"""Split-model runtime: locate / install llama.cpp on the host.

Phase 2 of the compute pool runs `llama-server` on the host (with
`--rpc <worker>:<port>` flags) so a single big model's layers fan
across the host's CPU+GPU and one or more compute workers' CPU+GPU.
The orchestration is `llama.cpp` — which Ollama uses internally but
doesn't expose RPC for — so we install a separate `llama.cpp` build
alongside Ollama and shell out to it for the split path.

This module is the **detection + install** layer:

  * `find_llama_server()` / `find_rpc_server()` — check PATH and our
    private install directory for an existing binary; return None if
    not found.
  * `download_llama_cpp_for_host()` — fetch the prebuilt CUDA Windows
    zip from GitHub, extract into `~/.gigachat/llama-cpp/`, and verify
    the binaries land. Used by the Settings UI's "Install llama.cpp"
    button (commit 6) — never auto-fired so the user doesn't get a
    surprise multi-hundred-MB download on first app boot.
  * `get_install_status()` — read-only summary the API + UI can render
    to show "installed at X (version Y)" or "not installed".

Lifecycle (start/stop a `llama-server` process for a registered split
model, surface output) lives in commit 4 — `split_lifecycle.py`. This
module deliberately stops at "binaries on disk".

Layout once installed::

    ~/.gigachat/llama-cpp/
        llama-server.exe
        llama-cli.exe
        rpc-server.exe         (only on the worker side; host doesn't run rpc-server)
        ggml-cuda.dll          (or ggml-vulkan.dll for worker variant)
        ...support DLLs

Variant choice: CUDA build on the host (RTX 3060 Ti), Vulkan build on
each worker (Intel iGPU / AMD iGPU / NVIDIA without CUDA toolkit
worth bothering about). The host module here only cares about the
host variant; worker-side install is handled separately via SSH or
documented manual install steps.
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import httpx

log = logging.getLogger(__name__)


# Pinned llama.cpp release tag. Refresh when new features land that we
# need (RPC stability fixes especially). Bumping is a deliberate act —
# we don't track main because the prebuilt zip layout has shifted in
# the past, and our extractor expects a specific shape.
LLAMA_CPP_VERSION = "b8934"

# Where we keep our private llama.cpp install. Lives outside the repo
# so a `git clean` doesn't wipe it; lives under the user's home so
# multiple Gigachat checkouts (dev / prod) share the same binaries.
LLAMA_CPP_INSTALL_DIR = Path.home() / ".gigachat" / "llama-cpp"

# Binaries we expect to find after install. `llama-server.exe` is the
# OpenAI-compatible HTTP API the routing layer talks to;
# `rpc-server.exe` is what runs on each compute worker. Keeping the
# host's install copy of `rpc-server.exe` around even though the host
# doesn't run it makes pushing the worker variant via SCP easier later.
HOST_BINARIES = ("llama-server", "llama-cli")
WORKER_BINARIES = ("rpc-server",)


@dataclass
class InstallStatus:
    """What `get_install_status()` reports back to the API/UI.

    `installed` is the bottom-line boolean the UI flips a banner on.
    `llama_server_path` is None when not installed, else the absolute
    path so the user can copy it to verify or replace manually.
    """
    installed: bool
    install_dir: str
    llama_server_path: str | None
    rpc_server_path: str | None
    version: str
    platform_supported: bool
    platform_reason: str | None


def _candidate_dirs() -> Iterable[Path]:
    """Where we look for an existing `llama-server` binary.

    Priority order — earliest match wins:
      1. Our private install dir (pinned version, our control).
      2. Anything on PATH (lets a power user point at their own build
         by symlinking; useful when llama.cpp's prebuilt CUDA zip
         doesn't match their toolkit and they've compiled locally).
    """
    yield LLAMA_CPP_INSTALL_DIR
    # PATH directories — split by os.pathsep, dedupe, keep order.
    seen: set[str] = set()
    for raw in (os.environ.get("PATH") or "").split(os.pathsep):
        d = raw.strip()
        if not d or d in seen:
            continue
        seen.add(d)
        yield Path(d)


def _resolve_binary(name: str) -> Path | None:
    """Find the named binary in any candidate dir. Windows-only path
    extension handling — we don't support Linux/macOS for now since the
    host this app runs on is Windows."""
    suffixes = (".exe",) if platform.system() == "Windows" else ("",)
    for d in _candidate_dirs():
        for suffix in suffixes:
            p = d / f"{name}{suffix}"
            if p.is_file():
                return p
    return None


def find_llama_server() -> Path | None:
    """Return the absolute path to `llama-server` if installed, else None."""
    return _resolve_binary("llama-server")


def find_rpc_server() -> Path | None:
    """Return the absolute path to `rpc-server` if installed locally.

    The host doesn't run rpc-server during inference (workers do), but
    we keep a copy around so pushing it to a freshly-registered worker
    over SSH/SCP is a one-liner — no internet pull on the worker side.
    """
    return _resolve_binary("rpc-server")


def _platform_support() -> tuple[bool, str | None]:
    """Phase 2's auto-install path is Windows-only for now.

    Linux / macOS hosts can still USE the feature by building llama.cpp
    themselves and dropping `llama-server` onto PATH — `find_llama_server`
    will pick it up. But the auto-download flow targets the host this
    app actually ships on (Windows desktop)."""
    if platform.system() != "Windows":
        return False, f"auto-install only supports Windows; got {platform.system()}"
    if platform.machine().lower() not in ("amd64", "x86_64"):
        return False, f"only x86_64 supported; got {platform.machine()}"
    return True, None


def get_install_status() -> InstallStatus:
    """Snapshot the current install state. Cheap; safe to call often."""
    server = find_llama_server()
    rpc = find_rpc_server()
    supported, reason = _platform_support()
    return InstallStatus(
        installed=server is not None,
        install_dir=str(LLAMA_CPP_INSTALL_DIR),
        llama_server_path=str(server) if server else None,
        rpc_server_path=str(rpc) if rpc else None,
        version=LLAMA_CPP_VERSION,
        platform_supported=supported,
        platform_reason=reason,
    )


# ---------------------------------------------------------------------------
# Download + extract
# ---------------------------------------------------------------------------

def _release_url(variant: str) -> str:
    """Build the GitHub releases zip URL for one of the prebuilt
    Windows variants.

    `variant` is the suffix llama.cpp uses on its release artifacts —
    e.g. `cuda-12.4` for the host's RTX, `vulkan` for the laptop's
    iGPU, `cpu` for a fallback. The release filename schema has been
    stable for ~30 releases as of `b5174`; if a future bump breaks
    this, it's a one-line edit.
    """
    fname = f"llama-{LLAMA_CPP_VERSION}-bin-win-{variant}-x64.zip"
    return (
        f"https://github.com/ggml-org/llama.cpp/releases/download/"
        f"{LLAMA_CPP_VERSION}/{fname}"
    )


# Variants we know how to fetch. The keys are LOGICAL roles; the
# values are llama.cpp's release suffixes. Roles map to hardware:
#
#   host   = CUDA build for an NVIDIA host running the orchestrating
#            llama-server. CUDA 12.4 covers any recent Nvidia driver.
#   worker = Vulkan build for the worker's rpc-server. Cross-vendor
#            (Intel iGPU, AMD iGPU, NVIDIA without CUDA toolkit, etc).
#            Default for "I just want it to work" — universal.
#   sycl   = Intel-native SYCL/oneAPI build. ~15-25% faster than
#            Vulkan on Intel iGPUs and Intel Arc dGPUs because it
#            uses Intel's native compute API. Bundles the oneAPI
#            runtime DLLs (~150 MB zip) so no separate install
#            needed. Pick this on Intel-only worker fleets.
#   cpu    = pure-CPU build, last resort. Smaller zip, no GPU
#            backend at all. Useful when the worker has no usable
#            GPU and you'd rather not pull a Vulkan/SYCL DLL set
#            you'll never use.
_VARIANTS = {
    "host": "cuda-12.4",
    "worker": "vulkan",
    "sycl": "sycl",
    "cpu": "cpu",
}


def recommend_worker_variant(gpus: list[dict] | None) -> str:
    """Pick the right llama.cpp worker variant for a worker's hardware.

    `gpus` is the `gpus` array captured by `compute_pool._probe_worker_specs_via_ssh`
    — a list of dicts with at least a `name` field. Decision rules,
    in priority order:

      * Any Intel GPU (iGPU or Arc dGPU) → `"sycl"`. The SYCL backend
        uses Intel's native compute API and is ~15-25% faster than
        Vulkan on Intel hardware. Bundled DPC++/oneAPI runtime DLLs
        ship in the prebuilt zip — no separate Intel oneAPI install
        needed on the worker.
      * Any AMD or NVIDIA GPU (without dedicated CUDA toolkit) →
        `"worker"` (Vulkan). Cross-vendor backend, ships with the
        GPU driver on Windows.
      * No GPUs detected → `"cpu"`. Smallest zip, no GPU backend
        DLLs to ship.

    Workers we have no spec data for (SSH probe never ran, ssh_host
    unset, etc.) get `"worker"` — the safe universal default.
    """
    if not gpus:
        return "worker"  # no signal — use universal Vulkan build
    has_intel = False
    has_other = False
    for g in gpus:
        name = ((g or {}).get("name") or "").lower()
        if "intel" in name:
            has_intel = True
        elif name:
            has_other = True
    if has_intel and not has_other:
        return "sycl"
    if has_intel and has_other:
        # Mixed Intel + something else (e.g. Intel iGPU + NVIDIA dGPU
        # in a gaming laptop). Vulkan covers both.
        return "worker"
    if has_other:
        return "worker"
    return "cpu"


ProgressCallback = Callable[[str, int, int], None]
"""Signature for download progress hooks: (phase, bytes_so_far, total)."""


def _download_to_disk(
    url: str,
    dest: Path,
    *,
    on_progress: ProgressCallback | None = None,
) -> None:
    """Stream a remote file to disk.

    Uses httpx so we share the same HTTP stack as the rest of the
    backend — caller can monkey-patch `split_runtime.httpx` to stub
    network access in tests. Streams chunks rather than buffering the
    whole zip in memory (these zips are 150–250 MB).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    # follow_redirects: GitHub releases issue a 302 to S3.
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", "0") or 0)
        seen = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(chunk_size=1024 * 256):
                f.write(chunk)
                seen += len(chunk)
                if on_progress:
                    on_progress("downloading", seen, total)
    tmp.replace(dest)


def _extract_zip(zip_path: Path, into: Path) -> list[Path]:
    """Extract zip flat into `into/`. llama.cpp's prebuilt zips have a
    flat layout (no top-level dir wrapping the binaries), so a plain
    `extractall` lands `llama-server.exe` directly in our install dir.
    Returns the list of files written so the caller can sanity-check."""
    into.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            # Defend against zip-slip — refuse paths that escape `into`.
            target = (into / member).resolve()
            try:
                target.relative_to(into.resolve())
            except ValueError:
                log.warning("split_runtime: refusing zip-slip path %r", member)
                continue
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            written.append(target)
    return written


def download_llama_cpp(
    variant: str = "host",
    *,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Download and extract one of the known llama.cpp Windows variants
    into our private install dir.

    `variant` ∈ {host, worker, cpu}. Returns the install dir on
    success; raises on any failure (network, zip corruption, missing
    binary post-extract). Idempotent on repeat calls — re-extracting
    just overwrites in place.
    """
    supported, reason = _platform_support()
    if not supported:
        raise RuntimeError(f"llama.cpp auto-install: {reason}")
    if variant not in _VARIANTS:
        raise ValueError(
            f"unknown llama.cpp variant {variant!r}; "
            f"valid: {sorted(_VARIANTS)}"
        )

    LLAMA_CPP_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    url = _release_url(_VARIANTS[variant])
    zip_path = LLAMA_CPP_INSTALL_DIR / f"llama-cpp-{variant}.zip"

    if on_progress:
        on_progress("starting", 0, 0)
    _download_to_disk(url, zip_path, on_progress=on_progress)

    if on_progress:
        on_progress("extracting", 0, 0)
    written = _extract_zip(zip_path, LLAMA_CPP_INSTALL_DIR)

    # Drop the zip once extracted — it's ~150 MB of redundant bytes
    # we don't need around. Keep it on extract failure (the caller's
    # exception handler can preserve evidence).
    try:
        zip_path.unlink()
    except OSError:
        pass

    # Verify the variant put down what we expected — different builds
    # ship different DLL sets but `llama-server` is in every variant.
    expected = HOST_BINARIES if variant in ("host", "cpu") else ("rpc-server",)
    for binname in expected:
        if not _resolve_binary(binname):
            paths = ", ".join(str(p) for p in written[:5])
            raise RuntimeError(
                f"llama.cpp install: expected '{binname}' missing after "
                f"extract from {url}. First few extracted paths: {paths}"
            )

    if on_progress:
        on_progress("done", 1, 1)
    return LLAMA_CPP_INSTALL_DIR


def uninstall_llama_cpp() -> None:
    """Wipe the private install dir. Manual recovery hatch for the user
    when an extracted DLL is corrupted or they want to force a fresh
    download after bumping `LLAMA_CPP_VERSION`. Doesn't touch any
    binaries the user installed via PATH."""
    if LLAMA_CPP_INSTALL_DIR.exists():
        shutil.rmtree(LLAMA_CPP_INSTALL_DIR)
