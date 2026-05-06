"""Local llama-server lifecycle, spawned by a paired peer over the
encrypted P2P channel.

Why this exists
===============
When a chat model is too big for any single peer's free RAM but a
paired LAN peer already has the GGUF on disk, the right architecture
is "peer-orchestrated split": the peer with the GGUF runs llama-server
locally and uses one or more remote `--rpc` backends (typically the
orchestrator host's rpc-server) to fan layer compute across the LAN.

* No GGUF transfer — the peer keeps its existing copy.
* No internet pull — model bytes never re-downloaded.
* No host disk pressure — large MoE / dense models that wouldn't fit
  the orchestrator's disk run anyway because they live on the peer.

Verified case: dolphin-mixtral:8x7b (26.4 GB) lives on FBS (16.6 GB
free, can't fit alone in plain Ollama). Host has 8 GB CUDA + 32 GB
RAM and rpc-server already listening on 50052. This module lets the
orchestrator drive FBS to start its own llama-server with
`--rpc <host>:50052` so layers fan across both nodes — model runs
without any data movement.

API surface
===========
Three functions, one for each lifecycle verb:
* ``start_local_llama_server(...)`` — idempotent spawn keyed by port.
* ``get_local_llama_server_status(port)`` — snapshot for orchestrator.
* ``stop_local_llama_server(port=None)`` — kill scoped to one port
  or every llama-server we own.

Authentication
==============
The matching `/api/p2p/llama-server/start|status|stop` endpoints are
whitelisted in ``p2p_secure_proxy._FORWARDABLE_PATHS`` and routed to
local Gigachat (not Ollama) via ``_GIGACHAT_INTERNAL_PATHS``. So the
only callers are (a) loopback, (b) a paired peer whose envelope
passes X25519+ChaCha20+Ed25519 verification.

Scope
=====
Spawn / status / stop only. Model resolution is handled here (we
look at the local Ollama blob store), but layer placement and
adaptive rebalance live in ``split_lifecycle`` on the orchestrator.
This module just makes sure the llama-server process is running
locally with the right `--model` and `--rpc` flags.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import socket
import subprocess
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# How long llama-server may sit idle (no in-flight inference) before
# the watchdog kills it to free RAM. 5 min matches Ollama's default
# `OLLAMA_KEEP_ALIVE` so users have a familiar mental model — recent
# conversations stay warm, dormant ones release memory for other
# models. Configurable via env so power users can pin a model.
_IDLE_TIMEOUT_SEC = int(os.environ.get("GIGACHAT_LLAMA_IDLE_TIMEOUT", "300"))


# Where llama-server.exe lives on this install. Same path as
# ``p2p_rpc_server._RPC_SERVER_BIN_DIR`` — the `p2p_binary_fetch`
# auto-installer drops everything llama.cpp into one directory, so
# both binaries are siblings.
#
# IMPORTANT — pinned llama.cpp build: **b8840** (system_fingerprint
# `b8840-9e5647aff`). Releases ≥ b8841 ship PR #21998 ("rpc:
# refactor the RPC transport") which introduced a regression: any
# multi-node split chat crashes mid-decode at ggml-rpc.cpp:640
# ("Remote RPC server crashed or returned malformed response /
# send failed bytes_sent=0") AFTER prompt eval but BEFORE the
# first generated token. Reproduces with both SYCL and pure-CPU
# rpc-servers, regardless of GGML_RPC_TIMEOUT.
#
# b8840 is the last release before that PR landed (PR merged
# Apr 19, b8840 cut Apr 18). Verified live: dolphin-mixtral 26 GB
# layer-split across host + Naresh CPU rpc-servers + FBS local
# CUDA returns tokens cleanly on b8840, crashes on b9002.
#
# When upstream patches the regression (track at
# https://github.com/ggml-org/llama.cpp/issues/...), bump to that
# build and remove this pin. The download URLs:
#   https://github.com/ggml-org/llama.cpp/releases/tag/b8840
# (CUDA 12.4 zip + cudart for NVIDIA peers, SYCL zip for Intel
# iGPU peers — both layer onto cpu zip's base RPC + CPU DLLs.)
_LLAMA_SERVER_BIN_DIR = Path.home() / ".gigachat" / "llama-cpp"
_LLAMA_SERVER_EXE = _LLAMA_SERVER_BIN_DIR / (
    "llama-server.exe" if platform.system() == "Windows" else "llama-server"
)

# Default port we bind llama-server to when the orchestrator doesn't
# specify one. Chosen above the rpc-server pair (50052/50053) so a
# single host can run both rpc-server (so other peers use it as a
# backend) AND llama-server (so it serves chats locally) without
# port conflict.
_DEFAULT_PORT = 8090

# Map (port -> {model, pid, started_at}) so the orchestrator can ask
# "what's running on port 8090?" and we can short-circuit
# already-running spawns. Module-level state, mirrored across the
# spawn / status / stop verbs.
_active_servers: dict[int, dict] = {}

# Stability env vars — same set p2p_rpc_server uses. Intel Xe2 / Meteor
# Lake SYCL paths are sensitive to upstream bugs; these knobs prevent
# silent weight corruption and warmup crashes.
_SPAWN_ENV: dict[str, str] = {
    "GGML_SYCL_DISABLE_OPT": "1",
    "GGML_SYCL_DISABLE_GRAPH": "1",
    "SYCL_CACHE_PERSISTENT": "1",
    # llama-server reads this to find DLLs / drivers when run from a
    # detached process whose CWD isn't the binary dir on every Windows
    # release. Belt-and-suspenders alongside subprocess `cwd=`.
    "PATH": str(_LLAMA_SERVER_BIN_DIR) + os.pathsep + os.environ.get("PATH", ""),
    # Bump RPC TCP timeouts. Default is ~5 s which trips the
    # "Remote RPC server crashed or returned malformed response"
    # crash on big models — the rpc-server is busy doing compute
    # and doesn't ack a control message in time, llama-server
    # interprets the silence as a crash and aborts the whole
    # forward pass. 120 s gives slow CPU rpc-servers time to
    # finish a layer before we declare them dead.
    "GGML_RPC_TIMEOUT": "120000",
}

# How long to wait for the freshly-spawned llama-server to start
# listening on its TCP port. llama-server's startup includes loading
# the GGUF (mmap is fast) AND initializing all `--rpc` backends
# (slow — TCP handshake to each remote rpc-server, then their
# model-weight upload).
#
# 60 s was empirically too tight: a 26 GB Mixtral 8x7b model uploading
# layer weights to 4 RPC endpoints (host iGPU SYCL + host CPU + Naresh
# iGPU SYCL + Naresh CPU) over a 25 MB/s LAN takes ~3-5 minutes for
# the layer-weight push alone, plus SYCL JIT warmup on the receiving
# rpc-servers. Symptom from the recent test: every dolphin-mixtral
# spawn returned "spawned_but_not_listening" at 60 s, the chat
# fell back to host Ollama (which 404'd because host doesn't have
# the model), and the user saw zero deltas.
#
# 600 s (10 min) gives big models room to load on a slow LAN /
# Wi-Fi link without the orchestrator giving up early. Small
# models (8-15 layers) still bind in <30 s — the longer ceiling
# is bounded by HOW LONG WE WAIT, not how long it takes for fast
# models. Once bound, the chat traffic flows at the model's natural
# token rate.
_LISTEN_WAIT_SEC = 600.0


# ============================================================================
# Gigachat patched llama.cpp manifest.
#
# We ship our own patched build that adds RPC-resilience: instead of aborting
# the whole llama-server process when an RPC backend hiccups (the upstream
# behaviour, which manifests as STATUS_STACK_BUFFER_OVERRUN on Windows and
# kills every in-flight chat), the patched build throws a recoverable
# `rpc_remote_failure` exception that the server catches + surfaces to the
# client as a 5xx — the chat layer in `agent._stream_llama_server_chat`
# auto-retries the request once, so the user sees a brief stutter instead
# of a broken chat.
#
# `gigachat_patch_marker.txt` lives next to the binaries. When present, this
# install is using the patched build and the safety net is engaged. When
# absent, we know we're running stock upstream — chat dispatch's retry path
# still applies to (almost any) HTTP-level failure but the underlying RPC
# crash will hard-kill llama-server. The marker is checked at every spawn so
# a user who manually drops in newer (or older) binaries gets an automatic
# warning that they've lost the patch.
# ============================================================================
_GIGACHAT_PATCH_MARKER = _LLAMA_SERVER_BIN_DIR / "gigachat_patch_marker.txt"


# Patched llama.cpp release — published on the Gigachat repo's GitHub
# Releases page. Pinned by tag so a user `git pull`-ing later doesn't
# silently pick up a different build. Bumping this tag (e.g. when we
# rebuild against a newer upstream `b9XXX`) just means the next
# install / re-install pulls the new zip.
_PATCHED_RELEASE_TAG = "gigachat-llamacpp-b9030-1"
_PATCHED_RELEASE_URL = (
    "https://github.com/HyphenFrox/Gigachat/releases/download/"
    f"{_PATCHED_RELEASE_TAG}/gigachat-llamacpp-b9030-windows-x64.zip"
)
# sha256 of the published zip — verified after download to catch both
# corruption and (paranoid) supply-chain swaps. Update alongside the
# release tag whenever the binaries are rebuilt.
_PATCHED_RELEASE_SHA256 = (
    "f2e7e0ddb31ab61d5cc757d2be5ab52e3307fe5c0f01078c07696e6e0b6b1cc7"
)


def is_patched_llama_cpp_installed() -> bool:
    """Return True iff the binaries in ``_LLAMA_SERVER_BIN_DIR`` are the
    Gigachat-patched build (RPC throws-instead-of-aborts + transport
    layer retries on transient send/recv failures).

    Read by the chat dispatcher so the UI can show "split-mode is
    crash-resilient" vs "split-mode may abort on RPC blip" hints.
    """
    return _GIGACHAT_PATCH_MARKER.is_file()


def fetch_patched_llama_cpp(*, force: bool = False) -> dict[str, Any]:
    """Download + extract the published patched llama.cpp build.

    Pulls the platform-matched zip from the pinned GitHub Release
    (`_PATCHED_RELEASE_URL`), verifies its sha256, and unpacks it into
    `~/.gigachat/llama-cpp/`. Idempotent: returns the existing
    install when the marker is already present (unless `force=True`).

    Returns a structured result so install.bat / the UI can display
    progress / errors:

      {
        "ok":          bool,
        "skipped":     bool,    # already installed and not forced
        "downloaded":  bool,    # we hit the network this call
        "platform_supported": bool,  # only Windows x64 right now
        "bytes":       int,     # downloaded size on success
        "install_dir": str,
        "tag":         str,     # release tag we pulled
        "error":       str,     # populated on failure
      }

    Best-effort. Failure modes (no network, sha256 mismatch, zip
    corrupt, target dir not writable) all return ok=false rather
    than raising — the caller can fall back to its existing
    "warn the user" path.
    """
    import hashlib
    import platform as _platform
    import shutil
    import tempfile
    import urllib.request
    import zipfile

    install_dir = _LLAMA_SERVER_BIN_DIR
    result: dict[str, Any] = {
        "ok": False, "skipped": False, "downloaded": False,
        "platform_supported": False, "bytes": 0,
        "install_dir": str(install_dir),
        "tag": _PATCHED_RELEASE_TAG, "error": "",
    }

    # Right now we only ship Windows x64 binaries (the user base + the
    # build host's toolchain). Linux / macOS users still have to build
    # from source via vendor/llama.cpp-patches/README.md — surface that
    # cleanly instead of trying to install an incompatible binary.
    if _platform.system() != "Windows" or _platform.machine().lower() not in (
        "amd64", "x86_64",
    ):
        result["error"] = (
            f"unsupported platform {_platform.system()} {_platform.machine()}; "
            "build from source — see vendor/llama.cpp-patches/README.md"
        )
        return result
    result["platform_supported"] = True

    if is_patched_llama_cpp_installed() and not force:
        result["ok"] = True
        result["skipped"] = True
        return result

    install_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".zip", delete=False,
            dir=str(install_dir),
        ) as tmp:
            tmp_path = tmp.name
            log.info(
                "p2p_llama_server: downloading patched llama.cpp from %s ...",
                _PATCHED_RELEASE_URL,
            )
            req = urllib.request.Request(
                _PATCHED_RELEASE_URL,
                headers={"User-Agent": "Gigachat/llamacpp-installer"},
            )
            with urllib.request.urlopen(req, timeout=600) as r:
                # Stream in chunks so we don't blow memory on the
                # 100 MB zip + can hash on the fly.
                hasher = hashlib.sha256()
                bytes_done = 0
                while True:
                    chunk = r.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
                    hasher.update(chunk)
                    bytes_done += len(chunk)
        digest = hasher.hexdigest()
        result["bytes"] = bytes_done
        result["downloaded"] = True
        if digest != _PATCHED_RELEASE_SHA256:
            result["error"] = (
                f"sha256 mismatch: expected {_PATCHED_RELEASE_SHA256[:16]}…, "
                f"got {digest[:16]}… — refusing to extract"
            )
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            return result
        log.info(
            "p2p_llama_server: download OK (%d bytes, sha256 verified); "
            "extracting to %s",
            bytes_done, install_dir,
        )
        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(install_dir)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        # Final sanity check: the marker should now exist.
        if not is_patched_llama_cpp_installed():
            result["error"] = (
                "extracted zip but gigachat_patch_marker.txt not present "
                f"in {install_dir} — release may be malformed"
            )
            return result
        result["ok"] = True
        log.info(
            "p2p_llama_server: patched llama.cpp install complete (tag=%s)",
            _PATCHED_RELEASE_TAG,
        )
        return result
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        log.warning(
            "p2p_llama_server: patched llama.cpp fetch failed: %s",
            result["error"],
        )
        return result


def _ensure_patched_llama_cpp_or_warn() -> None:
    """Make sure the patched llama.cpp build is installed. On a fresh
    system this triggers a one-time download from the Gigachat
    release page; on subsequent calls it's a marker-file check and
    no-op. Idempotent — only attempts download once per process.

    Failure modes are non-fatal: an unsupported platform, no network,
    or sha256 mismatch all log a warning and let the caller fall
    back to whatever stock binaries are already in place. With stock
    binaries split-mode still works for clean runs but a transient
    RPC blip will hard-kill llama-server.
    """
    global _PATCHED_BUILD_WARNED
    try:
        already = _PATCHED_BUILD_WARNED
    except NameError:
        already = False
    if already:
        return
    _PATCHED_BUILD_WARNED = True
    if is_patched_llama_cpp_installed():
        log.info(
            "p2p_llama_server: patched llama.cpp build detected "
            "(gigachat_patch_marker.txt present) — RPC failures will be "
            "auto-recovered by the chat layer"
        )
        return
    # Not installed — try the auto-fetch from the GitHub Release.
    res = fetch_patched_llama_cpp()
    if res.get("ok"):
        log.info(
            "p2p_llama_server: patched llama.cpp auto-installed from %s "
            "(%d MB)", _PATCHED_RELEASE_URL, res["bytes"] // (1024 * 1024),
        )
        return
    log.warning(
        "p2p_llama_server: STOCK llama.cpp at %s (no gigachat_patch_marker.txt) "
        "and auto-install failed: %s. RPC backend hiccups will hard-kill "
        "llama-server. Run `python -c \"from backend.p2p_llama_server import "
        "fetch_patched_llama_cpp as f; print(f(force=True))\"` to retry, or "
        "build from source — see vendor/llama.cpp-patches/README.md",
        str(_LLAMA_SERVER_BIN_DIR), res.get("error", "unknown error"),
    )


_PATCHED_BUILD_WARNED = False


def _is_listening_on(port: int) -> bool:
    """Return True iff TCP port `port` has at least one listener."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def _kill_running_llama_servers(only_listening_on_port: int | None = None) -> int:
    """Kill llama-server processes owned by this user.

    `only_listening_on_port`: when set, only kill the llama-server
    whose listener includes that port (sibling llama-servers on
    other ports keep running). When None, kills every llama-server
    we own — used by full-shutdown / unpair paths.

    Returns the count killed. Best-effort: psutil-based, silent
    on per-process errors.
    """
    killed = 0
    try:
        import psutil
    except ImportError:
        log.debug("p2p_llama_server: psutil missing, can't enumerate processes")
        return 0
    me = os.getlogin() if hasattr(os, "getlogin") else ""
    for p in psutil.process_iter(["pid", "name", "username"]):
        try:
            name = (p.info.get("name") or "").lower()
            if not name.startswith("llama-server"):
                continue
            if me and (p.info.get("username") or "").lower().split("\\")[-1] != me.lower():
                continue
            if only_listening_on_port is not None:
                try:
                    proc = psutil.Process(p.info["pid"])
                    ports = {
                        c.laddr.port for c in proc.net_connections(kind="inet")
                        if c.status == psutil.CONN_LISTEN
                    }
                    if only_listening_on_port not in ports:
                        continue
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            p.terminate()
            killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if killed:
        # Give terminate a moment to settle so the next bind() of the
        # same port doesn't race a half-dead process.
        time.sleep(0.8)
    return killed


def _spawn_detached(cmd: list[str], cwd: str, env: dict[str, str]) -> int:
    """Spawn `cmd` so it survives this process exiting.

    Same detachment dance ``p2p_rpc_server._spawn_detached`` uses.
    stdout + stderr go into ``llama-server.log`` next to the binary
    so the orchestrator can diagnose load failures (e.g. "model file
    not found", "RPC backend connect failed") that wouldn't surface
    via the API otherwise.
    """
    log_path = _LLAMA_SERVER_BIN_DIR / "llama-server.log"
    try:
        _LLAMA_SERVER_BIN_DIR.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "ab")
    except OSError:
        log_f = subprocess.DEVNULL  # type: ignore[assignment]
    if platform.system() == "Windows":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_BREAKAWAY_FROM_JOB = 0x01000000
        flags = (
            DETACHED_PROCESS
            | CREATE_NEW_PROCESS_GROUP
            | CREATE_BREAKAWAY_FROM_JOB
        )
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            close_fds=True,
        )
        return proc.pid
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    return proc.pid


def _resolve_local_ollama_gguf(model_name: str) -> str | None:
    """Resolve a model name to its on-disk GGUF path via the local
    Ollama manifest store.

    Mirrors the resolution logic embedded in
    ``compute_pool.ensure_worker_chat_server``'s SSH PowerShell payload
    — but runs locally so we don't need PowerShell-via-SSH. Returns
    None when the model isn't installed in the local Ollama.

    Why we don't reuse ``compute_pool.resolve_ollama_model`` here:
    that function calls Ollama's HTTP /api/show endpoint, which on
    Windows includes a Modelfile parse step that fails for some
    older models. Reading the manifest directly is one fewer moving
    part — purely a filesystem walk.
    """
    base = Path.home() / ".ollama" / "models"
    if ":" in model_name:
        ns_model, tag = model_name.rsplit(":", 1)
        if "/" in ns_model:
            ns, model = ns_model.split("/", 1)
        else:
            ns, model = "library", ns_model
    else:
        ns, model, tag = "library", model_name, "latest"
    manifest = base / "manifests" / "registry.ollama.ai" / ns / model / tag
    if not manifest.exists():
        return None
    try:
        m = json.loads(manifest.read_text())
    except Exception as e:
        log.warning(
            "p2p_llama_server: failed to parse manifest %s: %s",
            manifest, e,
        )
        return None
    digest = None
    for layer in m.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            digest = layer.get("digest")
            break
    if not digest:
        return None
    blob = base / "blobs" / digest.replace(":", "-")
    return str(blob) if blob.exists() else None


def start_local_llama_server(
    *,
    model: str,
    port: int = _DEFAULT_PORT,
    rpc_targets: list[str] | None = None,
    n_gpu_layers: int | None = None,
    context_size: int = 4096,
    parallel: int = 1,
) -> dict:
    """Bring up llama-server locally, idempotent w.r.t. (port, model).

    Args:
        model: Ollama model name (e.g. "dolphin-mixtral:8x7b"). We
            resolve to its on-disk GGUF via the local Ollama manifest
            store. Required.
        port: TCP port to bind. Defaults to 8090. Caller can pick a
            different port if 8090 is in use; we'll happily run
            multiple llama-servers concurrently on different ports.
        rpc_targets: Optional list of "host:port" strings for
            additional `--rpc` backends. Each target is a remote
            rpc-server we'll fan layers to. Empty list means
            single-node mode (this peer's GPU + CPU only).
        n_gpu_layers: How many layers to request on this peer's GPU.
            None (default) lets llama.cpp's `--fit` auto-distribute
            layers across local GPU + local CPU + every `--rpc`
            backend based on each device's reported free memory —
            the right choice when the model exceeds any single
            device. Pass an explicit value (e.g. 99) only when you
            KNOW the model fits the local GPU alone.
        context_size: --ctx-size value. 4096 is a safe default for
            chat; the orchestrator may bump to 8192 / 16384 for
            longer-context models.
        parallel: --parallel slot count for continuous batching.
            1 = single chat, no batching overhead. The orchestrator
            sets higher values when multiple subagents share the
            same llama-server.

    Returns:
        ``{"ok": bool, "port": int, "url": str, "pid": int|None,
           "model_path": str|None, "status": str, "rpc_targets":
           list[str], "log_path": str}``

        Status values:
        * ``already_running`` — port had this exact (model,
          rpc_targets) running; no spawn.
        * ``started`` — spawned, listening verified.
        * ``no_binary`` — llama-server.exe not installed.
        * ``model_not_found`` — couldn't resolve model GGUF locally.
        * ``spawn_failed`` — Popen raised; error in payload.
        * ``spawned_but_not_listening`` — process started but never
          opened the port; likely RPC connect or model-load failure
          (check log_path).
    """
    rpc_targets = list(rpc_targets or [])
    # Surface a one-shot warning if the binaries here aren't the
    # Gigachat-patched build — the user needs to know that an RPC
    # blip will hard-kill the process instead of being auto-recovered.
    _ensure_patched_llama_cpp_or_warn()
    out: dict = {
        "ok": False,
        "binary_path": str(_LLAMA_SERVER_EXE),
        "model": model,
        "port": port,
        "rpc_targets": rpc_targets,
        "log_path": str(_LLAMA_SERVER_BIN_DIR / "llama-server.log"),
        "patched_build": is_patched_llama_cpp_installed(),
    }

    # Already running with the exact same model+rpc_targets on this
    # port? Re-verify TCP listener and short-circuit. Saves a
    # multi-second restart on every chat turn that re-routes here.
    current = _active_servers.get(port)
    if (
        current
        and current.get("model") == model
        and current.get("rpc_targets") == rpc_targets
        and _is_listening_on(port)
    ):
        out["ok"] = True
        out["status"] = "already_running"
        out["pid"] = current.get("pid")
        out["model_path"] = current.get("model_path")
        out["url"] = f"http://0.0.0.0:{port}"
        return out

    if not _LLAMA_SERVER_EXE.is_file():
        out["status"] = "no_binary"
        out["error"] = (
            f"llama-server binary missing at {_LLAMA_SERVER_EXE}; "
            "install llama.cpp on this peer (e.g. via the binary "
            "auto-fetch path or download from the upstream releases)."
        )
        return out

    # Resolve the model to an on-disk GGUF before spending the
    # process-spawn cycle. A fast-fail here saves 60 s of waiting
    # for a listener that'll never come up.
    model_path = _resolve_local_ollama_gguf(model)
    if not model_path:
        out["status"] = "model_not_found"
        out["error"] = (
            f"model {model!r} not found in local Ollama store. "
            "Pull it on this peer first (`ollama pull <model>`)."
        )
        return out
    out["model_path"] = model_path

    # Different model / different rpc_targets / not listening — kill
    # whatever's on this port and respawn fresh.
    killed = _kill_running_llama_servers(only_listening_on_port=port)
    out["killed_stale"] = killed
    _active_servers.pop(port, None)

    # Build the spawn command. Match the flags
    # ``compute_pool.ensure_worker_chat_server`` uses on host so
    # behaviour is consistent across peer-led and host-led splits:
    #   * --flash-attn on + KV q8_0 → halve KV memory at <1 % accuracy
    #     loss, frees room for layer weights.
    #   * --cache-reuse 256 → multi-turn chat reuses prefix KV.
    #   * --jinja → prompt template rendering for chat-completion API.
    #   * --no-warmup → skip the dummy forward pass; the first real
    #     request warms the engine implicitly. Saves 5-10 s of
    #     unnecessary GPU work on cold start.
    cmd = [
        str(_LLAMA_SERVER_EXE),
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--flash-attn", "on",
        "-ctk", "q8_0", "-ctv", "q8_0",
        "--cache-reuse", "256",
        "--jinja",
        "-c", str(context_size),
        "--no-warmup",
        "--parallel", str(parallel),
    ]
    # `-ngl` is the explicit "force this many layers on the local GPU"
    # knob. Only set it when the caller provided a value — otherwise
    # llama.cpp's `--fit` defaults to auto-distributing layers across
    # every device (local GPU + local CPU + every `--rpc` backend)
    # based on each one's reported free memory. That's the right
    # choice for big-model peer-orchestrated split where the model
    # exceeds any single GPU.
    if n_gpu_layers is not None:
        cmd += ["-ngl", str(n_gpu_layers)]
    if rpc_targets:
        cmd += ["--rpc", ",".join(rpc_targets)]

    env = dict(os.environ)
    env.update(_SPAWN_ENV)
    try:
        pid = _spawn_detached(
            cmd, cwd=str(_LLAMA_SERVER_BIN_DIR), env=env,
        )
    except Exception as e:
        out["status"] = "spawn_failed"
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    out["pid"] = pid

    # Wait for the listener. llama-server with `--rpc` backends takes
    # 30-60 s to come up: TCP handshake + Ollama-style metadata exchange
    # + initial layer-weight upload to each rpc-server. Worth the
    # blocking wait so the orchestrator gets a definitive
    # "ready / failed" signal in one round trip.
    deadline = time.time() + _LISTEN_WAIT_SEC
    listening = False
    while time.time() < deadline:
        if _is_listening_on(port):
            listening = True
            break
        time.sleep(0.5)

    if listening:
        out["ok"] = True
        out["status"] = "started"
        out["url"] = f"http://0.0.0.0:{port}"
        _active_servers[port] = {
            "model": model,
            "rpc_targets": rpc_targets,
            "pid": pid,
            "model_path": model_path,
            "started_at": time.time(),
            "last_active_at": time.time(),
            # Persisted spawn args so the crash-respawn watchdog can
            # rebuild the exact same command line if llama-server dies
            # unexpectedly (e.g. RPC backend crash that propagates
            # through C frames past our patched throw to terminate).
            "spawn_args": {
                "n_gpu_layers": n_gpu_layers,
                "context_size": context_size,
                "parallel": parallel,
            },
        }
        # Spawn a daemon thread that watches for idle and kills the
        # llama-server when it's been doing nothing for too long.
        t = threading.Thread(
            target=_idle_watchdog,
            args=(port, pid, _IDLE_TIMEOUT_SEC),
            daemon=True,
            name=f"llama-server-idle-watch-{port}",
        )
        t.start()
        # Spawn a SECOND daemon thread that watches for unexpected
        # death and auto-respawns. The patched ggml-rpc.cpp throws a
        # recoverable exception on transient RPC failure instead of
        # GGML_ABORT()ing, but the throw still unwinds through C
        # frames in ggml's backend dispatcher and terminates the
        # process. This watchdog catches that case (process gone
        # while we still EXPECT it to be running) and respawns with
        # the same args. Net effect: a transient RPC blip becomes a
        # 30-60 s pause (model reload) instead of a permanent
        # broken split-mode chat. The chat layer's retry logic in
        # `agent._stream_llama_server_chat` papers over the gap.
        crash_t = threading.Thread(
            target=_crash_respawn_watchdog,
            args=(port, pid, model, rpc_targets, n_gpu_layers,
                  context_size, parallel),
            daemon=True,
            name=f"llama-server-crash-watch-{port}",
        )
        crash_t.start()
    else:
        out["status"] = "spawned_but_not_listening"
        out["error"] = (
            f"llama-server PID {pid} did not start listening on "
            f"127.0.0.1:{port} within {_LISTEN_WAIT_SEC}s. "
            f"Check {out['log_path']!r} for the spawn-side error — "
            "common causes: rpc-server not reachable on a target, "
            "model file missing on this peer, or out-of-memory "
            "during weight load."
        )
    return out


def _crash_respawn_watchdog(
    port: int,
    initial_pid: int,
    model: str,
    rpc_targets: list[str],
    n_gpu_layers: int | None,
    context_size: int,
    parallel: int,
) -> None:
    """Background thread: detect when llama-server dies UNEXPECTEDLY
    (i.e. _active_servers still expects it on this port AND the
    process is gone AND we're not in a clean-shutdown path) and
    auto-respawn with the original args.

    The patched ggml-rpc.cpp + ggml-sycl.cpp throw recoverable
    exceptions on transient RPC failures / SYCL DEVICE_LOST instead
    of GGML_ABORT()ing — but the throws still unwind through ggml's
    C backend-dispatcher frames and terminate the process. Without
    this watchdog every transient hardware blip leaves the user with
    a permanently dead llama-server until manual restart.

    Polls every 5 s. Caps respawn attempts at 5 within 60 s — past
    that we assume the failure is structural (model OOM, GPU truly
    dead, etc.) and stop trying so a runaway respawn loop can't pin
    the box. Each successful respawn resets the counter.

    Cooperates with `_idle_watchdog`: when the idle watchdog kills
    the process intentionally, it pops the port from `_active_servers`
    BEFORE killing — so the crash watchdog sees an empty entry and
    exits without respawning.
    """
    import urllib.error
    import urllib.request
    POLL_SEC = 5.0
    RESPAWN_BUDGET = 5
    BUDGET_WINDOW_SEC = 60.0
    respawn_history: list[float] = []
    expected_pid = initial_pid
    log.info(
        "p2p_llama_server: crash-respawn watchdog started for port %d "
        "pid %d (model=%s, rpc_targets=%d)",
        port, expected_pid, model, len(rpc_targets),
    )
    try:
        import psutil
    except ImportError:
        log.warning(
            "p2p_llama_server: crash-respawn watchdog disabled — "
            "psutil missing, can't detect process exit reliably"
        )
        return
    while True:
        time.sleep(POLL_SEC)
        # If the orchestrator no longer expects llama-server on this
        # port (clean stop, or idle-watchdog killed it), exit silently.
        active = _active_servers.get(port)
        if not active:
            return
        # If the model + rpc_targets on the active record drifted
        # (someone called start_local_llama_server with different
        # args), let that handler take over — we'd be respawning the
        # OLD model otherwise.
        if active.get("model") != model or active.get("rpc_targets") != rpc_targets:
            log.info(
                "p2p_llama_server: crash watchdog port %d — active model "
                "or rpc_targets changed, exiting (replacement watchdog "
                "will take over)", port,
            )
            return
        if psutil.pid_exists(expected_pid):
            continue
        # Process gone but we still expect it. RESPAWN.
        now = time.time()
        respawn_history = [t for t in respawn_history if (now - t) < BUDGET_WINDOW_SEC]
        if len(respawn_history) >= RESPAWN_BUDGET:
            log.warning(
                "p2p_llama_server: crash watchdog port %d — exhausted "
                "respawn budget (%d in last %.0fs); stopping respawn "
                "attempts. Manually restart via "
                "/api/p2p/llama-server/start when the underlying issue "
                "is resolved.",
                port, RESPAWN_BUDGET, BUDGET_WINDOW_SEC,
            )
            _active_servers.pop(port, None)
            return
        log.warning(
            "p2p_llama_server: crash watchdog port %d — process %d died "
            "unexpectedly (likely transient RPC backend crash or SYCL "
            "DEVICE_LOST that unwound past our patched catch). "
            "Respawning attempt %d/%d with same args.",
            port, expected_pid, len(respawn_history) + 1, RESPAWN_BUDGET,
        )
        respawn_history.append(now)
        # Drop the dead record before re-spawn so start_local_llama_server's
        # short-circuit doesn't think it's already running.
        _active_servers.pop(port, None)
        try:
            result = start_local_llama_server(
                model=model,
                port=port,
                rpc_targets=rpc_targets,
                n_gpu_layers=n_gpu_layers,
                context_size=context_size,
                parallel=parallel,
            )
        except Exception as e:
            log.warning(
                "p2p_llama_server: crash watchdog respawn raised: %s: %s",
                type(e).__name__, e,
            )
            continue
        if not result.get("ok"):
            log.warning(
                "p2p_llama_server: crash watchdog respawn failed for "
                "port %d: status=%s error=%s",
                port, result.get("status"), result.get("error"),
            )
            # Don't return — we'll try again next tick (subject to budget).
            continue
        new_pid = result.get("pid")
        if new_pid:
            expected_pid = int(new_pid)
            log.info(
                "p2p_llama_server: crash watchdog respawned port %d as "
                "pid %d", port, expected_pid,
            )
        # NOTE: start_local_llama_server already started a new
        # _crash_respawn_watchdog thread for the new pid, so the next
        # iteration of THIS thread will find _active_servers[port]
        # pointing at the new entry and exit on the model/rpc check.


def _idle_watchdog(port: int, pid: int, idle_timeout_sec: int) -> None:
    """Background thread: poll llama-server's `/slots` endpoint and
    kill the process when it's been idle for `idle_timeout_sec`.

    We use `/slots` (not just /health) because /health returns 200
    even while inference is running. /slots returns the actual slot
    state — when every slot's `state == 0` (SLOT_STATE_IDLE), the
    server has nothing in flight. We track the last time we saw
    ANY slot busy; if that's too far in the past, kill.

    Polls every 30 s. Exits when the process is gone (someone else
    killed it, e.g. user restart) or after the kill we trigger.

    Critical for FBS: dolphin-mixtral's llama-server holds 11 GB of
    RAM. Without this, FBS sits at 96 % memory permanently after a
    chat — every other model on FBS Ollama then 500's with
    "model requires more system memory than is available".
    """
    poll_interval_sec = 30.0
    url = f"http://127.0.0.1:{port}/slots"
    last_active = time.time()
    log.info(
        "p2p_llama_server: idle watchdog started for port %d pid %d "
        "(timeout %ds)", port, pid, idle_timeout_sec,
    )
    while True:
        try:
            time.sleep(poll_interval_sec)
        except Exception:
            return
        # If process is gone, exit cleanly.
        try:
            import psutil
            if not psutil.pid_exists(pid):
                log.info(
                    "p2p_llama_server: idle watchdog port %d — process "
                    "%d already gone, watchdog exiting",
                    port, pid,
                )
                _active_servers.pop(port, None)
                return
        except ImportError:
            # psutil unavailable — fall back to socket probe.
            if not _is_listening_on(port):
                log.info(
                    "p2p_llama_server: idle watchdog port %d — listener "
                    "gone, watchdog exiting", port,
                )
                _active_servers.pop(port, None)
                return
        # Probe /slots and check whether ANY slot is busy.
        any_busy = False
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as r:
                slots = json.loads(r.read())
                # Newer llama-server returns a list of slot dicts;
                # older builds return a {"slots": [...]} envelope.
                if isinstance(slots, dict):
                    slots = slots.get("slots") or []
                for slot in (slots or []):
                    state = slot.get("state")
                    # state 0 == SLOT_STATE_IDLE per llama.cpp.
                    # Anything else (1, 2, 3) means the slot is
                    # processing a request right now.
                    if state and state != 0:
                        any_busy = True
                        break
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError) as e:
            # Probe failed — could be llama-server not yet ready, or
            # crashed mid-call. Don't reset last_active on probe
            # error so a hung llama-server gets killed quickly.
            log.debug(
                "p2p_llama_server: /slots probe failed for port %d: %s",
                port, e,
            )
            continue
        if any_busy:
            last_active = time.time()
            continue
        idle_for = time.time() - last_active
        if idle_for >= idle_timeout_sec:
            log.info(
                "p2p_llama_server: port %d pid %d idle for %.0fs "
                "(threshold %ds) — killing to free RAM. Next chat "
                "for this model will respawn fresh.",
                port, pid, idle_for, idle_timeout_sec,
            )
            try:
                _kill_running_llama_servers(only_listening_on_port=port)
            except Exception as e:
                log.warning(
                    "p2p_llama_server: idle-kill of port %d failed: %s",
                    port, e,
                )
            _active_servers.pop(port, None)
            return


def get_local_llama_server_status(port: int = _DEFAULT_PORT) -> dict:
    """Snapshot of the local llama-server's state on `port`.

    Returns whether the port is listening, what model+rpc_targets we
    spawned (if known), and the PID. Used by orchestrators to decide
    whether to skip a redundant spawn.
    """
    state = _active_servers.get(port)
    return {
        "binary_path": str(_LLAMA_SERVER_EXE),
        "binary_present": _LLAMA_SERVER_EXE.is_file(),
        "port": port,
        "listening": _is_listening_on(port),
        "active_model": (state or {}).get("model"),
        "active_rpc_targets": (state or {}).get("rpc_targets"),
        "active_pid": (state or {}).get("pid"),
        "started_at": (state or {}).get("started_at"),
        "active_servers": dict(_active_servers),
    }


def stop_local_llama_server(port: int | None = None) -> dict:
    """Kill llama-server process(es).

    `port`: scope kill to one port (siblings on other ports keep
    running). When None, kills every llama-server we own.
    """
    if port is not None:
        killed = _kill_running_llama_servers(only_listening_on_port=port)
        _active_servers.pop(port, None)
        return {
            "killed": killed,
            "port": port,
            "listening": _is_listening_on(port),
        }
    killed = _kill_running_llama_servers()
    _active_servers.clear()
    return {"killed": killed, "listening": _is_listening_on(_DEFAULT_PORT)}
