"""Split-model lifecycle: start / stop the local `llama-server` process.

For each registered `split_models` row (commit 1 schema), this module
brings up a single `llama-server.exe` instance on the host with:

  llama-server
    --model     <gguf_path>
    --host      127.0.0.1                 (loopback only — the routing
                                           layer runs in-process here;
                                           no need to expose to LAN)
    --port      <split_models.llama_port>
    --rpc       <worker1_addr>:<port>,<worker2_addr>:<port>,...
    -ngl        <gpu_layers>               (default tuned per row;
                                            commit 4 hard-codes 99 to
                                            "as many as fit" — schema
                                            knob lands later)
    --jinja                                (use the GGUF's embedded
                                            chat template if present)

Architecture notes:

* **Resource pooling is native to llama.cpp.** With `-ngl N`, N layers
  go to GPUs in the pool (host VRAM via CUDA + each worker's GPU via
  Vulkan via rpc-server). Layers that don't fit GPUs cascade to the
  orchestrator's CPU+RAM and to each rpc-server's CPU+RAM. All four
  resource tiers (host VRAM, host RAM, worker GPU, worker RAM) get
  used; no resource is wasted. This module just supplies the flags;
  llama-server figures out per-layer placement.

* **One llama-server per split_model row.** Two rows can run
  concurrently on different `llama_port`s (default 11500, 11501, …).
  The chat router (commit 5) picks which one based on the conversation's
  active model.

* **Process supervision is intentionally minimal.** If llama-server
  exits, we record `status=error` with stderr tail and stop. We do
  NOT restart automatically — most failures are misconfiguration
  (wrong GGUF path, port already bound, GGUF newer than this
  llama.cpp build) and a silent retry-loop would just spam logs.
  Caller restarts manually after fixing the config.

* **In-process registry.** We track running processes in a
  module-level dict keyed by split_model_id. The DB status reflects
  the same state but the dict holds the Popen handle for stop().
  The dict survives the app process; DB status survives across
  restarts. On boot, anything DB-marked `running` is cross-checked
  against the OS — if no process exists, status is reset to `stopped`
  (commit 7 polish).

* **Log files** land in `~/.gigachat/llama-cpp/logs/<split_id>.log`
  for stdout+stderr merged, so the Settings UI can tail them.

Test surface kept tight: this module's pure-Python helpers
(command-building, log-rotation paths) are tested directly; the
actual `subprocess.Popen` spawn is exercised through stubbing in
test_split_lifecycle.py — we don't try to integration-test against
a real llama-server binary in CI.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from . import db, split_runtime

log = logging.getLogger(__name__)


# Default GPU-offload setting: 99 means "offload as many layers as
# possible to GPU memory" — llama.cpp clamps to the model's actual
# layer count and falls through to CPU+RAM for layers that don't fit.
# The user can override per row once the schema gains an `ngl` field
# (deferred — most users won't tune this manually).
_DEFAULT_NGL = 99


def _lower_priority_posix() -> None:
    """preexec hook that nice's the child to +10 on POSIX. Called only
    on Linux/macOS — Windows uses BELOW_NORMAL_PRIORITY_CLASS via
    creationflags. Both achieve the same goal: the OS scheduler
    de-prioritizes our inference when other workloads compete for CPU,
    so the user's foreground apps stay responsive without us giving up
    full speed on an otherwise-idle machine.
    """
    try:
        os.nice(10)
    except Exception:
        # Best-effort; if it fails (rare), child still runs at default
        # priority. Not worth crashing the spawn for.
        pass

# How long we wait for `llama-server` to come up before declaring the
# start a failure. Loading large models is slow:
#   * Reading GGUF from disk: ~10 s for a 17 GB model on NVMe.
#   * Pushing layer weights to host VRAM via CUDA: ~5–10 s.
#   * Streaming layer weights to each worker's rpc-server over LAN:
#     dominated by LAN bandwidth — at 100 Mb/s Wi-Fi, 10 GB takes
#     ~14 minutes; at 1 Gb/s Ethernet, ~80 s. We size the timeout for
#     the median home-LAN case (5 GHz Wi-Fi, ~30–50 MB/s effective):
#     a 10 GB push lands in ~3–5 minutes.
# 600 s (10 min) is conservative enough that any reasonable hardware
# combination gets a fair shot, and short enough that a permanently-
# wedged llama-server (port-bound, OOM, GGUF mismatch) doesn't make
# the user wait forever.
_BOOT_TIMEOUT_SEC = 600.0

# Polling cadence for the readiness check. llama-server's `/health`
# becomes available the moment the HTTP server binds; we poll often
# enough to react fast but not so often that we spin during the
# (potentially) tens of seconds of model loading.
_HEALTH_POLL_SEC = 0.5


# ---------------------------------------------------------------------------
# Process registry — running llama-server instances keyed by split_model id
# ---------------------------------------------------------------------------

@dataclass
class _RunningProcess:
    """Bookkeeping for one live llama-server child."""
    proc: subprocess.Popen
    port: int
    started_at: float
    log_path: Path
    cmd: list[str] = field(default_factory=list)


_running: dict[str, _RunningProcess] = {}


# ---------------------------------------------------------------------------
# Command building (pure; tested directly)
# ---------------------------------------------------------------------------

def _resolve_rpc_endpoints(worker_ids: list[str]) -> list[str]:
    """Look up each worker_id and produce a `<host>:<port>` string.

    Skips:
      * worker rows that have been deleted since the split_model row
        was created (DB referential integrity is loose by design — the
        split row holds string IDs, not foreign keys, so a worker can
        be removed without breaking the split row).
      * disabled workers (the user toggled them off explicitly).
      * workers whose probe last reported `rpc_server_reachable=False`
        — passing those to llama-server would just cause connection
        refusals during the actual inference.

    Returns the endpoints in the same order as `worker_ids`. Order
    matters: llama.cpp uses it to assign layer ranges (worker[0] gets
    the first GPU-layer chunk, worker[1] the next, etc.).
    """
    out: list[str] = []
    for wid in worker_ids:
        w = db.get_compute_worker(wid)
        if not w:
            log.info("split_lifecycle: skipping unknown worker_id %r", wid)
            continue
        if not w.get("enabled"):
            log.info(
                "split_lifecycle: skipping disabled worker %r (%s)",
                w["label"], wid,
            )
            continue
        caps = w.get("capabilities") or {}
        if not caps.get("rpc_server_reachable"):
            log.info(
                "split_lifecycle: skipping worker %r — rpc-server not reachable",
                w["label"],
            )
            continue
        host = (w.get("address") or "").strip()
        # Strip http(s):// like _worker_base_url does in compute_pool.
        for prefix in ("http://", "https://"):
            if host.startswith(prefix):
                host = host[len(prefix):]
        host = host.rstrip("/")
        port = caps.get("rpc_port") or 50052
        out.append(f"{host}:{port}")
    return out


def _is_moe_model(gguf_path: str) -> bool:
    """Heuristic: does this GGUF describe a Mixture-of-Experts model?

    Reads `<arch>.expert_count` from the metadata; > 0 means MoE.
    Used to decide whether to add the `-ot` flag that pins MoE expert
    tensors to the host CUDA backend (see `_build_command` docstring
    for why).

    Returns False on any read error so non-MoE models keep their
    current command path unchanged.
    """
    try:
        import gguf
        reader = gguf.GGUFReader(gguf_path)
    except Exception:
        return False

    arch_field = reader.fields.get("general.architecture")
    if not arch_field or not arch_field.types:
        return False
    try:
        arch = arch_field.parts[arch_field.data[0]].tobytes().decode(
            "utf-8", errors="replace",
        )
    except Exception:
        return False

    f = reader.fields.get(f"{arch}.expert_count")
    if f and f.data:
        try:
            return int(f.parts[f.data[0]][0]) > 0
        except Exception:
            return False
    return False


def _host_primary_backend() -> str:
    """Map the host's detected GPU vendor to the llama.cpp backend
    name suitable for `-ot <pattern>=<backend>`.

    Used by the Gemma 3n PLE workaround to pin gemma3n-specific
    tensors to the host (so Gated Delta Net stays local). On a host
    without a recognized GPU we fall back to `CPU` — it still keeps
    those ops off RPC, just on the host's CPU instead of its GPU.
    """
    try:
        from . import sysdetect
        kind = (sysdetect.detect_system() or {}).get("gpu_kind") or ""
    except Exception:
        kind = ""
    return {
        "nvidia": "CUDA0",
        "amd": "Vulkan0",
        "intel": "SYCL0",
        "apple": "Metal",
    }.get(kind, "CPU")


def _model_needs_fit_off(gguf_path: str) -> bool:
    """Return True only for Gemma 3n PLE variants whose forward graph
    is wide enough to trip `GGML_ASSERT(n_inputs <
    GGML_SCHED_MAX_SPLIT_INPUTS)` during llama-server's auto-fit
    pre-pass (upstream issue
    https://github.com/ggml-org/llama.cpp/issues/21730).

    Empirically:
      * E2B (block_count=30) loads cleanly with the default `-fit on`
        path and produces tokens at full speed. We do NOT need
        `-fit off` for it — adding it would suppress a genuinely
        useful auto-context-fit feature.
      * E4B (block_count=35) crashes the auto-fit pre-pass; only this
        variant (and any future PLE Gemma 3n with even more blocks)
        needs the workaround.

    Detection rule:
        arch in ("gemma4", "gemma3n")        # the model family
        AND embedding_length_per_layer_input > 0    # PLE marker
        AND block_count > 30                 # graph wide enough to trip

    Standard dense Gemmas (gemma4:26b/31b) lack the PLE key entirely,
    so they're never matched and keep llama-server's normal
    adaptive-fit behavior.
    """
    try:
        import gguf
    except ImportError:
        return False
    try:
        reader = gguf.GGUFReader(gguf_path)
    except Exception:
        return False
    arch_field = reader.fields.get("general.architecture")
    if not arch_field:
        return False
    try:
        arch = arch_field.parts[arch_field.data[0]].tobytes().decode(
            "utf-8", errors="replace",
        )
    except Exception:
        return False
    if arch not in ("gemma4", "gemma3n"):
        return False
    # PLE marker present?
    has_ple = False
    for key in (
        f"{arch}.embedding_length_per_layer_input",
        "gemma3n.embedding_length_per_layer_input",
    ):
        f = reader.fields.get(key)
        if f and f.data:
            try:
                if int(f.parts[f.data[0]][0]) > 0:
                    has_ple = True
                    break
            except Exception:
                continue
    if not has_ple:
        return False
    # Graph-width guard: E2B (block_count=30) is fine; E4B (35)+ trips it.
    for key in (f"{arch}.block_count", "gemma3n.block_count",
                "gemma4.block_count"):
        f = reader.fields.get(key)
        if f and f.data:
            try:
                blocks = int(f.parts[f.data[0]][0])
                return blocks > 30
            except Exception:
                continue
    return False


def _build_command(
    *,
    llama_server: Path,
    gguf_path: str,
    port: int,
    rpc_endpoints: list[str],
    ngl: int = _DEFAULT_NGL,
    mmproj_path: str | None = None,
    draft_gguf_path: str | None = None,
) -> list[str]:
    """Assemble the argv for `llama-server`.

    Pure: takes resolved endpoints, returns a list. Caller is in charge
    of validation (port range etc.) and supplying the binary path.
    Keeping this pure means the test suite can pin the exact flag
    layout without spawning a process.

    `mmproj_path` is the path to a CLIP-format multimodal projector
    GGUF. When provided, llama-server is launched with `--mmproj`
    enabling vision/image inputs in `/v1/chat/completions`. Required
    for multimodal models like gemma4:26b whose Ollama blob bundles
    the vision tower — we extract it into a separate mmproj file
    (or download Unsloth's pre-built one) and point llama-server at
    both. Phase 2 RPC layer-split still applies to the LLM tensors;
    the CLIP graph runs on the host backend by default.

    `draft_gguf_path` is the path to a smaller GGUF that llama-server
    runs as a speculative-decoding draft alongside the main model.
    When provided, llama-server is launched with `-md <path>` plus
    tuning flags (`--draft-max 8 --draft-min 1`) so it proposes a
    short run of cheap tokens from the draft and verifies them in a
    single batched pass on the main model. Net effect: 1.3-2× single-
    stream throughput on a same-family target/draft pair, at the cost
    of holding both models in memory. The picker in
    `compute_pool.pick_draft_for` enforces the family/size constraints
    that make speculative decoding actually win.
    """
    cmd: list[str] = [
        str(llama_server),
        "--model", gguf_path,
        # Loopback bind: only the host's own routing layer talks to
        # llama-server. Exposing it on LAN would re-create Ollama's
        # auth-vs-no-auth dilemma without any benefit; the workers
        # access the model via the rpc-server protocol, not HTTP.
        "--host", "127.0.0.1",
        "--port", str(port),
        # Flash-attention: newer llama.cpp builds require an explicit
        # value (on/off/auto). 'auto' lets llama-server enable it when
        # the model + backend support it; older code passed bare `-fa`
        # which the new parser rejects (consumes the next arg as a
        # value, e.g. swallows `--jinja`).
        "-fa", "auto",
        # Use the GGUF's embedded chat template — most modern GGUFs
        # ship a Jinja template under metadata; without --jinja the
        # server falls back to a generic role-marker format that some
        # models reject.
        "--jinja",
        # Number of layers to offload to GPUs in the pool. llama.cpp
        # clamps to the model's actual layer count, so 99 effectively
        # means "as many as fit". Layers that don't fit GPUs cascade
        # to CPU+RAM.
        "-ngl", str(ngl),
        # Context size: cap at 4096 tokens. llama-server otherwise
        # allocates the model's full native context (e.g. llama3.1
        # defaults to 131k tokens / 54k effective with 4 seqs), which
        # blows up KV-cache memory on small workers — the actual
        # crash mode in early bench runs was OOM during KV buffer
        # allocation on a laptop, AFTER the model layers had already
        # loaded successfully across the pool. 4096 is plenty for
        # chat turns and keeps KV memory bounded.
        "-c", "4096",
        # Skip the empty-run warmup. By default llama-server does one
        # forward pass on an empty input to JIT-compile kernels and
        # prime caches. Across an RPC pool with 32 layers × 3 nodes
        # over LAN, that warmup takes 10+ minutes — long enough that
        # /health never reports OK within our boot timeout. The first
        # real request pays the JIT cost instead (one-shot ~5–10 s
        # extra TTFT), which is dramatically less than waiting for
        # warmup before any traffic at all.
        "--no-warmup",
        # Reuse decoded KV cache across turns when the new prompt
        # shares a prefix with the previous one. `--cache-reuse N` is
        # the minimum prefix length (in tokens) we'll bother
        # rebinding — below 256 tokens the recompute cost is comparable
        # to the cache-lookup cost, so the win shows up only on
        # longer conversations. For chat-with-context turns this is
        # huge: each follow-up message reuses the system prompt + the
        # whole prior conversation, recomputing only the new tail.
        # llama-server matches as much of the in-flight context as
        # the cache holds, so capping at 256 just means short prompts
        # skip the path. Default is 0 (disabled) upstream; we enable
        # always.
        "--cache-reuse", "256",
    ]
    # Gemma 3n PLE variants whose graph trips
    # `GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS)` need TWO
    # workarounds together (llama.cpp issue #21730):
    #
    #   1. `-fit off` — skip the auto-fit pre-pass that fires the
    #      assertion early.
    #   2. Force full-pool offload (`-ngl 99`). Empirically the
    #      assertion ALSO fires later in `sched_reserve` whenever
    #      llama-server keeps *any* layer on the host's CPU
    #      (because that creates an extra graph split with a
    #      different topology). Pushing every layer to a GPU
    #      device (host CUDA + each worker's RPC backend) keeps
    #      the split count low enough.
    #
    # Every other model keeps the normal adaptive `-fit on` path AND
    # the adaptive `-ngl` we computed above — auto-fit is genuinely
    # useful for them (sizes context to free memory, avoids OOM).
    if _model_needs_fit_off(gguf_path):
        cmd.extend(["-fit", "off"])
        # Replace the adaptive -ngl with a high constant so all
        # layers go to GPU/RPC devices.
        for i, a in enumerate(cmd):
            if a == "-ngl" and i + 1 < len(cmd):
                cmd[i + 1] = "99"
                break
        # Disable Flash Attention and force single-slot dispatch.
        # Gemma 3n's Gated Delta Net (recurrent linear-attention)
        # crashes worker rpc-servers during multi-slot init when FA
        # is enabled; both kernels rely on host-side fused paths
        # that don't have an RPC-serialized equivalent. Falling back
        # to the eager attention kernel + single slot keeps the
        # compute graph on a code path the RPC backend can dispatch.
        # Replace `-fa auto` with `-fa off`:
        for i, a in enumerate(cmd):
            if a == "-fa" and i + 1 < len(cmd):
                cmd[i + 1] = "off"
                break
        cmd.extend(["--parallel", "1"])
        # Pin Gemma 3n's PLE / MatFormer / AltUp / Laurel tensors to
        # the host's primary backend. These tensors participate in the
        # Gated Delta Net compute, which has no working RPC-dispatch
        # path in stock llama.cpp — pushing them to the workers crashes
        # rpc-server during slot init's warmup forward pass. Standard
        # transformer tensors (attn_*, ffn_*) keep auto-distributing
        # across the pool, so we still get pool memory benefits for
        # the bulk of the model; only the gemma3n-specific paths stay
        # host-local. The destination is chosen at runtime so AMD /
        # Intel / CPU-only hosts work too — not just NVIDIA.
        host_dev = _host_primary_backend()
        cmd.extend([
            "-ot", f".*(altup|laurel|per_layer|inp_gate).*={host_dev}",
        ])
    if mmproj_path:
        # Multimodal projector: required for vision-capable inference
        # of models whose Ollama blob bundles a vision tower.
        # llama-server keeps the CLIP graph on its own backend (host
        # by default) while the LLM tensors layer-split via --rpc.
        cmd.extend(["--mmproj", mmproj_path])
    if draft_gguf_path:
        # Speculative decoding: draft model runs alongside the target
        # in the same llama-server process. `--draft-max 8` caps the
        # number of tokens the draft proposes per round (8 is the
        # llama.cpp default and a balanced choice — higher values
        # benefit only when the draft accepts at >80% rate); `--draft-min 1`
        # lets short, low-confidence draft runs still try one token
        # rather than skip the round entirely. `-ngld 99` mirrors
        # `-ngl 99` for the draft so its layers also offload to GPU
        # when there's room — speculative decoding only wins when the
        # draft generates faster than the target verifies.
        cmd.extend([
            "-md", draft_gguf_path,
            "--draft-max", "8",
            "--draft-min", "1",
            "-ngld", "99",
        ])
    # No `-ot` flag here. The MoE+SYCL+RPC bug is dodged at a higher
    # layer: `compute_pool._ensure_split_running_for` switches each
    # worker's rpc-server to `-d CPU` (no SYCL exposure) before
    # spawning llama-server for an MoE model, then restores
    # `-d SYCL0,CPU` afterwards. With workers exposing only CPU,
    # llama.cpp's auto-distribution naturally fans expert tensors
    # across host CPU + every worker CPU — using ALL pool memory
    # without ever touching SYCL. Non-MoE models keep the full
    # `-d SYCL0,CPU` pool with iGPU acceleration. See
    # `_set_workers_backend` in compute_pool.
    if rpc_endpoints:
        # llama-server takes --rpc as a comma-separated list of
        # `<host>:<port>` endpoints. Order controls layer assignment.
        cmd.extend(["--rpc", ",".join(rpc_endpoints)])
    return cmd


def _log_path_for(split_id: str) -> Path:
    """Per-split log file under our private install dir.

    Living next to the binaries keeps everything Phase 2 in one
    directory; `~/.gigachat/llama-cpp/logs/<id>.log`. Created lazily.
    """
    log_dir = split_runtime.LLAMA_CPP_INSTALL_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{split_id}.log"


# ---------------------------------------------------------------------------
# Lifecycle: start / stop / status
# ---------------------------------------------------------------------------

class SplitLifecycleError(RuntimeError):
    """Raised on any pre-flight failure (binary missing, GGUF missing,
    port already in use, etc.). Caller surfaces the message to the UI."""


def _preflight(row: dict) -> tuple[Path, list[str]]:
    """Resolve binary + endpoints + sanity-check the GGUF before spawn.

    Returns (llama_server_path, rpc_endpoints). Raises
    SplitLifecycleError with a user-readable message on any miss.
    """
    server = split_runtime.find_llama_server()
    if not server:
        raise SplitLifecycleError(
            "llama.cpp not installed — install via Settings → Compute → "
            "Install llama.cpp, then start this split model."
        )

    gguf = (row.get("gguf_path") or "").strip()
    if not gguf:
        raise SplitLifecycleError("split_model has no gguf_path")
    if not Path(gguf).is_file():
        raise SplitLifecycleError(
            f"gguf_path does not exist on host: {gguf}"
        )

    rpc = _resolve_rpc_endpoints(row.get("worker_ids") or [])
    return server, rpc


async def _wait_for_health(
    port: int,
    timeout: float | None = None,
    proc: subprocess.Popen | None = None,
) -> None:
    """Poll `http://127.0.0.1:<port>/health` until it returns 200, or
    raise SplitLifecycleError on timeout — or as soon as the child
    process exits, whichever comes first.

    llama-server's `/health` returns:
      200 + {"status":"ok"}     once the model is loaded and serving
      503                       while the model is still loading
      (no response)             before the HTTP server has bound

    We treat all non-200s and connection-refused as "not ready yet"
    and keep polling until the timeout fires.

    If `proc` is provided we also poll the child's exit status each
    loop. A dead child means the model failed to load (architecture
    mismatch, OOM, missing tensor, etc.); waiting the full timeout
    in that case is just dead time. Short-circuit and surface the
    failure immediately.
    """
    # Read the constant at call time, not at function-def time, so
    # tests / runtime tweaks of `_BOOT_TIMEOUT_SEC` are respected.
    if timeout is None:
        timeout = _BOOT_TIMEOUT_SEC
    deadline = time.monotonic() + timeout
    last_err: str | None = None
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            # Bail early if the child process has died — no point
            # polling a port that nothing is going to bind to.
            if proc is not None and proc.poll() is not None:
                raise SplitLifecycleError(
                    f"llama-server exited with code {proc.returncode} "
                    f"before becoming healthy (last error: {last_err or 'n/a'})"
                )
            try:
                r = await client.get(f"http://127.0.0.1:{port}/health")
                if r.status_code == 200:
                    return
                last_err = f"HTTP {r.status_code}"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(_HEALTH_POLL_SEC)
    raise SplitLifecycleError(
        f"llama-server on port {port} did not become healthy within "
        f"{timeout:.0f}s (last error: {last_err})"
    )


def _compute_optimal_ngl(
    gguf_path: str, worker_ids: list[str], *, safety: float = 0.7,
) -> int:
    """Decide how many layers to put on GPU pools (rest stay on host
    CPU + RAM, paged from disk via mmap).

    Why this matters: llama-server's auto-distribution with `-ngl 99`
    optimistically tries to put every layer on a GPU device (host CUDA
    + each RPC worker). If the combined GPU memory is barely enough,
    one device's allocator overcommits and the rpc-server crashes
    mid-load — this is the "Remote RPC server crashed" failure mode
    the targeted bench exposed for gemma4:26b on tight pool memory.

    Ollama's runtime sidesteps this by mmap'ing the GGUF on the host
    and lazily paging layers in only when computing them — RAM
    pressure pushes pages out, OS swap absorbs the overflow, no
    explicit OOM. We can't replicate mmap on the worker side
    (rpc-server has no GGUF file; layers arrive over network and
    must be resident), but we CAN cap GPU offload to the conservative
    fit and let the remaining layers ride host CPU + mmap'd file
    bytes — same trick Ollama uses on the host side.

    Algorithm: sum (host VRAM + each worker's free RAM) × `safety`,
    divide by average bytes per layer (file size / block_count from
    the GGUF metadata), clamp to [0, n_layers]. Returns the integer
    `-ngl` value to pass.

    Falls back to `_DEFAULT_NGL` if any input is missing — e.g.
    a worker without a probe yet, or a GGUF without the
    `<arch>.block_count` metadata key. The default `-ngl 99` keeps
    llama.cpp's old behaviour where it works (small models that fit
    comfortably).
    """
    try:
        import gguf
    except ImportError:
        return _DEFAULT_NGL

    # File size as a proxy for total weight bytes — close enough since
    # quantization metadata + tokenizer KV are small relative to weights.
    try:
        file_size = os.path.getsize(gguf_path)
    except OSError:
        return _DEFAULT_NGL
    if file_size <= 0:
        return _DEFAULT_NGL

    # Read block count from the GGUF metadata (the correct metadata key
    # is architecture-specific: `gemma4.block_count`, `llama.block_count`,
    # etc.). We probe the file once and walk known keys.
    try:
        reader = gguf.GGUFReader(gguf_path)
    except Exception:
        return _DEFAULT_NGL

    arch_field = reader.fields.get("general.architecture")
    if not arch_field or not arch_field.types:
        return _DEFAULT_NGL
    try:
        arch = arch_field.parts[arch_field.data[0]].tobytes().decode(
            "utf-8", errors="replace",
        )
    except Exception:
        return _DEFAULT_NGL

    n_layers = 0
    for key in (f"{arch}.block_count", "llama.block_count", "general.block_count"):
        f = reader.fields.get(key)
        if f and f.data:
            try:
                n_layers = int(f.parts[f.data[0]][0])
                break
            except Exception:
                continue
    if n_layers <= 0:
        return _DEFAULT_NGL

    avg_layer_bytes = file_size / n_layers

    # Sum free pool memory: host VRAM + each enabled worker's free RAM.
    # Worker iGPUs (Intel SYCL) draw from system RAM as shared GPU
    # memory, so `ram_free_gb` is a reasonable upper bound for what
    # SYCL can allocate on that worker.
    pool_free_bytes = 0
    try:
        from . import sysdetect
        spec = sysdetect.detect_system()
        # Host VRAM (CUDA / dGPU) — `vram_gb` is total; we approximate
        # free as 80% of total (OS / driver reservation).
        pool_free_bytes += int(float(spec.get("vram_gb") or 0) * (1024 ** 3) * 0.8)
    except Exception:
        pass

    for wid in worker_ids:
        w = db.get_compute_worker(wid)
        if not w or not w.get("enabled"):
            continue
        caps = w.get("capabilities") or {}
        free_gb = float(caps.get("ram_free_gb") or 0)
        pool_free_bytes += int(free_gb * (1024 ** 3))

    if pool_free_bytes <= 0:
        return _DEFAULT_NGL

    fits = int(pool_free_bytes * safety / avg_layer_bytes)
    fits = max(0, min(n_layers, fits))
    log.info(
        "split_lifecycle: adaptive ngl: file_size=%.2f GB n_layers=%d "
        "avg_layer=%.0f MiB pool_free=%.2f GB safety=%.2f -> ngl=%d",
        file_size / (1024 ** 3), n_layers, avg_layer_bytes / (1024 ** 2),
        pool_free_bytes / (1024 ** 3), safety, fits,
    )
    return fits


async def start(split_id: str) -> dict:
    """Bring up the llama-server for one split_model row.

    Idempotent: if it's already running according to our registry, we
    return the existing status without spawning a duplicate.
    """
    row = db.get_split_model(split_id)
    if not row:
        raise SplitLifecycleError("split_model not found")

    # Already running? Caller probably hit the API twice; just report.
    if split_id in _running and _running[split_id].proc.poll() is None:
        return {
            "ok": True,
            "status": "running",
            "port": _running[split_id].port,
            "note": "already running",
        }

    # Mark loading BEFORE spawning so the UI can render "starting…"
    # immediately on the next list call. If preflight raises, we'll
    # clear it back to error/stopped below.
    db.update_split_model_status(split_id, status="loading", last_error="")

    try:
        server, rpc_endpoints = _preflight(row)
    except SplitLifecycleError as e:
        db.update_split_model_status(split_id, status="error", last_error=str(e))
        return {"ok": False, "status": "error", "error": str(e)}

    # mmproj_path is optional — when set, we pass --mmproj so vision
    # input works. Older split rows (pre-migration) won't have the
    # column; .get() returns None which the builder ignores.
    mmproj = (row.get("mmproj_path") or "").strip() or None

    # Compute -ngl adaptively based on real pool free memory + GGUF
    # metadata. This is the Ollama-style "spill to host CPU + mmap"
    # trick: layers that don't fit GPU pools stay on host CPU/RAM
    # paged from the GGUF file, instead of the optimistic
    # `-ngl 99` push that overcommits and crashes one rpc-server.
    # Falls back to _DEFAULT_NGL if metadata or pool state is
    # missing — small models keep their old fast path.
    worker_ids = row.get("worker_ids") or []
    ngl = _compute_optimal_ngl(row["gguf_path"], worker_ids)

    # Speculative-decoding draft: optional, set by the router when a
    # smaller same-family chat model is available somewhere in the pool.
    # `_build_command` adds `-md <path>` + tuning flags when present.
    draft = (row.get("draft_gguf_path") or "").strip() or None

    cmd = _build_command(
        llama_server=server,
        gguf_path=row["gguf_path"],
        port=row["llama_port"],
        rpc_endpoints=rpc_endpoints,
        mmproj_path=mmproj,
        draft_gguf_path=draft,
        ngl=ngl,
    )
    log_path = _log_path_for(split_id)

    # Spawn detached enough that the child outlives our Python
    # process if it's restarted, but kept under the same console
    # group so we can SIGTERM it cleanly. On Windows that's
    # CREATE_NEW_PROCESS_GROUP; on POSIX, no special flags.
    #
    # Resource cooperation: launch llama-server at BELOW_NORMAL
    # priority. With this flag, the OS scheduler gives priority to
    # whatever app the user is interacting with — if they open a
    # browser / play a video / launch any other workload, the
    # scheduler de-prioritizes our background inference so the user's
    # foreground work stays responsive. On idle host (no other
    # workload), our process gets full CPU so inference still runs
    # at top speed. "Use as much as possible when free, yield to
    # foreground apps when busy" — the standard cooperative pattern.
    creationflags = 0
    if sys.platform == "win32":
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.BELOW_NORMAL_PRIORITY_CLASS
        )

    log_file = log_path.open("ab")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,   # merge into one log file
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            # POSIX: nice the process to +10 (lower priority) so the
            # kernel CFS scheduler also de-prioritizes it under load.
            preexec_fn=(_lower_priority_posix if sys.platform != "win32" else None),
        )
    except Exception as e:
        log_file.close()
        msg = f"failed to spawn llama-server: {type(e).__name__}: {e}"
        db.update_split_model_status(split_id, status="error", last_error=msg)
        return {"ok": False, "status": "error", "error": msg}

    _running[split_id] = _RunningProcess(
        proc=proc,
        port=row["llama_port"],
        started_at=time.time(),
        log_path=log_path,
        cmd=cmd,
    )

    # Wait for the HTTP server to come up. If health never reports OK,
    # we kill the child and surface the timeout. Pass `proc` so the
    # waiter short-circuits the moment the child crashes (e.g. the
    # GGUF architecture isn't supported by this llama.cpp build) —
    # otherwise we'd burn the full _BOOT_TIMEOUT_SEC polling a port
    # that nothing will ever bind to.
    try:
        await _wait_for_health(row["llama_port"], proc=proc)
    except SplitLifecycleError as e:
        # Best-effort cleanup so we don't leave a half-loaded process
        # hogging VRAM.
        await stop(split_id, _from_failed_start=True)
        db.update_split_model_status(split_id, status="error", last_error=str(e))
        return {"ok": False, "status": "error", "error": str(e)}

    db.update_split_model_status(split_id, status="running", last_error="")
    return {
        "ok": True,
        "status": "running",
        "port": row["llama_port"],
        "log_path": str(log_path),
    }


async def stop(split_id: str, *, _from_failed_start: bool = False) -> dict:
    """Terminate the llama-server child and update DB status.

    Idempotent: stopping an already-stopped row is a no-op success.
    `_from_failed_start` is internal — when start() bailed because
    /health never came up, we still need to kill the child but we
    DON'T want to overwrite the error status that start() already
    recorded.
    """
    rp = _running.pop(split_id, None)
    if rp is None:
        # Nothing in our registry. Reset DB if it claims `running` —
        # the row's status is stale (process died, app restarted, etc.).
        row = db.get_split_model(split_id)
        if row and row["status"] in ("loading", "running"):
            db.update_split_model_status(split_id, status="stopped")
        return {"ok": True, "status": "stopped", "note": "was not running"}

    if rp.proc.poll() is None:
        try:
            rp.proc.terminate()
            try:
                rp.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                rp.proc.kill()
                rp.proc.wait(timeout=2.0)
        except Exception as e:
            log.warning("split_lifecycle: error stopping %s: %s", split_id, e)

    if not _from_failed_start:
        db.update_split_model_status(split_id, status="stopped")
    return {"ok": True, "status": "stopped", "port": rp.port}


def status(split_id: str) -> dict:
    """Read-only status snapshot — combines DB row + process registry.

    The DB row is authoritative for "what the user sees" (status,
    last_error). The registry tells us if the OS-level process is
    still alive. Mismatches (DB says `running` but process is gone)
    are common after a crash; we report them as `crashed`.
    """
    row = db.get_split_model(split_id)
    if not row:
        return {"ok": False, "error": "not found"}
    rp = _running.get(split_id)
    pid = None
    alive = False
    if rp:
        pid = rp.proc.pid
        alive = rp.proc.poll() is None
    out = {
        "ok": True,
        "id": split_id,
        "label": row["label"],
        "db_status": row["status"],
        "last_error": row["last_error"],
        "pid": pid,
        "alive": alive,
        "port": rp.port if rp else row["llama_port"],
    }
    # Effective status reconciles DB vs registry.
    if row["status"] == "running" and not alive:
        out["effective_status"] = "crashed"
    else:
        out["effective_status"] = row["status"]
    return out


async def stop_all() -> None:
    """Shutdown helper — terminate every running llama-server child.

    Called from the FastAPI shutdown hook so uvicorn exits cleanly
    (otherwise the children leak GPU memory until OS reclaims them).
    """
    ids = list(_running.keys())
    for sid in ids:
        try:
            await stop(sid)
        except Exception as e:
            log.warning("split_lifecycle: stop_all error for %s: %s", sid, e)


def reconcile_on_boot() -> int:
    """Reset DB rows that claim `running` / `loading` but have no process.

    Scenario: the app crashed (or was Ctrl-C'd hard) while a split
    model was active. The DB row still says `running`, but the
    llama-server child is gone — Windows reclaimed it when uvicorn
    died. Without this reset, the UI would show a phantom green pill
    and the routing layer would happily try to send chat to a port
    nothing is listening on.

    Called once at app startup. Returns the number of rows that were
    reset so the startup hook can log it.
    """
    reset_count = 0
    for row in db.list_split_models():
        if row.get("status") in ("running", "loading"):
            db.update_split_model_status(
                row["id"],
                status="stopped",
                last_error="reset on app boot — process did not survive restart",
            )
            reset_count += 1
    return reset_count


def read_log_tail(split_id: str, lines: int = 100) -> str:
    """Return the last `lines` lines of the per-split log file.

    Used by the Settings UI to show what llama-server is actually
    saying — most start failures (port already bound, GGUF too new
    for this build, OOM during layer offload) surface here BEFORE
    the bare 'failed to start' status reaches the API. Returns empty
    string if the log doesn't exist yet.

    No streaming/follow — keep it simple. The UI fetches once on
    open and re-fetches on user click.
    """
    log_path = _log_path_for(split_id)
    if not log_path.is_file():
        return ""
    try:
        with log_path.open("rb") as f:
            # Cheap tail: read up to ~256 KB and split. For typical
            # llama-server logs (~few KB total in the first few minutes)
            # this is plenty; a runaway log gets truncated to the most
            # recent slice, which is what the operator wants anyway.
            f.seek(0, 2)  # end
            size = f.tell()
            read_bytes = min(size, 256 * 1024)
            f.seek(size - read_bytes)
            data = f.read()
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace")
    out_lines = text.splitlines()
    if len(out_lines) > lines:
        out_lines = out_lines[-lines:]
    return "\n".join(out_lines)
