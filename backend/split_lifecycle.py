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


def _read_gguf_int(reader, arch: str, *keys: str) -> int | None:
    """Look up the first available integer field in a GGUF reader.

    Tries each fully-qualified key in order (architecture-prefixed first,
    bare-architecture fallback last). Returns None when no key is
    present or the field doesn't decode to an int.
    """
    for key in keys:
        f = reader.fields.get(key)
        if f and f.data:
            try:
                return int(f.parts[f.data[0]][0])
            except Exception:
                continue
    return None


def _estimate_kv_bytes_per_slot(gguf_path: str, ctx_size: int = 4096) -> int:
    """Estimate the KV-cache size for ONE parallel slot at the given context.

    Each `--parallel` slot pre-allocates its own KV cache, so the per-slot
    size sets the upper bound on how many slots fit in remaining VRAM.
    Formula:

        2 (K+V) × n_layers × n_kv_heads × head_dim × ctx_size × 2 bytes (FP16)

    GQA-aware: `n_kv_heads` differs from `n_heads` on Llama 3 / Qwen 2 /
    most modern checkpoints. We read `<arch>.attention.head_count_kv` when
    present and fall back to `<arch>.attention.head_count` (MHA) when it
    isn't.

    Returns 0 on any metadata miss so callers fall back to a conservative
    `--parallel 1`.
    """
    try:
        import gguf
    except ImportError:
        return 0
    try:
        reader = gguf.GGUFReader(gguf_path)
    except Exception:
        return 0

    arch_field = reader.fields.get("general.architecture")
    if not arch_field or not arch_field.types:
        return 0
    try:
        arch = arch_field.parts[arch_field.data[0]].tobytes().decode(
            "utf-8", errors="replace",
        )
    except Exception:
        return 0

    n_layers = _read_gguf_int(
        reader, arch, f"{arch}.block_count", "llama.block_count",
    )
    embedding_length = _read_gguf_int(
        reader, arch, f"{arch}.embedding_length", "llama.embedding_length",
    )
    n_heads = _read_gguf_int(
        reader, arch, f"{arch}.attention.head_count", "llama.attention.head_count",
    )
    if not n_layers or not embedding_length or not n_heads:
        return 0

    n_kv_heads = _read_gguf_int(
        reader, arch,
        f"{arch}.attention.head_count_kv",
        "llama.attention.head_count_kv",
    ) or n_heads

    head_dim = embedding_length // n_heads
    if head_dim <= 0:
        return 0

    # 2 (K+V) × layers × kv_heads × head_dim × bytes_per_element (FP16=2)
    kv_per_token = 2 * n_layers * n_kv_heads * head_dim * 2
    return int(kv_per_token * max(ctx_size, 1))


# How much VRAM headroom to leave free after target + draft + KV slots.
# 15% mirrors the safety margin sysdetect already bakes into vram budgeting
# elsewhere — covers OS / display / driver overhead and small allocations
# llama.cpp does outside the headline buffers.
_PARALLEL_VRAM_HEADROOM = 0.15

# Hard cap on `--parallel`. llama-server's batched-verify is most efficient
# at 4-8 slots; beyond that, scheduling overhead and sub-slot cache
# pressure typically erode the win. 8 is also Ollama's default ceiling
# in recent releases — keeping the two engines in sync makes the chat
# experience consistent regardless of which one route_chat_for picks.
_PARALLEL_MAX_SLOTS = 8


def _compute_optimal_parallel(
    gguf_path: str,
    worker_ids: list[str],
    *,
    ctx_size: int = 4096,
    target_size_bytes: int | None = None,
    draft_size_bytes: int = 0,
    kv_size_multiplier: float = 1.0,
) -> int:
    """Decide how many `--parallel` decoding slots to allocate.

    Saturates GPU compute when multiple streams are in flight (concurrent
    chats, `delegate_parallel` subagents, speculative verify+draft) without
    paying KV-cache for slots we'll never use. Auto-adapts to whatever
    free memory the executing node actually has — no per-vendor / per-OS
    branches in the call sites.

    Algorithm:
      * Estimate KV-per-slot from the GGUF (architecture-aware, GQA-aware).
      * Sum free memory across the executing pool. For host-only paths
        (no rpc workers) that's host VRAM minus target+draft weights.
        For split paths (rpc workers present) the constraint becomes
        `min(host_free, every worker's free RAM)` — the slowest device
        bounds parallel slot count, since llama-server replicates the
        slot KV layout on each rpc-server.
      * Apply `_PARALLEL_VRAM_HEADROOM` so OS / driver / scratch
        allocations have room to grow.
      * Clamp to `[1, _PARALLEL_MAX_SLOTS]`.

    Returns 1 on any metadata miss (KV size unknown) so behavior matches
    the legacy single-slot path when we can't make an informed call.
    """
    kv_per_slot = _estimate_kv_bytes_per_slot(gguf_path, ctx_size)
    if kv_per_slot <= 0:
        return 1
    # KV quantization (Q8 ≈ 50%, Q4 ≈ 25%) shrinks per-slot KV by the
    # multiplier, so caller can simulate slot count under different
    # cache types. Default 1.0 = FP16 baseline.
    kv_per_slot = max(1, int(kv_per_slot * max(0.1, float(kv_size_multiplier))))

    if target_size_bytes is None:
        try:
            target_size_bytes = os.path.getsize(gguf_path)
        except OSError:
            target_size_bytes = 0

    # Host VRAM via sysdetect (vendor-agnostic — covers NVIDIA / AMD /
    # Intel / Apple / unified-memory / CPU-only).
    try:
        from . import sysdetect
        spec = sysdetect.detect_system()
        host_vram_total = int(float(spec.get("vram_gb") or 0) * (1024 ** 3))
        host_ram_total = int(float(spec.get("ram_gb") or 0) * (1024 ** 3))
    except Exception:
        host_vram_total = 0
        host_ram_total = 0

    # Effective per-node budget. CPU-only hosts (vram=0) use system RAM
    # for KV; GPU hosts use VRAM. Apple Silicon's unified memory shows up
    # as `vram_gb` already (sysdetect maps the unified pool there).
    host_budget = host_vram_total if host_vram_total > 0 else host_ram_total
    if host_budget <= 0:
        return 1

    host_free = int(host_budget * (1.0 - _PARALLEL_VRAM_HEADROOM))
    host_free -= int(target_size_bytes or 0)
    host_free -= int(draft_size_bytes or 0)
    if host_free <= 0:
        return 1

    bottleneck_free = host_free

    # Split path: each rpc-server replicates the slot KV layout for its
    # assigned layer chunk, so the worker with the smallest free RAM
    # gates the slot count. Walk every enabled worker; skip the ones
    # without spec data (probe hasn't filled them in yet) so absence
    # doesn't pin us to 1 slot needlessly.
    if worker_ids:
        for wid in worker_ids:
            w = db.get_compute_worker(wid)
            if not w or not w.get("enabled"):
                continue
            caps = w.get("capabilities") or {}
            free_gb = float(caps.get("ram_free_gb") or 0)
            if free_gb <= 0:
                continue
            worker_free = int(free_gb * (1024 ** 3) * (1.0 - _PARALLEL_VRAM_HEADROOM))
            if worker_free < bottleneck_free:
                bottleneck_free = worker_free

    slots = max(1, bottleneck_free // kv_per_slot)
    slots = min(_PARALLEL_MAX_SLOTS, int(slots))
    log.info(
        "split_lifecycle: adaptive parallel: kv_per_slot=%.0f MiB "
        "host_free=%.2f GB bottleneck_free=%.2f GB -> --parallel %d",
        kv_per_slot / (1024 ** 2),
        host_free / (1024 ** 3),
        bottleneck_free / (1024 ** 3),
        slots,
    )
    return slots


# KV cache quantization. Q8 halves the per-slot KV cost vs FP16 for a
# small accuracy drop (typ. < 1 % on public benchmarks). When the
# pool is memory-pressured (FP16 forces `--parallel 1` or 2), switching
# to Q8 lets us pack 2-4× more slots in the same VRAM — strictly more
# throughput on concurrent workloads (subagent fan-out, multi-conv).
# Pools that already fit `_PARALLEL_MAX_SLOTS` at FP16 stay on FP16.
_KV_QUANT_PARALLEL_FLOOR = 4

# Adaptive context-window upper bound. We never set `-c` above the
# model's own training-time max (read from GGUF metadata), and never
# below the chat baseline of 4096 tokens (the legacy hard-coded value
# — preserves behavior on tiny pools that can't pay for more).
_CTX_SIZE_FLOOR = 4096
_CTX_SIZE_CEILING = 131072  # 128 K — even huge pools stop here


def _decide_kv_precision_and_parallel(
    gguf_path: str,
    worker_ids: list[str],
    *,
    target_size_bytes: int,
    draft_size_bytes: int = 0,
    ctx_size: int = 4096,
) -> tuple[str | None, int]:
    """Joint decision: pick KV cache type AND `--parallel` slot count.

    Returns ``(cache_type, parallel)`` where ``cache_type`` is either
    ``None`` (FP16 — llama.cpp default) or ``"q8_0"`` (Q8 — half the
    KV memory). The slot count is computed against whichever precision
    we picked, so callers see a coherent (precision, slots) pair.

    Decision: try FP16 first. If the pool is memory-pressured enough
    that fewer than ``_KV_QUANT_PARALLEL_FLOOR`` slots fit in FP16,
    re-check at Q8 and switch when Q8 unlocks meaningfully more slots.
    Pools with plenty of headroom stay on FP16 (no precision penalty
    when there's no bottleneck to break).
    """
    fp16_parallel = _compute_optimal_parallel(
        gguf_path, worker_ids,
        ctx_size=ctx_size,
        target_size_bytes=target_size_bytes,
        draft_size_bytes=draft_size_bytes,
    )
    if fp16_parallel >= _KV_QUANT_PARALLEL_FLOOR:
        # Plenty of slots already at FP16 — keep precision, skip the
        # Q8 ladder.
        return None, fp16_parallel

    q8_parallel = _compute_optimal_parallel(
        gguf_path, worker_ids,
        ctx_size=ctx_size,
        target_size_bytes=target_size_bytes,
        draft_size_bytes=draft_size_bytes,
        kv_size_multiplier=0.5,
    )
    if q8_parallel > fp16_parallel:
        log.info(
            "split_lifecycle: KV quantized to q8_0 — fp16 fit %d slot(s), "
            "q8_0 fits %d", fp16_parallel, q8_parallel,
        )
        return "q8_0", q8_parallel
    return None, fp16_parallel


def _compute_optimal_ctx_size(
    gguf_path: str,
    worker_ids: list[str],
    *,
    parallel: int,
    cache_type: str | None,
    target_size_bytes: int,
    draft_size_bytes: int = 0,
) -> int:
    """Pick the largest context window the pool can hold.

    KV cache scales linearly with context size, so on a rich pool
    (host with 80 GB combined memory after target weights) we can
    afford 32 K-128 K context windows; on a tight pool we stick with
    the legacy 4 K floor.

    Algorithm:
      * Read GGUF's training-time max context (`<arch>.context_length`).
      * Compute KV bytes per token from `_estimate_kv_bytes_per_slot`
        at ctx=1, then scale by `parallel` and the cache-type multiplier.
      * Sum free pool memory (host + worker-min, same logic as
        `_compute_optimal_parallel`'s bottleneck).
      * Return ``min(model_max, free_mem // (kv_per_token *
        parallel))`` clamped to ``[_CTX_SIZE_FLOOR, _CTX_SIZE_CEILING]``.

    Returns ``_CTX_SIZE_FLOOR`` (4 K) when metadata is missing —
    matches the legacy hard-coded value so behavior is unchanged when
    we can't make an informed call.
    """
    kv_per_token = _estimate_kv_bytes_per_slot(gguf_path, ctx_size=1)
    if kv_per_token <= 0:
        return _CTX_SIZE_FLOOR
    if cache_type == "q8_0":
        kv_per_token = max(1, kv_per_token // 2)

    # Read model's training-time max context (don't exceed it — the
    # weights only learned positional encodings up to that length).
    try:
        import gguf
        reader = gguf.GGUFReader(gguf_path)
        arch_field = reader.fields.get("general.architecture")
        arch = (
            arch_field.parts[arch_field.data[0]].tobytes().decode("utf-8", errors="replace")
            if arch_field and arch_field.types
            else None
        )
        model_max_ctx = (
            _read_gguf_int(reader, arch, f"{arch}.context_length", "llama.context_length")
            if arch else None
        ) or _CTX_SIZE_CEILING
    except Exception:
        model_max_ctx = _CTX_SIZE_CEILING

    # Bottleneck-node free memory (same shape as parallel computation).
    try:
        from . import sysdetect
        spec = sysdetect.detect_system()
        host_vram = int(float(spec.get("vram_gb") or 0) * (1024 ** 3))
        host_ram = int(float(spec.get("ram_gb") or 0) * (1024 ** 3))
    except Exception:
        return _CTX_SIZE_FLOOR
    host_budget = host_vram if host_vram > 0 else host_ram
    if host_budget <= 0:
        return _CTX_SIZE_FLOOR

    free_after_target = int(host_budget * (1.0 - _PARALLEL_VRAM_HEADROOM))
    free_after_target -= int(target_size_bytes or 0) + int(draft_size_bytes or 0)
    if free_after_target <= 0:
        return _CTX_SIZE_FLOOR

    bottleneck_free = free_after_target
    for wid in worker_ids:
        w = db.get_compute_worker(wid)
        if not w or not w.get("enabled"):
            continue
        caps = w.get("capabilities") or {}
        free_gb = float(caps.get("ram_free_gb") or 0)
        if free_gb <= 0:
            continue
        worker_free = int(free_gb * (1024 ** 3) * (1.0 - _PARALLEL_VRAM_HEADROOM))
        if worker_free < bottleneck_free:
            bottleneck_free = worker_free

    fits = bottleneck_free // (kv_per_token * max(1, parallel))
    fits = min(fits, model_max_ctx, _CTX_SIZE_CEILING)
    fits = max(_CTX_SIZE_FLOOR, int(fits))
    log.info(
        "split_lifecycle: adaptive ctx: kv_per_token=%d parallel=%d "
        "bottleneck_free=%.2f GB model_max=%d -> -c %d",
        kv_per_token, parallel, bottleneck_free / (1024 ** 3),
        model_max_ctx, fits,
    )
    return fits


# Heterogeneity threshold — only emit `-ts` (per-device layer weights)
# when the pool is meaningfully uneven. Below this ratio, llama.cpp's
# internal per-device memory query already produces near-optimal splits
# and forcing weights can hurt by overriding its real-time view of free
# memory. 1.5× = strongest device has 50%+ more headroom than the
# weakest; that's the floor where weighted distribution is empirically
# better than equal split.
_TS_HETEROGENEITY_RATIO = 1.5

# Wired-LAN latency floor for engaging `--split-mode row` (tensor-parallel
# row dispatch). Row mode synchronizes per-layer matmul partial sums
# across devices, so per-token RPC chatter is much higher than layer-
# pipeline mode. Empirically wins on Gigabit-Ethernet (typ. 1-3 ms probe
# round-trip) and loses on Wi-Fi (typ. 8-30 ms). The check is
# conservative — when in doubt we stay on layer-pipeline.
_ROW_SPLIT_LATENCY_CEILING_MS = 4


def _compute_tensor_split_ratios(
    gguf_path: str, worker_ids: list[str],
) -> list[int] | None:
    """Compute per-device weights for `-ts` based on real free memory.

    When the pool is heterogeneous (e.g. host=24 GB GPU + worker=4 GB
    iGPU + worker=8 GB iGPU), equal-split sends 1/3 of the layers to
    each — but the smallest device OOMs. Weighted distribution sized
    by each node's actual free pool memory keeps every node within its
    budget while pushing as many layers onto fast devices as possible.

    Returns a list of integer weights `[host, worker_1, worker_2, ...]`
    in the SAME order `_resolve_rpc_endpoints` produces — so llama.cpp
    pairs them to `--rpc` endpoints correctly. Host weight is always
    first because llama-server enumerates the local backend before any
    RPC backends.

    Returns ``None`` when:
      * No RPC workers (no point — `-ts` only matters with multiple
        devices).
      * Pool is roughly homogeneous (max/min ≤ `_TS_HETEROGENEITY_RATIO`).
        llama.cpp's built-in per-device free-memory query handles even
        pools well enough that overriding can hurt.
      * Any device's free-memory data is missing — falls back to
        llama.cpp's defaults rather than assigning a guess.
    """
    if not worker_ids:
        return None

    # Host capacity: prefer VRAM (GPU is faster than CPU+RAM); fall back
    # to RAM for CPU-only hosts. Apple Silicon's unified pool already
    # shows up in vram_gb so this single-axis read covers all vendors.
    try:
        from . import sysdetect
        spec = sysdetect.detect_system()
        host_vram_gb = float(spec.get("vram_gb") or 0)
        host_ram_gb = float(spec.get("ram_gb") or 0)
    except Exception:
        return None
    host_capacity_gb = host_vram_gb if host_vram_gb > 0 else host_ram_gb
    if host_capacity_gb <= 0:
        return None

    weights: list[float] = [host_capacity_gb]
    for wid in worker_ids:
        w = db.get_compute_worker(wid)
        if not w or not w.get("enabled"):
            return None  # missing worker — can't size; let llama.cpp pick
        caps = w.get("capabilities") or {}
        # Worker capacity prefers `max_vram_seen_bytes` (proven GPU memory
        # from `/api/ps`) and falls back to `ram_free_gb` from the SSH
        # spec probe. iGPUs draw from system RAM so ram_free_gb is the
        # right ceiling for them; dGPUs report VRAM via /api/ps.
        vram_bytes = int(caps.get("max_vram_seen_bytes") or 0)
        ram_free_gb = float(caps.get("ram_free_gb") or 0)
        capacity_gb = max(vram_bytes / (1024 ** 3), ram_free_gb)
        if capacity_gb <= 0:
            return None
        weights.append(capacity_gb)

    if len(weights) < 2:
        return None

    smallest = min(weights)
    largest = max(weights)
    if smallest <= 0:
        return None
    if largest / smallest < _TS_HETEROGENEITY_RATIO:
        return None  # roughly homogeneous — let llama.cpp's auto handle

    # Convert to integer ratios — llama.cpp accepts `-ts a/b/c` with
    # whatever scale you pick. Multiply by 10 and round so a weight
    # of 7.4 GB becomes 74 (vs 7), preserving relative proportions
    # better at the rounding boundary.
    return [max(1, int(round(w * 10))) for w in weights]


def _should_pin_experts_to_cpu(gguf_path: str) -> bool:
    """Return True for MoE models whose expert tensors plausibly won't
    fit host GPU VRAM, so they're better placed on CPU+RAM (host or
    via RPC workers).

    MoE models (Mixtral, DeepSeek-V3, Qwen 2.5 MoE) carry expert
    sub-networks that dominate the model's on-disk size — typically
    70-90 % of the file. Stock llama.cpp tries to put experts on GPU
    by default, which OOMs the moment target_size > host_vram. Pinning
    experts to CPU keeps the attention path on GPU (where matmul
    speedups matter) while expert weights ride host RAM + every
    worker's RAM via the existing `-d CPU` RPC distribution.

    Heuristic: model is MoE (`_is_moe_model`) AND file_size > host
    VRAM. The constant `0.85` matches the safety factor used elsewhere
    for the OS / display / driver overhead. Models that fit comfortably
    in VRAM keep llama.cpp's default placement (no override here).
    Returns False on any read error so non-MoE models are unaffected.
    """
    try:
        if not _is_moe_model(gguf_path):
            return False
    except Exception:
        return False
    try:
        file_size = os.path.getsize(gguf_path)
    except OSError:
        return False

    try:
        from . import sysdetect
        spec = sysdetect.detect_system()
        host_vram_bytes = int(float(spec.get("vram_gb") or 0) * (1024 ** 3))
    except Exception:
        return False

    # Pin only when experts genuinely won't fit GPU. Threshold is
    # generous (file_size includes attn weights + KV scratch + overhead,
    # not just experts), so a model that's bigger than host VRAM almost
    # certainly has experts that won't either.
    return host_vram_bytes > 0 and file_size > int(host_vram_bytes * 0.85)


def _should_use_row_split(worker_ids: list[str]) -> bool:
    """Return True when the LAN is fast enough that `--split-mode row`
    might beat the default layer-pipeline.

    Row-mode tensor-parallel synchronizes partial sums per layer, so
    every token requires several round-trips per RPC worker. On a
    Gigabit-Ethernet LAN (typ. 1-3 ms RTT) the extra chatter is fast
    enough to amortize against the row-parallel matmul speedup; on
    typical Wi-Fi (8-30 ms RTT) the synchronization cost dominates and
    pipeline-layer wins.

    We use the worker-probe latency as a real-time bandwidth proxy —
    `_probe_one` already records `probe_latency_ms` on every sweep, so
    no extra measurement traffic is needed. When ANY worker exceeds
    the latency ceiling we stay on layer mode, since row-mode's slowest
    sync gates the whole thing.
    """
    if not worker_ids:
        return False
    for wid in worker_ids:
        w = db.get_compute_worker(wid)
        if not w or not w.get("enabled"):
            return False
        caps = w.get("capabilities") or {}
        latency_ms = int(caps.get("probe_latency_ms") or 999)
        if latency_ms > _ROW_SPLIT_LATENCY_CEILING_MS:
            return False
    return True


def _build_command(
    *,
    llama_server: Path,
    gguf_path: str,
    port: int,
    rpc_endpoints: list[str],
    ngl: int = _DEFAULT_NGL,
    mmproj_path: str | None = None,
    draft_gguf_path: str | None = None,
    parallel: int = 1,
    tensor_split: list[int] | None = None,
    split_mode: str | None = None,
    cache_type: str | None = None,
    ctx_size: int = _CTX_SIZE_FLOOR,
    prompt_cache_path: str | None = None,
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
        # Context size: adaptive — `_compute_optimal_ctx_size` picks
        # the largest window the pool can hold given the model's KV
        # geometry, the chosen `--parallel` slot count, and the
        # bottleneck-node free memory. Floors at 4096 (legacy default,
        # safe for tiny pools) and ceils at 128 K (enough for any
        # production chat). Wins on rich pools that previously
        # capped at 4 K despite having 80 GB of combined memory.
        "-c", str(max(_CTX_SIZE_FLOOR, int(ctx_size))),
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
        # Continuous batching — N concurrent decoding slots sharing the
        # same warm engine. Each slot pre-allocates its own KV cache, so
        # the size is auto-tuned by `_compute_optimal_parallel` from
        # GGUF metadata + actual free pool memory (vendor-agnostic).
        # Wins when multiple streams hit the same llama-server (parallel
        # subagents, concurrent chats, target+draft verification);
        # single-stream cost is identical to `--parallel 1` since the
        # other slots stay quiescent.
        "--parallel", str(max(1, int(parallel))),
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
        # Force single-slot dispatch — multi-slot init triggers the
        # rpc-server crash on Gated Delta Net + Flash Attention.
        # Replace whatever the adaptive picker selected with 1.
        for i, a in enumerate(cmd):
            if a == "--parallel" and i + 1 < len(cmd):
                cmd[i + 1] = "1"
                break
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
    # MoE expert auto-pin to CPU when GPU VRAM clearly can't fit
    # them. Engages only for MoE models whose file size exceeds host
    # VRAM (`_should_pin_experts_to_cpu`); the existing pool-side
    # `-d CPU` RPC distribution then fans the expert tensors across
    # host RAM + every worker's RAM. Attention layers stay on GPU.
    # Pattern matches the standard MoE expert tensor names
    # (`*_exps.weight`, `*_experts.weight`) used by Mixtral / DeepSeek-V3 /
    # Qwen 2.5 MoE family GGUFs.
    if _should_pin_experts_to_cpu(gguf_path):
        cmd.extend(["-ot", r".*_(exps|experts)\.weight=CPU"])
        log.info(
            "split_lifecycle: MoE expert tensors pinned to CPU for "
            "%s (file size > host VRAM)", gguf_path,
        )
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
    if tensor_split:
        # Per-device layer weights for heterogeneous pools. Format is
        # `a/b/c/...` where each entry corresponds to a device in the
        # order llama.cpp enumerates them: host backend(s) first, then
        # each --rpc endpoint. `_compute_tensor_split_ratios` returns
        # None for homogeneous pools so equal split (llama.cpp default)
        # stays in effect there.
        cmd.extend(["-ts", "/".join(str(int(w)) for w in tensor_split)])
    if split_mode and split_mode in ("layer", "row", "none"):
        # `--split-mode row` enables tensor-parallel matmul dispatch
        # (vs. the default sequential layer pipeline). Requires fast
        # interconnect — we engage only when `_should_use_row_split`
        # confirmed every RPC worker is below the latency ceiling.
        # `none` is single-device — included for completeness; we
        # don't engage it from the auto path.
        cmd.extend(["--split-mode", split_mode])
    if cache_type:
        # KV cache quantization. `q8_0` halves the per-slot KV cost
        # vs FP16 with < 1 % accuracy loss on standard benchmarks.
        # Engaged by `_decide_kv_precision_and_parallel` only when
        # FP16 fits fewer than `_KV_QUANT_PARALLEL_FLOOR` slots — at
        # that point Q8 is strictly more throughput.
        cmd.extend(["-ctk", cache_type, "-ctv", cache_type])
    if prompt_cache_path:
        # Prompt-cache-to-disk: persists decoded KV state across
        # llama-server restarts. On the next spawn with the same
        # path, llama-server loads the cache and skips re-tokenising
        # / re-decoding any matching prompt prefix. Massive win for
        # the system-prompt + tool-schema prefix that's stable per
        # (model, permission_mode) — every chat after a model switch
        # starts hot instead of cold.
        cmd.extend(["--prompt-cache", prompt_cache_path, "--prompt-cache-all"])
    return cmd


def _resolve_prompt_cache_path(gguf_path: str, cache_type: str | None) -> str | None:
    """Return a deterministic prompt-cache file path for this (model, KV
    precision) combination, or ``None`` when the cache shouldn't be used.

    Cache files live under ``~/.gigachat/llama-cpp/prompt-cache/`` keyed
    by a short hash of the GGUF path + cache-type so different precisions
    don't collide. llama-server overwrites the file on each save; reuse
    is automatic when the same path is passed on the next spawn.

    Returns ``None`` for paths that can't be resolved (e.g. no GGUF on
    disk yet) so the caller falls back to the no-cache code path —
    behavior matches the legacy spawn there.
    """
    try:
        if not Path(gguf_path).is_file():
            return None
    except (OSError, ValueError):
        return None
    try:
        import hashlib
        key = f"{gguf_path}:{cache_type or 'fp16'}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return None
    cache_dir = split_runtime.LLAMA_CPP_INSTALL_DIR / "prompt-cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return str(cache_dir / f"{digest}.bin")


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

    # Adaptive --parallel: KV-cost per slot * number of slots ≤ free
    # memory across the bottleneck node (host VRAM for host-only paths,
    # min(host, every worker free RAM) for split paths). Auto-scales
    # from 1 (CPU-only laptops, tight setups) up to 8 (workstations
    # with plenty of headroom). Speculative draft size is folded in so
    # parallel + speculative don't double-count free VRAM.
    draft_size = 0
    if draft:
        try:
            draft_size = os.path.getsize(draft)
        except OSError:
            draft_size = 0
    target_size_bytes = (
        os.path.getsize(row["gguf_path"]) if Path(row["gguf_path"]).is_file()
        else 0
    )

    # Joint precision + slot-count decision. FP16 is preferred when
    # the pool can afford it; under memory pressure the helper
    # transparently switches to Q8 KV (½ the memory per slot, < 1 %
    # accuracy drop) so we keep more concurrent decoding slots than
    # FP16 would allow.
    cache_type, parallel = _decide_kv_precision_and_parallel(
        row["gguf_path"], worker_ids,
        target_size_bytes=target_size_bytes,
        draft_size_bytes=draft_size,
    )

    # Adaptive context window. With (cache_type, parallel) decided we
    # know the per-token KV cost; sized against the bottleneck-node
    # free memory it tells us how big a context the pool can serve
    # without OOM. Floors at 4 K (legacy default) and ceils at the
    # model's training-time max.
    ctx_size = _compute_optimal_ctx_size(
        row["gguf_path"], worker_ids,
        parallel=parallel,
        cache_type=cache_type,
        target_size_bytes=target_size_bytes,
        draft_size_bytes=draft_size,
    )

    # Heterogeneous-pool weighting: when host VRAM and worker capacities
    # differ by more than `_TS_HETEROGENEITY_RATIO`, push more layers
    # onto the bigger nodes via `-ts`. Returns None for homogeneous
    # pools so llama.cpp's per-device free-memory query stays in
    # control there.
    tensor_split = _compute_tensor_split_ratios(row["gguf_path"], worker_ids)

    # Tensor-parallel row dispatch — engaged only on Gigabit-Ethernet-
    # class LANs (every worker probe latency below the ceiling). On
    # typical home Wi-Fi we stay on layer-pipeline (the default), since
    # row mode's per-layer sync chatter loses to plain pipeline above
    # ~5 ms RTT.
    split_mode = "row" if _should_use_row_split(worker_ids) else None

    # Prompt-cache-to-disk path — keyed by GGUF filename + cache_type
    # so different precisions get different caches. Persists across
    # llama-server restarts; first turn after a model-switch reloads
    # the cached system-prompt KV instead of re-decoding from scratch.
    prompt_cache_path = _resolve_prompt_cache_path(row["gguf_path"], cache_type)

    cmd = _build_command(
        llama_server=server,
        gguf_path=row["gguf_path"],
        port=row["llama_port"],
        rpc_endpoints=rpc_endpoints,
        mmproj_path=mmproj,
        draft_gguf_path=draft,
        ngl=ngl,
        parallel=parallel,
        tensor_split=tensor_split,
        split_mode=split_mode,
        cache_type=cache_type,
        ctx_size=ctx_size,
        prompt_cache_path=prompt_cache_path,
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
