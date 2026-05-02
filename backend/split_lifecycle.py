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
    # `ngl` is the GPU-offload layer count this process was launched
    # with. The adaptive watchdog compares it against the optimal
    # value computed from current free RAM and triggers a graceful
    # restart when the delta exceeds the rebalance threshold (peer
    # opened or closed an app, freeing or consuming RAM).
    ngl: int = 0
    # Cool-down marker — the watchdog won't issue back-to-back
    # restarts within `_REBALANCE_COOLDOWN_SEC` seconds of the last
    # restart for the same split_id, even if the layer math says we
    # could squeeze a few more on. Prevents thrash when free RAM
    # oscillates around a threshold.
    last_rebalance_at: float = 0.0
    # Last-touched-by-chat timestamp. Bumped every time the agent
    # streams against this split's base_url (see
    # `mark_split_touched_by_base_url`). The adaptive watchdog uses
    # this to decide when an idle split should be GC'd — holding a
    # multi-GB llama-server alive when no chat has used it in N min
    # wastes RAM that other processes (or other models) could use.
    last_touched_at: float = 0.0


def mark_split_touched_by_base_url(base_url: str) -> None:
    """Stamp ``last_touched_at`` on the running split whose base_url
    matches. No-op when the base_url isn't ours (split was just
    stopped, or it's an Ollama URL slipping through).

    Called from the agent every time it streams against a llama-server
    base_url so the idle GC knows the split is still in active use.
    """
    if not base_url:
        return
    try:
        port = int(base_url.rsplit(":", 1)[1].split("/", 1)[0])
    except (ValueError, IndexError):
        return
    now = time.time()
    for sid, rp in _running.items():
        if rp.port == port:
            rp.last_touched_at = now
            return


_running: dict[str, _RunningProcess] = {}


# ---------------------------------------------------------------------------
# Command building (pure; tested directly)
# ---------------------------------------------------------------------------

def _resolve_rpc_endpoints(worker_ids: list[str]) -> list[str]:
    """Look up each worker_id and produce a list of `<host>:<port>`
    strings. A single worker may emit MULTIPLE entries when it
    exposes more than one rpc-server (typical for the multi-rpc-
    server pattern where the worker runs an iGPU rpc-server on
    50052 AND a CPU rpc-server on 50053).

    Skips:
      * worker rows that have been deleted since the split_model row
        was created (DB referential integrity is loose by design — the
        split row holds string IDs, not foreign keys, so a worker can
        be removed without breaking the split row).
      * disabled workers (the user toggled them off explicitly).
      * workers whose probe last reported `rpc_server_reachable=False`
        — passing those to llama-server would just cause connection
        refusals during the actual inference.

    Returns the endpoints in the same order as `worker_ids`, with
    each worker's endpoints in the order produced by
    `compute_pool.select_multi_rpc_specs` (iGPU port first, CPU
    port second). Order matters: llama.cpp uses it to assign layer
    ranges and to map per-device --tensor-split weights, so the
    weights computed by `_compute_tensor_split_ratios` MUST list
    entries in the same order this function does.
    """
    # First pass: collect each worker's endpoints in CPU-then-GPU
    # order (we'll interleave across workers in a second pass).
    per_worker: list[list[str]] = []
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
        for prefix in ("http://", "https://"):
            if host.startswith(prefix):
                host = host[len(prefix):]
        host = host.rstrip("/")
        endpoints = caps.get("rpc_endpoints")
        worker_eps: list[str] = []
        if isinstance(endpoints, list) and endpoints:
            sorted_eps = sorted(
                endpoints,
                key=lambda e: 0 if (e.get("backend") or "").strip().upper() == "CPU" else 1,
            )
            for ep in sorted_eps:
                try:
                    port = int(ep.get("port") or 50052)
                except (TypeError, ValueError):
                    port = 50052
                worker_eps.append(f"{host}:{port}")
        else:
            port = caps.get("rpc_port") or 50052
            worker_eps.append(f"{host}:{port}")
        per_worker.append(worker_eps)

    # Second pass: INTERLEAVE across workers so no two consecutive
    # --rpc entries share an IP. Critical for llama.cpp's RPC client
    # which (verified live on b9002) FAILS to connect to the second
    # rpc-server on a host when the first one was just connected
    # to — looks like a connection-pool bug where the same-host
    # socket is reused. By emitting [W0[0], W1[0], W2[0], W0[1],
    # W1[1], ...], every consecutive pair is on a different IP and
    # all 4 rpc-server processes register cleanly. Verified to
    # produce ALL 4 RPC0..RPC3 entries in the device list with
    # the per-port memory_breakdown tracking each one.
    out: list[str] = []
    if not per_worker:
        return out
    max_per_worker = max(len(x) for x in per_worker)
    for round_idx in range(max_per_worker):
        for worker_eps in per_worker:
            if round_idx < len(worker_eps):
                out.append(worker_eps[round_idx])
    return out


def _is_moe_model(gguf_path: str) -> bool:
    """Heuristic: does this GGUF describe a Mixture-of-Experts model?

    Reads `<arch>.expert_count` from the cached metadata; > 0 means MoE.
    Used to decide whether to add the `-ot` flag that pins MoE expert
    tensors to the host CUDA backend (see `_build_command` docstring
    for why).

    Returns False on any read error so non-MoE models keep their
    current command path unchanged.
    """
    metadata = _get_gguf_metadata(gguf_path)
    expert_count = metadata.get("expert_count") or 0
    return int(expert_count) > 0


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
    metadata = _get_gguf_metadata(gguf_path)
    if not metadata:
        return False
    if metadata.get("arch") not in ("gemma4", "gemma3n"):
        return False
    if not (metadata.get("ple_embedding_length") or 0) > 0:
        return False
    # Graph-width guard: E2B (block_count=30) is fine; E4B (35)+ trips it.
    blocks = metadata.get("block_count") or 0
    return blocks > 30


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


# Per-path cache for parsed GGUF metadata. Keyed by (gguf_path) →
# (mtime, fields_dict). Six different helpers in this module open the
# same GGUF on every llama-server spawn — `_is_moe_model`,
# `_model_needs_fit_off`, `_estimate_kv_bytes_per_slot`,
# `_compute_optimal_ngl`, `_compute_optimal_ctx_size`,
# `_compute_optimal_batch_sizes`. Each open mmaps the file and parses
# headers (typ. 10-50 ms); 6× per spawn is 60-300 ms of redundant
# work. Caching by mtime lets a re-quantized blob trigger a re-parse
# while a stable file is parsed exactly once per backend lifetime.
_GGUF_METADATA_CACHE: dict[str, tuple[float, dict]] = {}


def _get_gguf_metadata(gguf_path: str) -> dict:
    """Return cached GGUF metadata for ``gguf_path``.

    The dict carries the architecture name plus every integer field
    the helpers in this module read. Cache invalidates on mtime
    change so an in-place rewrite (rare) is picked up on the next
    spawn. Returns an empty dict on any read failure — callers fall
    through to their default behaviour.
    """
    try:
        mtime = os.path.getmtime(gguf_path)
    except OSError:
        return {}
    cached = _GGUF_METADATA_CACHE.get(gguf_path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        import gguf
        reader = gguf.GGUFReader(gguf_path)
    except Exception:
        return {}

    arch_field = reader.fields.get("general.architecture")
    if not arch_field or not arch_field.types:
        return {}
    try:
        arch = arch_field.parts[arch_field.data[0]].tobytes().decode(
            "utf-8", errors="replace",
        )
    except Exception:
        return {}

    metadata: dict = {
        "arch": arch,
        "block_count": _read_gguf_int(
            reader, arch,
            f"{arch}.block_count", "llama.block_count", "general.block_count",
        ),
        "embedding_length": _read_gguf_int(
            reader, arch,
            f"{arch}.embedding_length", "llama.embedding_length",
        ),
        "head_count": _read_gguf_int(
            reader, arch,
            f"{arch}.attention.head_count", "llama.attention.head_count",
        ),
        "head_count_kv": _read_gguf_int(
            reader, arch,
            f"{arch}.attention.head_count_kv",
            "llama.attention.head_count_kv",
        ),
        "context_length": _read_gguf_int(
            reader, arch,
            f"{arch}.context_length", "llama.context_length",
        ),
        "expert_count": _read_gguf_int(
            reader, arch, f"{arch}.expert_count",
        ),
        # Gemma 3n PLE marker — read both the architecture-prefixed
        # form and the explicit gemma3n key so older / newer
        # quantisations both match.
        "ple_embedding_length": _read_gguf_int(
            reader, arch,
            f"{arch}.embedding_length_per_layer_input",
            "gemma3n.embedding_length_per_layer_input",
        ),
    }
    # Drop the reader so its mmap can release; we've extracted
    # everything we cared about into the dict.
    del reader
    _GGUF_METADATA_CACHE[gguf_path] = (mtime, metadata)
    return metadata


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
    metadata = _get_gguf_metadata(gguf_path)
    if not metadata:
        return 0
    n_layers = metadata.get("block_count") or 0
    embedding_length = metadata.get("embedding_length") or 0
    n_heads = metadata.get("head_count") or 0
    if not n_layers or not embedding_length or not n_heads:
        return 0
    # MHA fallback: when head_count_kv is absent the model uses
    # multi-head attention so kv heads == query heads.
    n_kv_heads = metadata.get("head_count_kv") or n_heads
    head_dim = embedding_length // n_heads
    if head_dim <= 0:
        return 0
    # 2 (K+V) × layers × kv_heads × head_dim × bytes_per_element (FP16=2)
    kv_per_token = 2 * n_layers * n_kv_heads * head_dim * 2
    return int(kv_per_token * max(ctx_size, 1))


# How much VRAM headroom to leave free after target + draft + KV slots.
# 5 % buffer to absorb allocator alignment overhead and avoid crashes
# on overcommit — the empirical floor where llama-server reliably
# loads a model that fits on paper. Smaller would push throughput
# higher in theory but tips into OOM crashes in practice.
_PARALLEL_VRAM_HEADROOM = 0.05

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

    Returns ``(cache_type, parallel)`` where ``cache_type`` is
    ``"q8_0"`` (default — half the KV memory at <1 % accuracy loss
    on standard benchmarks) or ``None`` (FP16 — only when the pool
    has so much headroom that quantising buys nothing).

    Q8 KV is the cheaper-and-faster default. We bias toward it
    because (a) the memory it frees is real — every freed byte goes
    to extra parallel slots, longer context, or fewer offload-to-CPU
    layers; (b) Flash Attention is on by default in `_build_command`
    so the q8 kernel is always available; (c) the precision floor
    matches Ollama's recommendation. The only reason to fall back to
    FP16 is when the pool already fits the Q8 maximum and the small
    quality margin matters more than freed VRAM — that's a corner
    we don't optimise for.
    """
    q8_parallel = _compute_optimal_parallel(
        gguf_path, worker_ids,
        ctx_size=ctx_size,
        target_size_bytes=target_size_bytes,
        draft_size_bytes=draft_size_bytes,
        kv_size_multiplier=0.5,
    )
    log.info(
        "split_lifecycle: KV cache type=q8_0 (default), fits %d parallel slot(s)",
        q8_parallel,
    )
    return "q8_0", q8_parallel


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
    # Comes from the cached metadata so a re-read doesn't re-mmap.
    metadata = _get_gguf_metadata(gguf_path)
    model_max_ctx = metadata.get("context_length") or _CTX_SIZE_CEILING

    # POOL-aware free memory budget. The previous formula compared
    # the model size against HOST VRAM only — for a 19 GB model on
    # an 8 GB GPU host that goes negative and we floored to 4 K
    # context. In split mode the model is DISTRIBUTED across host +
    # workers, so the right denominator is the combined pool. Use
    # 95 % of available (matches the user's explicit "5 % margin"
    # policy — see compute_pool._directive memory).
    try:
        from . import sysdetect
        spec = sysdetect.detect_system()
        host_vram = int(float(spec.get("vram_gb") or 0) * (1024 ** 3))
        host_ram = int(float(spec.get("ram_gb") or 0) * (1024 ** 3))
    except Exception:
        return _CTX_SIZE_FLOOR
    pool_total = host_vram + host_ram  # host's full envelope
    if pool_total <= 0:
        return _CTX_SIZE_FLOOR
    for wid in worker_ids:
        w = db.get_compute_worker(wid)
        if not w or not w.get("enabled"):
            continue
        caps = w.get("capabilities") or {}
        worker_total_gb = float(caps.get("ram_total_gb") or 0)
        if worker_total_gb <= 0:
            # Fallback: use ram_free_gb if total isn't populated
            worker_total_gb = float(caps.get("ram_free_gb") or 0)
        pool_total += int(worker_total_gb * (1024 ** 3))

    # KV cache budget = 95 % of pool minus the weight footprint.
    # Workers hold their RPC layer share, but the weights also live
    # on host (mmap or no-mmap) so we count target_size once. KV
    # cache then consumes whatever remains.
    free_after_target = int(pool_total * 0.95) - int(target_size_bytes or 0) - int(draft_size_bytes or 0)
    if free_after_target <= 0:
        return _CTX_SIZE_FLOOR

    fits = free_after_target // (kv_per_token * max(1, parallel))
    fits = min(fits, model_max_ctx, _CTX_SIZE_CEILING)
    fits = max(_CTX_SIZE_FLOOR, int(fits))
    # Replace the bottleneck variable name in the log line below.
    bottleneck_free = free_after_target
    log.info(
        "split_lifecycle: adaptive ctx: kv_per_token=%d parallel=%d "
        "bottleneck_free=%.2f GB model_max=%d -> -c %d",
        kv_per_token, parallel, bottleneck_free / (1024 ** 3),
        model_max_ctx, fits,
    )
    return fits


def _compute_optimal_batch_sizes(
    gguf_path: str,
    *,
    parallel: int,
    ctx_size: int,
    cache_type: str | None,
    target_size_bytes: int,
    draft_size_bytes: int = 0,
) -> tuple[int, int] | tuple[None, None]:
    """Pick `-b` (logical) and `-ub` (physical) batch sizes for prefill.

    llama-server defaults: ``-b 2048 -ub 512``. Bigger ``-ub`` makes
    prefill 2-4× faster on capable GPUs (per-token matmul amortizes
    over the batch) but costs activation memory roughly proportional
    to ``ub × hidden_size × n_layers``. On a workstation GPU with
    plenty of headroom the default leaves a lot of speed on the
    table; on a tight pool the default is exactly right.

    Algorithm:
      * Estimate activation memory per ubatch token from GGUF
        metadata (embedding_length × block_count × FP16 + 4× scratch).
      * Compute free VRAM after target+draft weights and KV cache.
      * Pick the largest ``ub`` whose activation budget fits half the
        remaining free VRAM (the other half is for scratch / paging /
        OS overhead).
      * Round to the next power of two so llama.cpp's batched kernels
        hit fast paths.

    Returns ``(b, ub)`` when bumping above defaults is safe; returns
    ``(None, None)`` to keep llama.cpp's defaults — for tight pools
    where activation memory is constrained, or when GGUF metadata
    can't be read (no informed call possible).
    """
    metadata = _get_gguf_metadata(gguf_path)
    embedding_length = metadata.get("embedding_length") or 0
    block_count = metadata.get("block_count") or 0
    if not embedding_length or not block_count:
        return None, None

    # Activation memory per token in the prefill pipeline:
    #   FP16 hidden state × layers × scratch multiplier
    # The 4× multiplier covers attn / FFN intermediate buffers that
    # llama.cpp allocates per layer; empirically matches what nvprof
    # shows on standard transformers.
    activation_per_token = embedding_length * block_count * 2 * 4

    # Free VRAM after weights + KV cache.
    try:
        from . import sysdetect
        spec = sysdetect.detect_system()
        host_vram = int(float(spec.get("vram_gb") or 0) * (1024 ** 3))
    except Exception:
        return None, None
    if host_vram <= 0:
        return None, None  # CPU-only — keep defaults; -ub bump won't help

    # KV bytes already accounted for by the parallel decision.
    kv_bytes = _estimate_kv_bytes_per_slot(gguf_path, ctx_size) * max(1, parallel)
    if cache_type == "q8_0":
        kv_bytes = kv_bytes // 2

    free_after = int(host_vram * (1.0 - _PARALLEL_VRAM_HEADROOM))
    free_after -= int(target_size_bytes or 0) + int(draft_size_bytes or 0)
    free_after -= kv_bytes
    if free_after <= 0:
        return None, None

    # Use half of remaining free VRAM for prefill activation. The
    # other half is reserve for kernel scratch / OS / driver overhead.
    activation_budget = free_after // 2
    max_ub_by_memory = activation_budget // max(1, activation_per_token)

    # Round down to the largest power of two ≤ max_ub. Fast llama.cpp
    # kernels are sized for power-of-two ubatches; non-aligned values
    # work but lose performance on some backends. Ceiling raised to
    # 8192 — with Flash Attention + Q8 KV freeing ~half the activation
    # budget vs the legacy FP16 path, big-VRAM workstations can absorb
    # the larger ubatch and prefill 2-4× faster on long prompts.
    if max_ub_by_memory < 1024:
        return None, None  # default 512 is already aggressive enough
    if max_ub_by_memory >= 8192:
        ub = 8192
    elif max_ub_by_memory >= 4096:
        ub = 4096
    elif max_ub_by_memory >= 2048:
        ub = 2048
    elif max_ub_by_memory >= 1024:
        ub = 1024
    else:
        return None, None

    # `-b` (logical) is generally 2× `-ub` (physical) — gives the
    # scheduler room to pack multiple ubatches per logical batch.
    # llama.cpp caps `-b` at 16384 in recent builds; 2×ub is safe.
    b = min(16384, ub * 2)
    log.info(
        "split_lifecycle: adaptive batch sizes: free_vram=%.2f GB "
        "activation_per_token=%d MiB -> -b %d -ub %d",
        free_after / (1024 ** 3),
        activation_per_token / (1024 ** 2),
        b, ub,
    )
    return b, ub


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

    # If ANY worker exposes multi-rpc endpoints, return None so
    # `_build_command` omits the `-ts` flag entirely. Reason:
    # llama.cpp's `common_fit_params` auto-fit ABORTS with
    # "model_params::tensor_split already set by user, abort"
    # when `-ts` is provided alongside `-ngl 0` (auto-fit mode).
    # The two flags are mutually exclusive in b9002+. Let auto-fit
    # do the per-device sizing — it knows each device's hard memory
    # cap and walks ngl down until everything fits.
    for wid in worker_ids:
        w = db.get_compute_worker(wid)
        if not w:
            continue
        eps = (w.get("capabilities") or {}).get("rpc_endpoints")
        if isinstance(eps, list) and len(eps) > 1:
            log.info(
                "split_lifecycle: multi-rpc endpoints detected, omitting "
                "-ts so llama.cpp's auto-fit can pick per-device shares",
            )
            return None

    weights: list[float] = [host_capacity_gb]
    for wid in worker_ids:
        w = db.get_compute_worker(wid)
        if not w or not w.get("enabled"):
            return None  # missing worker — can't size; let llama.cpp pick
        caps = w.get("capabilities") or {}
        # Multi-rpc-server path: the worker exposes one rpc-server
        # PER backend (e.g. iGPU on 50052, CPU on 50053). Emit a
        # weight per ENDPOINT, not per worker — `_resolve_rpc_endpoints`
        # produces matching `host:port` entries in the same order, so
        # llama.cpp's per-device --tensor-split maps cleanly. iGPU
        # endpoint capped at VRAM ceiling; CPU endpoint sized by
        # remaining RAM after the iGPU portion.
        endpoints = caps.get("rpc_endpoints")
        if isinstance(endpoints, list) and endpoints:
            ram_free_gb = float(caps.get("ram_free_gb") or 0)
            vram_total_gb = float(caps.get("vram_total_gb") or 0)
            ram_total_gb = float(caps.get("ram_total_gb") or 0)
            for ep in endpoints:
                ep_backend = (ep.get("backend") or "").lower()
                if any(k in ep_backend for k in ("sycl", "vulkan", "cuda")):
                    # iGPU / dGPU endpoint — cap at VRAM ceiling.
                    # 0.85 leaves headroom for KV cache + compute
                    # buffers also allocated on the device.
                    if vram_total_gb > 0:
                        weights.append(vram_total_gb * 0.85)
                    elif ram_free_gb > 0:
                        # No vram info — fall back to a small share
                        # so we don't over-commit.
                        weights.append(min(ram_free_gb * 0.30, 2.0))
                    else:
                        weights.append(1.0)
                else:
                    # CPU endpoint — gets the worker's free RAM
                    # MINUS what the iGPU endpoint will take from
                    # shared memory (Intel UMA: iGPU draws from
                    # the same physical RAM pool). Total per-worker
                    # contribution = ram_free, split between the
                    # two endpoints based on which device gets it.
                    cpu_capacity = max(0.5, ram_free_gb - vram_total_gb * 0.85)
                    weights.append(cpu_capacity)
            continue
        # Legacy single-rpc-server path — one weight per worker.
        # Worker capacity prefers `max_vram_seen_bytes` (proven GPU memory
        # from `/api/ps`) and falls back to `ram_free_gb` from the SSH
        # spec probe.
        vram_bytes = int(caps.get("max_vram_seen_bytes") or 0)
        ram_free_gb = float(caps.get("ram_free_gb") or 0)
        capacity_gb = max(vram_bytes / (1024 ** 3), ram_free_gb)
        current_backend = (caps.get("current_rpc_backend") or "").lower()
        has_gpu_device = (
            "sycl" in current_backend or "vulkan" in current_backend
            or "cuda" in current_backend
        )
        if has_gpu_device:
            vram_total_gb = float(caps.get("vram_total_gb") or 0)
            if vram_total_gb > 0:
                # 0.85 leaves room for KV cache + compute buffers
                # that are also allocated on the device, plus the
                # OS / driver overhead for the iGPU itself.
                capacity_gb = min(capacity_gb, vram_total_gb * 0.85)
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


def _compute_optimal_n_cpu_moe(
    gguf_path: str,
    worker_ids: list[str],
    *,
    ngl: int,
    parallel: int,
    cache_type: str | None,
    ctx_size: int,
    target_size_bytes: int,
) -> int | None:
    """Decide how many MoE layers' expert FFNs to keep on CPU.

    Only meaningful for MoE models where the GPU pool can't hold
    every expert; for dense models OR for MoE models that fit GPU
    completely, returns ``None`` (no override — llama.cpp's default
    placement is best).

    Algorithm:
      1. Bail early if the model isn't MoE (`_is_moe_model`).
      2. Estimate per-layer expert size: total weight bytes minus
         attention bytes, divided by block_count. Heuristic — most
         MoE weights are FFN experts (typically 70-90 %).
      3. Compute the GPU memory budget after KV cache reservations.
         Budget = (host_vram + each worker's free RAM) - kv_bytes.
      4. ngl bounds how many layers attempt GPU residency. The first
         ``ngl`` layers' attention + dense parts already live on GPU.
         Their experts compete for the remaining budget.
      5. Walk layers from the END (highest-indexed) and pin one
         layer's experts to CPU per ``avg_expert_bytes`` of overshoot.
         llama.cpp counts ``--n-cpu-moe`` from the highest-numbered
         layer downward, matching this walk direction.

    Returns the integer count of MoE layers to send to CPU, clamped
    to ``[0, n_layers]``. ``0`` means "no override needed — pool fits
    every expert"; ``None`` means "model isn't MoE — skip the flag".
    """
    if not _is_moe_model(gguf_path):
        return None
    metadata = _get_gguf_metadata(gguf_path)
    n_layers = metadata.get("block_count") or 0
    if n_layers <= 0:
        return None

    try:
        file_size = os.path.getsize(gguf_path)
    except OSError:
        return None

    # GPU pool budget: host VRAM + each worker's free RAM (workers
    # exposing iGPU SYCL via shared system memory, so RAM is the
    # right proxy). Apply the same 5 % buffer used by the
    # adaptive-ngl decision so the n-cpu-moe placement stays in
    # sync with what `-ngl` will actually try to fit on GPU.
    pool_bytes = 0
    try:
        from . import sysdetect
        spec = sysdetect.detect_system()
        pool_bytes += int(float(spec.get("vram_gb") or 0) * (1024 ** 3))
    except Exception:
        pass
    for wid in worker_ids:
        w = db.get_compute_worker(wid)
        if not w or not w.get("enabled"):
            continue
        caps = w.get("capabilities") or {}
        pool_bytes += int(float(caps.get("ram_free_gb") or 0) * (1024 ** 3))
    pool_bytes = int(pool_bytes * 0.95)
    if pool_bytes <= 0:
        return None

    # Subtract KV cache footprint so we don't tell the GPU it has
    # bytes that are already spoken for.
    # Note: `_estimate_kv_bytes_per_slot` takes a gguf_path string,
    # not the metadata dict. Passing the dict here was the cause of
    # a TypeError that bubbled out of `start()` as a 500.
    kv_bytes_per_token = _estimate_kv_bytes_per_slot(gguf_path, 1)
    kv_total = (
        kv_bytes_per_token
        * max(1, ctx_size)
        * max(1, parallel)
        * (0.5 if cache_type == "q8_0" else 1.0)
    )
    pool_bytes -= int(kv_total)
    if pool_bytes <= 0:
        # Even KV doesn't fit — push everything except attention off GPU.
        return n_layers

    # Compare what GPU layers want to occupy vs the budget. ngl
    # caps how many layers attempt GPU residency.
    ngl_effective = min(int(ngl), n_layers)
    if ngl_effective <= 0:
        return n_layers
    avg_layer_bytes = file_size / n_layers
    layer_bytes_on_gpu = avg_layer_bytes * ngl_effective
    if layer_bytes_on_gpu <= pool_bytes:
        # Whole model fits — no override needed.
        return 0

    # Excess gets shipped to CPU. Each MoE layer offloaded saves
    # `avg_layer_bytes` (close enough — experts dominate the layer).
    excess = layer_bytes_on_gpu - pool_bytes
    n_offload = int((excess + avg_layer_bytes - 1) / avg_layer_bytes)
    n_offload = min(n_layers, max(0, n_offload))
    log.info(
        "split_lifecycle: --n-cpu-moe: model=MoE, file=%.1f GB, ngl=%d, "
        "kv=%.1f GB, pool_after_kv=%.1f GB -> offload %d/%d MoE layer(s)",
        file_size / (1024 ** 3), ngl_effective,
        kv_total / (1024 ** 3), pool_bytes / (1024 ** 3),
        n_offload, n_layers,
    )
    return n_offload


def _should_use_row_split(worker_ids: list[str]) -> bool:
    """Return True when `--split-mode row` should engage.

    Two independent triggers, EITHER is sufficient:

    1. LAN is fast enough (low latency) that the per-token sync chatter
       amortizes against the row-parallel matmul speedup. Original
       criterion — preserved for the original tensor-parallel-on-fast-
       LAN use case.

    2. ANY worker exposes a small iGPU (<= 4 GB VRAM) AND models we
       run are big enough that layer-mode would drop that iGPU from
       placement (its proportional share is smaller than one whole
       layer). With layer-mode, llama.cpp's auto-fit silently excludes
       devices whose share rounds below 1 layer — the iGPU sits idle
       even though it has free memory. Row-mode splits every layer's
       matmul across ALL devices so the iGPU contributes a row-stripe
       of every layer instead of needing to host whole layers. This is
       the ONLY way to keep small iGPUs in the inference rotation.

    Returns False only when the latency ceiling is exceeded AND no
    small iGPU is in the pool (no reason to engage row-mode then).
    """
    if not worker_ids:
        return False
    has_small_igpu = False
    fast_lan = True
    for wid in worker_ids:
        w = db.get_compute_worker(wid)
        if not w or not w.get("enabled"):
            return False
        caps = w.get("capabilities") or {}
        latency_ms = int(caps.get("probe_latency_ms") or 999)
        if latency_ms > _ROW_SPLIT_LATENCY_CEILING_MS:
            fast_lan = False
        # Small-iGPU detection: any worker exposing a GPU device with
        # <= 4 GB VRAM. Intel Iris Xe shared pool typically reports
        # 2-3 GB; AMD Vega iGPU similar. dGPUs (NVIDIA, Arc Bxx) are
        # bigger than this threshold and don't trigger.
        gpu_kind = (caps.get("gpu_kind") or "").lower()
        vram_total = float(caps.get("vram_total_gb") or 0)
        if gpu_kind in ("intel", "amd") and 0 < vram_total <= 4.0:
            has_small_igpu = True
    return fast_lan or has_small_igpu


def _should_mlock_weights(target_size_bytes: int) -> bool:
    """Return True when locking the GGUF in RAM is safe + likely useful.

    `--mlock` pins weight pages so the OS can't page them out under
    memory pressure. On a system with plenty of free RAM this is a
    small but real win — token-rate degrades over hours when memory
    fragmentation forces the kernel to evict warm weight pages.

    Engages only when the model file is comfortably smaller than free
    RAM (target ≤ 50 % of free) so locking can't OOM the system or
    starve other processes. Below that threshold, mmap-on-demand is
    the safer default and llama.cpp falls back to it cleanly.

    Returns False on any read failure so undetectable-memory hosts
    keep llama.cpp's default mmap behavior.
    """
    if target_size_bytes <= 0:
        return False
    try:
        import psutil
        free_ram_bytes = int(psutil.virtual_memory().available)
    except Exception:
        return False
    if free_ram_bytes <= 0:
        return False
    return target_size_bytes * 2 <= free_ram_bytes


def _should_disable_mmap(
    target_size_bytes: int, worker_ids: list[str],
) -> bool:
    """Decide whether to pass `--no-mmap` to llama-server.

    Mirrors Ollama's empirically-derived heuristic (see
    ``ollama/llm/server.go`` — "Windows CUDA should not use mmap for
    best performance" / "Linux with a model larger than free space,
    mmap leads to thrashing"):

    Disable mmap (load weights into private process RAM) when ANY:
      * Host is Windows + has a CUDA GPU — mmap on Windows CUDA
        triggers page-fault stalls during decode that visibly tank
        throughput.
      * Model fits comfortably in pool RAM (model ≤ 80 % of pool
        free) — `--no-mmap` materialises the weights so the OS can't
        evict them under pressure, which is what produces visible
        process-RSS at the user's "95 % memory usage on every device"
        target. Without `--no-mmap` weights ride the file cache and
        Task Manager under-counts the load.

    Keep mmap enabled when:
      * Model exceeds pool RAM — mmap is the only way the model
        loads at all (host CPU streams pages from SSD on demand).
        This is the "use SSD/HDD as a compute tier" path the user
        explicitly wanted.
      * Pool memory is unknown — fall through to llama.cpp's safer
        default rather than risk OOM.
    """
    if target_size_bytes <= 0:
        return False
    # Pool-fit check first — even Windows CUDA needs mmap when the
    # model exceeds physical RAM, otherwise the load triggers an
    # OOM kill (verified empirically: dolphin-mixtral 26 GB on a
    # 16 GB FBS host with --no-mmap pegged the host at 96 % then
    # llama-server died). The user wants SSD as an overflow tier
    # — that's literally what mmap does for layers that don't fit.
    try:
        import psutil
        host_free = int(psutil.virtual_memory().available)
    except Exception:
        return False
    pool_free = host_free
    try:
        from . import sysdetect
        spec = sysdetect.detect_system()
        pool_free += int(float(spec.get("vram_gb") or 0) * (1024 ** 3))
    except Exception:
        pass
    for wid in worker_ids:
        w = db.get_compute_worker(wid)
        if not w or not w.get("enabled"):
            continue
        caps = w.get("capabilities") or {}
        free_gb = float(caps.get("ram_free_gb") or 0)
        pool_free += int(free_gb * (1024 ** 3))
    if pool_free <= 0:
        return False
    # If the model exceeds 80 % of pool free, KEEP mmap on. The
    # SSD/HDD-as-overflow path is the only way mega-models load at
    # all on a host with limited RAM. Workers' RAM still contributes
    # for the layers we explicitly offload via -ngl.
    fits = target_size_bytes <= int(pool_free * 0.80)
    if not fits:
        return False
    # Model fits in pool — now decide based on platform whether
    # disabling mmap is the win Ollama claims:
    if sys.platform == "win32":
        try:
            from . import sysdetect
            spec = sysdetect.detect_system()
            if (spec.get("gpu_kind") or "").lower() == "nvidia":
                # Windows CUDA: Ollama's empirical rule says no-mmap
                # is faster (avoids decode-time page faults).
                return True
        except Exception:
            pass
    # Linux / non-CUDA / non-Windows: disable mmap when fits so the
    # weights show up in process RSS and can't be evicted under
    # pressure. Hits the user's "95 % memory usage" success metric.
    return True


def _recommend_thread_counts() -> tuple[int, int] | tuple[None, None]:
    """Pick `--threads` (decode) and `--threads-batch` (prefill) values.

    llama-server defaults to ``os.cpu_count()`` which on hyperthreaded
    CPUs is the LOGICAL count — typically 2× the physical count. For
    AVX2 / matmul workloads, hyperthreads compete for the same FPU and
    physical core count usually wins by 5-15 % on decode. Prefill is
    more parallel-friendly, so it can use the logical count.

    Returns ``(decode_threads, prefill_threads)`` when psutil reports
    a physical / logical split worth acting on; ``(None, None)`` keeps
    the llama-server default. Single-core or unable-to-detect → fall
    through to default.
    """
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
    except Exception:
        return None, None
    if not physical or not logical:
        return None, None
    if physical <= 1:
        return None, None
    # Only override when there's a meaningful split (hyperthreading
    # active). On a system where logical == physical (no SMT), the
    # default already matches and there's no benefit to overriding.
    if logical <= physical:
        return None, None
    # Decode: physical cores. Prefill: logical (or physical*2 if even
    # higher SMT). Cap at 32 — beyond that NUMA and cache-line
    # contention erode the win on consumer hardware.
    decode_threads = min(32, int(physical))
    prefill_threads = min(32, int(logical))
    log.info(
        "split_lifecycle: adaptive threads — decode=%d prefill=%d "
        "(physical=%d logical=%d)",
        decode_threads, prefill_threads, physical, logical,
    )
    return decode_threads, prefill_threads


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
    batch_size: int | None = None,
    ubatch_size: int | None = None,
    threads: int | None = None,
    threads_batch: int | None = None,
    mlock: bool = False,
    flash_attn: bool = True,
    cache_reuse: int = 256,
    n_cpu_moe: int | None = None,
    no_mmap: bool = False,
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
        #
        # `ngl=0` is a sentinel meaning "let llama.cpp's auto-fit
        # pick" — emitted by `_compute_optimal_ngl` when the pool
        # has small heterogeneous devices (e.g. multi-rpc-server
        # workers with ~2 GB iGPUs) where a fixed user-set ngl
        # is more likely to abort llama-server's
        # `common_fit_params: failed to fit ... abort` than help.
        # We omit the flag entirely in that case so auto-fit runs.
        *(["-ngl", str(ngl)] if ngl > 0 else []),
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
    if prompt_cache_path and _llama_server_supports_flag(llama_server, "--prompt-cache"):
        # Prompt-cache-to-disk: persists decoded KV state across
        # llama-server restarts. On the next spawn with the same
        # path, llama-server loads the cache and skips re-tokenising
        # / re-decoding any matching prompt prefix. Massive win for
        # the system-prompt + tool-schema prefix that's stable per
        # (model, permission_mode) — every chat after a model switch
        # starts hot instead of cold.
        #
        # Only emitted when the build accepts it — recent llama.cpp
        # builds (post-2025-12) removed the disk prompt-cache CLI in
        # favour of in-process slot-state persistence; passing the
        # flag to those builds aborts startup with "invalid argument".
        cmd.extend(["--prompt-cache", prompt_cache_path, "--prompt-cache-all"])
    if batch_size and ubatch_size:
        # Adaptive prompt-eval batch sizes. llama-server defaults
        # (`-b 2048 -ub 512`) are conservative — bumping `-ub` can
        # speed prefill 2-4× on capable GPUs at the cost of activation
        # memory. `_compute_optimal_batch_sizes` only returns values
        # when the pool has the headroom to absorb the extra activation
        # memory; tight pools keep llama.cpp's defaults.
        cmd.extend(["-b", str(batch_size), "-ub", str(ubatch_size)])
    if threads:
        # Decode threads — defaults to logical CPU count which on
        # hyperthreaded CPUs over-subscribes the FPU. Physical core
        # count typically wins 5-15 % on AVX2 decode; the picker
        # only returns a value when there's a meaningful SMT split.
        cmd.extend(["--threads", str(int(threads))])
    if threads_batch:
        # Prefill threads — separate from decode because batched
        # matmul scales with logical core count more cleanly than
        # single-token decode does.
        cmd.extend(["--threads-batch", str(int(threads_batch))])
    if mlock:
        # Pin weights in RAM so the OS can't page them out on memory
        # pressure. Real win on long-running hosts where token-rate
        # would otherwise degrade as memory fragments. The picker
        # (`_should_mlock_weights`) only engages when free RAM is at
        # least 2× the model size, so locking can't OOM the system.
        cmd.append("--mlock")
    if no_mmap:
        # Disable mmap so the model loads into private process pages
        # rather than the file cache. Critical for two reasons:
        #   1. Resource visibility — mmap'd pages count as FS cache,
        #      not process RSS, so Task Manager / Activity Monitor
        #      shows a hugely under-counted memory footprint and the
        #      OS can evict layers under pressure (causing per-token
        #      disk thrashing).
        #   2. Performance on Windows CUDA — Ollama's empirical rule:
        #      "Windows CUDA should not use mmap for best performance"
        #      (see ollama/llm/server.go). The mmap path triggers
        #      page faults during decode that stall the GPU.
        # The selector (`_should_disable_mmap`) decides when to enable
        # this flag based on platform + GPU vendor + free RAM vs
        # model size, mirroring Ollama's heuristic.
        cmd.append("--no-mmap")
    if flash_attn:
        # Flash Attention switches the attention block to the fused
        # kernel — typically 5-30 % faster generation depending on
        # context length, with bigger gains at long contexts. It's
        # also the prerequisite for KV cache quantisation: llama.cpp
        # silently falls back to FP16 KV when -fa is off, so without
        # this flag the q8_0 cache_type above wouldn't actually engage.
        # `--flash-attn on` (vs the older `-fa` boolean toggle) gives
        # an explicit value across versions; the runtime warns and
        # disables for models that don't implement FA so it's safe
        # to leave on by default.
        cmd.extend(["--flash-attn", "on"])
    if cache_reuse and cache_reuse > 0:
        # KV-shift-based prompt-prefix reuse. When a follow-up turn
        # shares a prefix with the previous turn (same system prompt,
        # same first N user/assistant rounds, same tool schemas), the
        # server skips re-decoding that prefix and shifts the cached
        # KV state forward by the diff. Non-trivial speed-up on
        # multi-turn chat where the prefix dominates the prompt
        # length. Default 256 is the chunk granularity — smaller
        # values reuse more aggressively at the cost of bookkeeping;
        # larger values miss small in-prefix edits.
        cmd.extend(["--cache-reuse", str(int(cache_reuse))])
    if n_cpu_moe is not None and n_cpu_moe > 0:
        # MoE expert offload to CPU. Counts the number of MoE layers
        # whose expert FFN tensors stay on system RAM while attention
        # + dense parts remain on GPU. Critical for big-MoE models
        # (DeepSeek-V3, Qwen3-MoE 235B, GPT-OSS 120B) that don't fit
        # GPU even with -ngl tuned: experts dominate the file size
        # but each token only routes through a handful of them, so
        # the GPU↔CPU activation hop costs less than streaming all
        # weights from disk via mmap. Counts from the highest-numbered
        # layer downward, so smaller values keep more experts on GPU.
        cmd.extend(["--n-cpu-moe", str(int(n_cpu_moe))])
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


# Cache of supported flags per llama-server binary (keyed by exe
# path). Probing involves a subprocess.run + help-text parse; do
# it once per binary and reuse for the lifetime of this process.
_supported_flags_cache: dict[str, set[str]] = {}


def _llama_server_supports_flag(server_path: str, flag: str) -> bool:
    """Cached probe: does this llama-server build accept the flag?

    llama.cpp churns the llama-server CLI surface — flags appear and
    disappear between builds without a stable deprecation cycle.
    Older builds had ``--prompt-cache``; recent builds removed it
    and replaced the disk-cache feature with in-process slot
    persistence. Building commands without checking blows up the
    spawn with "invalid argument" before the model even loads.

    First call per binary path runs ``llama-server --help``, parses
    every long-form flag out of the output, and caches the set.
    Subsequent calls are O(1) lookups. ``True`` on probe failure so
    we don't accidentally suppress flags on a flaky probe (a fail-
    open here means at worst we get the original "invalid argument"
    we'd see without the cache).
    """
    if not server_path:
        return True
    cached = _supported_flags_cache.get(server_path)
    if cached is None:
        cached = set()
        try:
            r = subprocess.run(
                [server_path, "--help"],
                capture_output=True,
                text=True,
                timeout=10.0,
            )
            help_text = (r.stdout or "") + "\n" + (r.stderr or "")
            # Match every `--<flag>` token. Greedy across word chars
            # plus dash so multi-word flags (`--cache-type-k`) round-trip.
            import re
            for m in re.finditer(r"--[A-Za-z][A-Za-z0-9\-]*", help_text):
                cached.add(m.group(0))
        except Exception as e:
            log.debug(
                "split_lifecycle: --help probe failed on %s (%s); "
                "assuming all flags supported (fail-open)",
                server_path, e,
            )
            # Fail-open: caching the empty set would suppress every
            # flag. Cache None-equivalent so we re-try later.
            return True
        _supported_flags_cache[server_path] = cached
    return flag in cached


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
    gguf_path: str,
    worker_ids: list[str],
    *,
    safety: float = 0.95,
    live_stats: dict | None = None,
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

    `safety=0.95` means "use 95 % of the reported free pool memory" —
    a 5 % buffer purely to absorb allocator alignment overhead and
    avoid crashes on overcommit. The user-set policy is "use as much
    as possible without crashing"; 5 % is the empirical sweet spot
    where llama-server reliably loads a model that fits on paper.

    Falls back to `_DEFAULT_NGL` if any input is missing — e.g.
    a worker without a probe yet, or a GGUF without the
    `<arch>.block_count` metadata key. The default `-ngl 99` keeps
    llama.cpp's old behaviour where it works (small models that fit
    comfortably).

    Realtime adaptation: when ``live_stats`` is supplied (mapping
    ``worker_id -> stats_dict`` from
    ``compute_pool.probe_worker_live_stats``), each worker's
    ``ram_free_gb`` is taken from the FRESH probe rather than the
    capabilities cache. This is what makes the adaptive watchdog in
    ``adaptive_watchdog`` recompute the split when a peer's free
    memory shifts (user opens or closes other apps mid-inference).
    """
    # File size as a proxy for total weight bytes — close enough since
    # quantization metadata + tokenizer KV are small relative to weights.
    try:
        file_size = os.path.getsize(gguf_path)
    except OSError:
        return _DEFAULT_NGL
    if file_size <= 0:
        return _DEFAULT_NGL

    # Block count from cached metadata (mtime-keyed, parses each GGUF
    # exactly once across helpers).
    metadata = _get_gguf_metadata(gguf_path)
    n_layers = metadata.get("block_count") or 0
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
        host_vram_bytes = int(float(spec.get("vram_gb") or 0) * (1024 ** 3))
        pool_free_bytes += host_vram_bytes
        # When the host has no dGPU (vram=0) it's a CPU-only machine —
        # in that case its own system RAM IS the offload budget. We
        # don't count host RAM when there's a real dGPU because the
        # non-GPU layers ride mmap from disk anyway, but for a
        # CPU-only host (laptop with iGPU only) we'd otherwise
        # vastly under-report the pool. psutil.virtual_memory().available
        # reflects what the OS will actually hand us right now.
        if host_vram_bytes == 0:
            try:
                import psutil
                host_ram_free = int(psutil.virtual_memory().available)
                pool_free_bytes += host_ram_free
            except Exception:
                pass
    except Exception:
        pass

    for wid in worker_ids:
        w = db.get_compute_worker(wid)
        if not w or not w.get("enabled"):
            continue
        # Prefer fresh live-probe stats when supplied by the adaptive
        # watchdog. Falls back to the capabilities cache so existing
        # one-shot callers (e.g. start()) keep their old behaviour.
        free_gb = 0.0
        if live_stats and wid in live_stats:
            try:
                free_gb = float(live_stats[wid].get("ram_free_gb") or 0)
            except (TypeError, ValueError):
                free_gb = 0.0
        if free_gb <= 0:
            caps = w.get("capabilities") or {}
            free_gb = float(caps.get("ram_free_gb") or 0)
        else:
            caps = w.get("capabilities") or {}
        # Per-endpoint contribution (multi-rpc-server world):
        # When the worker exposes BOTH an iGPU (SYCL/Vulkan) endpoint
        # AND a CPU endpoint via rpc_endpoints, the pool budget for
        # this worker is the SUM of:
        #   - iGPU capacity (capped at vram_total * 0.85 — the iGPU
        #     can't allocate more than its shared-memory pool)
        #   - CPU capacity (worker's free RAM, minus what the iGPU
        #     will draw from the same physical pool on Intel UMA)
        # Sum != worker's free RAM because Intel UMA charges iGPU
        # use against system RAM. The cap matches what
        # `_compute_tensor_split_ratios` will actually request.
        endpoints = caps.get("rpc_endpoints")
        if isinstance(endpoints, list) and endpoints:
            vram_total_gb = float(caps.get("vram_total_gb") or 0)
            has_gpu = any(
                any(k in (ep.get("backend") or "").lower() for k in ("sycl", "vulkan", "cuda"))
                for ep in endpoints
            )
            has_cpu = any(
                "cpu" == (ep.get("backend") or "").lower().strip()
                or (ep.get("backend") or "").lower().endswith(",cpu")
                for ep in endpoints
            )
            worker_pool_gb = 0.0
            if has_gpu and vram_total_gb > 0:
                worker_pool_gb += vram_total_gb * 0.85
            if has_cpu:
                # Intel UMA: iGPU draws from system RAM, so subtract
                # what the iGPU will take. Floor at 0.5 GB so a
                # tiny CPU contribution still counts.
                cpu_share = max(0.5, free_gb - vram_total_gb * 0.85)
                worker_pool_gb += cpu_share
            elif not has_gpu:
                # CPU-only worker — no iGPU subtraction.
                worker_pool_gb += free_gb
            pool_free_bytes += int(worker_pool_gb * (1024 ** 3))
        else:
            # Legacy single-rpc-server worker: full ram_free as the
            # contribution. Matches the prior behaviour.
            pool_free_bytes += int(free_gb * (1024 ** 3))

    if pool_free_bytes <= 0:
        return _DEFAULT_NGL

    # When ANY worker exposes multiple rpc-server endpoints (multi-
    # rpc-server pattern), defer to llama.cpp's auto-fit by emitting
    # the sentinel ngl=0. Reason: with small heterogeneous iGPU
    # devices (~2 GB Intel Iris Xe / Arc), a fixed `-ngl` causes
    # b9002+ llama-server to abort with "common_fit_params: failed
    # to fit ... abort" because the per-device share for the iGPU
    # endpoints exceeds their VRAM ceiling. Auto-fit walks devices
    # and shrinks ngl until everything fits — much more reliable
    # than our pool-summed math which doesn't know about per-device
    # GPU memory hard caps that llama.cpp's allocator enforces.
    has_multi_rpc = False
    for wid in worker_ids:
        w = db.get_compute_worker(wid)
        if not w:
            continue
        eps = (w.get("capabilities") or {}).get("rpc_endpoints")
        if isinstance(eps, list) and len(eps) > 1:
            has_multi_rpc = True
            break
    if has_multi_rpc:
        log.info(
            "split_lifecycle: multi-rpc endpoints detected, deferring to "
            "llama.cpp auto-fit (ngl=0 sentinel)",
        )
        return 0

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

    # Ensure every worker's rpc-server is up with the right backend
    # before the split spawn. `_select_worker_backend` returns
    # `SYCL0,CPU` for Intel workers (iGPU + CPU both exposed to the
    # layer-placement pool) — the user-set policy is "use every
    # resource available". The historical SYCL+RPC crashes that used
    # to require dropping SYCL during split (#21420 / #20259 /
    # #21474) are all fixed upstream by build 8940. SYCL kernel
    # quirks that remain open (#21893) are handled by the env vars
    # (`GGML_SYCL_DISABLE_OPT=1`, `GGML_SYCL_DISABLE_GRAPH=1`) we
    # set when spawning rpc-server.
    #
    # Idempotent: `_set_workers_backend` is a no-op when the
    # current backend already matches. Routes through the new P2P
    # path (via `_attempt_rpc_server_restart` -> P2P-first) so this
    # works on paired peers without ssh_host.
    worker_ids = row.get("worker_ids") or []
    workers_for_split = [db.get_compute_worker(wid) for wid in worker_ids]
    workers_for_split = [w for w in workers_for_split if w]
    if workers_for_split:
        try:
            from . import compute_pool as _pool
            await _pool._set_workers_backend(workers_for_split, in_split=True)
        except Exception as e:
            log.warning(
                "split_lifecycle: backend-switch to CPU before split "
                "spawn failed (%s); continuing — split may crash on "
                "RPC slot init if workers are still on SYCL", e,
            )

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

    # Adaptive prompt-eval batch sizes. Returns (None, None) on tight
    # pools so llama.cpp's defaults stay in effect there; bumps `-b`
    # and `-ub` on workstations with VRAM headroom for 2-4× faster
    # prefill on long prompts.
    batch_size, ubatch_size = _compute_optimal_batch_sizes(
        row["gguf_path"],
        parallel=parallel,
        ctx_size=ctx_size,
        cache_type=cache_type,
        target_size_bytes=target_size_bytes,
        draft_size_bytes=draft_size,
    )

    # Adaptive thread counts: on hyperthreaded CPUs the llama-server
    # default (logical core count) over-subscribes the FPU and slows
    # decode. Decode runs at physical core count; prefill stays at
    # logical (it's batched matmul that benefits from SMT).
    threads, threads_batch = _recommend_thread_counts()

    # Adaptive --mlock: pin weights in RAM so the OS can't page them
    # out under memory pressure. Engages only when free RAM is at
    # least 2× the model file so locking can't OOM the system.
    use_mlock = _should_mlock_weights(target_size_bytes)
    # Decide --no-mmap: lift weights into private RAM whenever the
    # model fits + on Windows CUDA always. See `_should_disable_mmap`
    # for the rationale (mirrors Ollama's policy). This is what
    # produces visible 95 %-class memory utilisation on every
    # participating device — without it the weights ride the file
    # cache and the OS under-reports the load.
    use_no_mmap = _should_disable_mmap(target_size_bytes, worker_ids)

    # Adaptive --n-cpu-moe: when the model is MoE AND the file is
    # bigger than the GPU pool, expert FFN tensors get pinned to
    # CPU/RAM. Attention + dense weights stay on GPU. Big-MoE
    # (DeepSeek-V3, Qwen3-MoE 235B, GPT-OSS 120B) wins enormously
    # from this — token routing only fires a handful of experts per
    # forward pass, so the GPU↔CPU activation hop costs less than
    # streaming weights from disk via mmap.
    n_cpu_moe = _compute_optimal_n_cpu_moe(
        row["gguf_path"], row["worker_ids"],
        ngl=ngl, parallel=parallel, cache_type=cache_type,
        ctx_size=ctx_size, target_size_bytes=target_size_bytes,
    )

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
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        threads=threads,
        threads_batch=threads_batch,
        mlock=use_mlock,
        no_mmap=use_no_mmap,
        n_cpu_moe=n_cpu_moe,
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
        # DETACHED_PROCESS + CREATE_BREAKAWAY_FROM_JOB so llama-server
        # outlives whichever Python process spawned it. Without these,
        # a CLI / test script that calls split_lifecycle.start() and
        # then exits drags llama-server with it (Windows job-object
        # inheritance), which silently breaks the next chat against
        # the same model. The live FastAPI backend doesn't need this
        # for its own lifecycle (it stays running) but adding the
        # flags makes `start()` work the same way regardless of caller.
        DETACHED_PROCESS = 0x00000008
        CREATE_BREAKAWAY_FROM_JOB = 0x01000000
        creationflags = (
            DETACHED_PROCESS
            | CREATE_BREAKAWAY_FROM_JOB
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.BELOW_NORMAL_PRIORITY_CLASS
        )

    # Inject SYCL safety env vars so a host with an Intel iGPU
    # doesn't hit #21893 (still OPEN upstream — Intel Xe2 / Meteor
    # Lake silently corrupts weights / panics during decode without
    # `GGML_SYCL_DISABLE_OPT=1`). We unconditionally set them on
    # every llama-server spawn — no harm on hosts without SYCL,
    # mandatory on hosts with it. Same env vars that
    # `p2p_rpc_server._RPC_SPAWN_ENV` sets on rpc-server.
    spawn_env = dict(os.environ)
    spawn_env.setdefault("GGML_SYCL_DISABLE_OPT", "1")
    spawn_env.setdefault("GGML_SYCL_DISABLE_GRAPH", "1")
    spawn_env.setdefault("SYCL_CACHE_PERSISTENT", "1")
    # Per-RPC-call timeout. Without this, when an rpc-server peer
    # silently dies (laptop sleep, OOM, network drop, SYCL crash),
    # llama-server's RPC dispatch hangs indefinitely waiting for
    # the dead peer's response — the user sees a chat that never
    # finishes. 30 seconds is generous (a healthy peer answers a
    # layer push in tens of ms even on Wi-Fi); anything longer
    # than this and the orchestrator should treat the peer as
    # gone, surface an error, and let the watchdog's crash detector
    # mark + auto-fall-back. The env var is read by ggml-rpc.cpp's
    # send/recv path inside llama-server.
    spawn_env.setdefault("GGML_RPC_TIMEOUT", "30000")  # milliseconds

    log_file = log_path.open("ab")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,   # merge into one log file
            stdin=subprocess.DEVNULL,
            env=spawn_env,
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

    now = time.time()
    _running[split_id] = _RunningProcess(
        proc=proc,
        port=row["llama_port"],
        started_at=now,
        # Initialise touched-at to the start time so a freshly-loaded
        # split gets its full idle grace period before the GC could
        # reclaim it (otherwise a split that's loaded but not yet
        # used by chat would be eligible immediately).
        last_touched_at=now,
        log_path=log_path,
        cmd=cmd,
        ngl=ngl,
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
        # Record per-worker iGPU-backend failure so the auto-fallback
        # selector tries the next preference (SYCL -> Vulkan -> CPU)
        # on the next start attempt. We mark every Intel worker that
        # was attached to this split — we can't always tell which
        # worker's iGPU was the one that crashed, but the cool-down
        # window means a worker whose iGPU is fine just retries
        # tomorrow with no permanent loss.
        try:
            from . import compute_pool as _pool
            for wid in (row.get("worker_ids") or []):
                w = db.get_compute_worker(wid)
                if not w:
                    continue
                caps = w.get("capabilities") or {}
                current = caps.get("current_rpc_backend") or ""
                if "sycl" in current.lower() or "vulkan" in current.lower():
                    _pool.record_backend_failure(wid, current)
        except Exception as fe:
            log.debug(
                "split_lifecycle: failure-detection hook failed: %s", fe,
            )
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


# ---------------------------------------------------------------------------
# Adaptive layer-rebalance watchdog
# ---------------------------------------------------------------------------
#
# Goal: when a worker's free RAM changes (user opens Chrome → less RAM,
# user closes Chrome → more RAM), shift the GPU-offload layer count
# (`-ngl`) so the running split-model uses what's actually available.
#
# Without this loop the layer split is decided ONCE at start and
# baked into the running llama-server's argv. If the worker's free
# RAM drops by 50 % mid-inference we either crash an rpc-server or
# leave the worker doing nothing while the host CPU pages from
# disk — both directly contradict the user's "use as much resource
# as available with only 5 % margin" policy.
#
# Algorithm (one tick):
#   for each running split-model:
#     1. Probe live free RAM on each worker (encrypted P2P call).
#     2. Recompute optimal ngl with the fresh stats.
#     3. If |new_ngl - current_ngl| >= rebalance_threshold AND the
#        last restart for this split_id was > cool-down ago:
#          stop() then start() — start() re-reads live stats again.
#     4. Cool-down stamp on rebalance to prevent thrash.
#
# Cool-down + threshold are both required: threshold alone allows
# oscillation between two adjacent layer counts; cool-down alone
# allows a single big drop to trigger immediately even if the
# delta is just one layer (we'd rather be slightly stale than
# constantly restarting a multi-GB process).

# How often to sample live stats. 10 s is short enough to react to a
# user opening Slack or Chrome (those typically stabilise within a
# few seconds), long enough that a /24-paired-peer probe doesn't
# meaningfully tax the LAN. Idle when no split-model is loaded.
_ADAPTIVE_TICK_SEC = 10.0

# Minimum layer-count delta to trigger a rebalance restart. Anything
# below this is in the noise — a single layer of `gemma4:e4b` is
# ~150 MB and llama-server alignment overhead can move that much
# between probes. Set conservatively so we only restart when there's
# a real win to be had.
_REBALANCE_MIN_DELTA_LAYERS = 3

# Minimum interval between rebalance restarts of the same split_id.
# llama-server warm-up is 5-30 s depending on model size; restarting
# inside that window would mean the user spends more time waiting
# than running. 60 s ensures the model gets a meaningful chunk of
# productive runtime between any two rebalances.
_REBALANCE_COOLDOWN_SEC = 60.0

# Idle window after which the watchdog reclaims a split-model that's
# loaded but unused. Set to 8 minutes — long enough that a user who
# stepped away for a coffee comes back to a hot model, short enough
# that an abandoned split-model doesn't permanently squat on
# multi-GB of orchestrator RAM and worker RAM. The user can re-engage
# at any time by sending a chat; route_chat_for re-creates and
# re-starts the split with whatever pool resources are available
# at that moment.
_IDLE_GC_SEC = 480.0

# Peer-loss detection. The watchdog probes every worker attached to
# a running split each tick; if a worker's stats probe fails this
# many ticks in a row, we treat that peer as gone (laptop closed,
# Wi-Fi dropped, backend crashed) and drop it from the split's
# worker list. The split is then restarted with the surviving
# peers — better to lose a layer-budget chunk than have llama-server
# hang waiting for a dead RPC endpoint.
_PEER_LOSS_FAIL_THRESHOLD = 3

# Per-(split_id, worker_id) consecutive-failure counter for the
# peer-loss detector. Reset to 0 on every successful probe.
# Module-level to survive across ticks.
_peer_fail_counts: dict[tuple[str, str], int] = {}


_adaptive_task: asyncio.Task | None = None


async def _adaptive_tick() -> None:
    """One pass of the adaptive watchdog. Public so tests can drive it.

    Reads `_running` (the live process registry) and adjusts each
    entry's `-ngl` if the optimal value computed from current free
    RAM has drifted far enough. Errors are caught per-entry so a
    flaky worker doesn't take the whole loop down.
    """
    if not _running:
        return
    from . import compute_pool as _pool
    now = time.time()
    # Snapshot the registry — `stop()` mutates `_running` and we
    # don't want a "dictionary changed during iteration" error.
    live_split_ids = list(_running.keys())

    # Crash-detection pass: any split whose llama-server process
    # exited unexpectedly gets cleaned up here, AND each Intel
    # worker that was attached gets its current iGPU backend
    # marked failed so the next start picks the next-down option
    # (SYCL -> Vulkan -> CPU). This is what catches the
    # mid-decode crash mode where llama-server dies silently after
    # the chat returns "ConnectError" to the agent.
    crashed_targets = []
    for split_id in live_split_ids:
        rp = _running.get(split_id)
        if rp is None:
            continue
        rc = rp.proc.poll()
        if rc is not None:
            crashed_targets.append((split_id, rc))
    for split_id, rc in crashed_targets:
        row = db.get_split_model(split_id) or {}
        worker_ids = row.get("worker_ids") or []
        log.warning(
            "adaptive_watchdog: split %s exited unexpectedly (rc=%s); "
            "marking attached Intel workers' current iGPU backend as "
            "failed so the next start auto-falls-back",
            split_id, rc,
        )
        for wid in worker_ids:
            try:
                w = db.get_compute_worker(wid)
                if not w:
                    continue
                caps = w.get("capabilities") or {}
                current = caps.get("current_rpc_backend") or ""
                if "sycl" in current.lower() or "vulkan" in current.lower():
                    _pool.record_backend_failure(wid, current)
            except Exception as e:
                log.debug(
                    "adaptive_watchdog: failure-record skip for %s: %s",
                    wid, e,
                )
        # Drop from registry so we don't re-enter on next tick.
        _running.pop(split_id, None)
        try:
            db.update_split_model_status(
                split_id, status="error",
                last_error=f"llama-server exited mid-run (rc={rc})",
            )
        except Exception:
            pass

        # Auto-rebuild with surviving workers. Volatile-network case:
        # a worker laptop sleeps mid-inference, the orchestrator
        # detects the crash, marks the SYCL backend failed, and
        # NOW we re-spawn with whichever workers are still
        # heartbeat-reachable. Falls through cleanly if NO workers
        # remain (split row stays in error status; next chat turn
        # triggers route_chat_for which goes single-host).
        try:
            surviving = []
            for wid in worker_ids:
                w = db.get_compute_worker(wid)
                if not w:
                    continue
                caps = w.get("capabilities") or {}
                if caps.get("rpc_server_reachable"):
                    surviving.append(wid)
            if surviving and len(surviving) < len(worker_ids):
                log.info(
                    "adaptive_watchdog: rebuilding split %s with %d "
                    "surviving worker(s) (was %d)",
                    split_id, len(surviving), len(worker_ids),
                )
                # Update worker_ids to only living peers and restart.
                db.update_split_model(
                    split_id, worker_ids=surviving,
                )
                res = await start(split_id)
                if not res.get("ok"):
                    log.warning(
                        "adaptive_watchdog: rebuild failed: %s",
                        res.get("error"),
                    )
            elif surviving:
                # All original workers still reachable — just retry.
                log.info(
                    "adaptive_watchdog: retrying split %s with same "
                    "workers (auto-fallback should now pick a safer "
                    "backend per failure flags)",
                    split_id,
                )
                res = await start(split_id)
                if not res.get("ok"):
                    log.warning(
                        "adaptive_watchdog: retry failed: %s",
                        res.get("error"),
                    )
        except Exception as e:
            log.warning(
                "adaptive_watchdog: post-crash rebuild failed for "
                "%s: %s", split_id, e,
            )
    if crashed_targets:
        live_split_ids = list(_running.keys())

    # Idle GC pass — reclaim any split that hasn't seen chat traffic
    # in `_IDLE_GC_SEC`. Walks the snapshot, stops eligible entries,
    # then rebuilds `live_split_ids` to skip them in the rebalance
    # pass below. This keeps the orchestrator's RAM (and each
    # worker's RAM) free for other work when the user has moved
    # on to a different model — the next chat will re-create the
    # split with whatever resources are available at that moment.
    gc_targets = []
    for split_id in live_split_ids:
        rp = _running.get(split_id)
        if rp is None:
            continue
        if rp.proc.poll() is not None:
            # Process already gone; let the boot-reconcile clean
            # the registry on next restart. Skip from both passes.
            continue
        idle_for = now - (rp.last_touched_at or rp.started_at)
        if idle_for >= _IDLE_GC_SEC:
            gc_targets.append((split_id, idle_for))
    for split_id, idle_for in gc_targets:
        log.info(
            "adaptive_watchdog: reclaiming split %s — idle for %.0fs "
            "(no chat traffic in %d s window). User can re-engage at "
            "any time by sending a chat against the model.",
            split_id, idle_for, int(_IDLE_GC_SEC),
        )
        try:
            await stop(split_id)
        except Exception as e:
            log.warning(
                "adaptive_watchdog: idle GC stop failed for %s: %s",
                split_id, e,
            )
    if gc_targets:
        # Refresh the snapshot so the rebalance pass below doesn't
        # try to operate on splits we just stopped.
        live_split_ids = list(_running.keys())
    if not live_split_ids:
        return
    for split_id in live_split_ids:
        running = _running.get(split_id)
        if running is None:
            continue
        # Process gone? Let `stop()` / boot-reconcile clean up — the
        # watchdog only cares about live processes whose layer count
        # we can change.
        if running.proc.poll() is not None:
            continue
        row = db.get_split_model(split_id)
        if not row:
            continue
        worker_ids = row.get("worker_ids") or []
        if not worker_ids:
            continue
        # Fetch fresh stats from each worker. We deliberately do this
        # one peer at a time rather than gather()-all so a single
        # slow peer doesn't block the whole rebalance — we can still
        # decide based on the workers that responded, and a missing
        # worker falls through to its capabilities-cached value.
        live_stats: dict[str, dict] = {}
        for wid in worker_ids:
            w = db.get_compute_worker(wid)
            if not w:
                continue
            try:
                stats = await _pool.probe_worker_live_stats(w)
            except Exception as e:
                log.debug(
                    "adaptive_watchdog: probe failed for %s: %s",
                    w.get("label"), e,
                )
                stats = {}
            if stats:
                live_stats[wid] = stats
                # Persist the fresh ram_free_gb back to the worker's
                # capabilities so other code paths (UI display, the
                # 5-min sweep) see the latest value too. Cheap — one
                # JSON merge + UPDATE.
                try:
                    caps = dict(w.get("capabilities") or {})
                    caps["ram_free_gb"] = float(stats.get("ram_free_gb") or 0)
                    caps["ram_free_probed_at"] = now
                    db.update_compute_worker_capabilities(
                        wid, capabilities=caps,
                    )
                except Exception as e:
                    log.debug(
                        "adaptive_watchdog: capability persist failed for "
                        "%s: %s", w.get("label"), e,
                    )
        # Recompute optimal ngl from the fresh data.
        try:
            new_ngl = _compute_optimal_ngl(
                row["gguf_path"], worker_ids, live_stats=live_stats,
            )
        except Exception as e:
            log.warning(
                "adaptive_watchdog: ngl recompute failed for split %s: %s",
                split_id, e,
            )
            continue
        old_ngl = running.ngl or 0
        delta = abs(new_ngl - old_ngl)
        if delta < _REBALANCE_MIN_DELTA_LAYERS:
            continue
        # Cool-down — never restart back-to-back.
        if (now - running.last_rebalance_at) < _REBALANCE_COOLDOWN_SEC:
            log.debug(
                "adaptive_watchdog: split %s wants ngl=%d (was %d), "
                "but in cool-down for %.1fs more",
                split_id, new_ngl, old_ngl,
                _REBALANCE_COOLDOWN_SEC - (now - running.last_rebalance_at),
            )
            continue
        # Mark BEFORE stop/start so a restart that takes 30 s of
        # warm-up doesn't immediately re-trigger on the next tick.
        running.last_rebalance_at = now
        log.info(
            "adaptive_watchdog: rebalancing split %s: ngl %d -> %d "
            "(pool free RAM shifted; restarting llama-server with "
            "new offload count)",
            split_id, old_ngl, new_ngl,
        )
        try:
            await stop(split_id)
            res = await start(split_id)
            if not res.get("ok"):
                log.warning(
                    "adaptive_watchdog: restart failed for split %s: %s",
                    split_id, res.get("error"),
                )
        except Exception as e:
            log.warning(
                "adaptive_watchdog: rebalance failed for split %s: %s",
                split_id, e,
            )


async def _adaptive_loop() -> None:
    """Background task body. Runs forever until cancelled."""
    while True:
        try:
            await _adaptive_tick()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("adaptive_watchdog: tick failed: %s", e)
        try:
            await asyncio.sleep(_ADAPTIVE_TICK_SEC)
        except asyncio.CancelledError:
            raise


def start_adaptive_watchdog() -> None:
    """Schedule the adaptive watchdog. Idempotent — calling twice is
    a no-op. Called from app.py's startup hook."""
    global _adaptive_task
    if _adaptive_task is not None and not _adaptive_task.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    _adaptive_task = loop.create_task(
        _adaptive_loop(), name="split_adaptive_watchdog",
    )
    log.info("split_lifecycle: adaptive watchdog started (tick=%ds)", _ADAPTIVE_TICK_SEC)


def stop_adaptive_watchdog() -> None:
    """Cancel the watchdog. Called from app shutdown so the test
    runner doesn't see a stranded task."""
    global _adaptive_task
    if _adaptive_task is not None and not _adaptive_task.done():
        _adaptive_task.cancel()
    _adaptive_task = None
