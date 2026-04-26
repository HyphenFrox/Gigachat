"""Adaptive `-ngl` with multi-tenant accounting.

`compute_optimal_ngl(gguf, workers, *, reserved_bytes, my_priority,
other_priorities)` returns a `(ngl, estimated_bytes)` tuple that:

* respects pool memory already reserved by other live claims,
* applies priority-weighted fair share when the pool is contested,
* degrades gracefully (ngl=0 → caller runs host-only) when even our
  share can't fit a single layer rather than overcommitting,
* falls back to llama.cpp's default `99` when GGUF metadata is
  unreadable, so we never block a workload on missing precision.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_NGL = 99
SAFETY_FACTOR = 0.7  # leave 30% headroom for KV cache + work buffers


def _read_gguf_n_layers(gguf_path: str) -> int:
    """Open the GGUF, return its block_count (transformer layer count)
    or 0 if metadata is missing / unreadable.
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

    for key in (f"{arch}.block_count", "llama.block_count", "general.block_count"):
        f = reader.fields.get(key)
        if f and f.data:
            try:
                return int(f.parts[f.data[0]][0])
            except Exception:
                continue
    return 0


def _detect_host_vram_gb() -> float:
    """Best-effort host VRAM total. Tries `nvidia-smi`, returns 0 if
    not available (caller treats 0 as "no GPU").
    """
    import shutil
    if not shutil.which("nvidia-smi"):
        return 0.0
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return 0.0
        # Memory in MB; pick the first GPU.
        first = r.stdout.strip().splitlines()[0].strip()
        return float(first) / 1024.0
    except Exception:
        return 0.0


def compute_optimal_ngl(
    gguf_path: str,
    workers: list[dict[str, Any]] | None = None,
    *,
    safety: float = SAFETY_FACTOR,
    reserved_bytes: int = 0,
    my_priority: int = 100,
    other_priorities: list[int] | None = None,
) -> tuple[int, int]:
    """Decide how many layers to put on the pool, accounting for any
    other workloads currently using it.

    Returns `(ngl, estimated_bytes)`:
      * `ngl`: integer for `-ngl` flag.
      * `estimated_bytes`: how much pool memory this allocation will
        consume (caller registers this with the claim so the *next*
        workload knows what's reserved).

    Allocation policy
    -----------------
    1. `pool_total = host_vram + sum(worker free RAM)` × safety
       (default safety 0.7 = 30 % headroom for KV cache + buffers).
    2. `pool_free = pool_total - reserved_bytes` (subtract what other
       active workloads have already claimed).
    3. Under contention, split `pool_free` proportionally by priority:
         my_share = pool_free × (my_priority /
                                 (my_priority + sum(other_priorities)))
       When uncontested, `my_share = pool_free`.
    4. `ngl = floor(my_share / avg_layer_bytes)`, clamped to
       `[0, n_layers]`.
    5. Graceful degradation: if `ngl == 0` (pool can't fit even one
       layer for our share), we return `(0, 0)` — caller skips `--rpc`
       and runs host-only rather than overcommitting.
    6. If GGUF metadata is unreadable / pool is empty, return
       `(DEFAULT_NGL, 0)` — caller delegates to llama.cpp's default
       "fit as many as possible" probe.

    Parameters
    ----------
    reserved_bytes :
        Sum of bytes already budgeted by other live claims. Pass
        `registry.total_reserved_bytes(exclude_pid=os.getpid())`.
    my_priority, other_priorities :
        Integer weights. Higher = more pool share when contested.
        Default 100. Pass priorities from `registry.get_active_claims()`
        for a fair split.
    """
    workers = workers or []
    other_priorities = other_priorities or []

    n_layers = _read_gguf_n_layers(gguf_path)
    if n_layers <= 0:
        return DEFAULT_NGL, 0

    try:
        file_size = os.path.getsize(gguf_path)
    except OSError:
        return DEFAULT_NGL, 0
    if file_size <= 0:
        return DEFAULT_NGL, 0

    avg_layer_bytes = file_size / n_layers

    pool_total = 0
    pool_total += int(_detect_host_vram_gb() * (1024 ** 3) * 0.8)
    for w in workers:
        if not w.get("enabled", True):
            continue
        free_gb = float(w.get("ram_free_gb") or 0)
        pool_total += int(free_gb * (1024 ** 3))

    if pool_total <= 0:
        return DEFAULT_NGL, 0

    pool_free = max(0, pool_total - int(reserved_bytes))

    # Priority-weighted share when contested.
    total_priority = my_priority + sum(max(1, p) for p in other_priorities)
    my_share = (
        pool_free * (my_priority / total_priority)
        if other_priorities else pool_free
    )

    fits = int(my_share * safety / avg_layer_bytes)
    fits = max(0, min(n_layers, fits))
    estimated = int(fits * avg_layer_bytes)
    log.info(
        "adaptive ngl: file=%.2fGB n_layers=%d avg_layer=%.0fMiB "
        "pool_total=%.2fGB reserved=%.2fGB my_share=%.2fGB "
        "priority=%d/(self+%d others) safety=%.2f -> ngl=%d "
        "(reserves %.2fGB)",
        file_size / (1024 ** 3), n_layers, avg_layer_bytes / (1024 ** 2),
        pool_total / (1024 ** 3), reserved_bytes / (1024 ** 3),
        my_share / (1024 ** 3),
        my_priority, len(other_priorities), safety, fits,
        estimated / (1024 ** 3),
    )
    return fits, estimated
