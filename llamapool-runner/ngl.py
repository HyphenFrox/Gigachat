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
# Cooperative mode: leave 30 % headroom for KV cache + work buffers
# AND for fluctuations in other apps' working sets (so we don't
# accidentally evict their pages).
SAFETY_FACTOR_COOPERATIVE = 0.7
# Aggressive mode: tighter margin since we're explicitly NOT yielding
# to other apps. Still keep some headroom because KV cache + work
# buffers are real even when the pool owns everything.
SAFETY_FACTOR_AGGRESSIVE = 0.85
SAFETY_FACTOR = SAFETY_FACTOR_COOPERATIVE  # back-compat default


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
    safety: float | None = None,
    reserved_bytes: int = 0,
    my_priority: int = 100,
    other_priorities: list[int] | None = None,
    aggressive: bool = False,
) -> tuple[int, int]:
    """Decide how many layers to put on the pool, accounting for any
    other workloads currently using it.

    Returns `(ngl, estimated_bytes)`:
      * `ngl`: integer for `-ngl` flag.
      * `estimated_bytes`: how much pool memory this allocation will
        consume (caller registers this with the claim so the *next*
        workload knows what's reserved).

    Two memory-source modes
    -----------------------
    * **Cooperative** (`aggressive=False`, default) — pool budget is
      built from each device's *currently free* memory and the host's
      VRAM scaled by 0.8. Yields gracefully to whatever else the OS
      is running. Safety factor 0.7 leaves headroom for KV cache +
      work buffers + other-app fluctuations. **This is the right
      mode for hosts where the user is also doing other things.**
    * **Aggressive** (`aggressive=True`) — pool budget is built from
      each device's *total* RAM (×0.85) and the host's VRAM (×0.95).
      We don't yield to other apps; the pool will displace the OS
      page cache to take what it wants. Safety factor 0.85 (still
      keeping ~15 % for KV/buffers since those are real regardless
      of OS cooperation). Use ONLY on dedicated machines or when the
      operator explicitly accepts that other apps may slow down.

    Inter-pool sharing (claims registry, priority-weighted shares)
    works identically in both modes — `aggressive` only changes the
    boundary between "the pool" and "everything else on the OS".

    Allocation policy
    -----------------
    1. `pool_total = host_vram_bytes + sum(worker memory bytes)` —
       sourced from `ram_free_gb` (cooperative) or `ram_total_gb`
       (aggressive).
    2. `pool_free = pool_total - reserved_bytes` (subtract what other
       active claims have already budgeted — same in both modes).
    3. Priority-weighted share when contested.
    4. `ngl = floor(my_share × safety / avg_layer_bytes)`, clamped
       to `[0, n_layers]`.

    Parameters
    ----------
    reserved_bytes :
        Sum of bytes already budgeted by other live claims. Pass
        `registry.total_reserved_bytes(exclude_pid=os.getpid())`.
    my_priority, other_priorities :
        Integer weights. Higher = more pool share when contested.
    aggressive :
        Toggle the mode described above.
    safety :
        Override the safety factor explicitly. When `None`, picks
        0.7 (cooperative) or 0.85 (aggressive) automatically.
    """
    workers = workers or []
    other_priorities = other_priorities or []
    if safety is None:
        safety = SAFETY_FACTOR_AGGRESSIVE if aggressive else SAFETY_FACTOR_COOPERATIVE

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

    # Build the pool budget. Aggressive uses total memory with high
    # utilisation factors; cooperative uses currently-free memory.
    pool_total = 0
    host_vram_factor = 0.95 if aggressive else 0.8
    pool_total += int(_detect_host_vram_gb() * (1024 ** 3) * host_vram_factor)
    for w in workers:
        if not w.get("enabled", True):
            continue
        if aggressive:
            # Use worker's TOTAL RAM. Fall back to ram_free_gb if
            # the snapshot doesn't include total (older configs).
            total_gb = float(w.get("ram_total_gb") or w.get("ram_free_gb") or 0)
            pool_total += int(total_gb * (1024 ** 3) * 0.85)
        else:
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
        "adaptive ngl (%s): file=%.2fGB n_layers=%d avg_layer=%.0fMiB "
        "pool_total=%.2fGB reserved=%.2fGB my_share=%.2fGB "
        "priority=%d/(self+%d others) safety=%.2f -> ngl=%d "
        "(reserves %.2fGB)",
        "aggressive" if aggressive else "cooperative",
        file_size / (1024 ** 3), n_layers, avg_layer_bytes / (1024 ** 2),
        pool_total / (1024 ** 3), reserved_bytes / (1024 ** 3),
        my_share / (1024 ** 3),
        my_priority, len(other_priorities), safety, fits,
        estimated / (1024 ** 3),
    )
    return fits, estimated
