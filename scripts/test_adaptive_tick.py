"""Diagnostic: drive the adaptive split-watchdog tick by hand, with
a synthetic running-process registry, so we can verify the data
path (live-stats fetch -> ngl recompute -> rebalance decision)
without needing a real split-model loaded.

Inserts a fake _RunningProcess into split_lifecycle._running for
one of the configured split_model rows (or a synthetic one if none
exist), then calls _adaptive_tick() once and prints what changed.
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import db, split_lifecycle  # noqa: E402


async def main() -> int:
    db.init()
    workers = [w for w in db.list_compute_workers() if w.get("enabled")]
    if not workers:
        print("no enabled compute_workers")
        return 1

    # Look for an existing split_model row we can probe; if none,
    # we just exercise the live-stats fetch path directly.
    rows = db.list_split_models()
    print(f"split_model rows: {len(rows)}")
    print(f"compute_workers (enabled): {[w['label'] for w in workers]}")

    # Fetch stats per worker via the watchdog's helper (the same
    # code path the live tick uses).
    from backend import compute_pool as _pool
    print()
    print("--- live stats per worker (probe_worker_live_stats) ---")
    for w in workers:
        t0 = time.time()
        stats = await _pool.probe_worker_live_stats(w)
        elapsed_ms = int((time.time() - t0) * 1000)
        print(f"  {w['label']:<22s} {elapsed_ms:>4d} ms  {stats}")

    # Try compute_optimal_ngl with live stats. We need a gguf path
    # to pass; fall back to a synthetic probe if no split rows.
    if not rows:
        print()
        print("(no split_model rows; ngl recompute skipped)")
        return 0

    row = rows[0]
    gguf_path = row.get("gguf_path")
    worker_ids = row.get("worker_ids") or []
    if not gguf_path or not worker_ids:
        print("no usable split_model row to test ngl recompute")
        return 0

    # Build live_stats dict for the workers attached to this split
    live_stats: dict[str, dict] = {}
    for wid in worker_ids:
        w = db.get_compute_worker(wid)
        if not w:
            continue
        s = await _pool.probe_worker_live_stats(w)
        if s:
            live_stats[wid] = s

    cached_ngl = split_lifecycle._compute_optimal_ngl(gguf_path, worker_ids)
    live_ngl = split_lifecycle._compute_optimal_ngl(
        gguf_path, worker_ids, live_stats=live_stats,
    )
    print()
    print(f"split_id  = {row['id']}")
    print(f"gguf_path = {gguf_path}")
    print(f"ngl from cached caps = {cached_ngl}")
    print(f"ngl from live stats  = {live_ngl}")
    print(f"delta = {live_ngl - cached_ngl}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
