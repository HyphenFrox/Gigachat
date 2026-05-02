"""Print one peer's live ram_free_gb. Single-line output for easy
piping. Filters by device-id substring or label substring.

Usage:
    python probe_one_peer.py DESKTOP-0692HOK
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import db, compute_pool as _pool  # noqa: E402


async def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: probe_one_peer.py <label_or_devid_substr>")
        return 2
    needle = argv[1].lower()
    db.init()
    workers = [
        w for w in db.list_compute_workers()
        if w.get("enabled")
        and (
            needle in (w.get("label") or "").lower()
            or needle in (w.get("gigachat_device_id") or "").lower()
        )
    ]
    if not workers:
        print(f"no enabled worker matches {argv[1]!r}")
        return 1
    w = workers[0]
    stats = await _pool.probe_worker_live_stats(w)
    if not stats:
        print(f"{w['label']}: NO STATS")
        return 1
    import time as _t
    print(
        f"{_t.strftime('%H:%M:%S')}  {w['label']:<22s}  "
        f"free={stats.get('ram_free_gb'):.2f} GB  "
        f"used={stats.get('ram_used_pct'):.1f}%  "
        f"cpu={stats.get('cpu_pct'):.1f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv)))
