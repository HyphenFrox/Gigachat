"""Diagnostic: fetch each paired peer's live /api/p2p/system-stats
through the encrypted P2P secure proxy. Used to verify the new
adaptive-watchdog data path end-to-end (wire format, whitelist,
Gigachat-internal routing, decrypt, parse).

Run via the project's .venv:

    & .\\.venv\\Scripts\\python.exe scripts/probe_peer_stats.py
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import db  # noqa: E402


async def main() -> int:
    db.init()
    # The secure-proxy fetcher takes a worker dict (compute_workers
    # row), not a paired_devices row directly — so look up the
    # worker linked to each paired peer.
    workers = db.list_compute_workers()
    if not workers:
        print("no compute_workers configured")
        return 1
    from backend import compute_pool as _pool  # local import — needs db.init first
    for w in workers:
        if not w.get("enabled"):
            continue
        label = w.get("label")
        print(f"-- {label} ({w.get('address')}:{w.get('ollama_port')}) --")
        stats = await _pool.probe_worker_live_stats(w)
        if not stats:
            print("  (no stats — peer didn't respond or proxy refused)")
            continue
        print(f"  {json.dumps(stats, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
