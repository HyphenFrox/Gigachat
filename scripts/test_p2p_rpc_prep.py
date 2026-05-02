"""Diagnostic: drive ensure_rpc_server_via_proxy() against every paired
peer and report whether each peer's rpc-server is listening after.

Used to verify the new P2P-based prep path: no SSH involvement, just
the encrypted secure-proxy carrying the start request to the peer
who then spawns rpc-server in its own session.

Run via the project's .venv on the orchestrator:

    & .\\.venv\\Scripts\\python.exe scripts/test_p2p_rpc_prep.py
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import db, compute_pool as _pool  # noqa: E402


async def main() -> int:
    db.init()
    workers = [w for w in db.list_compute_workers() if w.get("enabled")]
    if not workers:
        print("no enabled workers")
        return 1
    for w in workers:
        label = w.get("label")
        print(f"-- {label} ({w.get('address')}:{w.get('ollama_port')}) --")
        t0 = time.time()
        ok = await _pool.ensure_rpc_server_via_proxy(w)
        elapsed = time.time() - t0
        print(f"   ensure_rpc_server_via_proxy -> {ok}  ({elapsed:.1f}s)")
        # Re-read the worker capabilities from DB to see what got stamped.
        fresh = db.get_compute_worker(w["id"])
        if fresh:
            caps = fresh.get("capabilities") or {}
            print(
                f"   capabilities.rpc_server_reachable = "
                f"{caps.get('rpc_server_reachable')}"
            )
            print(
                f"   capabilities.current_rpc_backend  = "
                f"{caps.get('current_rpc_backend')!r}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
