"""Drive POST /api/p2p/rpc-server/start against one peer via the
encrypted secure-proxy and print the verbose response."""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import db  # noqa: E402


async def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: start_rpc_on_peer.py <label_substr>")
        return 2
    needle = argv[1].lower()
    db.init()
    workers = [
        w for w in db.list_compute_workers()
        if w.get("enabled") and needle in (w.get("label") or "").lower()
    ]
    if not workers:
        print(f"no enabled worker matching {argv[1]!r}")
        return 1
    w = workers[0]
    print(f"starting rpc-server on {w['label']} via P2P")
    from backend import p2p_secure_client as _secure
    status, body = await _secure.forward(
        w, method="POST", path="/api/p2p/rpc-server/start",
        body={"backend": "CPU"},
    )
    print(f"HTTP {status}")
    try:
        print(json.dumps(json.loads(body), indent=2))
    except Exception:
        print(body[:500])
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv)))
