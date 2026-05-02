"""Quick: from this orchestrator, ask each paired peer's
/api/p2p/rpc-server/status (over the encrypted secure-proxy) what
their local rpc-server state is. Used to debug why
ensure_rpc_server_via_proxy returns False."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import db  # noqa: E402


async def main() -> int:
    db.init()
    workers = [w for w in db.list_compute_workers() if w.get("enabled")]
    if not workers:
        print("no workers")
        return 1
    from backend import p2p_secure_client as _secure  # noqa
    import json as _json
    for w in workers:
        label = w.get("label")
        try:
            status, body = await _secure.forward(
                w, method="GET", path="/api/p2p/rpc-server/status",
                body=None,
            )
        except Exception as e:
            print(f"{label}: ERR {type(e).__name__}: {e}")
            continue
        print(f"{label} (HTTP {status}):")
        try:
            print(f"  {_json.loads(body)}")
        except Exception:
            print(f"  {body[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
