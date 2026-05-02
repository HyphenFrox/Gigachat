"""Diagnostic: trigger a pair-notify push and report the result.

Useful when the symmetric pair-notify silently fails. Reads the
target peer from the FIRST entry in paired_devices on this box, then
calls push_pair_notify and prints what came back.

Run via the project's .venv\\Scripts\\python.exe so backend imports
work, e.g.:

    & .\\.venv\\Scripts\\python.exe scripts/probe_pair_notify.py
"""
import asyncio
import sys
import os

# Bootstrap path so `from backend import ...` works when run from
# the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import db, p2p_lan_client


def main() -> int:
    # Initialise the DB so list_paired_devices works.
    try:
        db.init()
    except Exception:
        pass

    paired = db.list_paired_devices()
    if not paired:
        print("no paired devices on this box — pair one first")
        return 1

    target = paired[0]
    print(f"target: device_id={target['device_id']} label={target['label']!r}")
    print(f"        ip={target.get('ip')!r} port={target.get('port')!r}")
    print(
        f"        x25519={(target.get('x25519_public_b64') or '')[:20]}…"
    )

    if not target.get("ip"):
        print("FAIL: stored ip is null — cannot push pair-notify")
        return 1

    res = asyncio.run(
        p2p_lan_client.push_pair_notify(
            peer_ip=target["ip"],
            peer_port=int(target.get("port") or 8000),
            peer_device_id=target["device_id"],
            peer_x25519_public_b64=target.get("x25519_public_b64"),
        )
    )
    print(f"push_pair_notify returned: {res}")
    return 0 if res else 1


if __name__ == "__main__":
    sys.exit(main())
