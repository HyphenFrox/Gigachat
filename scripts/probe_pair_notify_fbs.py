"""One-off: trigger a pair-notify push from Gautam-FBS to whoever is
paired on it (the user's main laptop). Mirrors probe_pair_notify.py
but is meant to run ON Gautam-FBS to drive the symmetric record onto
the OTHER side after a partial pair.

Run on Gautam-FBS via SSH:
    & .\\.venv\\Scripts\\python.exe scripts/probe_pair_notify_fbs.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import db, p2p_lan_client  # noqa: E402


def main() -> int:
    db.init()
    paired = db.list_paired_devices()
    if not paired:
        print("no paired devices on this box")
        return 1
    target = paired[0]
    print(
        f"target: {target['device_id']} {target['label']!r} "
        f"@ {target.get('ip')}:{target.get('port')}"
    )
    if not target.get("ip"):
        print("FAIL: stored ip is null")
        return 1
    res = asyncio.run(
        p2p_lan_client.push_pair_notify(
            peer_ip=target["ip"],
            peer_port=int(target.get("port") or 8000),
            peer_device_id=target["device_id"],
            peer_x25519_public_b64=target.get("x25519_public_b64"),
        )
    )
    print(f"push_pair_notify -> {res}")
    return 0 if res else 1


if __name__ == "__main__":
    sys.exit(main())
