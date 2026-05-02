"""Diagnostic: send one encrypted-proxy GET to a paired peer and
print the FULL response body so we can see exactly why the receiver
returns 4xx/5xx.

Reads the FIRST entry in paired_devices and posts a real envelope to
its `/api/p2p/secure/forward` carrying a `GET /api/version`. Prints
the HTTP status, response headers, and body verbatim.

Run via the project's .venv:

    & .\\.venv\\Scripts\\python.exe scripts/probe_secure_forward.py
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402

from backend import db, p2p_crypto  # noqa: E402


async def main() -> int:
    db.init()
    paired = db.list_paired_devices()
    if not paired:
        print("no paired devices on this box")
        return 1
    target = paired[0]
    peer_x25519 = target.get("x25519_public_b64")
    peer_device_id = target["device_id"]
    peer_ip = target.get("ip")
    peer_port = int(target.get("port") or 8000)
    if not peer_x25519:
        print("FAIL: paired record has no x25519 key — re-pair needed")
        return 1
    if not peer_ip:
        print("FAIL: paired record has no ip")
        return 1
    print(f"target: {peer_device_id} @ {peer_ip}:{peer_port}")
    envelope = p2p_crypto.seal_json(
        recipient_x25519_pub_b64=peer_x25519,
        recipient_device_id=peer_device_id,
        payload={
            "method": "GET",
            "path": "/api/version",
            "body": None,
            "headers": {},
        },
    )
    url = f"http://{peer_ip}:{peer_port}/api/p2p/secure/forward"
    print(f"POST {url}")
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(url, json=envelope)
    print(f"status: {r.status_code}")
    print(f"body:   {r.text[:500]}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
