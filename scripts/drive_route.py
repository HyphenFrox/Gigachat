"""Drive compute_pool.route_chat_for(model) on the LIVE backend so the
spawned llama-server lifecycle is owned by the FastAPI process (not
the test script). Use this for end-to-end split verification when
test_route_decision.py would orphan the spawn on script timeout.

Calls the loopback endpoint that runs route_chat_for inside the
backend process.
"""
import asyncio
import json
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: drive_route.py <model>")
        return 2
    model = argv[1]
    # Use the chat-bench endpoint if it exists; otherwise call
    # route_chat_for via a direct dispatch through the agent path.
    # We DELIBERATELY do NOT use a generative path so we don't burn
    # tokens — we only want to know whether routing engaged split.
    base = "http://127.0.0.1:8000"
    print(f"resolving + routing {model!r} via live backend at {base}")
    t0 = time.time()
    async with httpx.AsyncClient(timeout=180.0) as client:
        # Direct route-only endpoint doesn't exist, so we use the
        # split-models list before/after to detect a row creation.
        before = await client.get(f"{base}/api/split-models")
        before_rows = before.json().get("rows") or before.json()
        before_ids = {r.get("id") for r in (before_rows if isinstance(before_rows, list) else [])}
        # Trigger by hitting model-router via the public route helper.
        # We use POST /api/compute-pool/route which doesn't exist —
        # fall back to creating a real conversation message.
        # Simpler: just call route_chat_for via the test endpoint we'll add
        r = await client.post(
            f"{base}/api/_internal/route-test",
            json={"model": model},
        )
        if r.status_code == 404:
            print("internal endpoint missing; this script needs /api/_internal/route-test")
            return 1
        elapsed = time.time() - t0
        print(f"HTTP {r.status_code} in {elapsed:.1f}s")
        try:
            print(json.dumps(r.json(), indent=2, default=str))
        except Exception:
            print(r.text[:500])
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv)))
