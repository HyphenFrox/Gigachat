"""Send a chat completion to a running split llama-server and print
the response. Used to verify the split is actually answering tokens."""
import asyncio
import json
import os
import sys
import time

import httpx


async def main(argv: list[str]) -> int:
    port = int(argv[1]) if len(argv) > 1 else 11500
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    body = {
        "model": "llama3.1:8b",
        "messages": [
            {"role": "user", "content": "Reply with exactly: SPLIT WORKS"},
        ],
        "max_tokens": 40,
        "stream": False,
    }
    print(f"POST {url}")
    t0 = time.time()
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(url, json=body)
    elapsed = time.time() - t0
    print(f"HTTP {r.status_code} in {elapsed:.1f}s")
    try:
        data = r.json()
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        print(f"reply: {choice!r}")
        print(f"usage: {usage}")
    except Exception as e:
        print(f"parse error: {e}")
        print(r.text[:1000])
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv)))
