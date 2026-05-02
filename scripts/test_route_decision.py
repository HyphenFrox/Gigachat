"""Drive compute_pool.route_chat_for() against a model and report
the decision: did the auto-promote router engage split, or did it
keep the chat on host Ollama?

Pass the model name as the only argument:

    & .\\.venv\\Scripts\\python.exe scripts/test_route_decision.py gemma4:26b
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import db, compute_pool as _pool  # noqa: E402


async def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: test_route_decision.py <model_name>")
        return 2
    model = argv[1]
    db.init()
    print(f"routing for model: {model!r}")
    print("--- enabled workers ---")
    for w in db.list_compute_workers():
        if not w.get("enabled"):
            continue
        caps = w.get("capabilities") or {}
        print(
            f"  {w['label']:<22s} use_for_chat={w.get('use_for_chat')} "
            f"rpc_reachable={caps.get('rpc_server_reachable')} "
            f"ram_free_gb={caps.get('ram_free_gb')}"
        )
    print()
    print("--- calling route_chat_for ---")
    t0 = time.time()
    try:
        decision = await _pool.route_chat_for(model)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return 1
    elapsed = time.time() - t0
    print(f"elapsed: {elapsed:.1f}s")
    print(f"decision: {json.dumps(decision, indent=2, default=str)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv)))
