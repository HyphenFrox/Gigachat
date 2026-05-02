"""Resolve an Ollama model to its GGUF, create or get a split_models
row for it that uses every enabled chat-worker, and start it.
Prints the final state."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import db, compute_pool as _pool, split_lifecycle  # noqa: E402


async def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: create_split_for.py <model>")
        return 2
    model = argv[1]
    db.init()
    info = _pool.resolve_ollama_model(model)
    if not info:
        print(f"could not resolve {model!r}")
        return 1
    print(f"gguf={info['gguf_path']} size={info['size_bytes']/1e9:.2f} GB")
    workers = [
        w["id"] for w in db.list_compute_workers(enabled_only=True)
        if w.get("use_for_chat")
        and (w.get("capabilities") or {}).get("rpc_server_reachable")
    ]
    print(f"workers: {workers}")
    if not workers:
        print("no rpc-reachable workers")
        return 1
    rows = db.list_split_models()
    target_row = next((r for r in rows if r.get("label") == model), None)
    if target_row is None:
        sid = db.create_split_model(
            label=model, gguf_path=info["gguf_path"],
            mmproj_path=info.get("mmproj_path"),
            worker_ids=workers,
        )
        print(f"created split row {sid}")
    else:
        sid = target_row["id"]
        db.update_split_model(
            sid, gguf_path=info["gguf_path"],
            worker_ids=workers,
            mmproj_path=info.get("mmproj_path"),
        )
        print(f"reused split row {sid}")
    print(f"starting...")
    result = await split_lifecycle.start(sid)
    print(f"result: {result}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv)))
