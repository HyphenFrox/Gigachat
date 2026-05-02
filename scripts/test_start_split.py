"""Run split_lifecycle.start() directly and print any exception.
Used to extract the actual traceback when /api/split-models/{sid}/start
returns 500 with an empty body."""
import asyncio
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import db, split_lifecycle  # noqa: E402


async def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: test_start_split.py <split_id>")
        return 2
    sid = argv[1]
    db.init()
    row = db.get_split_model(sid)
    if not row:
        print(f"no split row {sid}")
        return 1
    print(f"starting {row['label']} (gguf={row['gguf_path']})")
    try:
        result = await split_lifecycle.start(sid)
        print(f"result: {result}")
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv)))
