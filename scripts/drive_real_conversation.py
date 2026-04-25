"""Drive a single agent turn end-to-end against a real Ollama model.

Used to verify the `_resolve` / bash_cwd fix holds up on a live conversation
the way the user reported. Creates a fresh conversation pointed at a clean
workspace, issues the same prompt the user gave ("Create a react app that
displays stock prices of popular stocks"), streams events to stdout as they
arrive, and prints the resulting conversation id on completion so a follow-up
script can inspect the DB.

Run directly: `python scripts/drive_real_conversation.py`
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

# Make the package importable when run as a bare script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import agent, db


PROMPT = "Create a react app that displays stock prices of popular stocks"
MODEL = "gemma4:e4b"
CWD = r"C:\Users\gauta\Downloads\Gigachat\.claude-test"


async def main() -> None:
    conv = db.create_conversation(
        title="real-test-after-fix",
        model=MODEL,
        cwd=CWD,
        # Auto-approve every tool — no UI, no human in the loop for this
        # scripted run. Same practical effect as the user clicking "allow all"
        # in the frontend.
        auto_approve=True,
        permission_mode="allow_all",
    )
    cid = conv["id"]
    # Record the id up-front so a parallel watcher can start polling even if
    # the stream takes a while to reach completion.
    Path("scripts/.last_conv_id").write_text(cid, encoding="utf-8")
    print(f"[conv] {cid}", flush=True)

    t0 = time.time()
    event_count = 0
    async for evt in agent.run_turn(cid, user_text=PROMPT):
        event_count += 1
        etype = evt.get("type", "?")
        # Keep the console readable: print a one-line summary per event.
        if etype == "tool_call":
            name = evt.get("name")
            args = evt.get("args") or {}
            brief = json.dumps(args, ensure_ascii=False)[:120]
            print(f"[{event_count}] tool_call {name} {brief}", flush=True)
        elif etype == "tool_result":
            name = evt.get("name")
            ok = evt.get("ok")
            out = (evt.get("output") or "")[:120].replace("\n", " ")
            print(f"[{event_count}] tool_result {name} ok={ok} | {out}", flush=True)
        elif etype == "error":
            print(f"[{event_count}] ERROR: {evt.get('message')}", flush=True)
        elif etype == "message_delta":
            # Too chatty — skip streaming token deltas.
            pass
        else:
            print(f"[{event_count}] {etype}", flush=True)

    dt = time.time() - t0
    print(f"[done] conv={cid} events={event_count} elapsed={dt:.1f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
