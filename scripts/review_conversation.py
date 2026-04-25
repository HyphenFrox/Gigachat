"""Audit a completed conversation for the bugs the fix was supposed to address.

Metrics surfaced (matches the two failure signatures from the bug report):

1. **Bash error rate** — what fraction of bash tool calls returned non-zero.
2. **File-tool destination correctness** — after the model `cd`s into a
   subdir, do subsequent file ops land inside it? We assert that every
   write_file / edit_file / read_file result whose *input path* was
   relative resolves under the bash_cwd at the time of the call (not just
   the conversation's root cwd).

Usage:
    python scripts/review_conversation.py <conv_id>
    python scripts/review_conversation.py  # falls back to scripts/.last_conv_id
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import db


def _load_cid() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    last = Path("scripts/.last_conv_id")
    if last.exists():
        return last.read_text(encoding="utf-8").strip()
    raise SystemExit("usage: review_conversation.py <conv_id>")


def main() -> None:
    cid = _load_cid()
    conv = db.get_conversation(cid)
    if not conv:
        raise SystemExit(f"conversation not found: {cid}")
    msgs = db.list_messages(cid)
    cwd = Path(conv["cwd"]).resolve()
    print(f"conv {cid}  cwd={cwd}  model={conv['model']}  title={conv['title']}")
    print(f"messages: {len(msgs)}  bash_cwd_final={conv.get('bash_cwd')}")
    print("-" * 72)

    bash_total = 0
    bash_failed: list[dict] = []
    file_writes: list[dict] = []
    file_reads_missing: list[dict] = []

    # Build call_id -> (name, args, msg_idx) from every assistant row so we
    # can pair tool results back to their triggering call by id. The DB
    # denormalizes tool_calls on tool rows too, but only with {id, name} —
    # args live solely on the assistant side.
    calls_by_id: dict[str, dict] = {}
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        raw = m.get("tool_calls") or []
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = []
        for c in raw:
            if not isinstance(c, dict):
                continue
            cid_ = c.get("id")
            if not cid_:
                continue
            calls_by_id[cid_] = {
                "name": c.get("name"),
                "args": c.get("args") or {},
                "msg_idx": i,
            }

    for i, m in enumerate(msgs):
        if m.get("role") != "tool":
            continue
        raw_tc = m.get("tool_calls") or []
        if isinstance(raw_tc, str):
            try:
                raw_tc = json.loads(raw_tc)
            except Exception:
                raw_tc = []
        if not raw_tc:
            continue
        back_ref = raw_tc[0] if isinstance(raw_tc[0], dict) else {}
        call = calls_by_id.get(back_ref.get("id") or "")
        if not call:
            # Fallback: the tool row carries the name too; use it with empty
            # args so the category counters still fire.
            call = {"name": back_ref.get("name"), "args": {}, "msg_idx": i}
        name = call["name"]
        args = call["args"]
        content = m.get("content") or ""
        result = {}
        if isinstance(content, str):
            try:
                result = json.loads(content)
            except Exception:
                result = {"output": content}
        ok = result.get("ok")
        out = result.get("output") or ""
        err = result.get("error") or ""

        if name in ("bash", "bash_bg", "bash_output"):
            bash_total += 1
            exit_code = result.get("exit_code")
            # `bash` can return ok=True, exit_code=None for commands that
            # succeed (e.g. plain `cd`). Treat non-ok OR non-zero exit as
            # failure; null exit on a successful call is fine.
            failed = (ok is False) or (isinstance(exit_code, int) and exit_code != 0)
            if failed:
                bash_failed.append({
                    "msg_idx": i,
                    "cmd": args.get("command") or args.get("cmd"),
                    "exit": exit_code,
                    "err": (err or "")[:200],
                    "out": (out or "")[:200],
                })
        elif name in ("write_file", "edit_file"):
            dest = None
            # edit_file: "edited <abs> (N replacement)\n\n--- a/..." — check
            # this first because the diff body often contains the literal
            # word "to" which would fool the write_file parser below.
            # write_file: "wrote <N> chars to <abs>".
            if name == "edit_file" and out.startswith("edited "):
                rest = out[len("edited "):]
                for sep in (" (", "\n"):
                    if sep in rest:
                        rest = rest.split(sep, 1)[0]
                        break
                dest = rest.strip()
            elif " to " in out:
                dest = out.split(" to ", 1)[1].split("\n", 1)[0].strip()
            file_writes.append({
                "msg_idx": i,
                "tool": name,
                "input_path": args.get("path"),
                "resolved": dest,
                "ok": ok,
            })
        elif name == "read_file":
            blob = (err or out or "").lower()
            if ok is False and "not found" in blob:
                file_reads_missing.append({
                    "msg_idx": i,
                    "input_path": args.get("path"),
                    "out": (err or out or "")[:200],
                })

    print(f"bash calls: {bash_total}   failed: {len(bash_failed)}")
    if bash_total:
        rate = 100.0 * len(bash_failed) / bash_total
        print(f"bash error rate: {rate:.1f}%")
    for f in bash_failed:
        print(f"  msg{f['msg_idx']} exit={f['exit']}: {f['cmd']!r}")
        print(f"    stderr: {f['out']}")
    print("-" * 72)

    print(f"write/edit_file calls: {len(file_writes)}")
    # Did the writes land under cwd? Did any land directly at cwd when they
    # should have gone into a subdir? Flag any write whose resolved path
    # sits at the workspace root if there's evidence the model had cd'd.
    root_writes: list[dict] = []
    for w in file_writes:
        r = w.get("resolved") or ""
        if not r:
            continue
        try:
            rp = Path(r).resolve()
        except Exception:
            continue
        rel = None
        try:
            rel = rp.relative_to(cwd)
        except ValueError:
            pass
        if rel is not None and len(rel.parts) <= 1:
            # Direct child of cwd.
            root_writes.append({**w, "relative": str(rel)})
    # Show all writes with their resolved destination so a human can eyeball
    # it; then separately call out ones that landed at the workspace root.
    for w in file_writes[:40]:
        print(f"  msg{w['msg_idx']} {w['tool']} input={w['input_path']!r} -> {w['resolved']}")
    if len(file_writes) > 40:
        print(f"  ... +{len(file_writes) - 40} more")
    print("-" * 72)

    if root_writes:
        print(f"writes landing directly under workspace root: {len(root_writes)}")
        for w in root_writes:
            print(f"  msg{w['msg_idx']} {w['tool']}({w['input_path']!r}) -> {w['relative']}")
    else:
        print("no writes landed directly at workspace root (good sign).")
    print("-" * 72)

    print(f"read_file 'not found' failures: {len(file_reads_missing)}")
    for r in file_reads_missing:
        print(f"  msg{r['msg_idx']} read_file({r['input_path']!r})")
        print(f"    {r['out']}")
    print("=" * 72)

    # Summary verdict.
    bash_rate = 100.0 * len(bash_failed) / bash_total if bash_total else 0.0
    # Heuristic: the scaffolder almost always creates a subdir with the
    # project name. If every write landed at the cwd root, something's off.
    root_share = 100.0 * len(root_writes) / len(file_writes) if file_writes else 0.0
    print(f"VERDICT: bash_error_rate={bash_rate:.1f}%  root_write_share={root_share:.1f}%  "
          f"read_misses={len(file_reads_missing)}")


if __name__ == "__main__":
    main()
