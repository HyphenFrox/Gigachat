"""Regression: when the agent writes to a file under an indexed cwd,
the matching chunks must be evicted from the codebase index so
`codebase_search` doesn't return pre-edit content.

These tests cover the SYNCHRONOUS half of the invalidation path —
the immediate `delete_doc_chunks_for` call that runs inside
`write_file` / `edit_file`. The asynchronous re-embed half goes
through `_codebase_index_cwd_impl`'s per-file inner loop, which is
covered by the existing index tests; we don't exercise it here
because it'd need a live Ollama embeddings endpoint.
"""
from __future__ import annotations

import asyncio

import pytest

from backend import tools

pytestmark = pytest.mark.smoke


def _run(coro):
    return asyncio.run(coro)


def _seed_index(db, cwd: str, file_paths: list[str]) -> None:
    """Mark the index ready and drop a fake chunk per file so we can
    assert it's been evicted after a subsequent write."""
    db.upsert_codebase_index(cwd, status="ready", file_count=len(file_paths))
    for path in file_paths:
        db.insert_doc_chunk(
            path=path,
            ordinal=0,
            text=f"original content of {path}",
            vector=[0.0] * 8,
            model="test-embed",
        )


def test_write_file_evicts_stale_chunks(isolated_db, tmp_path):
    """write_file under an indexed cwd → that file's chunks vanish so
    codebase_search can't return pre-edit content."""
    cwd = str(tmp_path)
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir(parents=True)
    target.write_text("# original\n", encoding="utf-8")
    _seed_index(isolated_db, cwd, [str(target.resolve())])
    # Fake the read-first guard by recording a prior read.
    tools._record_read(None, target)
    # Sanity: chunk exists pre-write.
    assert any(
        r["path"] == str(target.resolve())
        for r in isolated_db.all_doc_chunks()
    )
    # Now write through the tool.
    res = _run(tools.write_file(cwd, str(target), "# new content\n"))
    assert res["ok"] is True
    # The synchronous delete should have removed the file's chunks.
    remaining = [
        r for r in isolated_db.all_doc_chunks()
        if r["path"] == str(target.resolve())
    ]
    assert remaining == []


def test_write_file_outside_cwd_does_not_touch_index(isolated_db, tmp_path):
    """A write to a file OUTSIDE the indexed cwd must not invalidate
    chunks that happen to share a basename / segment."""
    cwd = str(tmp_path)
    inside = tmp_path / "app.py"
    outside = tmp_path.parent / "other_repo_app.py"
    inside.write_text("inside\n", encoding="utf-8")
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("outside\n", encoding="utf-8")
    _seed_index(isolated_db, cwd, [str(inside.resolve())])
    tools._record_read(None, outside)
    # Write to the OUTSIDE file. We use its absolute path; the index
    # is only for `tmp_path` so this should be a no-op.
    res = _run(tools.write_file(cwd, str(outside), "x"))
    assert res["ok"] is True
    # The unrelated chunk for `inside` is still there.
    assert any(
        r["path"] == str(inside.resolve())
        for r in isolated_db.all_doc_chunks()
    )


def test_write_file_does_nothing_when_index_not_ready(isolated_db, tmp_path):
    """No invalidation if the index isn't in `ready` state — saves
    cycles when the user hasn't built the index yet."""
    cwd = str(tmp_path)
    target = tmp_path / "x.py"
    target.write_text("old\n", encoding="utf-8")
    isolated_db.upsert_codebase_index(cwd, status="pending")
    isolated_db.insert_doc_chunk(
        path=str(target.resolve()),
        ordinal=0,
        text="old",
        vector=[0.0] * 8,
        model="test-embed",
    )
    tools._record_read(None, target)
    res = _run(tools.write_file(cwd, str(target), "new\n"))
    assert res["ok"] is True
    # Pending index → no invalidation, chunk survives.
    assert any(
        r["path"] == str(target.resolve())
        for r in isolated_db.all_doc_chunks()
    )


def test_edit_file_evicts_stale_chunks(isolated_db, tmp_path):
    """edit_file goes through the same invalidation path."""
    cwd = str(tmp_path)
    target = tmp_path / "lib.py"
    target.write_text("def foo():\n    return 1\n", encoding="utf-8")
    _seed_index(isolated_db, cwd, [str(target.resolve())])
    tools._record_read(None, target)
    assert any(
        r["path"] == str(target.resolve())
        for r in isolated_db.all_doc_chunks()
    )
    res = _run(tools.edit_file(
        cwd, str(target), "return 1", "return 42",
    ))
    assert res["ok"] is True
    remaining = [
        r for r in isolated_db.all_doc_chunks()
        if r["path"] == str(target.resolve())
    ]
    assert remaining == []


def test_invalidation_does_not_raise_on_missing_index(isolated_db, tmp_path):
    """The invalidator is best-effort — calling it for a path with no
    matching index row must not raise (would break write_file)."""
    cwd = str(tmp_path)
    target = tmp_path / "fresh.py"
    target.write_text("hi\n", encoding="utf-8")
    tools._record_read(None, target)
    # No upsert_codebase_index call — there's no index for this cwd.
    res = _run(tools.write_file(cwd, str(target), "bye\n"))
    assert res["ok"] is True
