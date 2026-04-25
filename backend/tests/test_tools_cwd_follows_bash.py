"""Regression tests: file tools must follow wherever bash `cd`-ed to.

Scenario that motivated the fix: user asks "scaffold a React app", model runs
`npm create vite@latest myapp …` then `cd myapp && npm install`, then
`write_file({"path": "src/App.jsx", …})`. Under the old behaviour, that
last call wrote to `<workspace>/src/App.jsx` instead of
`<workspace>/myapp/src/App.jsx` because `_resolve` hard-coded the
conversation's original cwd. That diverged the model's mental model ("I
cd'd, so relative paths follow") from the tool's actual behaviour, and
every subsequent file operation landed at the wrong place.

Fix: `_resolve(cwd, path, conv_id)` now consults the persisted bash_cwd for
relative paths. These tests assert that invariant directly — touching the
DB-backed bash_cwd state and verifying each file tool lands where a shell
user would expect.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from backend import db, tools

pytestmark = pytest.mark.smoke


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def workspace(isolated_db, tmp_path):
    """Create a workspace with a `sub/` subdir pre-populated with one file.

    Returns (cwd, conv_id). The conversation is registered in the DB so
    `db.get_bash_cwd` / `db.set_bash_cwd` work end-to-end.
    """
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "note.txt").write_text("hello from sub\n", encoding="utf-8")
    conv = isolated_db.create_conversation(
        title="cwd-follow", model="test", cwd=str(tmp_path)
    )
    # Simulate the model having run `cd sub` earlier — the bash layer would
    # have persisted this via `_persist_new_cwd`.
    isolated_db.set_bash_cwd(conv["id"], str(tmp_path / "sub"))
    return str(tmp_path), conv["id"]


def test_resolve_uses_bash_cwd_for_relative_paths(workspace):
    """_resolve with conv_id set routes relative paths through bash_cwd."""
    cwd, cid = workspace
    # With conv_id: should resolve inside sub/
    resolved = tools._resolve(cwd, "note.txt", cid)
    assert resolved.name == "note.txt"
    assert resolved.parent.name == "sub"


def test_resolve_ignores_bash_cwd_without_conv_id(workspace):
    """_resolve without conv_id keeps the old "relative to cwd" behaviour."""
    cwd, _ = workspace
    resolved = tools._resolve(cwd, "note.txt")
    # Without conv_id: resolves at workspace root — the file does not exist
    # there, but that's the old behaviour we're preserving.
    assert resolved.parent == Path(cwd).resolve()


def test_resolve_absolute_path_overrides_bash_cwd(workspace, tmp_path):
    """An absolute path always wins, regardless of bash_cwd."""
    cwd, cid = workspace
    absolute = tmp_path / "other.txt"
    resolved = tools._resolve(cwd, str(absolute), cid)
    assert resolved == absolute.resolve()


def test_read_file_follows_bash_cwd(workspace):
    cwd, cid = workspace
    res = _run(tools.read_file(cwd, "note.txt", conv_id=cid))
    assert res["ok"] is True, res
    assert "hello from sub" in res["output"]


def test_write_file_follows_bash_cwd(workspace, tmp_path):
    cwd, cid = workspace
    res = _run(
        tools.write_file(cwd, "fresh.txt", "wrote inside sub\n", conv_id=cid)
    )
    assert res["ok"] is True, res
    # The file must land inside sub/, not at the workspace root.
    assert (tmp_path / "sub" / "fresh.txt").read_text(encoding="utf-8") == (
        "wrote inside sub\n"
    )
    assert not (tmp_path / "fresh.txt").exists(), (
        "write_file ignored bash_cwd and wrote to the workspace root"
    )


def test_list_dir_follows_bash_cwd(workspace):
    cwd, cid = workspace
    res = _run(tools.list_dir(cwd, ".", conv_id=cid))
    assert res["ok"] is True, res
    # Pre-populated note.txt lives in sub/. A list_dir(".") must see it.
    assert "note.txt" in res["output"]


def test_edit_file_follows_bash_cwd(workspace, tmp_path):
    cwd, cid = workspace
    # Prime the read-first guard with a read (which also goes through the
    # same bash_cwd-aware _resolve).
    _run(tools.read_file(cwd, "note.txt", conv_id=cid))
    res = _run(
        tools.edit_file(
            cwd, "note.txt", "hello from sub", "edited in place",
            conv_id=cid,
        )
    )
    assert res["ok"] is True, res
    assert (tmp_path / "sub" / "note.txt").read_text(encoding="utf-8") == (
        "edited in place\n"
    )


def test_dispatch_threads_bash_cwd_through_write_file(workspace, tmp_path):
    """End-to-end via dispatch: a bare `write_file` call must land in sub/."""
    cwd, cid = workspace
    res = _run(
        tools.dispatch(
            "write_file",
            {"path": "dispatched.txt", "content": "via dispatch"},
            cwd=cwd,
            conv_id=cid,
        )
    )
    assert res["ok"] is True, res
    assert (tmp_path / "sub" / "dispatched.txt").exists(), (
        "dispatch didn't pass conv_id through to _resolve — "
        "file landed at the workspace root"
    )


def test_bash_cwd_fallback_when_directory_deleted(workspace, tmp_path):
    """If bash_cwd points at a deleted dir, _resolve falls back to cwd.

    We previously had the same fallback in `_effective_bash_cwd` for the
    bash tool; extending it via `_resolve` means file tools get the same
    safety net instead of raising ENOENT on every call.
    """
    cwd, cid = workspace
    import shutil
    shutil.rmtree(tmp_path / "sub")
    # With the subdir gone, relative resolves fall back to the workspace root.
    resolved = tools._resolve(cwd, "fresh.txt", cid)
    assert resolved.parent == Path(cwd).resolve()
