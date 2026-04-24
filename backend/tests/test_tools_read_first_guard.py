"""Read-first guard: edit_file / write_file must be preceded by read_file.

The guard exists to prevent a common LLM failure mode — the model
hallucinating the contents of a file it never actually read, then
emitting an `edit_file` whose `old_string` matches training-data priors
instead of the real file. With the guard in place, an unread-file edit
returns `{ok: False}` with a clear hint to call `read_file` first.

We exercise:
  * `edit_file` on an unread file → refused.
  * `edit_file` after `read_file` on the same conv → allowed.
  * `edit_file` without conv_id (tests / CLI) → allowed (guard is a
    no-op without a session to bind to).
  * `write_file` overwriting an unread existing file → refused.
  * `write_file` creating a NEW file → allowed (nothing to have read).
  * Per-conversation isolation: a read in conv A doesn't unlock the
    same path in conv B.
  * `clear_read_state_for_conversation` drops the tracker as expected
    (called from the delete-conversation endpoint).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from backend import tools

pytestmark = pytest.mark.smoke


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _wipe_read_state():
    """Each test starts with an empty tracker — other tests can't leak in."""
    tools._READ_FILES_BY_CONV.clear()
    yield
    tools._READ_FILES_BY_CONV.clear()


# --- edit_file -----------------------------------------------------------


def test_edit_file_refused_without_prior_read(tmp_path: Path):
    target = tmp_path / "code.py"
    target.write_text("print('hi')\n", encoding="utf-8")

    res = _run(
        tools.edit_file(
            str(tmp_path),
            "code.py",
            "hi",
            "bye",
            conv_id="conv-A",
        )
    )

    assert res["ok"] is False
    err = (res.get("error") or "").lower()
    assert "has not been read" in err
    # Error must name `read_file` so the model knows the remediation.
    assert "read_file" in (res.get("error") or "")


def test_edit_file_allowed_after_read(tmp_path: Path):
    target = tmp_path / "code.py"
    target.write_text("print('hi')\n", encoding="utf-8")

    r = _run(tools.read_file(str(tmp_path), "code.py", conv_id="conv-A"))
    assert r["ok"] is True

    res = _run(
        tools.edit_file(
            str(tmp_path),
            "code.py",
            "hi",
            "bye",
            conv_id="conv-A",
        )
    )
    assert res["ok"] is True, res
    assert target.read_text(encoding="utf-8") == "print('bye')\n"


def test_edit_file_skipped_without_conv_id(tmp_path: Path):
    """Tests / CLI callers pass conv_id=None and bypass the guard."""
    target = tmp_path / "code.py"
    target.write_text("print('hi')\n", encoding="utf-8")

    res = _run(
        tools.edit_file(
            str(tmp_path),
            "code.py",
            "hi",
            "bye",
            conv_id=None,
        )
    )
    assert res["ok"] is True, res


# --- write_file ----------------------------------------------------------


def test_write_file_refused_when_overwriting_unread(tmp_path: Path):
    target = tmp_path / "config.json"
    target.write_text("{}", encoding="utf-8")

    res = _run(
        tools.write_file(
            str(tmp_path),
            "config.json",
            '{"changed": true}',
            conv_id="conv-A",
        )
    )
    assert res["ok"] is False
    assert "has not been read" in (res.get("error") or "").lower()
    # Untouched file — the refusal is total.
    assert target.read_text(encoding="utf-8") == "{}"


def test_write_file_creates_new_without_read(tmp_path: Path):
    """Brand-new file: no prior contents to have read. Creation is fine."""
    target = tmp_path / "fresh.txt"
    assert not target.exists()

    res = _run(
        tools.write_file(
            str(tmp_path),
            "fresh.txt",
            "hello\n",
            conv_id="conv-A",
        )
    )
    assert res["ok"] is True, res
    assert target.read_text(encoding="utf-8") == "hello\n"


def test_write_file_then_edit_without_re_read(tmp_path: Path):
    """Post-write, the path is considered read — a follow-up edit works."""
    target = tmp_path / "new.py"
    wr = _run(
        tools.write_file(
            str(tmp_path),
            "new.py",
            "x = 1\n",
            conv_id="conv-A",
        )
    )
    assert wr["ok"] is True

    ed = _run(
        tools.edit_file(
            str(tmp_path),
            "new.py",
            "x = 1",
            "x = 2",
            conv_id="conv-A",
        )
    )
    assert ed["ok"] is True, ed


# --- conversation isolation ---------------------------------------------


def test_read_in_conv_a_does_not_unlock_conv_b(tmp_path: Path):
    target = tmp_path / "shared.txt"
    target.write_text("alpha\n", encoding="utf-8")

    _run(tools.read_file(str(tmp_path), "shared.txt", conv_id="conv-A"))

    res = _run(
        tools.edit_file(
            str(tmp_path),
            "shared.txt",
            "alpha",
            "beta",
            conv_id="conv-B",
        )
    )
    assert res["ok"] is False
    assert "has not been read" in (res.get("error") or "").lower()


def test_clear_read_state_drops_tracking(tmp_path: Path):
    target = tmp_path / "a.txt"
    target.write_text("one\n", encoding="utf-8")

    _run(tools.read_file(str(tmp_path), "a.txt", conv_id="conv-X"))
    # Before clearing: edit is allowed.
    ok = _run(
        tools.edit_file(
            str(tmp_path),
            "a.txt",
            "one",
            "two",
            conv_id="conv-X",
        )
    )
    assert ok["ok"] is True

    tools.clear_read_state_for_conversation("conv-X")

    # After clearing: edit needs a fresh read.
    refused = _run(
        tools.edit_file(
            str(tmp_path),
            "a.txt",
            "two",
            "three",
            conv_id="conv-X",
        )
    )
    assert refused["ok"] is False
    assert "has not been read" in (refused.get("error") or "").lower()


def test_dispatch_threads_conv_id_through_read_file(tmp_path: Path):
    """End-to-end via the real dispatcher: read via dispatch, then edit
    via dispatch — the second call must succeed."""
    target = tmp_path / "d.txt"
    target.write_text("hello\n", encoding="utf-8")

    async def run():
        r = await tools.dispatch(
            "read_file",
            {"path": "d.txt"},
            cwd=str(tmp_path),
            conv_id="conv-D",
        )
        assert r["ok"] is True, r
        e = await tools.dispatch(
            "edit_file",
            {"path": "d.txt", "old_string": "hello", "new_string": "world"},
            cwd=str(tmp_path),
            conv_id="conv-D",
        )
        return e

    res = _run(run())
    assert res["ok"] is True, res
    assert target.read_text(encoding="utf-8") == "world\n"
