"""Tests for disk retention (checkpoints + memory files).

Three behaviors to pin down:
  * `trim_conv_checkpoints` keeps the newest N stamps for a conversation
    and deletes the rest.
  * `sweep` removes stamps older than the age cutoff AND removes
    conv-dirs / memory files whose conversation id is no longer in the
    DB.
  * `sweep` refuses to delete anything when the DB returns an empty set
    (a failed query must not wipe every file on disk).

The tests use `tmp_path` for isolation and monkey-patch
`db.list_conversations` so we don't need the full isolated_db fixture
(and so we can simulate the "DB unavailable" edge case).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from backend import retention

pytestmark = pytest.mark.smoke


# --- Helpers ---------------------------------------------------------------


def _make_stamp(conv_dir: Path, name: str, mtime: float | None = None) -> Path:
    """Create a fake stamp directory with one content file.

    Mimics the layout `_checkpoint_file` produces so retention sees
    something realistic. Accepts `mtime` to backdate a stamp for the
    age-based sweep test.
    """
    d = conv_dir / name
    d.mkdir(parents=True)
    (d / "data.bin").write_bytes(b"x")
    if mtime is not None:
        os.utime(d, (mtime, mtime))
        os.utime(d / "data.bin", (mtime, mtime))
    return d


@pytest.fixture()
def fake_db(monkeypatch):
    """Stub `db.list_conversations` so the sweep sees a specific id set.

    Each test calls `fake_db([...])` with the conv-ids the DB should
    report; everything else on disk becomes an orphan.
    """
    def _set(ids: list[str] | None):
        def _list():
            if ids is None:
                # Simulate query failure — sweep must NOT wipe everything.
                raise RuntimeError("db unavailable")
            return [{"id": i} for i in ids]
        from backend import db as _db
        monkeypatch.setattr(_db, "list_conversations", _list)
    return _set


# --- trim_conv_checkpoints -------------------------------------------------


def test_trim_keeps_most_recent_n(tmp_path):
    """60 stamps → trim to 50 → 10 deleted, newest 50 survive."""
    cp = tmp_path / "checkpoints"
    conv = cp / "A"
    for i in range(60):
        _make_stamp(conv, f"20260425T{i:06d}_000_aaaa")
    deleted = retention.trim_conv_checkpoints(cp, "A", keep=50)
    assert deleted == 10
    survivors = sorted(p.name for p in conv.iterdir())
    assert len(survivors) == 50
    # The 10 oldest (lex-min) are gone.
    assert survivors[0] == "20260425T000010_000_aaaa"


def test_trim_is_noop_below_threshold(tmp_path):
    """If there are fewer stamps than the keep count, nothing is deleted."""
    cp = tmp_path / "checkpoints"
    conv = cp / "A"
    for i in range(5):
        _make_stamp(conv, f"stamp-{i}")
    deleted = retention.trim_conv_checkpoints(cp, "A", keep=50)
    assert deleted == 0
    assert len(list(conv.iterdir())) == 5


def test_trim_handles_missing_conv_dir(tmp_path):
    """A non-existent conv dir is a no-op, not an error."""
    cp = tmp_path / "checkpoints"
    cp.mkdir()
    assert retention.trim_conv_checkpoints(cp, "ghost", keep=50) == 0


def test_trim_without_conv_id_is_noop(tmp_path):
    """conv_id=None disables trim (matches `_checkpoint_file`'s contract)."""
    cp = tmp_path / "checkpoints"
    cp.mkdir()
    assert retention.trim_conv_checkpoints(cp, None) == 0


# --- sweep -----------------------------------------------------------------


def test_sweep_deletes_old_stamps(tmp_path, fake_db):
    """Stamps older than the age cutoff disappear; fresh ones stay."""
    fake_db(["A"])
    cp = tmp_path / "checkpoints"
    mem = tmp_path / "memory"
    mem.mkdir()
    conv = cp / "A"
    # Fresh stamp (now) and ancient stamp (40 days ago — past 30 day cap).
    fresh = _make_stamp(conv, "fresh")
    ancient = _make_stamp(conv, "ancient", mtime=time.time() - 40 * 86400)

    counts = retention.sweep(cp, mem)
    assert counts["old_stamps"] == 1
    assert fresh.exists()
    assert not ancient.exists()


def test_sweep_removes_orphan_conv_dirs(tmp_path, fake_db):
    """Conv dirs whose id is not in the DB set are removed wholesale."""
    fake_db(["A"])  # only A is real
    cp = tmp_path / "checkpoints"
    (cp / "A").mkdir(parents=True)
    (cp / "ghost").mkdir(parents=True)
    _make_stamp(cp / "ghost", "stamp")
    mem = tmp_path / "memory"
    mem.mkdir()

    counts = retention.sweep(cp, mem)
    assert counts["orphan_conv_dirs"] == 1
    assert (cp / "A").exists()
    assert not (cp / "ghost").exists()


def test_sweep_removes_orphan_memory_files(tmp_path, fake_db):
    """Memory files for deleted conversations are unlinked."""
    fake_db(["A"])
    cp = tmp_path / "checkpoints"
    cp.mkdir()
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "A.md").write_text("real conv")
    (mem / "ghost.md").write_text("orphan")
    (mem / "not-a-markdown.txt").write_text("ignored")  # non-.md stays

    counts = retention.sweep(cp, mem)
    assert counts["orphan_memory_files"] == 1
    assert (mem / "A.md").exists()
    assert not (mem / "ghost.md").exists()
    assert (mem / "not-a-markdown.txt").exists()


def test_sweep_refuses_when_db_query_fails(tmp_path, fake_db):
    """DB error → empty known-set → orphan deletion is suppressed.

    This is the critical safety property: a failed `list_conversations`
    must not cause us to delete every checkpoint on disk.
    """
    fake_db(None)  # simulates DB failure
    cp = tmp_path / "checkpoints"
    (cp / "anything").mkdir(parents=True)
    _make_stamp(cp / "anything", "stamp")
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "X.md").write_text("keep me")

    counts = retention.sweep(cp, mem)
    # No orphans deleted even though the known-set is empty.
    assert counts["orphan_conv_dirs"] == 0
    assert counts["orphan_memory_files"] == 0
    assert (cp / "anything" / "stamp").exists()
    assert (mem / "X.md").exists()
