"""Tests for the global_memories CRUD layer.

Global memories are SQLite-backed notes injected into every conversation's
system prompt. The CRUD covered here also powers the REST API
(`/api/memories`) and the `remember(scope="global")` / `forget(scope=...)`
agent tools, so we want strong coverage of edge cases:

  - blank/whitespace input is rejected (would otherwise pollute the table)
  - oversize content is truncated, never rejected (don't lose data quietly)
  - case-insensitive substring delete (matches the agent's mental model)
  - id mismatches return None / 0 instead of raising (HTTP layer maps to 404)
"""

from __future__ import annotations

import pytest

# Whole module is fast + offline — runs in the smoke tier.
pytestmark = pytest.mark.smoke


def test_add_and_list_global_memory(isolated_db):
    """A new memory should round-trip every field and appear in the list."""
    db = isolated_db
    row = db.add_global_memory("user prefers SCSS", topic="preferences")
    assert row["content"] == "user prefers SCSS"
    assert row["topic"] == "preferences"
    assert row["id"]
    assert row["created_at"] == row["updated_at"]

    rows = db.list_global_memories()
    assert len(rows) == 1
    assert rows[0]["id"] == row["id"]


def test_add_global_memory_strips_and_normalizes(isolated_db):
    """Content/topic are trimmed; empty topic becomes None."""
    db = isolated_db
    row = db.add_global_memory("   surrounded by spaces   ", topic="   ")
    assert row["content"] == "surrounded by spaces"
    assert row["topic"] is None


def test_add_global_memory_rejects_blank(isolated_db):
    """Empty / whitespace-only content must raise — protects against typos."""
    db = isolated_db
    with pytest.raises(ValueError):
        db.add_global_memory("")
    with pytest.raises(ValueError):
        db.add_global_memory("   ")


def test_add_global_memory_truncates_oversize(isolated_db):
    """Very long content is truncated to the cap, not rejected."""
    db = isolated_db
    big = "x" * (db.GLOBAL_MEMORY_CONTENT_MAX + 500)
    row = db.add_global_memory(big)
    assert len(row["content"]) == db.GLOBAL_MEMORY_CONTENT_MAX


def test_list_global_memories_oldest_first(isolated_db):
    """list_global_memories returns rows in insertion order (oldest-first)."""
    db = isolated_db
    a = db.add_global_memory("first")
    b = db.add_global_memory("second")
    c = db.add_global_memory("third")
    rows = db.list_global_memories()
    assert [r["id"] for r in rows] == [a["id"], b["id"], c["id"]]


def test_update_global_memory_partial_patch(isolated_db):
    """Passing None for a field leaves it unchanged."""
    db = isolated_db
    row = db.add_global_memory("original content", topic="origin")
    # Patch only content — topic should survive.
    updated = db.update_global_memory(row["id"], content="new content")
    assert updated["content"] == "new content"
    assert updated["topic"] == "origin"
    # Patch only topic — content should survive.
    updated2 = db.update_global_memory(row["id"], topic="other")
    assert updated2["content"] == "new content"
    assert updated2["topic"] == "other"


def test_update_global_memory_clears_topic_with_empty_string(isolated_db):
    """Passing an empty string explicitly clears the topic to NULL."""
    db = isolated_db
    row = db.add_global_memory("hello", topic="something")
    updated = db.update_global_memory(row["id"], topic="")
    assert updated["topic"] is None


def test_update_global_memory_blank_content_raises(isolated_db):
    """Updating content to blank is treated like add — refuse it."""
    db = isolated_db
    row = db.add_global_memory("hello")
    with pytest.raises(ValueError):
        db.update_global_memory(row["id"], content="   ")


def test_update_global_memory_missing_id_returns_none(isolated_db):
    """Patching a non-existent id should return None, not raise."""
    db = isolated_db
    assert db.update_global_memory("does-not-exist", content="x") is None


def test_delete_global_memory(isolated_db):
    """delete_global_memory returns 1 on success, 0 on miss."""
    db = isolated_db
    row = db.add_global_memory("doomed")
    assert db.delete_global_memory(row["id"]) == 1
    assert db.delete_global_memory(row["id"]) == 0
    assert db.list_global_memories() == []


def test_delete_global_memories_matching_case_insensitive(isolated_db):
    """Substring delete is case-insensitive and returns the count removed."""
    db = isolated_db
    db.add_global_memory("user lives in Tokyo")
    db.add_global_memory("user works on Money Maker")
    db.add_global_memory("favourite IDE is VSCode")

    n = db.delete_global_memories_matching("USER")
    assert n == 2
    remaining = [r["content"] for r in db.list_global_memories()]
    assert remaining == ["favourite IDE is VSCode"]


def test_delete_global_memories_matching_treats_like_metachars_literally(isolated_db):
    """`%` / `_` in the pattern must NOT act as SQL LIKE wildcards.

    Users (and the agent calling `forget(scope="global", pattern=...)`) think
    of the pattern as a literal substring. If the LIKE metachars leaked
    through, a pattern like `_` would match every row.
    """
    db = isolated_db
    db.add_global_memory("budget capped at 100%")
    db.add_global_memory("under_score in this one")
    db.add_global_memory("plain entry, no specials")

    # `%` should match only the literal-percent row.
    n = db.delete_global_memories_matching("100%")
    assert n == 1
    contents = sorted(r["content"] for r in db.list_global_memories())
    assert contents == ["plain entry, no specials", "under_score in this one"]

    # `_` should match only the literal-underscore row, NOT every row.
    n = db.delete_global_memories_matching("under_score")
    assert n == 1
    contents = [r["content"] for r in db.list_global_memories()]
    assert contents == ["plain entry, no specials"]


def test_delete_global_memories_matching_blank_pattern_raises(isolated_db):
    """A blank pattern must NOT wipe the entire table — raise instead."""
    db = isolated_db
    db.add_global_memory("keep me")
    with pytest.raises(ValueError):
        db.delete_global_memories_matching("")
    with pytest.raises(ValueError):
        db.delete_global_memories_matching("   ")
    # Sanity check: nothing was deleted.
    assert len(db.list_global_memories()) == 1


def test_get_global_memory_missing_returns_none(isolated_db):
    """get_global_memory must return None (not raise) for unknown ids."""
    db = isolated_db
    assert db.get_global_memory("nope") is None
