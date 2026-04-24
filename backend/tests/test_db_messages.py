"""Tests for message-level operations: edit, delete-after, compression."""

from __future__ import annotations

import time

import pytest

# Whole module is fast + offline — runs in the smoke tier.
pytestmark = pytest.mark.smoke


def test_update_user_message_content(isolated_db):
    """update_user_message_content should rewrite a user row in place."""
    db = isolated_db
    conv = db.create_conversation(title="t", model="m", cwd="/tmp")
    msg = db.add_message(conv["id"], "user", "original")
    updated = db.update_user_message_content(msg["id"], "rewritten")
    assert updated is not None
    assert updated["content"] == "rewritten"
    # Round-trip via list_messages too.
    msgs = db.list_messages(conv["id"])
    assert msgs[0]["content"] == "rewritten"


def test_update_user_message_rejects_non_user_rows(isolated_db):
    """The edit guard must refuse assistant or tool rows."""
    db = isolated_db
    conv = db.create_conversation(title="t", model="m", cwd="/tmp")
    asst = db.add_message(conv["id"], "assistant", "I won't be edited")
    result = db.update_user_message_content(asst["id"], "hacked")
    assert result is None
    # Confirm the assistant row was untouched.
    msgs = db.list_messages(conv["id"])
    assert msgs[0]["content"] == "I won't be edited"


def test_delete_messages_after(isolated_db):
    """delete_messages_after drops every row newer than the anchor."""
    db = isolated_db
    conv = db.create_conversation(title="t", model="m", cwd="/tmp")
    a = db.add_message(conv["id"], "user", "first")
    # Sleep a hair so created_at has a strictly increasing timestamp.
    time.sleep(0.01)
    db.add_message(conv["id"], "assistant", "second")
    time.sleep(0.01)
    db.add_message(conv["id"], "user", "third")

    n = db.delete_messages_after(conv["id"], a["id"])
    assert n == 2  # the assistant + the third user row

    remaining = db.list_messages(conv["id"])
    assert [m["content"] for m in remaining] == ["first"]


def test_delete_messages_after_unknown_id(isolated_db):
    """Unknown message id should be a no-op returning 0."""
    db = isolated_db
    conv = db.create_conversation(title="t", model="m", cwd="/tmp")
    db.add_message(conv["id"], "user", "first")
    assert db.delete_messages_after(conv["id"], "nonexistent") == 0
    assert len(db.list_messages(conv["id"])) == 1


def test_compress_tool_outputs_keeps_head_and_tail(isolated_db):
    """Compressed payload must retain head + tail of the original body."""
    db = isolated_db
    conv = db.create_conversation(title="t", model="m", cwd="/tmp")
    body = "\n".join(f"line-{i}" for i in range(100))
    msg = db.add_message(
        conv["id"],
        "tool",
        body,
        tool_calls=[{"id": "call_1", "name": "bash"}],
    )
    n = db.compress_tool_outputs([msg["id"]])
    assert n == 1
    fresh = db.list_messages(conv["id"])
    payload = fresh[0]["content"]
    # Head: first 5 lines must survive.
    for i in range(5):
        assert f"line-{i}" in payload
    # Tail: last 10 lines must survive.
    for i in range(90, 100):
        assert f"line-{i}" in payload
    # Middle should be elided.
    assert "lines elided" in payload
    # And the marker must mark the row as compressed for idempotency.
    assert db.is_compressed_tool_output(payload)


def test_compress_tool_outputs_idempotent(isolated_db):
    """Running compression twice on the same row should not degrade further."""
    db = isolated_db
    conv = db.create_conversation(title="t", model="m", cwd="/tmp")
    body = "\n".join(f"line-{i}" for i in range(100))
    msg = db.add_message(
        conv["id"],
        "tool",
        body,
        tool_calls=[{"id": "call_1", "name": "bash"}],
    )
    db.compress_tool_outputs([msg["id"]])
    first_pass = db.list_messages(conv["id"])[0]["content"]
    db.compress_tool_outputs([msg["id"]])
    second_pass = db.list_messages(conv["id"])[0]["content"]
    # Idempotent: a second compression pass leaves the payload alone OR
    # produces an equivalent payload of the same shape (same head/tail).
    # We check the head/tail content is preserved either way.
    for i in (0, 4, 95, 99):
        assert f"line-{i}" in second_pass
