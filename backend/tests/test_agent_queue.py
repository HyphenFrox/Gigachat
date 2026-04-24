"""Tests for the DB-backed user-input queue used by 'send while busy'.

The queue used to be an in-memory dict inside `agent` (`_QUEUED_INPUT`); it's
now persisted in the `queued_inputs` SQLite table so messages survive a
server crash. These tests exercise the same public surface
(`enqueue_user_input` / `_drain_queued_input`) and assert state via
`db.has_queued_inputs` instead of reaching into module-level dicts.

Each test creates a real conversation first because `queued_inputs` has a
FOREIGN KEY onto `conversations(id)` — the FK is what guarantees that stale
queue rows get cleaned up when a conversation is deleted.
"""

from __future__ import annotations

import pytest

# Whole module is fast + offline — runs in the smoke tier.
pytestmark = pytest.mark.smoke


def _new_conv(agent, title: str = "t") -> str:
    """Create a conversation through the db module and return its id."""
    return agent.db.create_conversation(title, "test-model", "/tmp")["id"]


def test_enqueue_basic(fresh_agent_queue):
    """Enqueuing some text should land in the per-conversation FIFO."""
    agent = fresh_agent_queue
    cid = _new_conv(agent)
    assert agent.enqueue_user_input(cid, "hello") is True
    assert agent.db.has_queued_inputs(cid) is True
    # Draining surfaces the exact payload we enqueued.
    drained = agent._drain_queued_input(cid)
    assert drained == [{"text": "hello", "images": None}]


def test_enqueue_strips_whitespace(fresh_agent_queue):
    """Whitespace-only / empty payloads with no images are rejected."""
    agent = fresh_agent_queue
    cid = _new_conv(agent)
    assert agent.enqueue_user_input(cid, "   ") is False
    assert agent.enqueue_user_input(cid, "") is False
    assert agent.enqueue_user_input(cid, None) is False
    assert agent.db.has_queued_inputs(cid) is False


def test_enqueue_with_images_only(fresh_agent_queue):
    """An image attachment alone is enough to queue (no text required)."""
    agent = fresh_agent_queue
    cid = _new_conv(agent)
    assert agent.enqueue_user_input(cid, "", images=["abc.png"]) is True
    drained = agent._drain_queued_input(cid)
    assert drained == [{"text": "", "images": ["abc.png"]}]


def test_drain_returns_fifo_then_clears(fresh_agent_queue):
    """drain returns everything in FIFO order then empties the slot."""
    agent = fresh_agent_queue
    cid = _new_conv(agent)
    agent.enqueue_user_input(cid, "first")
    agent.enqueue_user_input(cid, "second")
    drained = agent._drain_queued_input(cid)
    assert [d["text"] for d in drained] == ["first", "second"]
    # Subsequent drain returns [] — no more items, no exception.
    assert agent._drain_queued_input(cid) == []
    assert agent.db.has_queued_inputs(cid) is False


def test_drain_independent_per_conversation(fresh_agent_queue):
    """Queues are keyed by conversation_id; one drain doesn't touch others."""
    agent = fresh_agent_queue
    c1 = _new_conv(agent, "c1")
    c2 = _new_conv(agent, "c2")
    agent.enqueue_user_input(c1, "from c1")
    agent.enqueue_user_input(c2, "from c2")
    drained = agent._drain_queued_input(c1)
    assert [d["text"] for d in drained] == ["from c1"]
    # c2 is still untouched.
    assert agent.db.has_queued_inputs(c2) is True
    remaining = agent._drain_queued_input(c2)
    assert [d["text"] for d in remaining] == ["from c2"]
