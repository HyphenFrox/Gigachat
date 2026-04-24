"""Subagent progress bus: lets subagents (run inside a `delegate` tool)
publish progress events onto a per-parent-conversation in-process queue so
the parent turn can forward them onto its SSE stream in real time.

The bus has three contracts:
  * Publishes keyed to an UNREGISTERED parent conv_id are dropped silently
    — this matters for tests, CLI callers, and subagent runs that happen
    outside of a live turn.
  * Publishes keyed to a registered conv_id land in the right queue and
    nowhere else (no cross-conversation leakage).
  * `put_nowait` never raises even if we hit the unlikely QueueFull path —
    losing a progress chip is strictly better than wedging a subagent.

We exercise the low-level publish/register helpers directly rather than
spinning up a real subagent, because the subagent path requires Ollama
and the bus semantics are what we care about here.
"""
from __future__ import annotations

import asyncio

import pytest

from backend import agent

pytestmark = pytest.mark.smoke


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _wipe_bus():
    """Every test starts with an empty registry so they don't leak into
    each other.
    """
    agent._SUBAGENT_PROGRESS_BUS.clear()
    yield
    agent._SUBAGENT_PROGRESS_BUS.clear()


def test_publish_without_registered_parent_is_noop():
    """No conv registered → publish is dropped silently (no exception)."""
    # Must not raise.
    agent._publish_subagent_event("never-registered", {"type": "subagent_started"})
    agent._publish_subagent_event(None, {"type": "subagent_started"})
    # Nothing was created as a side effect.
    assert agent._SUBAGENT_PROGRESS_BUS == {}


def test_register_returns_a_queue_and_is_idempotent():
    """Registering twice for the same conv returns the SAME queue — we
    don't want a race between two registrations to split events across
    two queues.
    """
    async def run():
        q1 = agent._register_subagent_bus("conv-A")
        q2 = agent._register_subagent_bus("conv-A")
        assert q1 is q2
        assert isinstance(q1, asyncio.Queue)

    _run(run())


def test_publish_lands_in_registered_queue():
    """Events published after register() are retrievable via get()."""
    async def run():
        q = agent._register_subagent_bus("conv-A")
        agent._publish_subagent_event("conv-A", {"type": "subagent_started", "subagent_id": "s1"})
        agent._publish_subagent_event("conv-A", {"type": "subagent_done", "subagent_id": "s1"})
        a = await q.get()
        b = await q.get()
        assert a["type"] == "subagent_started"
        assert b["type"] == "subagent_done"

    _run(run())


def test_publish_does_not_leak_across_parent_conversations():
    """A publish keyed to conv-A must never show up in conv-B's queue."""
    async def run():
        qa = agent._register_subagent_bus("conv-A")
        qb = agent._register_subagent_bus("conv-B")
        agent._publish_subagent_event("conv-A", {"type": "subagent_started", "subagent_id": "sA"})
        agent._publish_subagent_event("conv-B", {"type": "subagent_started", "subagent_id": "sB"})
        # conv-A sees only its own event.
        a = await qa.get()
        assert a["subagent_id"] == "sA"
        # conv-B sees only its own event.
        b = await qb.get()
        assert b["subagent_id"] == "sB"
        # Both queues are now empty.
        assert qa.empty()
        assert qb.empty()

    _run(run())


def test_unregister_drops_the_queue():
    """After unregister, a subsequent publish is a no-op."""
    async def run():
        agent._register_subagent_bus("conv-X")
        agent._unregister_subagent_bus("conv-X")
        assert "conv-X" not in agent._SUBAGENT_PROGRESS_BUS
        # Must not raise, must not recreate the queue.
        agent._publish_subagent_event("conv-X", {"type": "subagent_started"})
        assert "conv-X" not in agent._SUBAGENT_PROGRESS_BUS

    _run(run())


def test_publish_swallows_queue_full(monkeypatch):
    """Synthetic QueueFull — the publisher must NOT raise, or a subagent
    could deadlock waiting to emit a progress chip.
    """
    async def run():
        # Register a queue, then replace it with one whose put_nowait
        # always raises QueueFull. That models a saturated bus.
        class _AlwaysFull:
            def put_nowait(self, evt):
                raise asyncio.QueueFull()

        agent._SUBAGENT_PROGRESS_BUS["conv-F"] = _AlwaysFull()
        # Must not raise.
        agent._publish_subagent_event("conv-F", {"type": "subagent_started"})

    _run(run())
