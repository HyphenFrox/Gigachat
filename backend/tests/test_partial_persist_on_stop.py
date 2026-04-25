"""Regression: when a turn is interrupted mid-stream (Stop button or
client disconnect), whatever the model has already streamed must land
in the DB with a "[stopped mid-response]" marker — not vanish entirely.

The bug this guards against: previously the frontend's Stop button
aborted the SSE fetch, which raised CancelledError at the Ollama-chunk
await inside `_run_turn_impl`. The post-stream `db.add_message` call
never ran, so the partial assistant text the user had been watching
disappeared from the transcript on the next refresh.

These tests stub out the Ollama call so we can deterministically inject
a CancelledError mid-stream and verify the persisted row.
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator
from unittest.mock import patch

import pytest

from backend import agent

pytestmark = pytest.mark.smoke


def _run(coro):
    return asyncio.run(coro)


async def _fake_stream_two_chunks_then_cancel() -> AsyncIterator[dict]:
    """Stub for `_stream_ollama_chat`: yields two text chunks, then
    raises CancelledError to simulate the user clicking Stop after
    seeing some streamed output."""
    yield {"message": {"content": "Hello, "}, "done": False}
    yield {"message": {"content": "I'm working on it"}, "done": False}
    raise asyncio.CancelledError("client aborted")


def test_partial_text_persists_on_cancel(isolated_db):
    """Cancellation during streaming → assistant row exists with
    accumulated text + the stopped marker."""
    conv = isolated_db.create_conversation(
        title="t", model="gemma4:e4b", cwd="C:/tmp",
    )
    cid = conv["id"]

    async def driver():
        # Patch `_stream_ollama_chat` to our fake, then iterate
        # run_turn until it raises (the cancel propagates out).
        with patch("backend.agent._stream_ollama_chat",
                   side_effect=lambda *a, **kw: _fake_stream_two_chunks_then_cancel()):
            try:
                async for _ev in agent.run_turn(cid, user_text="hi"):
                    pass
            except asyncio.CancelledError:
                pass

    _run(driver())

    msgs = isolated_db.list_messages(cid)
    # At least one user row + one assistant partial row.
    roles = [m["role"] for m in msgs]
    assert "assistant" in roles, (
        f"Expected an assistant row to be persisted on cancel, got: "
        f"{[(m['role'], m.get('content', '')[:30]) for m in msgs]}"
    )
    assistant = [m for m in msgs if m["role"] == "assistant"][-1]
    body = assistant["content"]
    # Both streamed chunks landed.
    assert "Hello," in body and "working on it" in body
    # The marker tells the user (and the model on next replay) that
    # this row isn't a finished answer.
    assert "[stopped mid-response]" in body


async def _fake_stream_thinking_then_cancel() -> AsyncIterator[dict]:
    """Model wedged in reasoning — produces only thinking tokens, then
    gets cancelled before any final answer."""
    yield {"message": {"thinking": "Let me think... "}, "done": False}
    yield {"message": {"thinking": "Maybe the answer is X"}, "done": False}
    raise asyncio.CancelledError()


def test_partial_persist_surfaces_thinking_when_no_answer(isolated_db):
    """Pure-thinking cancellation → assistant row gets the thinking
    body so the user sees SOMETHING rather than a blank row."""
    conv = isolated_db.create_conversation(
        title="t", model="gemma4:e4b", cwd="C:/tmp",
    )
    cid = conv["id"]

    async def driver():
        with patch("backend.agent._stream_ollama_chat",
                   side_effect=lambda *a, **kw: _fake_stream_thinking_then_cancel()):
            try:
                async for _ev in agent.run_turn(cid, user_text="hi"):
                    pass
            except asyncio.CancelledError:
                pass

    _run(driver())

    msgs = isolated_db.list_messages(cid)
    assistants = [m for m in msgs if m["role"] == "assistant"]
    assert assistants, "expected partial assistant row even with only thinking"
    body = assistants[-1]["content"]
    assert "Let me think" in body or "answer is X" in body
    assert "[stopped mid-response]" in body


async def _fake_stream_immediate_cancel() -> AsyncIterator[dict]:
    """Cancellation before any content streams — no row should be
    written (writing an empty row would clutter the transcript)."""
    if False:
        yield {}  # mark this as an async generator
    raise asyncio.CancelledError()


def test_no_persist_when_nothing_streamed(isolated_db):
    """If cancellation hits BEFORE any tokens or tool calls came in,
    don't write a blank assistant row."""
    conv = isolated_db.create_conversation(
        title="t", model="gemma4:e4b", cwd="C:/tmp",
    )
    cid = conv["id"]

    async def driver():
        with patch("backend.agent._stream_ollama_chat",
                   side_effect=lambda *a, **kw: _fake_stream_immediate_cancel()):
            try:
                async for _ev in agent.run_turn(cid, user_text="hi"):
                    pass
            except asyncio.CancelledError:
                pass

    _run(driver())

    msgs = isolated_db.list_messages(cid)
    assistants = [m for m in msgs if m["role"] == "assistant"]
    assert assistants == [], (
        "no assistant row should be written when cancel hits before "
        "any content; got: " + str([m["content"][:40] for m in assistants])
    )
