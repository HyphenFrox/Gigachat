"""Regression: split-model chat routing — Phase 2 commit 5.

Pins:
  * `compute_pool.pick_split_chat_target(model_name)` recognizes
    `split:<label>` model identifiers and resolves them to a running
    llama-server URL+label only when the row's status is `running`.
  * `agent._stream_llama_server_chat` translates llama-server's
    OpenAI-shape SSE chunks into the Ollama-shape dicts the rest of
    the agent loop expects (`{message: {role, content}, done: bool}`).

Network is fully stubbed via `httpx.MockTransport`; no real
llama-server needed.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from backend import agent, compute_pool

pytestmark = pytest.mark.smoke


def _run(coro):
    return asyncio.run(coro)


# --- pick_split_chat_target ----------------------------------------------


def test_pick_split_target_none_for_normal_model(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    assert compute_pool.pick_split_chat_target("gemma4:e4b") is None
    assert compute_pool.pick_split_chat_target("") is None
    assert compute_pool.pick_split_chat_target(None) is None


def test_pick_split_target_none_when_label_unknown(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    assert compute_pool.pick_split_chat_target("split:no-such-label") is None


def test_pick_split_target_skips_stopped_rows(isolated_db, monkeypatch):
    """A stopped split model must not route — no llama-server is bound
    to its port. Routing there would 502 every chat turn."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    isolated_db.create_split_model(label="big", gguf_path="/m.gguf")
    # Status defaults to 'stopped'.
    assert compute_pool.pick_split_chat_target("split:big") is None


def test_pick_split_target_resolves_running_row(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    sid = isolated_db.create_split_model(
        label="big-q3", gguf_path="/m.gguf", llama_port=11500,
    )
    isolated_db.update_split_model_status(sid, status="running")
    target = compute_pool.pick_split_chat_target("split:big-q3")
    assert target is not None
    base, label = target
    assert base == "http://127.0.0.1:11500"
    assert label == "big-q3"


def test_pick_split_target_skips_disabled_rows(isolated_db, monkeypatch):
    """A row that's been disabled but happens to still have a
    llama-server alive must not route here — disabling is the user's
    explicit signal of 'don't use this'."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    sid = isolated_db.create_split_model(
        label="big", gguf_path="/m.gguf", enabled=False,
    )
    isolated_db.update_split_model_status(sid, status="running")
    assert compute_pool.pick_split_chat_target("split:big") is None


# --- _stream_llama_server_chat (OpenAI SSE → Ollama-shape) ---------------


def _sse(chunks: list[dict | str]) -> bytes:
    """Build a fake SSE stream payload. Pass dicts to be JSON-encoded as
    `data: <json>`, or the literal string '[DONE]' for the terminator."""
    parts = []
    for c in chunks:
        if isinstance(c, str):
            parts.append(f"data: {c}\n\n")
        else:
            parts.append(f"data: {json.dumps(c)}\n\n")
    return "".join(parts).encode("utf-8")


def _install_mock_http(monkeypatch, handler):
    real = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real(*args, **kwargs)

    monkeypatch.setattr(agent.httpx, "AsyncClient", _factory)


def test_stream_translates_content_deltas(monkeypatch):
    """Three OpenAI deltas → three Ollama-shape chunks + a final
    done=True from the [DONE] terminator."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["model"] == "big-q3"
        sse = _sse([
            {"choices": [{"delta": {"role": "assistant", "content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {"choices": [{"delta": {"content": "!"}}]},
            "[DONE]",
        ])
        return httpx.Response(
            200,
            content=sse,
            headers={"content-type": "text/event-stream"},
        )

    _install_mock_http(monkeypatch, handler)

    async def go():
        chunks = []
        async for c in agent._stream_llama_server_chat(
            model="big-q3",
            messages=[{"role": "user", "content": "hi"}],
            base_url="http://127.0.0.1:11500",
        ):
            chunks.append(c)
        return chunks

    chunks = _run(go())
    # 3 content chunks + 1 final done=True.
    assert len(chunks) == 4
    text = "".join(c["message"]["content"] for c in chunks)
    assert text == "Hello!"
    assert chunks[-1]["done"] is True
    assert all(c["message"]["role"] == "assistant" for c in chunks)


def test_stream_skips_non_data_lines(monkeypatch):
    """SSE allows comment / id / event lines we don't care about; they
    must be silently skipped instead of breaking the parse."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            ": this is a comment\n"
            "event: progress\n"
            f"data: {json.dumps({'choices': [{'delta': {'content': 'hi'}}]})}\n\n"
            "data: [DONE]\n\n"
        ).encode("utf-8")
        return httpx.Response(
            200, content=body,
            headers={"content-type": "text/event-stream"},
        )

    _install_mock_http(monkeypatch, handler)

    async def go():
        chunks = []
        async for c in agent._stream_llama_server_chat(
            model="any", messages=[],
            base_url="http://127.0.0.1:11500",
        ):
            chunks.append(c)
        return chunks

    chunks = _run(go())
    # 1 content + 1 done.
    assert len(chunks) == 2
    assert chunks[0]["message"]["content"] == "hi"


def test_stream_raises_on_4xx(monkeypatch):
    """A 400/500 from llama-server must surface — this is how the user
    sees real errors like 'model has no chat template'."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, content=b"model loading failed")

    _install_mock_http(monkeypatch, handler)

    async def go():
        chunks = []
        async for c in agent._stream_llama_server_chat(
            model="any", messages=[],
            base_url="http://127.0.0.1:11500",
        ):
            chunks.append(c)
        return chunks

    with pytest.raises(httpx.HTTPStatusError):
        _run(go())


def test_stream_sends_bearer_when_token_set(monkeypatch):
    seen_auth: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization"))
        return httpx.Response(
            200,
            content=_sse([{"choices": [{"delta": {"content": ""}}]}, "[DONE]"]),
            headers={"content-type": "text/event-stream"},
        )

    _install_mock_http(monkeypatch, handler)

    async def go():
        async for _ in agent._stream_llama_server_chat(
            model="any", messages=[],
            base_url="http://127.0.0.1:11500",
            auth_token="hunter2",
        ):
            pass

    _run(go())
    assert seen_auth == ["Bearer hunter2"]
