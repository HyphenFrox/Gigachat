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


# --- route_chat_for (auto-router) ----------------------------------------


def _stub_split_lifecycle(monkeypatch, started_ports: dict[str, int]):
    """Replace split_lifecycle.start/stop with stubs that just flip
    DB status. Avoids spawning real subprocesses in tests."""
    import asyncio
    from backend import split_lifecycle

    async def _start(sid):
        # Flip the row to running and pick its port from the DB row.
        from backend import db as _db
        row = _db.get_split_model(sid)
        if not row:
            return {"ok": False, "status": "error", "error": "not found"}
        _db.update_split_model_status(sid, status="running", last_error="")
        started_ports[sid] = row["llama_port"]
        return {"ok": True, "status": "running", "port": row["llama_port"]}

    async def _stop(sid, *, _from_failed_start=False):
        from backend import db as _db
        _db.update_split_model_status(sid, status="stopped")
        started_ports.pop(sid, None)
        return {"ok": True, "status": "stopped"}

    monkeypatch.setattr(split_lifecycle, "start", _start)
    monkeypatch.setattr(split_lifecycle, "stop", _stop)


def _seed_eligible_split_worker(isolated_db, *, label, address, model_present="gemma3:27b"):
    """Worker with rpc-server reachable + use_for_chat enabled."""
    import time
    wid = isolated_db.create_compute_worker(
        label=label, address=address, use_for_chat=True,
    )
    isolated_db.update_compute_worker_capabilities(
        wid,
        capabilities={
            "version": "0.5.4",
            "models": [{"name": model_present, "details": {}}],
            "rpc_server_reachable": True,
            "rpc_port": 50052,
            "rpc_error": None,
        },
        last_seen=time.time() - 5.0,
        last_error="",
    )
    return wid


def test_route_uses_ollama_when_model_not_in_manifest(isolated_db, monkeypatch):
    """Model name we can't resolve to a GGUF blob → fall through to
    Ollama (Ollama itself will surface the right error if the model
    truly doesn't exist anywhere)."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    # Make resolve_ollama_model return None.
    monkeypatch.setattr(compute_pool, "resolve_ollama_model", lambda _: None)
    decision = _run(compute_pool.route_chat_for("custom:thing"))
    assert decision == {"engine": "ollama"}


def test_route_uses_ollama_when_model_fits_host_vram(isolated_db, monkeypatch):
    """Model size < host budget → host Ollama (fastest path; no LAN
    overhead, workers stay free for parallel embed/subagent)."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    # 4 GB model.
    monkeypatch.setattr(
        compute_pool, "resolve_ollama_model",
        lambda name: {"gguf_path": "/m.gguf", "size_bytes": 4 * 1024 ** 3, "manifest": {}},
    )
    # Host has 8 GB VRAM (host_budget = 8 * 0.85 = 6.8 GB).
    monkeypatch.setattr(compute_pool, "_host_vram_budget_bytes", lambda: 8 * 1024 ** 3 * 85 // 100)
    decision = _run(compute_pool.route_chat_for("small:7b"))
    assert decision == {"engine": "ollama"}


def test_route_falls_back_to_ollama_when_model_too_big_but_no_workers(isolated_db, monkeypatch):
    """Model exceeds host VRAM but no eligible workers exist → fall
    back to Ollama. Ollama's CPU offload handles it (slowly) — strictly
    better than refusing.

    The decision MAY carry extra fields (`mega_model`, `model_size_gb`,
    `pool_memory_gb`) when the model exceeds host total memory — those
    are advisory metadata for the UI, not behavioural changes. We
    assert on `engine` only so the test doesn't break when the
    metadata schema evolves.
    """
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    monkeypatch.setattr(
        compute_pool, "resolve_ollama_model",
        lambda name: {"gguf_path": "/m.gguf", "size_bytes": 12 * 1024 ** 3, "manifest": {}},
    )
    monkeypatch.setattr(compute_pool, "_host_vram_budget_bytes", lambda: 8 * 1024 ** 3 * 85 // 100)
    decision = _run(compute_pool.route_chat_for("big:27b"))
    assert decision.get("engine") == "ollama"


def test_route_engages_split_when_model_too_big_and_workers_eligible(isolated_db, monkeypatch):
    """Model exceeds host VRAM AND a worker has rpc-server reachable
    → auto-spawn llama-server and route there."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_eligible_split_worker(isolated_db, label="A", address="a.local")

    monkeypatch.setattr(
        compute_pool, "resolve_ollama_model",
        lambda name: {"gguf_path": "/big.gguf", "size_bytes": 17 * 1024 ** 3, "manifest": {}},
    )
    monkeypatch.setattr(compute_pool, "_host_vram_budget_bytes", lambda: 8 * 1024 ** 3 * 85 // 100)

    started_ports: dict[str, int] = {}
    _stub_split_lifecycle(monkeypatch, started_ports)

    decision = _run(compute_pool.route_chat_for("big:27b"))
    assert decision["engine"] == "llama_server"
    assert decision["label"] == "big:27b"
    assert decision["base_url"].startswith("http://127.0.0.1:")
    # Auto-created a split_models row for this exact model.
    rows = isolated_db.list_split_models()
    assert any(r["label"] == "big:27b" for r in rows)
    # And started llama-server for it (stub flipped status to running).
    assert any(r["status"] == "running" for r in rows)


def test_route_reuses_existing_running_split_for_same_model(isolated_db, monkeypatch):
    """Second turn for the same big model → reuse the warm
    llama-server (no fresh spawn, no double-start)."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_eligible_split_worker(isolated_db, label="A", address="a.local")
    monkeypatch.setattr(
        compute_pool, "resolve_ollama_model",
        lambda name: {"gguf_path": "/big.gguf", "size_bytes": 17 * 1024 ** 3, "manifest": {}},
    )
    monkeypatch.setattr(compute_pool, "_host_vram_budget_bytes", lambda: 8 * 1024 ** 3 * 85 // 100)
    started_ports: dict[str, int] = {}
    _stub_split_lifecycle(monkeypatch, started_ports)

    _run(compute_pool.route_chat_for("big:27b"))
    rows1 = isolated_db.list_split_models()
    sid1 = rows1[0]["id"]

    _run(compute_pool.route_chat_for("big:27b"))
    rows2 = isolated_db.list_split_models()
    # Same row, same id — no second auto-create.
    assert len(rows2) == 1
    assert rows2[0]["id"] == sid1
    assert rows2[0]["status"] == "running"


def test_route_stops_other_split_when_switching_models(isolated_db, monkeypatch):
    """Switching from big-model-A to big-model-B should stop A's
    llama-server (one big model hot at a time — finite VRAM)."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_eligible_split_worker(isolated_db, label="A", address="a.local")

    sizes = {"a:27b": 17, "b:32b": 19}
    monkeypatch.setattr(
        compute_pool, "resolve_ollama_model",
        lambda name: {"gguf_path": f"/{name}.gguf", "size_bytes": sizes[name] * 1024 ** 3, "manifest": {}},
    )
    monkeypatch.setattr(compute_pool, "_host_vram_budget_bytes", lambda: 8 * 1024 ** 3 * 85 // 100)
    started_ports: dict[str, int] = {}
    _stub_split_lifecycle(monkeypatch, started_ports)

    _run(compute_pool.route_chat_for("a:27b"))
    _run(compute_pool.route_chat_for("b:32b"))

    rows = {r["label"]: r for r in isolated_db.list_split_models()}
    assert rows["a:27b"]["status"] == "stopped"
    assert rows["b:32b"]["status"] == "running"


def test_route_stops_split_when_switching_to_small_model(isolated_db, monkeypatch):
    """Switching from a big split model back to a small Ollama-fits
    model → free the VRAM by stopping the llama-server."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_eligible_split_worker(isolated_db, label="A", address="a.local")

    sizes = {"big:27b": 17, "small:7b": 4}
    monkeypatch.setattr(
        compute_pool, "resolve_ollama_model",
        lambda name: {"gguf_path": f"/{name}.gguf", "size_bytes": sizes[name] * 1024 ** 3, "manifest": {}},
    )
    monkeypatch.setattr(compute_pool, "_host_vram_budget_bytes", lambda: 8 * 1024 ** 3 * 85 // 100)
    started_ports: dict[str, int] = {}
    _stub_split_lifecycle(monkeypatch, started_ports)

    _run(compute_pool.route_chat_for("big:27b"))
    decision = _run(compute_pool.route_chat_for("small:7b"))
    assert decision == {"engine": "ollama"}
    rows = {r["label"]: r for r in isolated_db.list_split_models()}
    # Big was stopped during the switch.
    assert rows["big:27b"]["status"] == "stopped"
