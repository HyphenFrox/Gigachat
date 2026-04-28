"""Regression: `agent._embed_text` routes to compute workers when eligible.

The compute pool's first real production wiring lives in `agent._embed_text`:
when a registered worker has `use_for_embeddings=True` and the embed model
installed, the embed call goes there instead of the host's local Ollama.
This frees the host's GPU for chat / subagent work.

Invariants tested here:
  * No eligible worker → request goes to OLLAMA_URL (host).
  * Eligible worker → request goes to that worker's base URL, carrying the
    bearer token if one is set.
  * Worker is unreachable / 5xx → silent fallback to host. Embedding is an
    enhancement; a flaky worker must not break recall.

httpx is fully stubbed via MockTransport so this runs offline.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from backend import agent, compute_pool

pytestmark = pytest.mark.smoke


def _install_mock_http_on(module, handler):
    """Wrap `httpx.AsyncClient` in `module`'s namespace with MockTransport."""
    real_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    # agent.py does `import httpx` so the class is at `agent.httpx.AsyncClient`.
    return _factory


def _run(coro):
    return asyncio.run(coro)


def _seed_eligible_worker(isolated_db, *, address: str, auth_token=None) -> str:
    import time as _t
    wid = isolated_db.create_compute_worker(
        label="W", address=address, auth_token=auth_token, use_for_embeddings=True,
    )
    isolated_db.update_compute_worker_capabilities(
        wid,
        capabilities={
            "version": "0.5.4",
            "models": [{"name": agent.EMBED_MODEL + ":latest", "details": {}}],
        },
        last_seen=_t.time() - 5.0,
        last_error="",
    )
    return wid


def test_embed_falls_through_to_host_when_no_workers(isolated_db, monkeypatch):
    """Baseline: no workers registered, embed must hit OLLAMA_URL."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)

    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, json={"embedding": [1.0, 0.0, 0.0]})

    monkeypatch.setattr(agent.httpx, "AsyncClient", _install_mock_http_on(agent, handler))
    vec = _run(agent._embed_text("hello"))
    assert vec is not None
    assert seen_urls, "embed request never fired"
    assert seen_urls[0].startswith(agent.OLLAMA_URL)


def test_embed_routes_to_eligible_worker(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_eligible_worker(isolated_db, address="worker.local", auth_token="hunter2")

    seen_urls: list[str] = []
    seen_auth: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        seen_auth.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"embedding": [0.0, 1.0, 0.0]})

    monkeypatch.setattr(agent.httpx, "AsyncClient", _install_mock_http_on(agent, handler))
    vec = _run(agent._embed_text("hello"))
    assert vec is not None
    assert seen_urls, "embed request never fired"
    assert seen_urls[0].startswith("http://worker.local:11434")
    assert seen_auth[0] == "Bearer hunter2"


def test_embed_falls_back_to_host_when_worker_errors(isolated_db, monkeypatch):
    """Worker returns 500 → silently fall through to host Ollama. The user's
    recall feature must not break because a registered laptop is rebooting."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_eligible_worker(isolated_db, address="worker.local")

    seen_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(request.url.host)
        if request.url.host == "worker.local":
            return httpx.Response(500, json={"error": "boom"})
        # Host Ollama (localhost) — happy path.
        return httpx.Response(200, json={"embedding": [0.0, 0.0, 1.0]})

    monkeypatch.setattr(agent.httpx, "AsyncClient", _install_mock_http_on(agent, handler))
    vec = _run(agent._embed_text("hello"))
    assert vec is not None
    # Worker tried first, host tried second.
    assert seen_hosts[0] == "worker.local"
    assert "localhost" in seen_hosts[-1] or "127.0.0.1" in seen_hosts[-1]


def test_embed_returns_none_when_both_fail(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_eligible_worker(isolated_db, address="worker.local")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    monkeypatch.setattr(agent.httpx, "AsyncClient", _install_mock_http_on(agent, handler))
    vec = _run(agent._embed_text("hello"))
    assert vec is None
