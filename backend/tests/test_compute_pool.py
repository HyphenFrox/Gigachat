"""Regression: compute-pool capability probe + liveness sweep.

These tests cover commit #2 of the multi-PC feature — the layer above the
DB CRUD that talks to each worker's Ollama and persists what it learns:

  * `_worker_base_url` builds the right URL regardless of how the user
    pasted the address (bare hostname / `http://...` / `https://...` /
    trailing slash). Defends against the most common copy-paste shapes.
  * `probe_worker` happy-path → capabilities + last_seen persisted, no
    error recorded.
  * `probe_worker` network failure → `last_error` recorded; the row stays
    queryable so the UI can show "unreachable since X".
  * `probe_worker` partial failure (one of the two endpoints returns 5xx)
    → the working endpoint's data is still saved; only the broken half is
    surfaced as an error. Prevents an old worker without `/api/version`
    from looking entirely dead.
  * Auth token, when set, is sent as `Authorization: Bearer <token>` on
    BOTH probe requests.
  * `probe_all_enabled` returns an empty list when nothing is registered
    (boot path) and aggregates summaries when workers exist.

Network is fully stubbed via `httpx.MockTransport` — these tests never
actually open a socket, so they're safe to run alongside the rest of the
smoke suite without an Ollama instance present.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from backend import compute_pool

pytestmark = pytest.mark.smoke


# --- helpers --------------------------------------------------------------


def _install_mock_http(monkeypatch, handler):
    """Wrap `httpx.AsyncClient` so every probe call uses a MockTransport.

    `compute_pool.probe_worker` constructs its own `httpx.AsyncClient`, so
    we can't pass a mock client in directly. Instead we replace the class
    in the module's namespace with a factory that always supplies our
    mock transport — caller-side code is unchanged.
    """
    real_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def _factory(*args, **kwargs):
        # Caller may pass `timeout=...`; honor it but force our transport.
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(compute_pool.httpx, "AsyncClient", _factory)


def _run(coro):
    """Run an async coroutine to completion in a fresh event loop."""
    return asyncio.run(coro)


# --- _worker_base_url -----------------------------------------------------


def test_base_url_plain_hostname():
    url = compute_pool._worker_base_url(
        {"address": "desktop-0692hok.local", "ollama_port": 11434},
    )
    assert url == "http://desktop-0692hok.local:11434"


def test_base_url_strips_http_scheme():
    """User pastes `http://1.2.3.4` from a browser tab — strip it."""
    url = compute_pool._worker_base_url(
        {"address": "http://1.2.3.4", "ollama_port": 11434},
    )
    assert url == "http://1.2.3.4:11434"


def test_base_url_strips_https_scheme_and_trailing_slash():
    url = compute_pool._worker_base_url(
        {"address": "https://my-host.local/", "ollama_port": 8080},
    )
    assert url == "http://my-host.local:8080"


def test_base_url_default_port_when_missing():
    url = compute_pool._worker_base_url({"address": "x.local"})
    assert url == "http://x.local:11434"


# --- probe_worker: bookkeeping ---------------------------------------------


def test_probe_worker_unknown_id_returns_not_found(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    result = _run(compute_pool.probe_worker("nonexistent"))
    assert result["ok"] is False
    assert "not found" in result["error"]


def test_probe_worker_disabled_short_circuits(isolated_db, monkeypatch):
    """Disabled workers must not be probed at all — the user toggled them
    off for a reason (offline, being maintained, etc.)."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    wid = isolated_db.create_compute_worker(
        label="A", address="a", transport="lan", enabled=False,
    )
    result = _run(compute_pool.probe_worker(wid))
    assert result["ok"] is False
    assert "disabled" in result["error"]


# --- probe_worker: happy path ---------------------------------------------


def test_probe_worker_persists_capabilities_on_success(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    wid = isolated_db.create_compute_worker(
        label="A", address="x.local", transport="lan",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.5.4"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {
                    "name": "gemma4:e4b",
                    "size": 4500000000,
                    "details": {
                        "family": "gemma",
                        "parameter_size": "4B",
                        "quantization_level": "Q4_K_M",
                    },
                },
                {
                    "name": "nomic-embed-text:latest",
                    "size": 274000000,
                    "details": {"family": "nomic-bert"},
                },
            ]})
        return httpx.Response(404)

    _install_mock_http(monkeypatch, handler)

    result = _run(compute_pool.probe_worker(wid))
    assert result["ok"] is True
    assert result["error"] is None
    assert result["capabilities"]["version"] == "0.5.4"
    names = [m["name"] for m in result["capabilities"]["models"]]
    assert "gemma4:e4b" in names

    # Persisted on the row?
    row = isolated_db.get_compute_worker(wid)
    assert row["capabilities"]["version"] == "0.5.4"
    assert row["last_seen"] is not None
    assert row["last_error"] is None  # cleared on success


# --- probe_worker: network failure ----------------------------------------


def test_probe_worker_records_network_failure(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    wid = isolated_db.create_compute_worker(
        label="A", address="dead.local", transport="lan",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        # Simulate connection refused at the transport layer.
        raise httpx.ConnectError("connection refused")

    _install_mock_http(monkeypatch, handler)

    result = _run(compute_pool.probe_worker(wid))
    assert result["ok"] is False
    assert "ConnectError" in result["error"] or "connection" in result["error"].lower()

    row = isolated_db.get_compute_worker(wid)
    assert row["last_error"] is not None
    assert row["last_seen"] is not None  # we still recorded the attempt time


# --- probe_worker: partial failure ----------------------------------------


def test_probe_worker_partial_failure_keeps_working_half(isolated_db, monkeypatch):
    """Old Ollama builds may not implement `/api/version`. We must still
    surface the model list from `/api/tags` rather than treating the whole
    worker as dead — the routing layer can then send chat traffic to it."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    wid = isolated_db.create_compute_worker(
        label="A", address="x.local", transport="lan",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(404, json={"error": "not implemented"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "gemma4:e4b", "size": 1, "details": {}},
            ]})
        return httpx.Response(404)

    _install_mock_http(monkeypatch, handler)

    result = _run(compute_pool.probe_worker(wid))
    # Not fully ok — version missing — but models came through.
    assert result["ok"] is False
    assert result["error"]
    assert any(m["name"] == "gemma4:e4b" for m in result["capabilities"]["models"])

    row = isolated_db.get_compute_worker(wid)
    # last_error captures the version-side problem.
    assert row["last_error"]
    assert any(m["name"] == "gemma4:e4b" for m in row["capabilities"]["models"])


# --- auth header ----------------------------------------------------------


def test_probe_worker_sends_bearer_token_when_set(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    wid = isolated_db.create_compute_worker(
        label="A", address="x.local", transport="tailscale",
        auth_token="hunter2",
    )

    seen_auth_headers: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth_headers.append(request.headers.get("authorization"))
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.5.4"})
        return httpx.Response(200, json={"models": []})

    _install_mock_http(monkeypatch, handler)
    _run(compute_pool.probe_worker(wid))

    assert seen_auth_headers, "no probe requests fired"
    # Both requests must carry the bearer.
    assert all(h == "Bearer hunter2" for h in seen_auth_headers)


def test_probe_worker_omits_auth_when_token_unset(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    wid = isolated_db.create_compute_worker(
        label="A", address="x.local", transport="lan",
    )

    seen_auth_headers: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth_headers.append(request.headers.get("authorization"))
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.5.4"})
        return httpx.Response(200, json={"models": []})

    _install_mock_http(monkeypatch, handler)
    _run(compute_pool.probe_worker(wid))

    assert seen_auth_headers, "no probe requests fired"
    assert all(h is None for h in seen_auth_headers)


# --- probe_all_enabled ----------------------------------------------------


def test_probe_all_enabled_empty_when_no_workers(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    assert _run(compute_pool.probe_all_enabled()) == []


def test_probe_all_enabled_aggregates_summaries(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    isolated_db.create_compute_worker(label="A", address="a.local", transport="lan")
    isolated_db.create_compute_worker(label="B", address="b.local", transport="lan")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a.local":
            if request.url.path == "/api/version":
                return httpx.Response(200, json={"version": "0.5.4"})
            return httpx.Response(200, json={"models": [{"name": "x", "details": {}}]})
        # b.local: simulate a hard failure
        raise httpx.ConnectError("nope")

    _install_mock_http(monkeypatch, handler)

    summaries = _run(compute_pool.probe_all_enabled())
    by_label = {s["label"]: s for s in summaries}
    assert by_label["A"]["ok"] is True
    assert by_label["B"]["ok"] is False
    assert by_label["B"]["error"]


def test_probe_all_enabled_skips_disabled(isolated_db, monkeypatch):
    """Disabled rows should never appear in the sweep. They're toggled off
    for a reason (e.g. user knows the laptop is travelling) and probing
    them just spams `last_error`."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    isolated_db.create_compute_worker(label="A", address="a.local", transport="lan", enabled=True)
    isolated_db.create_compute_worker(label="B", address="b.local", transport="lan", enabled=False)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.5.4"})
        return httpx.Response(200, json={"models": []})

    _install_mock_http(monkeypatch, handler)
    summaries = _run(compute_pool.probe_all_enabled())
    labels = {s["label"] for s in summaries}
    assert labels == {"A"}
