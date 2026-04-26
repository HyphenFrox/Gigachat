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


def _install_mock_http(monkeypatch, handler, *, rpc_reachable: bool = False):
    """Wrap `httpx.AsyncClient` so every probe call uses a MockTransport.

    `compute_pool.probe_worker` constructs its own `httpx.AsyncClient`, so
    we can't pass a mock client in directly. Instead we replace the class
    in the module's namespace with a factory that always supplies our
    mock transport — caller-side code is unchanged.

    Also stubs `_probe_rpc_server` to a no-op (fast return) so tests
    that don't care about Phase 2's rpc-server probe don't pay the
    multi-second TCP timeout per test. Pass `rpc_reachable=True` if a
    test wants to assert the rpc path is reflected in capabilities.
    """
    real_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def _factory(*args, **kwargs):
        # Caller may pass `timeout=...`; honor it but force our transport.
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(compute_pool.httpx, "AsyncClient", _factory)

    async def _stub_rpc_probe(host, port, timeout=2.0):
        return (rpc_reachable, None if rpc_reachable else "stubbed: not reachable")

    monkeypatch.setattr(compute_pool, "_probe_rpc_server", _stub_rpc_probe)


def _run(coro):
    """Run an async coroutine to completion in a fresh event loop."""
    return asyncio.run(coro)


# --- _worker_base_url -----------------------------------------------------


def test_base_url_plain_hostname():
    url = compute_pool._worker_base_url(
        {"address": "worker.local", "ollama_port": 11434},
    )
    assert url == "http://worker.local:11434"


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


# --- model name matching --------------------------------------------------


def test_model_matches_exact():
    assert compute_pool._model_matches("nomic-embed-text", "nomic-embed-text") is True
    assert compute_pool._model_matches("gemma4:e4b", "gemma4:e4b") is True


def test_model_matches_latest_tag_implicit_either_side():
    """`nomic-embed-text` (no tag) matches `nomic-embed-text:latest` (Ollama
    appends `:latest` when listing) and vice-versa."""
    assert compute_pool._model_matches("nomic-embed-text:latest", "nomic-embed-text") is True
    assert compute_pool._model_matches("nomic-embed-text", "nomic-embed-text:latest") is True


def test_model_matches_explicit_tags_must_match():
    """An explicit tag mismatch is NOT coerced — `gemma4:e4b` must not be
    routed to a worker that only has `gemma4:e2b`."""
    assert compute_pool._model_matches("gemma4:e2b", "gemma4:e4b") is False
    assert compute_pool._model_matches("gemma4:e4b", "gemma4:e2b") is False


def test_model_matches_different_models_never_match():
    assert compute_pool._model_matches("llama3.1:8b", "gemma4:e4b") is False


# --- pick_embed_target ----------------------------------------------------


def _seed_eligible_worker(
    isolated_db,
    *,
    label: str = "A",
    address: str = "a.local",
    models=("nomic-embed-text:latest",),
    use_for_embeddings: bool = True,
    enabled: bool = True,
    last_seen_age_sec: float = 30.0,
    last_error: str = "",
    auth_token: str | None = None,
) -> str:
    """Helper: create a worker AND simulate a successful recent probe."""
    import time as _t
    wid = isolated_db.create_compute_worker(
        label=label,
        address=address,
        transport="lan",
        enabled=enabled,
        use_for_embeddings=use_for_embeddings,
        auth_token=auth_token,
    )
    isolated_db.update_compute_worker_capabilities(
        wid,
        capabilities={
            "version": "0.5.4",
            "models": [{"name": m, "details": {}} for m in models],
        },
        last_seen=_t.time() - last_seen_age_sec,
        last_error=last_error,
    )
    return wid


def test_pick_embed_target_none_when_no_workers(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    assert compute_pool.pick_embed_target("nomic-embed-text") is None


def test_pick_embed_target_picks_eligible_worker(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_eligible_worker(isolated_db, address="a.local", auth_token="hunter2")
    target = compute_pool.pick_embed_target("nomic-embed-text")
    assert target is not None
    base, token = target
    assert base == "http://a.local:11434"
    assert token == "hunter2"


def test_pick_embed_target_skips_disabled(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_eligible_worker(isolated_db, label="A", address="a.local", enabled=False)
    assert compute_pool.pick_embed_target("nomic-embed-text") is None


def test_pick_embed_target_skips_use_for_embeddings_false(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_eligible_worker(isolated_db, address="a.local", use_for_embeddings=False)
    assert compute_pool.pick_embed_target("nomic-embed-text") is None


def test_pick_embed_target_skips_workers_without_model(isolated_db, monkeypatch):
    """Routing to a worker that doesn't have the embed model would force
    Ollama there to pull it (slow + uses internet bandwidth). Skip those
    workers — the host can serve the embed locally instead."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_eligible_worker(
        isolated_db, address="a.local",
        models=("gemma4:e4b",),  # chat model, no embedder
    )
    assert compute_pool.pick_embed_target("nomic-embed-text") is None


def test_pick_embed_target_skips_stale_workers(isolated_db, monkeypatch):
    """Last seen > 1 hour ago — treat as unhealthy until the next probe."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_eligible_worker(
        isolated_db, address="a.local", last_seen_age_sec=2 * 60 * 60,
    )
    assert compute_pool.pick_embed_target("nomic-embed-text") is None


def test_pick_embed_target_skips_workers_with_last_error(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_eligible_worker(
        isolated_db, address="a.local", last_error="connection refused",
    )
    assert compute_pool.pick_embed_target("nomic-embed-text") is None


def test_pick_embed_target_skips_never_probed(isolated_db, monkeypatch):
    """A row the user just added but the periodic sweep hasn't touched
    yet (last_seen=None). Don't route until we've confirmed it works."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    isolated_db.create_compute_worker(
        label="A", address="a.local", transport="lan",
    )
    assert compute_pool.pick_embed_target("nomic-embed-text") is None


def test_pick_embed_target_prefers_freshest_when_multiple(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_eligible_worker(
        isolated_db, label="old", address="old.local", last_seen_age_sec=1800,
    )
    _seed_eligible_worker(
        isolated_db, label="new", address="new.local", last_seen_age_sec=10,
    )
    target = compute_pool.pick_embed_target("nomic-embed-text")
    assert target is not None
    assert target[0] == "http://new.local:11434"


def test_pick_embed_target_no_token_when_unset(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_eligible_worker(isolated_db, address="a.local")  # no auth_token
    target = compute_pool.pick_embed_target("nomic-embed-text")
    assert target is not None
    assert target[1] is None


# --- list_subagent_workers ------------------------------------------------


def _seed_eligible_subagent_worker(isolated_db, **kwargs):
    """Helper for subagent picker tests — defaults to a chat model + the
    `use_for_subagents` flag, but still uses the same eligibility shape
    as the embed helper so we test the same plumbing."""
    import time as _t
    kwargs.setdefault("models", ("gemma4:e4b",))
    kwargs.setdefault("use_for_embeddings", False)  # not relevant here
    last_seen_age = kwargs.pop("last_seen_age_sec", 30.0)
    last_error = kwargs.pop("last_error", "")
    auth_token = kwargs.pop("auth_token", None)
    label = kwargs.pop("label", "A")
    address = kwargs.pop("address", "a.local")
    enabled = kwargs.pop("enabled", True)
    use_for_subagents = kwargs.pop("use_for_subagents", True)
    models = kwargs.pop("models")
    _ = kwargs.pop("use_for_embeddings", None)  # discard; we only need subagents flag
    wid = isolated_db.create_compute_worker(
        label=label,
        address=address,
        transport="lan",
        enabled=enabled,
        use_for_subagents=use_for_subagents,
        auth_token=auth_token,
    )
    isolated_db.update_compute_worker_capabilities(
        wid,
        capabilities={
            "version": "0.5.4",
            "models": [{"name": m, "details": {}} for m in models],
        },
        last_seen=_t.time() - last_seen_age,
        last_error=last_error,
    )
    return wid


def test_list_subagent_workers_empty_when_no_workers(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    assert compute_pool.list_subagent_workers("gemma4:e4b") == []


def test_list_subagent_workers_returns_eligible_only(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_eligible_subagent_worker(isolated_db, label="A", address="a.local")
    _seed_eligible_subagent_worker(
        isolated_db, label="off", address="off.local", use_for_subagents=False,
    )
    _seed_eligible_subagent_worker(
        isolated_db, label="bad-model",
        address="bad.local", models=("llama3.1:8b",),
    )
    targets = compute_pool.list_subagent_workers("gemma4:e4b")
    assert len(targets) == 1
    assert targets[0][0] == "http://a.local:11434"


def test_list_subagent_workers_orders_freshest_first(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_eligible_subagent_worker(
        isolated_db, label="old", address="old.local", last_seen_age_sec=1800,
    )
    _seed_eligible_subagent_worker(
        isolated_db, label="new", address="new.local", last_seen_age_sec=10,
    )
    targets = compute_pool.list_subagent_workers("gemma4:e4b")
    assert [t[0] for t in targets] == [
        "http://new.local:11434",
        "http://old.local:11434",
    ]


def test_list_subagent_workers_skips_workers_without_model(isolated_db, monkeypatch):
    """If none of the workers has the requested chat model installed, we
    can't route subagents there — Ollama would try to pull on the fly
    and the fan-out latency would blow up. Caller falls back to host."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_eligible_subagent_worker(
        isolated_db, address="a.local", models=("nomic-embed-text:latest",),
    )
    assert compute_pool.list_subagent_workers("gemma4:e4b") == []


# --- pick_chat_target -----------------------------------------------------


def _seed_chat_worker(isolated_db, **kwargs):
    """Same shape as _seed_eligible_worker but for the chat flag."""
    import time as _t
    label = kwargs.pop("label", "C")
    address = kwargs.pop("address", "c.local")
    models = kwargs.pop("models", ("gemma4:e4b",))
    use_for_chat = kwargs.pop("use_for_chat", True)
    enabled = kwargs.pop("enabled", True)
    last_seen_age = kwargs.pop("last_seen_age_sec", 30.0)
    last_error = kwargs.pop("last_error", "")
    auth_token = kwargs.pop("auth_token", None)
    wid = isolated_db.create_compute_worker(
        label=label, address=address, transport="lan",
        enabled=enabled, use_for_chat=use_for_chat, auth_token=auth_token,
    )
    isolated_db.update_compute_worker_capabilities(
        wid,
        capabilities={
            "version": "0.5.4",
            "models": [{"name": m, "details": {}} for m in models],
        },
        last_seen=_t.time() - last_seen_age,
        last_error=last_error,
    )
    return wid


def test_pick_chat_target_none_when_no_workers(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    assert compute_pool.pick_chat_target("gemma4:e4b") is None


def test_pick_chat_target_picks_eligible(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_chat_worker(isolated_db, address="c.local", auth_token="t")
    target = compute_pool.pick_chat_target("gemma4:e4b")
    assert target is not None
    assert target == ("http://c.local:11434", "t")


def test_pick_chat_target_skips_use_for_chat_false(isolated_db, monkeypatch):
    """Toggling chat off but keeping the row enabled means the user wants
    this worker for embeddings/subagents only — chat must NOT route here."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_chat_worker(isolated_db, address="c.local", use_for_chat=False)
    assert compute_pool.pick_chat_target("gemma4:e4b") is None


def test_pick_chat_target_skips_workers_without_model(isolated_db, monkeypatch):
    """The chat picker must enforce model presence — Ollama would
    otherwise pull a multi-GB model mid-turn just to fail the request."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_chat_worker(
        isolated_db, address="c.local", models=("nomic-embed-text:latest",),
    )
    assert compute_pool.pick_chat_target("gemma4:e4b") is None


# --- Phase 2 commit 3: rpc-server probe -----------------------------------


def test_probe_records_rpc_reachable_in_capabilities(isolated_db, monkeypatch):
    """When rpc-server is up on the worker (the laptop has installed +
    started llama.cpp's rpc-server.exe), the probe must surface that on
    the row's capabilities so the Settings UI can render a green "RPC"
    badge separately from the Ollama status."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    wid = isolated_db.create_compute_worker(
        label="A", address="x.local", transport="lan",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.5.4"})
        return httpx.Response(200, json={"models": [
            {"name": "gemma4:e4b", "size": 1, "details": {}},
        ]})

    _install_mock_http(monkeypatch, handler, rpc_reachable=True)

    result = _run(compute_pool.probe_worker(wid))
    assert result["ok"] is True
    caps = result["capabilities"]
    assert caps["rpc_server_reachable"] is True
    assert caps["rpc_port"] == compute_pool._DEFAULT_RPC_PORT
    assert caps["rpc_error"] is None

    # Persisted on the row.
    row = isolated_db.get_compute_worker(wid)
    assert row["capabilities"]["rpc_server_reachable"] is True


def test_probe_records_rpc_unreachable_separately_from_last_error(
    isolated_db, monkeypatch
):
    """rpc-server being down is NOT a worker-level error — Phase 1
    routing (chat / embed / subagent via Ollama) still works without
    rpc-server. The row's `last_error` must stay clean so the worker
    counts as online for those workloads, while `capabilities.rpc_*`
    reflects the missing rpc daemon for the Settings UI."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    wid = isolated_db.create_compute_worker(
        label="A", address="x.local", transport="lan",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.5.4"})
        return httpx.Response(200, json={"models": [
            {"name": "gemma4:e4b", "size": 1, "details": {}},
        ]})

    # Default rpc_reachable=False from the helper.
    _install_mock_http(monkeypatch, handler)

    result = _run(compute_pool.probe_worker(wid))
    assert result["ok"] is True              # ollama still good
    assert result["error"] is None           # last_error stays clean
    caps = result["capabilities"]
    assert caps["rpc_server_reachable"] is False
    assert caps["rpc_error"]                 # error string present


def test_probe_rpc_server_real_call_against_unbound_port(monkeypatch):
    """Smoke-test the actual TCP probe (no stub) against a port that's
    very unlikely to be bound. Confirms the function returns a (False,
    error) tuple in a reasonable time rather than hanging forever."""
    import asyncio
    # Use 127.0.0.1:1 — port 1 is reserved/unused on practically every
    # system. asyncio.open_connection will get ECONNREFUSED quickly.
    ok, err = asyncio.run(
        compute_pool._probe_rpc_server("127.0.0.1", 1, timeout=2.0),
    )
    assert ok is False
    assert err is not None
    # Some flavor of "connection refused" / "actively refused".
    assert "rpc-server probe:" in err


# --- Phase 2 commit 9: per-worker hardware capability capture -------------


def test_probe_records_gpu_present_when_loaded_model_in_vram(isolated_db, monkeypatch):
    """When /api/ps reports a model with size_vram>0, the worker's
    capabilities must surface gpu_present=True. The router uses this
    to prefer GPU-equipped workers over CPU-only ones."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    wid = isolated_db.create_compute_worker(
        label="gpu-worker", address="g.local", transport="lan",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.5.4"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "gemma4:e4b", "details": {}},
            ]})
        if request.url.path == "/api/ps":
            # 6 GB loaded into VRAM — clear GPU signal.
            return httpx.Response(200, json={"models": [{
                "name": "gemma4:e4b",
                "size": 4500000000,
                "size_vram": 6442450944,  # 6 GB
            }]})
        return httpx.Response(404)

    _install_mock_http(monkeypatch, handler, rpc_reachable=False)
    result = _run(compute_pool.probe_worker(wid))
    caps = result["capabilities"]
    assert caps["gpu_present"] is True
    assert caps["max_vram_seen_bytes"] == 6442450944
    assert caps["loaded_count"] == 1


def test_probe_records_no_gpu_when_ps_empty(isolated_db, monkeypatch):
    """A worker with nothing currently loaded gives us no hardware
    signal — gpu_present must be False (we treat absence of evidence
    as 'unknown, treat as no-GPU' so CPU-only workers don't get
    artificially preferred)."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    wid = isolated_db.create_compute_worker(
        label="idle", address="i.local", transport="lan",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.5.4"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "gemma4:e4b", "details": {}},
            ]})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(404)

    _install_mock_http(monkeypatch, handler, rpc_reachable=False)
    result = _run(compute_pool.probe_worker(wid))
    caps = result["capabilities"]
    assert caps["gpu_present"] is False
    assert caps["max_vram_seen_bytes"] == 0
    assert caps["loaded_count"] == 0


def test_capability_score_orders_gpu_above_no_gpu():
    """The router's ranking key must treat GPU-equipped workers as
    strictly more powerful than CPU-only ones (within the same
    eligibility class)."""
    cpu_only = {
        "id": "a", "last_seen": 9999.0,
        "capabilities": {"gpu_present": False, "max_vram_seen_bytes": 0},
    }
    gpu_worker = {
        "id": "b", "last_seen": 1.0,
        "capabilities": {"gpu_present": True, "max_vram_seen_bytes": 4 * 1024 ** 3},
    }
    # GPU should sort first even though its last_seen is older.
    sorted_workers = sorted(
        [cpu_only, gpu_worker], key=compute_pool._capability_score, reverse=True,
    )
    assert sorted_workers[0]["id"] == "b"


def test_capability_score_orders_more_vram_above_less():
    small = {
        "id": "small", "last_seen": 9999.0,
        "capabilities": {"gpu_present": True, "max_vram_seen_bytes": 4 * 1024 ** 3},
    }
    big = {
        "id": "big", "last_seen": 1.0,
        "capabilities": {"gpu_present": True, "max_vram_seen_bytes": 24 * 1024 ** 3},
    }
    sorted_workers = sorted(
        [small, big], key=compute_pool._capability_score, reverse=True,
    )
    assert sorted_workers[0]["id"] == "big"


def test_pick_chat_target_picks_gpu_worker_over_cpu_only(isolated_db, monkeypatch):
    """End-to-end: with two eligible chat workers, the GPU one must
    win the route. Maps to the user's 'prefer the more powerful
    worker' intent."""
    import time
    monkeypatch.setattr(compute_pool, "db", isolated_db)

    # CPU-only worker, fresher probe.
    cpu_id = isolated_db.create_compute_worker(
        label="cpu", address="cpu.local", transport="lan", use_for_chat=True,
    )
    isolated_db.update_compute_worker_capabilities(
        cpu_id,
        capabilities={
            "version": "0.5.4",
            "models": [{"name": "gemma4:e4b", "details": {}}],
            "rpc_server_reachable": True, "rpc_port": 50052, "rpc_error": None,
            "gpu_present": False, "max_vram_seen_bytes": 0, "loaded_count": 0,
        },
        last_seen=time.time() - 5.0, last_error="",
    )
    # GPU worker, slightly older probe.
    gpu_id = isolated_db.create_compute_worker(
        label="gpu", address="gpu.local", transport="lan", use_for_chat=True,
    )
    isolated_db.update_compute_worker_capabilities(
        gpu_id,
        capabilities={
            "version": "0.5.4",
            "models": [{"name": "gemma4:e4b", "details": {}}],
            "rpc_server_reachable": True, "rpc_port": 50052, "rpc_error": None,
            "gpu_present": True, "max_vram_seen_bytes": 4 * 1024 ** 3, "loaded_count": 1,
        },
        last_seen=time.time() - 30.0, last_error="",
    )

    target = compute_pool.pick_chat_target("gemma4:e4b")
    assert target is not None
    assert target[0] == "http://gpu.local:11434"
