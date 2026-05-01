"""Regression: P2P pool inventory cache + public-pool routing wiring.

Covers the new "rendezvous = bootstrap only" architecture:
  * Bootstrap discovery: `lookup_peers()` cross-checks the
    device_id ↔ pubkey binding before accepting any rendezvous
    response — a malicious rendezvous can't substitute identities.
  * Inventory cache: `find_peers_with_model` and `list_all_models`
    only return peers whose model inventory was polled within the
    TTL; stale entries are silently dropped from results.
  * Discovered-peer fallback: `get_discovered_peer` returns a row
    shaped to be drop-in compatible with `db.get_paired_device` so
    the secure-proxy verify path doesn't need to special-case it.
  * Per-role whitelist: discovered peers can only call read-only
    metadata endpoints (/api/tags, /api/show, /api/ps); /api/chat
    and /api/embed are refused even with a valid signature.
  * Public-pool routing: `ensure_public_peer_worker` reads from the
    LOCAL inventory cache (not the rendezvous), seeds the new
    worker's capabilities.models so it's immediately routable, and
    is idempotent across repeated calls.
"""
from __future__ import annotations

import asyncio
import base64
import time

import pytest

from backend import (
    identity,
    p2p_crypto,
    p2p_pool_inventory,
    p2p_pool_routing,
    p2p_rendezvous,
    p2p_secure_proxy,
)


pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_peer(device_id: str = "GLAB1234567890AB") -> dict:
    """Build a peer record shaped like what `lookup_peers` returns.

    The returned record has matching device_id ↔ pubkey so the
    cross-check inside lookup_peers wouldn't reject it.
    """
    # Generate a real Ed25519 keypair so the device_id check passes.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    from cryptography.hazmat.primitives.asymmetric.x25519 import (
        X25519PrivateKey,
    )
    from cryptography.hazmat.primitives import serialization

    ed_priv = Ed25519PrivateKey.generate()
    ed_pub_bytes = ed_priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    pub_b64 = base64.b64encode(ed_pub_bytes).decode("ascii")
    real_did = (
        base64.b32encode(ed_pub_bytes)
        .decode("ascii").rstrip("=")[:16].upper()
    )
    x_priv = X25519PrivateKey.generate()
    x_pub_b64 = base64.b64encode(
        x_priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ),
    ).decode("ascii")

    return {
        "device_id": real_did,
        "public_key_b64": pub_b64,
        "x25519_public_b64": x_pub_b64,
        "candidates": [{"ip": "203.0.113.5", "port": 8000, "source": "stun"}],
        "last_seen_at": time.time(),
    }


# ---------------------------------------------------------------------------
# Bootstrap discovery — device_id ↔ pubkey cross-check
# ---------------------------------------------------------------------------


def test_lookup_peers_rejects_substituted_identity(monkeypatch):
    """A peer entry whose device_id doesn't match base32(pubkey)[:16]
    must be rejected — defends against a malicious rendezvous trying
    to substitute identities.
    """
    real = _make_fake_peer()
    # Tamper: keep the (real) pubkey, but lie about the device_id.
    tampered = dict(real)
    tampered["device_id"] = "AAAAAAAAAAAAAAAA"

    class _FakeResponse:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {"peers": [real, tampered]}

    class _FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            return _FakeResponse()

    monkeypatch.setattr(p2p_rendezvous, "_current_rendezvous_url", lambda: "https://x")
    monkeypatch.setattr(p2p_rendezvous.httpx, "AsyncClient", _FakeClient)

    out = asyncio.run(p2p_rendezvous.lookup_peers())
    assert len(out) == 1
    assert out[0]["device_id"] == real["device_id"]


def test_lookup_peers_returns_empty_when_url_unset(monkeypatch):
    monkeypatch.setattr(p2p_rendezvous, "_current_rendezvous_url", lambda: "")
    out = asyncio.run(p2p_rendezvous.lookup_peers())
    assert out == []


def test_lookup_peers_handles_404_gracefully(monkeypatch):
    """An old rendezvous deployment without /peers should degrade
    gracefully — empty list, no exception."""
    class _FakeResponse:
        status_code = 404
        def raise_for_status(self):
            raise RuntimeError("should not be called for 404")

    class _FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            return _FakeResponse()

    monkeypatch.setattr(p2p_rendezvous, "_current_rendezvous_url", lambda: "https://x")
    monkeypatch.setattr(p2p_rendezvous.httpx, "AsyncClient", _FakeClient)

    out = asyncio.run(p2p_rendezvous.lookup_peers())
    assert out == []


# ---------------------------------------------------------------------------
# Inventory cache — find / list / TTL
# ---------------------------------------------------------------------------


def test_find_peers_with_model_returns_only_fresh_entries(monkeypatch):
    p2p_pool_inventory.clear_cache()
    fresh = _make_fake_peer()
    fresh["models"] = [{"name": "llama3:8b", "size_bytes": 4_000_000_000}]
    fresh["models_fetched_at"] = time.time()
    p2p_pool_inventory._peers[fresh["device_id"]] = fresh

    stale = _make_fake_peer()
    stale["models"] = [{"name": "llama3:8b", "size_bytes": 4_000_000_000}]
    stale["models_fetched_at"] = time.time() - 99_999  # well past TTL
    p2p_pool_inventory._peers[stale["device_id"]] = stale

    found = p2p_pool_inventory.find_peers_with_model("llama3:8b")
    found_ids = {p["device_id"] for p in found}
    assert fresh["device_id"] in found_ids
    assert stale["device_id"] not in found_ids


def test_list_all_models_skips_stale_peers():
    p2p_pool_inventory.clear_cache()
    fresh = _make_fake_peer()
    fresh["models"] = [{
        "name": "qwen3:32b", "family": "qwen",
        "parameter_size": "32B", "quantization_level": "Q4_K_M",
        "size_bytes": 18_000_000_000,
    }]
    fresh["models_fetched_at"] = time.time()
    p2p_pool_inventory._peers[fresh["device_id"]] = fresh

    out = p2p_pool_inventory.list_all_models()
    assert len(out) == 1
    entry = out[0]
    assert entry["name"] == "qwen3:32b"
    assert entry["source_device_id"] == fresh["device_id"]
    # All public-pool routes are E2E encrypted, so the badge claim is fixed.
    assert entry["encrypted"] is True


def test_get_discovered_peer_returns_pairable_shape():
    """The fallback record returned to the secure-proxy verify path
    must look like a paired_device row so the proxy doesn't need a
    special case."""
    p2p_pool_inventory.clear_cache()
    peer = _make_fake_peer()
    p2p_pool_inventory._peers[peer["device_id"]] = peer

    rec = p2p_pool_inventory.get_discovered_peer(peer["device_id"])
    assert rec is not None
    # The fields the secure proxy reads.
    assert rec["device_id"] == peer["device_id"]
    assert rec["public_key_b64"] == peer["public_key_b64"]
    assert rec["x25519_public_b64"] == peer["x25519_public_b64"]
    assert rec["role"] == "discovered"


def test_get_discovered_peer_unknown_returns_none():
    p2p_pool_inventory.clear_cache()
    assert p2p_pool_inventory.get_discovered_peer("UNKNOWN") is None


# ---------------------------------------------------------------------------
# Secure proxy — per-role whitelist
# ---------------------------------------------------------------------------


def test_secure_proxy_whitelist_paths_unchanged_for_paired():
    # Paired peers (LAN or 'public') get the full whitelist plus the
    # newly-added /api/pull endpoint for auto-pull.
    assert "/api/chat" in p2p_secure_proxy._FORWARDABLE_PATHS
    assert "/api/embed" in p2p_secure_proxy._FORWARDABLE_PATHS
    assert "/api/embeddings" in p2p_secure_proxy._FORWARDABLE_PATHS
    assert "/api/tags" in p2p_secure_proxy._FORWARDABLE_PATHS
    assert "/api/pull" in p2p_secure_proxy._FORWARDABLE_PATHS


def test_secure_proxy_whitelist_discovered_is_read_only():
    # Discovered peers can only read metadata. /api/chat etc. is refused.
    assert "/api/tags" in p2p_secure_proxy._DISCOVERED_PEER_PATHS
    assert "/api/show" in p2p_secure_proxy._DISCOVERED_PEER_PATHS
    assert "/api/ps" in p2p_secure_proxy._DISCOVERED_PEER_PATHS
    assert "/api/chat" not in p2p_secure_proxy._DISCOVERED_PEER_PATHS
    assert "/api/embed" not in p2p_secure_proxy._DISCOVERED_PEER_PATHS
    assert "/api/embeddings" not in p2p_secure_proxy._DISCOVERED_PEER_PATHS
    assert "/api/pull" not in p2p_secure_proxy._DISCOVERED_PEER_PATHS


# ---------------------------------------------------------------------------
# Public-pool routing — ensure_public_peer_worker
# ---------------------------------------------------------------------------


def test_ensure_public_peer_worker_disabled_returns_none(isolated_db, monkeypatch):
    isolated_db.set_setting("p2p_public_pool_enabled", "0")
    out = asyncio.run(p2p_pool_routing.ensure_public_peer_worker("llama3:8b"))
    assert out is None


def test_ensure_public_peer_worker_no_peers_returns_none(isolated_db, monkeypatch):
    isolated_db.set_setting("p2p_public_pool_enabled", "1")
    p2p_pool_inventory.clear_cache()

    async def _no_refresh(**kw):
        return None
    monkeypatch.setattr(p2p_pool_inventory, "ensure_fresh", _no_refresh)

    out = asyncio.run(p2p_pool_routing.ensure_public_peer_worker("absent"))
    assert out is None


def test_ensure_public_peer_worker_creates_and_seeds_capabilities(
    isolated_db, monkeypatch,
):
    """End-to-end happy path: peer in cache → paired_devices upsert →
    compute_workers row → capabilities.models seeded.

    The capability seed is what makes the very next routing call see
    the new worker as eligible (without waiting for the periodic
    probe loop).
    """
    isolated_db.set_setting("p2p_public_pool_enabled", "1")
    p2p_pool_inventory.clear_cache()

    peer = _make_fake_peer()
    peer["models"] = [{
        "name": "qwen3:32b",
        "family": "qwen",
        "parameter_size": "32B",
        "quantization_level": "Q4_K_M",
        "size_bytes": 18_000_000_000,
    }]
    peer["models_fetched_at"] = time.time()
    p2p_pool_inventory._peers[peer["device_id"]] = peer

    async def _no_refresh(**kw):
        return None
    monkeypatch.setattr(p2p_pool_inventory, "ensure_fresh", _no_refresh)

    worker = asyncio.run(
        p2p_pool_routing.ensure_public_peer_worker("qwen3:32b"),
    )
    assert worker is not None
    assert worker["use_encrypted_proxy"] is True
    assert worker["gigachat_device_id"] == peer["device_id"]
    # The seed must include the model that triggered the registration
    # so `_worker_has_model` returns True immediately.
    caps = worker.get("capabilities") or {}
    seeded_names = {m.get("name") for m in caps.get("models") or []}
    assert "qwen3:32b" in seeded_names

    # Idempotency: a second call with the same model returns the same
    # worker (refreshes address, preserves capabilities) — does NOT
    # create a duplicate row.
    worker2 = asyncio.run(
        p2p_pool_routing.ensure_public_peer_worker("qwen3:32b"),
    )
    assert worker2 is not None
    assert worker2["id"] == worker["id"]
