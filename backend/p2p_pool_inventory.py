"""Local cache of "what models does each peer in the public swarm have".

Why this module exists
======================
The rendezvous server's job is **bootstrap discovery only** — it tells us
who's online, where to reach them, and (crucially) their pubkey. It does
NOT track what each peer offers; that would centralise the "who has what
model" graph on a single server, which is exactly the centralisation a
P2P app should avoid.

Instead, every peer maintains its OWN view of the swarm:

  1. Periodically GET ``/peers`` from the rendezvous → list of online
     peers (identity + STUN candidates).
  2. For each known peer, periodically POST a sealed envelope with
     ``method=GET, path=/api/tags`` to their secure-proxy → that peer's
     model list, encrypted end-to-end.
  3. Store both in this in-memory cache with a TTL.
  4. ``find_peers_with_model(name)`` answers from the cache — zero
     extra network calls during chat routing.

The cache is in-memory (no DB writes per refresh), backed up by the
secure proxy's authenticated channel so a malicious rendezvous can't
poison our inventory: even if the rendezvous lies and inserts a fake
peer in the /peers list, we cross-check ``device_id == hash(pubkey)``
before accepting them, and the encrypted envelope to ``/api/tags``
verifies against THAT pubkey — so an attacker who only controls the
rendezvous can't read or fake responses.

Discovered-peer trust gate
==========================
A peer in this cache is "discovered, not paired": we trust their
identity (pubkey ↔ device_id checked) but we don't allow them to use
our compute. The secure proxy enforces this by reading the cache as a
fall-back lookup for envelope verification, with a TIGHTER whitelist
than for paired peers — discovered peers can only call read-only
metadata endpoints (``/api/tags``, ``/api/show``, ``/api/ps``). They
can't drive ``/api/chat`` or ``/api/embed`` until the user explicitly
picks one of their models, which promotes them to ``role='public'``
in the paired_devices table.

Refresh cadence
===============
* Peer list (``/peers``): every 60 s. Cheap call (one HTTP GET to the
  rendezvous), keeps us in sync as peers join/leave.
* Per-peer model inventory: every 5 min, staggered. Querying every
  peer's /api/tags constantly would saturate small home-internet
  uploads on the swarm. 5 min is fast enough that "I just installed
  this model" appears in other people's pickers within minutes.
* On-demand: ``ensure_fresh()`` forces a refresh older than the
  caller-specified age. The model picker uses this so opening the
  dropdown gets the latest inventory if it's been a while.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from . import db, p2p_crypto, p2p_rendezvous

log = logging.getLogger("gigachat.p2p.pool_inventory")

# Per-peer model-inventory TTL. After this long without a successful
# /api/tags poll, we drop the model list (the peer entry stays in the
# cache; only the models go stale). Keeps the picker honest about
# which peers actually have anything to offer.
_INVENTORY_TTL_SEC = 600.0  # 10 min

# Refresh peer LIST (from rendezvous /peers) at this cadence. Cheap
# call, mostly bounded by rendezvous latency. Important to be faster
# than the rendezvous's own peer TTL (60 s) so we don't miss peers
# that just joined.
_PEER_LIST_REFRESH_SEC = 60.0

# Refresh per-peer MODEL inventory at this cadence. Bigger interval
# — calling every peer's /api/tags constantly would be rude on a
# large swarm. 5 min is fast enough for "I just pulled X" to show
# up everywhere within the same coffee break.
_INVENTORY_REFRESH_SEC = 300.0

# Cap on parallel /api/tags polls to avoid hammering all peers at
# once when the loop fires. Spread the cost both for ourselves
# (file descriptors) and for them (polled concurrently).
_INVENTORY_PARALLELISM = 8

# Per-peer /api/tags timeout. This is to a peer's secure proxy so
# it includes their crypto roundtrip + their local Ollama call.
# 6 s is generous; faster timeout keeps a single dead peer from
# stalling the refresh round.
_INVENTORY_PEER_TIMEOUT_SEC = 6.0

# Hard cap on cached peers — defense against a runaway rendezvous
# returning huge peer lists. 10k is enormous for a P2P swarm; if
# we ever get close we have bigger problems than memory.
_MAX_CACHED_PEERS = 10_000


# ---------------------------------------------------------------------------
# Cache state
# ---------------------------------------------------------------------------

# {device_id: PeerEntry}
#
# PeerEntry shape (kept as a plain dict for trivial JSON-ability and
# zero allocator pressure on hot paths):
#   {
#     "device_id": str,
#     "public_key_b64": str,        # Ed25519 (signing) — trust anchor
#     "x25519_public_b64": str,     # X25519 (encryption)
#     "candidates": [{ip, port, source}, ...],
#     "models": [<ModelEntry-shaped dict>],  # filled by inventory poll
#     "models_fetched_at": float,   # epoch seconds; 0 if never fetched
#     "last_seen_via_rendezvous": float,
#     "last_inventory_error": str,  # debug only — surfaced in /status
#   }
_peers: dict[str, dict] = {}
_peers_lock = asyncio.Lock()

# Background refresh task handle (None when not running).
_loop_task: asyncio.Task | None = None
_loop_stop_event: asyncio.Event | None = None


# ---------------------------------------------------------------------------
# Public read API
# ---------------------------------------------------------------------------

def get_discovered_peer(device_id: str) -> dict | None:
    """Look up a discovered peer's record by device_id.

    Used by the secure proxy as a fallback when an envelope arrives
    from a peer that isn't in `paired_devices`. The returned dict is
    shaped to be drop-in compatible with `db.get_paired_device(...)`
    so the verify path doesn't need to special-case it.

    Returns None when the peer isn't in our cache (dropped from the
    rendezvous, never seen, or cache cleared).
    """
    if not device_id:
        return None
    rec = _peers.get(device_id)
    if not rec:
        return None
    # Mirror the paired_device row shape — only the fields the secure
    # proxy reads.
    return {
        "device_id": rec["device_id"],
        "public_key_b64": rec["public_key_b64"],
        "x25519_public_b64": rec.get("x25519_public_b64"),
        "label": (rec.get("device_id") or "")[:16],
        # Distinct role so the proxy can pick a tighter whitelist for
        # discovered peers vs. paired peers.
        "role": "discovered",
    }


def find_peers_with_model(model_name: str) -> list[dict]:
    """Return cached peer records that advertise ``model_name``.

    Reads ONLY from the local cache — no network call. The model
    picker / routing layer use this to decide which public-pool peer
    to dispatch a chat to.

    Returned dicts are full peer records (device_id, pubkeys,
    candidates, models). Caller picks one (typically the first) and
    materialises it as a compute_worker via
    `p2p_pool_routing.ensure_public_peer_worker`.

    Skips entries whose model inventory has gone stale (older than
    `_INVENTORY_TTL_SEC`) — those peers might have uninstalled the
    model since we last polled.
    """
    if not model_name:
        return []
    now = time.time()
    out: list[dict] = []
    for rec in _peers.values():
        if (now - (rec.get("models_fetched_at") or 0)) > _INVENTORY_TTL_SEC:
            continue
        for m in rec.get("models") or []:
            if m.get("name") == model_name:
                out.append(rec)
                break
    return out


def list_all_models() -> list[dict]:
    """Aggregated model list across every cached peer.

    Returns ``[{name, family, parameter_size, quantization_level,
    size_bytes, source_device_id, source_label, encrypted}, ...]`` —
    same shape `/api/models/all-sources` wants for its `public`
    section. Callers should already have filtered out stale peers
    (`models_fetched_at` older than TTL).

    Cheap: pure in-memory scan; safe to call on every model-picker
    open.
    """
    now = time.time()
    out: list[dict] = []
    for rec in _peers.values():
        if (now - (rec.get("models_fetched_at") or 0)) > _INVENTORY_TTL_SEC:
            continue
        did = rec.get("device_id") or ""
        for m in rec.get("models") or []:
            if not m.get("name"):
                continue
            out.append({
                "name": m["name"],
                "family": m.get("family"),
                "parameter_size": m.get("parameter_size"),
                "quantization_level": m.get("quantization_level"),
                "size_bytes": int(m.get("size_bytes") or m.get("size") or 0),
                "source_device_id": did,
                "source_label": did[:16],
                # Public-pool dispatch is always end-to-end encrypted
                # via the secure proxy — the badge in the picker says so.
                "encrypted": True,
            })
    return out


def status() -> dict:
    """Snapshot of inventory state for the diagnostics endpoint.

    Cheap pure-read; safe to call on every /api/p2p/status hit.
    """
    now = time.time()
    fresh = sum(
        1 for rec in _peers.values()
        if (now - (rec.get("models_fetched_at") or 0)) <= _INVENTORY_TTL_SEC
    )
    return {
        "running": _loop_task is not None and not _loop_task.done(),
        "peer_count": len(_peers),
        "peers_with_fresh_inventory": fresh,
        "inventory_ttl_sec": _INVENTORY_TTL_SEC,
    }


# ---------------------------------------------------------------------------
# Refresh logic
# ---------------------------------------------------------------------------

async def refresh_peer_list() -> int:
    """Pull the current peer list from the rendezvous and merge into
    the cache.

    Adds new peers (with empty model lists; the inventory loop fills
    them in). Updates existing peers' candidates / last_seen. Drops
    peers that have aged out of the rendezvous beyond a generous
    grace period (so a brief rendezvous outage doesn't wipe our
    inventory).

    Returns the new peer-count.
    """
    raw = await p2p_rendezvous.lookup_peers()
    now = time.time()
    seen_ids: set[str] = set()
    async with _peers_lock:
        for entry in raw:
            did = entry.get("device_id") or ""
            if not did:
                continue
            seen_ids.add(did)
            existing = _peers.get(did)
            if existing:
                # Update only the bootstrap-side fields — keep model
                # inventory intact (it's owned by the inventory poll).
                existing["public_key_b64"] = entry.get("public_key_b64") or ""
                existing["x25519_public_b64"] = entry.get(
                    "x25519_public_b64"
                ) or ""
                existing["candidates"] = entry.get("candidates") or []
                existing["last_seen_via_rendezvous"] = now
            else:
                if len(_peers) >= _MAX_CACHED_PEERS:
                    # Already at the cap — refuse to grow further.
                    # Caller could log; cheaper to silently no-op.
                    continue
                _peers[did] = {
                    "device_id": did,
                    "public_key_b64": entry.get("public_key_b64") or "",
                    "x25519_public_b64": entry.get("x25519_public_b64") or "",
                    "candidates": entry.get("candidates") or [],
                    "models": [],
                    "models_fetched_at": 0.0,
                    "last_seen_via_rendezvous": now,
                    "last_inventory_error": "",
                }
        # Drop peers absent from the rendezvous for >2 grace periods.
        # Keeps stale entries out of the picker without flapping when
        # a peer briefly disappears between polls.
        cutoff = now - (_PEER_LIST_REFRESH_SEC * 3)
        for did in list(_peers.keys()):
            if did in seen_ids:
                continue
            if _peers[did].get("last_seen_via_rendezvous", 0) < cutoff:
                _peers.pop(did, None)
    return len(_peers)


async def _poll_one_peer_inventory(peer_rec: dict) -> None:
    """Send an encrypted ``GET /api/tags`` to one peer and store their
    model list in the cache.

    Errors are caught + logged; the cache entry's
    ``last_inventory_error`` field gets set so /status surfaces it.
    Empty model list (peer has no Ollama running) is treated as
    success — the cache reflects "I asked and they have nothing".
    """
    did = peer_rec.get("device_id") or ""
    x25519 = peer_rec.get("x25519_public_b64") or ""
    public_key = peer_rec.get("public_key_b64") or ""
    candidates = peer_rec.get("candidates") or []
    if not (did and x25519 and public_key and candidates):
        return
    # Pick the first STUN candidate (or LAN, then anything else).
    cand = next(
        (c for c in candidates if c.get("source") == "stun"),
        next(
            (c for c in candidates if c.get("source") == "lan"),
            candidates[0] if candidates else None,
        ),
    )
    if not cand:
        return
    ip = cand.get("ip")
    port = int(cand.get("port") or 0)
    if not ip or not port:
        return

    # Build sealed envelope addressed to this peer.
    try:
        envelope = p2p_crypto.seal_json(
            recipient_x25519_pub_b64=x25519,
            recipient_device_id=did,
            payload={
                "method": "GET",
                "path": "/api/tags",
                "body": None,
                "headers": {},
            },
        )
    except Exception as e:
        peer_rec["last_inventory_error"] = f"seal: {type(e).__name__}: {e}"
        return

    # Lazy import — httpx is heavy and we don't want to drag it into
    # the module import path (this module's API is otherwise pure).
    import httpx

    url = f"http://{ip}:{port}/api/p2p/secure/forward"
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_INVENTORY_PEER_TIMEOUT_SEC),
        ) as client:
            r = await client.post(url, json=envelope)
        if r.status_code >= 400:
            peer_rec["last_inventory_error"] = (
                f"http {r.status_code}: {r.text[:120]}"
            )
            return
        response_envelope = r.json()
    except Exception as e:
        peer_rec["last_inventory_error"] = (
            f"transport: {type(e).__name__}: {e}"
        )
        return

    # Decrypt the response and parse Ollama's /api/tags shape.
    try:
        payload, _ = p2p_crypto.open_envelope_json(
            response_envelope,
            expected_sender_ed25519_pub_b64=public_key,
        )
    except p2p_crypto.CryptoError as e:
        peer_rec["last_inventory_error"] = f"verify: {e}"
        return

    body_text = payload.get("body") or "{}"
    try:
        data = json.loads(body_text) if isinstance(body_text, str) else body_text
    except Exception as e:
        peer_rec["last_inventory_error"] = f"parse: {e}"
        return

    # Length-cap every field a malicious peer could exploit to bloat
    # our memory or break the picker UI. The secure proxy already
    # caps the whole response at 4 MB, but per-field caps keep one
    # bad peer from injecting a giant string into our cache.
    def _trim(val: Any, max_len: int) -> str | None:
        if val is None:
            return None
        s = str(val)
        return s[:max_len] if s else None

    out: list[dict] = []
    for m in (data or {}).get("models") or []:
        if not isinstance(m, dict):
            continue
        name = _trim(m.get("name"), 128)
        if not name:
            continue
        details = m.get("details") or {}
        if not isinstance(details, dict):
            details = {}
        try:
            size_bytes = int(m.get("size") or 0)
        except (TypeError, ValueError):
            size_bytes = 0
        size_bytes = max(0, min(size_bytes, 1 << 50))  # ≤ 1 PiB sanity cap
        out.append({
            "name": name,
            "family": _trim(details.get("family"), 32),
            "parameter_size": _trim(details.get("parameter_size"), 16),
            "quantization_level": _trim(details.get("quantization_level"), 16),
            "size_bytes": size_bytes,
        })
    # Cap at 200 to match the size we'd accept from /api/tags anywhere
    # else — defends against a runaway peer with an absurd model count.
    peer_rec["models"] = out[:200]
    peer_rec["models_fetched_at"] = time.time()
    peer_rec["last_inventory_error"] = ""


async def refresh_inventories(*, only_stale: bool = True) -> int:
    """Poll every cached peer for its current /api/tags.

    With ``only_stale=True`` (the default), skips peers polled within
    the last `_INVENTORY_REFRESH_SEC`. Pass ``False`` to force a
    full re-poll (used by ensure_fresh()).

    Concurrency is bounded by `_INVENTORY_PARALLELISM` — we don't
    open hundreds of sockets at once even on a large swarm.

    Returns the number of peers polled.
    """
    now = time.time()
    targets: list[dict] = []
    for rec in _peers.values():
        if only_stale:
            age = now - (rec.get("models_fetched_at") or 0)
            if age < _INVENTORY_REFRESH_SEC:
                continue
        targets.append(rec)
    if not targets:
        return 0
    sem = asyncio.Semaphore(_INVENTORY_PARALLELISM)

    async def _bounded(peer: dict) -> None:
        async with sem:
            await _poll_one_peer_inventory(peer)

    await asyncio.gather(
        *(_bounded(p) for p in targets),
        return_exceptions=True,
    )
    return len(targets)


async def ensure_fresh(*, max_age_sec: float = 60.0) -> None:
    """Force an inventory refresh if any peer's data is older than
    ``max_age_sec``.

    Cheap when everything's already fresh. The model picker calls
    this on dropdown-open so the visible inventory is current
    without waiting for the next periodic tick.
    """
    now = time.time()
    needs = any(
        (now - (rec.get("models_fetched_at") or 0)) > max_age_sec
        for rec in _peers.values()
    )
    if not needs:
        return
    # Refresh the peer list first if it itself is stale — otherwise
    # we'll poll inventory on stale-or-missing peers.
    await refresh_peer_list()
    await refresh_inventories(only_stale=False)


# ---------------------------------------------------------------------------
# Background loop lifecycle
# ---------------------------------------------------------------------------

async def _background_loop(stop_event: asyncio.Event) -> None:
    """Long-running task: refresh peer list at one cadence, model
    inventories at another, both off the same loop.

    Robust to individual failures — every iteration catches its own
    exceptions so a single bad peer can't derail the whole loop.
    Sleeps with `asyncio.wait` against the stop event so shutdown is
    near-instant even mid-cycle.
    """
    last_inventory_at = 0.0
    while not stop_event.is_set():
        # Peer list every loop iteration (cheap).
        try:
            n = await refresh_peer_list()
            if n > 0:
                log.debug("p2p_pool_inventory: %d peers in cache", n)
        except Exception as e:
            log.info("p2p_pool_inventory: peer-list refresh failed: %s", e)

        # Inventory poll on its own slower cadence.
        now = time.time()
        if (now - last_inventory_at) >= _INVENTORY_REFRESH_SEC:
            try:
                polled = await refresh_inventories(only_stale=False)
                if polled:
                    log.debug(
                        "p2p_pool_inventory: polled %d peer inventories",
                        polled,
                    )
                last_inventory_at = time.time()
            except Exception as e:
                log.info(
                    "p2p_pool_inventory: inventory refresh failed: %s", e,
                )

        # Sleep until next peer-list tick or shutdown signal.
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=_PEER_LIST_REFRESH_SEC,
            )
        except asyncio.TimeoutError:
            continue


async def start() -> None:
    """Kick off the background refresh loop.

    No-op if already running. Safe to call from FastAPI's lifespan
    startup hook.
    """
    global _loop_task, _loop_stop_event
    if _loop_task is not None and not _loop_task.done():
        return
    _loop_stop_event = asyncio.Event()
    _loop_task = asyncio.create_task(
        _background_loop(_loop_stop_event),
        name="p2p_pool_inventory_loop",
    )
    log.info("p2p_pool_inventory: background loop started")


async def stop() -> None:
    """Stop the background loop and wait for it to drain.

    Safe to call multiple times; idempotent.
    """
    global _loop_task, _loop_stop_event
    if _loop_stop_event is not None:
        _loop_stop_event.set()
    if _loop_task is not None:
        try:
            await asyncio.wait_for(_loop_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            _loop_task.cancel()
        except Exception:
            pass
    _loop_task = None
    _loop_stop_event = None


def clear_cache() -> None:
    """Wipe the cache. Used by tests + on Public Pool toggle off so
    we don't keep stale "discovered peer" records that grant the
    secure proxy permission to read our /api/tags."""
    _peers.clear()
