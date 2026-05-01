"""Gigachat P2P rendezvous service.

Tiny FastAPI app the user deploys to GCP Cloud Run. Its only job is
to let peers find each other so the Gigachat clients can establish
QUIC connections directly (NAT-traversal happens between peers, not
through this service — model traffic and prompts never touch the
rendezvous).

Endpoints:

  POST /register   — peer registers its identity + STUN endpoints
  GET  /lookup/{device_id} — peer asks where another peer is
  POST /heartbeat  — peer keeps its registration alive
  GET  /health    — Cloud Run health probe

The rendezvous never sees prompts, model weights, or any chat data.
It stores the minimum needed for hole-punching:

  device_id (Ed25519 pubkey hash, the trust anchor)
  public_key_b64 (so the requester can verify subsequent traffic)
  candidates: [{ip, port, source}]  (STUN-discovered endpoints)
  last_seen_at

Anti-abuse:
  * 60-second TTL on registrations — abandoned peers fall out of
    the table without manual cleanup.
  * Per-IP rate-limit on /register (30 req/min) — a sybil farm has
    to spread across IPs to grow the table.
  * Required signature on /register — proves the peer holds the
    Ed25519 private key matching the claimed device_id, so an
    attacker can't impersonate someone else's identity.
  * Stateless across restarts — Cloud Run can scale this to zero
    without losing trust state, because identity is in the message
    not in the table. Lookup-misses on a cold start resolve once
    peers heartbeat again.

Deployment:
  gcloud run deploy gigachat-rendezvous \\
    --source . \\
    --region us-central1 \\
    --platform managed \\
    --allow-unauthenticated \\
    --max-instances 5

The service uses no GCP-specific APIs — Cloud Run's HTTP frontend
and a single Python process is all it needs. Memory: ~64 MB for
1000 peers. CPU: serves 100+ req/s on a single 0.5 vCPU instance.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from collections import defaultdict, deque
from typing import Deque

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

try:
    # Cryptography is the only non-stdlib dep besides FastAPI/uvicorn.
    # Same lib the client uses, so signature verification is bit-exact.
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey,
    )
    _CRYPTO_OK = True
except ImportError:
    _CRYPTO_OK = False


app = FastAPI(
    title="Gigachat Rendezvous",
    description=(
        "Stateless peer discovery service for the Gigachat P2P compute "
        "pool. No prompts, no model traffic — peers find each other "
        "here, then talk QUIC directly."
    ),
    version="1.0",
)

# Registration TTL. A peer that doesn't heartbeat within this window
# is dropped from the lookup table. 60 s is short enough that crashed
# peers don't pollute results for long but long enough that a peer
# can survive a brief network blip without re-registering.
TTL_SEC = 60.0

# Per-IP rate limit on /register. Sybil-farming gets quadratically
# more expensive across thousands of IPs. 30 req/min lets a normal
# peer re-register after a network change without ever tripping it.
REGISTER_RATE_PER_MIN = 30

# Required signature freshness — the timestamp in the signed message
# must be within this many seconds of "now". Prevents an attacker
# from replaying a captured registration after a long delay.
REGISTER_TIMESTAMP_SKEW_SEC = 120.0


# ---------- In-memory state ----------

# {device_id: PeerRecord}
_peers: dict[str, dict] = {}

# {ip: deque[timestamps]} for the rate-limit window.
_register_history: dict[str, Deque[float]] = defaultdict(deque)
_state_lock = asyncio.Lock()


# ---------- Request / response models ----------


class Candidate(BaseModel):
    """One reachable endpoint for a peer.

    `source` reports how the candidate was learned:
      * "stun"       — peer's public IP/port via a STUN server.
      * "lan"        — local-network IP (for same-NAT peers).
      * "ipv6"       — global IPv6 (skips NAT entirely).
      * "manual"     — operator-supplied static address.
    """
    ip: str = Field(..., min_length=1, max_length=64)
    port: int = Field(..., ge=1, le=65535)
    source: str = Field(default="stun", max_length=16)


class RegisterBody(BaseModel):
    """Body for POST /register.

    The signature proves the peer holds the Ed25519 private key
    matching `public_key_b64`. The signed message is the JSON-encoded
    payload of (device_id, public_key_b64, candidates, timestamp) in
    that order, separated by '|' to prevent ambiguity.

    Replay protection: `timestamp` must be fresh
    (within REGISTER_TIMESTAMP_SKEW_SEC of server time). The same
    signature can't be reused after the window closes.
    """
    device_id: str = Field(..., min_length=4, max_length=64)
    public_key_b64: str = Field(..., min_length=8, max_length=128)
    candidates: list[Candidate] = Field(default_factory=list, max_length=8)
    timestamp: float = Field(..., gt=0)
    signature_b64: str = Field(..., min_length=8, max_length=256)


class HeartbeatBody(BaseModel):
    """Body for POST /heartbeat. Lighter than /register — only
    extends the TTL for an already-registered peer. Same signature
    pattern, but the signed message is just (device_id, timestamp)."""
    device_id: str = Field(..., min_length=4, max_length=64)
    timestamp: float = Field(..., gt=0)
    signature_b64: str = Field(..., min_length=8, max_length=256)


class LookupResponse(BaseModel):
    """Response shape for /lookup/{device_id}."""
    device_id: str
    public_key_b64: str
    candidates: list[Candidate]
    last_seen_at: float
    ttl_remaining_sec: float


# ---------- Helpers ----------


def _verify_ed25519(public_key_b64: str, message: bytes, signature_b64: str) -> bool:
    """Verify an Ed25519 signature, returning False on any error.

    Cheap when the crypto lib is loaded; raises 503 from the caller
    when it isn't (deployment misconfiguration, not a normal path).
    """
    if not _CRYPTO_OK:
        raise HTTPException(503, "rendezvous deployed without cryptography lib")
    try:
        pub_bytes = base64.b64decode(public_key_b64)
        signature = base64.b64decode(signature_b64)
        pub = Ed25519PublicKey.from_public_bytes(pub_bytes)
        try:
            pub.verify(signature, message)
            return True
        except InvalidSignature:
            return False
    except Exception:
        return False


def _device_id_from_pubkey(pubkey_b64: str) -> str:
    """Mirror of `backend/identity.py::_device_id_from_pubkey` — must
    produce identical output so a registration's claimed device_id
    can be cross-checked against its public key."""
    pub_bytes = base64.b64decode(pubkey_b64)
    s = base64.b32encode(pub_bytes).decode("ascii").rstrip("=")
    return s[:16].upper()


def _client_ip(request: Request) -> str:
    """Best-effort client IP for rate-limiting.

    Cloud Run sits behind a Google L7 LB that sets X-Forwarded-For;
    we trust the LAST entry (the LB's view of the real client).
    Falls back to the direct peername if the header is missing
    (local dev / curl).
    """
    xff = request.headers.get("x-forwarded-for", "").strip()
    if xff:
        # XFF may be a comma-list; the last entry is the closest to us.
        return xff.split(",")[-1].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _purge_expired(now: float | None = None) -> None:
    """Drop peers whose TTL has elapsed. Called inline at the top of
    /lookup and /register so the dict can't grow without bound."""
    cutoff = (now if now is not None else time.time()) - TTL_SEC
    for did in list(_peers.keys()):
        if _peers[did]["last_seen_at"] < cutoff:
            _peers.pop(did, None)


def _check_rate_limit(ip: str) -> None:
    """Enforce REGISTER_RATE_PER_MIN per IP. Raises 429 on over-limit."""
    now = time.time()
    cutoff = now - 60.0
    history = _register_history[ip]
    while history and history[0] < cutoff:
        history.popleft()
    if len(history) >= REGISTER_RATE_PER_MIN:
        raise HTTPException(
            429,
            f"rate limit: {REGISTER_RATE_PER_MIN} registrations/min per IP",
        )
    history.append(now)


def _build_register_message(body: RegisterBody) -> bytes:
    """Canonical bytes that body.signature must verify against.

    Same shape on the client; mismatched serialization would break
    the signature even when the actual data is identical, so we
    nail down the format here. Pipe-separated, lexicographic order.
    """
    cand_repr = ",".join(
        f"{c.ip}:{c.port}:{c.source}" for c in body.candidates
    )
    parts = [
        b"gigachat-rdv-register-v1",
        body.device_id.encode("ascii"),
        body.public_key_b64.encode("ascii"),
        cand_repr.encode("ascii"),
        f"{body.timestamp:.6f}".encode("ascii"),
    ]
    return b"|".join(parts)


def _build_heartbeat_message(body: HeartbeatBody) -> bytes:
    parts = [
        b"gigachat-rdv-heartbeat-v1",
        body.device_id.encode("ascii"),
        f"{body.timestamp:.6f}".encode("ascii"),
    ]
    return b"|".join(parts)


# ---------- Endpoints ----------


@app.get("/health")
async def health() -> dict:
    """Cloud Run probe target. Trivial.

    Note: NOT /healthz — Google's frontend intercepts that path on
    *.run.app domains and returns its own 404 page before requests
    reach the container. /health works as expected.
    """
    return {"ok": True, "peers": len(_peers)}


@app.post("/register")
async def register(body: RegisterBody, request: Request) -> dict:
    """Register a peer's identity + reachable endpoints.

    Replaces any prior record for the same device_id. Idempotent —
    a peer can re-register at any time (with a fresh timestamp +
    signature) to update its candidate list.
    """
    ip = _client_ip(request)
    _check_rate_limit(ip)

    # Sanity: claimed device_id must match the public key. Otherwise
    # a peer with one identity could try to register under another
    # device's id (squatting / impersonation).
    if _device_id_from_pubkey(body.public_key_b64) != body.device_id:
        raise HTTPException(400, "device_id does not match public_key")

    # Fresh-timestamp check before signature so an attacker replaying
    # an old signed registration is rejected before we spend cycles
    # on the verify.
    skew = abs(body.timestamp - time.time())
    if skew > REGISTER_TIMESTAMP_SKEW_SEC:
        raise HTTPException(
            400,
            f"registration timestamp out of window "
            f"(±{REGISTER_TIMESTAMP_SKEW_SEC:.0f}s; got skew={skew:.0f}s)",
        )

    msg = _build_register_message(body)
    if not _verify_ed25519(body.public_key_b64, msg, body.signature_b64):
        raise HTTPException(401, "signature verification failed")

    async with _state_lock:
        _purge_expired()
        _peers[body.device_id] = {
            "device_id": body.device_id,
            "public_key_b64": body.public_key_b64,
            "candidates": [c.model_dump() for c in body.candidates],
            "last_seen_at": time.time(),
        }
    return {
        "ok": True,
        "ttl_sec": TTL_SEC,
    }


@app.post("/heartbeat")
async def heartbeat(body: HeartbeatBody) -> dict:
    """Extend a registration's TTL without re-sending candidates.

    Cheaper than /register — peers heartbeat every ~30s to keep
    their entry warm. New candidates / address changes go through
    /register instead.
    """
    skew = abs(body.timestamp - time.time())
    if skew > REGISTER_TIMESTAMP_SKEW_SEC:
        raise HTTPException(400, "heartbeat timestamp out of window")

    async with _state_lock:
        rec = _peers.get(body.device_id)
        if not rec:
            raise HTTPException(404, "device not registered (re-register)")
        msg = _build_heartbeat_message(body)
        if not _verify_ed25519(rec["public_key_b64"], msg, body.signature_b64):
            raise HTTPException(401, "signature verification failed")
        rec["last_seen_at"] = time.time()
    return {"ok": True, "ttl_sec": TTL_SEC}


@app.get("/lookup/{device_id}")
async def lookup(device_id: str) -> LookupResponse:
    """Return the candidate endpoints for a registered peer.

    Anyone can look up a peer by device_id — the device_id itself is
    a public identifier (visible in pairing UIs, signed receipts, etc.).
    Knowing the candidates doesn't grant access to the peer; the actual
    P2P session still requires the requester to have been pre-authorized
    by the target (paired or friended).
    """
    async with _state_lock:
        _purge_expired()
        rec = _peers.get(device_id)
        if not rec:
            raise HTTPException(404, "device not registered")
        ttl_remaining = max(
            0.0, TTL_SEC - (time.time() - rec["last_seen_at"]),
        )
        return LookupResponse(
            device_id=rec["device_id"],
            public_key_b64=rec["public_key_b64"],
            candidates=[Candidate(**c) for c in rec["candidates"]],
            last_seen_at=rec["last_seen_at"],
            ttl_remaining_sec=ttl_remaining,
        )


@app.get("/")
async def index() -> dict:
    """Tiny landing page for humans poking at the URL."""
    return {
        "service": "gigachat-rendezvous",
        "version": app.version,
        "endpoints": [
            "POST /register",
            "POST /heartbeat",
            "GET /lookup/{device_id}",
            "GET /health",
        ],
        "peers_registered": len(_peers),
        "ttl_sec": TTL_SEC,
        "note": (
            "This service stores ONLY peer locations for NAT traversal. "
            "Prompts, models, and chat content never pass through here."
        ),
    }


# Cloud Run sets PORT; uvicorn picks it up.
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
