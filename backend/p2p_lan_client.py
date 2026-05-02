"""HTTP client for talking to peer Gigachat instances on the LAN.

Used by the symmetric pair flow to:

  * push a "you are paired with us" notice to the claimant after
    the host accepts a PIN,
  * push an "unpaired" notice to the other side when the local
    user removes a pairing.

Every outbound message carries an Ed25519 signature over a canonical
byte representation so the receiver can verify it came from the
claimed peer (no MITM injecting fake "you got unpaired" messages).

This module never sees prompts or chat content — pair / unpair are
metadata only. The chat data path stays on Ollama / llama-server
and is governed by the privacy guard.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import time

import httpx

from . import identity

log = logging.getLogger("gigachat.p2p.lan_client")

# How long to wait for a peer to ACK a notify. Short — these are
# fire-and-forget metadata pings; if the peer is briefly offline
# we'd rather give up and leave the symmetric state inconsistent
# than block the user's UI for seconds.
_NOTIFY_TIMEOUT_SEC = 5.0


def _sign_pair_notify(
    *,
    host_device_id: str,
    host_label: str,
    host_public_key_b64: str,
    host_x25519_public_b64: str,
    claimant_device_id: str,
    timestamp: float,
) -> bytes:
    """Canonical bytes for the pair-notify signature.

    Pipe-separated, version-prefixed (v2 — added X25519 pubkey).
    Mismatched serialisation breaks the signature even when the
    data is identical.
    """
    parts = [
        b"gigachat-p2p-pair-notify-v2",
        host_device_id.encode("ascii"),
        host_label.encode("utf-8"),
        host_public_key_b64.encode("ascii"),
        host_x25519_public_b64.encode("ascii"),
        claimant_device_id.encode("ascii"),
        f"{timestamp:.6f}".encode("ascii"),
    ]
    return hashlib.sha256(b"|".join(parts)).digest()


def _sign_unpair_notify(
    *,
    initiator_device_id: str,
    initiator_public_key_b64: str,
    peer_device_id: str,
    timestamp: float,
) -> bytes:
    """Canonical bytes for the unpair-notify signature."""
    parts = [
        b"gigachat-p2p-unpair-notify-v1",
        initiator_device_id.encode("ascii"),
        initiator_public_key_b64.encode("ascii"),
        peer_device_id.encode("ascii"),
        f"{timestamp:.6f}".encode("ascii"),
    ]
    return hashlib.sha256(b"|".join(parts)).digest()


async def push_pair_notify(
    *,
    peer_ip: str,
    peer_port: int,
    peer_device_id: str,
    peer_x25519_public_b64: str | None = None,
) -> bool:
    """Tell the freshly-paired claimant that the host accepted them.

    Called by the host's `accept_pairing` after the trust anchor is
    persisted locally. The claimant's POST handler verifies the
    signature, stores the host's identity, and the friendship is
    symmetric.

    Encryption: when `peer_x25519_public_b64` is supplied the body
    is wrapped in a `p2p_crypto` envelope addressed to the peer.
    For backwards compatibility with peers paired before E2E shipped
    (no X25519 key on file), we fall back to plaintext + signed
    body — the receiver accepts both shapes.

    Returns True on success, False on any error.
    """
    if not peer_ip:
        log.info("pair-notify skipped — no peer ip yet")
        return False
    me = identity.get_identity()
    ts = time.time()
    digest = _sign_pair_notify(
        host_device_id=me.device_id,
        host_label=me.label,
        host_public_key_b64=me.public_key_b64,
        host_x25519_public_b64=me.x25519_public_b64,
        claimant_device_id=peer_device_id,
        timestamp=ts,
    )
    sig = base64.b64encode(me.sign(digest)).decode("ascii")
    # Tell the peer which port to reach us on. Unsigned routing
    # metadata — NOT in the signature digest, because tampering it
    # is harmless (worst case: the peer can't reach us and falls
    # back to mDNS / active scan). Keeping it outside the signed
    # bytes also makes the wire format additive: peers running
    # older code that don't expect `host_port` ignore it (Pydantic
    # default), and we don't have to bump the signature version.
    try:
        from . import p2p_discovery as _disc
        host_port = _disc.get_advertised_port()
    except Exception:
        host_port = 8000
    inner_body = {
        "host_device_id": me.device_id,
        "host_label": me.label,
        "host_public_key_b64": me.public_key_b64,
        "host_x25519_public_b64": me.x25519_public_b64,
        "host_port": host_port,
        "claimant_device_id": peer_device_id,
        "timestamp": ts,
        "signature_b64": sig,
    }
    if peer_x25519_public_b64:
        # Encrypted envelope wraps the (already-signed) inner body.
        # The envelope's own signature pins the SENDER (us) — the
        # inner signature pins the PAIR-NOTIFY ACTION's authenticity.
        # Two layers means a tampered envelope fails AEAD before we
        # spend cycles on the inner verify.
        from . import p2p_crypto as _pc
        outer = _pc.seal_json(
            recipient_x25519_pub_b64=peer_x25519_public_b64,
            recipient_device_id=peer_device_id,
            payload=inner_body,
        )
        post_body = {"encrypted": outer}
    else:
        # Legacy / first-pair path — peer doesn't have our X25519
        # key yet, so we can't address an envelope to them. The
        # signed inner body still proves authenticity.
        post_body = inner_body

    url = f"http://{peer_ip}:{peer_port}/api/p2p/pair/notify"
    try:
        async with httpx.AsyncClient(timeout=_NOTIFY_TIMEOUT_SEC) as client:
            r = await client.post(url, json=post_body)
            r.raise_for_status()
        log.info(
            "pair-notify delivered to %s:%d (%s)",
            peer_ip, peer_port,
            "encrypted" if peer_x25519_public_b64 else "plaintext",
        )
        return True
    except Exception as e:
        log.info(
            "pair-notify to %s:%d failed: %s — claimant won't have a "
            "symmetric record until it pairs back",
            peer_ip, peer_port, e,
        )
        return False


async def push_unpair_notify(
    *,
    peer_ip: str,
    peer_port: int,
    peer_device_id: str,
    peer_x25519_public_b64: str | None = None,
) -> bool:
    """Tell the friend that we unpaired so they drop their record too.

    Symmetric unpair: when EITHER side removes the friendship, the
    OTHER side gets removed too. The signed message proves the
    request really came from the claimed initiator (an attacker
    can't drop someone else's pairing without their private key).

    Encryption: same treatment as `push_pair_notify` — body is
    wrapped in a `p2p_crypto` envelope when we have the peer's
    X25519 key on file; falls back to plaintext+signed for legacy
    peers paired before E2E shipped.

    Failure non-fatal: the local-side removal already succeeded.
    Worst case the peer's record stays orphaned until they prune
    it manually or pair again.
    """
    if not peer_ip:
        return False
    me = identity.get_identity()
    ts = time.time()
    digest = _sign_unpair_notify(
        initiator_device_id=me.device_id,
        initiator_public_key_b64=me.public_key_b64,
        peer_device_id=peer_device_id,
        timestamp=ts,
    )
    sig = base64.b64encode(me.sign(digest)).decode("ascii")
    inner_body = {
        "initiator_device_id": me.device_id,
        "initiator_public_key_b64": me.public_key_b64,
        "peer_device_id": peer_device_id,
        "timestamp": ts,
        "signature_b64": sig,
    }
    if peer_x25519_public_b64:
        from . import p2p_crypto as _pc
        outer = _pc.seal_json(
            recipient_x25519_pub_b64=peer_x25519_public_b64,
            recipient_device_id=peer_device_id,
            payload=inner_body,
        )
        post_body = {"encrypted": outer}
    else:
        post_body = inner_body

    url = f"http://{peer_ip}:{peer_port}/api/p2p/pair/unpair-notify"
    try:
        async with httpx.AsyncClient(timeout=_NOTIFY_TIMEOUT_SEC) as client:
            r = await client.post(url, json=post_body)
            r.raise_for_status()
        log.info(
            "unpair-notify delivered to %s:%d (%s)",
            peer_ip, peer_port,
            "encrypted" if peer_x25519_public_b64 else "plaintext",
        )
        return True
    except Exception as e:
        log.info(
            "unpair-notify to %s:%d failed: %s — friend's record may "
            "stay orphaned until they prune it",
            peer_ip, peer_port, e,
        )
        return False


# Reference to the main asyncio loop the FastAPI app is running on.
# Captured during lifespan startup (see `register_main_loop`) so that
# code running on a threadpool worker (FastAPI dispatches sync `def`
# handlers there) can still schedule coroutines on the main loop via
# `run_coroutine_threadsafe`.
#
# Why this matters: until 2026-05 `fire_and_forget` only worked when
# called from an async context, because `asyncio.get_event_loop()` in
# a threadpool thread returns a fresh, non-running loop and the
# `is_running()` check sent every coroutine straight to `coro.close()`.
# That silently dropped every pair-notify / unpair-notify scheduled
# from sync handlers — symmetric pairings only "worked" via manual
# probe scripts that called `asyncio.run()` directly, never through
# the natural pair flow.
_MAIN_LOOP: asyncio.AbstractEventLoop | None = None


def register_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Capture the FastAPI app's main asyncio loop.

    Called once during app startup so `fire_and_forget` can dispatch
    coroutines onto the right loop even when invoked from a sync
    handler running on a threadpool worker.
    """
    global _MAIN_LOOP
    _MAIN_LOOP = loop


def fire_and_forget(coro) -> None:
    """Schedule an async coroutine without awaiting it.

    Used at the API endpoint level so the user's pair/unpair click
    returns immediately while the notify is still in flight to the
    peer. Errors are swallowed inside the coroutine.

    Three cases, in order of preference:
      1. We're already on the running loop — `create_task` it directly.
      2. We're on a threadpool worker (sync handler) but the main loop
         is registered and running — `run_coroutine_threadsafe` dispatches
         it cross-thread.
      3. No usable loop — drop the coroutine cleanly. Reaching here
         means we're being called during shutdown or before the app
         finished booting; either way there's nowhere to schedule.
    """
    # Case 1: same-thread async dispatch.
    try:
        running = asyncio.get_running_loop()
        running.create_task(coro)
        return
    except RuntimeError:
        # No running loop on this thread — fall through to cross-thread.
        pass
    # Case 2: cross-thread dispatch onto the registered main loop.
    main = _MAIN_LOOP
    if main is not None and main.is_running():
        try:
            asyncio.run_coroutine_threadsafe(coro, main)
            return
        except RuntimeError:
            # Loop closed between is_running() check and submission;
            # fall through to drop.
            pass
    # Case 3: nowhere to schedule. Closing the coroutine prevents the
    # "coroutine was never awaited" RuntimeWarning at GC time.
    coro.close()
