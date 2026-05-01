"""TURN-style relay client — fallback when direct P2P connection fails.

Why this exists
===============
The bulk of P2P traffic flows directly between peers via STUN-discovered
candidates. That works for cone-NAT routers (most ISPs) but FAILS for
symmetric-NAT routers (most home routers without UPnP) when both
peers are behind symmetric NATs at the same time.

This module bridges that gap by relaying encrypted envelopes through
the rendezvous server when the direct path fails. The relay sees
ONLY ciphertext — confidentiality is end-to-end via the existing
p2p_crypto envelope (X25519 ECDH + ChaCha20-Poly1305 AEAD), so the
relay's trust profile doesn't widen by adding it.

Latency / scope
===============
HTTP POST + long-poll roundtrip is ~100-300 ms on Cloud Run, vs.
sub-millisecond on a direct LAN hop. Acceptable for one-shot calls
(chat completions, embeddings) but TOO SLOW for streaming where each
NDJSON chunk would pay the relay tax. Streaming over the relay would
need WebSocket transport — deferred to a follow-up.

API
===
    relay_send(recipient_device_id, envelope) -> bool
        POST a sealed envelope to the rendezvous's relay queue.
        Returns True on accepted-by-server, False on any failure.

    start_inbox_loop()  /  stop_inbox_loop()
        Long-poll the relay inbox and dispatch any inbound
        envelopes to the secure-proxy verify path. Lifetime is
        managed by the FastAPI lifespan, gated on the public-pool
        toggle.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from . import db, identity, p2p_crypto, p2p_rendezvous

log = logging.getLogger("gigachat.p2p.relay")

# Per-attempt POST timeout. The relay should respond quickly; long
# tail means the rendezvous is overloaded — bail and let the caller
# decide whether to retry direct or surface failure.
_RELAY_SEND_TIMEOUT_SEC = 8.0

# Long-poll request timeout — must be > server's RELAY_POLL_MAX_WAIT_SEC
# (currently 25 s) but with safety margin for DNS / TLS round-trips.
_INBOX_POLL_TIMEOUT_SEC = 35.0

# Time to back off after a poll failure. Keeps a flapping rendezvous
# from getting hammered while still recovering quickly when it comes
# back up.
_INBOX_BACKOFF_SEC = 5.0

# How often to check whether the public-pool toggle changed without a
# full inbox poll cycle. Cheap (one DB read).
_TOGGLE_CHECK_INTERVAL_SEC = 5.0


# ---------------------------------------------------------------------------
# Public read API
# ---------------------------------------------------------------------------

_loop_task: asyncio.Task | None = None
_loop_stop_event: asyncio.Event | None = None

# Pending request/response correlation. The relay is asynchronous —
# we POST a request envelope, the recipient processes it eventually,
# then they POST their response envelope back to OUR inbox. To pair
# the two, we embed `_relay_req_id` in the request payload; the
# secure-proxy server-side path echoes it into the response payload.
# This dict maps {req_id: asyncio.Future} so the inbox loop can hand
# the matched response to the awaiting forward_via_relay() call.
_pending_responses: dict[str, "asyncio.Future[dict]"] = {}


# Default timeout for a relay request/response round-trip. Includes
# rendezvous-poll latency on both sides PLUS the recipient's actual
# Ollama call. Generous to absorb slow rendezvous + slow peers.
_RELAY_RT_TIMEOUT_SEC = 60.0


def _public_pool_enabled() -> bool:
    val = db.get_setting("p2p_public_pool_enabled")
    if val is None:
        return True
    return str(val).lower() in ("1", "true", "yes", "on")


async def relay_send(
    recipient_device_id: str, envelope: dict,
) -> bool:
    """POST a sealed envelope to the rendezvous's relay queue.

    Returns True iff the rendezvous accepted the envelope (the
    recipient may still never poll for it; this only confirms the
    relay holds it). Returns False on any failure — caller decides
    whether to retry or surface to the user.
    """
    if not recipient_device_id or not isinstance(envelope, dict):
        return False
    url = p2p_rendezvous._current_rendezvous_url()
    if not url:
        return False
    body = {"recipient": recipient_device_id, "envelope": envelope}
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_RELAY_SEND_TIMEOUT_SEC),
        ) as client:
            r = await client.post(f"{url.rstrip('/')}/relay/send", json=body)
            if r.status_code >= 400:
                log.debug(
                    "p2p_relay: relay_send rejected by server: HTTP %d %s",
                    r.status_code, r.text[:200],
                )
                return False
            return True
    except Exception as e:
        log.debug("p2p_relay: relay_send transport error: %s", e)
        return False


# ---------------------------------------------------------------------------
# Inbox poll loop — picks up envelopes addressed to us via the relay
# ---------------------------------------------------------------------------


async def _dispatch_inbound_envelope(envelope: dict) -> None:
    """Route a relay-delivered envelope to the right handler.

    Two cases distinguished by inspecting the (encrypted) envelope:
      1. RESPONSE — sealed for us, contains our `_relay_req_id` in
         the payload. Means a peer is sending back the answer to a
         request we made earlier; we hand it to the awaiting
         Future and we're done.
      2. REQUEST — sealed for us, no matching pending req_id.
         A peer is asking us to do work. Hand to the secure-proxy
         verify+forward path, then ship the response envelope back
         via the relay.

    Drop on any verify failure — the relay can't tell us who tried.
    """
    # Cheap pre-decrypt path: try to verify + open the envelope so
    # we can inspect the payload to decide which branch to take.
    # Verify costs the same as the secure-proxy verify path would
    # have; we pay it once here and reuse the result if this turns
    # out to be a request.
    me = identity.get_identity()
    if not isinstance(envelope, dict):
        return
    sender_id = envelope.get("sender") or ""
    if not sender_id:
        return
    # Sender pubkey lookup — paired_devices first, then discovered
    # cache. Mirrors p2p_secure_proxy._peer_record_for to keep the
    # trust model identical for direct vs relay paths.
    paired = db.get_paired_device(sender_id)
    if paired:
        sender_pub = paired.get("public_key_b64") or ""
    else:
        from . import p2p_pool_inventory as _inv
        rec = _inv.get_discovered_peer(sender_id)
        sender_pub = (rec or {}).get("public_key_b64") or ""
    if not sender_pub:
        log.debug("p2p_relay: dropped envelope from unknown sender %s", sender_id)
        return
    try:
        payload, _ = p2p_crypto.open_envelope_json(
            envelope, expected_sender_ed25519_pub_b64=sender_pub,
        )
    except p2p_crypto.CryptoError as e:
        log.info("p2p_relay: inbound envelope verify failed: %s", e)
        return

    # Branch: response to one of our pending requests?
    rid = payload.get("_relay_req_id")
    if isinstance(rid, str) and rid:
        fut = _pending_responses.pop(rid, None)
        if fut is not None and not fut.done():
            fut.set_result(payload)
            return
        # Stale response (we already timed out the future, or this
        # is a duplicate retry from the peer). Drop silently — no
        # harm done.
        log.debug("p2p_relay: response for unknown rid %s; dropping", rid)
        return

    # Branch: it's a request from a peer. Hand to secure-proxy
    # for verify+forward (which will redo the verify; cheap, and
    # keeps the proxy's auth path the single source of truth).
    from . import p2p_secure_proxy as _sp
    try:
        response_env = await _sp.serve_forward_one_shot(envelope)
    except p2p_crypto.CryptoError as e:
        log.info("p2p_relay: inbound forward verify failed: %s", e)
        return
    except Exception as e:
        log.info("p2p_relay: inbound forward failed: %s", e)
        return
    sent = await relay_send(sender_id, response_env)
    if not sent:
        log.info(
            "p2p_relay: failed to ship response back to %s via relay",
            sender_id,
        )


async def forward_via_relay(
    *,
    recipient_device_id: str,
    recipient_x25519_pub_b64: str,
    recipient_ed25519_pub_b64: str,
    method: str,
    path: str,
    body: Any = None,
    headers: dict | None = None,
    timeout_sec: float = _RELAY_RT_TIMEOUT_SEC,
) -> tuple[int, str]:
    """One-shot encrypted request/response over the rendezvous relay.

    Drop-in replacement for ``p2p_secure_client.forward`` that takes
    the same logical inputs (method, path, body) but routes through
    the rendezvous's relay queue instead of a direct HTTP POST. Used
    as a fallback when the recipient is behind a symmetric NAT and
    the direct path can't reach them.

    Generates a `_relay_req_id` for correlation, parks an asyncio
    Future in `_pending_responses[rid]`, ships the request, awaits
    the matching response from our inbox loop, returns
    ``(status, body_text)``.

    Raises ``RuntimeError`` (or asyncio.TimeoutError) on any failure.
    Caller may catch and surface to the user.
    """
    import uuid as _uuid

    if not (recipient_device_id and recipient_x25519_pub_b64
            and recipient_ed25519_pub_b64):
        raise RuntimeError("relay forward needs full recipient identity")
    rid = _uuid.uuid4().hex
    payload = {
        "method": (method or "GET").upper(),
        "path": path,
        "body": body,
        "headers": headers or {},
        "_relay_req_id": rid,
    }
    envelope = p2p_crypto.seal_json(
        recipient_x25519_pub_b64=recipient_x25519_pub_b64,
        recipient_device_id=recipient_device_id,
        payload=payload,
    )
    loop = asyncio.get_event_loop()
    fut: asyncio.Future[dict] = loop.create_future()
    _pending_responses[rid] = fut
    try:
        ok = await relay_send(recipient_device_id, envelope)
        if not ok:
            raise RuntimeError("relay rejected the send")
        try:
            response_payload = await asyncio.wait_for(fut, timeout=timeout_sec)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"relay round-trip timed out after {timeout_sec}s"
            )
    finally:
        _pending_responses.pop(rid, None)
    status = int(response_payload.get("status") or 0)
    body_text = str(response_payload.get("body") or "")
    return status, body_text


async def _inbox_loop(stop_event: asyncio.Event) -> None:
    """Long-poll the relay for envelopes addressed to us. Forever.

    Re-checks the public-pool toggle on every iteration so a quick
    flip OFF stops the polling without waiting for the in-flight
    long-poll's deadline. Also tolerates rendezvous downtime —
    a long-poll failure backs off briefly and retries; the loop
    survives transient cloud blips.
    """
    me = identity.get_identity()
    last_toggle_check = 0.0
    enabled = _public_pool_enabled()
    while not stop_event.is_set():
        # Cheap toggle re-check — avoids waiting an entire poll
        # window after the user disables Public Pool.
        now = asyncio.get_event_loop().time()
        if (now - last_toggle_check) >= _TOGGLE_CHECK_INTERVAL_SEC:
            enabled = _public_pool_enabled()
            last_toggle_check = now
        if not enabled:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=2.0)
                return
            except asyncio.TimeoutError:
                continue

        url = p2p_rendezvous._current_rendezvous_url()
        if not url:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=10.0)
                return
            except asyncio.TimeoutError:
                continue

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(_INBOX_POLL_TIMEOUT_SEC),
            ) as client:
                r = await client.get(
                    f"{url.rstrip('/')}/relay/inbox/{me.device_id}",
                )
            if r.status_code == 404:
                # Old rendezvous deployment without the relay
                # endpoint — back off and retry later. New deploy
                # will start working once it lands.
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=30.0)
                    return
                except asyncio.TimeoutError:
                    continue
            if r.status_code >= 400:
                log.debug(
                    "p2p_relay: inbox poll HTTP %d %s",
                    r.status_code, r.text[:200],
                )
                try:
                    await asyncio.wait_for(
                        stop_event.wait(), timeout=_INBOX_BACKOFF_SEC,
                    )
                    return
                except asyncio.TimeoutError:
                    continue
            data = r.json() or {}
        except Exception as e:
            log.debug("p2p_relay: inbox poll transport error: %s", e)
            try:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=_INBOX_BACKOFF_SEC,
                )
                return
            except asyncio.TimeoutError:
                continue

        envelopes = data.get("envelopes") or []
        if not envelopes:
            continue
        # Dispatch in parallel — peers may be sending us bursts;
        # serialising would let one slow Ollama call block all
        # others.
        await asyncio.gather(
            *(_dispatch_inbound_envelope(env) for env in envelopes),
            return_exceptions=True,
        )


async def start() -> None:
    """Kick off the relay inbox poll loop. No-op if already running."""
    global _loop_task, _loop_stop_event
    if _loop_task is not None and not _loop_task.done():
        return
    _loop_stop_event = asyncio.Event()
    _loop_task = asyncio.create_task(
        _inbox_loop(_loop_stop_event),
        name="p2p_relay_inbox_loop",
    )
    log.info("p2p_relay: inbox poll loop started")


async def stop() -> None:
    """Stop the inbox loop and wait for it to drain. Idempotent."""
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
