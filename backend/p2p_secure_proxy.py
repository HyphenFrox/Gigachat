"""End-to-end encrypted proxy for Gigachat-to-Gigachat compute traffic.

When this Gigachat install routes a compute request to a paired peer
(embed, chat, etc.), the request must NOT go in plaintext over the
network — anyone on the LAN or internet path could read user prompts.
This module replaces the direct Gigachat→peer-Ollama hop with an
encrypted Gigachat→peer-Gigachat hop:

    [host's agent.py]
       ↓ envelope = p2p_crypto.seal_json({method, path, headers, body})
       ↓ POST /api/p2p/secure/forward
    [peer's Gigachat]
       ↓ p2p_crypto.open_envelope_json(envelope, expected=peer.pubkey)
       ↓ verify peer is paired with us
       ↓ httpx.post(http://localhost:11434/api/embeddings, json=body)
       ↓ envelope_response = p2p_crypto.seal_json(response_dict)
       → response

  For STREAMING (chat) the proxy reads NDJSON from local Ollama,
  wraps each line in its own envelope, writes them back as the
  HTTP response body. The client (host's agent.py) reads those
  envelope-wrapped lines, decrypts each, and feeds the plaintext
  NDJSON to the existing Ollama-stream parser. The wire format on
  the network is: NDJSON of envelopes; the upstream parser sees
  exactly the original Ollama NDJSON.

Two endpoints, one helper module:

  * ``serve_forward_one_shot(envelope)`` — server side; returns one
    envelope holding the full response.
  * ``serve_forward_stream(envelope) -> AsyncGenerator[str]`` —
    server side; yields NDJSON envelopes line by line for SSE-style
    streams.
  * Outbound: ``backend/p2p_secure_client.py`` — caller-facing
    wrappers that build the envelope, POST it, decrypt the
    response.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import threading
import time
from typing import Any, AsyncGenerator

import httpx

from . import db, identity, p2p_crypto

log = logging.getLogger("gigachat.p2p.secure_proxy")

# Local Ollama endpoint we forward to. Always loopback because this
# proxy runs on the same machine as the Ollama instance it serves.
_LOCAL_OLLAMA = "http://127.0.0.1:11434"

# Per-peer rate-limit on inbound forward requests. Defence against a
# friend (or compromised friend keypair) pointing a torrent of
# requests at us — we keep our own compute available for our own
# use. 600 req/min per peer per hosted endpoint — ample headroom for
# an active chat conversation that fans out tags + show + chat +
# embed + system-stats every turn.
_INBOUND_RATE_PER_MIN = 600

# Hard cap on the request envelope size we'll deserialise. Protects
# us from a malicious peer trying to exhaust memory by sending a
# multi-GB body before we can validate it. 4 MB covers full chat
# history with tool definitions + injected codebase context (the
# 256 KB cap was triggering 400 Bad Request mid-chat on long
# conversations).
_MAX_INBOUND_ENVELOPE_BYTES = 4_000_000

# Hard cap on the upstream Ollama response we'll proxy back. Stops
# us from being used as a one-way amplifier (peer asks for a tiny
# thing → we stream gigabytes back). 4 MB covers chat completions
# at any reasonable length.
_MAX_UPSTREAM_RESPONSE_BYTES = 4_000_000

# Streaming chunk timeout. If the upstream Ollama stops emitting
# tokens for this long we cut the stream — prevents a stuck request
# from holding our resources indefinitely.
_STREAM_IDLE_TIMEOUT_SEC = 60.0

# Whitelist of upstream Ollama paths the secure proxy will forward
# for PAIRED peers (LAN pair or role='public' — i.e. a peer the user
# has accepted into their compute pool, directly or transitively).
# Tighter than "not /api/p2p/* and not /api/conversations" because
# even an authenticated peer should not be able to call into
# arbitrary Ollama admin endpoints (model deletion, etc.) through
# our proxy. Anything outside this set is refused at the verify
# stage — the request never hits the local Ollama.
_FORWARDABLE_PATHS = frozenset({
    "/api/embeddings",
    "/api/embed",
    "/api/chat",
    "/api/generate",
    "/api/version",     # Ollama version string — read-only metadata
    "/api/tags",        # model list — read-only metadata
    "/api/show",        # model info — read-only metadata
    "/api/ps",          # which models are currently loaded — read-only
    "/api/pull",        # auto-pull from official source on the executor
    # Gigachat-internal stats endpoint. Routes to local Gigachat (port
    # 8000), not Ollama (11434). The orchestrator polls this on each
    # worker every ~10 s while a split-model is loaded so the layer
    # split adapts in real time as the user opens/closes other apps
    # on the worker.
    "/api/p2p/system-stats",
    # Local rpc-server lifecycle endpoints — let an orchestrator
    # bring up llama.cpp's rpc-server on this peer over the
    # encrypted P2P channel, replacing the old SSH-based prep path.
    # The orchestrator calls these to make the peer eligible for
    # split-model layer placement (see ``compute_pool``).
    "/api/p2p/rpc-server/start",
    "/api/p2p/rpc-server/status",
    "/api/p2p/rpc-server/stop",
    # Multi-rpc-server orchestration: lets the orchestrator ask the
    # worker to run BOTH a SYCL rpc-server (iGPU only) and a CPU
    # rpc-server (CPU+RAM only) on different ports concurrently.
    # Each is single-backend so ggml-rpc.cpp's hybrid layout-mismatch
    # bug doesn't apply, but the worker contributes both compute
    # tiers to the orchestrator's --tensor-split.
    "/api/p2p/rpc-server/ensure-multi",
    # Peer-led split — orchestrator asks a paired peer to spawn its
    # OWN llama-server using its local GGUF, with the orchestrator's
    # rpc-server as a `--rpc` backend. Used when the model lives on
    # the peer but is too big for the peer alone — fans compute
    # across both nodes WITHOUT any data transfer (the peer's GGUF
    # stays in place).
    "/api/p2p/llama-server/start",
    "/api/p2p/llama-server/status",
    "/api/p2p/llama-server/stop",
    # LAN-first binary fetch — orchestrator installs missing
    # llama-cpp DLLs by pulling from a peer that already has the
    # file (saves hundreds of MB of internet bandwidth per fresh
    # peer). The list path is light; the get path streams the file.
    "/api/p2p/binary/list",
})

# Paths that should be forwarded to the LOCAL GIGACHAT backend rather
# than to Ollama. Everything outside this set goes to Ollama via
# `_LOCAL_OLLAMA`. Kept as a tight enum so a malicious peer can't
# coax our proxy into hitting arbitrary Gigachat internals — only
# the explicitly listed paths leak.
_GIGACHAT_INTERNAL_PATHS = frozenset({
    "/api/p2p/system-stats",
    "/api/p2p/rpc-server/start",
    "/api/p2p/rpc-server/status",
    "/api/p2p/rpc-server/stop",
    "/api/p2p/rpc-server/ensure-multi",
    "/api/p2p/llama-server/start",
    "/api/p2p/llama-server/status",
    "/api/p2p/llama-server/stop",
    "/api/p2p/binary/list",
    # GPU auto-recovery — orchestrator asks a peer to soft/hard reset
    # its own iGPU before demoting it to CPU for 24h. Tiny request
    # body and tiny JSON response so the encrypted-proxy size cap is
    # comfortably fine.
    "/api/p2p/gpu/recover",
})

# Note: ``/api/p2p/binary/get/<file>`` deliberately is NOT in the
# encrypted-proxy whitelist. The endpoint serves multi-hundred-MB
# DLLs which would blow the per-envelope size cap. The orchestrator
# fetches binaries via a direct LAN HTTP GET (the path is in
# ``app._P2P_LAN_PREFIXES`` so the auth middleware lets paired-peer
# IPs through). Confidentiality isn't a concern — the files are
# standard llama.cpp release artifacts, not user data.

# Where the Gigachat backend is reachable on this host. Loopback only —
# we never forward a peer's request to a remote Gigachat.
_LOCAL_GIGACHAT = "http://127.0.0.1:8000"

# Whitelist for DISCOVERED peers — peers we know about via the
# rendezvous bootstrap but haven't accepted into our compute pool
# yet. They can ask "what models do you have" but they cannot
# consume our compute (no /api/chat, no /api/embed, no /api/pull).
# This is the trust gate that lets the model-picker show "model X is
# on peer Y" across the whole swarm without giving every swarm peer
# a free ride on our GPU.
_DISCOVERED_PEER_PATHS = frozenset({
    "/api/version",
    "/api/tags",
    "/api/show",
    "/api/ps",
})


# ---------------------------------------------------------------------------
# Per-peer sliding-window rate limit. Module-level state, lock-guarded.
# ---------------------------------------------------------------------------
_peer_request_history: dict[str, collections.deque] = collections.defaultdict(
    collections.deque
)
_rate_lock = threading.Lock()


def _check_inbound_rate_limit(peer_device_id: str) -> None:
    """Raise CryptoError when a peer exceeds the per-minute cap.

    Sliding 60-second window; old entries are pruned each call so
    the dict can't grow without bound. The check itself is fast
    (O(window-size) — at most 60 entries per peer).
    """
    now = time.time()
    cutoff = now - 60.0
    with _rate_lock:
        history = _peer_request_history[peer_device_id]
        while history and history[0] < cutoff:
            history.popleft()
        if len(history) >= _INBOUND_RATE_PER_MIN:
            raise p2p_crypto.CryptoError(
                f"rate-limited: {_INBOUND_RATE_PER_MIN} requests/min per peer"
            )
        history.append(now)


def _peer_record_for(sender_device_id: str) -> dict | None:
    """Look up the trust record for an inbound request's sender.

    Two-tier lookup:
      1. ``paired_devices`` (DB, persistent) — the canonical store
         for peers the user has explicitly accepted (LAN pair or
         role='public' from the model picker).
      2. ``p2p_pool_inventory`` (in-memory, populated from the
         rendezvous /peers list) — peers we know about but haven't
         accepted. They get a TIGHTER whitelist via the role
         dispatch in ``_verify_inbound``.

    Returns the canonical-shape dict (with `role` set) or None when
    the sender is wholly unknown.
    """
    if not sender_device_id:
        return None
    paired = db.get_paired_device(sender_device_id)
    if paired:
        return paired
    # Lazy import to avoid a hard dependency cycle (pool_inventory
    # imports p2p_crypto which imports identity which imports db…).
    from . import p2p_pool_inventory
    return p2p_pool_inventory.get_discovered_peer(sender_device_id)


def _verify_inbound(envelope: dict) -> tuple[dict, dict]:
    """Common verify path — returns ``(payload_dict, peer_record)``
    on success, raises ``CryptoError`` on any failure.

    Steps applied (in order):
      1. Envelope is a dict and not pathologically large.
      2. Sender field present.
      3. Sender is in our paired_devices (refuses unknown identities).
      4. Sender has an X25519 pubkey on file (so we can encrypt
         the response back to them — refuse the request rather than
         send a half-secured response).
      5. Per-peer rate limit (defence against compromised friend
         keypair flooding us).
      6. Envelope signature verifies against the SENDER's stored
         Ed25519 pubkey (not whatever the envelope claims) —
         identity substitution attempts are caught here.
      7. Payload path is in the forwardable whitelist.
    """
    if not isinstance(envelope, dict):
        raise p2p_crypto.CryptoError("envelope must be a JSON object")
    # Quick size estimate — refuse pathological inputs before they
    # reach the crypto layer.
    try:
        approx_size = len(json.dumps(envelope))
    except Exception:
        raise p2p_crypto.CryptoError("envelope is not JSON-serialisable")
    if approx_size > _MAX_INBOUND_ENVELOPE_BYTES:
        raise p2p_crypto.CryptoError(
            f"envelope exceeds size cap ({approx_size} > "
            f"{_MAX_INBOUND_ENVELOPE_BYTES} bytes)"
        )
    sender_device_id = envelope.get("sender", "")
    if not isinstance(sender_device_id, str) or not sender_device_id:
        raise p2p_crypto.CryptoError("envelope missing sender")
    peer = _peer_record_for(sender_device_id)
    if not peer:
        raise p2p_crypto.CryptoError(
            "sender is not a paired peer; refusing to decrypt"
        )
    if not peer.get("x25519_public_b64"):
        raise p2p_crypto.CryptoError(
            "sender has no X25519 key on file — cannot encrypt "
            "response back. Re-pair to exchange keys."
        )
    # Public-pool toggle gate: when the user has disabled "join
    # public compute pool" in Settings, refuse compute requests
    # from peers we know about ONLY because of the rendezvous
    # (`role='public'`). LAN-paired peers (`role='local'`) are
    # untouched — they were explicitly trusted via the PIN flow
    # and the public-pool toggle is about INTERNET strangers,
    # not LAN friends.
    sender_role = (peer.get("role") or "").lower()
    if sender_role == "public":
        try:
            from . import p2p_pool_routing as _ppr
            public_pool_on = _ppr._public_pool_enabled()
        except Exception:
            public_pool_on = True
        if not public_pool_on:
            raise p2p_crypto.CryptoError(
                "public-pool sharing is disabled in Settings; "
                "refusing compute request from public-pool peer"
            )
    _check_inbound_rate_limit(sender_device_id)
    payload, sender = p2p_crypto.open_envelope_json(
        envelope,
        expected_sender_ed25519_pub_b64=peer.get("public_key_b64"),
    )
    if sender != sender_device_id:
        # `open_envelope_json` returns the verified sender from the
        # envelope; this should match what we used for the lookup.
        raise p2p_crypto.CryptoError(
            "envelope sender mismatch after verify"
        )
    path = payload.get("path") or ""
    # Per-role whitelist dispatch — paired peers (full set) vs.
    # discovered peers (read-only metadata only). See whitelist
    # docstrings above for the rationale.
    role = peer.get("role") or ""
    allowed = (
        _DISCOVERED_PEER_PATHS
        if role == "discovered"
        else _FORWARDABLE_PATHS
    )
    if path not in allowed:
        raise p2p_crypto.CryptoError(
            f"path {path!r} is not on the secure-proxy whitelist "
            f"for role {role!r}"
        )
    return payload, peer


async def serve_forward_one_shot(envelope: dict) -> dict:
    """Decrypt + verify an inbound forward envelope, call local Ollama,
    encrypt the response, return.

    Raises ``p2p_crypto.CryptoError`` on any verification failure —
    the caller surfaces this as HTTP 400/401 to the requester.
    """
    payload, peer = _verify_inbound(envelope)
    method = (payload.get("method") or "GET").upper()
    path = payload.get("path") or ""
    body = payload.get("body")
    extra_headers = payload.get("headers") or {}

    # Pick the forward target. Gigachat-internal paths (e.g. the
    # adaptive system-stats endpoint) hit our own loopback Gigachat
    # backend; everything else goes to local Ollama. Keeping this
    # routing decision next to the path-whitelist check above makes
    # it harder to accidentally widen the surface — a path has to be
    # in BOTH whitelists for it to land on Gigachat.
    if path in _GIGACHAT_INTERNAL_PATHS:
        upstream = _LOCAL_GIGACHAT
        upstream_label = "Gigachat"
    else:
        upstream = _LOCAL_OLLAMA
        upstream_label = "Ollama"
    url = f"{upstream}{path}"
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            r = await client.request(
                method, url, json=body if isinstance(body, (dict, list)) else None,
                content=body if isinstance(body, str) else None,
                headers=extra_headers,
            )
        except Exception as e:
            raise p2p_crypto.CryptoError(
                f"upstream {upstream_label} request failed: "
                f"{type(e).__name__}: {e}"
            )

    # Refuse to forward a response larger than the cap — defends
    # us from being used as a one-way amplifier.
    if len(r.content or b"") > _MAX_UPSTREAM_RESPONSE_BYTES:
        raise p2p_crypto.CryptoError(
            f"upstream response too large "
            f"({len(r.content)} > {_MAX_UPSTREAM_RESPONSE_BYTES} bytes)"
        )

    # Build the response envelope. Sealed FOR the sender (encryption
    # key derived from their X25519 pubkey, signed by our identity).
    response_payload = {
        "status": r.status_code,
        "content_type": r.headers.get("content-type", "application/json"),
        "body": r.text,  # plaintext-of-Ollama-response, will be encrypted in seal
    }
    # Relay-mode correlation id: when the request arrived via the
    # rendezvous's relay (as opposed to a direct HTTP POST), the
    # client embedded `_relay_req_id` in the request payload so it
    # could match the eventual response envelope back to the
    # awaiting future. Echo it verbatim. Cheap; the field is
    # ignored by direct-HTTP callers.
    rid = payload.get("_relay_req_id")
    if isinstance(rid, str) and rid:
        response_payload["_relay_req_id"] = rid
    return p2p_crypto.seal_json(
        recipient_x25519_pub_b64=peer.get("x25519_public_b64") or "",
        recipient_device_id=peer["device_id"],
        payload=response_payload,
    )


async def serve_forward_stream(envelope: dict) -> AsyncGenerator[str, None]:
    """Decrypt + verify a streaming forward request, proxy to local
    Ollama, yield envelope-wrapped NDJSON lines back.

    Each yielded string is a JSON envelope on its own line — the
    receiver reads line-by-line, decrypts each envelope, and the
    plaintext is the original NDJSON line that Ollama would have
    emitted directly. This means the existing Ollama stream parser
    works unchanged on the receiving side.

    The terminating "stream is done" envelope carries
    `{"_stream": "done", "status": <int>}` so the receiver knows
    the upstream completed (vs the connection just dropped).
    """
    payload, peer = _verify_inbound(envelope)
    method = (payload.get("method") or "POST").upper()
    path = payload.get("path") or ""
    body = payload.get("body")
    url = f"{_LOCAL_OLLAMA}{path}"
    recipient_x25519 = peer.get("x25519_public_b64") or ""
    sender_id = peer["device_id"]

    # Per-stream byte budget — same amplification cap as the
    # one-shot path, but enforced incrementally as we forward.
    bytes_forwarded = 0

    timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream(
                method, url,
                json=body if isinstance(body, (dict, list)) else None,
            ) as r:
                # If upstream returned a 4xx/5xx, surface that as a
                # `_stream: error` envelope BEFORE forwarding any line
                # data. Otherwise the receiver sees the error JSON
                # body as a normal NDJSON chunk, fails to parse it,
                # and the stream silently ends with no tokens — the
                # user just sees an empty assistant reply.
                #
                # Verified-against-bug case: peer Ollama returns
                # `{"error":"model requires more system memory ..."}`
                # with HTTP 500 when a new chat picks a worker whose
                # llama-server is already squatting all the RAM.
                # Without this branch, the user sees nothing; with it,
                # they see the actual OOM message.
                if r.status_code >= 400:
                    body_bytes = await r.aread()
                    err_body = body_bytes.decode("utf-8", errors="replace")[:400]
                    err_envelope = p2p_crypto.seal_json(
                        recipient_x25519_pub_b64=recipient_x25519,
                        recipient_device_id=sender_id,
                        payload={
                            "_stream": "error",
                            "message": (
                                f"upstream HTTP {r.status_code}: "
                                f"{err_body.strip()}"
                            ),
                        },
                    )
                    yield json.dumps(err_envelope) + "\n"
                    return
                # Each upstream NDJSON line gets its own envelope. The
                # envelope itself is a single JSON object; we write
                # one per line so the receiver can split on '\n'.
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    bytes_forwarded += len(line)
                    if bytes_forwarded > _MAX_UPSTREAM_RESPONSE_BYTES:
                        log.warning(
                            "secure_proxy stream cut at %d bytes "
                            "(amplification cap hit)",
                            bytes_forwarded,
                        )
                        cap_envelope = p2p_crypto.seal_json(
                            recipient_x25519_pub_b64=recipient_x25519,
                            recipient_device_id=sender_id,
                            payload={
                                "_stream": "error",
                                "message": "response cap exceeded",
                            },
                        )
                        yield json.dumps(cap_envelope) + "\n"
                        return
                    chunk_envelope = p2p_crypto.seal_json(
                        recipient_x25519_pub_b64=recipient_x25519,
                        recipient_device_id=sender_id,
                        payload={"line": line},
                    )
                    yield json.dumps(chunk_envelope) + "\n"
                # Final marker so the receiver can distinguish a
                # graceful end from a dropped connection.
                done_envelope = p2p_crypto.seal_json(
                    recipient_x25519_pub_b64=recipient_x25519,
                    recipient_device_id=sender_id,
                    payload={"_stream": "done", "status": r.status_code},
                )
                yield json.dumps(done_envelope) + "\n"
        except Exception as e:
            log.warning("secure_proxy stream upstream error: %s", e)
            err_envelope = p2p_crypto.seal_json(
                recipient_x25519_pub_b64=recipient_x25519,
                recipient_device_id=sender_id,
                payload={"_stream": "error", "message": f"{type(e).__name__}: {e}"},
            )
            yield json.dumps(err_envelope) + "\n"
