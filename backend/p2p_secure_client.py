"""Client side of the encrypted Gigachat-to-Gigachat compute proxy.

When the routing layer wants to send a request to a paired peer's
Gigachat (rather than direct-to-Ollama), it calls into one of the
helpers here. They:

  1. Wrap the (method, path, body) tuple in a `p2p_crypto.seal_json`
     envelope addressed to the peer.
  2. POST the envelope to the peer's `/api/p2p/secure/forward` (one-
     shot) or `/forward-stream` (streaming) endpoint.
  3. Decrypt the response envelope(s) using our X25519 private key
     and the peer's pinned Ed25519 pubkey for sender authenticity.
  4. Hand the plaintext back as if the caller had hit Ollama directly.

The one-shot path returns ``(status, body_text)`` — same shape the
caller would get from a direct Ollama call.

The streaming path is an async generator yielding plaintext NDJSON
lines exactly as Ollama would have emitted them — drop-in
substitute for an `httpx.AsyncClient.stream(...)` over a raw
Ollama URL.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator

import httpx

from . import db, p2p_crypto

log = logging.getLogger("gigachat.p2p.secure_client")

# How long to wait for the peer's secure-proxy round-trip. Most
# embed/show calls return in <1 s; chat completions can take 30+ s
# but those use the streaming path. Keep one-shot reasonable.
_ONE_SHOT_TIMEOUT_SEC = 120.0


class SecureProxyError(RuntimeError):
    """Raised when an outbound encrypted-proxy request fails — peer
    unreachable, signature verify failed on response, malformed
    response envelope, etc. Caller may fall back to direct-Ollama
    path or surface the error.
    """


def _peer_for_worker(worker: dict) -> dict | None:
    """Resolve the paired-device record for a worker row.

    The compute_workers row carries `gigachat_device_id`; we use
    that to look up the trust anchor (X25519 pubkey + Ed25519 pubkey)
    in `paired_devices`. Returns None when the worker isn't paired
    (manually-added IP rather than pair-flow row).
    """
    did = worker.get("gigachat_device_id")
    if not did:
        return None
    return db.get_paired_device(did)


async def forward(
    worker: dict,
    *,
    method: str,
    path: str,
    body: dict | list | str | None = None,
    headers: dict | None = None,
    timeout: float | None = None,
) -> tuple[int, str]:
    """Send a one-shot encrypted request to a peer's secure proxy.

    Returns ``(status_code, response_body_text)``. Raises
    ``SecureProxyError`` on any failure: peer unreachable, peer
    refused the envelope (rate-limited, unknown sender, replay), or
    response failed to decrypt.

    `timeout` overrides the default ``_ONE_SHOT_TIMEOUT_SEC`` (120 s).
    Heartbeat / liveness callers should pass a tight value (5 s) so
    a dead peer doesn't stall the calling loop for 2 minutes; chat /
    embed callers should leave it None to keep the generous default
    that accommodates a slow LLM response.

    Caller is expected to interpret status_code + body the same way
    they would from a direct Ollama call — the proxy preserves both.
    """
    peer = _peer_for_worker(worker)
    if not peer:
        raise SecureProxyError(
            f"worker {worker.get('label')!r} has no paired-device record"
        )
    peer_x25519 = peer.get("x25519_public_b64") or ""
    if not peer_x25519:
        raise SecureProxyError(
            f"peer {peer.get('label')!r} has no X25519 key on file; "
            "re-pair to exchange encryption keys"
        )
    envelope = p2p_crypto.seal_json(
        recipient_x25519_pub_b64=peer_x25519,
        recipient_device_id=peer["device_id"],
        payload={
            "method": (method or "GET").upper(),
            "path": path,
            "body": body,
            "headers": headers or {},
        },
    )
    url = (
        f"http://{worker.get('address')}:"
        f"{int(worker.get('ollama_port') or 8000)}/api/p2p/secure/forward"
    )
    direct_failed: Exception | None = None
    response_envelope: dict | None = None
    try:
        async with httpx.AsyncClient(timeout=(timeout or _ONE_SHOT_TIMEOUT_SEC)) as client:
            r = await client.post(url, json=envelope)
            r.raise_for_status()
            response_envelope = r.json()
    except Exception as e:
        direct_failed = e

    if direct_failed is not None:
        # Direct path failed — typical cause for a public-pool peer
        # is symmetric NAT on either end. Fall back to the rendezvous
        # relay if it's available + the peer's role suggests
        # internet-routing (LAN-paired peers stay direct because
        # relay can't beat sub-ms LAN RTT). The relay sees only
        # ciphertext: confidentiality is preserved.
        try:
            from . import p2p_relay as _relay
            log.info(
                "p2p_secure_client: direct forward to %s failed (%s); "
                "falling back to relay",
                peer.get("label"), type(direct_failed).__name__,
            )
            status, body_text = await _relay.forward_via_relay(
                recipient_device_id=peer["device_id"],
                recipient_x25519_pub_b64=peer_x25519,
                recipient_ed25519_pub_b64=peer.get("public_key_b64") or "",
                method=method, path=path, body=body, headers=headers,
            )
            return status, body_text
        except Exception as relay_err:
            raise SecureProxyError(
                f"both direct and relay paths to {peer.get('label')!r} "
                f"failed: direct={type(direct_failed).__name__}: "
                f"{direct_failed}; relay={type(relay_err).__name__}: "
                f"{relay_err}"
            )
    try:
        # Verify the response was sealed BY THE PEER (signature
        # pinned to their Ed25519 pubkey). Substitution attacks
        # (e.g. a MITM swapping in their own envelope) fail here.
        payload, _ = p2p_crypto.open_envelope_json(
            response_envelope,
            expected_sender_ed25519_pub_b64=peer["public_key_b64"],
        )
    except p2p_crypto.CryptoError as e:
        raise SecureProxyError(
            f"secure proxy response from {peer.get('label')!r} "
            f"failed verification: {e}"
        )
    status = int(payload.get("status") or 0)
    body_text = str(payload.get("body") or "")
    return status, body_text


async def forward_stream(
    worker: dict,
    *,
    method: str,
    path: str,
    body: dict | list | str | None = None,
) -> AsyncGenerator[str, None]:
    """Streaming variant — yields plaintext NDJSON lines as the peer
    streams them back through the encrypted proxy.

    Each line is exactly what Ollama emitted on the peer side, so
    the existing Ollama stream parser works unchanged on the
    receiving side. The terminating `_stream: done` envelope is
    swallowed (we don't yield it) — the consumer naturally hits
    end-of-stream when our generator completes.
    """
    peer = _peer_for_worker(worker)
    if not peer:
        raise SecureProxyError(
            f"worker {worker.get('label')!r} has no paired-device record"
        )
    peer_x25519 = peer.get("x25519_public_b64") or ""
    if not peer_x25519:
        raise SecureProxyError(
            f"peer {peer.get('label')!r} has no X25519 key on file; "
            "re-pair to exchange encryption keys"
        )
    envelope = p2p_crypto.seal_json(
        recipient_x25519_pub_b64=peer_x25519,
        recipient_device_id=peer["device_id"],
        payload={
            "method": (method or "POST").upper(),
            "path": path,
            "body": body,
        },
    )
    url = (
        f"http://{worker.get('address')}:"
        f"{int(worker.get('ollama_port') or 8000)}"
        "/api/p2p/secure/forward-stream"
    )
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", url, json=envelope,
        ) as r:
            if r.status_code >= 400:
                body_bytes = await r.aread()
                raise SecureProxyError(
                    f"secure proxy stream from {peer.get('label')!r} "
                    f"failed: HTTP {r.status_code} "
                    f"{body_bytes[:200].decode('utf-8', errors='replace')}"
                )
            async for raw_line in r.aiter_lines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    chunk_env = json.loads(line)
                    payload, _ = p2p_crypto.open_envelope_json(
                        chunk_env,
                        expected_sender_ed25519_pub_b64=peer["public_key_b64"],
                    )
                except (json.JSONDecodeError, p2p_crypto.CryptoError) as e:
                    raise SecureProxyError(
                        f"secure proxy stream chunk failed verify: {e}"
                    )
                # Stream terminator / error markers.
                if "_stream" in payload:
                    if payload["_stream"] == "done":
                        return
                    if payload["_stream"] == "error":
                        raise SecureProxyError(
                            f"peer reported stream error: "
                            f"{payload.get('message', 'unknown')}"
                        )
                    continue
                # Ordinary chunk — yield the original NDJSON line.
                line_text = payload.get("line")
                if isinstance(line_text, str) and line_text:
                    yield line_text
