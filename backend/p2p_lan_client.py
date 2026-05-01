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
    claimant_device_id: str,
    timestamp: float,
) -> bytes:
    """Canonical bytes for the pair-notify signature.

    Pipe-separated, version-prefixed so a future format change can
    bump the version without an ambiguous parse on the receiver side.
    Same shape on both sides; mismatched serialisation breaks the
    signature even when the data is identical.
    """
    parts = [
        b"gigachat-p2p-pair-notify-v1",
        host_device_id.encode("ascii"),
        host_label.encode("utf-8"),
        host_public_key_b64.encode("ascii"),
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
) -> bool:
    """Tell the freshly-paired claimant that the host accepted them.

    Called by the host's `accept_pairing` after the trust anchor is
    persisted locally. The claimant's POST handler verifies the
    signature, stores the host's identity, and the friendship is
    symmetric.

    Returns True on success, False on any error. Failure is non-fatal:
    the host's record is intact, only the claimant's side is missing
    its symmetric copy. The user can re-pair to repair.
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
        claimant_device_id=peer_device_id,
        timestamp=ts,
    )
    sig = base64.b64encode(me.sign(digest)).decode("ascii")
    body = {
        "host_device_id": me.device_id,
        "host_label": me.label,
        "host_public_key_b64": me.public_key_b64,
        "claimant_device_id": peer_device_id,
        "timestamp": ts,
        "signature_b64": sig,
    }
    url = f"http://{peer_ip}:{peer_port}/api/p2p/pair/notify"
    try:
        async with httpx.AsyncClient(timeout=_NOTIFY_TIMEOUT_SEC) as client:
            r = await client.post(url, json=body)
            r.raise_for_status()
        log.info("pair-notify delivered to %s:%d", peer_ip, peer_port)
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
) -> bool:
    """Tell the friend that we unpaired so they drop their record too.

    Symmetric unpair: when EITHER side removes the friendship, the
    OTHER side gets removed too. The signed message proves the
    request really came from the claimed initiator (an attacker
    can't drop someone else's pairing without their private key).

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
    body = {
        "initiator_device_id": me.device_id,
        "initiator_public_key_b64": me.public_key_b64,
        "peer_device_id": peer_device_id,
        "timestamp": ts,
        "signature_b64": sig,
    }
    url = f"http://{peer_ip}:{peer_port}/api/p2p/pair/unpair-notify"
    try:
        async with httpx.AsyncClient(timeout=_NOTIFY_TIMEOUT_SEC) as client:
            r = await client.post(url, json=body)
            r.raise_for_status()
        log.info("unpair-notify delivered to %s:%d", peer_ip, peer_port)
        return True
    except Exception as e:
        log.info(
            "unpair-notify to %s:%d failed: %s — friend's record may "
            "stay orphaned until they prune it",
            peer_ip, peer_port, e,
        )
        return False


def fire_and_forget(coro) -> None:
    """Schedule an async coroutine without awaiting it.

    Used at the API endpoint level so the user's pair/unpair click
    returns immediately while the notify is still in flight to the
    peer. Errors are swallowed inside the coroutine.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(coro)
            return
    except RuntimeError:
        pass
    # No running loop — drop. Background tasks during shutdown end
    # up here; the caller's user-facing action has already returned.
    coro.close()
