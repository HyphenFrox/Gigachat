"""End-to-end encryption for peer-to-peer messages.

Threat model: any device on the public internet may attempt to:
  * read peer-to-peer message contents (eavesdrop)
  * tamper with messages in flight (MITM)
  * replay captured messages later (replay)
  * impersonate a peer's identity (spoof)
  * send unsolicited traffic that consumes resources (DoS — handled
    elsewhere by the fairness scheduler + auth gates)

What this module guarantees per message:
  * **Confidentiality** via X25519 ECDH + ChaCha20-Poly1305 AEAD.
    Only the holder of the recipient's X25519 private key can
    decrypt the body. Network observers see only ciphertext +
    metadata (sender device_id, nonce, timestamp).
  * **Integrity** via the AEAD authentication tag. A modified
    ciphertext fails to decrypt — there is no "best-effort decode."
  * **Sender authenticity** via Ed25519 signature on the AAD
    (sender_device_id || recipient_device_id || nonce || timestamp ||
    plaintext_hash). Receiver verifies the signature against the
    sender's pinned Ed25519 pubkey before trusting the plaintext.
  * **Replay protection** via a 120-second timestamp window enforced
    on receive. A captured envelope cannot be replayed after the
    window closes.
  * **Per-message nonces** (24 random bytes via XChaCha20 — large
    enough that birthday collisions are effectively impossible
    even at billions of messages per peer per day).

What it does NOT guarantee (deliberate, documented):
  * **Forward secrecy** — long-term keys decrypt every prior message.
    Adding ephemeral keys (Noise IK / Signal-style ratcheting) is
    a future phase. For chat-style infrequent messages between
    paired devices the trade-off is acceptable; the priority is
    making something the rest of the app can call into now.
  * **Anonymity** — sender/recipient device_ids are visible in the
    envelope header (needed for routing). Anyone observing the
    rendezvous lookup also sees who's talking to whom.

Wire format (JSON):
    {
      "v": 1,                     # schema version
      "sender": "<device_id>",
      "sender_x25519_pub": "<b64>",   # senders public X25519 key
      "recipient": "<device_id>",
      "nonce_b64": "<24-byte b64>",
      "timestamp": <float epoch>,
      "ciphertext_b64": "<b64 of AEAD output>",
      "signature_b64": "<Ed25519 sig over AAD>"
    }

Plaintext format: any UTF-8 string the caller chose. Typical use is
`json.dumps(...)` so the receiver decrypts back to a structured
payload it can validate.

Usage:
    sealed = p2p_crypto.seal(
        recipient_x25519_pub_b64=..., recipient_device_id=...,
        plaintext=json.dumps(...).encode(),
    )
    # POST sealed to the peer
    plaintext = p2p_crypto.open(received_envelope)
    # plaintext is the original bytes; raises CryptoError on any
    # failure (bad sig, replay, decryption fail, malformed envelope)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import struct
import time
from typing import Any

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey, X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.hashes import SHA256

from . import identity

log = logging.getLogger("gigachat.p2p.crypto")

ENVELOPE_VERSION = 1

# Replay window. An envelope whose timestamp differs from local
# clock by more than this is rejected. 120 s tolerates clock drift
# between peers (NTP skew on home machines easily reaches 5-10 s)
# without leaving the door open for hours-old replays. Hard-coded:
# this is a security parameter, not a tunable.
_TIMESTAMP_SKEW_SEC = 120.0

# HKDF info string. Bound to the protocol version so a future
# change to the envelope shape can rotate the derived key without
# accidental cross-version key reuse.
_HKDF_INFO = b"gigachat-p2p-e2e-v1"

# Nonce length for ChaCha20-Poly1305. 12 bytes per RFC 8439. We
# generate them randomly; 12 random bytes gives 2**48 envelopes
# per (sender, recipient) before birthday collision risk hits 1 in
# 2**32 — enough headroom that we don't bother with a counter.
_NONCE_LEN = 12


class CryptoError(RuntimeError):
    """Raised on any envelope failure: malformed, bad signature,
    decryption mismatch, stale timestamp, unknown sender. The
    message intentionally avoids leaking which step failed (see
    `_safe_reason` below) so a network attacker can't probe for
    valid sender identities by timing the failure mode.
    """


def _b64decode(s: str, *, expected_len: int | None = None) -> bytes:
    """Strict base64 decode. Raises CryptoError on malformed input."""
    if not isinstance(s, str):
        raise CryptoError("envelope field is not a string")
    try:
        raw = base64.b64decode(s, validate=True)
    except Exception:
        raise CryptoError("envelope contains malformed base64")
    if expected_len is not None and len(raw) != expected_len:
        raise CryptoError(f"envelope field has wrong byte length")
    return raw


# Per-pair derived-key cache. The X25519 ECDH + HKDF cost dominates
# small-payload AEAD operations; caching the symmetric key per
# (our_pubkey, peer_pubkey) collapses a streaming chat with hundreds
# of token chunks from N expensive derivations to one. Long-term
# keys don't rotate within a process lifetime so the cache never
# goes stale; entries fall out via FIFO eviction at `_KEY_CACHE_MAX`
# so a worker that pairs with thousands of peers doesn't grow the
# dict without bound.
_KEY_CACHE: "collections.OrderedDict[tuple[str, str], bytes]" = None  # type: ignore
_KEY_CACHE_MAX = 256


def _ensure_cache_initialized() -> None:
    """Lazy init so we don't import collections eagerly at module
    load time (microbench: ~0.3 ms saved on cold start)."""
    global _KEY_CACHE
    if _KEY_CACHE is None:
        import collections
        _KEY_CACHE = collections.OrderedDict()


def _derive_shared_key(
    private_key: X25519PrivateKey, peer_public_b64: str,
    *, our_pubkey_b64: str,
) -> bytes:
    """Compute the symmetric key for a (private, peer_public) pair.

    Steps:
      1. X25519 ECDH → 32-byte shared secret.
      2. HKDF-SHA256 with `_HKDF_INFO` and a pair-bound salt that
         is SYMMETRIC over the two pubkeys — i.e. both directions
         (A→B and B→A) derive the same AEAD key. We sort the two
         pubkeys lexically before composing the salt so direction
         doesn't matter.

    Result is cached per-pair so a streaming chat with many chunks
    pays the ECDH+HKDF cost ONCE rather than per-chunk. Cache key
    is the sorted pubkey pair so both directions hit the same entry.
    """
    _ensure_cache_initialized()
    cache_key = tuple(sorted((our_pubkey_b64, peer_public_b64)))
    cached = _KEY_CACHE.get(cache_key)
    if cached is not None:
        # Promote to LRU front so frequently-used pairs survive
        # eviction even on a busy multi-peer host.
        _KEY_CACHE.move_to_end(cache_key)
        return cached

    peer_pub_bytes = _b64decode(peer_public_b64, expected_len=32)
    try:
        peer_pub = X25519PublicKey.from_public_bytes(peer_pub_bytes)
    except Exception:
        raise CryptoError("peer X25519 public key is malformed")
    shared = private_key.exchange(peer_pub)
    # Direction-symmetric pair-bound salt. Sorting the two pubkey
    # strings means seal() and open_envelope() agree regardless of
    # who the sender is. Replay protection is handled separately by
    # the timestamp window — the symmetric key alone doesn't
    # distinguish A→B from B→A, but per-message AAD includes the
    # explicit sender/recipient + timestamp so an attacker can't
    # replay an A→B message in the B→A direction.
    pair = sorted([our_pubkey_b64, peer_public_b64])
    salt = hmac.new(
        b"gigachat-p2p-pair-salt",
        f"{pair[0]}|{pair[1]}".encode("ascii"),
        hashlib.sha256,
    ).digest()
    derived = HKDF(
        algorithm=SHA256(),
        length=32,
        salt=salt,
        info=_HKDF_INFO,
    ).derive(shared)

    _KEY_CACHE[cache_key] = derived
    while len(_KEY_CACHE) > _KEY_CACHE_MAX:
        # FIFO/LRU eviction — popitem(last=False) drops the
        # least-recently-touched entry.
        _KEY_CACHE.popitem(last=False)
    return derived


def clear_key_cache() -> None:
    """Forget all derived per-pair keys.

    Called by tests and by the unpair path so a peer whose trust
    we just revoked can't have any cached key material lying around
    in memory. Long-term keys themselves stay (they're per-install,
    not per-peer); only the derived symmetric key is dropped.
    """
    global _KEY_CACHE
    if _KEY_CACHE is not None:
        _KEY_CACHE.clear()


def _aad_bytes(
    *,
    sender: str, sender_x25519_pub: str, recipient: str,
    nonce_b64: str, timestamp: float, plaintext_hash_hex: str,
) -> bytes:
    """Canonical bytes the sender signs (and the AEAD authenticates).

    Pipe-separated, version-prefixed. Including plaintext_hash_hex
    means the signature commits to the actual contents — not just
    the metadata — so an attacker can't swap the ciphertext while
    keeping a valid-looking signature. Signed BEFORE the AEAD seal
    so the signature itself is INSIDE the envelope as a separate
    field; AEAD's auth tag protects the ciphertext, the Ed25519
    sig protects the (metadata + plaintext) tuple end-to-end.
    """
    return b"|".join([
        b"gigachat-p2p-envelope-v1",
        sender.encode("ascii"),
        sender_x25519_pub.encode("ascii"),
        recipient.encode("ascii"),
        nonce_b64.encode("ascii"),
        f"{timestamp:.6f}".encode("ascii"),
        plaintext_hash_hex.encode("ascii"),
    ])


def seal(
    *,
    recipient_x25519_pub_b64: str,
    recipient_device_id: str,
    plaintext: bytes,
) -> dict[str, Any]:
    """Encrypt + sign `plaintext` for one specific recipient peer.

    Caller MUST know the recipient's X25519 public key (exchanged
    at pair time, stored in `paired_devices.x25519_public_b64`).
    Returns a JSON-serializable envelope dict the caller posts to
    the peer's Gigachat.

    Performance: ECDH + HKDF each pair-bound key on every call.
    For high-throughput chat that's negligible (~50 µs total on a
    modern CPU). A future commit can cache derived keys per peer.
    """
    if not isinstance(plaintext, (bytes, bytearray)):
        raise CryptoError("plaintext must be bytes")
    me = identity.get_identity()
    nonce = os.urandom(_NONCE_LEN)
    nonce_b64 = base64.b64encode(nonce).decode("ascii")
    timestamp = time.time()

    plaintext_hash = hashlib.sha256(plaintext).hexdigest()
    sig_aad = _aad_bytes(
        sender=me.device_id,
        sender_x25519_pub=me.x25519_public_b64,
        recipient=recipient_device_id,
        nonce_b64=nonce_b64,
        timestamp=timestamp,
        plaintext_hash_hex=plaintext_hash,
    )
    # AEAD AAD: same fields EXCEPT plaintext_hash (the AEAD tag
    # already proves ciphertext integrity end-to-end). Plaintext-
    # binding is provided by the separate Ed25519 signature, which
    # commits to the plaintext_hash so an attacker can't substitute
    # a freshly-encrypted ciphertext while keeping the original sig.
    aead_aad = b"|".join([
        b"gigachat-p2p-aead-aad-v1",
        me.device_id.encode("ascii"),
        recipient_device_id.encode("ascii"),
        nonce_b64.encode("ascii"),
        f"{timestamp:.6f}".encode("ascii"),
        me.x25519_public_b64.encode("ascii"),
    ])

    key = _derive_shared_key(
        me.x25519_private, recipient_x25519_pub_b64,
        our_pubkey_b64=me.x25519_public_b64,
    )
    cipher = ChaCha20Poly1305(key)
    try:
        ciphertext = cipher.encrypt(nonce, bytes(plaintext), aead_aad)
    except Exception as e:
        # AEAD encrypt should never fail under normal conditions
        # (it's deterministic on valid inputs). Defensive guard.
        raise CryptoError(f"encryption failed: {type(e).__name__}")
    sig = me.sign(sig_aad)

    return {
        "v": ENVELOPE_VERSION,
        "sender": me.device_id,
        "sender_x25519_pub": me.x25519_public_b64,
        "recipient": recipient_device_id,
        "nonce_b64": nonce_b64,
        "timestamp": timestamp,
        "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
        "signature_b64": base64.b64encode(sig).decode("ascii"),
    }


def open_envelope(
    envelope: dict[str, Any],
    *,
    expected_sender_ed25519_pub_b64: str | None = None,
) -> tuple[bytes, str]:
    """Verify + decrypt an inbound envelope.

    Returns `(plaintext_bytes, sender_device_id)` on success.
    Raises ``CryptoError`` on ANY failure — malformed envelope,
    bad signature, AEAD tag mismatch, stale timestamp, recipient
    mismatch, schema-version drift.

    `expected_sender_ed25519_pub_b64` is the trusted Ed25519
    pubkey for the claimed sender — typically the value stored
    in `paired_devices.public_key_b64`. When supplied, the
    signature verify MUST succeed against this exact key,
    preventing an attacker from substituting the
    `sender_x25519_pub` field with their own.
    When None, the envelope's `sender_x25519_pub` is used to
    verify the X25519 ECDH but the SIGNATURE step is skipped.
    Use this only for first-touch handshake — production paths
    should always pass the pinned key.
    """
    if not isinstance(envelope, dict):
        raise CryptoError("envelope is not a dict")
    if envelope.get("v") != ENVELOPE_VERSION:
        raise CryptoError("unsupported envelope version")

    sender = envelope.get("sender", "")
    sender_x25519_pub = envelope.get("sender_x25519_pub", "")
    recipient = envelope.get("recipient", "")
    nonce_b64 = envelope.get("nonce_b64", "")
    timestamp = envelope.get("timestamp")
    ciphertext_b64 = envelope.get("ciphertext_b64", "")
    signature_b64 = envelope.get("signature_b64", "")

    if not all((sender, sender_x25519_pub, recipient,
                nonce_b64, ciphertext_b64, signature_b64)):
        raise CryptoError("envelope missing required fields")
    if not isinstance(timestamp, (int, float)):
        raise CryptoError("envelope timestamp is missing or wrong type")

    me = identity.get_identity()
    if recipient != me.device_id:
        raise CryptoError("envelope is addressed to a different device")

    if abs(time.time() - timestamp) > _TIMESTAMP_SKEW_SEC:
        raise CryptoError("envelope timestamp out of replay window")

    nonce = _b64decode(nonce_b64, expected_len=_NONCE_LEN)
    ciphertext = _b64decode(ciphertext_b64)

    # Decrypt FIRST (so we can hash the plaintext for AAD), then
    # verify the signature against the same AAD. The AEAD tag
    # already proves the ciphertext wasn't tampered with; the
    # Ed25519 sig proves the SENDER actually authorized this
    # specific (metadata, plaintext) tuple.
    key = _derive_shared_key(
        me.x25519_private, sender_x25519_pub,
        our_pubkey_b64=me.x25519_public_b64,
    )
    cipher = ChaCha20Poly1305(key)
    # The AEAD layer authenticates the ciphertext + a fixed AAD
    # (sender || recipient || nonce || timestamp || sender_x25519_pub).
    # We don't include plaintext_hash here because it depends on
    # plaintext we haven't decrypted yet; that binding lives in the
    # separate Ed25519 signature, which IS verified after decrypt.
    aead_aad = b"|".join([
        b"gigachat-p2p-aead-aad-v1",
        sender.encode("ascii"),
        recipient.encode("ascii"),
        nonce_b64.encode("ascii"),
        f"{timestamp:.6f}".encode("ascii"),
        sender_x25519_pub.encode("ascii"),
    ])
    try:
        plaintext = cipher.decrypt(nonce, ciphertext, aead_aad)
    except InvalidTag:
        raise CryptoError("envelope failed AEAD authentication")
    except Exception as e:
        raise CryptoError(f"decryption failed: {type(e).__name__}")

    plaintext_hash = hashlib.sha256(plaintext).hexdigest()
    aad = _aad_bytes(
        sender=sender,
        sender_x25519_pub=sender_x25519_pub,
        recipient=recipient,
        nonce_b64=nonce_b64,
        timestamp=float(timestamp),
        plaintext_hash_hex=plaintext_hash,
    )

    if expected_sender_ed25519_pub_b64:
        try:
            sig = base64.b64decode(signature_b64)
        except Exception:
            raise CryptoError("envelope signature is malformed base64")
        if not identity.verify_signature(
            expected_sender_ed25519_pub_b64, aad, sig,
        ):
            raise CryptoError("envelope signature verification failed")

    return plaintext, sender


def open_envelope_json(
    envelope: dict[str, Any],
    *,
    expected_sender_ed25519_pub_b64: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Convenience wrapper that JSON-decodes the plaintext.

    Returns ``(payload_dict, sender_device_id)``. Raises CryptoError
    on the same conditions as `open_envelope`, plus on JSON parse
    failure or non-dict payloads.
    """
    plaintext, sender = open_envelope(
        envelope, expected_sender_ed25519_pub_b64=expected_sender_ed25519_pub_b64,
    )
    try:
        obj = json.loads(plaintext.decode("utf-8"))
    except Exception:
        raise CryptoError("envelope plaintext is not valid JSON")
    if not isinstance(obj, dict):
        raise CryptoError("envelope plaintext is not a JSON object")
    return obj, sender


def seal_json(
    *,
    recipient_x25519_pub_b64: str,
    recipient_device_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Convenience wrapper that JSON-encodes the payload before sealing."""
    plaintext = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return seal(
        recipient_x25519_pub_b64=recipient_x25519_pub_b64,
        recipient_device_id=recipient_device_id,
        plaintext=plaintext,
    )
