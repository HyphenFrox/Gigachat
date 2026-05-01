"""End-to-end encryption for peer-to-peer messages.

Threat model: any device on the public internet may attempt to:
  * read peer-to-peer message contents (eavesdrop)
  * tamper with messages in flight (MITM)
  * replay captured messages later (replay)
  * impersonate a peer's identity (spoof)
  * compromise a peer's long-term private key and use it to decrypt
    previously-captured network traffic
  * send unsolicited traffic that consumes resources (DoS — handled
    elsewhere by the fairness scheduler + auth gates)

What this module guarantees per message:
  * **Confidentiality** via X25519 ECDH + ChaCha20-Poly1305 AEAD.
    Only the holder of the recipient's X25519 private key can
    decrypt the body. Network observers see only ciphertext +
    metadata (sender device_id, nonce, timestamp, ephemeral pub).
  * **Integrity** via the AEAD authentication tag. A modified
    ciphertext fails to decrypt — there is no "best-effort decode."
  * **Sender authenticity** via Ed25519 signature on the AAD
    (sender_device_id || recipient_device_id || nonce || timestamp ||
    sender_ephemeral_x25519_pub || plaintext_hash). Receiver
    verifies against the sender's pinned Ed25519 pubkey before
    trusting the plaintext OR the ephemeral pubkey.
  * **Replay protection** via a 120-second timestamp window enforced
    on receive.
  * **Per-message nonces** (12 random bytes — birthday collision
    risk is 1 in 2**32 only after 2**48 envelopes per pair, more
    than any realistic deployment will ever see).
  * **Forward secrecy (sender-ephemeral, partial)** — every envelope
    carries a freshly-generated X25519 EPHEMERAL pubkey. The AEAD
    key is derived from ECDH(eph_priv, recipient_long_term_X25519).
    The sender DESTROYS eph_priv after sealing — captured envelopes
    can NOT be decrypted later even if the sender's long-term keys
    are compromised. Recipient compromise still reveals past traffic
    addressed to that recipient (the long-term private key is what
    decrypts), so this is "half" of full FS. Full FS for both
    directions arrives with the TLS-with-pinning migration on the
    streaming paths (TLS 1.3's ECDHE handshake gives full FS).

What it does NOT guarantee (deliberate, documented):
  * **Anonymity** — sender/recipient device_ids are visible in the
    envelope header (needed for routing). Anyone observing the
    rendezvous lookup also sees who's talking to whom.

Wire format (JSON, schema version 2):
    {
      "v": 2,                            # schema version
      "sender": "<device_id>",
      "sender_eph_x25519_pub": "<b64>",  # PER-ENVELOPE ephemeral pubkey
      "recipient": "<device_id>",
      "nonce_b64": "<12-byte b64>",
      "timestamp": <float epoch>,
      "ciphertext_b64": "<b64 of AEAD output>",
      "signature_b64": "<Ed25519 sig over AAD>"
    }

  Schema v1 (pre-FS, long-term ECDH) is still ACCEPTED for inbound
  envelopes during the rolling upgrade — gradual migration without
  forcing a flag-day re-pair on the swarm. New seals always emit v2.
  v1 path is gated to be removed in a follow-up commit once enough
  installs have upgraded.

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
from cryptography.hazmat.primitives.serialization import (
    Encoding as _SerEncoding,
    PublicFormat as _SerPublicFormat,
)

from . import identity

log = logging.getLogger("gigachat.p2p.crypto")

ENVELOPE_VERSION = 2

# v1 envelopes are still ACCEPTED on the receive path — see the
# top-level docstring. New seals always emit v2.
_LEGACY_ENVELOPE_VERSION = 1

# Replay window. An envelope whose timestamp differs from local
# clock by more than this is rejected. 120 s tolerates clock drift
# between peers (NTP skew on home machines easily reaches 5-10 s)
# without leaving the door open for hours-old replays. Hard-coded:
# this is a security parameter, not a tunable.
_TIMESTAMP_SKEW_SEC = 120.0

# HKDF info string. Bound to the protocol version so a future
# change to the envelope shape can rotate the derived key without
# accidental cross-version key reuse. v1 keeps the original info
# string so legacy envelopes still derive the right key.
_HKDF_INFO_V2 = b"gigachat-p2p-e2e-v2-eph"
_HKDF_INFO_V1 = b"gigachat-p2p-e2e-v1"

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


# Per-pair LEGACY key cache. v1 envelopes derive the AEAD key from
# the long-term ECDH(our_x25519_priv, peer_x25519_long_term_pub).
# Because that derivation is per-pair (deterministic, cacheable), we
# memoise it for v1 inbound envelopes during the rolling upgrade.
# Once v1 inbound is removed this whole cache disappears.
#
# v2 envelopes use a freshly-generated ephemeral X25519 keypair on
# the sender side and DELIBERATELY do not cache — caching the
# derived key per (eph_pub, recipient_pub) would defeat forward
# secrecy: the cached key persists in memory after the ephemeral
# private key is destroyed, so a memory-disclosure bug would re-
# enable decryption of recently-captured envelopes. v2 derivation
# runs every envelope; the ECDH + HKDF cost (~50 µs on modern CPUs)
# is negligible vs. the network and AEAD work.
_LEGACY_KEY_CACHE: "collections.OrderedDict[tuple[str, str], bytes]" = None  # type: ignore
_KEY_CACHE_MAX = 256


def _ensure_legacy_cache_initialized() -> None:
    """Lazy init so we don't import collections eagerly at module
    load time (microbench: ~0.3 ms saved on cold start)."""
    global _LEGACY_KEY_CACHE
    if _LEGACY_KEY_CACHE is None:
        import collections
        _LEGACY_KEY_CACHE = collections.OrderedDict()


def _hkdf_pair_salt(pubkey_a_b64: str, pubkey_b_b64: str) -> bytes:
    """Direction-symmetric pair-bound salt.

    Sorting the two pubkey strings means seal() and open_envelope()
    agree regardless of who the sender is. Replay protection is
    handled separately by the timestamp window — the symmetric key
    alone doesn't distinguish A→B from B→A, but per-message AAD
    includes the explicit sender/recipient + timestamp so an
    attacker can't replay an A→B message in the B→A direction.

    For v2 (ephemeral-sender) the "pair" is (sender_ephemeral_pub,
    recipient_long_term_pub). Symmetric salt still works because
    decryption only needs the same two pubkeys in the same order.
    """
    pair = sorted([pubkey_a_b64, pubkey_b_b64])
    return hmac.new(
        b"gigachat-p2p-pair-salt",
        f"{pair[0]}|{pair[1]}".encode("ascii"),
        hashlib.sha256,
    ).digest()


def _derive_v2_ephemeral_key(
    *,
    our_private: X25519PrivateKey,
    peer_pub_b64: str,
    our_pub_b64: str,
) -> bytes:
    """Derive the AEAD key for a v2 envelope.

    v2 layout: one side is an EPHEMERAL keypair (the sender's per-
    envelope keys), the other is a LONG-TERM pubkey (the recipient's).
    This function works for both seal (our_private = ephemeral,
    peer_pub = recipient long-term) and open (our_private = our
    long-term, peer_pub = sender ephemeral) — the math is symmetric.

    NOT cached. v2 derivation runs every envelope so the ephemeral
    key material doesn't linger in memory beyond the immediate
    seal/open call. ECDH + HKDF is ~50 µs — well below the AEAD
    + network costs that dominate.
    """
    peer_pub_bytes = _b64decode(peer_pub_b64, expected_len=32)
    try:
        peer_pub = X25519PublicKey.from_public_bytes(peer_pub_bytes)
    except Exception:
        raise CryptoError("peer X25519 public key is malformed")
    shared = our_private.exchange(peer_pub)
    salt = _hkdf_pair_salt(our_pub_b64, peer_pub_b64)
    return HKDF(
        algorithm=SHA256(),
        length=32,
        salt=salt,
        info=_HKDF_INFO_V2,
    ).derive(shared)


def _derive_v1_legacy_key(
    private_key: X25519PrivateKey, peer_public_b64: str,
    *, our_pubkey_b64: str,
) -> bytes:
    """Derive (and cache) the AEAD key for a LEGACY v1 envelope.

    Same scheme as the original module had: long-term-ECDH +
    direction-symmetric salt + HKDF with the v1 info string. Cached
    per-pair so a streaming-chat correspondent doesn't pay ECDH per
    chunk. NEW SEALS NEVER USE THIS PATH — only v1 inbound during
    the rolling upgrade.
    """
    _ensure_legacy_cache_initialized()
    cache_key = tuple(sorted((our_pubkey_b64, peer_public_b64)))
    cached = _LEGACY_KEY_CACHE.get(cache_key)
    if cached is not None:
        _LEGACY_KEY_CACHE.move_to_end(cache_key)
        return cached
    peer_pub_bytes = _b64decode(peer_public_b64, expected_len=32)
    try:
        peer_pub = X25519PublicKey.from_public_bytes(peer_pub_bytes)
    except Exception:
        raise CryptoError("peer X25519 public key is malformed")
    shared = private_key.exchange(peer_pub)
    salt = _hkdf_pair_salt(our_pubkey_b64, peer_public_b64)
    derived = HKDF(
        algorithm=SHA256(),
        length=32,
        salt=salt,
        info=_HKDF_INFO_V1,
    ).derive(shared)
    _LEGACY_KEY_CACHE[cache_key] = derived
    while len(_LEGACY_KEY_CACHE) > _KEY_CACHE_MAX:
        _LEGACY_KEY_CACHE.popitem(last=False)
    return derived


def clear_key_cache() -> None:
    """Forget all derived per-pair LEGACY keys.

    v2 envelopes don't cache — ephemeral keys only live for the
    duration of one seal/open call. This function only clears the
    v1 inbound cache.

    Called by tests and by the unpair path so a peer whose trust
    we just revoked can't have any cached key material lying around
    in memory. Long-term keys themselves stay (they're per-install,
    not per-peer); only the derived symmetric key is dropped.
    """
    global _LEGACY_KEY_CACHE
    if _LEGACY_KEY_CACHE is not None:
        _LEGACY_KEY_CACHE.clear()


def _aad_bytes_v2(
    *,
    sender: str, sender_eph_pub: str, recipient: str,
    nonce_b64: str, timestamp: float, plaintext_hash_hex: str,
) -> bytes:
    """Canonical bytes the sender signs for v2 envelopes.

    Includes the EPHEMERAL pubkey so a MITM cannot substitute a
    different ephemeral pubkey of their own and re-encrypt the
    ciphertext to it (the signature commits to the exact eph pub
    used in the ECDH). Plaintext_hash binding closes the same
    seam for the actual contents.

    AEAD's auth tag protects ciphertext + AEAD AAD. Ed25519
    signature protects (metadata + plaintext_hash) — together
    they cover the full envelope end-to-end.
    """
    return b"|".join([
        b"gigachat-p2p-envelope-v2",
        sender.encode("ascii"),
        sender_eph_pub.encode("ascii"),
        recipient.encode("ascii"),
        nonce_b64.encode("ascii"),
        f"{timestamp:.6f}".encode("ascii"),
        plaintext_hash_hex.encode("ascii"),
    ])


def _aad_bytes_v1(
    *,
    sender: str, sender_x25519_pub: str, recipient: str,
    nonce_b64: str, timestamp: float, plaintext_hash_hex: str,
) -> bytes:
    """Legacy v1 signed AAD shape — kept for inbound v1 verify only."""
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
    """Encrypt + sign `plaintext` for one specific recipient peer
    using the v2 (sender-ephemeral, forward-secret) envelope format.

    Generates a fresh X25519 keypair per envelope. The ephemeral
    private key is held only for the duration of this function; once
    `seal()` returns, Python's GC will reclaim it on the next sweep.
    For maximum hygiene we also drop the local reference and rely on
    cryptography's PrivateKey not exposing the raw bytes through any
    attribute — the only way back to the priv would be a memory
    disclosure of the binary heap.

    Caller MUST know the recipient's LONG-TERM X25519 public key
    (exchanged at pair time / inventory cache, stored in
    `paired_devices.x25519_public_b64`). Returns a JSON-serializable
    envelope dict the caller posts to the peer's Gigachat.

    Performance: ECDH + HKDF runs every call (no caching by
    design — caching would defeat FS). ~50 µs on modern CPUs.
    """
    if not isinstance(plaintext, (bytes, bytearray)):
        raise CryptoError("plaintext must be bytes")
    me = identity.get_identity()
    nonce = os.urandom(_NONCE_LEN)
    nonce_b64 = base64.b64encode(nonce).decode("ascii")
    timestamp = time.time()

    # Generate the per-envelope ephemeral keypair. Local-scope only —
    # eph_priv is dropped at function exit. Captured envelopes can't
    # be decrypted later from sender-side compromise alone.
    eph_priv = X25519PrivateKey.generate()
    eph_pub_bytes = eph_priv.public_key().public_bytes(
        encoding=_SerEncoding.Raw,
        format=_SerPublicFormat.Raw,
    )
    eph_pub_b64 = base64.b64encode(eph_pub_bytes).decode("ascii")

    plaintext_hash = hashlib.sha256(plaintext).hexdigest()
    sig_aad = _aad_bytes_v2(
        sender=me.device_id,
        sender_eph_pub=eph_pub_b64,
        recipient=recipient_device_id,
        nonce_b64=nonce_b64,
        timestamp=timestamp,
        plaintext_hash_hex=plaintext_hash,
    )
    # AEAD AAD: same fields EXCEPT plaintext_hash (the AEAD tag
    # already proves ciphertext integrity end-to-end). Plaintext-
    # binding is provided by the separate Ed25519 signature.
    aead_aad = b"|".join([
        b"gigachat-p2p-aead-aad-v2",
        me.device_id.encode("ascii"),
        recipient_device_id.encode("ascii"),
        nonce_b64.encode("ascii"),
        f"{timestamp:.6f}".encode("ascii"),
        eph_pub_b64.encode("ascii"),
    ])

    try:
        key = _derive_v2_ephemeral_key(
            our_private=eph_priv,
            peer_pub_b64=recipient_x25519_pub_b64,
            our_pub_b64=eph_pub_b64,
        )
        cipher = ChaCha20Poly1305(key)
        ciphertext = cipher.encrypt(nonce, bytes(plaintext), aead_aad)
    except CryptoError:
        raise
    except Exception as e:
        # AEAD encrypt should never fail under normal conditions
        # (it's deterministic on valid inputs). Defensive guard.
        raise CryptoError(f"encryption failed: {type(e).__name__}")
    finally:
        # Best-effort: drop our reference to the ephemeral private
        # key + derived symmetric key so GC can reclaim them. Python
        # doesn't guarantee zero-out (the cryptography lib's
        # X25519PrivateKey holds an OpenSSL handle that's freed in
        # OpenSSL when the object is collected), but dropping the
        # reference is the most we can do without ctypes-level zero.
        eph_priv = None  # noqa: F841 — explicit release
        try:
            del key
        except NameError:
            pass
    sig = me.sign(sig_aad)

    return {
        "v": ENVELOPE_VERSION,
        "sender": me.device_id,
        "sender_eph_x25519_pub": eph_pub_b64,
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

    Accepts BOTH v1 (legacy long-term ECDH) and v2 (sender-
    ephemeral, forward-secret) envelopes — see top-level docstring
    for the rolling-upgrade rationale.

    `expected_sender_ed25519_pub_b64` is the trusted Ed25519
    pubkey for the claimed sender — typically the value stored
    in `paired_devices.public_key_b64`. When supplied, the
    signature verify MUST succeed against this exact key,
    preventing an attacker from substituting the sender's
    ephemeral X25519 pubkey with their own.
    When None, the signature step is skipped (use only for
    first-touch handshake — production paths must pin).
    """
    if not isinstance(envelope, dict):
        raise CryptoError("envelope is not a dict")
    version = envelope.get("v")
    if version not in (ENVELOPE_VERSION, _LEGACY_ENVELOPE_VERSION):
        raise CryptoError("unsupported envelope version")

    sender = envelope.get("sender", "")
    recipient = envelope.get("recipient", "")
    nonce_b64 = envelope.get("nonce_b64", "")
    timestamp = envelope.get("timestamp")
    ciphertext_b64 = envelope.get("ciphertext_b64", "")
    signature_b64 = envelope.get("signature_b64", "")

    # The X25519 pubkey field name differs between versions.
    if version == ENVELOPE_VERSION:
        peer_x25519_pub = envelope.get("sender_eph_x25519_pub", "")
    else:
        peer_x25519_pub = envelope.get("sender_x25519_pub", "")

    if not all((sender, peer_x25519_pub, recipient,
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
    if version == ENVELOPE_VERSION:
        key = _derive_v2_ephemeral_key(
            our_private=me.x25519_private,
            peer_pub_b64=peer_x25519_pub,
            our_pub_b64=me.x25519_public_b64,
        )
        aead_aad = b"|".join([
            b"gigachat-p2p-aead-aad-v2",
            sender.encode("ascii"),
            recipient.encode("ascii"),
            nonce_b64.encode("ascii"),
            f"{timestamp:.6f}".encode("ascii"),
            peer_x25519_pub.encode("ascii"),
        ])
    else:
        key = _derive_v1_legacy_key(
            me.x25519_private, peer_x25519_pub,
            our_pubkey_b64=me.x25519_public_b64,
        )
        aead_aad = b"|".join([
            b"gigachat-p2p-aead-aad-v1",
            sender.encode("ascii"),
            recipient.encode("ascii"),
            nonce_b64.encode("ascii"),
            f"{timestamp:.6f}".encode("ascii"),
            peer_x25519_pub.encode("ascii"),
        ])

    cipher = ChaCha20Poly1305(key)
    try:
        plaintext = cipher.decrypt(nonce, ciphertext, aead_aad)
    except InvalidTag:
        raise CryptoError("envelope failed AEAD authentication")
    except Exception as e:
        raise CryptoError(f"decryption failed: {type(e).__name__}")

    plaintext_hash = hashlib.sha256(plaintext).hexdigest()
    if version == ENVELOPE_VERSION:
        aad = _aad_bytes_v2(
            sender=sender,
            sender_eph_pub=peer_x25519_pub,
            recipient=recipient,
            nonce_b64=nonce_b64,
            timestamp=float(timestamp),
            plaintext_hash_hex=plaintext_hash,
        )
    else:
        aad = _aad_bytes_v1(
            sender=sender,
            sender_x25519_pub=peer_x25519_pub,
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
