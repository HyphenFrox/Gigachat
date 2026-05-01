"""Regression: p2p_crypto envelope — v2 forward-secret seal + v1 back-compat.

These tests pin the security properties of the envelope format:
  * v2 round-trips correctly under the sender-ephemeral pattern.
  * Each v2 envelope carries a FRESH ephemeral X25519 pubkey
    (forward secrecy depends on the ephemeral being unique per
    envelope; reusing it would defeat the whole point).
  * Legacy v1 envelopes still verify (rolling-upgrade compat).
  * Tampering with any field of a v2 envelope causes a hard fail.
  * Recipient-mismatch + replay-window failures stay enforced.
"""
from __future__ import annotations

import base64
import json
import time

import pytest

from backend import identity, p2p_crypto


pytestmark = pytest.mark.smoke


def _own_envelope(payload: dict) -> dict:
    """Seal an envelope addressed to this install (loopback round-trip)."""
    me = identity.get_identity()
    return p2p_crypto.seal_json(
        recipient_x25519_pub_b64=me.x25519_public_b64,
        recipient_device_id=me.device_id,
        payload=payload,
    )


def test_v2_envelope_round_trip():
    me = identity.get_identity()
    env = _own_envelope({"hello": "world", "n": 42})
    assert env["v"] == 2
    assert "sender_eph_x25519_pub" in env
    # The legacy long-term field is GONE from v2 — its presence
    # would defeat forward secrecy on the sender side.
    assert "sender_x25519_pub" not in env

    payload, sender = p2p_crypto.open_envelope_json(
        env, expected_sender_ed25519_pub_b64=me.public_key_b64,
    )
    assert payload == {"hello": "world", "n": 42}
    assert sender == me.device_id


def test_each_v2_envelope_carries_a_unique_ephemeral_pubkey():
    """Forward-secrecy property: every seal generates a fresh
    ephemeral X25519 keypair. Two consecutive seals to the same
    recipient must produce DIFFERENT sender_eph_x25519_pub values.
    A bug like 'cache the eph keypair across calls' would break
    FS silently — this test catches that.
    """
    a = _own_envelope({"k": "v1"})
    b = _own_envelope({"k": "v2"})
    assert a["sender_eph_x25519_pub"] != b["sender_eph_x25519_pub"]
    # And both must verify independently — sanity check that the
    # uniqueness didn't break decryption.
    me = identity.get_identity()
    p2p_crypto.open_envelope_json(
        a, expected_sender_ed25519_pub_b64=me.public_key_b64,
    )
    p2p_crypto.open_envelope_json(
        b, expected_sender_ed25519_pub_b64=me.public_key_b64,
    )


def test_v2_tampered_ciphertext_fails():
    me = identity.get_identity()
    env = _own_envelope({"x": 1})
    # Flip a byte in the ciphertext — AEAD tag must reject.
    raw = base64.b64decode(env["ciphertext_b64"])
    mutated = bytearray(raw)
    mutated[0] ^= 0x01
    env["ciphertext_b64"] = base64.b64encode(bytes(mutated)).decode("ascii")
    with pytest.raises(p2p_crypto.CryptoError):
        p2p_crypto.open_envelope_json(
            env, expected_sender_ed25519_pub_b64=me.public_key_b64,
        )


def test_v2_substituted_ephemeral_fails_signature():
    """If a MITM swaps the ephemeral pubkey to one of their own
    (and re-encrypts to a different shared secret), the signature
    over the original ephemeral pub no longer verifies."""
    me = identity.get_identity()
    env = _own_envelope({"x": 1})
    # Generate a fresh keypair and substitute its pubkey.
    from cryptography.hazmat.primitives.asymmetric.x25519 import (
        X25519PrivateKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding, PublicFormat,
    )
    rogue = X25519PrivateKey.generate()
    rogue_pub = base64.b64encode(
        rogue.public_key().public_bytes(
            encoding=Encoding.Raw, format=PublicFormat.Raw,
        ),
    ).decode("ascii")
    env["sender_eph_x25519_pub"] = rogue_pub
    with pytest.raises(p2p_crypto.CryptoError):
        p2p_crypto.open_envelope_json(
            env, expected_sender_ed25519_pub_b64=me.public_key_b64,
        )


def test_v2_recipient_mismatch_fails():
    me = identity.get_identity()
    env = _own_envelope({"x": 1})
    env["recipient"] = "DIFFERENT_DEVICE_ID"
    with pytest.raises(p2p_crypto.CryptoError):
        p2p_crypto.open_envelope_json(
            env, expected_sender_ed25519_pub_b64=me.public_key_b64,
        )


def test_v2_stale_timestamp_fails():
    me = identity.get_identity()
    env = _own_envelope({"x": 1})
    env["timestamp"] = time.time() - 9999
    with pytest.raises(p2p_crypto.CryptoError):
        p2p_crypto.open_envelope_json(
            env, expected_sender_ed25519_pub_b64=me.public_key_b64,
        )


def test_v1_legacy_envelope_still_verifies():
    """Roll-forward back-compat: a v1 envelope (long-term ECDH + v1
    AAD) must still verify under the new code so peers running the
    pre-FS build can keep talking to upgraded ones until everyone
    rolls forward.
    """
    me = identity.get_identity()
    # Hand-build a v1 envelope using the old derivation. We can
    # reuse the legacy path's primitives directly.
    import hashlib
    import os
    from cryptography.hazmat.primitives.ciphers.aead import (
        ChaCha20Poly1305,
    )
    nonce = os.urandom(12)
    nonce_b64 = base64.b64encode(nonce).decode("ascii")
    timestamp = time.time()
    payload = {"legacy": True}
    plaintext = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    plaintext_hash = hashlib.sha256(plaintext).hexdigest()

    # v1 derived key — we're sealing TO ourselves so peer pub == our pub.
    key = p2p_crypto._derive_v1_legacy_key(
        me.x25519_private, me.x25519_public_b64,
        our_pubkey_b64=me.x25519_public_b64,
    )
    aead_aad = b"|".join([
        b"gigachat-p2p-aead-aad-v1",
        me.device_id.encode("ascii"),
        me.device_id.encode("ascii"),
        nonce_b64.encode("ascii"),
        f"{timestamp:.6f}".encode("ascii"),
        me.x25519_public_b64.encode("ascii"),
    ])
    cipher = ChaCha20Poly1305(key)
    ciphertext = cipher.encrypt(nonce, plaintext, aead_aad)

    sig_aad = p2p_crypto._aad_bytes_v1(
        sender=me.device_id,
        sender_x25519_pub=me.x25519_public_b64,
        recipient=me.device_id,
        nonce_b64=nonce_b64,
        timestamp=timestamp,
        plaintext_hash_hex=plaintext_hash,
    )
    sig = me.sign(sig_aad)

    env = {
        "v": 1,
        "sender": me.device_id,
        "sender_x25519_pub": me.x25519_public_b64,
        "recipient": me.device_id,
        "nonce_b64": nonce_b64,
        "timestamp": timestamp,
        "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
        "signature_b64": base64.b64encode(sig).decode("ascii"),
    }
    out, sender = p2p_crypto.open_envelope_json(
        env, expected_sender_ed25519_pub_b64=me.public_key_b64,
    )
    assert out == payload
    assert sender == me.device_id


def test_unknown_envelope_version_rejected():
    env = _own_envelope({"x": 1})
    env["v"] = 999
    me = identity.get_identity()
    with pytest.raises(p2p_crypto.CryptoError):
        p2p_crypto.open_envelope_json(
            env, expected_sender_ed25519_pub_b64=me.public_key_b64,
        )


def test_clear_key_cache_only_touches_legacy():
    """`clear_key_cache` is documented to only clear the v1 legacy
    cache — v2 has nothing to clear (no caching by design). Verify
    the function runs cleanly on a fresh module (no AttributeError
    from accessing v2-only state)."""
    p2p_crypto.clear_key_cache()  # must not raise
    # And still works after another v2 round-trip.
    me = identity.get_identity()
    env = _own_envelope({"x": 1})
    p2p_crypto.open_envelope_json(
        env, expected_sender_ed25519_pub_b64=me.public_key_b64,
    )
