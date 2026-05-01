"""Regression: p2p_tls — self-signed cert + pubkey pinning.

Pins the security primitives the streaming-TLS migration depends on:

  * ``ensure_identity_cert()`` is idempotent + binds to the running
    install's Ed25519 identity (the cert pubkey == identity pubkey,
    which is what makes pinning trivial).
  * ``cert_pubkey_b64()`` extracts an Ed25519 pubkey from PEM or DER.
  * ``verify_peer_cert()`` is a constant-time string compare — wrong
    pin returns False without raising.
  * ``make_pinned_client_ssl_context()`` refuses to build a context
    without a pin (would silently accept any cert, defeating the
    whole point).
"""
from __future__ import annotations

import base64
import os
import shutil

import pytest

from backend import identity, p2p_tls


pytestmark = pytest.mark.smoke


@pytest.fixture()
def isolated_identity(tmp_path, monkeypatch):
    """Point the identity + cert paths at a tmp dir so tests don't
    touch the real install's data/identity*.* files.

    We create a fresh identity + fresh cert in the tmp dir, return
    the (identity, cert_path, key_path) so the test can check
    against it.
    """
    monkeypatch.setattr(
        identity, "_IDENTITY_PATH", tmp_path / "identity.json",
    )
    monkeypatch.setattr(identity, "_CACHED", None)
    monkeypatch.setattr(
        p2p_tls, "_CERT_PATH", tmp_path / "identity-cert.pem",
    )
    monkeypatch.setattr(
        p2p_tls, "_CERT_KEY_PATH", tmp_path / "identity-cert-key.pem",
    )
    me = identity.get_identity()
    cert_path, key_path = p2p_tls.ensure_identity_cert()
    yield me, cert_path, key_path


def test_ensure_identity_cert_creates_files(isolated_identity):
    me, cert_path, key_path = isolated_identity
    assert cert_path.exists()
    assert key_path.exists()
    # Cert PEM has the standard armor.
    pem = cert_path.read_bytes()
    assert pem.startswith(b"-----BEGIN CERTIFICATE-----")
    assert b"-----END CERTIFICATE-----" in pem


def test_cert_pubkey_matches_identity(isolated_identity):
    me, cert_path, _ = isolated_identity
    pem = cert_path.read_bytes()
    extracted = p2p_tls.cert_pubkey_b64(pem)
    assert extracted == me.public_key_b64


def test_ensure_is_idempotent(isolated_identity):
    """A second call should NOT regenerate when the existing cert
    still matches the identity AND has plenty of validity left."""
    me, cert_path, key_path = isolated_identity
    pem_first = cert_path.read_bytes()
    serial_first = base64.b64encode(pem_first[:64])  # rough fingerprint
    p2p_tls.ensure_identity_cert()
    pem_second = cert_path.read_bytes()
    assert pem_first == pem_second


def test_verify_peer_cert_correct_pin_passes(isolated_identity):
    me, cert_path, _ = isolated_identity
    pem = cert_path.read_bytes()
    assert p2p_tls.verify_peer_cert(pem, me.public_key_b64) is True


def test_verify_peer_cert_wrong_pin_rejects(isolated_identity):
    _, cert_path, _ = isolated_identity
    pem = cert_path.read_bytes()
    # Generate a different valid b64 of correct length; pin must
    # match by EXACT string compare.
    wrong = base64.b64encode(b"\x00" * 32).decode("ascii")
    assert p2p_tls.verify_peer_cert(pem, wrong) is False


def test_verify_peer_cert_handles_garbage(isolated_identity):
    _, _, _ = isolated_identity
    # Bytes that aren't a valid cert at all → False, no exception.
    assert p2p_tls.verify_peer_cert(b"not a cert", "anything") is False
    assert p2p_tls.verify_peer_cert(b"", "anything") is False


def test_pinned_context_refuses_empty_pin(isolated_identity):
    """Passing an empty pin would cause us to accept ANY cert — a
    silent loss of security. Refuse rather than pretend to pin."""
    with pytest.raises(ValueError):
        p2p_tls.make_pinned_client_ssl_context("")


def test_pinned_context_built_when_pin_present(isolated_identity):
    me, _, _ = isolated_identity
    ctx = p2p_tls.make_pinned_client_ssl_context(me.public_key_b64)
    # The context is configured for caller-side post-handshake pin
    # check — chain verification is OFF, hostname check is OFF.
    import ssl
    assert ctx.verify_mode == ssl.CERT_NONE
    assert ctx.check_hostname is False


def test_cert_extraction_rejects_non_ed25519(isolated_identity, tmp_path):
    """A cert with an RSA / ECDSA subject must NOT be accepted —
    our pinning scheme is Ed25519-only."""
    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric.rsa import (
        generate_private_key,
    )
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.serialization import Encoding
    from cryptography.x509.oid import NameOID
    import datetime

    rsa = generate_private_key(public_exponent=65537, key_size=2048)
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "rsa-impostor"),
        ]))
        .issuer_name(x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "rsa-impostor"),
        ]))
        .public_key(rsa.public_key())
        .serial_number(1234)
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=10),
        )
        .sign(private_key=rsa, algorithm=hashes.SHA256())
    )
    pem = cert.public_bytes(Encoding.PEM)
    with pytest.raises(ValueError):
        p2p_tls.cert_pubkey_b64(pem)
