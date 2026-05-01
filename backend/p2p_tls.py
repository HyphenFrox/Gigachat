"""TLS-with-pinning for streaming P2P traffic.

Why this exists
===============
The custom envelope (`p2p_crypto`) gives confidentiality, integrity,
and sender auth per message. For one-shot calls that's exactly what
you want — single cryptographic operation, no handshake overhead.

For streaming (chat completions, long NDJSON token streams) the
envelope-per-chunk pattern adds overhead AND only gives PARTIAL
forward secrecy (sender ephemeral; recipient compromise still
reveals past traffic). TLS 1.3 with ECDHE handshake gives:

  * **Full forward secrecy** in BOTH directions — the handshake
    exchanges ephemeral keys on both sides, so neither side's
    long-term key compromise leaks past traffic.
  * **Mutual authentication** via mTLS — both peers present an
    identity certificate; we pin each cert to the peer's expected
    Ed25519 pubkey. An attacker who steals a peer's cert can't
    impersonate them without ALSO stealing the matching priv.
  * **Standardised protocol** — well-vetted code path (OpenSSL),
    no per-chunk crypto arithmetic.

What this module provides
=========================
1. `ensure_identity_cert()` — generate (once) a self-signed X.509
   cert bound to our Ed25519 identity. Stored at
   ``~/.gigachat/identity-cert.pem`` alongside `identity.json`.
2. `cert_pubkey_b64(cert_bytes)` — extract the cert's Ed25519
   subject pubkey for pinning lookups.
3. `verify_peer_cert(cert_der, expected_ed25519_pub_b64)` —
   constant-time check that a presented cert's Subject Pubkey
   matches what we expected.
4. `make_pinned_client_ssl_context(expected_pub_b64)` — SSL context
   for httpx that accepts ANY cert chain BUT enforces pin match
   against `expected_pub_b64` after the handshake.

Why pinning, not a CA?
======================
This is a P2P swarm, not a corporate intranet. There's no CA we
trust globally. Pinning the cert's pubkey to the value we already
have in `paired_devices` / `pool_inventory` is the simplest way to
get authenticated TLS without a CA.

Server-side mTLS comes in a follow-up — keeping this commit
narrowly scoped to the cert + outbound-pin pieces so the migration
can roll out behind a feature flag instead of a flag-day cutover.
"""

from __future__ import annotations

import base64
import datetime
import logging
import os
import ssl
import stat
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import (
    Encoding, NoEncryption, PrivateFormat, PublicFormat,
)
from cryptography.x509.oid import NameOID

from . import identity as _identity

log = logging.getLogger("gigachat.p2p.tls")

# Where the identity cert lives. Co-located with identity.json so the
# user only has one file to back up. Same 0600 perms applied (POSIX).
_CERT_PATH = _identity._IDENTITY_PATH.parent / "identity-cert.pem"
_CERT_KEY_PATH = _identity._IDENTITY_PATH.parent / "identity-cert-key.pem"

# Cert validity. We auto-rotate to avoid the 1-yr browser cap,
# so 365 days is fine. Identity (Ed25519 keypair) is what's actually
# trusted; the cert wrapping it is just X.509 plumbing for TLS.
_CERT_VALIDITY_DAYS = 365


def _cert_subject(device_id: str) -> x509.Name:
    """X.509 subject — purely informational, not used for trust.

    Trust comes from the pinned Ed25519 pubkey, not from the X.509
    name fields. We populate CN with the device_id so a tcpdump /
    Wireshark peek shows which peer the cert belongs to.
    """
    return x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, device_id),
        x509.NameAttribute(
            NameOID.ORGANIZATION_NAME, "Gigachat P2P",
        ),
    ])


def ensure_identity_cert() -> tuple[Path, Path]:
    """Generate (or load) the self-signed identity cert.

    Returns the (cert_path, key_path) tuple. Idempotent — if the
    files already exist AND match the current identity's Ed25519
    pubkey, they're reused. Otherwise we regenerate (e.g. identity
    was rotated, cert expired).

    The cert key IS the identity's Ed25519 keypair — we don't
    generate a separate one. That way the cert pin is the same as
    the pubkey in paired_devices, and pinning logic stays trivial.
    """
    me = _identity.get_identity()

    if _CERT_PATH.exists() and _CERT_KEY_PATH.exists():
        try:
            existing = x509.load_pem_x509_certificate(_CERT_PATH.read_bytes())
            existing_pub = existing.public_key().public_bytes(
                encoding=Encoding.Raw, format=PublicFormat.Raw,
            )
            if base64.b64encode(existing_pub).decode("ascii") == me.public_key_b64:
                # Also check expiry — leave a generous buffer so a
                # certificate doesn't lapse mid-session.
                now = datetime.datetime.now(datetime.timezone.utc)
                if existing.not_valid_after_utc > now + datetime.timedelta(days=30):
                    return _CERT_PATH, _CERT_KEY_PATH
        except Exception as e:
            log.info(
                "p2p_tls: existing cert failed to load (%s); regenerating",
                e,
            )

    cert = (
        x509.CertificateBuilder()
        .subject_name(_cert_subject(me.device_id))
        .issuer_name(_cert_subject(me.device_id))  # self-signed
        .public_key(me.private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=_CERT_VALIDITY_DAYS)
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .add_extension(
            # Generic SAN so OpenSSL doesn't complain about hostname
            # mismatch — pinning is what we use for trust, not the
            # SAN name.
            x509.SubjectAlternativeName([
                x509.DNSName("gigachat-peer"),
            ]),
            critical=False,
        )
        .sign(private_key=me.private_key, algorithm=None)
    )

    cert_pem = cert.public_bytes(Encoding.PEM)
    key_pem = me.private_key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.PKCS8,
        encryption_algorithm=NoEncryption(),
    )

    _CERT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CERT_PATH.write_bytes(cert_pem)
    _CERT_KEY_PATH.write_bytes(key_pem)
    if sys.platform != "win32":
        try:
            os.chmod(_CERT_KEY_PATH, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
    log.info(
        "p2p_tls: generated identity cert for device_id=%s (CN=%s, valid=%dd)",
        me.device_id, me.device_id, _CERT_VALIDITY_DAYS,
    )
    return _CERT_PATH, _CERT_KEY_PATH


# ---------------------------------------------------------------------------
# Pin helpers — extract pubkey from a cert + verify pin match
# ---------------------------------------------------------------------------


def cert_pubkey_b64(cert_pem_or_der: bytes) -> str:
    """Extract the cert's Ed25519 subject pubkey, base64-encoded.

    Accepts either PEM or DER bytes. Returns the same base64 form
    we use everywhere else (raw 32-byte Ed25519 pubkey, base64
    standard alphabet). Comparison is then a string compare.

    Raises ``ValueError`` on malformed cert OR on a non-Ed25519
    subject (we deliberately accept only Ed25519 pubkeys; an RSA
    cert means the peer is using a different identity scheme).
    """
    if not cert_pem_or_der:
        raise ValueError("cert bytes are empty")
    try:
        if cert_pem_or_der.lstrip().startswith(b"-----BEGIN"):
            cert = x509.load_pem_x509_certificate(cert_pem_or_der)
        else:
            cert = x509.load_der_x509_certificate(cert_pem_or_der)
    except Exception as e:
        raise ValueError(f"could not parse cert: {e}")
    pub = cert.public_key()
    try:
        raw = pub.public_bytes(
            encoding=Encoding.Raw, format=PublicFormat.Raw,
        )
    except Exception:
        raise ValueError("cert subject is not an Ed25519 pubkey")
    if len(raw) != 32:
        raise ValueError(
            f"cert subject pubkey is the wrong length ({len(raw)} != 32)"
        )
    return base64.b64encode(raw).decode("ascii")


def verify_peer_cert(
    cert_pem_or_der: bytes, expected_ed25519_pub_b64: str,
) -> bool:
    """Constant-time pin check for a presented peer cert.

    Compares the cert's Ed25519 subject pubkey against the value we
    already have on file (from paired_devices / inventory cache /
    rendezvous /peers). Returns True iff they match exactly.

    Constant-time compare via `hmac.compare_digest` so a network
    timing attacker can't extract the expected pubkey one byte at
    a time by measuring response latency.
    """
    import hmac
    if not expected_ed25519_pub_b64:
        return False
    try:
        actual = cert_pubkey_b64(cert_pem_or_der)
    except ValueError as e:
        log.debug("p2p_tls: cert pubkey extract failed: %s", e)
        return False
    return hmac.compare_digest(actual, expected_ed25519_pub_b64)


# ---------------------------------------------------------------------------
# SSL contexts — server / client
# ---------------------------------------------------------------------------


def make_server_ssl_context() -> ssl.SSLContext:
    """SSL context for the FastAPI/uvicorn server.

    Uses our identity cert + key. Pure server-mode — clients verify
    OUR cert against their pin; we don't currently demand a client
    cert (mTLS), but the server-mode context here is the foundation
    for that follow-up.
    """
    cert_path, key_path = ensure_identity_cert()
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    # Disable compression (CRIME mitigation) and old protocols.
    # Python's create_default_context already does this in 3.10+, but
    # being explicit makes the policy auditable.
    ctx.options |= ssl.OP_NO_COMPRESSION
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    return ctx


def make_pinned_client_ssl_context(
    expected_peer_ed25519_pub_b64: str,
) -> ssl.SSLContext:
    """SSL context that accepts ANY cert chain BUT enforces a pin
    match after the TLS handshake completes.

    Why ANY chain: peer certs are self-signed; there's no CA we
    trust to validate them. The trust anchor is the pubkey pin we
    have on file, not the X.509 chain.

    Why pinning rather than skipping verification entirely: skipping
    verification leaves us open to MITM where any cert with the
    right hostname (we use a generic SAN, so basically any cert)
    would be accepted. The pin closes that — the handshake
    completes against any cert, then the caller MUST call
    `verify_peer_cert(socket.getpeercert(binary_form=True), ...)`
    to confirm the pin BEFORE sending the request body.

    Caller is responsible for the post-handshake pin check —
    enforcing it inside the SSL context's verify_callback would
    require a non-trivial wrapper around httpx's transport layer.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # we verify via pin, not chain
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2  # 1.3 not always avail
    if not expected_peer_ed25519_pub_b64:
        # Caller asked for a context but didn't supply a pin —
        # this would let any cert through silently. Refuse rather
        # than pretend to do pinning.
        raise ValueError("expected pubkey is required for pinned context")
    return ctx
