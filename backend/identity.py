"""Per-install Ed25519 identity for the P2P compute pool.

Generated on first launch and persisted to ``data/identity.json`` with
mode 0600 so other users on a shared machine can't read it. The public
half doubles as the device's network identity (`device_id`, a base32
prefix of the public key) — used by mDNS pairing, friend lookups,
signed receipts, and the future internet-P2P transport.

This module deliberately has zero runtime dependencies on the rest of
the app. It can be imported at process start before the DB is open
and before any other state is touched.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import socket
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding, NoEncryption, PrivateFormat, PublicFormat,
)

log = logging.getLogger("gigachat.identity")

# Persistent state lives next to the SQLite DB. Same disk hygiene story
# as `data/vapid.json` (the existing Web Push keypair).
_IDENTITY_PATH = Path(os.environ.get("GIGACHAT_DATA_DIR", "data")) / "identity.json"


@dataclass(frozen=True)
class Identity:
    """The four pieces of state a peer needs to act in the P2P network.

    `device_id` is short (16 chars, base32) and human-quotable — what the
    user reads off the screen when verifying a friend's identity.
    `public_key_b64` is the full 32-byte Ed25519 public key (base64), the
    canonical wire-format identifier used in pairing / receipts / handshakes.
    `label` is a friendly name (defaults to the OS hostname) shown next to
    the id in pairing UIs so users can tell their devices apart.
    """
    device_id: str
    label: str
    public_key_b64: str
    private_key: Ed25519PrivateKey

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self.private_key.public_key()

    def public_key_bytes(self) -> bytes:
        return self.public_key.public_bytes(
            encoding=Encoding.Raw, format=PublicFormat.Raw,
        )

    def sign(self, message: bytes) -> bytes:
        """Ed25519 sign — used by pairing handshake + receipt signing."""
        return self.private_key.sign(message)


def _default_label() -> str:
    """Best-effort device name for the UI. Falls back to a generic
    string when the platform doesn't expose a hostname (rare; usually
    a sandboxed container)."""
    try:
        name = socket.gethostname() or ""
    except OSError:
        name = ""
    name = name.strip()
    if not name:
        name = f"gigachat-{sys.platform}"
    # Trim a trailing ".local" / ".lan" — common Windows / Bonjour
    # suffixes that look ugly in the UI.
    for suffix in (".local", ".lan", ".home"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name[:64]


def _device_id_from_pubkey(pubkey_bytes: bytes) -> str:
    """Render a public key as a short, human-quotable identifier.

    Base32 (RFC 4648) without padding, uppercased, first 16 chars
    (≈80 bits — enough collision resistance for the discovery layer
    even on a swarm of millions of devices). Looks like
    ``GLAB-TZQ2-K7VA-X5R3`` when grouped for display.
    """
    s = base64.b32encode(pubkey_bytes).decode("ascii").rstrip("=")
    return s[:16].upper()


def format_device_id(device_id: str) -> str:
    """Human-readable form: groups of 4 separated by hyphens.

    Used for display only; the canonical persisted form is the raw
    16-char string so JSON / DB storage is compact.
    """
    s = (device_id or "").replace("-", "").upper()
    return "-".join(s[i:i + 4] for i in range(0, len(s), 4)) if s else ""


def _save_identity(priv: Ed25519PrivateKey, label: str) -> None:
    """Write identity to disk with conservative file permissions.

    Format is plain JSON (private key DER-base64) so a future tool
    that needs to read it without importing cryptography can do so.
    On POSIX we chmod the file to 0600. On Windows the equivalent is
    "current user only" via ACLs — we don't try to set those (would
    require pywin32) but the data dir itself is in the user's home,
    so multi-user disclosure is unlikely.
    """
    _IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    der = priv.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.PKCS8,
        encryption_algorithm=NoEncryption(),
    ).decode("ascii")
    pub_bytes = priv.public_key().public_bytes(
        encoding=Encoding.Raw, format=PublicFormat.Raw,
    )
    payload = {
        "version": 1,
        "label": label,
        "device_id": _device_id_from_pubkey(pub_bytes),
        "public_key_b64": base64.b64encode(pub_bytes).decode("ascii"),
        "private_key_pem": der,
    }
    tmp = _IDENTITY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if sys.platform != "win32":
        try:
            os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)  # 0600
        except OSError:
            pass
    os.replace(tmp, _IDENTITY_PATH)


_CACHED: Identity | None = None


def get_identity() -> Identity:
    """Return the singleton identity, generating it on first call.

    Subsequent calls return the cached object — keypair is
    immutable for the lifetime of the install.
    """
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    if _IDENTITY_PATH.exists():
        try:
            payload = json.loads(_IDENTITY_PATH.read_text(encoding="utf-8"))
            from cryptography.hazmat.primitives.serialization import (
                load_pem_private_key,
            )
            priv = load_pem_private_key(
                payload["private_key_pem"].encode("ascii"),
                password=None,
            )
            if not isinstance(priv, Ed25519PrivateKey):
                raise ValueError("identity.json holds a non-Ed25519 key")
            label = str(payload.get("label") or _default_label())
            _CACHED = Identity(
                device_id=str(payload["device_id"]),
                label=label,
                public_key_b64=str(payload["public_key_b64"]),
                private_key=priv,
            )
            return _CACHED
        except Exception as e:
            # Corrupt or unreadable identity — back it up and regenerate.
            # Losing identity is mild (you have to re-pair devices) but
            # acceptable when the alternative is the app refusing to start.
            log.warning(
                "identity: failed to load %s (%s); regenerating",
                _IDENTITY_PATH, e,
            )
            try:
                _IDENTITY_PATH.rename(
                    _IDENTITY_PATH.with_suffix(".broken")
                )
            except OSError:
                pass

    # First launch: generate + persist.
    priv = Ed25519PrivateKey.generate()
    label = _default_label()
    _save_identity(priv, label)
    pub_bytes = priv.public_key().public_bytes(
        encoding=Encoding.Raw, format=PublicFormat.Raw,
    )
    _CACHED = Identity(
        device_id=_device_id_from_pubkey(pub_bytes),
        label=label,
        public_key_b64=base64.b64encode(pub_bytes).decode("ascii"),
        private_key=priv,
    )
    log.info(
        "identity: generated new Ed25519 identity %s (%s)",
        _CACHED.device_id, _CACHED.label,
    )
    return _CACHED


def set_label(new_label: str) -> Identity:
    """Update the user-facing device label and re-persist.

    Identity (the keypair) is unchanged — only the friendly name is
    updated. Used by the Settings UI when the user wants their
    devices to show up as e.g. "Office desktop" rather than
    `DESKTOP-AB4XQZ`.
    """
    label = (new_label or "").strip() or _default_label()
    label = label[:64]
    cur = get_identity()
    if label == cur.label:
        return cur
    _save_identity(cur.private_key, label)
    global _CACHED
    _CACHED = Identity(
        device_id=cur.device_id,
        label=label,
        public_key_b64=cur.public_key_b64,
        private_key=cur.private_key,
    )
    log.info("identity: label updated to %r", label)
    return _CACHED


def verify_signature(public_key_b64: str, message: bytes, signature: bytes) -> bool:
    """Convenience: check an Ed25519 signature against a base64 pubkey.

    Used by the pairing handshake (verifying the claimant proved
    knowledge of the PIN) and by the receipt-validation path. Returns
    False on any decode / signature error rather than raising — most
    callers want the boolean rather than an exception ladder.
    """
    try:
        from cryptography.exceptions import InvalidSignature
        pub_bytes = base64.b64decode(public_key_b64)
        pub = Ed25519PublicKey.from_public_bytes(pub_bytes)
        try:
            pub.verify(signature, message)
            return True
        except InvalidSignature:
            return False
    except Exception:
        return False
