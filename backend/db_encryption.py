"""At-rest encryption for the most-sensitive SQLite columns.

Threat model
============
Defends against a third party who obtains the SQLite file (`app.db`)
without also obtaining the user's `identity.json`. Concrete scenarios
this protects:

  * Cloud backup of the DB file leaks (e.g. iCloud / OneDrive / Drive
    backup misconfiguration). The attacker has the .db blob; they
    cannot read message content without identity.json.
  * Stolen USB drive with a manual DB copy. Same as above.
  * Malware that exfiltrates the .db but doesn't know to also pull
    identity.json from a different directory.
  * Forensic analysis of an old physical disk.

Does NOT defend against an attacker who has BOTH `app.db` AND
`identity.json` — they can run our key-derivation code and decrypt
everything. Defense against full-disk compromise requires a user
passphrase wrapping the master key (Settings → Security → "Lock
chat history with a password" — left as a future opt-in).

Why identity-derived keys
=========================
* No user prompt on every app launch — the chat app stays "just works".
* Same identity that already binds peer pairings; one secret to back up.
* If identity is rotated (re-install, broken file), encrypted rows
  become unreadable; identity.json is the single source of truth.

Wire format on the column
=========================
Encrypted strings have the prefix ``"\\x01gcae1:"`` followed by
url-safe base64 of ``nonce(12 bytes) || ChaCha20Poly1305_ciphertext``.
Plaintext rows (legacy / never-encrypted) lack the prefix and are
returned as-is by `decrypt()`. The 0x01 (SOH) leading byte is
effectively absent from real user-typed text, so the prefix collision
risk is negligible — and even if a user pastes content starting with
that exact byte sequence, decryption fails cleanly (AEAD tag
mismatch) and the original value is returned unchanged.

Per-call performance
====================
ChaCha20-Poly1305 on a modern CPU is ~1 GB/s. Typical chat message
is a few hundred bytes, so encryption/decryption is microseconds.
Master-key derivation is amortised over the process lifetime
(cached after first use).
"""

from __future__ import annotations

import base64
import logging
import os
import threading
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.serialization import (
    Encoding, PrivateFormat, NoEncryption,
)

log = logging.getLogger("gigachat.db.encryption")

# Prefix marker on encrypted column values. The 0x01 (SOH) leading
# byte is essentially never present at the start of user-typed text;
# combined with the literal "gcae1:" it's effectively unique. v1 in
# the prefix lets us bump the format if the AEAD or KDF changes.
_ENC_PREFIX = "\x01gcae1:"

# Master key length (ChaCha20-Poly1305 takes 32 bytes).
_KEY_LEN = 32

# Nonce length per RFC 8439 (12 bytes for ChaCha20-Poly1305).
_NONCE_LEN = 12

# HKDF info — bound to format version so a future scheme change
# can't accidentally reuse the same derived key under the new shape.
_HKDF_INFO = b"gigachat-db-at-rest-v1"

# Master key is derived once per process from identity. Held in
# module-level state with a lock so concurrent readers don't all
# race the derivation. Cleared on key-rotation (test paths).
_MASTER_KEY: bytes | None = None
_MASTER_KEY_LOCK = threading.Lock()


def _derive_master_key() -> bytes:
    """Derive the master at-rest key from the user's X25519 identity.

    Why X25519 priv bytes (not Ed25519)? Either works — both are
    32-byte private scalars with full entropy. We pick X25519
    arbitrarily; the key is bound to the install identity, not to
    any particular crypto purpose.

    HKDF salt is hard-coded; the secrecy lives in the priv-key
    bytes. info string binds to "v1 at-rest" so a future scheme
    bump (or a different at-rest table) won't collide.
    """
    # Lazy import to avoid circular dep — db_encryption is imported
    # FROM db.py, but identity is small and clean to import.
    from . import identity as _id

    me = _id.get_identity()
    # X25519 private keys don't expose raw_bytes() in cryptography
    # 41+; the documented way is to serialize PEM-PKCS8 and parse
    # back, OR use private_bytes(Raw, Raw). Raw is supported.
    priv_bytes = me.x25519_private.private_bytes(
        encoding=Encoding.Raw,
        format=PrivateFormat.Raw,
        encryption_algorithm=NoEncryption(),
    )
    return HKDF(
        algorithm=SHA256(),
        length=_KEY_LEN,
        salt=b"gigachat-db-at-rest-salt-v1",
        info=_HKDF_INFO,
    ).derive(priv_bytes)


def _master_key() -> bytes:
    """Return the cached master key, deriving on first call.

    Thread-safe: the lock guards the cache miss so concurrent
    readers do not all run HKDF redundantly.
    """
    global _MASTER_KEY
    if _MASTER_KEY is not None:
        return _MASTER_KEY
    with _MASTER_KEY_LOCK:
        if _MASTER_KEY is None:
            _MASTER_KEY = _derive_master_key()
    return _MASTER_KEY


def _reset_master_key_for_tests() -> None:
    """Drop the cached master key. Called by test fixtures that
    swap the identity so the next call re-derives against the
    fresh identity. Production code never invokes this."""
    global _MASTER_KEY
    with _MASTER_KEY_LOCK:
        _MASTER_KEY = None


# ---------------------------------------------------------------------------
# Public API: encrypt() / decrypt() / is_encrypted()
# ---------------------------------------------------------------------------


def is_encrypted(value: Any) -> bool:
    """True iff ``value`` looks like it was produced by `encrypt()`.

    Recognises the prefix marker only — does NOT attempt decryption.
    Cheap; safe to call in tight loops to skip already-handled rows
    in a migration pass.
    """
    return isinstance(value, str) and value.startswith(_ENC_PREFIX)


def encrypt(plaintext: Any) -> Any:
    """Wrap a string value with at-rest AEAD.

    Returns a new string of the form ``"\\x01gcae1:<urlsafe-b64>"``.
    Pass-through for None / non-string / already-encrypted inputs
    so callers can apply this unconditionally to columns that may
    legitimately hold None.

    Empty string maps to empty string (no point burning a nonce on
    nothing — and avoids the call site needing to special-case it).
    """
    if plaintext is None:
        return None
    if not isinstance(plaintext, str):
        # Defensive: callers should pass strings, but if they pass a
        # bytes/object, fall through unchanged. Encryption is an
        # additive guard, never a barrier to legitimate writes.
        return plaintext
    if not plaintext:
        return plaintext
    if is_encrypted(plaintext):
        return plaintext  # idempotent — never double-wrap
    nonce = os.urandom(_NONCE_LEN)
    cipher = ChaCha20Poly1305(_master_key())
    try:
        ct = cipher.encrypt(nonce, plaintext.encode("utf-8"), None)
    except Exception as e:
        # AEAD encrypt only fails on malformed inputs we just
        # constructed — defensive guard rather than a hot path.
        log.warning("db_encryption: encrypt failed (%s); storing plaintext", e)
        return plaintext
    blob = nonce + ct
    return _ENC_PREFIX + base64.urlsafe_b64encode(blob).decode("ascii")


def decrypt(value: Any) -> Any:
    """Reverse `encrypt()`. Pass-through for unencrypted values.

    On AEAD failure (corrupt ciphertext, wrong identity / lost key)
    the original value is returned and a WARNING is logged. The
    caller (typically `_row_to_message`) gets back something
    string-shaped either way so the UI doesn't crash mid-render.
    Failure path should be rare: it implies identity rotation, disk
    corruption, or a deliberate downgrade.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    if not is_encrypted(value):
        return value
    encoded = value[len(_ENC_PREFIX):]
    try:
        blob = base64.urlsafe_b64decode(encoded)
    except Exception:
        log.warning("db_encryption: decrypt b64 failed; returning ciphertext")
        return value
    if len(blob) < _NONCE_LEN + 16:  # 16 = AEAD tag size
        log.warning("db_encryption: ciphertext too short; returning as-is")
        return value
    nonce = blob[:_NONCE_LEN]
    ct = blob[_NONCE_LEN:]
    cipher = ChaCha20Poly1305(_master_key())
    try:
        plaintext = cipher.decrypt(nonce, ct, None)
    except InvalidTag:
        # Most likely cause: identity was regenerated since this row
        # was encrypted (e.g. identity.json deleted/corrupted). The
        # data is unrecoverable without the original priv key.
        log.warning(
            "db_encryption: AEAD tag mismatch — identity changed? "
            "row contents unrecoverable",
        )
        return value
    except Exception as e:
        log.warning("db_encryption: decrypt failed (%s); returning ciphertext", e)
        return value
    try:
        return plaintext.decode("utf-8")
    except Exception:
        # Plaintext was sealed as something other than UTF-8 string.
        # Not produced by our `encrypt()` path; return raw bytes
        # in repr form so the caller can see something rather than
        # nothing.
        return repr(plaintext)


# ---------------------------------------------------------------------------
# Bulk migration helper
# ---------------------------------------------------------------------------


def encrypt_legacy_rows(
    conn,
    *,
    table: str,
    column: str,
    id_column: str = "id",
    batch: int = 200,
) -> int:
    """Walk a table and encrypt unencrypted rows in-place.

    Used by a one-shot migration tool to upgrade pre-encryption
    databases. Idempotent: rows already carrying the encrypted
    prefix are skipped, so re-running is harmless.

    Returns the number of rows it actually mutated. Caller is
    responsible for the surrounding transaction (we don't commit
    here so a partial failure can be rolled back).
    """
    cursor = conn.execute(
        f"SELECT {id_column}, {column} FROM {table} WHERE {column} IS NOT NULL"
    )
    converted = 0
    pending: list[tuple[str, Any]] = []
    while True:
        rows = cursor.fetchmany(batch)
        if not rows:
            break
        for row in rows:
            rid = row[id_column] if hasattr(row, "keys") else row[0]
            val = row[column] if hasattr(row, "keys") else row[1]
            if not isinstance(val, str) or not val:
                continue
            if is_encrypted(val):
                continue
            new_val = encrypt(val)
            if new_val != val:
                pending.append((new_val, rid))
                converted += 1
            if len(pending) >= batch:
                conn.executemany(
                    f"UPDATE {table} SET {column} = ? WHERE {id_column} = ?",
                    pending,
                )
                pending.clear()
    if pending:
        conn.executemany(
            f"UPDATE {table} SET {column} = ? WHERE {id_column} = ?",
            pending,
        )
    return converted
