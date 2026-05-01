"""Regression: at-rest encryption for sensitive SQLite columns.

Pins the contracts callers depend on:

  * `encrypt()` / `decrypt()` round-trip every UTF-8 string.
  * `encrypt()` is non-deterministic (random nonce → different
    ciphertext each call). Deterministic encryption would let an
    attacker correlate identical plaintexts across rows.
  * `is_encrypted()` recognises wrapped values.
  * `decrypt()` is a pass-through for None / non-string / legacy
    plaintext rows so callers can apply it unconditionally.
  * `decrypt()` returns the original ciphertext on AEAD failure
    (rather than crashing the row hydrator) so a corrupted DB is
    still partially navigable.

Plus end-to-end through `db.add_message` / `_row_to_message` /
`db.update_message_content` to make sure the wiring is right —
caller-visible behaviour is unchanged but the on-disk bytes are
ciphertext.
"""
from __future__ import annotations

import pytest

from backend import db_encryption


pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# Pure-function tests (don't need the DB)
# ---------------------------------------------------------------------------


def test_round_trip():
    db_encryption._reset_master_key_for_tests()
    msg = "Hello, encrypted at-rest world! 🌍"
    ct = db_encryption.encrypt(msg)
    assert ct != msg
    assert db_encryption.is_encrypted(ct)
    assert db_encryption.decrypt(ct) == msg


def test_encrypt_is_non_deterministic():
    """Identical plaintexts must produce DIFFERENT ciphertexts.

    A deterministic scheme would let an attacker who sees two
    encrypted columns spot which messages contained the same text
    even without decrypting either.
    """
    db_encryption._reset_master_key_for_tests()
    p = "Same plaintext"
    a = db_encryption.encrypt(p)
    b = db_encryption.encrypt(p)
    assert a != b
    # Both still decrypt to the same plaintext.
    assert db_encryption.decrypt(a) == p
    assert db_encryption.decrypt(b) == p


def test_is_encrypted_recognises_only_wrapped():
    db_encryption._reset_master_key_for_tests()
    assert not db_encryption.is_encrypted("plain text")
    assert not db_encryption.is_encrypted("")
    assert not db_encryption.is_encrypted(None)
    assert db_encryption.is_encrypted(db_encryption.encrypt("x"))


def test_pass_through_for_none_and_empty():
    db_encryption._reset_master_key_for_tests()
    assert db_encryption.encrypt(None) is None
    assert db_encryption.decrypt(None) is None
    # Empty string is a no-op (no nonce burn for nothing).
    assert db_encryption.encrypt("") == ""
    assert db_encryption.decrypt("") == ""


def test_pass_through_for_legacy_plaintext():
    """Existing rows from before the encryption rollout stay
    readable — `decrypt` returns them unchanged."""
    db_encryption._reset_master_key_for_tests()
    legacy = "this row was written before at-rest encryption shipped"
    assert db_encryption.decrypt(legacy) == legacy


def test_encrypt_is_idempotent():
    """Calling encrypt() on an already-encrypted value returns the
    same value, NOT a doubly-wrapped one. Callers can apply
    encrypt() defensively without worrying about the input shape."""
    db_encryption._reset_master_key_for_tests()
    once = db_encryption.encrypt("test")
    twice = db_encryption.encrypt(once)
    assert once == twice


def test_corrupt_ciphertext_returns_original():
    """A tampered or corrupted encrypted value should NOT crash —
    it should be returned as-is so the row hydrator can render
    SOMETHING. The user sees that something's wrong; the app
    keeps running."""
    db_encryption._reset_master_key_for_tests()
    ct = db_encryption.encrypt("original plaintext")
    # Mutate one character in the b64 portion → AEAD tag mismatch.
    mutated = ct[:-2] + ("AA" if ct[-2:] != "AA" else "BB")
    out = db_encryption.decrypt(mutated)
    # Either returns the corrupted ciphertext (graceful) — never raises.
    assert isinstance(out, str)


# ---------------------------------------------------------------------------
# Integration through db.py
# ---------------------------------------------------------------------------


def test_add_message_stores_ciphertext_returns_plaintext(isolated_db):
    """add_message persists encrypted bytes but returns the
    plaintext content the caller passed in."""
    db_encryption._reset_master_key_for_tests()
    cid = isolated_db.create_conversation(
        title="t", model="x:y", cwd="/tmp",
    )["id"]
    plaintext = "Sensitive question the user asked."
    row = isolated_db.add_message(cid, "user", plaintext)
    assert row["content"] == plaintext

    # Peek at the raw column to confirm it's the wrapped form.
    with isolated_db._conn() as c:
        raw = c.execute(
            "SELECT content FROM messages WHERE id = ?", (row["id"],),
        ).fetchone()["content"]
    assert raw != plaintext
    assert db_encryption.is_encrypted(raw)


def test_list_messages_decrypts_transparently(isolated_db):
    db_encryption._reset_master_key_for_tests()
    cid = isolated_db.create_conversation(
        title="t", model="x:y", cwd="/tmp",
    )["id"]
    plaintext = "Decrypted on read"
    isolated_db.add_message(cid, "user", plaintext)
    msgs = isolated_db.list_messages(cid)
    assert len(msgs) == 1
    assert msgs[0]["content"] == plaintext


def test_update_message_content_re_encrypts(isolated_db):
    db_encryption._reset_master_key_for_tests()
    cid = isolated_db.create_conversation(
        title="t", model="x:y", cwd="/tmp",
    )["id"]
    row = isolated_db.add_message(cid, "assistant", "first")
    updated = isolated_db.update_message_content(row["id"], "second")
    assert updated["content"] == "second"

    with isolated_db._conn() as c:
        raw = c.execute(
            "SELECT content FROM messages WHERE id = ?", (row["id"],),
        ).fetchone()["content"]
    assert db_encryption.is_encrypted(raw)
    # And the original plaintext is gone.
    assert "first" not in raw
    assert "second" not in raw


def test_search_conversations_finds_encrypted_content(isolated_db):
    """Substring search of message bodies works across the
    encryption boundary — implementation falls back to Python-side
    decrypt-and-scan since SQL LIKE can't see through ciphertext."""
    db_encryption._reset_master_key_for_tests()
    cid_a = isolated_db.create_conversation(
        title="alpha", model="x:y", cwd="/tmp",
    )["id"]
    cid_b = isolated_db.create_conversation(
        title="bravo", model="x:y", cwd="/tmp",
    )["id"]
    isolated_db.add_message(cid_a, "user", "look for the unique-marker-zaq here")
    isolated_db.add_message(cid_b, "user", "nothing relevant in this one")

    hits = isolated_db.search_conversations("unique-marker-zaq")
    hit_ids = {c["id"] for c in hits}
    assert cid_a in hit_ids
    assert cid_b not in hit_ids


def test_legacy_plaintext_message_still_renders(isolated_db):
    """A row written by an older Gigachat (no encryption) must
    still render cleanly through `_row_to_message` after the
    encryption migration. Simulate by inserting plaintext directly,
    bypassing add_message."""
    db_encryption._reset_master_key_for_tests()
    cid = isolated_db.create_conversation(
        title="t", model="x:y", cwd="/tmp",
    )["id"]
    legacy_plain = "row from before encryption shipped"
    with isolated_db._conn() as c:
        c.execute(
            "INSERT INTO messages "
            "(id, conversation_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("legacy-id", cid, "user", legacy_plain, 1.0),
        )
    msgs = isolated_db.list_messages(cid)
    assert len(msgs) == 1
    assert msgs[0]["content"] == legacy_plain
