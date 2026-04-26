"""Regression: compute-pool worker registry — DB CRUD + invariants.

These tests cover commit #1 of the multi-PC feature:
  * Workers can be added with label / address / transport (lan / tailscale).
  * Capability snapshots and last-seen timestamps round-trip independently
    of user-facing edits (separated so the background probe loop and the
    Settings UI don't fight over `updated_at`).
  * Auth tokens stay server-side: the standard list / get rows include
    only an `auth_token_set` boolean — the secret itself is reachable
    only via `get_compute_worker_auth_token`.
  * Field validation rejects bad inputs (unknown transport, blank label /
    address, out-of-range port).

Routing tests (which worker handles a given request) come with the
capability-probe + routing commits — this file only pins the data
layer + API shape.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.smoke


# --- create / list / get --------------------------------------------------


def test_create_basic_lan_worker(isolated_db):
    wid = isolated_db.create_compute_worker(
        label="laptop",
        address="desktop-0692hok.local",
        transport="lan",
    )
    row = isolated_db.get_compute_worker(wid)
    assert row is not None
    assert row["label"] == "laptop"
    assert row["address"] == "desktop-0692hok.local"
    assert row["ollama_port"] == 11434
    assert row["transport"] == "lan"
    assert row["enabled"] is True
    # All three workload flags default to enabled.
    assert row["use_for_chat"] is True
    assert row["use_for_embeddings"] is True
    assert row["use_for_subagents"] is True
    # No probe has run yet.
    assert row["last_seen"] is None
    assert row["capabilities"] is None
    # No token set on this row.
    assert row["auth_token_set"] is False


def test_create_with_auth_token_keeps_it_internal(isolated_db):
    wid = isolated_db.create_compute_worker(
        label="t",
        address="100.91.9.91",
        transport="tailscale",
        auth_token="hunter2",
    )
    row = isolated_db.get_compute_worker(wid)
    # Boolean flag IS present on the row dict (so UI can render "set").
    assert row["auth_token_set"] is True
    # But the actual token MUST NOT appear anywhere in the row dict.
    assert "auth_token" not in row
    assert "hunter2" not in repr(row)
    # Internal fetcher still works for outbound requests.
    assert isolated_db.get_compute_worker_auth_token(wid) == "hunter2"


def test_list_returns_newest_first(isolated_db):
    a = isolated_db.create_compute_worker(label="A", address="a", transport="lan")
    import time; time.sleep(0.01)
    b = isolated_db.create_compute_worker(label="B", address="b", transport="lan")
    rows = isolated_db.list_compute_workers()
    assert [r["label"] for r in rows[:2]] == ["B", "A"]


def test_list_enabled_only(isolated_db):
    a = isolated_db.create_compute_worker(label="A", address="a", transport="lan", enabled=True)
    b = isolated_db.create_compute_worker(label="B", address="b", transport="lan", enabled=False)
    enabled = isolated_db.list_compute_workers(enabled_only=True)
    labels = {r["label"] for r in enabled}
    assert labels == {"A"}


# --- field validation ----------------------------------------------------


def test_blank_label_rejected(isolated_db):
    with pytest.raises(ValueError):
        isolated_db.create_compute_worker(label="", address="x", transport="lan")
    with pytest.raises(ValueError):
        isolated_db.create_compute_worker(label="  ", address="x", transport="lan")


def test_blank_address_rejected(isolated_db):
    with pytest.raises(ValueError):
        isolated_db.create_compute_worker(label="L", address="", transport="lan")


def test_unknown_transport_rejected(isolated_db):
    with pytest.raises(ValueError):
        isolated_db.create_compute_worker(
            label="L", address="x", transport="bogus",
        )


def test_port_clamped_to_valid_range(isolated_db):
    wid = isolated_db.create_compute_worker(
        label="L", address="x", transport="lan", ollama_port=99999,
    )
    assert isolated_db.get_compute_worker(wid)["ollama_port"] == 65535
    wid2 = isolated_db.create_compute_worker(
        label="L2", address="x", transport="lan", ollama_port=0,
    )
    assert isolated_db.get_compute_worker(wid2)["ollama_port"] == 1


# --- update ---------------------------------------------------------------


def test_update_partial_fields(isolated_db):
    wid = isolated_db.create_compute_worker(
        label="A", address="a.local", transport="lan",
    )
    updated = isolated_db.update_compute_worker(
        wid, label="A renamed", use_for_embeddings=False,
    )
    assert updated["label"] == "A renamed"
    assert updated["use_for_embeddings"] is False
    # Fields not patched are untouched.
    assert updated["address"] == "a.local"
    assert updated["use_for_chat"] is True


def test_update_rejects_invalid_transport(isolated_db):
    wid = isolated_db.create_compute_worker(label="A", address="a", transport="lan")
    with pytest.raises(ValueError):
        isolated_db.update_compute_worker(wid, transport="bogus")


def test_update_unknown_field_silently_ignored(isolated_db):
    """Patches with unknown keys (e.g. an old client sending a removed
    field) MUST NOT raise — they should ignore the unknown key and patch
    the rest. Otherwise rolling forward the schema breaks old clients."""
    wid = isolated_db.create_compute_worker(label="A", address="a", transport="lan")
    updated = isolated_db.update_compute_worker(
        wid, label="A2", legacy_field="goodbye",
    )
    assert updated["label"] == "A2"


# --- capability snapshot path --------------------------------------------


def test_update_capabilities_persists(isolated_db):
    wid = isolated_db.create_compute_worker(label="A", address="a", transport="lan")
    snapshot = {
        "models": ["gemma4:e4b", "llama3.1:8b"],
        "ram_gb": 16,
        "vram_gb": 8,
        "gpu": "RTX 3060 Ti",
    }
    isolated_db.update_compute_worker_capabilities(
        wid, capabilities=snapshot, last_seen=12345.0,
    )
    row = isolated_db.get_compute_worker(wid)
    assert row["capabilities"] == snapshot
    assert row["last_seen"] == 12345.0
    assert row["last_error"] is None


def test_update_capabilities_with_error(isolated_db):
    wid = isolated_db.create_compute_worker(label="A", address="a", transport="lan")
    isolated_db.update_compute_worker_capabilities(
        wid, last_error="connection refused", last_seen=99.0,
    )
    row = isolated_db.get_compute_worker(wid)
    assert row["last_error"] == "connection refused"
    assert row["last_seen"] == 99.0
    # Capabilities not touched by an error-only update.
    assert row["capabilities"] is None


def test_update_capabilities_clear_error(isolated_db):
    """Passing empty-string `last_error` clears it (success after a
    previous failure)."""
    wid = isolated_db.create_compute_worker(label="A", address="a", transport="lan")
    isolated_db.update_compute_worker_capabilities(wid, last_error="oops")
    isolated_db.update_compute_worker_capabilities(wid, last_error="")
    assert isolated_db.get_compute_worker(wid)["last_error"] is None


# --- delete ---------------------------------------------------------------


def test_delete_removes_row(isolated_db):
    wid = isolated_db.create_compute_worker(label="A", address="a", transport="lan")
    n = isolated_db.delete_compute_worker(wid)
    assert n == 1
    assert isolated_db.get_compute_worker(wid) is None


def test_delete_unknown_returns_zero(isolated_db):
    n = isolated_db.delete_compute_worker("nonexistent-id")
    assert n == 0
