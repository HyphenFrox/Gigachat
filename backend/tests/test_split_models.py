"""Regression: split-model registry — DB CRUD + invariants.

Phase 2 of the compute-pool feature. A "split model" is a single GGUF
served by a host-local `llama-server` with `--rpc <worker>:<port>`
flags so layers fan across one or more compute workers. This file
pins the data layer; lifecycle (start/stop the server, surface
errors) tests come with the lifecycle commit.

Things this file checks:
  * Create / list / get round-trip the user-facing fields cleanly.
  * `worker_ids` is stored as JSON and rehydrated as a Python list.
  * Field validation rejects the obvious foot-guns: blank label /
    blank gguf path / non-list worker_ids / out-of-range port.
  * `update_split_model` is user-facing — it MUST NOT touch
    `status` or `last_error`. Those belong to the runtime layer
    (`update_split_model_status`) so a malicious / buggy client can
    never lie about a server being "running" when nothing is bound.
  * `update_split_model_status` accepts only known statuses.
  * Status updates clear `last_error` cleanly when passed `""`,
    leave it alone when passed `None`.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.smoke


# --- create / list / get --------------------------------------------------


def test_create_basic(isolated_db):
    sid = isolated_db.create_split_model(
        label="big-q3",
        gguf_path="C:/path/to/model.gguf",
    )
    row = isolated_db.get_split_model(sid)
    assert row is not None
    assert row["label"] == "big-q3"
    assert row["gguf_path"] == "C:/path/to/model.gguf"
    assert row["worker_ids"] == []
    assert row["llama_port"] == 11500
    assert row["enabled"] is True
    # Never created in a "live" state — lifecycle module flips this.
    assert row["status"] == "stopped"
    assert row["last_error"] is None


def test_create_with_worker_ids_persists_as_list(isolated_db):
    sid = isolated_db.create_split_model(
        label="m",
        gguf_path="/m.gguf",
        worker_ids=["w1", "w2"],
    )
    row = isolated_db.get_split_model(sid)
    assert row["worker_ids"] == ["w1", "w2"]


def test_list_returns_newest_first(isolated_db):
    isolated_db.create_split_model(label="A", gguf_path="/a.gguf")
    import time
    time.sleep(0.01)
    isolated_db.create_split_model(label="B", gguf_path="/b.gguf")
    rows = isolated_db.list_split_models()
    assert [r["label"] for r in rows[:2]] == ["B", "A"]


def test_list_enabled_only(isolated_db):
    isolated_db.create_split_model(label="A", gguf_path="/a.gguf", enabled=True)
    isolated_db.create_split_model(label="B", gguf_path="/b.gguf", enabled=False)
    rows = isolated_db.list_split_models(enabled_only=True)
    assert {r["label"] for r in rows} == {"A"}


# --- field validation ----------------------------------------------------


def test_blank_label_rejected(isolated_db):
    with pytest.raises(ValueError):
        isolated_db.create_split_model(label="", gguf_path="/a.gguf")
    with pytest.raises(ValueError):
        isolated_db.create_split_model(label="   ", gguf_path="/a.gguf")


def test_blank_gguf_path_rejected(isolated_db):
    with pytest.raises(ValueError):
        isolated_db.create_split_model(label="A", gguf_path="")
    with pytest.raises(ValueError):
        isolated_db.create_split_model(label="A", gguf_path="   ")


def test_non_string_worker_id_rejected(isolated_db):
    with pytest.raises(ValueError):
        isolated_db.create_split_model(
            label="A", gguf_path="/a.gguf", worker_ids=["w1", 42],
        )
    with pytest.raises(ValueError):
        isolated_db.create_split_model(
            label="A", gguf_path="/a.gguf", worker_ids=[""],
        )


def test_port_out_of_range_rejected(isolated_db):
    with pytest.raises(ValueError):
        isolated_db.create_split_model(
            label="A", gguf_path="/a.gguf", llama_port=0,
        )
    with pytest.raises(ValueError):
        isolated_db.create_split_model(
            label="A", gguf_path="/a.gguf", llama_port=99999,
        )


# --- update --------------------------------------------------------------


def test_update_partial_fields(isolated_db):
    sid = isolated_db.create_split_model(label="A", gguf_path="/a.gguf")
    updated = isolated_db.update_split_model(
        sid, label="A renamed", llama_port=11600,
    )
    assert updated["label"] == "A renamed"
    assert updated["llama_port"] == 11600
    # Untouched fields stay.
    assert updated["gguf_path"] == "/a.gguf"
    assert updated["enabled"] is True


def test_update_replaces_worker_ids(isolated_db):
    sid = isolated_db.create_split_model(
        label="A", gguf_path="/a.gguf", worker_ids=["w1", "w2"],
    )
    updated = isolated_db.update_split_model(sid, worker_ids=["w3"])
    assert updated["worker_ids"] == ["w3"]


def test_update_unknown_field_silently_ignored(isolated_db):
    """Unknown keys (e.g. an old client sending a removed field) MUST
    NOT raise — patch the rest, ignore the unknown. Same contract as
    the compute_workers patch endpoint."""
    sid = isolated_db.create_split_model(label="A", gguf_path="/a.gguf")
    updated = isolated_db.update_split_model(
        sid, label="A2", legacy_field="goodbye",
    )
    assert updated["label"] == "A2"


def test_update_cannot_set_status_directly(isolated_db):
    """The user-facing patch endpoint MUST NOT touch status. Otherwise
    a client could falsely flip a stopped row to 'running' and fool the
    routing layer into sending chat to a dead llama-server."""
    sid = isolated_db.create_split_model(label="A", gguf_path="/a.gguf")
    isolated_db.update_split_model(sid, status="running")
    assert isolated_db.get_split_model(sid)["status"] == "stopped"


# --- runtime status path -------------------------------------------------


def test_status_round_trip(isolated_db):
    sid = isolated_db.create_split_model(label="A", gguf_path="/a.gguf")
    isolated_db.update_split_model_status(sid, status="loading")
    assert isolated_db.get_split_model(sid)["status"] == "loading"
    isolated_db.update_split_model_status(sid, status="running")
    assert isolated_db.get_split_model(sid)["status"] == "running"


def test_status_rejects_unknown_value(isolated_db):
    sid = isolated_db.create_split_model(label="A", gguf_path="/a.gguf")
    with pytest.raises(ValueError):
        isolated_db.update_split_model_status(sid, status="exploded")


def test_status_with_error_then_clear(isolated_db):
    sid = isolated_db.create_split_model(label="A", gguf_path="/a.gguf")
    isolated_db.update_split_model_status(
        sid, status="error", last_error="port already in use",
    )
    row = isolated_db.get_split_model(sid)
    assert row["status"] == "error"
    assert row["last_error"] == "port already in use"
    # Empty string clears it; None would leave it alone.
    isolated_db.update_split_model_status(sid, status="running", last_error="")
    row = isolated_db.get_split_model(sid)
    assert row["status"] == "running"
    assert row["last_error"] is None


def test_status_last_error_none_leaves_alone(isolated_db):
    sid = isolated_db.create_split_model(label="A", gguf_path="/a.gguf")
    isolated_db.update_split_model_status(sid, status="error", last_error="oops")
    # last_error=None default — should NOT wipe the previous error.
    isolated_db.update_split_model_status(sid, status="loading")
    row = isolated_db.get_split_model(sid)
    assert row["status"] == "loading"
    assert row["last_error"] == "oops"


# --- delete --------------------------------------------------------------


def test_delete_removes_row(isolated_db):
    sid = isolated_db.create_split_model(label="A", gguf_path="/a.gguf")
    n = isolated_db.delete_split_model(sid)
    assert n == 1
    assert isolated_db.get_split_model(sid) is None


def test_delete_unknown_returns_zero(isolated_db):
    n = isolated_db.delete_split_model("nonexistent-id")
    assert n == 0
