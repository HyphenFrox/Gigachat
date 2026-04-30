"""Shared test fixtures.

Every test that touches the DB gets a fresh, isolated SQLite file under a
temporary directory. We monkey-patch `db.DB_PATH` BEFORE importing anything
that depends on it, so module-level state inside `db` starts clean per test.

This keeps tests fast (no real Ollama, no real network) and deterministic
(no shared state between runs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


# Make the project root importable so `from backend import …` works when
# pytest is run from the project root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """Return a freshly-initialized `db` module pointing at a temp file.

    Use this fixture in any test that calls `db.create_conversation`,
    `db.add_message`, etc. The file is deleted automatically when the test
    finishes (tmp_path cleanup is handled by pytest).

    Also resets module-level routing state (`_HOST_MEGA_MODEL_BUSY_UNTIL`)
    so an earlier test that engaged the mega-model path doesn't leak its
    "host is busy" flag into a routing test that expects the host to be
    in the rotation. Strictly an isolation hygiene fix, not a behaviour
    change in production.
    """
    from backend import db as _db
    from backend import compute_pool as _cp

    db_path = tmp_path / "test_app.db"
    monkeypatch.setattr(_db, "DB_PATH", db_path)
    _db.init()
    _cp._HOST_MEGA_MODEL_BUSY_UNTIL = 0.0
    return _db


@pytest.fixture()
def fresh_agent_queue(isolated_db):
    """Provide the `agent` module with a clean DB-backed user-input queue.

    The queue used to be an in-memory dict inside `agent`; it's now backed by
    a `queued_inputs` SQLite table so messages survive a crash. This fixture
    therefore just forwards `isolated_db` (which gives the queue a fresh
    table) and returns the agent module so tests read naturally.
    """
    from backend import agent as _agent

    return _agent
