"""Regression: `delegate_parallel` distributes subagents across host + workers.

The compute pool's second routing wire-up: when a parent agent issues
`delegate_parallel(tasks=[t1, t2, t3, ...])`, the subagent fan-out
schedules each task to host or one of the eligible workers in
round-robin order so the work actually parallelizes on hardware.

These tests verify the *targeting* layer — `run_subagents_parallel`
distributes correctly across `host + workers` — without spinning up
real Ollama. We monkeypatch `agent.run_subagent` itself to record
which (base_url, auth_token) each task was given; the chat streaming
path is covered separately by `test_compute_pool.py` and
`test_embed_routing.py`.

Invariants:
  * No registered workers → every task goes to OLLAMA_URL.
  * 1 host + 2 workers, 6 tasks → 2 each, host first.
  * Workers without the chat model → host-only fallback.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from backend import agent, compute_pool

pytestmark = pytest.mark.smoke


def _run(coro):
    return asyncio.run(coro)


def _seed_subagent_worker(isolated_db, *, label, address, model="gemma4:e4b", auth_token=None) -> str:
    """Helper: register a worker eligible for subagents with a fresh probe."""
    wid = isolated_db.create_compute_worker(
        label=label, address=address, transport="lan",
        use_for_subagents=True, auth_token=auth_token,
    )
    isolated_db.update_compute_worker_capabilities(
        wid,
        capabilities={
            "version": "0.5.4",
            "models": [{"name": model, "details": {}}],
        },
        last_seen=time.time() - 5.0,
        last_error="",
    )
    return wid


def _capture_run_subagent(monkeypatch):
    """Replace `agent.run_subagent` with a stub that records its kwargs.

    Returns the call-log list. Each entry is the dict of kwargs the stub
    was invoked with — `base_url` and `auth_token` are the fields the
    routing layer is responsible for; the rest is just bookkeeping the
    caller passes through unchanged.
    """
    calls: list[dict] = []

    async def _stub(*args, **kwargs):
        # Capture every call so we can assert distribution at the end.
        calls.append(dict(kwargs))
        return {"ok": True, "output": "stubbed"}

    monkeypatch.setattr(agent, "run_subagent", _stub)
    return calls


def test_parallel_routes_all_to_host_when_no_workers(isolated_db, monkeypatch):
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    calls = _capture_run_subagent(monkeypatch)

    res = _run(agent.run_subagents_parallel(
        tasks=["t1", "t2", "t3"],
        cwd=".",
        model="gemma4:e4b",
    ))

    assert res["ok"] is True
    assert len(calls) == 3
    # Every call must target the host's local Ollama.
    for c in calls:
        assert c["base_url"] == agent.OLLAMA_URL
        assert c["auth_token"] is None


def test_parallel_round_robins_across_host_and_workers(isolated_db, monkeypatch):
    """1 host + 2 workers, 6 tasks → exactly 2 per machine, host first."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_subagent_worker(isolated_db, label="A", address="a.local", auth_token="ta")
    _seed_subagent_worker(isolated_db, label="B", address="b.local", auth_token="tb")
    calls = _capture_run_subagent(monkeypatch)

    res = _run(agent.run_subagents_parallel(
        tasks=[f"t{i}" for i in range(6)],
        cwd=".",
        model="gemma4:e4b",
    ))
    assert res["ok"] is True
    assert len(calls) == 6

    # Tally targets — order on the wire isn't guaranteed (gather), but the
    # round-robin ASSIGNMENT is deterministic by index since we iterate
    # `enumerate(zip(cleaned, branch_ids))` in run_subagents_parallel. Group
    # by base_url to verify fairness.
    bases = [c["base_url"] for c in calls]
    counts = {b: bases.count(b) for b in set(bases)}
    assert counts == {
        agent.OLLAMA_URL: 2,
        "http://a.local:11434": 2,
        "http://b.local:11434": 2,
    }
    # Tokens must travel with the right URL.
    by_base = {c["base_url"]: c["auth_token"] for c in calls}
    assert by_base[agent.OLLAMA_URL] is None
    # Workers got ordered freshest-first by `list_subagent_workers`. Both
    # were seeded at the same fresh-age so order between A and B is a
    # tiebreak — what matters is that the right token rides with the URL.
    assert by_base["http://a.local:11434"] == "ta"
    assert by_base["http://b.local:11434"] == "tb"


def test_parallel_skips_workers_without_chat_model(isolated_db, monkeypatch):
    """Worker only has the embed model installed → routing for a chat
    fan-out must skip it and stay host-only."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_subagent_worker(
        isolated_db, label="A", address="a.local",
        model="nomic-embed-text:latest",
    )
    calls = _capture_run_subagent(monkeypatch)

    res = _run(agent.run_subagents_parallel(
        tasks=["t1", "t2"],
        cwd=".",
        model="gemma4:e4b",
    ))
    assert res["ok"] is True
    assert len(calls) == 2
    for c in calls:
        assert c["base_url"] == agent.OLLAMA_URL


def test_parallel_assigns_host_first_when_few_tasks(isolated_db, monkeypatch):
    """1 host + 1 worker, 1 task → host gets it (slot 0). Avoids waking a
    sleeping laptop just for a single subagent call."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _seed_subagent_worker(isolated_db, label="A", address="a.local")
    calls = _capture_run_subagent(monkeypatch)

    res = _run(agent.run_subagents_parallel(
        tasks=["only-one"],
        cwd=".",
        model="gemma4:e4b",
    ))
    assert res["ok"] is True
    assert len(calls) == 1
    assert calls[0]["base_url"] == agent.OLLAMA_URL
