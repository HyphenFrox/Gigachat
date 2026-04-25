"""Regression: workflow-extended hooks (`tool_error`,
`consecutive_failures`, `max_fires_per_conv` cap, longer timeout cap).

These pin the user-facing contract for the workflow feature:

  * The new event names are accepted by `create_hook` and rejected if
    misspelled.
  * `error_threshold` and `max_fires_per_conv` round-trip through CRUD
    and are clamped to sane bounds.
  * The fire counter increments per-(hook, conv) and survives reads.
  * Timeout caps are event-specific — diagnosis events get 900 s.
"""
from __future__ import annotations

import pytest

from backend import db

pytestmark = pytest.mark.smoke


# --- new event names ------------------------------------------------------


def test_new_hook_events_accepted():
    """`tool_error` and `consecutive_failures` join the existing event set."""
    assert "tool_error" in db.HOOK_EVENTS
    assert "consecutive_failures" in db.HOOK_EVENTS


def test_create_hook_rejects_unknown_event(isolated_db):
    with pytest.raises(ValueError):
        isolated_db.create_hook(event="nope", command="echo hi")


def test_create_hook_with_error_threshold(isolated_db):
    hid = isolated_db.create_hook(
        event="consecutive_failures",
        command="echo three-in-a-row",
        error_threshold=3,
    )
    row = isolated_db.get_hook(hid)
    assert row["error_threshold"] == 3


def test_error_threshold_clamped(isolated_db):
    """Out-of-range values get clamped (1..50)."""
    hid_lo = isolated_db.create_hook(
        event="consecutive_failures", command="echo lo", error_threshold=0,
    )
    hid_hi = isolated_db.create_hook(
        event="consecutive_failures", command="echo hi", error_threshold=999,
    )
    assert isolated_db.get_hook(hid_lo)["error_threshold"] == 1
    assert isolated_db.get_hook(hid_hi)["error_threshold"] == 50


def test_max_fires_per_conv_round_trip(isolated_db):
    hid = isolated_db.create_hook(
        event="tool_error", command="echo capped", max_fires_per_conv=5,
    )
    row = isolated_db.get_hook(hid)
    assert row["max_fires_per_conv"] == 5
    # Patch through update_hook
    updated = isolated_db.update_hook(hid, max_fires_per_conv=10)
    assert updated["max_fires_per_conv"] == 10
    # NULL clears the cap
    updated2 = isolated_db.update_hook(hid, max_fires_per_conv=None)
    assert updated2["max_fires_per_conv"] is None


# --- per-conversation fire counter ---------------------------------------


def test_hook_fire_counter_starts_at_zero(isolated_db):
    hid = isolated_db.create_hook(event="tool_error", command="echo x")
    assert isolated_db.get_hook_fire_count(hid, "conv-1") == 0


def test_hook_fire_counter_increments(isolated_db):
    hid = isolated_db.create_hook(event="tool_error", command="echo x")
    assert isolated_db.incr_hook_fire(hid, "conv-1") == 1
    assert isolated_db.incr_hook_fire(hid, "conv-1") == 2
    assert isolated_db.get_hook_fire_count(hid, "conv-1") == 2
    # Independent counters per conversation.
    assert isolated_db.incr_hook_fire(hid, "conv-2") == 1
    assert isolated_db.get_hook_fire_count(hid, "conv-1") == 2


def test_hook_fire_counter_reset(isolated_db):
    hid = isolated_db.create_hook(event="tool_error", command="echo x")
    isolated_db.incr_hook_fire(hid, "conv-1")
    isolated_db.incr_hook_fire(hid, "conv-2")
    isolated_db.reset_hook_fires(hook_id=hid)
    assert isolated_db.get_hook_fire_count(hid, "conv-1") == 0
    assert isolated_db.get_hook_fire_count(hid, "conv-2") == 0


# --- timeout caps --------------------------------------------------------


def test_default_timeout_cap_is_120():
    """Existing fire-and-forget events keep the original 120 s ceiling."""
    assert db.hook_timeout_cap("post_tool") == 120
    assert db.hook_timeout_cap("user_prompt_submit") == 120
    assert db.hook_timeout_cap("turn_done") == 120
    assert db.hook_timeout_cap("pre_tool") == 120


def test_diagnostic_events_get_longer_timeout_cap():
    """tool_error and consecutive_failures often invoke long-running
    diagnostic commands (a linter, the Claude CLI, a test suite). They
    get a 900 s cap so a runaway diagnosis still terminates eventually."""
    assert db.hook_timeout_cap("tool_error") == 900
    assert db.hook_timeout_cap("consecutive_failures") == 900


def test_timeout_clamped_to_event_specific_cap(isolated_db):
    """Pass a huge timeout and watch it clamp to the event's ceiling."""
    hid_short = isolated_db.create_hook(
        event="post_tool", command="echo a", timeout_seconds=99999,
    )
    hid_long = isolated_db.create_hook(
        event="consecutive_failures", command="echo b", timeout_seconds=99999,
    )
    assert isolated_db.get_hook(hid_short)["timeout_seconds"] == 120
    assert isolated_db.get_hook(hid_long)["timeout_seconds"] == 900


# --- agent-level consecutive-failure tracker ------------------------------


def test_consecutive_failure_tracker_resets_on_success():
    """The in-memory tracker zeros out on the next ok=True for the
    same (conv, tool). Without this, a transient model goof + recovery
    would still trip a `consecutive_failures` hook on the next bad
    call hours later."""
    from backend import agent
    # Three failures in a row.
    assert agent._bump_consec_failures("conv-A", "bash", ok=False) == 1
    assert agent._bump_consec_failures("conv-A", "bash", ok=False) == 2
    assert agent._bump_consec_failures("conv-A", "bash", ok=False) == 3
    # A success — counter resets.
    assert agent._bump_consec_failures("conv-A", "bash", ok=True) == 0
    # Failure starts the streak fresh.
    assert agent._bump_consec_failures("conv-A", "bash", ok=False) == 1


def test_consecutive_failure_tracker_is_per_tool():
    """A `bash` failure shouldn't bump `read_file`'s counter. They're
    independent streaks within the same conversation."""
    from backend import agent
    assert agent._bump_consec_failures("conv-B", "bash", ok=False) == 1
    assert agent._bump_consec_failures("conv-B", "bash", ok=False) == 2
    # Different tool — separate counter.
    assert agent._bump_consec_failures("conv-B", "read_file", ok=False) == 1
    # bash count unchanged
    assert agent._bump_consec_failures("conv-B", "bash", ok=False) == 3
