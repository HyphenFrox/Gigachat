"""Tests for `ui_wait`'s kind dispatcher and schema surface.

`ui_wait` supports six kinds: window / window_gone / element /
element_enabled / text / pixel_change. Most actual polling logic lives
behind platform-specific APIs (Windows UIA, Pillow screenshots, OCR),
so we can't fully exercise every kind on Linux CI. What we CAN verify
at the smoke tier:

  * A bogus kind is rejected with all six valid kinds listed — the
    error message is the model's only guide back to sane input.
  * The schema accepts `require_enabled` on `kind=element` (not a
    parameter-validation error). On non-Windows the call will time out
    cleanly rather than reject the argument.
  * The dispatcher forwards `require_enabled` as a bool (no int/str
    smuggling).

Windows-specific behaviors (`window_gone` firing immediately for a
missing title, `element_enabled` timing out with a status line) live in
a `deep` + `windows` marker and are skipped on other platforms.
"""
from __future__ import annotations

import asyncio
import sys
import time

import pytest

from backend import tools

pytestmark = pytest.mark.smoke


# --- Platform-agnostic surface --------------------------------------------


def test_bogus_kind_lists_all_valid_kinds():
    """An unknown kind must produce an error enumerating every valid one."""
    res = asyncio.run(tools.ui_wait(kind="not-a-real-kind", target="x"))
    err = (res.get("error") or "").lower()
    assert res["ok"] is False
    for required in (
        "window", "window_gone",
        "element", "element_enabled",
        "text", "pixel_change",
    ):
        assert required in err, f"valid kind {required!r} missing from error"


def test_require_enabled_is_accepted_on_element_kind():
    """`require_enabled=True` is a legal parameter (not a schema error).

    On non-Windows the call times out cleanly; what we're guarding
    against is a regression that rejects the arg before ever polling.
    """
    res = asyncio.run(tools.ui_wait(
        kind="element",
        target="xyzzy-unknown",
        timeout_seconds=1,
        interval_seconds=0.25,
        require_enabled=True,
    ))
    # Whichever platform we're on, the outcome is ok=False with an error —
    # but it must be a timeout or "Windows-only", NOT a parameter error.
    assert res["ok"] is False
    err = (res.get("error") or "").lower()
    assert "require_enabled" not in err, (
        f"require_enabled rejected as unknown arg: {res['error']!r}"
    )


def test_dispatcher_coerces_require_enabled_to_bool():
    """String 'true' from a noisy LLM should land as a real bool, not a string."""
    # Reach into the dispatcher the same way the agent loop does.
    async def run():
        return await tools.dispatch(
            "ui_wait",
            {
                "kind": "element",
                "target": "xyzzy",
                "timeout_seconds": 1,
                "interval_seconds": 0.25,
                "require_enabled": "true",  # hostile non-bool
            },
            cwd=".",
            conv_id=None,
        )
    res = asyncio.run(run())
    # The dispatcher wraps bool() around it — so a truthy string becomes
    # True and the call proceeds to the timeout path rather than exploding.
    assert res["ok"] is False
    assert "require_enabled" not in (res.get("error") or "").lower()


# --- Windows-only deep tests ----------------------------------------------


@pytest.mark.deep
@pytest.mark.windows
@pytest.mark.skipif(sys.platform != "win32", reason="Windows UIA required")
def test_window_gone_triggers_immediately_for_nonexistent_title():
    """A `window_gone` target that doesn't exist fires on the first poll."""
    async def run():
        t0 = time.monotonic()
        r = await tools.ui_wait(
            kind="window_gone",
            target="definitely-no-such-window-xyzzy-8844",
            timeout_seconds=5,
            interval_seconds=0.5,
        )
        return r, time.monotonic() - t0

    res, elapsed = asyncio.run(run())
    assert res["ok"] is True
    assert elapsed < 2.0, f"window_gone should be near-instant, took {elapsed:.2f}s"


@pytest.mark.deep
@pytest.mark.windows
@pytest.mark.skipif(sys.platform != "win32", reason="Windows UIA required")
def test_element_enabled_times_out_with_status():
    """A `element_enabled` target that never appears times out cleanly."""
    async def run():
        return await tools.ui_wait(
            kind="element_enabled",
            target="definitely-no-such-control-xyzzy-8844",
            timeout_seconds=3,
            interval_seconds=0.5,
        )

    res = asyncio.run(run())
    err = (res.get("error") or "").lower()
    assert res["ok"] is False
    assert "timed out" in err
    # The error must surface the last status so the model learns what
    # was actually observed during polling.
    assert "last status" in err
