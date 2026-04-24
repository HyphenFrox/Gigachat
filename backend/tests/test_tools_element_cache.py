"""Regression tests for the LRU element cache that backs `click_element_id`.

The cache is a `threading.Lock`-guarded `OrderedDict` keyed by the `elN`
strings `inspect_window` / `screenshot(with_elements=True)` mint. Two
invariants matter:

  * Reads promote the entry to MRU — so an active control doesn't age out
    under a noisy backdrop of less-used buttons. `OrderedDict.move_to_end`
    gives us this for free; the test catches anyone who accidentally
    reverts to a plain dict.
  * `_element_id_status` distinguishes three miss reasons: bad format
    (typo), not minted (id past high-water mark), and evicted (real id
    aged out). `click_element_id`'s error leans on this to tell the model
    whether to re-type or re-inspect.

These tests touch module globals directly under the shared lock, matching
how `inspect_window` itself operates. Each test resets cache state first
so ordering assertions are deterministic.
"""
from __future__ import annotations

import asyncio

import pytest

from backend import tools

pytestmark = pytest.mark.smoke


def _reset_cache():
    """Wipe the module-global cache between tests (use the real lock)."""
    with tools._ELEMENT_LOCK:
        tools._ELEMENT_CACHE.clear()
        tools._ELEMENT_NEXT_ID = 0
        tools._ELEMENT_HIGH_WATER = 0


# --- LRU semantics ---------------------------------------------------------


def test_read_promotes_entry_to_mru():
    """After minting e1, e2, e3 and reading e1, the order must be [e2, e3, e1]."""
    _reset_cache()
    e1 = tools._element_cache_put(10, 20, "first")
    e2 = tools._element_cache_put(30, 40, "second")
    e3 = tools._element_cache_put(50, 60, "third")

    hit = tools._element_cache_get(e1)
    assert hit is not None
    assert hit["cx"] == 10 and hit["cy"] == 20
    assert hit["label"] == "first"

    with tools._ELEMENT_LOCK:
        order = list(tools._ELEMENT_CACHE.keys())
    assert order == [e2, e3, e1], (
        f"expected LRU promotion to move {e1} to the tail, got {order}"
    )


def test_get_returns_a_copy_not_a_live_reference():
    """Mutating the returned dict must not leak into the cache."""
    _reset_cache()
    eid = tools._element_cache_put(1, 2, "label")
    first = tools._element_cache_get(eid)
    first["cx"] = 99999
    second = tools._element_cache_get(eid)
    assert second["cx"] == 1, "cache returned a live reference — mutations leaked"


# --- status codes ----------------------------------------------------------


def test_status_bad_format_for_garbage_id():
    """Anything that doesn't match the `elN` shape is 'bad format'."""
    _reset_cache()
    assert tools._element_id_status("not-an-id") == "bad format"
    assert tools._element_id_status("") == "bad format"
    assert tools._element_id_status("el") == "bad format"


def test_status_not_minted_for_future_id():
    """An id numerically past the high-water mark was never issued."""
    _reset_cache()
    tools._element_cache_put(0, 0, "x")  # mints el1, high-water = 1
    assert tools._element_id_status("el999") == "not minted"


def test_status_evicted_for_minted_then_dropped_id():
    """An id that was minted but no longer in the cache is 'evicted'."""
    _reset_cache()
    eid = tools._element_cache_put(0, 0, "ghost")
    with tools._ELEMENT_LOCK:
        del tools._ELEMENT_CACHE[eid]
    assert tools._element_id_status(eid) == "evicted"


# --- click_element_id error surface ---------------------------------------


def test_click_element_id_error_explains_not_minted():
    """The user-facing error must include the status so the model can
    pick the right recovery (retype vs. re-inspect)."""
    _reset_cache()

    async def run():
        return await tools.click_element_id("el9999")

    out = asyncio.run(run())
    assert out["ok"] is False
    assert "not minted" in (out.get("error") or "").lower()
