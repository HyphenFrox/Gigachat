"""Regression: lazy tool loading lets the model start with a small
toolbelt and grow it on demand instead of paying the full ~18 K-token
schema cost every turn.

Pinned invariants:

  * The two meta-tools (`tool_search`, `tool_load`) are always present
    even when the conversation has never called `tool_load` — without
    them the model has no way to bootstrap the rest of the toolbelt.
  * `tool_search` runs without a `conv_id` and is honest about the
    "loaded" flag — `False` for everything when there's no conversation.
  * `tool_load` refuses without a `conv_id` (no place to persist), and
    is idempotent (re-loading is a no-op + reports as already-loaded).
  * Unknown names from `tool_load` come back as errors instead of
    silently corrupting the loaded set.
  * The system prompt embeds the manifest section so the model can
    discover tool names + 1-liners without paying the schema cost.
  * Subagent palettes intentionally omit the meta-tools — they're
    short-lived and don't have a persisted loaded set.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from backend import db, prompts, tools

pytestmark = pytest.mark.smoke


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def conv(isolated_db):
    """Brand-new conversation row in an isolated DB."""
    return isolated_db.create_conversation(
        title="lazy-load",
        model="gemma4:e4b",
        cwd="C:/tmp",
    )


# --- DB layer -------------------------------------------------------------


def test_loaded_tools_default_empty(isolated_db, conv):
    assert isolated_db.get_loaded_tools(conv["id"]) == []


def test_add_loaded_tools_is_idempotent(isolated_db, conv):
    cid = conv["id"]
    first = isolated_db.add_loaded_tools(cid, ["read_file", "write_file"])
    assert sorted(first) == ["read_file", "write_file"]
    # Adding overlapping set: existing entries kept, new ones appended,
    # no duplicates.
    second = isolated_db.add_loaded_tools(cid, ["write_file", "bash"])
    assert sorted(second) == ["bash", "read_file", "write_file"]
    # Re-reading from the DB returns the same merged set.
    assert sorted(isolated_db.get_loaded_tools(cid)) == ["bash", "read_file", "write_file"]


def test_add_loaded_tools_nonexistent_conv(isolated_db):
    """Calling against a missing conversation returns [] without raising."""
    assert isolated_db.add_loaded_tools("does-not-exist", ["read_file"]) == []


# --- tool_search ----------------------------------------------------------


def test_tool_search_finds_by_summary_keyword():
    res = _run(tools.tool_search("read pdf", limit=5))
    assert res["ok"] is True
    out = res["output"]
    assert "read_doc" in out, f"expected read_doc in:\n{out}"


def test_tool_search_empty_query_errors():
    res = _run(tools.tool_search("", limit=5))
    assert res["ok"] is False
    assert "query" in (res.get("error") or "").lower()


def test_tool_search_excludes_meta_tools():
    """The manifest must not advertise tool_search / tool_load — the
    model shouldn't lazy-load the things that ARE the lazy-load
    bootstrap. We check the bullet-list region specifically: the
    instructional footer naturally mentions `tool_load(...)` as a
    usage hint, which is fine."""
    res = _run(tools.tool_search("tool", limit=30))
    out = res.get("output") or ""
    bullet_lines = [ln for ln in out.splitlines() if ln.lstrip().startswith("•")]
    bullet_blob = "\n".join(bullet_lines)
    assert "tool_search" not in bullet_blob
    assert "tool_load" not in bullet_blob


def test_tool_search_marks_loaded_flag(isolated_db, conv):
    cid = conv["id"]
    isolated_db.add_loaded_tools(cid, ["read_file"])
    res = _run(tools.tool_search("read", limit=10, conv_id=cid))
    out = res.get("output") or ""
    # `read_file` was loaded in the fixture, so the search result
    # should mark it as such.
    lines = [ln for ln in out.splitlines() if "read_file" in ln]
    assert lines, f"expected read_file line in:\n{out}"
    assert any("(loaded)" in ln for ln in lines)


# --- tool_load ------------------------------------------------------------


def test_tool_load_refuses_without_conv_id():
    res = _run(tools.tool_load(["read_file"]))
    assert res["ok"] is False
    assert "conversation" in (res.get("error") or "").lower()


def test_tool_load_basic_round_trip(isolated_db, conv):
    cid = conv["id"]
    res = _run(tools.tool_load(["read_file", "write_file"], conv_id=cid))
    assert res["ok"] is True
    assert sorted(res["loaded"]) == ["read_file", "write_file"]
    assert isolated_db.get_loaded_tools(cid) == ["read_file", "write_file"]


def test_tool_load_idempotent(isolated_db, conv):
    cid = conv["id"]
    _run(tools.tool_load(["read_file"], conv_id=cid))
    res = _run(tools.tool_load(["read_file", "bash"], conv_id=cid))
    out = res["output"]
    # Already-loaded tools are reported as such; the new one is loaded.
    assert "Already loaded" in out
    assert "Loaded 1 tool(s): bash" in out
    assert sorted(res["loaded"]) == ["bash", "read_file"]


def test_tool_load_unknown_name_reported(isolated_db, conv):
    cid = conv["id"]
    res = _run(tools.tool_load(["read_file", "nope_does_not_exist"], conv_id=cid))
    # Mixed result — known name loads, unknown is reported.
    assert "nope_does_not_exist" in (res.get("output") or "")
    assert res["unknown"] == ["nope_does_not_exist"]
    # The known name still made it through.
    assert "read_file" in res["loaded"]


def test_tool_load_empty_names_errors(isolated_db, conv):
    cid = conv["id"]
    res = _run(tools.tool_load([], conv_id=cid))
    assert res["ok"] is False
    assert "empty" in (res.get("error") or "").lower()


def test_tool_load_accepts_string_for_forgiveness(isolated_db, conv):
    """Smaller models sometimes pass a bare string instead of a list.
    Don't bounce the call — coerce and proceed."""
    cid = conv["id"]
    res = _run(tools.tool_load("read_file", conv_id=cid))
    assert res["ok"] is True
    assert "read_file" in res["loaded"]


# --- system prompt manifest ----------------------------------------------


def test_system_prompt_includes_manifest():
    sp = prompts.build_system_prompt("C:/tmp")
    assert "Available tools (lazy-loaded)" in sp
    # Real tools should appear in the manifest by name.
    assert "read_file" in sp
    assert "bash" in sp
    # Meta-tools must NOT appear (they're always loaded and listing them
    # would confuse the model into re-loading them).
    # Allow the bootstrap instructions to mention them once each in
    # narrative; we just want them missing from the bullet list. So
    # check the list section shape:
    assert "  • tool_search" not in sp
    assert "  • tool_load" not in sp


# --- meta-tools registered ------------------------------------------------


def test_meta_tools_registered():
    assert "tool_search" in tools.TOOL_REGISTRY
    assert "tool_load" in tools.TOOL_REGISTRY
    assert tools.TOOL_REGISTRY["tool_search"] is tools.tool_search
    assert tools.TOOL_REGISTRY["tool_load"] is tools.tool_load


def test_meta_tools_categorised_as_read():
    """Meta-tools must NOT be write-class — they have no external side
    effects (just bookkeeping in the conversations table) and should
    run silently in `approve_edits` mode."""
    assert tools.classify_tool("tool_search") == "read"
    assert tools.classify_tool("tool_load") == "read"


def test_always_loaded_constant_matches_meta_tools():
    """Single source of truth — the constant the agent loop uses to
    decide what's always in the toolbelt must match the meta-tools we
    actually registered."""
    assert set(tools.ALWAYS_LOADED_TOOLS) == {"tool_search", "tool_load"}


# --- meta-tool schemas exposed -------------------------------------------


def test_meta_tool_schemas_in_TOOL_SCHEMAS():
    names = {
        ((s.get("function") or {}).get("name"))
        for s in prompts.TOOL_SCHEMAS
    }
    assert "tool_search" in names
    assert "tool_load" in names
