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


def test_tool_search_no_results_points_at_primitives():
    """Real failure mode from a hermes3:8b conversation: model called
    tool_search('get current btc price'), got "No tools match X. Try
    fewer or broader terms", and gave up — pivoted to a tutorial
    response. The empty-result message wasn't actionable enough.

    New behaviour: point the model at the universal toolkit
    (write_file/read_file/edit_file/bash/python_exec) so it has a
    concrete next step instead of an open-ended 'try broader terms'."""
    res = _run(tools.tool_search("get current bitcoin price tool", limit=5))
    assert res["ok"] is True
    out = (res.get("output") or "")
    # Negative result still leads.
    assert "No tools match" in out
    # And the response steers the model at concrete primitives rather
    # than just telling it to try broader terms.
    for primitive in ("write_file", "read_file", "edit_file", "bash"):
        assert primitive in out, (
            f"empty-search response should mention {primitive} as a "
            f"primitive; got:\n{out}"
        )
    # Steers AWAY from the "write a tutorial" failure mode.
    assert "tutorial" in out.lower() or "execute the task" in out.lower()
    # Concrete tool_load call shape so the model can copy it.
    assert "tool_load" in out


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
    # Already-loaded tools are reported as such; the new request brings
    # in `bash` plus its bundle siblings (bash_bg, bash_output,
    # kill_shell) since they share state via shell_id.
    assert "Already loaded" in out
    assert "bash" in out
    assert "read_file" in res["loaded"]
    assert "bash" in res["loaded"]
    # Bundle siblings come along for the ride.
    assert "bash_bg" in res["loaded"]
    assert "bash_output" in res["loaded"]
    assert "kill_shell" in res["loaded"]


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


# --- required-field hints in the manifest --------------------------------


def test_manifest_shows_required_fields():
    """Without the required-field hint, adapter-mode models call tools
    by name with only `reason` filled (because reason is on every
    schema) and forget tool-specific required fields like `command` or
    `path`. The manifest section must surface those names so the model
    has enough info to make a correct first call without loading."""
    sp = prompts.build_system_prompt("C:/tmp")
    # Bash needs `command`. Without this hint, gemma4:e4b kept emitting
    # `args={"reason": "..."}` and getting "empty command" errors.
    assert "bash" in sp
    bash_lines = [ln for ln in sp.splitlines() if " bash " in ln or "• bash —" in ln]
    assert any("required: command" in ln for ln in bash_lines), (
        f"expected `required: command` on the bash manifest line, got:\n"
        + "\n".join(bash_lines)
    )
    # read_file needs `path`.
    read_lines = [ln for ln in sp.splitlines() if "• read_file —" in ln]
    assert any("required: path" in ln for ln in read_lines), (
        f"expected `required: path` on the read_file manifest line, got:\n"
        + "\n".join(read_lines)
    )


def test_manifest_excludes_reason_from_required():
    """`reason` is a UX field added to every schema by `_with_reason`;
    listing it on every manifest line would be noise. It should be
    stripped before rendering."""
    sp = prompts.build_system_prompt("C:/tmp")
    bullet_lines = [ln for ln in sp.splitlines() if ln.lstrip().startswith("•")]
    bullet_blob = "\n".join(bullet_lines)
    # No bullet line should advertise `reason` as required.
    assert "required: reason" not in bullet_blob
    assert "required: reason," not in bullet_blob


# --- auto-load on dispatch ------------------------------------------------


def test_dispatch_auto_loads_tool_on_first_call(isolated_db, conv):
    """Calling a tool that hasn't been explicitly loaded should add it
    to the conversation's loaded set, so the NEXT turn carries the full
    schema. Eliminates the "first call has wrong args because the
    schema wasn't visible" failure mode in adapter mode."""
    cid = conv["id"]
    assert isolated_db.get_loaded_tools(cid) == []
    # Dispatch any read-class tool. We use list_dir because it's
    # synchronous, no external dependencies, and exercises the
    # registry+conv_id path.
    _run(tools.dispatch(
        "list_dir",
        {"path": "."},
        cwd=str(__import__("pathlib").Path(__file__).parent),
        conv_id=cid,
    ))
    # `list_dir` is now loaded for this conversation.
    assert "list_dir" in isolated_db.get_loaded_tools(cid)


def test_dispatch_auto_load_skips_meta_tools(isolated_db, conv):
    """The two meta-tools are always-loaded; auto-loading them would
    just clutter the persisted set and round-trip the DB pointlessly."""
    cid = conv["id"]
    _run(tools.dispatch(
        "tool_search",
        {"query": "read"},
        cwd=".",
        conv_id=cid,
    ))
    # tool_search ran, but it's NOT in the persisted loaded set.
    assert "tool_search" not in isolated_db.get_loaded_tools(cid)


def test_dispatch_auto_load_skips_unknown_names(isolated_db, conv):
    """Unknown tool names should not pollute the loaded set — the
    dispatcher returns a "did you mean" error and that's it."""
    cid = conv["id"]
    res = _run(tools.dispatch(
        "definitely_not_a_real_tool",
        {},
        cwd=".",
        conv_id=cid,
    ))
    assert res["ok"] is False
    assert isolated_db.get_loaded_tools(cid) == []


# --- bash missing-command error message ----------------------------------


# --- tool bundles --------------------------------------------------------


def test_tool_load_pulls_shell_bundle(isolated_db, conv):
    """Loading `bash` should also bring in `bash_bg`, `bash_output`,
    `kill_shell` — they share state (the `shell_id`) and a model that
    holds only `bash` can't recover when a long-running command needs
    the background variant. This was the failure mode in conversation
    0f2136d8: model loaded `bash`, hit a `npm run dev` step, had no
    tool that fit, and punted to the user."""
    cid = conv["id"]
    res = _run(tools.tool_load(["bash"], conv_id=cid))
    assert res["ok"] is True
    loaded = set(isolated_db.get_loaded_tools(cid))
    for member in ("bash", "bash_bg", "bash_output", "kill_shell"):
        assert member in loaded, f"{member} should be in loaded set after tool_load(['bash'])"


def test_bundle_expansion_is_symmetric(isolated_db, conv):
    """Loading any member of a bundle pulls the whole bundle. Calling
    `tool_load(['bash_output'])` first should still bring in `bash`
    so the model can actually USE the shell_id it'd be polling."""
    cid = conv["id"]
    res = _run(tools.tool_load(["bash_output"], conv_id=cid))
    assert res["ok"] is True
    loaded = set(isolated_db.get_loaded_tools(cid))
    for member in ("bash", "bash_bg", "bash_output", "kill_shell"):
        assert member in loaded


def test_dispatch_auto_load_pulls_bundle(isolated_db, conv):
    """Same expansion in dispatch: when the model calls `bash`
    directly (in adapter mode the parser accepts any name regardless
    of loaded set), auto-load pulls the whole shell toolkit so a
    follow-up `bash_bg` call has its schema visible the next turn."""
    cid = conv["id"]
    _run(tools.dispatch(
        "bash",
        {"command": "echo hi"},
        cwd=str(__import__("pathlib").Path(__file__).parent),
        conv_id=cid,
    ))
    loaded = set(isolated_db.get_loaded_tools(cid))
    for member in ("bash", "bash_bg", "bash_output", "kill_shell"):
        assert member in loaded


def test_bundle_does_not_leak_unrelated_tools(isolated_db, conv):
    """Loading `read_file` (which has no bundle) should NOT pull in
    write_file or edit_file — bundling is reserved for state-sharing
    toolkits, not "merely related" tools. Bloating the loaded set
    with everything-related-to-X defeats lazy load."""
    cid = conv["id"]
    res = _run(tools.tool_load(["read_file"], conv_id=cid))
    assert res["ok"] is True
    loaded = set(isolated_db.get_loaded_tools(cid))
    assert "read_file" in loaded
    assert "write_file" not in loaded
    assert "edit_file" not in loaded


def test_bundle_expansion_preserves_order():
    """The originally-requested name should lead the expanded list
    so reporting in tool_load (which iterates in order) leads with
    what the model asked for, with siblings appended after."""
    expanded = tools._expand_with_bundles(["bash_output"])
    assert expanded[0] == "bash_output"
    assert set(expanded) == {"bash", "bash_bg", "bash_output", "kill_shell"}


def test_bundle_expansion_is_idempotent():
    """Passing an already-fully-expanded set in returns the same
    members — no duplicates, no oscillation."""
    full = ["bash", "bash_bg", "bash_output", "kill_shell"]
    expanded = tools._expand_with_bundles(full)
    assert set(expanded) == set(full)
    assert len(expanded) == 4


def test_bash_missing_command_returns_actionable_error():
    """The previous error was just 'empty command' — too terse for
    small adapter-mode models. They'd loop on the same mistake. The new
    error explicitly names the missing field, gives an example, and
    explains the difference vs `reason`."""
    res = _run(tools.dispatch(
        "bash",
        {"reason": "trying to cd somewhere"},  # forgot `command`
        cwd=".",
        conv_id=None,
    ))
    assert res["ok"] is False
    err = res.get("error") or ""
    assert "command" in err.lower()
    # Worth pinning the example shape — that's the bit the model
    # actually copies from on retry.
    assert "{" in err and "command" in err
