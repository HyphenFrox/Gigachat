"""Regression: project memory — the third memory scope.

Pinned invariants:

  * `project_memories` table accepts insert / list / delete-by-pattern
    keyed by `cwd` (the conversation's working directory path).
  * Two conversations pointed at the same `cwd` share the same memory
    set automatically — no user-side configuration required.
  * Cwd normalization: trailing slashes stripped; case-folded on
    Windows so "C:\\repo" and "c:\\repo" share state.
  * The `remember(scope="project")` / `forget(scope="project")` tool
    paths refuse cleanly when the conversation has no `cwd` set
    (defensive — every conversation should have one).
  * `load_project_memory_for_prompt` renders an empty string for
    blank-cwd / empty-table cases (so `build_system_prompt` can
    splice it unconditionally).
  * Schemas advertise all three scope values so adapter-mode models
    know `project` is a real choice.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from backend import prompts, tools

pytestmark = pytest.mark.smoke


def _run(coro):
    return asyncio.run(coro)


# --- DB layer -------------------------------------------------------------


def test_add_and_list_project_memory(isolated_db):
    cwd = "C:/repos/acme"
    isolated_db.add_project_memory(cwd, "uses pytest, never unittest")
    isolated_db.add_project_memory(cwd, "lint config is .eslintrc.cjs")
    rows = isolated_db.list_project_memories(cwd)
    contents = [r["content"] for r in rows]
    assert contents == ["uses pytest, never unittest", "lint config is .eslintrc.cjs"]


def test_list_project_memory_isolation_by_cwd(isolated_db):
    """A cwd's memories don't bleed into another cwd's set."""
    isolated_db.add_project_memory("C:/repos/acme", "ACME-only fact")
    isolated_db.add_project_memory("C:/repos/widgets", "Widgets-only fact")
    assert [r["content"] for r in isolated_db.list_project_memories("C:/repos/acme")] == ["ACME-only fact"]
    assert [r["content"] for r in isolated_db.list_project_memories("C:/repos/widgets")] == ["Widgets-only fact"]


def test_two_conversations_same_cwd_share_memory(isolated_db):
    """Core promise of the feature: same cwd → same memory set,
    even across separately-created conversations."""
    cwd = "C:/repos/shared"
    conv_a = isolated_db.create_conversation(title="A", model="m", cwd=cwd)
    conv_b = isolated_db.create_conversation(title="B", model="m", cwd=cwd)
    # Conv A writes a memory.
    res = _run(tools.remember(conv_a["id"], "deploy via fly.io", scope="project"))
    assert res["ok"] is True
    # Conv B sees it without doing anything.
    rows = isolated_db.list_project_memories(conv_b["cwd"])
    assert any("fly.io" in r["content"] for r in rows)


def test_cwd_normalization_strips_trailing_slash(isolated_db):
    """`C:/repos/acme` and `C:/repos/acme/` should hit the same row."""
    isolated_db.add_project_memory("C:/repos/acme", "first fact")
    rows = isolated_db.list_project_memories("C:/repos/acme/")
    assert any("first fact" in r["content"] for r in rows)


@pytest.mark.skipif(os.name != "nt", reason="case-folding only on Windows")
def test_cwd_normalization_case_insensitive_on_windows(isolated_db):
    """Windows filesystems are case-insensitive; "C:\\Users" and
    "c:\\users" should map to the same memory set."""
    isolated_db.add_project_memory("C:/Repos/Acme", "fact A")
    rows = isolated_db.list_project_memories("c:/repos/acme")
    assert any("fact A" in r["content"] for r in rows)


def test_add_project_memory_rejects_blank_cwd(isolated_db):
    with pytest.raises(ValueError):
        isolated_db.add_project_memory("", "some fact")
    with pytest.raises(ValueError):
        isolated_db.add_project_memory("   ", "some fact")


def test_delete_project_memory_substring(isolated_db):
    cwd = "C:/repos/acme"
    isolated_db.add_project_memory(cwd, "we use pytest, not unittest")
    isolated_db.add_project_memory(cwd, "linter is biome, not eslint")
    isolated_db.add_project_memory(cwd, "deploy via fly.io")
    n = isolated_db.delete_project_memories_matching(cwd, "lint")
    assert n == 1
    remaining = [r["content"] for r in isolated_db.list_project_memories(cwd)]
    assert "biome" not in " ".join(remaining)


def test_delete_project_memory_does_not_cross_cwd(isolated_db):
    """Pattern delete must stay scoped to the supplied cwd."""
    isolated_db.add_project_memory("C:/repos/a", "shared word here")
    isolated_db.add_project_memory("C:/repos/b", "shared word here too")
    n = isolated_db.delete_project_memories_matching("C:/repos/a", "shared")
    assert n == 1
    assert len(isolated_db.list_project_memories("C:/repos/b")) == 1


def test_delete_project_memory_refuses_blank_pattern(isolated_db):
    isolated_db.add_project_memory("C:/x", "x")
    with pytest.raises(ValueError):
        isolated_db.delete_project_memories_matching("C:/x", "")


# --- remember/forget tool paths ------------------------------------------


def test_remember_project_scope_uses_conv_cwd(isolated_db):
    """Round-trip via the tool: create conv, call remember(scope='project'),
    verify the row landed under the conv's cwd."""
    cwd = "C:/repos/acme"
    conv = isolated_db.create_conversation(title="t", model="m", cwd=cwd)
    res = _run(tools.remember(conv["id"], "uses pnpm not npm", scope="project"))
    assert res["ok"] is True
    rows = isolated_db.list_project_memories(cwd)
    assert any("pnpm" in r["content"] for r in rows)


def test_remember_project_refuses_without_conv_id(isolated_db):
    res = _run(tools.remember(None, "x", scope="project"))
    assert res["ok"] is False


def test_forget_project_scope_round_trip(isolated_db):
    cwd = "C:/repos/acme"
    conv = isolated_db.create_conversation(title="t", model="m", cwd=cwd)
    isolated_db.add_project_memory(cwd, "fact one")
    isolated_db.add_project_memory(cwd, "fact two")
    res = _run(tools.forget(conv["id"], "one", scope="project"))
    assert res["ok"] is True
    remaining = isolated_db.list_project_memories(cwd)
    assert all("one" not in r["content"] for r in remaining)


def test_invalid_scope_rejected_by_remember():
    res = _run(tools.remember("any", "x", scope="bogus"))
    assert res["ok"] is False
    err = (res.get("error") or "").lower()
    assert "conversation" in err and "project" in err and "global" in err


# --- prompt loader -------------------------------------------------------


def test_load_project_memory_empty_for_blank_cwd(isolated_db):
    assert tools.load_project_memory_for_prompt(None) == ""
    assert tools.load_project_memory_for_prompt("") == ""
    assert tools.load_project_memory_for_prompt("   ") == ""


def test_load_project_memory_renders_section(isolated_db):
    cwd = "C:/repos/acme"
    isolated_db.add_project_memory(cwd, "uses pnpm")
    isolated_db.add_project_memory(cwd, "biome for lint")
    text = tools.load_project_memory_for_prompt(cwd)
    assert "Project memory" in text
    assert "pnpm" in text
    assert "biome" in text


def test_build_system_prompt_includes_project_section(isolated_db):
    """End-to-end: a conversation gets the project memory section
    spliced into its system prompt, keyed by cwd."""
    cwd = "C:/repos/acme"
    isolated_db.add_project_memory(cwd, "deploy via fly.io")
    conv = isolated_db.create_conversation(title="t", model="m", cwd=cwd)
    sp = prompts.build_system_prompt(cwd, conv_id=conv["id"])
    assert "Project memory" in sp
    assert "fly.io" in sp


def test_build_system_prompt_omits_project_section_when_empty(isolated_db):
    """No memories for this cwd → section is omitted entirely."""
    conv = isolated_db.create_conversation(title="t", model="m", cwd="C:/empty")
    sp = prompts.build_system_prompt("C:/empty", conv_id=conv["id"])
    assert "Project memory" not in sp


# --- schema advertises all three scopes ----------------------------------


def test_remember_schema_lists_project_scope():
    sch = next(
        s for s in prompts.TOOL_SCHEMAS
        if (s.get("function") or {}).get("name") == "remember"
    )
    enum = sch["function"]["parameters"]["properties"]["scope"]["enum"]
    assert set(enum) == {"conversation", "project", "global"}


def test_forget_schema_lists_project_scope():
    sch = next(
        s for s in prompts.TOOL_SCHEMAS
        if (s.get("function") or {}).get("name") == "forget"
    )
    enum = sch["function"]["parameters"]["properties"]["scope"]["enum"]
    assert set(enum) == {"conversation", "project", "global"}
