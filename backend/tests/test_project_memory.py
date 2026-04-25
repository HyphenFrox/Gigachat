"""Regression: project memory — the third memory scope.

Pinned invariants:

  * `project_memories` table accepts insert / list / delete-by-pattern.
  * Memories are scoped to the conversation's `project` label —
    moving a chat to a different project shifts which set it sees.
  * The `remember(scope="project")` / `forget(scope="project")` tool
    paths refuse cleanly when the conversation has no project.
  * `load_project_memory_for_prompt` renders an empty string for
    unlabelled / empty-table cases (so `build_system_prompt` can
    splice it unconditionally).
  * Schemas advertise all three scope values so adapter-mode models
    know `project` is a real choice.
"""
from __future__ import annotations

import asyncio

import pytest

from backend import prompts, tools

pytestmark = pytest.mark.smoke


def _run(coro):
    return asyncio.run(coro)


# --- DB layer -------------------------------------------------------------


def test_add_and_list_project_memory(isolated_db):
    isolated_db.add_project_memory("acme", "uses pytest, never unittest")
    isolated_db.add_project_memory("acme", "lint config is .eslintrc.cjs")
    rows = isolated_db.list_project_memories("acme")
    contents = [r["content"] for r in rows]
    assert contents == ["uses pytest, never unittest", "lint config is .eslintrc.cjs"]


def test_list_project_memory_isolation(isolated_db):
    """A project's memories don't bleed into another project's set."""
    isolated_db.add_project_memory("acme", "ACME-only fact")
    isolated_db.add_project_memory("widgets", "Widgets-only fact")
    assert [r["content"] for r in isolated_db.list_project_memories("acme")] == ["ACME-only fact"]
    assert [r["content"] for r in isolated_db.list_project_memories("widgets")] == ["Widgets-only fact"]


def test_add_project_memory_rejects_blank_project(isolated_db):
    with pytest.raises(ValueError):
        isolated_db.add_project_memory("", "some fact")
    with pytest.raises(ValueError):
        isolated_db.add_project_memory("   ", "some fact")


def test_delete_project_memory_substring(isolated_db):
    isolated_db.add_project_memory("acme", "we use pytest, not unittest")
    isolated_db.add_project_memory("acme", "linter is biome, not eslint")
    isolated_db.add_project_memory("acme", "deploy via fly.io")
    n = isolated_db.delete_project_memories_matching("acme", "lint")
    assert n == 1
    remaining = [r["content"] for r in isolated_db.list_project_memories("acme")]
    assert "biome" not in " ".join(remaining)


def test_delete_project_memory_refuses_blank_pattern(isolated_db):
    isolated_db.add_project_memory("acme", "x")
    with pytest.raises(ValueError):
        isolated_db.delete_project_memories_matching("acme", "")


# --- remember/forget tool paths ------------------------------------------


def test_remember_project_scope_writes_to_project_label(isolated_db):
    """Round-trip via the tool: create conv with project label, call
    remember(scope='project'), verify the row landed under that label."""
    conv = isolated_db.create_conversation(
        title="t", model="m", cwd="C:/x", project="acme",
    )
    res = _run(tools.remember(conv["id"], "uses pnpm not npm", scope="project"))
    assert res["ok"] is True
    rows = isolated_db.list_project_memories("acme")
    assert any("pnpm" in r["content"] for r in rows)


def test_remember_project_refuses_when_no_project_label(isolated_db):
    conv = isolated_db.create_conversation(title="t", model="m", cwd="C:/x")
    # No `project` arg → unlabelled
    res = _run(tools.remember(conv["id"], "x", scope="project"))
    assert res["ok"] is False
    assert "project" in (res.get("error") or "").lower()


def test_remember_project_refuses_without_conv_id(isolated_db):
    res = _run(tools.remember(None, "x", scope="project"))
    assert res["ok"] is False


def test_forget_project_scope_round_trip(isolated_db):
    conv = isolated_db.create_conversation(
        title="t", model="m", cwd="C:/x", project="acme",
    )
    isolated_db.add_project_memory("acme", "fact one")
    isolated_db.add_project_memory("acme", "fact two")
    res = _run(tools.forget(conv["id"], "one", scope="project"))
    assert res["ok"] is True
    remaining = isolated_db.list_project_memories("acme")
    assert all("one" not in r["content"] for r in remaining)


def test_invalid_scope_rejected_by_remember():
    res = _run(tools.remember("any", "x", scope="bogus"))
    assert res["ok"] is False
    err = (res.get("error") or "").lower()
    assert "conversation" in err and "project" in err and "global" in err


# --- prompt loader -------------------------------------------------------


def test_load_project_memory_empty_for_blank_project(isolated_db):
    assert tools.load_project_memory_for_prompt(None) == ""
    assert tools.load_project_memory_for_prompt("") == ""
    assert tools.load_project_memory_for_prompt("   ") == ""


def test_load_project_memory_renders_section(isolated_db):
    isolated_db.add_project_memory("acme", "uses pnpm")
    isolated_db.add_project_memory("acme", "biome for lint")
    text = tools.load_project_memory_for_prompt("acme")
    assert "Project memory" in text
    assert "acme" in text
    assert "pnpm" in text
    assert "biome" in text


def test_build_system_prompt_includes_project_section(isolated_db):
    """End-to-end: a conversation with a project label gets the
    project memory section spliced into its system prompt."""
    isolated_db.add_project_memory("acme", "deploy via fly.io")
    conv = isolated_db.create_conversation(
        title="t", model="m", cwd="C:/x", project="acme",
    )
    sp = prompts.build_system_prompt("C:/x", conv_id=conv["id"])
    assert "Project memory" in sp
    assert "fly.io" in sp


def test_build_system_prompt_omits_project_section_when_unlabelled(isolated_db):
    isolated_db.add_project_memory("acme", "deploy via fly.io")
    conv = isolated_db.create_conversation(title="t", model="m", cwd="C:/x")
    sp = prompts.build_system_prompt("C:/x", conv_id=conv["id"])
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
