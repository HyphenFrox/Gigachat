"""Tests for conversation CRUD, search, pin, and tag features."""

from __future__ import annotations

import pytest

# Whole module is fast + offline — runs in the smoke tier.
pytestmark = pytest.mark.smoke


def _make_conv(db, title="Hello", model="gemma4:e4b", cwd="/tmp"):
    """Helper — create a conversation with sensible defaults."""
    return db.create_conversation(title=title, model=model, cwd=cwd)


def test_create_and_get_conversation(isolated_db):
    """Creating a conversation should round-trip every field correctly."""
    db = isolated_db
    conv = _make_conv(db, title="Test chat")
    assert conv["title"] == "Test chat"
    assert conv["model"] == "gemma4:e4b"
    # New conversations default to unpinned with no tags.
    assert conv["pinned"] is False
    assert conv["tags"] == []
    fetched = db.get_conversation(conv["id"])
    assert fetched == conv


def test_pin_unpin_conversation(isolated_db):
    """Toggling `pinned` via update_conversation should persist."""
    db = isolated_db
    conv = _make_conv(db)
    pinned = db.update_conversation(conv["id"], pinned=True)
    assert pinned["pinned"] is True
    unpinned = db.update_conversation(conv["id"], pinned=False)
    assert unpinned["pinned"] is False


def test_pinned_conversations_sort_first(isolated_db):
    """list_conversations should return pinned rows before unpinned ones."""
    db = isolated_db
    a = _make_conv(db, title="A")
    b = _make_conv(db, title="B")
    c = _make_conv(db, title="C")
    # Pin only B — it should jump to the top regardless of update order.
    db.update_conversation(b["id"], pinned=True)
    titles = [c["title"] for c in db.list_conversations()]
    assert titles[0] == "B"
    # The two unpinned rows still sort by updated_at among themselves.
    assert set(titles[1:]) == {"A", "C"}


def test_sort_by_last_user_message_not_last_activity(isolated_db):
    """User messaging in an old conversation must float it to the top,
    even if a different conversation has been chugging through agent
    activity in the background.

    Sidebar sorts by `last_user_message_at`, not `updated_at` — agent
    tool calls bump only `updated_at`, so a busy background turn
    shouldn't drift its conversation past one where the user just
    sent a message."""
    db = isolated_db
    a = _make_conv(db, title="A — user just messaged")
    b = _make_conv(db, title="B — agent still chugging")
    # Background agent activity in B (assistant + tool messages bump
    # updated_at but not last_user_message_at).
    db.add_message(b["id"], "user", "kick off")  # last_user_message_at on B = T0
    import time
    time.sleep(0.02)
    db.add_message(b["id"], "assistant", "working...")
    db.add_message(b["id"], "tool", '{"ok": true}')
    db.add_message(b["id"], "assistant", "still working...")
    # Slightly later, user sends a fresh message in A.
    time.sleep(0.02)
    db.add_message(a["id"], "user", "actually do this in A please")
    titles = [c["title"] for c in db.list_conversations()]
    # A's last user message is newer than B's, so A leads.
    assert titles[0] == "A — user just messaged", (
        f"Expected A first (most recent user message), got: {titles}"
    )


def test_assistant_only_activity_does_not_promote(isolated_db):
    """Just an assistant or tool message landing in a conversation
    must NOT promote it above another conversation with a fresher
    user message. This is the behaviour the user asked for: 'sort by
    where I last messaged', not 'sort by any activity'."""
    db = isolated_db
    a = _make_conv(db, title="A")
    b = _make_conv(db, title="B")
    import time
    db.add_message(a["id"], "user", "hello")  # A's last user msg = now
    time.sleep(0.02)
    db.add_message(b["id"], "user", "hi")     # B's last user msg = newer
    time.sleep(0.02)
    db.add_message(a["id"], "assistant", "background streamed reply")
    # Even though A had the most recent activity (the assistant), B
    # should still lead because its USER message is newer.
    titles = [c["title"] for c in db.list_conversations()]
    assert titles[0] == "B"


def test_edit_user_message_promotes_conversation(isolated_db):
    """Editing a user message via the edit-and-regenerate flow is
    itself a user action — sidebar should treat it like a fresh
    user message and float the conversation up."""
    db = isolated_db
    a = _make_conv(db, title="A")
    b = _make_conv(db, title="B")
    import time
    a_msg = db.add_message(a["id"], "user", "hello")  # A leads briefly
    time.sleep(0.02)
    db.add_message(b["id"], "user", "hi")  # B takes over
    time.sleep(0.02)
    db.update_user_message_content(a_msg["id"], "hello, edited")
    # Edit on A should bump its sort key past B.
    titles = [c["title"] for c in db.list_conversations()]
    assert titles[0] == "A"


def test_brand_new_conversation_sorts_by_created_at(isolated_db):
    """A chat that has zero user messages yet (just-created, agent
    hasn't started, etc.) falls back to created_at via the migration
    backfill so it still gets a sensible sidebar position."""
    db = isolated_db
    import time
    a = _make_conv(db, title="A")
    time.sleep(0.02)
    b = _make_conv(db, title="B newer")
    titles = [c["title"] for c in db.list_conversations()]
    # B is newer; sorts first even with no messages.
    assert titles[0] == "B newer"


def test_tags_serialize_and_clean(isolated_db):
    """Tag list should be stored as JSON; blanks/whitespace are dropped."""
    db = isolated_db
    conv = _make_conv(db)
    updated = db.update_conversation(
        conv["id"], tags=["work", "  ", "important", ""]
    )
    assert updated["tags"] == ["work", "important"]
    # Setting tags=None clears the list back to empty.
    cleared = db.update_conversation(conv["id"], tags=None)
    # None is treated as "no tag column update" because update_conversation
    # only includes a SET clause for non-None values that pass the type
    # checks. The previous value should still be there.
    # Setting an empty list explicitly DOES clear it (cleaned -> empty list
    # -> stored as NULL -> hydrates back as []).
    assert cleared["tags"] == ["work", "important"]
    cleared_explicit = db.update_conversation(conv["id"], tags=[])
    assert cleared_explicit["tags"] == []


def test_search_finds_by_title(isolated_db):
    """Search should match a substring of the conversation title."""
    db = isolated_db
    _make_conv(db, title="React refactor planning")
    _make_conv(db, title="Bash debug session")
    hits = db.search_conversations("refactor")
    assert len(hits) == 1
    assert hits[0]["title"] == "React refactor planning"


def test_search_finds_by_message_content(isolated_db):
    """Search should match a substring of any message body in the convo."""
    db = isolated_db
    conv = _make_conv(db, title="Misc")
    db.add_message(conv["id"], "user", "Please look at the SQL injection issue")
    db.add_message(conv["id"], "assistant", "Sure thing")
    hits = db.search_conversations("SQL injection")
    assert len(hits) == 1
    assert hits[0]["id"] == conv["id"]


def test_search_finds_by_tag(isolated_db):
    """Search should match a substring of a tag."""
    db = isolated_db
    conv = _make_conv(db, title="Anything")
    db.update_conversation(conv["id"], tags=["security-audit"])
    hits = db.search_conversations("security")
    assert len(hits) == 1
    assert hits[0]["id"] == conv["id"]


def test_search_empty_query_returns_empty(isolated_db):
    """An empty/whitespace query short-circuits to [] (not 'all')."""
    db = isolated_db
    _make_conv(db)
    assert db.search_conversations("") == []
    assert db.search_conversations("   ") == []


def test_search_pinned_first(isolated_db):
    """Search results should also respect the pinned-first ordering."""
    db = isolated_db
    older = _make_conv(db, title="older react")
    _ = _make_conv(db, title="newer react")
    db.update_conversation(older["id"], pinned=True)
    hits = db.search_conversations("react")
    assert hits[0]["id"] == older["id"]
