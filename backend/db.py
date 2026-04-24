"""SQLite-backed persistence for conversations and messages.

Schema:
  conversations  — id, title, model, auto_approve, cwd, created_at,
                   updated_at, pinned, tags
  messages       — id, conversation_id, role, content, tool_calls, images,
                   created_at, pinned

`tool_calls` is a JSON-encoded array carrying per-row metadata (tool call id,
name, arguments, attached screenshot filename). `images` is a JSON-encoded
array of filenames (relative to tools.UPLOAD_DIR) that the user attached to
the row — used to reconstruct multimodal history across reloads.

`tags` on conversations is a JSON-encoded list of short strings for free-form
labelling ("work", "experiments", …). `pinned` is a boolean flag — pinned
conversations sort to the top of the sidebar regardless of recency.

Migrations are intentionally additive (ADD COLUMN IF NOT EXISTS via a
try/except) so upgrading an existing DB in-place doesn't require a wipe.
"""

import sqlite3
import json
import re
import struct
import time
import uuid
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"


def _conn() -> sqlite3.Connection:
    """Open a connection with WAL mode + foreign-key enforcement.

    `check_same_thread=False` is safe here because FastAPI sync routes run on
    a threadpool and we always wrap writes in a context manager — sqlite3
    holds a write lock per-statement under WAL so there's no data-race.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode = WAL")
    c.execute("PRAGMA foreign_keys = ON")
    return c


def init() -> None:
    """Create tables on first run and apply additive column migrations."""
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                model TEXT NOT NULL,
                auto_approve INTEGER NOT NULL DEFAULT 0,
                cwd TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_calls TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
            -- Per-message embeddings for semantic recall (RAG).
            -- `embedding` is float32 little-endian, packed with struct.pack.
            -- One row per user/assistant message (tool rows are not embedded
            -- — their content is too noisy and rarely useful for recall).
            CREATE TABLE IF NOT EXISTS message_embeddings (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_embeddings_conv ON message_embeddings(conversation_id);
            -- Scheduled agent runs fired by the background daemon in app.py.
            -- `interval_seconds` is NULL for one-shots; set for recurring jobs
            -- (we re-schedule next_run_at = now + interval after each fire).
            -- `cwd` captures the working directory the scheduled run should
            -- use — we freeze it at creation time so the job is reproducible
            -- even if the user later edits the parent conversation.
            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                prompt TEXT NOT NULL,
                next_run_at REAL NOT NULL,
                interval_seconds INTEGER,
                cwd TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_scheduled_next ON scheduled_tasks(next_run_at);
            -- Doc-chunk store for the `doc_index` / `doc_search` tools. One
            -- row per text chunk after splitting a source file with overlap.
            -- `vector` is float32 little-endian (same pack/unpack helpers as
            -- message_embeddings). We do NOT enforce a foreign key to any
            -- other table — this index is independent of conversations and
            -- re-indexing the same file simply deletes then re-inserts.
            CREATE TABLE IF NOT EXISTS doc_chunks (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                vector BLOB NOT NULL,
                model TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_doc_chunks_path ON doc_chunks(path);
            -- User-defined lifecycle hooks. Each row is a shell command that
            -- fires on a specific event in the agent loop (pre-tool,
            -- post-tool, user_prompt_submit, turn_done). The command runs
            -- with a short timeout in a subprocess; its stdout/stderr is
            -- injected back into the conversation so the agent can see it
            -- on the next iteration. `matcher` is an optional substring —
            -- for pre-tool/post-tool events it restricts the hook to tool
            -- names that contain the matcher text; ignored for the other
            -- events. `enabled=0` disables without deleting so the user
            -- can toggle quickly from the settings UI.
            CREATE TABLE IF NOT EXISTS hooks (
                id TEXT PRIMARY KEY,
                event TEXT NOT NULL,
                matcher TEXT,
                command TEXT NOT NULL,
                timeout_seconds INTEGER NOT NULL DEFAULT 10,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_hooks_event ON hooks(event, enabled);

            -- Global memories: facts that should be visible to the model in
            -- EVERY conversation, not just the one where they were added.
            -- Distinct from per-conversation memory (data/memory/<conv>.md):
            -- - per-conv memory is for in-flight context ("we decided to use
            --   approach X for this refactor")
            -- - global memory is for durable facts ("user prefers SCSS over
            --   CSS", "this codebase uses pytest, never unittest")
            -- The block is injected into the system prompt of every new turn
            -- via prompts.build_system_prompt → tools.load_global_memory_for_prompt.
            -- Managed primarily through the Settings UI, but the agent can
            -- also write here via remember(scope="global").
            CREATE TABLE IF NOT EXISTS global_memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                topic TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_global_memories_topic
                ON global_memories(topic);

            -- Web Push subscriptions — one row per browser/device the user has
            -- granted push permission on. `endpoint` is the unique push-service
            -- URL the browser hands back during pushManager.subscribe(); we use
            -- it as the primary key so re-subscribing the same browser updates
            -- the keys in place rather than leaking duplicate rows.
            -- `p256dh` and `auth` are the subscription's keys (base64url-encoded
            -- strings) passed to pywebpush when we encrypt an outgoing payload.
            CREATE TABLE IF NOT EXISTS push_subscriptions (
                endpoint TEXT PRIMARY KEY,
                p256dh TEXT NOT NULL,
                auth TEXT NOT NULL,
                user_agent TEXT,
                created_at REAL NOT NULL
            );

            -- MCP (Model Context Protocol) servers — external processes we
            -- spawn over stdio and whose advertised tools get merged into
            -- TOOL_SCHEMAS so the agent can call them like native tools.
            -- `args_json` is a JSON array of CLI args; `env_json` is a JSON
            -- dict of env-var overrides. `name` is the user-visible label
            -- and also becomes the tool-namespace prefix (mcp__<name>__tool).
            CREATE TABLE IF NOT EXISTS mcp_servers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                command TEXT NOT NULL,
                args_json TEXT NOT NULL DEFAULT '[]',
                env_json TEXT NOT NULL DEFAULT '{}',
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            -- User-level settings (key/value). Any small piece of config that
            -- the user chooses in the UI and should persist across restarts
            -- lives here, one row per key. Values are stored as JSON text so
            -- a future setting can hold numbers, booleans, or structured data
            -- without a schema migration. Current keys:
            --   default_chat_model — model tag used for new conversations
            --                        (falls back to the auto-tuner's pick).
            CREATE TABLE IF NOT EXISTS user_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            -- User-defined tools, created only via Settings → Tools (the LLM
            -- has no equivalent tool-call route). Each row becomes a new
            -- first-class entry in the Ollama tool palette on the next turn
            -- (and for every future conversation). `code` is Python source
            -- that must define `def run(args: dict) -> dict`; it executes in
            -- an isolated subprocess under `data/tools_venv/` where `deps`
            -- have been pip-installed. `schema_json` is the JSON-schema
            -- `parameters` block advertised to the model. `category` mirrors
            -- TOOL_CATEGORIES (read / write) so the permission-mode gate
            -- works the same as for built-ins. `enabled=0` keeps the row in
            -- the DB but drops it from the palette, letting the user pause
            -- without deleting.
            CREATE TABLE IF NOT EXISTS user_tools (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                schema_json TEXT NOT NULL DEFAULT '{}',
                code TEXT NOT NULL,
                deps_json TEXT NOT NULL DEFAULT '[]',
                category TEXT NOT NULL DEFAULT 'write',
                timeout_seconds INTEGER NOT NULL DEFAULT 60,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_user_tools_enabled
                ON user_tools(enabled);

            -- API credentials / tokens the user pastes once and references
            -- from http_request via `{{secret:NAME}}` placeholders. Values
            -- are stored in the clear — the threat model is a single-user
            -- local app where an attacker with filesystem access already
            -- owns everything. The UI masks values by default so someone
            -- glancing at the screen doesn't see them; `data/app.db` sits
            -- inside the 0700 data/ directory that also holds the auth
            -- secret, so no new attack surface is introduced.
            CREATE TABLE IF NOT EXISTS secrets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                value TEXT NOT NULL,
                description TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            -- Side tasks flagged by the agent mid-turn via spawn_task. These
            -- are drive-by observations ("by the way, README badge is stale")
            -- that would bloat the current change if handled inline. The UI
            -- renders them as chips under the streaming assistant bubble;
            -- clicking Open spins the prompt off into a new conversation.
            -- status: 'pending' (chip visible), 'opened' (user clicked Open
            -- → opened_conversation_id set), 'dismissed' (user clicked X).
            CREATE TABLE IF NOT EXISTS side_tasks (
                id TEXT PRIMARY KEY,
                source_conversation_id TEXT NOT NULL,
                title TEXT NOT NULL,
                prompt TEXT NOT NULL,
                tldr TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at REAL NOT NULL,
                opened_at REAL,
                opened_conversation_id TEXT,
                FOREIGN KEY(source_conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_side_tasks_source
                ON side_tasks(source_conversation_id, status);

            -- Git worktrees created by the agent via create_worktree for risky
            -- edits. Each row maps a conversation to a throwaway worktree
            -- directory so the agent can later swap the conversation's cwd
            -- into it and merge/discard on completion. `branch` is the new
            -- branch name created off of `base_ref`; `path` is the absolute
            -- directory where git checked the worktree out. We track these
            -- in the DB so `git worktree list` stays the source of truth for
            -- git but we have a per-conversation index of what we created.
            CREATE TABLE IF NOT EXISTS worktrees (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                repo_path TEXT NOT NULL,
                path TEXT NOT NULL,
                branch TEXT NOT NULL,
                base_ref TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at REAL NOT NULL,
                removed_at REAL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_worktrees_conv
                ON worktrees(conversation_id, status);
            -- Auto-built per-cwd codebase indexes.
            --
            -- Tracks which working directories have been crawled + embedded for
            -- codebase_search, plus timestamps so the background indexer can
            -- decide whether to refresh. One row per distinct cwd (UNIQUE).
            -- Chunks themselves live in `doc_chunks` with `path LIKE '<cwd>%'`
            -- so we reuse the existing search code without duplicating tables.
            CREATE TABLE IF NOT EXISTS codebase_indexes (
                cwd TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                last_indexed_at REAL,
                file_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                updated_at REAL NOT NULL
            );
            -- Indexed documentation URLs — third-party docs the user pins
            -- into the semantic index so the agent can answer "what does
            -- React's useTransition do?" without a live web fetch.
            -- Chunks share the `doc_chunks` table; rows live under
            -- `path = "url:<page-url>"` so they can be isolated with a
            -- simple LIKE filter. `url` is the seed URL (not the per-page
            -- URL) and is the stable identity of the registry row.
            CREATE TABLE IF NOT EXISTS doc_urls (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL UNIQUE,
                title TEXT,
                max_pages INTEGER NOT NULL DEFAULT 20,
                same_origin_only INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'pending',
                pages_crawled INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                last_indexed_at REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            """
        )
        # Additive migration: images column (JSON array of upload filenames).
        # `ALTER TABLE ADD COLUMN` is idempotent-by-exception on SQLite —
        # running it twice raises OperationalError("duplicate column name"),
        # which we swallow so upgrades are a no-op.
        _migrated_permission_mode = False
        for ddl in (
            "ALTER TABLE messages ADD COLUMN images TEXT",
            # pinned rows survive compaction — user-level escape hatch for
            # "this detail must not be forgotten, no matter how long the
            # conversation gets".
            "ALTER TABLE messages ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0",
            # Conversation-level pin: pinned chats float to the top of the
            # sidebar even when older than the most-recently-touched ones.
            "ALTER TABLE conversations ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0",
            # Conversation-level tags: JSON-encoded list of short labels.
            # Stored as TEXT so we can do simple LIKE filters in search.
            "ALTER TABLE conversations ADD COLUMN tags TEXT",
            # Crash-resilience state machine. 'idle' means no turn is running;
            # 'running' means an agent loop is mid-turn; 'error' means the
            # last turn raised and never reset. The startup resumer uses this
            # (plus the queued_inputs table) to decide what to replay.
            "ALTER TABLE conversations ADD COLUMN state TEXT NOT NULL DEFAULT 'idle'",
            # Optional per-conversation persona — a free-text system-prompt
            # extension. NULL means "no persona override"; the system prompt
            # is built without any persona section in that case.
            "ALTER TABLE conversations ADD COLUMN persona TEXT",
            # Optional soft budgets. NULL = unbounded. When the agent loop
            # starts a turn, it checks cumulative usage against these caps
            # and refuses to begin a new turn if the budget is exhausted.
            # `budget_turns` counts visible user+assistant messages; tokens
            # are estimated from cumulative character length of the history.
            "ALTER TABLE conversations ADD COLUMN budget_turns INTEGER",
            "ALTER TABLE conversations ADD COLUMN budget_tokens INTEGER",
            # Per-conversation permission mode (replaces the old auto_approve
            # boolean with a three-value enum).
            #   read_only      — write tools are refused outright before any
            #                    approval, so the agent can still observe /
            #                    plan without being able to mutate anything.
            #   approve_edits  — DEFAULT. Read tools run silently, write tools
            #                    pause for manual approval (the old "manual
            #                    approve" behavior, minus the friction of
            #                    approving every harmless `read_file` call).
            #   allow_all      — no prompts, anything goes (equivalent to the
            #                    old auto_approve=1).
            # Default is 'approve_edits' so an upgraded DB with the legacy
            # default (auto_approve=0) still behaves identically from the
            # user's perspective — reads become zero-click but writes still
            # prompt. The backfill below promotes existing auto_approve=1
            # rows to 'allow_all' so the user's opt-in is preserved.
            "ALTER TABLE conversations ADD COLUMN permission_mode TEXT NOT NULL DEFAULT 'approve_edits'",
            # Scheduled-task target: when set, a scheduled fire RESUMES this
            # conversation (appending the prompt as the next user turn) instead
            # of creating a brand-new one. This is how the agent's self-paced
            # `schedule_wakeup` tool works — it enqueues a one-shot row that
            # points back at the originating conversation so the wake-up lands
            # in the same chat the agent was working in.
            "ALTER TABLE scheduled_tasks ADD COLUMN target_conversation_id TEXT",
            # Optional project label for grouping conversations in the sidebar.
            # NULL = ungrouped (rendered under a "No project" header). Just a
            # free-text string — no separate projects table — so users can
            # type any name and it auto-appears as a section.
            "ALTER TABLE conversations ADD COLUMN project TEXT",
            # Scheduled-task kind. Defaults to 'task' for classic scheduled_task
            # rows (fires in a brand-new conversation), 'wakeup' for one-shot
            # resume-this-chat rows, 'loop' for recurring resume-this-chat rows
            # (autonomous loop mode). Used by the UI to render appropriate
            # controls (Stop-loop button for loops, cancel for tasks).
            "ALTER TABLE scheduled_tasks ADD COLUMN kind TEXT NOT NULL DEFAULT 'task'",
            # Per-conversation "current bash directory". Persists across bash
            # calls so the model can `cd subdir` and have the next bash call
            # start there — mimics a long-lived shell session. NULL means
            # "no override" (bash starts at the conversation's fixed `cwd`).
            # File tools (read_file/edit_file/…) intentionally ignore this;
            # they always resolve against `cwd` so the workspace root stays
            # stable no matter where the shell wandered.
            "ALTER TABLE conversations ADD COLUMN bash_cwd TEXT",
        ):
            try:
                c.execute(ddl)
                # Track whether we just created the permission_mode column so
                # we know to run the one-time backfill below — running it on
                # every startup would silently stomp user edits made via the
                # new UI.
                if "permission_mode" in ddl:
                    _migrated_permission_mode = True
            except sqlite3.OperationalError:
                pass
        if _migrated_permission_mode:
            # Preserve intent for users who had flipped the legacy switch on.
            # Rows where auto_approve=1 want "run everything without asking",
            # which maps to 'allow_all'. Everything else stays at the column
            # default of 'approve_edits'.
            c.execute(
                "UPDATE conversations SET permission_mode = 'allow_all' WHERE auto_approve = 1"
            )
        # Persistent queue of user messages submitted while a turn was still
        # running. We used to keep this in-memory only, which meant queued
        # messages were dropped on process restart. Now they survive a crash
        # so the startup resumer can replay them when it restarts the turn.
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS queued_inputs (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                text TEXT NOT NULL,
                images TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_queued_conv ON queued_inputs(conversation_id, created_at);
            """
        )


def create_conversation(
    title: str,
    model: str,
    cwd: str,
    auto_approve: bool = False,
    permission_mode: str | None = None,
    project: str | None = None,
) -> dict:
    """Insert a new conversation row and return the hydrated dict.

    `permission_mode` is the new source of truth; `auto_approve` is kept as a
    parameter so legacy call-sites (e.g. the scheduled-task daemon that fires
    prompts unattended) still work. When both are given, permission_mode wins.

    `project` is a free-text label used purely for sidebar grouping. NULL /
    empty string means "ungrouped". Normalized with the same rule used in
    update_conversation (whitespace-collapsed, 80-char cap) so users can type
    any name and it auto-appears as a section.
    """
    cid = str(uuid.uuid4())
    now = time.time()
    if permission_mode not in {"read_only", "plan", "approve_edits", "allow_all"}:
        permission_mode = "allow_all" if auto_approve else "approve_edits"
    # Derive the legacy bit from permission_mode so the two fields never drift
    # apart inside a single INSERT — avoids a race where the column default
    # gets set before the UPDATE to allow_all.
    auto_bit = 1 if permission_mode == "allow_all" else 0
    proj = None
    if project is not None:
        s = " ".join(str(project).split())[:80]
        proj = s if s else None
    with _conn() as c:
        c.execute(
            "INSERT INTO conversations "
            "(id, title, model, auto_approve, permission_mode, cwd, project, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (cid, title, model, auto_bit, permission_mode, cwd, proj, now, now),
        )
    return get_conversation(cid)


def list_conversations() -> list[dict]:
    """Return all conversations, pinned first then most-recently-updated."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM conversations ORDER BY pinned DESC, updated_at DESC"
        ).fetchall()
    return [_row_to_conversation(r) for r in rows]


def search_conversations(query: str, limit: int = 50) -> list[dict]:
    """Substring-search title, tags, and message content; return matching
    conversations with pinned-first ordering.

    Implementation note — we deliberately do a single SQL pass with a UNION
    against the messages table rather than a separate full-text index, because
    the corpus is small (one user, a few hundred conversations max) and a
    LIKE scan finishes in milliseconds. If this ever stops feeling instant,
    swap in SQLite's FTS5 module — the wrapper signature is stable.

    `query` is escaped with `?` parameter binding to prevent SQL injection.
    Empty/whitespace-only queries return an empty list rather than
    everything (avoids accidentally pulling thousands of rows on a typo).
    """
    q = (query or "").strip()
    if not q:
        return []
    pattern = f"%{q}%"
    with _conn() as c:
        rows = c.execute(
            """
            SELECT c.* FROM conversations AS c
            WHERE c.id IN (
                SELECT id FROM conversations
                  WHERE title LIKE ? OR (tags IS NOT NULL AND tags LIKE ?)
                UNION
                SELECT conversation_id FROM messages
                  WHERE content LIKE ?
            )
            ORDER BY c.pinned DESC, c.updated_at DESC
            LIMIT ?
            """,
            (pattern, pattern, pattern, int(limit)),
        ).fetchall()
    return [_row_to_conversation(r) for r in rows]


def get_conversation(cid: str) -> dict | None:
    """Fetch one conversation by id, or None."""
    with _conn() as c:
        row = c.execute("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()
    return _row_to_conversation(row) if row else None


def update_conversation(cid: str, **fields: Any) -> dict | None:
    """Patch allowed fields on a conversation; touch updated_at.

    `tags` may be passed as a list/tuple — we serialize it to JSON. `pinned`
    is coerced to a 0/1 int. Unknown keys are silently ignored so the route
    handler can pass body.dict() without filtering.
    """
    allowed = {
        "title",
        "model",
        "auto_approve",
        "cwd",
        "pinned",
        "tags",
        "persona",
        "budget_turns",
        "budget_tokens",
        "permission_mode",
        "project",
    }
    valid_modes = {"read_only", "plan", "approve_edits", "allow_all"}
    sets = []
    values = []
    for k, v in fields.items():
        if k not in allowed:
            continue
        # None always means "leave this column alone" — mirrors how the
        # FastAPI patch handler filters None out of the body. This keeps
        # the layers consistent: callers pass None to mean "don't touch".
        if v is None:
            continue
        if k == "auto_approve" or k == "pinned":
            v = 1 if v else 0
        elif k == "permission_mode":
            # Reject unknown values silently (dropped from the SET list) so a
            # hostile / buggy caller can't smuggle arbitrary text into the
            # column. Legitimate UI callers only ever pass one of the three
            # known values.
            if str(v) not in valid_modes:
                continue
            v = str(v)
        elif k == "persona":
            # Persona is free-text; normalize whitespace-only to NULL so the
            # UI's "clear" affordance (empty the textarea + save) unsets the
            # override rather than storing a string of spaces.
            s = str(v).strip()
            v = s if s else None
        elif k == "project":
            # Project label: normalize to a short, whitespace-collapsed string
            # or NULL. Cap at 80 chars so the sidebar section headers stay
            # readable. Empty / whitespace-only input means "clear project".
            s = " ".join(str(v).split())[:80]
            v = s if s else None
        elif k in {"budget_turns", "budget_tokens"}:
            # Caller passed 0, "", or a literal zero → clear the budget.
            # Negative numbers are rejected (no meaningful interpretation).
            try:
                n = int(v)
            except (TypeError, ValueError):
                continue
            if n <= 0:
                v = None
            else:
                v = n
        elif k == "tags":
            # Normalize to a JSON-encoded list of short non-empty strings.
            # Accepts list/tuple — anything else is silently ignored. An
            # empty list (after cleaning) clears the column to NULL, which
            # is the canonical "no tags" state.
            if isinstance(v, (list, tuple)):
                cleaned = [str(t).strip() for t in v if str(t).strip()]
                v = json.dumps(cleaned) if cleaned else None
            else:
                continue
        sets.append(f"{k} = ?")
        values.append(v)
    if not sets:
        return get_conversation(cid)
    sets.append("updated_at = ?")
    values.append(time.time())
    values.append(cid)
    with _conn() as c:
        c.execute(f"UPDATE conversations SET {', '.join(sets)} WHERE id = ?", values)
    return get_conversation(cid)


def delete_conversation(cid: str) -> None:
    """Drop a conversation (messages cascade via FK)."""
    with _conn() as c:
        c.execute("DELETE FROM conversations WHERE id = ?", (cid,))


# ---------------------------------------------------------------------------
# Conversation state machine for crash resilience
#
# Every conversation has a `state` column: 'idle' | 'running' | 'error'. The
# agent loop transitions it at well-defined points; the startup resumer uses
# the final value to decide what to do with each conversation after a crash.
# ---------------------------------------------------------------------------
_VALID_STATES = {"idle", "running", "error"}


def set_conversation_state(cid: str, state: str) -> None:
    """Write the conversation's run-state. Unknown states are rejected so a
    typo in the agent loop can't poison the resumer."""
    if state not in _VALID_STATES:
        raise ValueError(f"invalid state: {state!r}")
    with _conn() as c:
        c.execute(
            "UPDATE conversations SET state = ? WHERE id = ?",
            (state, cid),
        )


def get_bash_cwd(cid: str) -> str | None:
    """Return the conversation's persistent bash cwd, or None if unset.

    Used by the `bash` tool to emulate a long-lived shell session: a
    `cd subdir` in one call is reflected in the next call's starting dir.
    """
    with _conn() as c:
        row = c.execute(
            "SELECT bash_cwd FROM conversations WHERE id = ?",
            (cid,),
        ).fetchone()
    if not row:
        return None
    return row["bash_cwd"]


def set_bash_cwd(cid: str, path: str | None) -> None:
    """Persist the conversation's bash cwd. Pass `None` to clear the override."""
    with _conn() as c:
        c.execute(
            "UPDATE conversations SET bash_cwd = ? WHERE id = ?",
            (path, cid),
        )


def list_conversations_by_state(state: str) -> list[dict]:
    """Return every conversation currently in the given state.

    Used at startup to find conversations that were mid-turn when the process
    died. Callers should decide whether to resume, mark errored, or reset
    each one.
    """
    if state not in _VALID_STATES:
        raise ValueError(f"invalid state: {state!r}")
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM conversations WHERE state = ? ORDER BY updated_at ASC",
            (state,),
        ).fetchall()
    return [_row_to_conversation(r) for r in rows]


# ---------------------------------------------------------------------------
# Persistent user-input queue
#
# When the user sends a follow-up message while a turn is still running, we
# stash it here. The agent loop drains the queue between iterations; the
# startup resumer drains it when it restarts an interrupted turn. DB-backed
# (not in-memory) so a server crash doesn't lose messages the user typed.
# ---------------------------------------------------------------------------
def enqueue_user_input(cid: str, text: str, images: list[str] | None) -> str:
    """Append one queued user-input row. Returns the new row id."""
    qid = str(uuid.uuid4())
    payload_images = json.dumps(images) if images else None
    with _conn() as c:
        c.execute(
            "INSERT INTO queued_inputs (id, conversation_id, text, images, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (qid, cid, text or "", payload_images, time.time()),
        )
    return qid


def drain_queued_inputs(cid: str) -> list[dict]:
    """Pop every queued input for `cid` in FIFO order and delete them.

    Returns a list of {text, images} dicts (images is a list or None).
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM queued_inputs WHERE conversation_id = ? ORDER BY created_at ASC",
            (cid,),
        ).fetchall()
        if not rows:
            return []
        c.execute(
            "DELETE FROM queued_inputs WHERE conversation_id = ?",
            (cid,),
        )
    out: list[dict] = []
    for r in rows:
        imgs = None
        try:
            imgs = json.loads(r["images"]) if r["images"] else None
        except Exception:
            imgs = None
        out.append({"text": r["text"] or "", "images": imgs})
    return out


def has_queued_inputs(cid: str) -> bool:
    """Return True if there's at least one pending queued message."""
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM queued_inputs WHERE conversation_id = ? LIMIT 1",
            (cid,),
        ).fetchone()
    return row is not None


def touch_conversation(cid: str) -> None:
    with _conn() as c:
        c.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (time.time(), cid))


def add_message(
    cid: str,
    role: str,
    content: str,
    tool_calls: list | None = None,
    images: list[str] | None = None,
) -> dict:
    """Append a message row and return the hydrated dict.

    `images` is stored as a JSON-encoded array of filenames (under
    tools.UPLOAD_DIR). Only user-role rows typically carry images today, but
    the column accepts them on any role for forward-compat.
    """
    mid = str(uuid.uuid4())
    now = time.time()
    tc = json.dumps(tool_calls) if tool_calls else None
    imgs = json.dumps(images) if images else None
    with _conn() as c:
        c.execute(
            "INSERT INTO messages (id, conversation_id, role, content, tool_calls, images, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (mid, cid, role, content, tc, imgs, now),
        )
        c.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, cid))
    return {
        "id": mid,
        "conversation_id": cid,
        "role": role,
        "content": content,
        "tool_calls": tool_calls or [],
        "images": images or [],
        "created_at": now,
    }


def add_system_summary(cid: str, content: str) -> dict:
    """Insert a synthetic `system` row carrying an auto-compacted summary.

    Flagged with a well-known prefix in `content` so we can recognize our
    own rows on the next compaction pass (prevents infinite recompression).
    """
    return add_message(cid, "system", content)


def delete_messages(ids: list[str]) -> None:
    """Drop a batch of messages by id. No-op on an empty list.

    Used by the auto-compactor to remove the slice of history it just
    replaced with a single summary row.
    """
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)
    with _conn() as c:
        c.execute(f"DELETE FROM messages WHERE id IN ({placeholders})", tuple(ids))


# Canonical "compressed tool result" payload — written in place of the
# original output when the compactor decides a tool row is too old to keep
# in full AND we couldn't extract anything useful from the body. Structured
# as valid JSON so the existing tool_result parsing code still works.
_COMPRESSED_TOOL_OUTPUT = (
    '{"ok": true, "output": "[older tool output compacted for context]"}'
)

# How many lines (head / tail) we keep when compressing a tool body that
# was too long to keep in full. Five lines of head + ten of tail covers the
# common case where a bash invocation prints a banner up top and the
# meaningful result at the bottom (test summary, exit code, error stack).
# Tunable; not exposed as config because the right value is empirical and
# rarely worth surfacing to end users.
_COMPRESS_HEAD_LINES = 5
_COMPRESS_TAIL_LINES = 10


# Marker substring used to detect rows that are ALREADY in compressed form
# so we don't re-compact them on the next pass (which would degrade further
# each time and eventually drop the head+tail snippet too).
_COMPRESSED_MARKER = "[older tool output compacted"


def is_compressed_tool_output(content: str | None) -> bool:
    """Return True when a tool row's content already carries our compression
    marker, so callers can skip it instead of recompressing.
    """
    if not content:
        return False
    return _COMPRESSED_MARKER in content


def _build_compressed_payload(original: str) -> str:
    """Return a compressed JSON tool-result payload that preserves the
    head and tail of the original body.

    Why preserve snippets at all? An empty `[older tool output compacted]`
    line throws away every signal in the original — exit codes, error
    messages, the path of a file that was just read. Even a few preserved
    lines often let the model resume the task without re-running the tool.

    The function is robust to non-JSON inputs: we always wrap the result
    in `{"ok": true, "output": "..."}` so the tool-message parser doesn't
    care whether the original was JSON or plain text. Newlines inside the
    snippet are preserved (json.dumps will escape them correctly).
    """
    text = original or ""
    lines = text.splitlines()
    if len(lines) <= _COMPRESS_HEAD_LINES + _COMPRESS_TAIL_LINES:
        # Original is already short enough — keep the whole thing but still
        # mark it as compressed so the model knows the row was visited.
        snippet = text
    else:
        head = lines[:_COMPRESS_HEAD_LINES]
        tail = lines[-_COMPRESS_TAIL_LINES:]
        elided = len(lines) - len(head) - len(tail)
        snippet = (
            "\n".join(head)
            + f"\n... [{elided} lines elided to save context] ...\n"
            + "\n".join(tail)
        )
    body = (
        "[older tool output compacted — head + tail preserved for context]\n\n"
        + snippet
    )
    return json.dumps({"ok": True, "output": body})


def compress_tool_outputs(ids: list[str]) -> int:
    """Replace the body of the given tool rows with a head+tail summary.

    This is the cheap first pass of compaction: we keep the row so the
    assistant's tool_call has a matching result (Ollama rejects unpaired
    tool calls), but we throw away the bulk of the output. Unlike the
    earlier blunt "marker only" approach, we now preserve head+tail
    snippets so the model can still see the exit signal / error message
    / first few hits without having to re-run the tool. Returns the
    number of rows touched.
    """
    if not ids:
        return 0
    placeholders = ",".join("?" for _ in ids)
    with _conn() as c:
        # Pull the originals so we can build per-row head+tail payloads.
        rows = c.execute(
            f"SELECT id, content FROM messages WHERE id IN ({placeholders}) AND role = 'tool'",
            tuple(ids),
        ).fetchall()
        touched = 0
        for r in rows:
            new_content = _build_compressed_payload(r["content"] or "")
            cur = c.execute(
                "UPDATE messages SET content = ? WHERE id = ?",
                (new_content, r["id"]),
            )
            touched += cur.rowcount or 0
        return touched


def set_message_pinned(mid: str, pinned: bool) -> dict | None:
    """Toggle the `pinned` flag on one message. Pinned rows are never
    dropped by the auto-compactor and never have their content compressed.
    """
    with _conn() as c:
        c.execute(
            "UPDATE messages SET pinned = ? WHERE id = ?",
            (1 if pinned else 0, mid),
        )
        row = c.execute("SELECT * FROM messages WHERE id = ?", (mid,)).fetchone()
    return _row_to_message(row) if row else None


def update_user_message_content(mid: str, content: str) -> dict | None:
    """Replace the content of one **user** message in place.

    Restricted to user-role rows so the UI's "edit and regenerate" flow can't
    accidentally rewrite assistant text or tool results — those would break
    the model's view of history. Returns the updated row, or None if no
    user-role row matched.
    """
    with _conn() as c:
        cur = c.execute(
            "UPDATE messages SET content = ? WHERE id = ? AND role = 'user'",
            (content or "", mid),
        )
        if cur.rowcount == 0:
            return None
        row = c.execute("SELECT * FROM messages WHERE id = ?", (mid,)).fetchone()
        # Bump conversation updated_at so the sidebar reorders it to the top.
        if row:
            c.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (time.time(), row["conversation_id"]),
            )
    return _row_to_message(row) if row else None


def delete_messages_after(cid: str, mid: str) -> int:
    """Drop every message in `cid` whose created_at is strictly newer than
    the given message's created_at. The anchor message itself is preserved.

    Used by the edit-and-regenerate flow: after the user rewrites the most
    recent message, we throw away everything that came after it (assistant
    reply, tool calls, tool results) so the agent loop can produce a fresh
    response from the corrected prompt.

    Returns the number of rows deleted (0 if mid doesn't exist).
    """
    with _conn() as c:
        anchor = c.execute(
            "SELECT created_at FROM messages WHERE id = ? AND conversation_id = ?",
            (mid, cid),
        ).fetchone()
        if not anchor:
            return 0
        cur = c.execute(
            "DELETE FROM messages WHERE conversation_id = ? AND created_at > ?",
            (cid, anchor["created_at"]),
        )
        return cur.rowcount or 0


def get_messages_by_ids(ids: list[str]) -> list[dict]:
    """Bulk lookup by message id, preserving the input order."""
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    with _conn() as c:
        rows = c.execute(
            f"SELECT * FROM messages WHERE id IN ({placeholders})",
            tuple(ids),
        ).fetchall()
    by_id = {r["id"]: _row_to_message(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


# ---------------------------------------------------------------------------
# Embedding storage — used by the semantic-recall layer in agent.py.
# ---------------------------------------------------------------------------
def _pack_vec(vec: list[float]) -> bytes:
    """Serialize a float list to float32 little-endian bytes."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack_vec(data: bytes) -> list[float]:
    """Inverse of _pack_vec. Size is derived from the byte length."""
    n = len(data) // 4
    if n <= 0:
        return []
    return list(struct.unpack(f"<{n}f", data))


def save_embedding(message_id: str, conversation_id: str, vector: list[float]) -> None:
    """Insert-or-replace one message embedding."""
    if not vector:
        return
    blob = _pack_vec(vector)
    with _conn() as c:
        c.execute(
            """
            INSERT INTO message_embeddings (message_id, conversation_id, embedding)
            VALUES (?, ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET embedding = excluded.embedding
            """,
            (message_id, conversation_id, blob),
        )


def list_embeddings_for_conv(conversation_id: str) -> list[tuple[str, list[float]]]:
    """Return every (message_id, vector) pair for a conversation."""
    with _conn() as c:
        rows = c.execute(
            "SELECT message_id, embedding FROM message_embeddings WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchall()
    return [(r["message_id"], _unpack_vec(r["embedding"])) for r in rows]


def list_all_embeddings() -> list[tuple[str, str, list[float]]]:
    """Return every (message_id, conversation_id, vector) in the database.

    Powers the user-facing cross-conversation semantic search — we load the
    full set once, dot-product on the Python side, and return top hits.
    The corpus is small per user (thousands of vectors at most) so an in-
    memory scan is simpler than wiring up a vector index.
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT message_id, conversation_id, embedding FROM message_embeddings"
        ).fetchall()
    return [
        (r["message_id"], r["conversation_id"], _unpack_vec(r["embedding"]))
        for r in rows
    ]


def list_unembedded_messages(limit: int = 500) -> list[dict]:
    """Return up to `limit` user/assistant rows that have no embedding yet.

    Used by the one-shot reindex endpoint to backfill embeddings for
    conversations created before semantic recall shipped. Tool rows are
    intentionally excluded — they're too noisy to be useful for recall.
    """
    with _conn() as c:
        rows = c.execute(
            """
            SELECT m.id, m.conversation_id, m.role, m.content
              FROM messages AS m
         LEFT JOIN message_embeddings AS e ON e.message_id = m.id
             WHERE e.message_id IS NULL
               AND m.role IN ('user', 'assistant')
               AND m.content IS NOT NULL
               AND TRIM(m.content) != ''
             LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [
        {
            "id": r["id"],
            "conversation_id": r["conversation_id"],
            "role": r["role"],
            "content": r["content"],
        }
        for r in rows
    ]


def count_embedded_vs_total() -> tuple[int, int]:
    """Return (embedded_count, total_eligible_user_assistant_count).

    The UI uses this to explain why semantic search may be missing hits
    ("12 of 48 messages indexed — run reindex to cover the rest").
    """
    with _conn() as c:
        embedded = c.execute(
            "SELECT COUNT(*) AS n FROM message_embeddings"
        ).fetchone()["n"]
        total = c.execute(
            "SELECT COUNT(*) AS n FROM messages "
            "WHERE role IN ('user', 'assistant') "
            "AND content IS NOT NULL AND TRIM(content) != ''"
        ).fetchone()["n"]
    return int(embedded), int(total)


def list_messages(cid: str) -> list[dict]:
    """Return every message for a conversation, oldest-first."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (cid,),
        ).fetchall()
    return [_row_to_message(r) for r in rows]


def list_pinned_messages(cid: str) -> list[dict]:
    """Return every pinned message in one conversation, oldest-first.

    Used by the ChatHeader → "Pinned messages" dialog so the user can see,
    jump to, or unpin every row they've marked sticky — without scrolling
    through the full transcript looking for pin icons.
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM messages "
            "WHERE conversation_id = ? AND pinned = 1 "
            "ORDER BY created_at ASC",
            (cid,),
        ).fetchall()
    return [_row_to_message(r) for r in rows]


def delete_message(cid: str, mid: str) -> bool:
    """Permanently delete one message by id, scoped to the given conversation.

    Returns True if a row was removed, False otherwise. Scoping by cid
    prevents a malicious client from deleting messages that belong to a
    different conversation by guessing a message id.
    """
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM messages WHERE id = ? AND conversation_id = ?",
            (mid, cid),
        )
        return (cur.rowcount or 0) > 0


def count_assistant_turns(cid: str) -> int:
    """Return the number of assistant messages in a conversation.

    The budget gauge counts a "turn" as one completed assistant reply (not
    user turns, not tool rows). That way, a user who sends 30 rapid-fire
    messages while the agent is churning through a single long turn
    doesn't burn through the budget on bookkeeping alone.
    """
    with _conn() as c:
        row = c.execute(
            "SELECT COUNT(*) AS n FROM messages "
            "WHERE conversation_id = ? AND role = 'assistant'",
            (cid,),
        ).fetchone()
    return int(row["n"] or 0) if row else 0


def conversation_content_chars(cid: str) -> int:
    """Return the total `content` character count across one conversation.

    Used as a cheap proxy for cumulative token usage when checking budgets.
    Multiplying by the frontend's CHARS_PER_TOKEN constant gives a rough
    token count — good enough for budget-enforcement UX, not precise enough
    to pay a provider by (we're local-only so that doesn't matter).
    """
    with _conn() as c:
        row = c.execute(
            "SELECT COALESCE(SUM(LENGTH(content)), 0) AS n FROM messages "
            "WHERE conversation_id = ?",
            (cid,),
        ).fetchone()
    return int(row["n"] or 0) if row else 0


# ---------------------------------------------------------------------------
# Scheduled tasks
#
# These are rows consumed by the polling daemon in app.py. The schedule_task
# tool inserts a row, the daemon reads rows whose next_run_at has passed,
# fires each one as a new conversation, then either deletes the row (one-shot)
# or bumps next_run_at forward by interval_seconds (recurring).
# ---------------------------------------------------------------------------
def create_scheduled_task(
    *,
    name: str,
    prompt: str,
    next_run_at: float,
    interval_seconds: int | None,
    cwd: str,
    target_conversation_id: str | None = None,
    kind: str = "task",
) -> str:
    """Insert a scheduled-task row and return its new id.

    When `target_conversation_id` is set, the daemon will resume that existing
    conversation (append `prompt` as a user message) instead of creating a new
    one. This powers `schedule_wakeup` — the agent scheduling its own follow-up
    in the same chat where it asked to wake up.

    `kind` distinguishes the three row types so the daemon and the UI can
    branch:
      - 'task'   — classic scheduled_task row: fires in a brand-new
                   conversation (one-shot or recurring via interval_seconds).
      - 'wakeup' — one-shot resume-this-chat row (schedule_wakeup).
      - 'loop'   — recurring resume-this-chat row for autonomous loop mode.
                   Keeps firing the same prompt back into the same conv until
                   explicitly stopped or the turn's stop-loop tool call runs.
    Unknown kinds are coerced to 'task' so bad input can't smuggle new
    behavior in through the daemon.
    """
    if kind not in {"task", "wakeup", "loop"}:
        kind = "task"
    tid = str(uuid.uuid4())
    with _conn() as c:
        c.execute(
            "INSERT INTO scheduled_tasks "
            "(id, name, prompt, next_run_at, interval_seconds, cwd, created_at, target_conversation_id, kind) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                tid,
                name,
                prompt,
                next_run_at,
                interval_seconds,
                cwd,
                time.time(),
                target_conversation_id,
                kind,
            ),
        )
    return tid


def list_scheduled_tasks() -> list[dict]:
    """Return every pending scheduled task, soonest-first."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM scheduled_tasks ORDER BY next_run_at ASC"
        ).fetchall()
    return [_row_to_scheduled(r) for r in rows]


def get_due_scheduled_tasks(now: float) -> list[dict]:
    """Return tasks whose `next_run_at` is at or before `now`. Used by the
    daemon to find work to fire on each polling tick.
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM scheduled_tasks WHERE next_run_at <= ? ORDER BY next_run_at ASC",
            (now,),
        ).fetchall()
    return [_row_to_scheduled(r) for r in rows]


def cancel_scheduled_task(id_prefix: str) -> int:
    """Delete scheduled tasks whose id starts with the given prefix.

    Returns the number of rows removed. Supports short-ids (e.g. first 8 hex
    chars) to match the abbreviated form we show in `list_scheduled_tasks`.
    """
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM scheduled_tasks WHERE id = ? OR id LIKE ?",
            (id_prefix, id_prefix + "%"),
        )
        return cur.rowcount or 0


def update_scheduled_task_next_run(id: str, next_run_at: float) -> None:
    """Bump a recurring task's next_run_at after a successful fire."""
    with _conn() as c:
        c.execute(
            "UPDATE scheduled_tasks SET next_run_at = ? WHERE id = ?",
            (next_run_at, id),
        )


def delete_scheduled_task(id: str) -> None:
    """Remove a one-shot task after it has fired."""
    with _conn() as c:
        c.execute("DELETE FROM scheduled_tasks WHERE id = ?", (id,))


def _row_to_scheduled(row: sqlite3.Row) -> dict:
    # target_conversation_id / kind are post-migration columns — older rows
    # predate them. Use dict(row).get(...) so a freshly upgraded DB still
    # hydrates cleanly without a schema-repair step.
    row_dict = dict(row)
    return {
        "id": row["id"],
        "name": row["name"],
        "prompt": row["prompt"],
        "next_run_at": row["next_run_at"],
        "interval_seconds": row["interval_seconds"],
        "cwd": row["cwd"],
        "created_at": row["created_at"],
        "target_conversation_id": row_dict.get("target_conversation_id"),
        "kind": row_dict.get("kind") or "task",
    }


def get_active_loop_for_conversation(cid: str) -> dict | None:
    """Return the single pending loop row for a conversation, or None.

    The UI shows a banner + Stop button whenever a loop is active in the
    current chat, and the `start_loop` tool guards against double-starts by
    calling this first — at most one loop per conversation, keeps the UX
    easy to reason about.
    """
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM scheduled_tasks WHERE target_conversation_id = ? "
            "AND kind = 'loop' ORDER BY next_run_at ASC LIMIT 1",
            (cid,),
        ).fetchone()
    return _row_to_scheduled(row) if row else None


def cancel_loops_for_conversation(cid: str) -> int:
    """Delete every pending loop row bound to this conversation.

    Returns the count removed — the tool/HTTP handler uses it to tell the
    caller whether a loop was actually running. Only loops are deleted;
    classic scheduled_task rows with the same target_conversation_id (if
    anyone ever creates them) are left alone.
    """
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM scheduled_tasks WHERE target_conversation_id = ? "
            "AND kind = 'loop'",
            (cid,),
        )
        return cur.rowcount or 0


# ---------------------------------------------------------------------------
# Doc-chunk index (embeddings for `doc_index` / `doc_search`)
#
# Table schema mirrors message_embeddings: float32 LE bytes via struct.pack.
# Kept separate from message_embeddings because it is file-scoped, not
# conversation-scoped, and has a different lifecycle (explicit re-index).
# ---------------------------------------------------------------------------
def insert_doc_chunk(
    *,
    path: str,
    ordinal: int,
    text: str,
    vector: list[float],
    model: str,
) -> str:
    """Insert one chunk + its embedding. Returns the new row id."""
    cid = str(uuid.uuid4())
    blob = _pack_vec(vector)
    with _conn() as c:
        c.execute(
            "INSERT INTO doc_chunks "
            "(id, path, ordinal, text, vector, model, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (cid, path, ordinal, text, blob, model, time.time()),
        )
    return cid


def delete_doc_chunks_for(path: str) -> int:
    """Drop every chunk associated with a single file path.

    Called before re-indexing a file so the tool is idempotent: running
    `doc_index` twice on the same directory leaves exactly one set of rows
    per file instead of accumulating duplicates.
    """
    with _conn() as c:
        cur = c.execute("DELETE FROM doc_chunks WHERE path = ?", (path,))
        return cur.rowcount or 0


def delete_doc_chunks_for_prefix(prefix: str) -> int:
    """Drop every chunk whose path starts with `prefix`.

    Used by the URL crawler to wipe a seed's old pages before a re-crawl.
    The argument is treated as a literal prefix — any SQL-LIKE wildcards
    inside it are escaped so a path containing `%` can't accidentally
    widen the match.
    """
    if not prefix:
        return 0
    # SQLite LIKE uses `%` and `_` as wildcards — escape both so callers
    # can pass arbitrary strings (URLs included) without surprise.
    esc = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM doc_chunks WHERE path LIKE ? ESCAPE '\\'",
            (esc + "%",),
        )
        return cur.rowcount or 0


def all_doc_chunks(path_substr: str | None = None) -> list[dict]:
    """Return every doc_chunks row with its vector unpacked to a list.

    `path_substr` optionally filters rows whose path contains the given
    substring (case-sensitive — SQLite LIKE uses the default collation).
    The caller (`doc_search`) scores similarity in Python because the full
    index typically fits easily in memory.
    """
    with _conn() as c:
        if path_substr:
            rows = c.execute(
                "SELECT * FROM doc_chunks WHERE path LIKE ?",
                (f"%{path_substr}%",),
            ).fetchall()
        else:
            rows = c.execute("SELECT * FROM doc_chunks").fetchall()
    out: list[dict] = []
    for r in rows:
        out.append({
            "id": r["id"],
            "path": r["path"],
            "ordinal": r["ordinal"],
            "text": r["text"],
            "vector": _unpack_vec(r["vector"]),
            "model": r["model"],
            "created_at": r["created_at"],
        })
    return out


# ---------------------------------------------------------------------------
# Codebase index registry
#
# One row per cwd that's been submitted for background indexing. The actual
# chunks still live in `doc_chunks` — we just track status/progress/last-run
# metadata per root so the UI can render "indexed 400/500 files" and the
# `codebase_search` tool can answer "is this cwd ready yet?" without scanning
# doc_chunks. The row is keyed by the absolute cwd path so re-indexing the
# same directory upserts rather than accumulating rows.
# ---------------------------------------------------------------------------
_CODEBASE_INDEX_STATES = {"pending", "indexing", "ready", "error"}


def upsert_codebase_index(
    cwd: str,
    *,
    status: str,
    file_count: int = 0,
    chunk_count: int = 0,
    last_indexed_at: float | None = None,
    error: str | None = None,
) -> None:
    """Create or update the registry row for `cwd`.

    `status` is validated against the closed set; bad input is coerced to
    'error' to keep the UI's state machine predictable.
    """
    if status not in _CODEBASE_INDEX_STATES:
        status = "error"
    now = time.time()
    with _conn() as c:
        c.execute(
            "INSERT INTO codebase_indexes "
            "(cwd, status, last_indexed_at, file_count, chunk_count, error, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(cwd) DO UPDATE SET "
            "  status = excluded.status, "
            "  last_indexed_at = COALESCE(excluded.last_indexed_at, codebase_indexes.last_indexed_at), "
            "  file_count = excluded.file_count, "
            "  chunk_count = excluded.chunk_count, "
            "  error = excluded.error, "
            "  updated_at = excluded.updated_at",
            (cwd, status, last_indexed_at, file_count, chunk_count, error, now),
        )


def get_codebase_index(cwd: str) -> dict | None:
    """Return the registry row for `cwd`, or None if it was never indexed."""
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM codebase_indexes WHERE cwd = ?", (cwd,),
        ).fetchone()
    if not row:
        return None
    return {
        "cwd": row["cwd"],
        "status": row["status"],
        "last_indexed_at": row["last_indexed_at"],
        "file_count": row["file_count"],
        "chunk_count": row["chunk_count"],
        "error": row["error"],
        "updated_at": row["updated_at"],
    }


def list_codebase_indexes() -> list[dict]:
    """Return every codebase-index row, most-recently-touched first.

    The settings panel uses this to show which roots have been indexed so a
    user can delete stale entries (e.g. a folder they no longer work in).
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM codebase_indexes ORDER BY updated_at DESC",
        ).fetchall()
    return [
        {
            "cwd": r["cwd"],
            "status": r["status"],
            "last_indexed_at": r["last_indexed_at"],
            "file_count": r["file_count"],
            "chunk_count": r["chunk_count"],
            "error": r["error"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]


def delete_codebase_index(cwd: str) -> int:
    """Drop the registry row AND every doc_chunk whose path starts with `cwd`.

    Two-statement delete: the registry row goes first (so a concurrent status
    GET sees "gone" before we burn through chunks), then the chunks. Returns
    the number of registry rows removed (0 or 1) — the chunk count is
    incidental and not propagated to the caller.
    """
    with _conn() as c:
        cur = c.execute("DELETE FROM codebase_indexes WHERE cwd = ?", (cwd,))
        removed = cur.rowcount or 0
        # doc_chunks uses full absolute paths, so anchor the LIKE on cwd + os.sep.
        # Append a path separator so we don't accidentally match a sibling
        # directory that shares a name prefix (e.g. /tmp/foo matching /tmp/foobar).
        prefix = cwd.rstrip("/\\")
        c.execute(
            "DELETE FROM doc_chunks WHERE path LIKE ? OR path LIKE ?",
            (prefix + "/%", prefix + "\\%"),
        )
    return removed


# ---------------------------------------------------------------------------
# Indexed documentation URLs — tiny CRUD for the doc_urls registry.
#
# Mirrors the codebase_indexes shape: each row tracks crawl status for one
# seed URL, while the actual chunks live in doc_chunks under
# `path = "url:<page-url>"` — that convention keeps URL docs filterable
# with one LIKE prefix without duplicating the vector-store schema.
# ---------------------------------------------------------------------------
_DOC_URL_STATES = {"pending", "crawling", "ready", "error"}

# Path-prefix marker for URL-sourced chunks in the doc_chunks table. Kept
# as a module constant so callers don't hardcode `"url:"` in multiple
# places.
DOC_URL_CHUNK_PREFIX = "url:"


def create_doc_url(
    *,
    url: str,
    title: str | None = None,
    max_pages: int = 20,
    same_origin_only: bool = True,
) -> dict:
    """Register a new URL for docs-indexing, or return the existing row.

    Idempotent on `url` (the table's UNIQUE constraint): re-adding the
    same URL returns the existing row without throwing. Use
    `update_doc_url` or `delete_doc_url` + `create_doc_url` to reset
    options on an existing seed.
    """
    now = time.time()
    did = _new_id()
    with _conn() as c:
        try:
            c.execute(
                "INSERT INTO doc_urls "
                "(id, url, title, max_pages, same_origin_only, status, "
                " pages_crawled, chunk_count, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'pending', 0, 0, ?, ?)",
                (
                    did, url, title, int(max_pages),
                    1 if same_origin_only else 0, now, now,
                ),
            )
            row = c.execute(
                "SELECT * FROM doc_urls WHERE id = ?", (did,),
            ).fetchone()
        except sqlite3.IntegrityError:
            row = c.execute(
                "SELECT * FROM doc_urls WHERE url = ?", (url,),
            ).fetchone()
    return _doc_url_row_to_dict(row) if row else {}


def _doc_url_row_to_dict(row) -> dict:
    """Serialize a doc_urls sqlite Row to a plain dict."""
    return {
        "id": row["id"],
        "url": row["url"],
        "title": row["title"],
        "max_pages": row["max_pages"],
        "same_origin_only": bool(row["same_origin_only"]),
        "status": row["status"],
        "pages_crawled": row["pages_crawled"],
        "chunk_count": row["chunk_count"],
        "error": row["error"],
        "last_indexed_at": row["last_indexed_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def list_doc_urls() -> list[dict]:
    """Return every indexed URL, most-recently-touched first."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM doc_urls ORDER BY updated_at DESC",
        ).fetchall()
    return [_doc_url_row_to_dict(r) for r in rows]


def get_doc_url(did: str) -> dict | None:
    """Return one URL row by id, or None."""
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM doc_urls WHERE id = ?", (did,),
        ).fetchone()
    return _doc_url_row_to_dict(row) if row else None


def get_doc_url_by_url(url: str) -> dict | None:
    """Return one URL row by its seed URL, or None."""
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM doc_urls WHERE url = ?", (url,),
        ).fetchone()
    return _doc_url_row_to_dict(row) if row else None


def update_doc_url(
    did: str,
    *,
    status: str | None = None,
    pages_crawled: int | None = None,
    chunk_count: int | None = None,
    error: str | None = None,
    last_indexed_at: float | None = None,
    title: str | None = None,
) -> dict | None:
    """Patch any subset of a doc_urls row in-place.

    Unlike `upsert_codebase_index` we don't rebuild the whole row — only
    the fields the crawler needs to tick over (status → pages_crawled →
    chunk_count → last_indexed_at) are touched. Invalid `status` values
    are coerced to 'error' to keep the UI's state machine predictable.
    """
    fields: list[str] = []
    params: list = []
    if status is not None:
        if status not in _DOC_URL_STATES:
            status = "error"
        fields.append("status = ?")
        params.append(status)
    if pages_crawled is not None:
        fields.append("pages_crawled = ?")
        params.append(int(pages_crawled))
    if chunk_count is not None:
        fields.append("chunk_count = ?")
        params.append(int(chunk_count))
    if error is not None:
        fields.append("error = ?")
        params.append(error)
    if last_indexed_at is not None:
        fields.append("last_indexed_at = ?")
        params.append(float(last_indexed_at))
    if title is not None:
        fields.append("title = ?")
        params.append(title)
    fields.append("updated_at = ?")
    params.append(time.time())
    params.append(did)
    with _conn() as c:
        c.execute(
            f"UPDATE doc_urls SET {', '.join(fields)} WHERE id = ?",
            params,
        )
    return get_doc_url(did)


def delete_doc_url(did: str) -> int:
    """Drop the URL row AND every doc_chunk whose path is a URL under it.

    Chunks are keyed by per-page URL (`path = "url:<page-url>"`) — we
    can't cheaply find "all pages crawled from seed X" without another
    join, so we remove chunks by `url` prefix: every chunk whose path
    starts with `"url:<seed-url>"` is dropped. This accepts a small
    risk of collateral: if the user indexed both
    `https://docs.foo.com/` and `https://docs.foo.com/guide/` the
    latter's chunks would also match the former's prefix. Callers who
    care can delete guides before parents.
    """
    row = get_doc_url(did)
    if not row:
        return 0
    seed = row["url"].rstrip("/")
    with _conn() as c:
        cur = c.execute("DELETE FROM doc_urls WHERE id = ?", (did,))
        removed = cur.rowcount or 0
    # Routed through the shared prefix-delete helper so LIKE wildcards in
    # the seed URL get escaped consistently with the crawler's reset path.
    delete_doc_chunks_for_prefix(f"{DOC_URL_CHUNK_PREFIX}{seed}")
    return removed


# ---------------------------------------------------------------------------
# Lifecycle hooks (user shell commands fired at specific agent-loop events)
#
# Design goals:
#   - Simple CRUD surface: create / list / update / delete / get-for-event.
#   - Events limited to a closed set (`HOOK_EVENTS`) so bad data never slips
#     in. Validation lives here to keep every writer honest.
#   - Never deletes silently: the UI shows disabled hooks greyed-out, so the
#     user always sees what would have fired.
# ---------------------------------------------------------------------------
HOOK_EVENTS = {"pre_tool", "post_tool", "user_prompt_submit", "turn_done"}


def create_hook(
    *,
    event: str,
    command: str,
    matcher: str | None = None,
    timeout_seconds: int = 10,
    enabled: bool = True,
) -> str:
    """Insert a new hook and return its id. Raises ValueError on bad input."""
    if event not in HOOK_EVENTS:
        raise ValueError(f"event must be one of {sorted(HOOK_EVENTS)}, got {event!r}")
    cmd = (command or "").strip()
    if not cmd:
        raise ValueError("command must not be empty")
    # Clamp timeout so a runaway hook can't block the agent forever. 1..120s
    # covers "quick linter" through "short test suite" without being abusable.
    ts = max(1, min(int(timeout_seconds or 10), 120))
    hid = str(uuid.uuid4())
    with _conn() as c:
        c.execute(
            "INSERT INTO hooks (id, event, matcher, command, timeout_seconds, enabled, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (hid, event, (matcher or None), cmd, ts, 1 if enabled else 0, time.time()),
        )
    return hid


def list_hooks() -> list[dict]:
    """Return every hook, newest-first. Disabled rows included."""
    with _conn() as c:
        rows = c.execute("SELECT * FROM hooks ORDER BY created_at DESC").fetchall()
    return [_row_to_hook(r) for r in rows]


def get_hooks_for_event(event: str) -> list[dict]:
    """Return ENABLED hooks registered for this event. Used by the agent loop."""
    if event not in HOOK_EVENTS:
        return []
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM hooks WHERE event = ? AND enabled = 1 ORDER BY created_at ASC",
            (event,),
        ).fetchall()
    return [_row_to_hook(r) for r in rows]


def update_hook(id: str, **fields: Any) -> dict | None:
    """Patch allowed fields on a hook. Returns the refreshed row or None."""
    allowed = {"event", "matcher", "command", "timeout_seconds", "enabled"}
    sets: list[str] = []
    values: list[Any] = []
    for k, v in fields.items():
        if k not in allowed:
            continue
        if k == "event" and v not in HOOK_EVENTS:
            raise ValueError(f"event must be one of {sorted(HOOK_EVENTS)}")
        if k == "enabled":
            v = 1 if v else 0
        if k == "timeout_seconds":
            v = max(1, min(int(v or 10), 120))
        sets.append(f"{k} = ?")
        values.append(v)
    if not sets:
        return get_hook(id)
    values.append(id)
    with _conn() as c:
        c.execute(f"UPDATE hooks SET {', '.join(sets)} WHERE id = ?", values)
    return get_hook(id)


def get_hook(id: str) -> dict | None:
    """Fetch a single hook by id."""
    with _conn() as c:
        row = c.execute("SELECT * FROM hooks WHERE id = ?", (id,)).fetchone()
    return _row_to_hook(row) if row else None


def delete_hook(id: str) -> int:
    """Remove a hook. Returns number of rows deleted (0 or 1)."""
    with _conn() as c:
        cur = c.execute("DELETE FROM hooks WHERE id = ?", (id,))
        return cur.rowcount or 0


def _row_to_hook(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "event": row["event"],
        "matcher": row["matcher"],
        "command": row["command"],
        "timeout_seconds": row["timeout_seconds"],
        "enabled": bool(row["enabled"]),
        "created_at": row["created_at"],
    }


# ---------------------------------------------------------------------------
# Global memories — facts that should be visible to the model in every
# conversation. Backed by SQLite so the Settings UI gets atomic CRUD and the
# agent can mutate the table from any conversation without file races.
#
# Single short cap (8 KB per entry) is enforced on the way in — long entries
# explode the system prompt and are almost always a bug. We don't cap the
# total number of entries because the user is the curator; if they want 200
# memories, that's their call (the prompt-builder will trim if it has to).
# ---------------------------------------------------------------------------
GLOBAL_MEMORY_CONTENT_MAX = 8 * 1024
GLOBAL_MEMORY_TOPIC_MAX = 80


def list_global_memories() -> list[dict]:
    """Return every global memory row, oldest-first.

    Oldest-first ordering matches how they read in the system prompt — the
    most recently added entry sits at the bottom, which is also the position
    the model anchors most strongly on (recency bias). Frontend can re-sort
    for display if the user wants newest-first.
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM global_memories ORDER BY created_at ASC"
        ).fetchall()
    return [_row_to_global_memory(r) for r in rows]


def get_global_memory(mid: str) -> dict | None:
    """Fetch a single global memory by id, or None."""
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM global_memories WHERE id = ?", (mid,)
        ).fetchone()
    return _row_to_global_memory(row) if row else None


def add_global_memory(content: str, topic: str | None = None) -> dict:
    """Insert a new global memory and return the hydrated dict.

    Empty / whitespace-only `content` raises ValueError so callers (API
    handler, agent tool) can surface a clean 400 / error to the user.
    """
    cleaned = (content or "").strip()
    if not cleaned:
        raise ValueError("content is required")
    if len(cleaned) > GLOBAL_MEMORY_CONTENT_MAX:
        cleaned = cleaned[:GLOBAL_MEMORY_CONTENT_MAX]
    t = (topic or "").strip()[:GLOBAL_MEMORY_TOPIC_MAX] or None
    mid = str(uuid.uuid4())
    now = time.time()
    with _conn() as c:
        c.execute(
            "INSERT INTO global_memories (id, content, topic, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (mid, cleaned, t, now, now),
        )
    return get_global_memory(mid)  # type: ignore[return-value]


def update_global_memory(
    mid: str,
    content: str | None = None,
    topic: str | None = None,
) -> dict | None:
    """Patch a global memory's content and/or topic.

    Pass `None` to leave a field unchanged. Returns the refreshed row, or
    None if no row matched. Mirrors update_conversation's "None-means-skip"
    convention so the API handler can pass body.dict() without filtering.
    """
    sets: list[str] = []
    values: list[Any] = []
    if content is not None:
        cleaned = content.strip()
        if not cleaned:
            raise ValueError("content cannot be blank")
        if len(cleaned) > GLOBAL_MEMORY_CONTENT_MAX:
            cleaned = cleaned[:GLOBAL_MEMORY_CONTENT_MAX]
        sets.append("content = ?")
        values.append(cleaned)
    if topic is not None:
        t = topic.strip()[:GLOBAL_MEMORY_TOPIC_MAX] or None
        sets.append("topic = ?")
        values.append(t)
    if not sets:
        return get_global_memory(mid)
    sets.append("updated_at = ?")
    values.append(time.time())
    values.append(mid)
    with _conn() as c:
        cur = c.execute(
            f"UPDATE global_memories SET {', '.join(sets)} WHERE id = ?",
            values,
        )
        if cur.rowcount == 0:
            return None
    return get_global_memory(mid)


def delete_global_memory(mid: str) -> int:
    """Remove one global memory. Returns rows deleted (0 or 1)."""
    with _conn() as c:
        cur = c.execute("DELETE FROM global_memories WHERE id = ?", (mid,))
        return cur.rowcount or 0


def delete_global_memories_matching(pattern: str) -> int:
    """Delete every global memory whose content contains `pattern` (case-
    insensitive substring). Used by the `forget(scope='global')` agent tool.

    Returns the number of rows deleted. An empty / whitespace pattern is
    refused (ValueError) so a typo can't accidentally wipe the table.

    SQL `LIKE` treats `%` and `_` as wildcards by default. The user (or the
    agent calling `forget(scope="global", pattern=...)`) thinks of the
    pattern as a literal substring, so we escape those metachars and use
    `ESCAPE` to match the per-conv `forget` semantics. Without this, a
    pattern like `100%` would match every row instead of just rows
    containing the literal text "100%".
    """
    needle = (pattern or "").strip()
    if not needle:
        raise ValueError("pattern is required")
    # Escape SQL LIKE metachars so the pattern stays literal. We pick `\` as
    # the escape char and announce it via ESCAPE.
    escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    like = f"%{escaped}%"
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM global_memories WHERE LOWER(content) LIKE LOWER(?) ESCAPE '\\'",
            (like,),
        )
        return cur.rowcount or 0


def _row_to_global_memory(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "content": row["content"],
        "topic": row["topic"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


# ---------------------------------------------------------------------------
# Web Push subscriptions.
#
# The UI calls `pushManager.subscribe()` in the service worker, POSTs the
# resulting {endpoint, keys} payload to /api/push/subscribe, and we persist
# it here. Outgoing pushes (see backend/push.py) walk every row and fire an
# encrypted payload at each endpoint; stale endpoints (410 Gone) are pruned
# at send-time via `delete_push_subscription`.
# ---------------------------------------------------------------------------
def upsert_push_subscription(
    endpoint: str, p256dh: str, auth: str, user_agent: str | None = None
) -> None:
    """Insert or update a browser's push subscription.

    Keyed on `endpoint` so re-subscribing the same browser refreshes the
    keys without leaking duplicates (Chrome rotates keys periodically).
    """
    with _conn() as c:
        c.execute(
            """
            INSERT INTO push_subscriptions (endpoint, p256dh, auth, user_agent, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(endpoint) DO UPDATE SET
                p256dh=excluded.p256dh,
                auth=excluded.auth,
                user_agent=excluded.user_agent
            """,
            (endpoint, p256dh, auth, user_agent, time.time()),
        )


def list_push_subscriptions() -> list[dict]:
    """Return every stored push subscription, one dict per browser."""
    with _conn() as c:
        rows = c.execute("SELECT * FROM push_subscriptions").fetchall()
    return [
        {
            "endpoint": r["endpoint"],
            "p256dh": r["p256dh"],
            "auth": r["auth"],
            "user_agent": r["user_agent"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def delete_push_subscription(endpoint: str) -> int:
    """Remove a subscription by its endpoint. Returns rows deleted (0 or 1)."""
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM push_subscriptions WHERE endpoint = ?", (endpoint,)
        )
        return cur.rowcount or 0


def count_push_subscriptions() -> int:
    """Count of registered browsers — surfaced in the settings UI."""
    with _conn() as c:
        row = c.execute("SELECT COUNT(*) AS n FROM push_subscriptions").fetchone()
    return int(row["n"]) if row else 0


# ---------------------------------------------------------------------------
# MCP servers.
#
# Rows here are configuration only — the live subprocess state is held in
# backend/mcp.py's in-memory session map. On server restart we re-spawn any
# row where `enabled=1`, fetch its tool list, and merge into TOOL_SCHEMAS so
# the agent sees those tools without a manual refresh.
# ---------------------------------------------------------------------------
def _row_to_mcp_server(row: sqlite3.Row) -> dict:
    try:
        args = json.loads(row["args_json"] or "[]")
        if not isinstance(args, list):
            args = []
    except Exception:
        args = []
    try:
        env = json.loads(row["env_json"] or "{}")
        if not isinstance(env, dict):
            env = {}
    except Exception:
        env = {}
    return {
        "id": row["id"],
        "name": row["name"],
        "command": row["command"],
        "args": [str(a) for a in args],
        "env": {str(k): str(v) for k, v in env.items()},
        "enabled": bool(row["enabled"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def create_mcp_server(
    name: str,
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    enabled: bool = True,
) -> dict:
    """Insert a new MCP server row. `name` must be unique."""
    now = time.time()
    sid = uuid.uuid4().hex
    with _conn() as c:
        c.execute(
            """
            INSERT INTO mcp_servers (id, name, command, args_json, env_json, enabled, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sid,
                name,
                command,
                json.dumps(list(args or [])),
                json.dumps(dict(env or {})),
                1 if enabled else 0,
                now,
                now,
            ),
        )
        row = c.execute("SELECT * FROM mcp_servers WHERE id = ?", (sid,)).fetchone()
    return _row_to_mcp_server(row)


def list_mcp_servers() -> list[dict]:
    """Return every configured MCP server, newest first."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM mcp_servers ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_mcp_server(r) for r in rows]


def get_mcp_server(sid: str) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM mcp_servers WHERE id = ?", (sid,)).fetchone()
    return _row_to_mcp_server(row) if row else None


def update_mcp_server(sid: str, patch: dict) -> dict | None:
    """Apply a partial update (name/command/args/env/enabled). Returns the
    refreshed row, or None if the id does not exist."""
    allowed = {"name", "command", "args", "env", "enabled"}
    fields: list[str] = []
    values: list[Any] = []
    for k, v in patch.items():
        if k not in allowed:
            continue
        if k == "args":
            fields.append("args_json = ?")
            values.append(json.dumps(list(v or [])))
        elif k == "env":
            fields.append("env_json = ?")
            values.append(json.dumps(dict(v or {})))
        elif k == "enabled":
            fields.append("enabled = ?")
            values.append(1 if v else 0)
        else:
            fields.append(f"{k} = ?")
            values.append(v)
    if not fields:
        return get_mcp_server(sid)
    fields.append("updated_at = ?")
    values.append(time.time())
    values.append(sid)
    with _conn() as c:
        c.execute(
            f"UPDATE mcp_servers SET {', '.join(fields)} WHERE id = ?", values
        )
        row = c.execute("SELECT * FROM mcp_servers WHERE id = ?", (sid,)).fetchone()
    return _row_to_mcp_server(row) if row else None


def delete_mcp_server(sid: str) -> int:
    with _conn() as c:
        cur = c.execute("DELETE FROM mcp_servers WHERE id = ?", (sid,))
        return cur.rowcount or 0


# ---------------------------------------------------------------------------
# User settings (key/value store)
#
# Small JSON-encoded values keyed by string. Used by /api/settings so the UI
# can persist user preferences (default chat model, future feature toggles…)
# across restarts without a dedicated table per setting.
# ---------------------------------------------------------------------------
def get_setting(key: str, default: Any = None) -> Any:
    """Return the stored value for `key`, or `default` if the key is unset.

    Values are JSON-decoded; a malformed row collapses to `default` rather
    than raising so a corrupt entry can't crash the settings route.
    """
    with _conn() as c:
        row = c.execute(
            "SELECT value FROM user_settings WHERE key = ?", (key,)
        ).fetchone()
    if not row:
        return default
    try:
        return json.loads(row["value"])
    except (json.JSONDecodeError, TypeError):
        return default


def set_setting(key: str, value: Any) -> None:
    """Insert-or-replace one setting. `value` must be JSON-serialisable."""
    payload = json.dumps(value)
    with _conn() as c:
        c.execute(
            """
            INSERT INTO user_settings (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (key, payload, time.time()),
        )


def delete_setting(key: str) -> int:
    """Remove one setting row. Returns rows deleted (0 or 1)."""
    with _conn() as c:
        cur = c.execute("DELETE FROM user_settings WHERE key = ?", (key,))
        return cur.rowcount or 0


def get_all_settings() -> dict[str, Any]:
    """Return every stored setting as a {key: value} dict (JSON-decoded)."""
    with _conn() as c:
        rows = c.execute("SELECT key, value FROM user_settings").fetchall()
    out: dict[str, Any] = {}
    for r in rows:
        try:
            out[r["key"]] = json.loads(r["value"])
        except (json.JSONDecodeError, TypeError):
            continue
    return out


def _row_to_conversation(row: sqlite3.Row) -> dict:
    # `pinned` and `tags` are post-migration columns — old rows may not have
    # them. Guard with try/except so a freshly upgraded DB still hydrates
    # cleanly until those rows get a write that fills the defaults.
    try:
        pinned = bool(row["pinned"])
    except (IndexError, KeyError):
        pinned = False
    try:
        raw_tags = row["tags"]
    except (IndexError, KeyError):
        raw_tags = None
    try:
        tags = json.loads(raw_tags) if raw_tags else []
    except (json.JSONDecodeError, TypeError):
        tags = []
    try:
        state = row["state"] or "idle"
    except (IndexError, KeyError):
        state = "idle"
    # Persona / budgets are late-migration columns — older rows may predate
    # them. Guard with the same try/except pattern used for pinned/tags so a
    # freshly upgraded DB still hydrates without a schema-repair step.
    try:
        persona = row["persona"] or None
    except (IndexError, KeyError):
        persona = None
    try:
        budget_turns = row["budget_turns"]
    except (IndexError, KeyError):
        budget_turns = None
    try:
        budget_tokens = row["budget_tokens"]
    except (IndexError, KeyError):
        budget_tokens = None
    try:
        project = row["project"] or None
    except (IndexError, KeyError):
        project = None
    # Permission mode supersedes the old auto_approve bit, but we keep the
    # legacy field in the response so any code still reading conv["auto_approve"]
    # keeps working during the transition. Source of truth is permission_mode.
    try:
        permission_mode = row["permission_mode"] or "approve_edits"
    except (IndexError, KeyError):
        permission_mode = "allow_all" if bool(row["auto_approve"]) else "approve_edits"
    if permission_mode not in {"read_only", "plan", "approve_edits", "allow_all"}:
        permission_mode = "approve_edits"
    return {
        "id": row["id"],
        "title": row["title"],
        "model": row["model"],
        "auto_approve": permission_mode == "allow_all",
        "permission_mode": permission_mode,
        "cwd": row["cwd"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "pinned": pinned,
        "tags": tags,
        "state": state,
        "persona": persona,
        "budget_turns": budget_turns,
        "budget_tokens": budget_tokens,
        "project": project,
    }


def _row_to_message(row: sqlite3.Row) -> dict:
    # sqlite3.Row supports column lookup but not `.get` — guard the images
    # and pinned columns with try/except so rows from pre-migration schemas
    # still hydrate cleanly on upgraded databases.
    try:
        raw_images = row["images"]
    except (IndexError, KeyError):
        raw_images = None
    try:
        pinned = bool(row["pinned"])
    except (IndexError, KeyError):
        pinned = False
    return {
        "id": row["id"],
        "conversation_id": row["conversation_id"],
        "role": row["role"],
        "content": row["content"],
        "tool_calls": json.loads(row["tool_calls"]) if row["tool_calls"] else [],
        "images": json.loads(raw_images) if raw_images else [],
        "pinned": pinned,
        "created_at": row["created_at"],
    }


# ---------------------------------------------------------------------------
# Secrets — credentials referenced from http_request via {{secret:NAME}}
# ---------------------------------------------------------------------------
SECRET_NAME_MAX = 64
SECRET_VALUE_MAX = 16_000  # enough for long JWTs, service-account JSON blobs
SECRET_DESC_MAX = 400

# Valid secret names are alphanumerics + underscore, starting with a letter /
# underscore. The same pattern is used to match `{{secret:...}}` placeholders
# in http_request, so making it restrictive here prevents accidental glob
# matches on values that happen to contain double-braces.
_SECRET_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def _row_to_secret(row: sqlite3.Row, include_value: bool = False) -> dict:
    out = {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"] or "",
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    if include_value:
        out["value"] = row["value"]
    return out


def list_secrets() -> list[dict]:
    """Return every secret's metadata (name + description + timestamps only).

    The raw value is intentionally omitted — callers who need to resolve a
    placeholder go through ``get_secret_value()``, which is a deliberate,
    narrow read path. The settings UI lists rows via this function and then
    calls ``get_secret(id)`` once the user clicks "reveal".
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM secrets ORDER BY name ASC"
        ).fetchall()
    return [_row_to_secret(r) for r in rows]


def get_secret(sid: str, include_value: bool = False) -> dict | None:
    """Fetch a single secret by id. ``include_value=True`` exposes the value."""
    with _conn() as c:
        row = c.execute("SELECT * FROM secrets WHERE id = ?", (sid,)).fetchone()
    return _row_to_secret(row, include_value=include_value) if row else None


def get_secret_value(name: str) -> str | None:
    """Resolve a secret by NAME (for ``{{secret:NAME}}`` substitution).

    Returns the raw value or None if the name isn't registered. Called from
    ``tools.http_request`` on every placeholder — kept separate from the
    metadata path so the placeholder resolver never accidentally hands back
    someone else's secret on a typo'd name.
    """
    if not name:
        return None
    with _conn() as c:
        row = c.execute(
            "SELECT value FROM secrets WHERE name = ?", (name,)
        ).fetchone()
    return row["value"] if row else None


def create_secret(name: str, value: str, description: str | None = None) -> dict:
    """Insert a new secret. Raises ValueError on bad input / dup name."""
    n = (name or "").strip()
    if not _SECRET_NAME_RE.match(n):
        raise ValueError(
            "name must be 1-64 chars, start with a letter/underscore, and "
            "contain only letters, digits, and underscores"
        )
    v = value or ""
    if not v:
        raise ValueError("value is required")
    if len(v) > SECRET_VALUE_MAX:
        raise ValueError(f"value must be at most {SECRET_VALUE_MAX} characters")
    d = (description or "").strip()[:SECRET_DESC_MAX] or None
    sid = str(uuid.uuid4())
    now = time.time()
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO secrets (id, name, value, description, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (sid, n, v, d, now, now),
            )
    except sqlite3.IntegrityError as e:
        # Surface the "name already exists" case with a clean message rather
        # than leaking the UNIQUE-constraint jargon into the UI.
        raise ValueError(f"a secret named {n!r} already exists") from e
    return get_secret(sid)  # type: ignore[return-value]


def update_secret(
    sid: str,
    name: str | None = None,
    value: str | None = None,
    description: str | None = None,
) -> dict | None:
    """Patch a secret. None-means-skip like ``update_global_memory``."""
    sets: list[str] = []
    values: list[Any] = []
    if name is not None:
        n = name.strip()
        if not _SECRET_NAME_RE.match(n):
            raise ValueError("invalid secret name")
        sets.append("name = ?")
        values.append(n)
    if value is not None:
        if not value:
            raise ValueError("value cannot be blank")
        if len(value) > SECRET_VALUE_MAX:
            raise ValueError(f"value must be at most {SECRET_VALUE_MAX} characters")
        sets.append("value = ?")
        values.append(value)
    if description is not None:
        d = description.strip()[:SECRET_DESC_MAX] or None
        sets.append("description = ?")
        values.append(d)
    if not sets:
        return get_secret(sid)
    sets.append("updated_at = ?")
    values.append(time.time())
    values.append(sid)
    try:
        with _conn() as c:
            cur = c.execute(
                f"UPDATE secrets SET {', '.join(sets)} WHERE id = ?", values
            )
            if cur.rowcount == 0:
                return None
    except sqlite3.IntegrityError as e:
        raise ValueError("another secret already uses that name") from e
    return get_secret(sid)


def delete_secret(sid: str) -> int:
    """Remove one secret. Returns rows deleted (0 or 1)."""
    with _conn() as c:
        cur = c.execute("DELETE FROM secrets WHERE id = ?", (sid,))
        return cur.rowcount or 0


# ---------------------------------------------------------------------------
# User-defined tools.
#
# The Settings → Tools UI (POST /api/user-tools) is the only way to write rows
# into `user_tools`; the LLM has no equivalent tool-call route. Each row
# becomes a first-class entry in the Ollama tool palette — the model discovers
# it the same way it discovers MCP tools, no retraining or redeploy. Code runs
# in a subprocess against a dedicated venv (`data/tools_venv/`, see
# `user_tools_runtime.py`), so a misbehaving user tool can't mutate the
# backend's globals or inject imports into other subsystems.
#
# Name validation mirrors the Ollama function-calling constraint: lowercase
# alpha + digits + underscore, starts with a letter, ≤48 chars so the combined
# namespace isn't at risk of collision with future built-ins. We also refuse
# names that already exist as built-in tools — the dispatcher would otherwise
# route to whichever one it found first, a confusing failure mode.
# ---------------------------------------------------------------------------
USER_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
USER_TOOL_CATEGORIES = {"read", "write"}
USER_TOOL_DESC_MAX = 2000
USER_TOOL_CODE_MAX = 64 * 1024
USER_TOOL_TIMEOUT_MIN = 1
USER_TOOL_TIMEOUT_MAX = 600


def _row_to_user_tool(row: sqlite3.Row) -> dict:
    """Unpack a user_tools row, deserialising JSON columns."""
    try:
        schema = json.loads(row["schema_json"]) if row["schema_json"] else {}
    except Exception:
        schema = {}
    try:
        deps = json.loads(row["deps_json"]) if row["deps_json"] else []
    except Exception:
        deps = []
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "schema": schema if isinstance(schema, dict) else {},
        "code": row["code"],
        "deps": deps if isinstance(deps, list) else [],
        "category": row["category"],
        "timeout_seconds": row["timeout_seconds"],
        "enabled": bool(row["enabled"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def create_user_tool(
    *,
    name: str,
    description: str,
    code: str,
    schema: dict | None = None,
    deps: list[str] | None = None,
    category: str = "write",
    timeout_seconds: int = 60,
    enabled: bool = True,
) -> dict:
    """Insert a user-defined tool. Validates input; raises ValueError on bad shape.

    Schema validation here is structural only — the caller (tools.create_tool)
    runs the deeper checks (AST parse, `def run(args)` presence, dep-spec
    regex, built-in-name collision). This function focuses on the DB-layer
    invariants: name pattern, length caps, enum membership.
    """
    n = (name or "").strip().lower()
    if not USER_TOOL_NAME_RE.match(n):
        raise ValueError(
            "tool name must match ^[a-z][a-z0-9_]{0,47}$ "
            "(lowercase alpha/digits/underscore, start with a letter)"
        )
    desc = (description or "").strip()
    if not desc:
        raise ValueError("description must not be empty")
    if len(desc) > USER_TOOL_DESC_MAX:
        raise ValueError(f"description must be at most {USER_TOOL_DESC_MAX} characters")
    body = code or ""
    if not body.strip():
        raise ValueError("code must not be empty")
    if len(body) > USER_TOOL_CODE_MAX:
        raise ValueError(f"code must be at most {USER_TOOL_CODE_MAX} characters")
    cat = (category or "write").strip().lower()
    if cat not in USER_TOOL_CATEGORIES:
        raise ValueError(f"category must be one of {sorted(USER_TOOL_CATEGORIES)}")
    t = max(USER_TOOL_TIMEOUT_MIN, min(int(timeout_seconds or 60), USER_TOOL_TIMEOUT_MAX))
    schema_obj = schema if isinstance(schema, dict) else {}
    deps_list = [str(d).strip() for d in (deps or []) if str(d).strip()]
    tid = str(uuid.uuid4())
    now = time.time()
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO user_tools "
                "(id, name, description, schema_json, code, deps_json, "
                " category, timeout_seconds, enabled, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    tid,
                    n,
                    desc,
                    json.dumps(schema_obj, ensure_ascii=False),
                    body,
                    json.dumps(deps_list, ensure_ascii=False),
                    cat,
                    t,
                    1 if enabled else 0,
                    now,
                    now,
                ),
            )
    except sqlite3.IntegrityError as e:
        raise ValueError(f"a tool named {n!r} already exists") from e
    return get_user_tool(tid) or {}


def list_user_tools(enabled_only: bool = False) -> list[dict]:
    """Every user tool, newest-first. Pass enabled_only=True for palette use."""
    with _conn() as c:
        if enabled_only:
            rows = c.execute(
                "SELECT * FROM user_tools WHERE enabled = 1 ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM user_tools ORDER BY created_at DESC"
            ).fetchall()
    return [_row_to_user_tool(r) for r in rows]


def get_user_tool(tid: str) -> dict | None:
    """Fetch one user tool by id."""
    with _conn() as c:
        row = c.execute("SELECT * FROM user_tools WHERE id = ?", (tid,)).fetchone()
    return _row_to_user_tool(row) if row else None


def get_user_tool_by_name(name: str) -> dict | None:
    """Fetch one user tool by name — the hot-path used by the dispatcher."""
    n = (name or "").strip().lower()
    if not n:
        return None
    with _conn() as c:
        row = c.execute("SELECT * FROM user_tools WHERE name = ?", (n,)).fetchone()
    return _row_to_user_tool(row) if row else None


def update_user_tool(tid: str, **fields: Any) -> dict | None:
    """Patch allowed fields on a user tool.

    Accepts: description, code, schema (dict), deps (list[str]), category,
    timeout_seconds, enabled. Unknown keys are silently dropped so callers can
    pass the full edit body without trimming. Name is intentionally NOT
    patchable — renaming would leave the model confused about a tool that
    disappeared mid-conversation and any stored references would break.
    """
    sets: list[str] = []
    values: list[Any] = []
    if "description" in fields and fields["description"] is not None:
        d = str(fields["description"]).strip()
        if not d:
            raise ValueError("description must not be empty")
        if len(d) > USER_TOOL_DESC_MAX:
            raise ValueError(f"description must be at most {USER_TOOL_DESC_MAX} characters")
        sets.append("description = ?")
        values.append(d)
    if "code" in fields and fields["code"] is not None:
        body = str(fields["code"])
        if not body.strip():
            raise ValueError("code must not be empty")
        if len(body) > USER_TOOL_CODE_MAX:
            raise ValueError(f"code must be at most {USER_TOOL_CODE_MAX} characters")
        sets.append("code = ?")
        values.append(body)
    if "schema" in fields and fields["schema"] is not None:
        schema_obj = fields["schema"] if isinstance(fields["schema"], dict) else {}
        sets.append("schema_json = ?")
        values.append(json.dumps(schema_obj, ensure_ascii=False))
    if "deps" in fields and fields["deps"] is not None:
        deps_list = [str(d).strip() for d in fields["deps"] if str(d).strip()]
        sets.append("deps_json = ?")
        values.append(json.dumps(deps_list, ensure_ascii=False))
    if "category" in fields and fields["category"] is not None:
        cat = str(fields["category"]).strip().lower()
        if cat not in USER_TOOL_CATEGORIES:
            raise ValueError(f"category must be one of {sorted(USER_TOOL_CATEGORIES)}")
        sets.append("category = ?")
        values.append(cat)
    if "timeout_seconds" in fields and fields["timeout_seconds"] is not None:
        t = max(USER_TOOL_TIMEOUT_MIN, min(int(fields["timeout_seconds"]), USER_TOOL_TIMEOUT_MAX))
        sets.append("timeout_seconds = ?")
        values.append(t)
    if "enabled" in fields and fields["enabled"] is not None:
        sets.append("enabled = ?")
        values.append(1 if fields["enabled"] else 0)
    if not sets:
        return get_user_tool(tid)
    sets.append("updated_at = ?")
    values.append(time.time())
    values.append(tid)
    with _conn() as c:
        cur = c.execute(
            f"UPDATE user_tools SET {', '.join(sets)} WHERE id = ?", values
        )
        if cur.rowcount == 0:
            return None
    return get_user_tool(tid)


def delete_user_tool(tid: str) -> int:
    """Remove one user tool by id. Returns rows deleted (0 or 1)."""
    with _conn() as c:
        cur = c.execute("DELETE FROM user_tools WHERE id = ?", (tid,))
        return cur.rowcount or 0


def delete_user_tool_by_name(name: str) -> int:
    """Remove one user tool by name. Returns rows deleted (0 or 1)."""
    n = (name or "").strip().lower()
    if not n:
        return 0
    with _conn() as c:
        cur = c.execute("DELETE FROM user_tools WHERE name = ?", (n,))
        return cur.rowcount or 0


# ---------------------------------------------------------------------------
# Side tasks (spawn_task / chip UI)
#
# Rows represent drive-by observations the agent made mid-turn. They live in
# the DB so the chip stays visible across reloads until the user opens or
# dismisses it. `status` transitions: pending → opened|dismissed (terminal).
# ---------------------------------------------------------------------------
SIDE_TASK_TITLE_MAX = 120
SIDE_TASK_TLDR_MAX = 400
SIDE_TASK_PROMPT_MAX = 8000


def create_side_task(
    *,
    source_conversation_id: str,
    title: str,
    prompt: str,
    tldr: str | None,
) -> dict:
    """Insert a side-task row and return the hydrated dict.

    All string fields are length-clamped at the DB layer so a misbehaving
    tool call can't stash an unbounded blob.
    """
    sid = str(uuid.uuid4())
    now = time.time()
    t = (title or "").strip()[:SIDE_TASK_TITLE_MAX]
    p = (prompt or "").strip()[:SIDE_TASK_PROMPT_MAX]
    l = (tldr or "").strip()[:SIDE_TASK_TLDR_MAX] if tldr else None
    with _conn() as c:
        c.execute(
            "INSERT INTO side_tasks "
            "(id, source_conversation_id, title, prompt, tldr, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
            (sid, source_conversation_id, t, p, l, now),
        )
    return get_side_task(sid) or {}


def get_side_task(sid: str) -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM side_tasks WHERE id = ?", (sid,)
        ).fetchone()
    return _row_to_side_task(row) if row else None


def list_side_tasks_for_conversation(conversation_id: str) -> list[dict]:
    """Return all side tasks (any status) that originated from a given chat."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM side_tasks WHERE source_conversation_id = ? "
            "ORDER BY created_at DESC",
            (conversation_id,),
        ).fetchall()
    return [_row_to_side_task(r) for r in rows]


def list_pending_side_tasks(conversation_id: str) -> list[dict]:
    """Return only pending side tasks — for rendering live chips in the UI."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM side_tasks WHERE source_conversation_id = ? "
            "AND status = 'pending' ORDER BY created_at ASC",
            (conversation_id,),
        ).fetchall()
    return [_row_to_side_task(r) for r in rows]


def mark_side_task_opened(sid: str, opened_conversation_id: str) -> dict | None:
    with _conn() as c:
        cur = c.execute(
            "UPDATE side_tasks SET status = 'opened', opened_at = ?, "
            "opened_conversation_id = ? WHERE id = ? AND status = 'pending'",
            (time.time(), opened_conversation_id, sid),
        )
        if cur.rowcount == 0:
            return None
    return get_side_task(sid)


def mark_side_task_dismissed(sid: str) -> dict | None:
    with _conn() as c:
        cur = c.execute(
            "UPDATE side_tasks SET status = 'dismissed' "
            "WHERE id = ? AND status = 'pending'",
            (sid,),
        )
        if cur.rowcount == 0:
            return None
    return get_side_task(sid)


def _row_to_side_task(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "source_conversation_id": row["source_conversation_id"],
        "title": row["title"],
        "prompt": row["prompt"],
        "tldr": row["tldr"],
        "status": row["status"],
        "created_at": row["created_at"],
        "opened_at": row["opened_at"],
        "opened_conversation_id": row["opened_conversation_id"],
    }


# ---------------------------------------------------------------------------
# Worktrees (create_worktree / list_worktrees / remove_worktree)
#
# Thin per-conversation index of git worktrees the agent spun up. The git
# binary itself is the source of truth for worktree state; we just track
# what we created so the UI can surface an "active worktree" badge and
# cleanup is scoped by conversation.
# ---------------------------------------------------------------------------
def create_worktree_row(
    *,
    conversation_id: str,
    repo_path: str,
    path: str,
    branch: str,
    base_ref: str,
) -> dict:
    wid = str(uuid.uuid4())
    with _conn() as c:
        c.execute(
            "INSERT INTO worktrees "
            "(id, conversation_id, repo_path, path, branch, base_ref, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'active', ?)",
            (wid, conversation_id, repo_path, path, branch, base_ref, time.time()),
        )
    return get_worktree(wid) or {}


def get_worktree(wid: str) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM worktrees WHERE id = ?", (wid,)).fetchone()
    return _row_to_worktree(row) if row else None


def list_worktrees_for_conversation(conversation_id: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM worktrees WHERE conversation_id = ? ORDER BY created_at DESC",
            (conversation_id,),
        ).fetchall()
    return [_row_to_worktree(r) for r in rows]


def list_active_worktrees_for_conversation(conversation_id: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM worktrees WHERE conversation_id = ? AND status = 'active' "
            "ORDER BY created_at DESC",
            (conversation_id,),
        ).fetchall()
    return [_row_to_worktree(r) for r in rows]


def mark_worktree_removed(wid: str) -> dict | None:
    with _conn() as c:
        cur = c.execute(
            "UPDATE worktrees SET status = 'removed', removed_at = ? "
            "WHERE id = ? AND status = 'active'",
            (time.time(), wid),
        )
        if cur.rowcount == 0:
            return None
    return get_worktree(wid)


def _row_to_worktree(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "conversation_id": row["conversation_id"],
        "repo_path": row["repo_path"],
        "path": row["path"],
        "branch": row["branch"],
        "base_ref": row["base_ref"],
        "status": row["status"],
        "created_at": row["created_at"],
        "removed_at": row["removed_at"],
    }
