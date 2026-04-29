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

import os
import sqlite3
import json
import re
import struct
import time
import uuid
from pathlib import Path
from typing import Any

# orjson is a Rust-based JSON parser ~3-5x faster than stdlib json
# for the row-hydration hot path (every message LIST query parses
# `tool_calls` + `images` columns through this). Falls back to
# stdlib json when orjson isn't installed; behaviour identical.
try:
    import orjson as _orjson  # type: ignore

    def _json_loads(data: str) -> Any:
        # orjson.loads accepts both str and bytes; we get str from
        # sqlite3 by default. The orjson decoder is strict about
        # trailing whitespace where stdlib is lenient — every JSON
        # value we store is dumped via json.dumps (or orjson) without
        # trailing whitespace, so this matches in practice.
        return _orjson.loads(data)
except ImportError:
    def _json_loads(data: str) -> Any:
        return json.loads(data)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"


def _conn() -> sqlite3.Connection:
    """Open a connection with WAL mode + foreign-key enforcement.

    `check_same_thread=False` is safe here because FastAPI sync routes run on
    a threadpool and we always wrap writes in a context manager — sqlite3
    holds a write lock per-statement under WAL so there's no data-race.

    Performance pragmas:
      * `journal_mode = WAL` — concurrent reads while a writer is active.
      * `synchronous = NORMAL` — fsync at each WAL checkpoint instead of
        every transaction. Recommended SQLite default for WAL mode;
        durability still survives an OS crash, only loses uncommitted
        transactions on a hard power-cut. ~2-5x faster writes than the
        FULL default with negligible risk for an interactive chat app.
      * `temp_store = MEMORY` — sort/group temporaries in RAM rather
        than spilling to disk; helps the doc_chunks ORDER BY paths.
      * `cache_size = -32000` — 32 MiB page cache per connection
        (negative = KiB). Default is 2 MiB; a 32 MiB cache fits the
        full doc_chunks index for a typical mid-size codebase, so
        repeat searches hit memory not disk.
      * `mmap_size = 268435456` — let SQLite mmap up to 256 MiB of the
        DB file, eliminating most read syscalls. No-op when SQLite
        was compiled without mmap support; harmless either way.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode = WAL")
    c.execute("PRAGMA synchronous = NORMAL")
    c.execute("PRAGMA temp_store = MEMORY")
    c.execute("PRAGMA cache_size = -32000")
    c.execute("PRAGMA mmap_size = 268435456")
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
                created_at REAL NOT NULL,
                error_threshold INTEGER,
                max_fires_per_conv INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_hooks_event ON hooks(event, enabled);
            -- Per-conversation fire counter — used to enforce
            -- `hooks.max_fires_per_conv` so a buggy hook can't infinite-
            -- loop. Persisted (vs. in-memory) so a backend restart
            -- doesn't reset the counter and re-open the loop.
            CREATE TABLE IF NOT EXISTS hook_fires (
                hook_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                fire_count INTEGER NOT NULL DEFAULT 0,
                last_fired_at REAL NOT NULL,
                PRIMARY KEY (hook_id, conversation_id)
            );

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

            -- Project memories: facts shared across every conversation
            -- working in the same directory. Keyed by the conversation's
            -- `cwd` (its absolute working-directory path), so any two
            -- chats pointed at the same repo automatically share the
            -- same memory set without the user having to assign a
            -- project label. Sits between global (user-wide) and
            -- conversation (per-chat) on the specificity axis.
            -- Examples: "this codebase uses pytest, not unittest", "the
            -- linter config is .eslintrc.cjs at repo root", "API tokens
            -- for staging are in 1Password entry X".
            CREATE TABLE IF NOT EXISTS project_memories (
                id TEXT PRIMARY KEY,
                cwd TEXT NOT NULL,
                content TEXT NOT NULL,
                topic TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            -- Note: the cwd index lives in the migration block below,
            -- not here. On a DB that booted the legacy `project`-column
            -- code first, the column rename hasn't run yet at this
            -- point, so creating an index on `cwd` would fail with
            -- "no such column". The migration handles both fresh and
            -- legacy DBs uniformly.

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
            -- Compute pool: other PCs the user has registered as workers.
            -- The host can route embedding calls, subagent runs, and chat
            -- turns to these workers to parallelise across machines. Each
            -- worker exposes an Ollama endpoint on a LAN address — all
            -- ongoing traffic stays on the local Wi-Fi/Ethernet so it
            -- never burns metered internet bandwidth.
            -- `auth_token` is a shared secret the worker validates; without
            -- it any LAN peer could drain the GPU.
            -- `ssh_host` is an optional SSH alias used for LAN-side model
            -- copy (scp). Same network as `address`.
            -- `tailscale_host` is an optional, stable Tailscale identifier
            -- (MagicDNS name or CGNAT IPv4) used ONLY by the auto-repair
            -- routine: when DHCP hands the worker a new LAN IP and the
            -- stored `address` goes stale, the host reaches the worker
            -- over Tailscale just long enough to ask for the new LAN IP,
            -- updates `address`, and resumes ordinary LAN traffic. Never
            -- used for chat / embeddings / model copy.
            -- `capabilities_json` is a cached snapshot of what the worker
            -- reports (installed Ollama models, RAM/VRAM/GPU info from
            -- sysdetect). Refreshed by the periodic probe.
            -- `last_seen` tracks liveness — a worker the host hasn't been
            -- able to reach for a while is greyed out in the picker.
            -- `use_for_*` flags let the user opt a worker into specific
            -- workloads (chat / embeddings / subagents) without having to
            -- delete-and-readd to change scope.
            CREATE TABLE IF NOT EXISTS compute_workers (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                address TEXT NOT NULL,
                ollama_port INTEGER NOT NULL DEFAULT 11434,
                auth_token TEXT,
                ssh_host TEXT,
                tailscale_host TEXT,
                enabled INTEGER NOT NULL DEFAULT 1,
                use_for_chat INTEGER NOT NULL DEFAULT 1,
                use_for_embeddings INTEGER NOT NULL DEFAULT 1,
                use_for_subagents INTEGER NOT NULL DEFAULT 1,
                capabilities_json TEXT,
                last_seen REAL,
                last_error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_compute_workers_enabled
                ON compute_workers(enabled);

            -- Split-model definitions (Phase 2 of the compute-pool feature).
            --
            -- A "split model" is a single GGUF served by `llama-server` on
            -- the host with `--rpc <worker>:<port>` flags so layers fan
            -- across one or more compute workers. Used for models too big
            -- for the host's own VRAM/RAM (Phase 1 routes WHOLE requests
            -- per machine; Phase 2 splits ONE inference across machines).
            --
            -- `gguf_path` is an absolute path on the host's disk — typically
            -- pointing at an Ollama-managed blob (`~/.ollama/models/blobs/
            -- sha256-…`). We re-use Ollama's already-downloaded GGUFs to
            -- avoid duplicating multi-GB files.
            -- `worker_ids_json` is a JSON array of `compute_workers.id`
            -- values whose rpc-server should be wired into llama-server's
            -- `--rpc` flag. Order matters: it controls layer assignment.
            -- `llama_port` is the local TCP port `llama-server` binds on
            -- the host (default 11500 — separate from Ollama's 11434 so
            -- both can coexist).
            -- `status` reflects the runtime state of the local
            -- `llama-server` process: stopped / loading / running / error.
            -- Updated by the lifecycle module, NOT by user edits.
            CREATE TABLE IF NOT EXISTS split_models (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                gguf_path TEXT NOT NULL,
                worker_ids_json TEXT NOT NULL DEFAULT '[]',
                llama_port INTEGER NOT NULL DEFAULT 11500,
                enabled INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'stopped',
                last_error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_split_models_enabled
                ON split_models(enabled);
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
            # Lazy tool-loading. JSON list of tool names whose full schemas
            # are currently visible to the model in this conversation. Empty
            # by default; the model uses `tool_load(["name", ...])` to add
            # entries on demand. The agent loop filters the schemas list
            # down to (meta-tools + this set) before each Ollama call, so
            # the system-prompt + tools-payload stays small until tools are
            # actually used. Survives restarts so a long conversation
            # doesn't have to re-load everything after a backend bounce.
            "ALTER TABLE conversations ADD COLUMN loaded_tools_json TEXT NOT NULL DEFAULT '[]'",
            # Workflow extension columns on `hooks`. The hooks table started
            # life as a simple "fire shell on lifecycle event" pipe; these
            # turn it into a general workflow trigger system without
            # introducing a parallel table that would duplicate the CRUD
            # surface. `error_threshold` is read by the
            # `on_consecutive_failures` event (default 1 = "fire on first
            # failure"); `max_fires_per_conv` caps how often a hook can
            # fire in one conversation so a buggy hook can't loop forever.
            "ALTER TABLE hooks ADD COLUMN error_threshold INTEGER",
            "ALTER TABLE hooks ADD COLUMN max_fires_per_conv INTEGER",
            # Re-key project_memories from the legacy `project` label to
            # `cwd`. Originally shipped as project-label-keyed in the
            # same release that shipped the third memory scope; we
            # corrected to cwd-keyed before any users wrote rows. The
            # rename keeps any in-flight test data intact, the index
            # follows automatically in modern SQLite. New tables get
            # the right shape from the CREATE TABLE above; this branch
            # only fires for DBs that booted the older code first.
            "ALTER TABLE project_memories RENAME COLUMN project TO cwd",
            "DROP INDEX IF EXISTS idx_project_memories_project",
            "CREATE INDEX IF NOT EXISTS idx_project_memories_cwd ON project_memories(cwd)",
            # Sidebar sort key. We used to order conversations by
            # `updated_at`, which the agent itself bumps every time it
            # commits an assistant or tool message — so a conversation
            # that's been running tools in the background drifts to
            # the top even when the user hasn't touched it. Tracking
            # the last USER message timestamp separately makes the
            # sidebar reflect "where the user has been talking" rather
            # than "where any agent activity happened". Backfill below
            # seeds the column from existing messages so post-migration
            # ordering is sensible from the first render.
            "ALTER TABLE conversations ADD COLUMN last_user_message_at REAL",
            # Drop the old (pinned, updated_at) index so the schema
            # block below can recreate it as (pinned,
            # last_user_message_at, updated_at) — the new sort needs
            # the extra column in the index to skip the sort step.
            # CREATE INDEX IF NOT EXISTS would otherwise see the old
            # index name and silently keep the wrong shape.
            "DROP INDEX IF EXISTS idx_conversations_sort",
            # Phase 2 commit 10: SSH alias / hostname for LAN-first
            # model copy. NULL means "no SSH configured" — the host
            # falls back to manual `ollama pull` instructions on the
            # worker side. Typical value is the alias the user has
            # in ~/.ssh/config (e.g. "laptop").
            "ALTER TABLE compute_workers ADD COLUMN ssh_host TEXT",
            # Optional Tailscale identifier (MagicDNS name or CGNAT
            # IPv4) used ONLY for the auto-repair routine in
            # compute_pool.py. When the LAN address goes stale (the
            # worker rejoined the network and DHCP gave it a new
            # lease) the host reaches the worker over Tailscale just
            # long enough to rediscover the new LAN IP, then resumes
            # ordinary LAN traffic. NULL means auto-repair is off
            # for this worker — a stale address triggers an
            # "unreachable" status until the user updates it manually.
            "ALTER TABLE compute_workers ADD COLUMN tailscale_host TEXT",
            # Phase 2 commit 24: optional path to a multimodal projector
            # GGUF (mmproj). When set, llama-server is launched with
            # `--mmproj <path>` so vision/image inputs work via Phase 2
            # split. Required for models like gemma4:26b whose Ollama
            # blob bundles a vision tower that stock llama-server can't
            # load directly. NULL means text-only.
            "ALTER TABLE split_models ADD COLUMN mmproj_path TEXT",
            # Speculative decoding: optional path to a smaller "draft"
            # GGUF that llama-server runs alongside the main model.
            # When set, llama-server is launched with `-md <path>` and
            # speculative-decoding tuning flags so it can use the
            # entire pool's model inventory to accelerate single-stream
            # generation on the node that's actually running the chat
            # — the draft proposes a few tokens fast, the main model
            # verifies them in a single batched pass. NULL means
            # vanilla single-model serving.
            "ALTER TABLE split_models ADD COLUMN draft_gguf_path TEXT",
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
        # Backfill last_user_message_at from the messages table so the
        # sidebar sort is meaningful from first render after migration.
        # Only touches rows where the column is currently NULL — running
        # this on every startup would be cheap (sub-millisecond on small
        # DBs) but stomp on conversations the user has just messaged in.
        c.execute(
            "UPDATE conversations SET last_user_message_at = ("
            "  SELECT MAX(created_at) FROM messages "
            "  WHERE messages.conversation_id = conversations.id "
            "  AND messages.role = 'user'"
            ") WHERE last_user_message_at IS NULL"
        )
        # Conversations with no user messages yet (just-created chats)
        # fall back to created_at so they still get a sensible position
        # in the list rather than sorting last forever.
        c.execute(
            "UPDATE conversations SET last_user_message_at = created_at "
            "WHERE last_user_message_at IS NULL"
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
            -- Sidebar sort: list_conversations + the outer search_conversations
            -- ORDER BY (pinned DESC, last_user_message_at DESC, updated_at
            -- DESC). A composite index covering the sort columns lets
            -- SQLite walk the B-tree in reverse and skip the sort entirely.
            -- `pinned` and `last_user_message_at` are post-migration columns
            -- on existing DBs, but the ALTER TABLEs above run first in the
            -- same init() call so both columns always exist by the time
            -- this index is created.
            CREATE INDEX IF NOT EXISTS idx_conversations_sort
                ON conversations(pinned, last_user_message_at, updated_at);
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
    """Return all conversations, pinned first then most-recently-updated.

    For sidebars that want pagination instead, see
    ``list_conversations_paginated``. The unbounded version is fine
    when the user has up to a few hundred chats; beyond that the
    JSON marshalling alone becomes a noticeable hitch on every
    sidebar refresh.
    """
    with _conn() as c:
        rows = c.execute(
            # Sidebar sort. Primary key is `last_user_message_at` so a
            # chat the user just messaged in floats to the top, even
            # if a different conversation has been chugging through
            # tool calls in the background. `updated_at` stays as the
            # tie-breaker for brand-new chats that haven't received a
            # user message yet (the migration backfills that column
            # with `created_at` in those cases).
            "SELECT * FROM conversations ORDER BY pinned DESC, "
            "last_user_message_at DESC, updated_at DESC"
        ).fetchall()
    return [_row_to_conversation(r) for r in rows]


def list_conversations_paginated(
    *,
    limit: int,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Return one page of conversations + the total count.

    Same sort order as ``list_conversations`` (pinned first, then
    last-user-message-at desc, then updated_at desc) so paged loads
    stitch together correctly. The total count comes from the same
    connection, eliminating the race between page fetch and count
    query that two-trip pagination would otherwise have.

    Cap is enforced at the DB layer via SQL ``LIMIT``; offset uses
    ``OFFSET`` which is fine at the ~thousand-conversation scale a
    user might reach. Beyond that, keyset pagination (``WHERE
    last_user_message_at < ?``) would be more scalable, but the
    sort key isn't strictly monotonic across the pinned/unpinned
    boundary so offset is the simpler correct choice here.
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM conversations ORDER BY pinned DESC, "
            "last_user_message_at DESC, updated_at DESC "
            "LIMIT ? OFFSET ?",
            (int(limit), int(offset)),
        ).fetchall()
        total_row = c.execute(
            "SELECT COUNT(*) AS n FROM conversations"
        ).fetchone()
        total = int(total_row["n"]) if total_row else 0
    return [_row_to_conversation(r) for r in rows], total


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


def get_loaded_tools(cid: str) -> list[str]:
    """Return the per-conversation set of tools the model has loaded.

    Used by the lazy-tool-loading flow: the agent loop filters its
    schemas list down to (meta-tools + this set) so the model's prompt
    stays small until tools are actually requested via `tool_load`.

    Empty list = no extra tools loaded yet (only the meta-tools are
    visible). Order is preserved as best-effort but callers should treat
    it as a set.
    """
    with _conn() as c:
        row = c.execute(
            "SELECT loaded_tools_json FROM conversations WHERE id = ?",
            (cid,),
        ).fetchone()
    if not row:
        return []
    raw = row["loaded_tools_json"] or "[]"
    try:
        names = _json_loads(raw)
    except Exception:
        return []
    if not isinstance(names, list):
        return []
    return [str(n) for n in names if isinstance(n, str)]


def add_loaded_tools(cid: str, names: list[str]) -> list[str]:
    """Mark `names` as loaded for the conversation.

    Returns the new full set after the union. Idempotent — re-adding an
    already-loaded name is a no-op. The caller is responsible for
    validating that the names actually exist in the registry; this
    function just tracks the bookkeeping.
    """
    if not names:
        return get_loaded_tools(cid)
    with _conn() as c:
        row = c.execute(
            "SELECT loaded_tools_json FROM conversations WHERE id = ?",
            (cid,),
        ).fetchone()
        if not row:
            return []
        try:
            current = _json_loads(row["loaded_tools_json"] or "[]")
            if not isinstance(current, list):
                current = []
        except Exception:
            current = []
        merged: list[str] = []
        seen: set[str] = set()
        for n in list(current) + list(names):
            if not isinstance(n, str) or n in seen:
                continue
            seen.add(n)
            merged.append(n)
        c.execute(
            "UPDATE conversations SET loaded_tools_json = ? WHERE id = ?",
            (json.dumps(merged), cid),
        )
    return merged


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
            imgs = _json_loads(r["images"]) if r["images"] else None
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
        # Always bump updated_at — keeps the existing crash-resilience
        # / staleness signals correct. Bump last_user_message_at ONLY
        # on user-role rows so the sidebar sort reflects "where the
        # user has been talking" rather than "where any agent
        # activity happened".
        if role == "user":
            c.execute(
                "UPDATE conversations SET updated_at = ?, "
                "last_user_message_at = ? WHERE id = ?",
                (now, now, cid),
            )
        else:
            c.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, cid),
            )
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
        # Bump both updated_at AND last_user_message_at — editing a user
        # message is itself a user action, so the sidebar should treat
        # it like a fresh user message for sort purposes.
        if row:
            now = time.time()
            c.execute(
                "UPDATE conversations SET updated_at = ?, "
                "last_user_message_at = ? WHERE id = ?",
                (now, now, row["conversation_id"]),
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
    """Return every (message_id, vector) pair for a conversation.

    For top-k search prefer ``search_embeddings_topk_for_conv`` — it
    skips the per-row Python list reconstruction and runs the dot
    products in a single numpy matmul (~10× faster on long
    conversations).
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT message_id, embedding FROM message_embeddings WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchall()
    return [(r["message_id"], _unpack_vec(r["embedding"])) for r in rows]


def search_embeddings_topk_for_conv(
    conversation_id: str,
    query_vector: list[float],
    *,
    top_k: int,
    min_score: float = 0.0,
    exclude_ids: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Numpy-vectorized top-k dot-product search over a conversation's
    message embeddings.

    Both stored and query vectors are expected to be unit-norm (the
    embed model emits normalized vectors); we therefore use plain dot
    product as the similarity score instead of a full cosine. Saves
    two `linalg.norm` calls per query at no accuracy cost.

    Returns ``[(message_id, score), ...]`` sorted by descending score,
    with `score >= min_score` and `id not in exclude_ids`. Empty list
    on any read failure or when no rows match.
    """
    if not query_vector or top_k <= 0:
        return []
    with _conn() as c:
        rows = c.execute(
            "SELECT message_id, embedding FROM message_embeddings WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchall()
    if not rows:
        return []

    excl = exclude_ids or set()
    expected = len(query_vector) * 4
    valid_indices: list[int] = []
    buffers: list[bytes] = []
    for i, r in enumerate(rows):
        if r["message_id"] in excl:
            continue
        v = r["embedding"]
        if not v or len(v) != expected:
            continue
        buffers.append(v)
        valid_indices.append(i)
    if not valid_indices:
        return []

    import numpy as np
    matrix = np.frombuffer(b"".join(buffers), dtype=np.float32)
    matrix = matrix.reshape(len(valid_indices), len(query_vector))
    qv = np.asarray(query_vector, dtype=np.float32)
    scores = matrix @ qv

    # Apply the min_score floor before top-k so we don't waste a slot
    # on a barely-relevant hit when good hits exist.
    mask = scores >= float(min_score)
    if not mask.any():
        return []
    valid_score_idx = np.where(mask)[0]
    valid_scores = scores[valid_score_idx]

    n = len(valid_scores)
    k = min(top_k, n)
    if k >= n:
        order = np.argsort(-valid_scores)
    else:
        partition = np.argpartition(-valid_scores, k)[:k]
        order = partition[np.argsort(-valid_scores[partition])]

    out: list[tuple[str, float]] = []
    for j in order:
        ix = valid_indices[int(valid_score_idx[int(j)])]
        out.append((rows[ix]["message_id"], float(valid_scores[int(j)])))
    return out


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
    """Return every message for a conversation, oldest-first.

    For very long conversations prefer ``list_messages_paginated`` —
    the unbounded version pulls every row, which can be slow on
    chats that grew to thousands of messages over weeks of use.
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (cid,),
        ).fetchall()
    return [_row_to_message(r) for r in rows]


def list_messages_paginated(
    cid: str,
    *,
    limit: int,
    before_id: str | None = None,
) -> tuple[list[dict], int]:
    """Return up to ``limit`` most-recent messages plus the conversation's
    total message count. Used for the scroll-up pagination path on
    large conversations.

    When ``before_id`` is set the page returns messages strictly older
    than that one (for scroll-up "load more" gestures). When unset
    the page returns the most-recent ``limit`` messages — the typical
    initial load.

    Result is always oldest-first within the page so the frontend
    can append pages above its existing list without re-sorting.
    Total count is computed in the same connection so the caller can
    show "showing 200 of 1834".
    """
    with _conn() as c:
        if before_id:
            cutoff_row = c.execute(
                "SELECT created_at FROM messages "
                "WHERE id = ? AND conversation_id = ?",
                (before_id, cid),
            ).fetchone()
            if not cutoff_row:
                # Unknown anchor — return empty page so the UI
                # doesn't double-render the existing tail.
                total_row = c.execute(
                    "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?",
                    (cid,),
                ).fetchone()
                total = int(total_row["n"]) if total_row else 0
                return [], total
            rows = c.execute(
                "SELECT * FROM messages "
                "WHERE conversation_id = ? AND created_at < ? "
                "ORDER BY created_at DESC LIMIT ?",
                (cid, cutoff_row["created_at"], int(limit)),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM messages "
                "WHERE conversation_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (cid, int(limit)),
            ).fetchall()
        total_row = c.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?",
            (cid,),
        ).fetchone()
        total = int(total_row["n"]) if total_row else 0
    # Reverse so the page is oldest-first within itself.
    rows = list(reversed(rows))
    return [_row_to_message(r) for r in rows], total


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
    """Insert one chunk + its embedding. Returns the new row id.

    For bulk inserts (the common indexing case where dozens-to-hundreds
    of chunks land per file), prefer `insert_doc_chunks_batch` — it
    bundles every row into a single transaction via `executemany`,
    cutting SQLite's per-row commit overhead by 3-5×.
    """
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


def insert_doc_chunks_batch(
    rows: list[dict],
) -> int:
    """Bulk-insert chunks via a single transaction. Returns row count.

    Each row dict must carry ``path``, ``ordinal``, ``text``, ``vector``,
    and ``model``. The function generates UUIDs and timestamps; callers
    only supply the embedded payload.

    Why batch: every individual `insert_doc_chunk` opens a transaction
    and forces an fsync at commit. For an indexer producing 1 000
    chunks that's 1 000 fsyncs — easily 3-10 s of disk wait on a
    typical SSD. `executemany` with one outer transaction collapses
    that to a single fsync, dropping the DB-write phase from seconds
    to tens of milliseconds. Combined with the pool-distributed
    embed fan-out, full indexing throughput is no longer DB-bound.

    Empty input is a no-op (returns 0). Malformed rows raise — the
    caller is expected to validate input shape.
    """
    if not rows:
        return 0
    now = time.time()
    payload = [
        (
            str(uuid.uuid4()),
            r["path"],
            int(r["ordinal"]),
            r["text"],
            _pack_vec(r["vector"]),
            r["model"],
            now,
        )
        for r in rows
    ]
    with _conn() as c:
        c.executemany(
            "INSERT INTO doc_chunks "
            "(id, path, ordinal, text, vector, model, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            payload,
        )
    return len(payload)


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

    For top-k cosine search, prefer ``search_doc_chunks_topk`` — it
    skips the per-row Python list allocation and uses numpy's
    vectorized matmul for the cosine score, ~10-50× faster on indexes
    of ~1000 chunks or more.
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


def search_doc_chunks_topk(
    query_vector: list[float],
    top_k: int,
    *,
    path_substr: str | None = None,
    path_prefix: str | None = None,
) -> list[tuple[float, dict]]:
    """Numpy-vectorized top-k cosine similarity search over doc_chunks.

    The previous path (``all_doc_chunks`` + per-row Python cosine) ran
    in O(N × D) pure-Python multiplications — ~5-10 ms per row at 768
    dims, so 1000-chunk indexes burned 5-10 s. This function:

      * pulls only id, path, ordinal, text, model, vector (bytes)
        in one query — skips the Python list reconstruction;
      * stacks the raw bytes into a single ``np.float32`` matrix
        via ``np.frombuffer`` (zero-copy where alignment allows);
      * computes cosine via a single matmul + norm divisions —
        ~10-50× faster than the equivalent Python loop;
      * returns only the top-k results via ``np.argpartition``,
        avoiding a full sort of the score array.

    ``path_prefix`` is a stricter filter than ``path_substr`` — applied
    *after* SQL fetches but before scoring, so callers that need
    "starts with X" semantics (codebase_search scoping by cwd, e.g.)
    can express it without a Python post-filter.

    Returns ``[(score, row_dict), ...]`` sorted by descending score.
    Each ``row_dict`` carries the same fields as ``all_doc_chunks``
    EXCEPT ``vector`` is omitted (caller didn't ask for it; saves
    serialization time when many candidates exist).
    """
    if not query_vector or top_k <= 0:
        return []

    with _conn() as c:
        if path_substr:
            rows = c.execute(
                "SELECT id, path, ordinal, text, vector, model, created_at "
                "FROM doc_chunks WHERE path LIKE ?",
                (f"%{path_substr}%",),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT id, path, ordinal, text, vector, model, created_at "
                "FROM doc_chunks"
            ).fetchall()

    if not rows:
        return []

    # Optional stricter prefix filter — applied after SQL because
    # SQL LIKE doesn't have anchored-prefix semantics without escaping
    # every special character.
    if path_prefix:
        rows = [r for r in rows if (r["path"] or "").startswith(path_prefix)]
        if not rows:
            return []

    import numpy as np
    expected_bytes = len(query_vector) * 4  # float32 = 4 bytes
    valid_indices: list[int] = []
    matrix_buffers: list[bytes] = []
    for i, r in enumerate(rows):
        v = r["vector"]
        # Skip rows whose vector dim doesn't match the query — happens
        # when the embed model changed since the chunk was indexed.
        # Mismatched-dim cosine is meaningless; better to drop than crash.
        if not v or len(v) != expected_bytes:
            continue
        matrix_buffers.append(v)
        valid_indices.append(i)
    if not valid_indices:
        return []

    matrix = np.frombuffer(b"".join(matrix_buffers), dtype=np.float32)
    matrix = matrix.reshape(len(valid_indices), len(query_vector))
    qv = np.asarray(query_vector, dtype=np.float32)

    # Cosine similarity: (M @ q) / (||M_row|| * ||q||)
    dot = matrix @ qv
    matrix_norms = np.linalg.norm(matrix, axis=1)
    qv_norm = np.linalg.norm(qv)
    # Replace zero norms with 1.0 — saves a divide-by-zero without
    # affecting the result (any vector with zero norm scores 0/1 = 0).
    matrix_norms = np.where(matrix_norms == 0, 1.0, matrix_norms)
    if qv_norm == 0:
        qv_norm = 1.0
    scores = dot / (matrix_norms * qv_norm)

    # Top-k via argpartition: O(N) average vs O(N log N) for full sort.
    # Worth it when N is in the thousands and k is small (typical).
    n = len(scores)
    k = min(top_k, n)
    if k >= n:
        top_idx = np.argsort(-scores)
    else:
        partition = np.argpartition(-scores, k)[:k]
        # Sort just the top-k slice so the caller sees descending order.
        top_idx = partition[np.argsort(-scores[partition])]

    out: list[tuple[float, dict]] = []
    for i in top_idx:
        r = rows[valid_indices[int(i)]]
        out.append((
            float(scores[int(i)]),
            {
                "id": r["id"],
                "path": r["path"],
                "ordinal": r["ordinal"],
                "text": r["text"],
                "model": r["model"],
                "created_at": r["created_at"],
            },
        ))
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
HOOK_EVENTS = {
    "pre_tool",
    "post_tool",
    "user_prompt_submit",
    "turn_done",
    # Fires when a tool call returns ok=False. Subset of post_tool —
    # cleaner trigger for "lint after every successful write" vs.
    # "diagnose after every failure" use cases without an in-hook
    # `if .ok` check.
    "tool_error",
    # Fires after N consecutive ok=False results from the SAME tool name
    # in the same conversation. N comes from `error_threshold` (default 1).
    # Designed for the "model looped on the same broken call, ask Claude
    # to step in" workflow.
    "consecutive_failures",
}

# Event-specific timeout caps. Stateful diagnosis hooks (consecutive_failures
# in particular, since a typical use is calling out to Claude Code or a
# local linter that takes minutes) need much longer headroom than the
# fire-and-forget post_tool hooks which the original 120 s cap was tuned
# for. We still cap them — a hung diagnosis hook would wedge the turn.
_HOOK_TIMEOUT_CAP_DEFAULT = 120
_HOOK_TIMEOUT_CAPS = {
    "tool_error": 900,
    "consecutive_failures": 900,
}


def hook_timeout_cap(event: str) -> int:
    """Return the maximum allowed `timeout_seconds` for a hook on this event."""
    return _HOOK_TIMEOUT_CAPS.get(event, _HOOK_TIMEOUT_CAP_DEFAULT)


def create_hook(
    *,
    event: str,
    command: str,
    matcher: str | None = None,
    timeout_seconds: int = 10,
    enabled: bool = True,
    error_threshold: int | None = None,
    max_fires_per_conv: int | None = None,
) -> str:
    """Insert a new hook and return its id. Raises ValueError on bad input."""
    if event not in HOOK_EVENTS:
        raise ValueError(f"event must be one of {sorted(HOOK_EVENTS)}, got {event!r}")
    cmd = (command or "").strip()
    if not cmd:
        raise ValueError("command must not be empty")
    # Clamp timeout so a runaway hook can't block the agent forever. The
    # ceiling is event-specific — diagnosis hooks need minutes.
    ts = max(1, min(int(timeout_seconds or 10), hook_timeout_cap(event)))
    # error_threshold: only meaningful for consecutive_failures. Clamp 1..50
    # so a typo ("threshold = 999") doesn't accidentally disable the hook.
    et = None
    if error_threshold is not None:
        et = max(1, min(int(error_threshold), 50))
    # max_fires_per_conv: NULL means unlimited. Otherwise clamp 1..1000.
    mfpc = None
    if max_fires_per_conv is not None:
        mfpc = max(1, min(int(max_fires_per_conv), 1000))
    hid = str(uuid.uuid4())
    with _conn() as c:
        c.execute(
            "INSERT INTO hooks (id, event, matcher, command, timeout_seconds, "
            "enabled, created_at, error_threshold, max_fires_per_conv) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (hid, event, (matcher or None), cmd, ts, 1 if enabled else 0,
             time.time(), et, mfpc),
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
    allowed = {
        "event", "matcher", "command", "timeout_seconds", "enabled",
        "error_threshold", "max_fires_per_conv",
    }
    # We need the current event for the timeout-cap clamp when only
    # `timeout_seconds` is being patched (the cap depends on event).
    current = get_hook(id)
    if not current:
        return None
    target_event = fields.get("event", current["event"])
    if target_event not in HOOK_EVENTS:
        raise ValueError(f"event must be one of {sorted(HOOK_EVENTS)}")
    sets: list[str] = []
    values: list[Any] = []
    for k, v in fields.items():
        if k not in allowed:
            continue
        if k == "enabled":
            v = 1 if v else 0
        elif k == "timeout_seconds":
            v = max(1, min(int(v or 10), hook_timeout_cap(target_event)))
        elif k == "error_threshold":
            v = None if v in (None, "") else max(1, min(int(v), 50))
        elif k == "max_fires_per_conv":
            v = None if v in (None, "") else max(1, min(int(v), 1000))
        sets.append(f"{k} = ?")
        values.append(v)
    if not sets:
        return current
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
    # Some columns were added in a later migration — using row[col] would
    # KeyError on rows from a DB that hasn't been migrated yet. Accessing
    # via .keys() is the cheap defensive check.
    cols = row.keys() if hasattr(row, "keys") else []
    return {
        "id": row["id"],
        "event": row["event"],
        "matcher": row["matcher"],
        "command": row["command"],
        "timeout_seconds": row["timeout_seconds"],
        "enabled": bool(row["enabled"]),
        "created_at": row["created_at"],
        "error_threshold": row["error_threshold"] if "error_threshold" in cols else None,
        "max_fires_per_conv": row["max_fires_per_conv"] if "max_fires_per_conv" in cols else None,
    }


# --- Hook fire counter (per-conversation cap enforcement) -----------------


def get_hook_fire_count(hook_id: str, conversation_id: str) -> int:
    """Return how many times this hook has fired in this conversation."""
    with _conn() as c:
        row = c.execute(
            "SELECT fire_count FROM hook_fires WHERE hook_id = ? AND conversation_id = ?",
            (hook_id, conversation_id),
        ).fetchone()
    return int(row["fire_count"]) if row else 0


def incr_hook_fire(hook_id: str, conversation_id: str) -> int:
    """Atomic +1 on the (hook, conv) counter. Returns the new count."""
    with _conn() as c:
        c.execute(
            "INSERT INTO hook_fires (hook_id, conversation_id, fire_count, last_fired_at) "
            "VALUES (?, ?, 1, ?) "
            "ON CONFLICT(hook_id, conversation_id) DO UPDATE SET "
            "fire_count = fire_count + 1, last_fired_at = excluded.last_fired_at",
            (hook_id, conversation_id, time.time()),
        )
        row = c.execute(
            "SELECT fire_count FROM hook_fires WHERE hook_id = ? AND conversation_id = ?",
            (hook_id, conversation_id),
        ).fetchone()
    return int(row["fire_count"]) if row else 1


def reset_hook_fires(hook_id: str | None = None, conversation_id: str | None = None) -> int:
    """Clear fire counters. Both args optional — pass either or both as a
    filter; pass neither to wipe everything (used by tests)."""
    sql = "DELETE FROM hook_fires WHERE 1=1"
    args: list[Any] = []
    if hook_id is not None:
        sql += " AND hook_id = ?"
        args.append(hook_id)
    if conversation_id is not None:
        sql += " AND conversation_id = ?"
        args.append(conversation_id)
    with _conn() as c:
        cur = c.execute(sql, args)
        return cur.rowcount or 0


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
# Project memories (per-cwd shared notes).
#
# Sit between `global_memories` (user-wide) and per-conversation memory
# (per-chat). Keyed by the conversation's `cwd` — every chat working in
# the same directory automatically shares the same memory set, with no
# user-side configuration required. Move a chat's cwd and it sees a
# different (or empty) project memory.
# ---------------------------------------------------------------------------


def _normalize_project_cwd(cwd: str | None) -> str:
    """Canonicalize a cwd path for project-memory lookups so two
    conversations entered in slightly different forms still match.

    Trims whitespace, removes trailing slashes (except for drive roots
    like ``C:\\``), and lowercases on Windows where the FS is
    case-insensitive — anywhere else, case is meaningful.
    """
    p = (cwd or "").strip()
    if not p:
        return ""
    # Strip a single trailing separator, but preserve `C:\` / `/` as
    # those are roots in their own right.
    if len(p) > 3 and p[-1] in ("/", "\\"):
        p = p.rstrip("/\\")
    if os.name == "nt":
        p = p.lower()
    return p


def list_project_memories(cwd: str) -> list[dict]:
    """Return every memory tied to this cwd, oldest-first."""
    key = _normalize_project_cwd(cwd)
    if not key:
        return []
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM project_memories WHERE cwd = ? "
            "ORDER BY created_at ASC",
            (key,),
        ).fetchall()
    return [_row_to_project_memory(r) for r in rows]


def add_project_memory(
    cwd: str, content: str, topic: str | None = None,
) -> dict:
    """Insert a project memory and return the row. Same length caps as
    global memory so a runaway agent can't blow up the prompt."""
    key = _normalize_project_cwd(cwd)
    if not key:
        raise ValueError("cwd is required")
    if len(key) > 1024:
        raise ValueError("cwd path must be ≤ 1024 chars")
    body = (content or "").strip()
    if not body:
        raise ValueError("content is required")
    if len(body) > 8000:
        raise ValueError("memory body must be ≤ 8 KB")
    t = (topic or "").strip() or None
    if t and len(t) > 80:
        raise ValueError("topic must be ≤ 80 chars")
    mid = str(uuid.uuid4())
    now = time.time()
    with _conn() as c:
        c.execute(
            "INSERT INTO project_memories (id, cwd, content, topic, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (mid, key, body, t, now, now),
        )
        row = c.execute(
            "SELECT * FROM project_memories WHERE id = ?", (mid,),
        ).fetchone()
    return _row_to_project_memory(row)


def delete_project_memories_matching(cwd: str, pattern: str) -> int:
    """Delete every project memory whose content contains `pattern`
    (case-insensitive substring), scoped to the given cwd. Same SQL-
    LIKE-metachar escaping as `delete_global_memories_matching`."""
    key = _normalize_project_cwd(cwd)
    if not key:
        raise ValueError("cwd is required")
    needle = (pattern or "").strip()
    if not needle:
        raise ValueError("pattern is required")
    escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    like = f"%{escaped}%"
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM project_memories "
            "WHERE cwd = ? AND LOWER(content) LIKE LOWER(?) ESCAPE '\\'",
            (key, like),
        )
        return cur.rowcount or 0


def _row_to_project_memory(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "cwd": row["cwd"],
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
        args = _json_loads(row["args_json"] or "[]")
        if not isinstance(args, list):
            args = []
    except Exception:
        args = []
    try:
        env = _json_loads(row["env_json"] or "{}")
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
# In-process micro-cache for `get_setting` reads. Settings change rarely
# (user toggles a UI option maybe once per session) but get read on
# every chat turn. The cache stores JSON-decoded values keyed by name;
# writes through `set_setting` / `delete_setting` invalidate the entry
# so changes propagate immediately. The "missing" sentinel
# `_SETTING_MISS` lets us cache negative lookups too — common for
# feature flags that are unset until the user explicitly opts in.
_SETTING_MISS = object()
_SETTING_CACHE: dict[str, Any] = {}


def get_setting(key: str, default: Any = None) -> Any:
    """Return the stored value for `key`, or `default` if the key is unset.

    Values are JSON-decoded; a malformed row collapses to `default` rather
    than raising so a corrupt entry can't crash the settings route.

    Cached in-process via `_SETTING_CACHE`; any `set_setting` or
    `delete_setting` call invalidates the matching key so the next
    read sees the fresh value. Cache hit avoids the SQLite open + JSON
    decode round-trip — measurably faster on chat turns that consult
    multiple settings (~0.5-1 ms saved per setting on warm cache).
    """
    cached = _SETTING_CACHE.get(key, _SETTING_MISS)
    if cached is not _SETTING_MISS:
        # `None` as a stored value is legitimate — distinguish from
        # "missing" via the sentinel pattern.
        if cached is None:
            return default
        return cached
    with _conn() as c:
        row = c.execute(
            "SELECT value FROM user_settings WHERE key = ?", (key,)
        ).fetchone()
    if not row:
        # Cache the negative result so repeated reads of an unset key
        # also hit the in-memory path.
        _SETTING_CACHE[key] = None
        return default
    try:
        value = _json_loads(row["value"])
    except (json.JSONDecodeError, TypeError):
        return default
    _SETTING_CACHE[key] = value
    return value


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
    # Invalidate so the next `get_setting` re-reads from disk and
    # caches the new value.
    _SETTING_CACHE.pop(key, None)


def delete_setting(key: str) -> int:
    """Remove one setting row. Returns rows deleted (0 or 1)."""
    with _conn() as c:
        cur = c.execute("DELETE FROM user_settings WHERE key = ?", (key,))
        rows = cur.rowcount or 0
    _SETTING_CACHE.pop(key, None)
    return rows


def get_all_settings() -> dict[str, Any]:
    """Return every stored setting as a {key: value} dict (JSON-decoded).

    Bypasses the per-key cache for the read but populates it from the
    result so subsequent `get_setting` calls hit warm.
    """
    with _conn() as c:
        rows = c.execute("SELECT key, value FROM user_settings").fetchall()
    out: dict[str, Any] = {}
    for r in rows:
        try:
            out[r["key"]] = _json_loads(r["value"])
            _SETTING_CACHE[r["key"]] = out[r["key"]]
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
        tags = _json_loads(raw_tags) if raw_tags else []
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
        "tool_calls": _json_loads(row["tool_calls"]) if row["tool_calls"] else [],
        "images": _json_loads(raw_images) if raw_images else [],
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
        schema = _json_loads(row["schema_json"]) if row["schema_json"] else {}
    except Exception:
        schema = {}
    try:
        deps = _json_loads(row["deps_json"]) if row["deps_json"] else []
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


# ---------------------------------------------------------------------------
# Compute pool (other PCs registered as workers).
#
# The host can route embeddings, subagents, and chat turns to these
# workers to parallelise across machines. See `compute_pool.py` for
# the routing logic and capability-probe loop; this layer is just CRUD.
# ---------------------------------------------------------------------------

# Hard cap on stored hostnames/IPs — comfortably above any real DNS name
# but tight enough to refuse a megabyte-of-junk DOS attempt.
_HOSTNAME_MAX_LEN = 256


def _normalise_optional_host(value: str | None) -> str | None:
    """Trim a user-entered hostname; return None for empty input.

    Used for `ssh_host` and `tailscale_host` columns where NULL means
    "feature off" and any non-empty string is the resolved identifier.
    """
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    if len(v) > _HOSTNAME_MAX_LEN:
        raise ValueError(f"host must be ≤ {_HOSTNAME_MAX_LEN} chars")
    return v


def create_compute_worker(
    *,
    label: str,
    address: str,
    ollama_port: int = 11434,
    auth_token: str | None = None,
    ssh_host: str | None = None,
    tailscale_host: str | None = None,
    enabled: bool = True,
    use_for_chat: bool = True,
    use_for_embeddings: bool = True,
    use_for_subagents: bool = True,
) -> str:
    """Insert a new compute worker and return its id.

    `address` is a LAN hostname or IPv4 — e.g. `worker.local` for mDNS
    or a `192.168.x.x` for a hardcoded LAN address. All ongoing traffic
    (chat / embeddings / subagents) flows over this address; the host
    probes `http://{address}:{ollama_port}` for capabilities.

    `tailscale_host` is optional and used ONLY for the auto-repair
    routine in `compute_pool.py`. It exists so a stale `address` after
    a DHCP rebind can be rediscovered without user intervention.
    """
    lbl = (label or "").strip()
    if not lbl:
        raise ValueError("label is required")
    if len(lbl) > 80:
        raise ValueError("label must be ≤ 80 chars")
    addr = (address or "").strip()
    if not addr:
        raise ValueError("address is required")
    if len(addr) > _HOSTNAME_MAX_LEN:
        raise ValueError(f"address must be ≤ {_HOSTNAME_MAX_LEN} chars")
    # Be careful with the falsy coalesce — `int(0 or 11434)` returns
    # 11434, but a caller passing 0 explicitly meant "clamp to 1", not
    # "use default". Treat None as "use default" and any int as a value
    # to clamp.
    port = 11434 if ollama_port is None else max(1, min(int(ollama_port), 65535))
    ssh_host_clean = _normalise_optional_host(ssh_host)
    tailscale_clean = _normalise_optional_host(tailscale_host)
    wid = str(uuid.uuid4())
    now = time.time()
    with _conn() as c:
        c.execute(
            "INSERT INTO compute_workers ("
            "id, label, address, ollama_port, auth_token, ssh_host, "
            "tailscale_host, enabled, use_for_chat, use_for_embeddings, "
            "use_for_subagents, created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                wid, lbl, addr, port, auth_token,
                ssh_host_clean, tailscale_clean,
                1 if enabled else 0,
                1 if use_for_chat else 0,
                1 if use_for_embeddings else 0,
                1 if use_for_subagents else 0,
                now, now,
            ),
        )
    _invalidate_compute_workers_cache()
    return wid


# Tiny in-process cache for the worker roster. The routing layer
# consults `list_compute_workers` on every chat turn, every embed
# call, every tool dispatch — for a 100-chunk indexing run that's
# 100+ identical SELECTs against an unchanging table. A 1-second
# TTL collapses the burst into a single read while keeping the
# data fresh for the periodic probe loop's writes.
#
# Cache holds both flavours (enabled_only=True/False) keyed
# separately. Writes (`update_compute_worker`,
# `update_compute_worker_capabilities`, `delete_compute_worker`,
# `add_compute_worker`) call `_invalidate_compute_workers_cache`
# so a fresh probe result becomes visible on the next read with
# zero TTL wait.
_COMPUTE_WORKERS_CACHE: dict[bool, tuple[float, list[dict]]] = {}
_COMPUTE_WORKERS_CACHE_TTL = 1.0


def _invalidate_compute_workers_cache() -> None:
    """Drop both flavours of the cached worker list. Called from every
    write path so subsequent reads see the change immediately.
    """
    _COMPUTE_WORKERS_CACHE.clear()


def list_compute_workers(*, enabled_only: bool = False) -> list[dict]:
    """Return every registered worker, newest-first.

    Cached for ``_COMPUTE_WORKERS_CACHE_TTL`` seconds across all
    callers — the routing layer can call this dozens of times per
    second on a busy indexing burst, but the underlying table only
    changes when a probe lands or the user edits a worker row in
    Settings. Both events explicitly invalidate the cache so users
    don't wait the full TTL for their changes to land.
    """
    cached = _COMPUTE_WORKERS_CACHE.get(enabled_only)
    now = time.time()
    if cached and (now - cached[0]) < _COMPUTE_WORKERS_CACHE_TTL:
        # Return a defensive copy so callers can't mutate the cached
        # list. The dicts inside are also references — callers don't
        # mutate those by convention, so we don't deep-copy.
        return list(cached[1])

    sql = "SELECT * FROM compute_workers"
    if enabled_only:
        sql += " WHERE enabled = 1"
    sql += " ORDER BY created_at DESC"
    with _conn() as c:
        rows = c.execute(sql).fetchall()
    workers = [_row_to_compute_worker(r) for r in rows]
    _COMPUTE_WORKERS_CACHE[enabled_only] = (now, workers)
    return list(workers)


def get_compute_worker(wid: str) -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM compute_workers WHERE id = ?", (wid,),
        ).fetchone()
    return _row_to_compute_worker(row) if row else None


def update_compute_worker(wid: str, **fields: Any) -> dict | None:
    """Patch allowed fields on a worker. Returns the refreshed row.

    For optional-host fields (`ssh_host`, `tailscale_host`) the
    convention is: pass an empty string to clear the column, ``None`` /
    omit the key to leave it unchanged. Non-empty strings are trimmed
    and length-checked the same way as in `create_compute_worker`.
    """
    allowed = {
        "label", "address", "ollama_port", "auth_token",
        "ssh_host", "tailscale_host",
        "enabled", "use_for_chat", "use_for_embeddings", "use_for_subagents",
    }
    sets: list[str] = []
    values: list[Any] = []
    for k, v in fields.items():
        if k not in allowed:
            continue
        if k in ("enabled", "use_for_chat", "use_for_embeddings",
                 "use_for_subagents"):
            v = 1 if v else 0
        elif k == "ollama_port":
            v = 11434 if v is None else max(1, min(int(v), 65535))
        elif k in ("ssh_host", "tailscale_host"):
            # Empty string clears; None leaves alone (handled by skipping
            # the key in the patch dict at the API layer).
            v = _normalise_optional_host(v)
        elif k == "address":
            addr = (v or "").strip()
            if not addr:
                raise ValueError("address must not be empty")
            if len(addr) > _HOSTNAME_MAX_LEN:
                raise ValueError(f"address must be ≤ {_HOSTNAME_MAX_LEN} chars")
            v = addr
        elif k == "label":
            lbl = (v or "").strip()
            if not lbl:
                raise ValueError("label must not be empty")
            if len(lbl) > 80:
                raise ValueError("label must be ≤ 80 chars")
            v = lbl
        sets.append(f"{k} = ?")
        values.append(v)
    if not sets:
        return get_compute_worker(wid)
    sets.append("updated_at = ?")
    values.append(time.time())
    values.append(wid)
    with _conn() as c:
        c.execute(
            f"UPDATE compute_workers SET {', '.join(sets)} WHERE id = ?",
            values,
        )
    _invalidate_compute_workers_cache()
    return get_compute_worker(wid)


def update_compute_worker_capabilities(
    wid: str,
    *,
    capabilities: dict | None = None,
    last_seen: float | None = None,
    last_error: str | None = None,
) -> None:
    """Cache the latest probe result on the worker row.

    Separated from `update_compute_worker` because the capability probe
    runs in the background and shouldn't be lumped in with user-facing
    edits (different audit trail / different concurrency story).
    """
    sets: list[str] = []
    values: list[Any] = []
    if capabilities is not None:
        sets.append("capabilities_json = ?")
        values.append(json.dumps(capabilities))
    if last_seen is not None:
        sets.append("last_seen = ?")
        values.append(float(last_seen))
    # last_error: empty string means "clear"; None means "leave alone".
    if last_error is not None:
        sets.append("last_error = ?")
        values.append(last_error or None)
    if not sets:
        return
    sets.append("updated_at = ?")
    values.append(time.time())
    values.append(wid)
    with _conn() as c:
        c.execute(
            f"UPDATE compute_workers SET {', '.join(sets)} WHERE id = ?",
            values,
        )
    _invalidate_compute_workers_cache()


def delete_compute_worker(wid: str) -> int:
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM compute_workers WHERE id = ?", (wid,),
        )
        rowcount = cur.rowcount or 0
    _invalidate_compute_workers_cache()
    return rowcount


def _row_to_compute_worker(row: sqlite3.Row) -> dict:
    caps_raw = row["capabilities_json"]
    try:
        caps = _json_loads(caps_raw) if caps_raw else None
    except Exception:
        caps = None
    # ssh_host / tailscale_host may not exist on older DBs that haven't
    # been migrated yet — sqlite3.Row raises IndexError on unknown
    # columns. Defensive accessors keep first-run reads safe.
    try:
        ssh_host = row["ssh_host"]
    except IndexError:
        ssh_host = None
    try:
        tailscale_host = row["tailscale_host"]
    except IndexError:
        tailscale_host = None
    return {
        "id": row["id"],
        "label": row["label"],
        "address": row["address"],
        "ollama_port": row["ollama_port"],
        # auth_token is NEVER returned to the API/UI — the column exists
        # for outbound requests only. Surfacing it would be a credential
        # exfiltration vector. Use `get_compute_worker_auth_token(wid)`
        # for the rare internal caller that needs it.
        "auth_token_set": bool(row["auth_token"]),
        "ssh_host": ssh_host,
        "tailscale_host": tailscale_host,
        "enabled": bool(row["enabled"]),
        "use_for_chat": bool(row["use_for_chat"]),
        "use_for_embeddings": bool(row["use_for_embeddings"]),
        "use_for_subagents": bool(row["use_for_subagents"]),
        "capabilities": caps,
        "last_seen": row["last_seen"],
        "last_error": row["last_error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get_compute_worker_auth_token(wid: str) -> str | None:
    """Internal-only: fetch the raw auth token for outbound requests.

    Kept off `_row_to_compute_worker` so the API surface can never
    accidentally leak tokens via the standard list/get endpoints.
    """
    with _conn() as c:
        row = c.execute(
            "SELECT auth_token FROM compute_workers WHERE id = ?", (wid,),
        ).fetchone()
    return row["auth_token"] if row else None


# ---------------------------------------------------------------------------
# Split models (Phase 2 of the compute-pool feature)
# ---------------------------------------------------------------------------
# A split model is a GGUF served by a host-local `llama-server` instance
# with `--rpc <worker>:<port>` flags so the model's layers fan across
# multiple machines. Each row here represents ONE split-model definition
# the user has registered; lifecycle (start/stop) lives in
# `layered_runtime.py` and updates `status` / `last_error` here.
#
# Routing layer reads these rows to know "is this conversation's model a
# split model? if so, send chat to the local llama-server rather than
# Ollama."
# ---------------------------------------------------------------------------

# Allowed values for split_models.status. The lifecycle module enforces
# these transitions; user-facing edits never write status directly.
SPLIT_MODEL_STATUSES = {"stopped", "loading", "running", "error"}

# Default port range for `llama-server`. We pick a default that doesn't
# collide with Ollama (11434) and sits in the unprivileged range. Users
# can override per row if they're already running something on 11500.
SPLIT_MODEL_DEFAULT_PORT = 11500


def create_split_model(
    *,
    label: str,
    gguf_path: str,
    worker_ids: list[str] | None = None,
    llama_port: int = SPLIT_MODEL_DEFAULT_PORT,
    enabled: bool = True,
    mmproj_path: str | None = None,
    draft_gguf_path: str | None = None,
) -> str:
    """Insert a new split-model definition and return its id.

    Validates aggressively up front because a malformed row (empty path,
    non-list worker_ids, out-of-range port) would later break the
    lifecycle module in ways that are harder to diagnose. Status is
    always created as `stopped` — the lifecycle module flips it to
    `loading` / `running` / `error` as it brings llama-server up.

    `draft_gguf_path`, when set, is the on-disk path to a smaller GGUF
    that llama-server runs as a speculative-decoding draft. It must
    share the target model's tokenizer (same family) for accept rates to
    be useful; the picker in `compute_pool.pick_draft_for` enforces this.
    Empty string / None disables speculative decoding for this row.
    """
    lbl = (label or "").strip()
    if not lbl:
        raise ValueError("label is required")
    if len(lbl) > 80:
        raise ValueError("label must be ≤ 80 chars")

    path = (gguf_path or "").strip()
    if not path:
        raise ValueError("gguf_path is required")

    # `worker_ids` must be a list of strings — empty list is allowed
    # (means "run on host alone, no rpc workers"); the lifecycle module
    # then invokes llama-server without any --rpc flags.
    wids = list(worker_ids or [])
    for i, w in enumerate(wids):
        if not isinstance(w, str) or not w.strip():
            raise ValueError(f"worker_ids[{i}] must be a non-empty string")

    port = SPLIT_MODEL_DEFAULT_PORT if llama_port is None else int(llama_port)
    if port < 1 or port > 65535:
        raise ValueError("llama_port must be 1–65535")

    # mmproj + draft are optional — we only validate they're non-empty
    # strings if set.
    mmproj = (mmproj_path or "").strip() or None
    draft = (draft_gguf_path or "").strip() or None

    sid = uuid.uuid4().hex
    now = time.time()
    with _conn() as c:
        c.execute(
            "INSERT INTO split_models ("
            "id, label, gguf_path, mmproj_path, draft_gguf_path, "
            "worker_ids_json, llama_port, enabled, status, "
            "created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'stopped', ?, ?)",
            (
                sid,
                lbl,
                path,
                mmproj,
                draft,
                json.dumps(wids),
                port,
                1 if enabled else 0,
                now, now,
            ),
        )
    return sid


def list_split_models(*, enabled_only: bool = False) -> list[dict]:
    """Return every split-model definition, newest first.

    `enabled_only=True` is what the routing layer uses — disabled rows
    don't influence chat routing even if their llama-server happens to
    still be running.
    """
    sql = "SELECT * FROM split_models"
    if enabled_only:
        sql += " WHERE enabled = 1"
    sql += " ORDER BY created_at DESC"
    with _conn() as c:
        rows = c.execute(sql).fetchall()
    return [_row_to_split_model(r) for r in rows]


def get_split_model(sid: str) -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM split_models WHERE id = ?", (sid,),
        ).fetchone()
    return _row_to_split_model(row) if row else None


def update_split_model(sid: str, **fields: Any) -> dict | None:
    """User-facing patch. Same shape as `update_compute_worker`:
    unknown keys are silently ignored so an old client sending a removed
    field doesn't break, and `status`/`last_error` are NOT updateable
    here — they belong to the lifecycle module."""
    if not get_split_model(sid):
        return None

    allowed = {
        "label", "gguf_path", "mmproj_path", "draft_gguf_path",
        "worker_ids", "llama_port", "enabled",
    }
    cols: list[str] = []
    vals: list[Any] = []
    for k, v in fields.items():
        if k not in allowed:
            continue
        if k == "label":
            lbl = (v or "").strip()
            if not lbl:
                raise ValueError("label cannot be blank")
            if len(lbl) > 80:
                raise ValueError("label must be ≤ 80 chars")
            cols.append("label = ?")
            vals.append(lbl)
        elif k == "gguf_path":
            path = (v or "").strip()
            if not path:
                raise ValueError("gguf_path cannot be blank")
            cols.append("gguf_path = ?")
            vals.append(path)
        elif k == "mmproj_path":
            # Optional — empty string / None means "remove mmproj".
            cleaned = (v or "").strip() or None
            cols.append("mmproj_path = ?")
            vals.append(cleaned)
        elif k == "draft_gguf_path":
            # Optional — empty string / None means "disable speculative
            # decoding for this row" (vanilla single-model serving).
            cleaned = (v or "").strip() or None
            cols.append("draft_gguf_path = ?")
            vals.append(cleaned)
        elif k == "worker_ids":
            wids = list(v or [])
            for i, w in enumerate(wids):
                if not isinstance(w, str) or not w.strip():
                    raise ValueError(f"worker_ids[{i}] must be a non-empty string")
            cols.append("worker_ids_json = ?")
            vals.append(json.dumps(wids))
        elif k == "llama_port":
            port = int(v) if v is not None else SPLIT_MODEL_DEFAULT_PORT
            if port < 1 or port > 65535:
                raise ValueError("llama_port must be 1–65535")
            cols.append("llama_port = ?")
            vals.append(port)
        elif k == "enabled":
            cols.append("enabled = ?")
            vals.append(1 if v else 0)

    if not cols:
        return get_split_model(sid)

    cols.append("updated_at = ?")
    vals.append(time.time())
    vals.append(sid)
    with _conn() as c:
        c.execute(
            f"UPDATE split_models SET {', '.join(cols)} WHERE id = ?",
            tuple(vals),
        )
    return get_split_model(sid)


def update_split_model_status(
    sid: str,
    *,
    status: str | None = None,
    last_error: str | None = None,
) -> None:
    """Internal: lifecycle module calls this when llama-server starts /
    stops / errors. Separated from `update_split_model` so a user PATCH
    can never set the row to a fictitious 'running' state.

    Pass `last_error=""` to clear an error after a successful start;
    `last_error=None` (the default) means "leave it alone".
    """
    cols: list[str] = []
    vals: list[Any] = []
    if status is not None:
        if status not in SPLIT_MODEL_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(SPLIT_MODEL_STATUSES)}"
            )
        cols.append("status = ?")
        vals.append(status)
    if last_error is not None:
        cols.append("last_error = ?")
        vals.append(last_error or None)
    if not cols:
        return
    cols.append("updated_at = ?")
    vals.append(time.time())
    vals.append(sid)
    with _conn() as c:
        c.execute(
            f"UPDATE split_models SET {', '.join(cols)} WHERE id = ?",
            tuple(vals),
        )


def delete_split_model(sid: str) -> int:
    """Remove a split-model row. Caller is responsible for stopping the
    llama-server process first — the data layer doesn't touch runtime
    state."""
    with _conn() as c:
        c.execute("DELETE FROM split_models WHERE id = ?", (sid,))
        return c.total_changes


def _row_to_split_model(row: sqlite3.Row) -> dict:
    try:
        wids = _json_loads(row["worker_ids_json"]) if row["worker_ids_json"] else []
    except Exception:
        wids = []
    # `mmproj_path` and `draft_gguf_path` were added in later migrations
    # — older rows return the columns as NULL, and the columns
    # themselves didn't exist before the migration ran. Use
    # sqlite3.Row.keys() to detect presence so the helper still works
    # against an upgrade-in-progress DB.
    def _safe_col(name: str):
        try:
            if name in row.keys():
                return row[name]
        except (IndexError, KeyError):
            pass
        return None

    return {
        "id": row["id"],
        "label": row["label"],
        "gguf_path": row["gguf_path"],
        "mmproj_path": _safe_col("mmproj_path"),
        "draft_gguf_path": _safe_col("draft_gguf_path"),
        "worker_ids": wids,
        "llama_port": row["llama_port"],
        "enabled": bool(row["enabled"]),
        "status": row["status"],
        "last_error": row["last_error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
