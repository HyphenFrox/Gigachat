# Architecture

Orientation for contributors. The [README](./README.md) covers *what* each
feature does and its security posture; this file covers *how* the pieces fit
together and *where to change what*.

---

## Shape of the system

Three processes, talking over two boundaries:

```
  Browser (React SPA)
    │  HTTP + SSE
    ▼
  FastAPI backend (uvicorn, single worker)       ◄── this is the locus of state
    │  HTTP (Ollama native API)
    ▼
  Ollama server (local LLM runtime)
```

The backend is the only thing that writes durable state. The frontend is a
thin rendering + input layer — close the tab and the only thing lost is the
open SSE connection; the DB and disk are untouched. Ollama is stateless from
our point of view (we resend the whole conversation every turn).

### Directory map

```
backend/            FastAPI app, agent loop, tool implementations
  app.py            HTTP routes (REST + SSE), request validation, auth
  agent.py          run_turn() — the Ollama ↔ tool-loop inner engine
  tools.py          every tool implementation + the dispatcher (~10k lines)
  prompts.py        tool schemas handed to Ollama; system-prompt assembly
  db.py             SQLite schema + CRUD; all persistent state goes here
  ollama_runtime.py Ollama HTTP client + model registry
  mcp.py            MCP server integration (external tool backends)
  tool_prompt_adapter.py   Converts structured tool calls to model dialect
  user_tools_runtime.py    Sandboxed venv execution for user-defined tools
  compute_pool.py   Worker registry, capability probe, intelligent chat router (auto-decides
                    Ollama-on-host vs Ollama-on-worker vs split-via-llama.cpp; ranks nodes
                    by GPU presence + proven VRAM).
  split_runtime.py  Detects / installs the host's llama.cpp binaries (CUDA build for orchestration).
  split_lifecycle.py llama-server process supervision: spawn with `--rpc <worker>:<port>`,
                    wait for /health, stop on model switch, boot reconcile after a crash.
  model_sync.py     LAN-first SCP of Ollama model blobs from host to a worker (saves internet
                    bandwidth on multi-GB pulls). Wired automatically into the chat router.
  auth.py           Session cookies + login/logout
  push.py           Web Push notifications
  sysdetect.py      OS / capability probes (monitor layout, UIA availability, host VRAM)
  server.py         uvicorn entry point
  tests/            pytest suite (conftest uses an isolated temp DB per test)

frontend/src/       React + Vite SPA
  App.jsx           Top-level router, auth gate, theme
  main.jsx          Vite entry
  components/       All UI. ChatView.jsx is the workhorse (~2k lines)
  components/ui/    ShadCN primitives (Sonner toaster lives here)
  lib/              Client-side helpers (fetch wrappers, formatters)

data/               Runtime state (not checked in)
  app.db            Primary SQLite database
  checkpoints/<conv_id>/<stamp>/<hash>.bin    File-edit undo snapshots
  memory/<conv_id>.md                          Per-conversation memory notes
  screenshots/                                 Cached screenshot PNGs
  uploads/                                     User-attached images
```

---

## Turn flow (happy path)

This is what happens between a user pressing Send and the SSE stream closing.
Line numbers are pointers, not contracts — grep the function name if they
drift.

1. **POST** from the browser to `/api/conversations/{cid}/messages`.
   Body carries `content` and optional `images` (paths into `data/uploads/`).
   → `app.py  api_send()`.

2. **Validate & open SSE.** `api_send` 404s on a missing conversation, runs
   `_filter_safe_images()` so a malicious client can't exfiltrate arbitrary
   files by claiming they're images, then returns a `StreamingResponse`
   wrapping the async generator `gen()`.

3. **Agent loop starts.** `gen()` iterates `agent.run_turn(cid, user_text,
   user_images)`. `run_turn` is a thin wrapper that marks the conversation
   `running` in SQLite (so a crash mid-turn is recoverable by the startup
   resumer) and delegates to `_run_turn_impl` which is the actual iteration
   state machine.

4. **Prompt assembly.** `_run_turn_impl` pulls conversation history from
   `db.list_messages`, injects global memories + per-conversation memory,
   assembles the system prompt via `prompts.build_system_prompt`, and picks
   the tool schema list based on the permission mode (`approve_all` /
   `approve_edits` / `read_only`).

5. **Call Ollama.** `_stream_ollama_chat()` opens a streaming HTTP request.
   Tokens arrive as NDJSON; each chunk is yielded upstream as an SSE
   `delta` event so the browser paints progressively.

6. **Tool-call detected.** When the model emits a `tool_calls` array, the
   loop stops streaming text and enters the tool phase. If the structured
   channel is empty but the streamed text contains
   `<tool_call>{...}</tool_call>` tags, `tool_prompt_adapter
   .parse_tool_calls_from_text()` recovers the call before persistence —
   covers both prompt-space mode (stub-template models) and the
   misbehaving-native-tool-aware case (gemma4:e4b, smaller Qwens).

7. **Approval gate (optional).** In `approve_edits` mode, write-class tools
   emit a `tool_request` SSE event and the loop blocks on an async Future
   that the browser resolves via `POST /api/conversations/{cid}/tool_decision`.
   `read_only` mode refuses write-class tools before the request even reaches
   the dispatcher.

8. **Dispatch.** `tools.dispatch(name, args, cwd, conv_id, model)`.
   Order of precedence:
   1. `resolve_tool_alias()` — canonicalize common misspellings
   2. MCP tools (prefix `mcp__`) route to `mcp.dispatch_tool`
   3. User-defined tools (`db.get_user_tool_by_name`) execute in a
      sandboxed venv
   4. Built-in tools look up in `TOOL_REGISTRY` and call the Python function
   Each tool returns a JSON-able dict `{ok, ...}`. Errors are reported as
   `{ok: False, error: ...}` not exceptions.

9. **Tool result → Ollama.** The result dict is serialized into a tool-role
   message, appended to history, and the loop goes back to step 5. When the
   model stops emitting tool calls and just produces text, we persist the
   final assistant message (`db.add_message`) and close the SSE stream.

10. **Finally block.** `run_turn` flips the conversation back to `idle` (or
    `error` if an exception escaped) so the sidebar un-busies. A stale
    `running` row more than 60 s old is force-flipped by the startup
    watchdog — that's what makes browser-refresh-mid-turn survivable.

---

## Load-bearing invariants

Each of these is relied on by multiple places. Break one and something
non-obvious stops working.

- **One SSE stream per conversation.** `agent.request_stop(cid)` sets a flag
  that the loop polls between Ollama chunks and before tool dispatch; two
  concurrent turns for the same `cid` would trample the flag and the Stop
  button stops working. The frontend enforces this client-side; the backend
  does not reject it, so don't add a second caller.

- **Element IDs are process-scoped (`_ELEMENT_CACHE`).** Minted by
  `inspect_window` or `screenshot(with_elements=True)`, evicted LRU at 5000
  entries. IDs don't survive `uvicorn` restart and don't survive a window
  relayout. The cache lives in module globals protected by `_ELEMENT_LOCK`;
  this assumes **single-worker uvicorn**. Don't run `--workers 2` — the
  cache would fragment silently and element ids minted by one worker would
  miss in the other.

- **Tool outputs are never trusted as instructions.** Anything flowing back
  from the web, UIA-accessible window titles, OCR text, file contents, etc.
  is quoted with `!r` or JSON-serialized so embedded newlines can't forge a
  fake instruction line. This is the defense against prompt-injection.
  When adding a new tool, follow the pattern: structured JSON fields good,
  free-form text dumps require escaping.

- **All persistent state goes through `db.py`.** No other module opens the
  SQLite file directly. This keeps schema migrations in one place (look for
  `CREATE TABLE IF NOT EXISTS` + column-check `ALTER TABLE` blocks near
  `db.init()`). Ad-hoc sqlite access from elsewhere would miss those
  migrations on startup and read from an outdated schema.

- **Write-class tools respect permission mode.** The classifier lives in
  `tools.classify_tool(name)` (backed by `TOOL_CATEGORIES` plus a fallback
  rule that tags MCP tools as write-class because their side effects are
  unknown). The agent loop reads that category to decide whether to refuse
  the call (`read_only` mode), pause for approval (`approve_edits`), or
  run silently (`allow_all`). A new tool that happens to write to disk or
  send input must be added to `TOOL_CATEGORIES` or `read_only` mode leaks.

- **File tools follow `bash`'s persisted cwd.** `tools._resolve(cwd, path,
  conv_id)` consults the per-conversation `bash_cwd` for relative paths
  whenever a `conv_id` is threaded through. Absolute paths still win.
  Every dispatcher call into a file tool passes `conv_id`; tests for the
  invariant live in `backend/tests/test_tools_cwd_follows_bash.py`. Break
  this and the model's mental model diverges from reality the moment it
  runs `cd subdir` in bash.

---

## Data at rest

### SQLite (`data/app.db`)

A handful of tables defined in `backend/db.py`'s schema block:

| Table                 | Purpose                                                                   |
|-----------------------|---------------------------------------------------------------------------|
| `conversations`       | Per-conversation metadata: title, state (idle/running/error), budgets, persisted `bash_cwd` |
| `messages`            | Append-only chat history (user / assistant / tool / system roles)         |
| `message_embeddings`  | Optional vector embeddings for recall search                              |
| `queued_inputs`       | Mid-turn message queue (persists across crashes)                          |
| `scheduled_tasks`     | `schedule_task` / `start_loop` / `schedule_wakeup` queue + next-run timestamps |
| `doc_chunks`          | RAG chunks: `doc_index` directories, codebase auto-index, and `docs:` URL crawls share this table |
| `codebase_indexes`    | One row per cwd — status, file/chunk counts, last-run timestamp           |
| `doc_urls`            | URL-crawl status for Settings → Docs                                      |
| `hooks`               | Lifecycle / tool hooks (user-configured)                                  |
| `global_memories`     | Notes injected into every conversation's system prompt                    |
| `push_subscriptions`  | Web Push endpoints                                                        |
| `mcp_servers`         | Registered MCP server configs                                             |
| `user_settings`       | Single-row key/value blob for app settings                                |
| `user_tools`          | User-defined tools (code + schema)                                        |
| `secrets`             | Named secrets (plaintext at rest; threat model is single-user local)       |
| `side_tasks`          | `spawn_task` chip queue under the composer                                |
| `worktrees`           | `create_worktree` / `list_worktrees` / `remove_worktree` registry         |

### Files on disk

- `data/checkpoints/<conv_id>/<YYYYMMDDTHHMMSS_micro_hash>/…` — Each
  `write_file` / `replace_in_file` call that overwrites an existing file
  first snapshots the old bytes here. `restore_checkpoint(conv_id, stamp)`
  walks this tree.
- `data/memory/<conv_id>.md` — Per-conversation scratchpad.
  `memory_put` / `memory_get` / `memory_delete` tools own this.
  Global memories are in SQLite (`global_memories` table), not here.
- `data/screenshots/screenshot-<timestamp>.png` — `screenshot` tool output;
  served to the browser via `/api/screenshots/{filename}`.
- `data/uploads/<uuid>.<ext>` — User-attached images; served via
  `/api/uploads/{filename}` after path validation in `_filter_safe_images`.

---

## Where to change what

| Goal                                         | Start here                                 |
|----------------------------------------------|--------------------------------------------|
| Add a new built-in tool                      | `tools.py` (impl) + `prompts.py` (schema)  |
| Change how a tool result is formatted        | `tool_prompt_adapter.py`                   |
| Add a new SSE event type                     | `agent.py` (emit) + frontend `ChatView.jsx`|
| Add a new REST endpoint                      | `app.py`                                   |
| New SQLite table or column                   | `db.py` (`init()` + CRUD functions)        |
| New approval tier or permission mode         | `tools.py` (`TOOL_CATEGORIES` + `classify_tool`) and the gate in `agent.py` |
| Change the system prompt                     | `prompts.py` (`build_system_prompt`)       |
| Add a new React panel                        | `frontend/src/components/`                 |
| Wire a new MCP server                        | `mcp.py` + `mcp_servers` table             |
| Tweak the chat-routing policy (host vs worker vs split) | `compute_pool.py` (`route_chat_for`, `pick_chat_target`, `_capability_score`, `_host_capability_score`) |
| Adjust the per-worker probe (new capability hint) | `compute_pool._probe_one` (returns dict merged into `capabilities_json`) |
| Add a new `llama-server` CLI flag            | `split_lifecycle._build_command`           |

---

## Compute pool

Two routing tiers; both live in `compute_pool.py` and are invoked from
`agent.run_turn` early in each turn.

**Whole-request routing (Phase 1).** A single chat / embedding /
parallel-subagent call goes to ONE machine. `pick_chat_target` (and the
embed/subagent siblings) ranks workers by `_capability_score` — a
`(gpu_present, max_vram_seen_bytes, last_seen)` tuple — and compares
the strongest worker to `_host_capability_score()` derived from
`sysdetect`. Workers only win when STRICTLY more capable than host;
ties go local (no LAN hop, KV cache stays warm). Workers ineligible
because they're missing the model auto-trigger an SCP from host
(`_maybe_kickoff_lan_sync`) when `ssh_host` is configured — fully
non-blocking, future turns route to the now-loaded worker.

**Layer-split routing (Phase 2).** When a model exceeds the strongest
single node's capacity, `route_chat_for` spawns
`llama-server --rpc <worker>:50052,...` on host port 11500 via
`split_lifecycle.start`. llama.cpp natively distributes layers across
host VRAM/RAM + each worker's `rpc-server` (which contributes its own
GPU + CPU + RAM). Workers don't need the model installed for the split
path — bytes stream over RPC from the host's local GGUF blob. Process
state is tracked in the `split_models` table; rows are auto-created
keyed by model name and auto-stopped when the user switches to a
different model (one big model hot at a time, finite VRAM).

The `split_models` table is internal — there's no user-facing CRUD UI.
Users just pick a model in the chat picker; the router decides.

---

## Frontend error-handling convention

Every `catch` block must do one of three things:

1. **Toast.** The default. `import { toast } from 'sonner'` and call
   `toast.error(label, { description: e.message })` so the user sees
   what broke. The Sonner toaster is mounted once at the root
   (`components/ui/sonner.jsx`) and picks up calls from anywhere.

2. **Deliberately swallow, with a comment.** Some failures are
   genuinely non-fatal — an auto-refresh that the next SSE event will
   re-populate, a typeahead miss that shouldn't spam the user while
   they're typing, a cosmetic `URL.revokeObjectURL` that's best-effort
   by design. These look like:
   ```js
   } catch {
     /* non-fatal — chips will repopulate on the next stream event */
   }
   ```
   The comment is not decoration; it's the review signal that this
   swallow was thought through.

3. **Fall back to a safe state.** A few places (e.g. the auth-status
   probe in `App.jsx`) treat a failed fetch as a specific signal
   ("assume unauthenticated, show login"). These document the fallback
   reasoning in a comment too.

`console.error` is reserved for contexts that have no toaster yet —
today, only service-worker registration in `lib/pwa.js`. Adding a new
`console.*` call to a component is almost certainly a mistake; use
`toast.error`.

---

## Deployment assumptions

- **Single machine, single user, single worker.** The app is a localhost
  desktop tool; it is not designed to be exposed on the public internet or
  served to multiple users. Computer-use tools drive the real mouse. In-
  memory caches (element IDs, SSE stop flags, screenshot signatures,
  click-marker globals) are process-local and don't survive a restart.
- **Hot reload is on by default.** `uvicorn --reload` watches `backend/`.
  Long-running tests that hit live HTTP must either disable the reloader
  or tolerate the mid-run restart. In-process tests (`backend/tests/`)
  don't care.
- **Ollama is a soft dependency.** The backend starts fine without Ollama
  running; chat turns fail at the `chat_stream` call with a clear error.
  Tool tests that don't hit Ollama run cleanly.
