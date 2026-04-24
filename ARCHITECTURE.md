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
  auth.py           Session cookies + login/logout
  push.py           Web Push notifications
  sysdetect.py      OS / capability probes (monitor layout, UIA availability)
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
   → `app.py:1671  api_send()`.

2. **Validate & open SSE.** `api_send` 404s on a missing conversation, runs
   `_filter_safe_images()` so a malicious client can't exfiltrate arbitrary
   files by claiming they're images, then returns a `StreamingResponse`
   wrapping the async generator `gen()`.

3. **Agent loop starts.** `gen()` iterates `agent.run_turn(cid, user_text,
   user_images)` → `agent.py:1248`. `run_turn` is a thin wrapper that marks
   the conversation `running` in SQLite (so a crash mid-turn is recoverable
   by the startup resumer) and delegates to `_run_turn_impl` which is the
   actual iteration state machine.

4. **Prompt assembly.** `_run_turn_impl` pulls conversation history from
   `db.list_messages`, injects global memories + per-conversation memory,
   assembles the system prompt via `prompts.build_system_prompt`, and picks
   the tool schema list based on the permission mode (`approve_all` /
   `approve_edits` / `read_only`).

5. **Call Ollama.** `ollama_runtime.chat_stream()` opens a streaming HTTP
   request. Tokens arrive as NDJSON; each chunk is yielded upstream as an
   SSE `assistant_delta` event so the browser paints progressively.

6. **Tool-call detected.** When the model emits a `tool_calls` array (or the
   dialect-specific equivalent handled by `tool_prompt_adapter`), the loop
   stops streaming text and enters the tool phase.

7. **Approval gate (optional).** In `approve_edits` mode, write-class tools
   emit a `tool_request` SSE event and the loop blocks on an async Future
   that the browser resolves via `POST /api/conversations/{cid}/tool_decision`.
   `read_only` mode refuses write-class tools before the request even reaches
   the dispatcher.

8. **Dispatch.** `tools.dispatch(name, args, cwd, conv_id, model)` →
   `tools.py:9825`. Order of precedence:
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

- **Write-class tools respect permission mode.** The gating lives in
  `agent.py` (see the `_tool_is_write_class` check inside `_run_turn_impl`),
  not in the tool itself. A new tool that happens to write to disk or send
  input must be classified on that allowlist or `read_only` mode leaks.

---

## Data at rest

### SQLite (`data/app.db`)

13 tables, all defined in `backend/db.py:54-256`:

| Table                 | Purpose                                                                   |
|-----------------------|---------------------------------------------------------------------------|
| `conversations`       | Per-conversation metadata: title, state (idle/running/error), budgets     |
| `messages`            | Append-only chat history (user / assistant / tool / system roles)         |
| `message_embeddings`  | Optional vector embeddings for recall search                              |
| `queued_inputs`       | Mid-turn message queue (persists across crashes)                          |
| `scheduled_tasks`     | `schedule_task` tool state + next-run timestamps                          |
| `doc_chunks`          | RAG chunks from `add_doc` tool                                            |
| `hooks`               | Lifecycle / tool hooks (user-configured)                                  |
| `global_memories`     | Notes injected into every conversation's system prompt                    |
| `push_subscriptions`  | Web Push endpoints                                                        |
| `mcp_servers`         | Registered MCP server configs                                             |
| `user_settings`       | Single-row key/value blob for app settings                                |
| `user_tools`          | User-defined tools (code + schema)                                        |
| `secrets`             | Named secrets (encrypted at rest, exposed to user tools on demand)        |
| `side_tasks`          | Background subagent bookkeeping (`delegate_parallel`)                     |

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
| New approval tier or permission mode         | `agent.py` (`_tool_is_write_class` + gate) |
| Change the system prompt                     | `prompts.py` (`build_system_prompt`)       |
| Add a new React panel                        | `frontend/src/components/`                 |
| Wire a new MCP server                        | `mcp.py` + `mcp_servers` table             |

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
