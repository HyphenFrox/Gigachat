# Tools the model can run

The agent uses Ollama's native function-calling — **any modern Ollama model with a tool/function-calling chat template plugs in the same way** (Gemma 4, Llama 3.1+, Qwen 2.5+, Mistral, etc.).

## File and shell

`bash`, `bash_bg` (long-running background shells with `bash_output` / `kill_shell`), `read_file`, `write_file`, `edit_file` (surgical old→new replacement with ambiguity rejection + `replace_all`), `list_dir`, `grep` (ripgrep with Python fallback), `glob` (`**/*.py`-style matching), `python_exec` (short Python snippet in an isolated subprocess — stdlib-only, 30 s wall-clock, 20 000-char output cap).

**File tools follow `bash`'s persisted cwd.** `bash` remembers where it `cd`'d to across calls, and after `cd myapp`, `write_file({"path": "src/App.jsx", ...})` lands inside `myapp/src/`, not back at the workspace root. Use absolute paths to bypass this when you need the workspace root explicitly.

## Documents

`read_doc` extracts readable text from PDF / .docx / .xlsx (pymupdf / python-docx / openpyxl). PDFs accept page-range selectors (`1-5,7,9`); xlsx accepts explicit sheet names. For plain text use `read_file`.

## Computer use

`screenshot`, `screenshot_window`, `list_monitors`, `list_windows`, `computer_click`, `computer_drag`, `computer_type`, `computer_key`, `computer_scroll`, `computer_mouse_move`, `click_element`, `click_element_id`, `focus_window`, `open_app`, `window_action`, `window_bounds`, `inspect_window`, `ocr_screenshot`, `ui_wait`, `computer_batch`.

The model can literally see your desktop (requires a multimodal Ollama model such as `gemma4`, `llava`, `qwen2.5-vl`) and drive your mouse and keyboard — same surface as Anthropic's computer use. Highlights:

- **Coordinate grid overlay** — every screenshot ships with a yellow 100-px grid (labels every 200 px) so a small vision model can name targets by nearest cell instead of eyeballing pixels.
- **Multi-monitor + window-cropped capture** — `list_monitors` + the `monitor` param target any attached display; `screenshot_window({"name": "Chrome"})` crops to one window's bounding rect (4-10× cheaper in vision tokens).
- **Accessibility-tree clicks** — `click_element({"name": "Guest mode"})` clicks by accessible name (Windows UI Automation, the same tech screen readers use), sidestepping pixel localization. `inspect_window` dumps the tree, mints stable `[elN]` IDs, and returns a Set-of-Mark annotated PNG with each ID painted on its control. `click_element_id({"id": "el7"})` clicks that exact control with no fuzzy matching — ideal when two buttons share a name.
- **Bounded waits** — `ui_wait` blocks until a state appears (`window` / `element` / `text` / `pixel_change` / `window_gone` / `element_enabled`), capped at 30 s, polled at ~250 ms — so the agent stops screenshot-spamming a slow load.
- **Batched primitives** — `computer_batch` runs an allowlisted sequence (move/click/drag/type/key/scroll/wait/focus/window/click_element/click_element_id/open_app/ocr) in one call, capped at 20 steps, single end-of-batch screenshot.

## Browser automation (Chrome DevTools Protocol)

`browser_tabs`, `browser_goto`, `browser_click`, `browser_type`, `browser_text`, `browser_eval`. Launch Chrome with `--remote-debugging-port=9222` (e.g. `open_app({"name": "chrome", "args": ["--remote-debugging-port=9222"]})`) and the agent drives it via CSS selectors — vastly more reliable than pixel clicks for web work. `browser_goto` enforces http/https schemes; `browser_eval` runs arbitrary JS (escape hatch, flagged in the schema).

## Web access

`web_search` (DuckDuckGo, no API key), `fetch_url` (downloads + cleans via [trafilatura](https://github.com/adbar/trafilatura)), `http_request` (full HTTP client — method / headers / query / body — for calling real APIs).

`http_request` integrates with the **Secrets** store: drop `{{secret:NAME}}` into any header or body and the backend substitutes the value just before the wire, never exposing the raw credential in the transcript. SSRF guard same as `fetch_url`; opt into LAN targets with `allow_private: true`.

## Local indexing & search

- **`doc_index` / `doc_search`** — walks a directory, chunks every matching file, embeds via Ollama (`nomic-embed-text` by default), stores vectors in SQLite. `doc_search` returns top-k by cosine similarity. Re-indexing is idempotent.
- **Codebase auto-index** — when a conversation's `cwd` is set, the backend kicks off a background index of that root: gitignore-aware on git repos (`git ls-files -co --exclude-standard -z`), rglob + noise blacklist otherwise. Live status chip in the chat header tracks `pending` / `indexing` / `ready` / `error` with file + chunk counts; click to reindex. The agent calls `codebase_search(query, top_k)` instead of grepping three times.
- **Docs by URL** — Settings → **Docs** crawls a public docs URL breadth-first (same-origin, capped at 100 pages, SSRF-guarded), extracts clean text, embeds, stores. `docs_search(query, top_k, url_prefix?)` pulls relevant passages — optionally scoped to one site — so "how do I pass auth headers in httpx?" hits real docs instead of stale training data.

## Memory and coordination

- **Long-term memory** — `remember` / `forget` save and prune durable facts. `scope="conversation"` (default) writes to `data/memory/<conv_id>.md` so the note survives compaction *inside that thread*. `scope="global"` writes to a SQLite-backed `global_memories` table injected into the system prompt of **every** conversation (and every subagent). Settings → **Memories** is the human-friendly editor for the global store; the chat header's "⋯" → **Conversation memory** edits the per-chat file.
- **Subagents** — `delegate` spawns one isolated subagent for a scoped sub-task; `delegate_parallel` fans out 2-6 independent subagents concurrently (one labelled result block per task, partial failures inline). Each call accepts a `type`: `general` (full toolbelt), `explorer` (read-only fast recon), `architect` (read-only planner), `reviewer` (read-only critic). Read-only types have every write-class tool stripped from their palette.
- **`todo_write`** — structured task list rendered in a side panel.
- **`ask_user_question`** — pauses the turn and renders 2-6 buttons under the composer; control returns only when you click. Subagents cannot call this — only the top-level loop can prompt the user.
- **`spawn_task`** — flag a drive-by issue (stale README line, dead config option, missing test) as a chip under the composer without derailing the current turn. Click **Open** to spin a fresh conversation seeded with the stored prompt.

## Scheduling and loops

- **`schedule_task`** — queue a prompt to run autonomously in a new conversation either at an ISO datetime (`run_at`) or on a recurring `every_minutes` interval. A 30-second-poll daemon fires due tasks with auto-approve enabled. `list_scheduled_tasks` / `cancel_scheduled_task` manage the queue.
- **`schedule_wakeup(delay_seconds, note)`** — schedule **this** conversation to resume itself later (60 s – 1 h). Use for "check the build in 10 minutes" without holding a streaming connection open. Push notification when the follow-up turn lands.
- **`start_loop(goal, interval_seconds)`** — turn the current chat into a self-driving worker. Every `interval_seconds` (60 s – 1 h) the daemon re-appends `goal` as a user turn. Idempotent — calling again replaces the existing loop. `stop_loop()` ends it; emerald banner above the composer shows live countdown + truncated goal.
- **`monitor`** — block on a file, HTTP URL, or bash command until a condition flips (`exists` / `missing` / `contains:` / `not_contains:` / `changed` / `status:` / `exit_code:` / `regex:`). Saves "run a tool, ask agent to retry in N seconds" loops. URL targets reuse the SSRF guard.

## Sandboxed containers (Docker)

`docker_run`, `docker_run_bg`, `docker_logs`, `docker_exec`, `docker_stop`, `docker_list`, `docker_pull` — run **any language or piece of software** inside an isolated container. Defaults: `--rm`, `--security-opt=no-new-privileges`, 512 MB memory, 1 CPU, conversation cwd mounted **read-only** at `/workspace`, bridge networking (inbound blocked unless explicitly published). Opt into `mount_mode: "rw"`, `network: "none"`, or published ports as needed. Image name is allowlist-validated; docker CLI invoked with argv list (no `shell=True`). Container management is scoped to containers Gigachat itself started (`gigachat_*` name prefix), so the agent can't tamper with your other containers.

## Universal API connector (OpenAPI / Swagger)

Register any REST API by its OpenAPI spec, then call any of its endpoints without writing a per-endpoint tool.

- **`openapi_load(spec_url, api_id, [auth_scheme, auth_secret_name, default_headers])`** — fetches and registers a spec. Auth schemes: `bearer` / `apikey` / `basic`, the credential is stored in the secrets table by name and substituted at call time so the raw value never reaches the model.
- **`openapi_list`** / **`openapi_list_ops(api_id, [query])`** / **`openapi_describe(api_id, operation_id)`** — discovery surface so the agent can browse big specs (capped at 500 ops per spec, 1 MB body).
- **`openapi_call(api_id, operation_id, args)`** — invoke. Path / query / header / body parameters dispatched per the spec; remaining args become the JSON body.
- All calls go through the existing `http_request`, so SSRF guard, secret redaction, and audit logging apply unchanged.

## Audio (transcription)

- **`transcribe_audio(path, [model_name, language])`** — local Whisper via faster-whisper. Returns full transcript + per-segment timestamps. Models: tiny / base / small / medium / large-v3. VAD trims silence. 500 MB file cap.

## SSH (remote machines)

- **`ssh_exec(host, command, [user, port, password_secret])`** — run a remote command, return combined stdout+stderr + exit code. Auth: stored secret, OR system ssh-agent / ~/.ssh keys.
- **`ssh_put` / `ssh_get`** — SCP a file up or down; 100 MB cap.

## Email

- **`email_send(smtp_host, smtp_port, user, password_secret, to, subject, body, ...)`** — stdlib smtplib, SSL by default. Use an APP password for Gmail / Outlook / Fastmail.
- **`email_read(imap_host, user, password_secret, [folder, limit, unseen_only])`** — stdlib imaplib, returns recent messages with ~2 KB body previews.

## Notifications (Slack / Discord / Telegram / generic)

- **`notify(channel, message, [title])`** — webhook URL lives in the secrets table under `WEBHOOK_<CHANNEL>` (uppercased). Auto-detects Slack, Discord, and Telegram Bot API URLs and adapts the JSON shape; generic targets receive `{title, message}`. The URL itself never appears in the conversation, only the channel alias.

## Smart home (Home Assistant)

- **`home_assistant_call(action, [base_url, entity_id, domain, service, service_data])`** — list_entities / get_state / call_service over the local HA REST API. Long-lived access token in secrets under `HOME_ASSISTANT_TOKEN`.

## Skill library (procedural memory)

- **`save_skill(name, description, body, [tags])`** — bank a named playbook the agent figured out. Distinct from `remember` (facts): skills are how-to procedures.
- **`find_skill(query)`** — search by substring; **`recall_skill(name)`** returns the full body (and bumps usage stats).
- **`list_skills` / `update_skill` / `delete_skill`** — round out the surface. Browse / prune in Settings → **Skills** later (API at `/api/skills`).

## Multi-agent orchestration

- **`orchestrate(task, [skip_review])`** — one-call ARCHITECT (plan) → GENERAL (execute) → REVIEWER (verify) pipeline. Same chat model in all three roles; diversity comes from the SUBAGENT_TYPES prompt overlays. For complex tasks where managing the sequencing yourself with raw `delegate` would be wasteful.

## Event-driven triggers

- **Webhooks** — `POST /api/webhooks` mints a `/webhook/<token>` URL; any service hitting that URL fires an agent turn against the configured target conversation, with the request body as the user message (or templated via `{body}`).
- **File watchers** — `POST /api/file-watchers` registers a path + glob; a 2-second polling daemon coalesces created / modified / deleted events into one debounced turn.

## Other tools

- **`clipboard_read` / `clipboard_write`** — share small bits of text with the desktop without typing.
- **`create_worktree(branch, base_ref)`** — `git worktree add` on a throwaway branch so the agent can do risky edits without touching your working tree. Pair with `list_worktrees` / `remove_worktree(id)`. Branch and base_ref regex-validated.
- **MCP servers** — connect external [Model Context Protocol](https://modelcontextprotocol.io) servers over stdio. Each server runs as a local subprocess and every advertised tool is auto-merged into the palette as `mcp__<server>__<tool>`. Settings → **MCP** for CRUD; live tool counts + stderr tail for troubleshooting. 20 s handshake timeout, 120 s tool-call ceiling.
- **User-defined Python tools** — Settings → **Tools** lets *you* register Python snippets that become first-class entries in the tool palette. Each has a `def run(args)` entry point, optional pip dependencies (PEP 508 subset, blocklist on pip/setuptools/wheel), JSON-schema parameters, and a stored `category` + `timeout_seconds` the model cannot override. **The LLM has no route to create, edit, or delete these** — deliberate safety boundary against self-extension. Kill switch: `GIGACHAT_DISABLE_USER_TOOLS=1`.

## Audit log

Every tool call across every conversation is recorded in the `audit_log` table — tool name, category, args, result summary (capped at 2 KB), ok/duration. `GET /api/audit-log` exposes a filterable read-only view: by conversation, by tool name, since-timestamp. Useful for "what did the agent do today" reviews and for post-mortem of long-running tasks.
