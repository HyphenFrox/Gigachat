<p align="center">
  <img src="frontend/src/assets/gigachat-logo.jpg" alt="Gigachat logo" width="180" />
</p>

# Gigachat

A self-hosted web app that turns **any locally-running Ollama model** (Gemma, Llama, Qwen, DeepSeek, Mistral — anything with function-calling) into a Claude-Code-style coding assistant: chat with conversation history, plus tools to run shell commands, read and write files, drive your desktop, search the web, and more — with a per-conversation permission gate on every tool call. Optional password auth + Tailscale-aware bind host for remote access.

```
┌────────────────┐          ┌─────────────────┐          ┌──────────────┐
│  React + Vite  │  SSE     │  FastAPI        │  chat    │   Ollama     │
│  + shadcn/ui   │◄────────►│  agent loop     │◄────────►│   any model  │
│  (port 5173)   │  /api/*  │  (port 8000)    │  :11434  │              │
└────────────────┘          └─────────────────┘          └──────────────┘
```

> ⚠ **The app's capability is heavily influenced by the model you run.** A bigger / better model gives you sharper reasoning, more reliable tool use, and longer plans; a small model will sometimes drop tool calls or hallucinate paths. If something feels broken, try a larger model first.

> **Contributing or learning the codebase?** Read [ARCHITECTURE.md](./ARCHITECTURE.md) for the turn flow, load-bearing invariants, and where to change what.

---

## Quickstart

```powershell
# 1. Install Ollama and a function-calling model.
ollama pull gemma4:e4b

# 2. Install Python + frontend deps. From the project root (Gigachat/):
python -m pip install -r backend/requirements.txt
cd frontend && npm install && cd ..

# 3. Run the dev servers (two console windows). Visit http://localhost:5173.
.\dev.bat
```

That's it for solo loopback use. Production build (`build.bat` then `start.bat`) and remote access (Tailscale or reverse proxy) are covered further down.

---

## What it does

- **Chat with full history**, persisted to SQLite — search, pin, tag, group by project, edit-and-regenerate the last user message.
- **Stream tokens live** over Server-Sent Events; queue follow-up messages while a turn is in flight without locking the composer.
- **Run real tools** (shell, file edit, screenshot, click, browser, web search, …) with a per-conversation permission mode that gates every write-class call.
- **Watch the reasoning** — a desktop right-strip shows the active tool, its args, and a pulsing "Thinking" card so you don't have to scroll the transcript to see what's happening.
- **Survive crashes** — every conversation has an `idle/running/error` state; the startup resumer either re-enters interrupted turns or flips state back to idle.
- **Stay readable on long sessions** — auto-compaction, proactive output aging, screenshot image aging, and a token-usage gauge in the header.

---

## Tools the model can run

The agent uses Ollama's native function-calling — **any modern Ollama model with a tool/function-calling chat template plugs in the same way** (Gemma 4, Llama 3.1+, Qwen 2.5+, Mistral, etc.). Tools are grouped below.

### File and shell

`bash`, `bash_bg` (long-running background shells with `bash_output` / `kill_shell`), `read_file`, `write_file`, `edit_file` (surgical old→new replacement with ambiguity rejection + `replace_all`), `list_dir`, `grep` (ripgrep with Python fallback), `glob` (`**/*.py`-style matching), `python_exec` (short Python snippet in an isolated subprocess — stdlib-only, 30 s wall-clock, 20 000-char output cap).

**File tools follow `bash`'s persisted cwd.** `bash` remembers where it `cd`'d to across calls, and after `cd stock-tracker`, `write_file({"path": "src/App.jsx", ...})` lands inside `stock-tracker/src/`, not back at the workspace root. Use absolute paths to bypass this when you need the workspace root explicitly.

### Documents

`read_doc` extracts readable text from PDF / .docx / .xlsx (pymupdf / python-docx / openpyxl). PDFs accept page-range selectors (`1-5,7,9`); xlsx accepts explicit sheet names. For plain text use `read_file`.

### Computer use

`screenshot`, `screenshot_window`, `list_monitors`, `list_windows`, `computer_click`, `computer_drag`, `computer_type`, `computer_key`, `computer_scroll`, `computer_mouse_move`, `click_element`, `click_element_id`, `focus_window`, `open_app`, `window_action`, `window_bounds`, `inspect_window`, `ocr_screenshot`, `ui_wait`, `computer_batch`.

The model can literally see your desktop (requires a multimodal Ollama model such as `gemma4`, `llava`, `qwen2.5-vl`) and drive your mouse and keyboard — same surface as Anthropic's computer use. Highlights:

- **Coordinate grid overlay** — every screenshot ships with a yellow 100-px grid (labels every 200 px) so a small vision model can name targets by nearest cell instead of eyeballing pixels.
- **Multi-monitor + window-cropped capture** — `list_monitors` + the `monitor` param target any attached display; `screenshot_window({"name": "Chrome"})` crops to one window's bounding rect (4-10× cheaper in vision tokens).
- **Accessibility-tree clicks** — `click_element({"name": "Guest mode"})` clicks by accessible name (Windows UI Automation, the same tech screen readers use), sidestepping pixel localization. `inspect_window` dumps the tree, mints stable `[elN]` IDs, and returns a Set-of-Mark annotated PNG with each ID painted on its control. `click_element_id({"id": "el7"})` clicks that exact control with no fuzzy matching — ideal when two buttons share a name.
- **Bounded waits** — `ui_wait` blocks until a state appears (`window` / `element` / `text` / `pixel_change` / `window_gone` / `element_enabled`), capped at 30 s, polled at ~250 ms — so the agent stops screenshot-spamming a slow load.
- **Batched primitives** — `computer_batch` runs an allowlisted sequence (move/click/drag/type/key/scroll/wait/focus/window/click_element/click_element_id/open_app/ocr) in one call, capped at 20 steps, single end-of-batch screenshot.

### Browser automation (Chrome DevTools Protocol)

`browser_tabs`, `browser_goto`, `browser_click`, `browser_type`, `browser_text`, `browser_eval`. Launch Chrome with `--remote-debugging-port=9222` (e.g. `open_app({"name": "chrome", "args": ["--remote-debugging-port=9222"]})`) and the agent drives it via CSS selectors — vastly more reliable than pixel clicks for web work. `browser_goto` enforces http/https schemes; `browser_eval` runs arbitrary JS (escape hatch, flagged in the schema).

### Web access

`web_search` (DuckDuckGo, no API key), `fetch_url` (downloads + cleans via [trafilatura](https://github.com/adbar/trafilatura)), `http_request` (full HTTP client — method / headers / query / body — for calling real APIs).

`http_request` integrates with the **Secrets** store: drop `{{secret:NAME}}` into any header or body and the backend substitutes the value just before the wire, never exposing the raw credential in the transcript. SSRF guard same as `fetch_url`; opt into LAN targets with `allow_private: true`.

### Local indexing & search

- **`doc_index` / `doc_search`** — walks a directory, chunks every matching file, embeds via Ollama (`nomic-embed-text` by default), stores vectors in SQLite. `doc_search` returns top-k by cosine similarity. Re-indexing is idempotent.
- **Codebase auto-index** — when a conversation's `cwd` is set, the backend kicks off a background index of that root: gitignore-aware on git repos (`git ls-files -co --exclude-standard -z`), rglob + noise blacklist otherwise. Live status chip in the chat header tracks `pending` / `indexing` / `ready` / `error` with file + chunk counts; click to reindex. The agent calls `codebase_search(query, top_k)` instead of grepping three times.
- **Docs by URL** — Settings → **Docs** crawls a public docs URL breadth-first (same-origin, capped at 100 pages, SSRF-guarded), extracts clean text, embeds, stores. `docs_search(query, top_k, url_prefix?)` pulls relevant passages — optionally scoped to one site — so "how do I pass auth headers in httpx?" hits real docs instead of stale training data.

### Memory and coordination

- **Long-term memory** — `remember` / `forget` save and prune durable facts. `scope="conversation"` (default) writes to `data/memory/<conv_id>.md` so the note survives compaction *inside that thread*. `scope="global"` writes to a SQLite-backed `global_memories` table injected into the system prompt of **every** conversation (and every subagent). Settings → **Memories** is the human-friendly editor for the global store; the chat header's "⋯" → **Conversation memory** edits the per-chat file.
- **Subagents** — `delegate` spawns one isolated subagent for a scoped sub-task; `delegate_parallel` fans out 2-6 independent subagents concurrently (one labelled result block per task, partial failures inline). Each call accepts a `type`: `general` (full toolbelt), `explorer` (read-only fast recon), `architect` (read-only planner), `reviewer` (read-only critic). Read-only types have every write-class tool stripped from their palette.
- **`todo_write`** — structured task list rendered in a side panel.
- **`ask_user_question`** — pauses the turn and renders 2-6 buttons under the composer; control returns only when you click. Subagents cannot call this — only the top-level loop can prompt the user.
- **`spawn_task`** — flag a drive-by issue (stale README line, dead config option, missing test) as a chip under the composer without derailing the current turn. Click **Open** to spin a fresh conversation seeded with the stored prompt.

### Scheduling and loops

- **`schedule_task`** — queue a prompt to run autonomously in a new conversation either at an ISO datetime (`run_at`) or on a recurring `every_minutes` interval. A 30-second-poll daemon fires due tasks with auto-approve enabled. `list_scheduled_tasks` / `cancel_scheduled_task` manage the queue.
- **`schedule_wakeup(delay_seconds, note)`** — schedule **this** conversation to resume itself later (60 s – 1 h). Use for "check the build in 10 minutes" without holding a streaming connection open. Push notification when the follow-up turn lands.
- **`start_loop(goal, interval_seconds)`** — turn the current chat into a self-driving worker. Every `interval_seconds` (60 s – 1 h) the daemon re-appends `goal` as a user turn. Idempotent — calling again replaces the existing loop. `stop_loop()` ends it; emerald banner above the composer shows live countdown + truncated goal.
- **`monitor`** — block on a file, HTTP URL, or bash command until a condition flips (`exists` / `missing` / `contains:` / `not_contains:` / `changed` / `status:` / `exit_code:` / `regex:`). Saves "run a tool, ask agent to retry in N seconds" loops. URL targets reuse the SSRF guard.

### Sandboxed containers (Docker)

`docker_run`, `docker_run_bg`, `docker_logs`, `docker_exec`, `docker_stop`, `docker_list`, `docker_pull` — run **any language or piece of software** inside an isolated container. Defaults: `--rm`, `--security-opt=no-new-privileges`, 512 MB memory, 1 CPU, conversation cwd mounted **read-only** at `/workspace`, bridge networking (inbound blocked unless explicitly published). Opt into `mount_mode: "rw"`, `network: "none"`, or published ports as needed. Image name is allowlist-validated; docker CLI invoked with argv list (no `shell=True`). Container management is scoped to containers Gigachat itself started (`gigachat_*` name prefix), so the agent can't tamper with your other containers.

### Other tools

- **`clipboard_read` / `clipboard_write`** — share small bits of text with the desktop without typing.
- **`create_worktree(branch, base_ref)`** — `git worktree add` on a throwaway branch so the agent can do risky edits without touching your working tree. Pair with `list_worktrees` / `remove_worktree(id)`. Branch and base_ref regex-validated.
- **MCP servers** — connect external [Model Context Protocol](https://modelcontextprotocol.io) servers over stdio. Each server runs as a local subprocess and every advertised tool is auto-merged into the palette as `mcp__<server>__<tool>`. Settings → **MCP** for CRUD; live tool counts + stderr tail for troubleshooting. 20 s handshake timeout, 120 s tool-call ceiling.
- **User-defined Python tools** — Settings → **Tools** lets *you* register Python snippets that become first-class entries in the tool palette. Each has a `def run(args)` entry point, optional pip dependencies (PEP 508 subset, blocklist on pip/setuptools/wheel), JSON-schema parameters, and a stored `category` + `timeout_seconds` the model cannot override. **The LLM has no route to create, edit, or delete these** — deliberate safety boundary against self-extension. Kill switch: `GIGACHAT_DISABLE_USER_TOOLS=1`.

---

## Permission modes

A header dropdown picks how tool calls are gated, per conversation:

| Mode | Icon | Read tools | Write tools |
|---|---|---|---|
| **Read-only** | 👁 | run silently | refused before approval card |
| **Plan mode** | 📋 | run silently | refused; agent must end with `[PLAN READY]` to unlock the **Execute plan** button (which flips to Approve edits + replays the plan) |
| **Approve edits** *(default)* | 🛡 | run silently | pause with diff/command/reason card, wait for click |
| **Allow everything** | ⚡ | run silently | run silently |

**Approval cards show the *full* command (bash), the *unified diff* (write/edit), and the model's `reason` field.** Every tool schema requires the `reason` so you can see *why* before clicking. Side-by-side diff toggle is one click.

⚠ **Use Allow everything only when actively watching.** A hostile tool output can try to prompt-inject the model into firing destructive tools. The default Approve edits is the safe choice — see the [Safety & security](#safety--security) section for the full threat model.

Tool categorisation lives in `backend/tools.py` (`TOOL_CATEGORIES` + `classify_tool()`). MCP tools default to write-class because their side effects are unknown.

---

## UI

- **Activity panel** (desktop only, right strip) — active tool with reason and 3-key args summary; "Thinking" card while drafting; recent-calls log with status icons. No need to scroll the transcript to see what's running.
- **Side-by-side diff** on every approval card with a one-click toggle to unified.
- **Auto-compaction** at ~75% context — older turns summarized into a synthetic system note. Compressed tool outputs preserve a head + tail snippet (first 5 + last 10 lines).
- **Proactive tool-output aging** — bulky results (>2 KB, >25 turns old) shrink to head+tail snippets even under the threshold.
- **Screenshot image aging** — only the most recent 5 screenshot frames carry their PNG into the next Ollama call; older frames become text-only descriptors. Vision tokens dominate cost in screenshot-heavy sessions.
- **Pinned messages** — pin a turn to keep it in context after compaction. Useful for sticky constraints (spec snippets, style guides). Manage from the chat header's "⋯" → **Pinned messages**.
- **Per-conversation persona** — "⋯" → **Persona** lets you paste a 4 000-char system-prompt fragment scoped to this chat ("act as a senior code reviewer who hates comments", etc.).
- **Soft conversation budget** — "⋯" → **Budget** caps a chat at N turns and/or N tokens. Header gauge turns amber at 60/80% and red at 100%.
- **Token-usage indicator** — header gauge shows distance to compaction. With a budget set, a Wallet gauge renders beside it; whichever is closer to its cap drives the percent.
- **Image / document paste / drag-drop** — drop PNG/JPEG/WebP/GIF (multimodal user message; needs vision model) or PDF/TXT/MD/CSV (server-side text extraction prepended to the next message). Mismatch warning if you attach an image while pointed at a text-only model.
- **Slash commands** — `/plan`, `/bug`, `/review`, `/explain`, `/search`, `/desktop` expand into proven prompt templates.
- **Push-to-talk dictation** — mic button uses the browser's Web Speech API for live speech→text. Falls back to a tooltip when the browser doesn't support it.
- **Realtime voice mode** — long-press for continuous dictation with built-in VAD; each finished utterance auto-sends. No TTS — the agent replies in streamed text.
- **Artifact / canvas preview** — `html` / `svg` / `mermaid` / `markdown` fenced blocks render a live preview pane beside the code. HTML is sandboxed in an iframe with `sandbox="allow-scripts"` (no `allow-same-origin`); SVG is sanitized through DOMPurify; Mermaid is dynamic-imported. Plain code blocks also qualify above 20 lines (line-numbered viewer + download button).
- **PWA + push notifications** — service worker + manifest, installs to home screen / Start menu. Completed scheduled tasks (and long turns finishing in a backgrounded tab) fire native OS notifications via Web Push (RFC 8291, ECDH + AES-128-GCM via pywebpush). VAPID keypair auto-generated on first run, persisted to `data/vapid.json` (0600).
- **Cross-conversation semantic search** — sparkle-badge mode in the sidebar embeds the query (`nomic-embed-text`) and returns top meaning-based matches across *every* conversation. Click a hit to open and scroll-to-match. Needs `ollama pull nomic-embed-text`; back-fill old messages once via `POST /api/conversations/reindex`.
- **Mobile-friendly UI** — shadcn/ui + Tailwind, dark theme, slide-in sidebar on phones.
- **Sonner toasts** for all errors.

---

## Models

- **Model picker** — choose any installed Ollama model per-conversation. Default filter shows only models whose Ollama `capabilities` list includes `tools`; a wrench-icon footer toggle (**Tool-capable** ↔ **Show all models**) flips it off when needed. Same toggle lives in Settings → **Default chat model**.
- **Auto-tuned default** — at first run the backend probes RAM / VRAM / GPU kind and picks the largest recommended Gemma 4 variant that fits. Override via Settings → **Default model**.
- **Auto-tuned context window** — `backend/sysdetect.py` picks the largest `num_ctx` Ollama can safely hold so the window grows with your hardware.
- **Auto-start / auto-pull** — backend launches Ollama as a detached subprocess if installed-but-not-running, then async-pulls the recommended default model and `nomic-embed-text` if missing. Set `GIGACHAT_NO_AUTO_PULL=1` to skip on metered connections.

### Tool-calling fallbacks

Some Ollama models advertise `tools` capability but ship a passthrough chat template (`{{ .Prompt }}`) that never renders `.Tools` — at time of writing, `gemma4:e4b` and `qwen3.5:9b` are in this bucket. Ollama silently discards `tools=[...]` and the model refuses to call anything.

Gigachat detects this on first use by probing `/api/show` and comparing the template against `.Tools` / `.ToolCalls` markers. When the template is a stub, the conversation switches to **prompt-space mode**: the tool list is serialized as an XML-tagged block in the system prompt, the model emits `<tool_call>{"name": ..., "args": ...}</tool_call>` inside its streamed text, and the agent loop parses those tags and feeds them into the same dispatch pipeline as native function calls.

The same parser also runs as a **safety net for natively-tool-aware models** — some smaller Gemmas and Qwens occasionally announce a call in prose and dump JSON between `<tool_call>` tags while leaving the structured channel empty, even when their template DOES render `.Tools`. The fallback recovers those silently dropped calls. No setting to toggle.

Gigachat also extends the picker to **tool-capable model families whose Ollama upload happens to strip the template** — `dolphin3:*` (Llama 3.1 base), `llama3.2-vision:11b`, `ikiru/Dolphin-Mistral-…` (Mistral 24B base), `deepseek-coder-v2:*`. Their weights were trained with function calling but the Modelfile shipped without the `{{ if .Tools }}` block, so Ollama drops the cap flag. Gigachat detects these by family pattern (`tool_prompt_adapter._matches_known_tool_capable`), shows them in the picker, and forces prompt-space mode for them — no `ollama create` workaround needed on your side.

### Picking a model

| Model | Size | Notes |
|---|---|---|
| `gemma4:e2b` | 7.2 GB | fastest, fits in 8 GB VRAM |
| `gemma4:e4b` / `gemma4:latest` | 9.6 GB | **recommended default** — best quality on 16 GB RAM + 8 GB VRAM |
| `gemma4:26b` | 18 GB | usually too big for ≤16 GB RAM |
| `gemma4:31b` | 20 GB | requires a workstation |
| `llama3.1:8b`, `qwen2.5:7b`, `mistral-nemo` | 4-5 GB | good chat alternatives; desktop-use needs a multimodal variant |
| `llava`, `qwen2.5-vl`, `gemma4:*` | varies | pick one for computer-use / screenshot tools |

---

## Settings drawer

One sidebar footer button (⚙ Settings) hosts eight tabs:

- **General** — default chat model, hardware summary, auto-pull status.
- **Memories** — global memory CRUD (one entry per row, optional `topic` for grouping; edits propagate immediately, no save button).
- **Secrets** — named API tokens / credentials referenced via `{{secret:NAME}}`. Values hidden by default; click reveal to show one. The agent has no write access here.
- **Schedules** — every queued prompt with next-run / interval / cwd. Add / delete from the UI; rows back the agent's `schedule_task` tool too.
- **Docs** — URL-indexed documentation sites for `docs_search`. Live status chip per site; reindex button.
- **Tools** — user-defined Python tools (review, pause, delete, or add new ones with code + schema + deps form).
- **Hooks** — register shell commands at agent lifecycle points (`user_prompt_submit`, `pre_tool` / `post_tool`, `turn_done`). Each receives a structured JSON payload on stdin; stdout is injected back as a system-note. Hooks run with your full shell privileges — UI warns on creation.
- **MCP** — external Model Context Protocol servers.

Notifications stays as its own footer entry because push permission is a per-device toggle, not a shared preference.

---

## Workflows (lifecycle hooks)

Settings → **Hooks** lets you register shell commands that fire at well-known points in the agent loop. Each hook receives a structured JSON payload on stdin; whatever it prints to stdout gets injected back into the conversation as a system-role message so the model sees it on the next turn. That's all — but it composes into surprisingly powerful workflows.

### Trigger events

| Event | Fires when |
|---|---|
| `user_prompt_submit` | Once per user turn, before the model runs. |
| `pre_tool` / `post_tool` | Around each tool call (optional tool-name substring matcher). |
| `tool_error` | A tool call returned `ok=False`. Cleaner than filtering inside a `post_tool` hook. |
| `consecutive_failures` | After **N** back-to-back failures of the same tool in the same conversation (N is per-hook via the `error_threshold` field, default 1). The "model is looping on the same broken call" signal. |
| `turn_done` | When the agent produces a final answer. |

### Per-hook settings

- **Tool matcher** — case-insensitive substring against the tool name. Empty = match every tool.
- **Timeout** — hard cap on the shell command. 120 s for the original four events, **900 s for `tool_error` / `consecutive_failures`** so a long-running diagnosis (Claude CLI, full pytest run) has time to finish.
- **Max fires per conversation** — cap on how often the hook can fire in one chat. Persisted in `hook_fires`, so a backend restart can't reset the counter and re-open a runaway loop. Empty = unlimited; **set this to 5-10 for any hook that calls a paid API**.

### Example: Claude self-fixer

Drop in a hook that asks Claude to diagnose the bug whenever the agent fails the same tool 3+ times. Claude reads the repo, decides if it's a real Gigachat bug or model misuse, fixes the source if needed, runs the smoke tests, commits, and replies with a one-paragraph verdict that lands as a system-note in the failing conversation. Backend hot-reload picks up any source changes, the model retries with the fix in place.

Hook config (Settings → **Hooks** → New hook):

| Field | Value |
|---|---|
| Event | `consecutive_failures` |
| Tool matcher | *(empty — match any tool)* |
| Fire after | `3` |
| Command | `bash /c/Users/gauta/Downloads/Gigachat/scripts/claude-fixer.sh` |
| Timeout (s) | `600` |
| Max fires / conversation | `5` |

The script — already shipped at [scripts/claude-fixer.sh](scripts/claude-fixer.sh) — uses python (already on the box for the Gigachat backend) for JSON parsing, so it works on a vanilla Windows + Git Bash install with no extra binaries:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Build the full diagnosis prompt in python — bash's $(...) silently
# strips embedded NUL bytes, so multi-field round-tripping back to
# shell variables loses data. Easier to just have python emit the
# entire prompt and pipe it straight to claude.
prompt=$(python -X utf8 -c '
import sys, json
d = json.loads(sys.stdin.read())
conv_id = d.get("conversation_id", "")
tool_name = d.get("tool_name", "")
streak = d.get("consecutive_count", 0)
err = (d.get("error") or d.get("output") or "")[:2000]
sys.stdout.write(f"""\
Gigachat conversation {conv_id} is wedged: tool {tool_name!r} just failed
{streak} times in a row. Last error:

{err}

Decide: real Gigachat bug or model mistake?
- Real bug: read backend/, fix it, run `python -m pytest -m smoke`, commit. Reply in ONE paragraph.
- Model mistake: tell the agent what to try differently. No code changes.

Reply under 600 chars and end with a clear next-step.
""")
')

timeout 540 claude --print "$prompt" 2>&1 | tail -c 1500
```

### Other useful workflow recipes

| Goal | Event | Matcher | Command (one-liner) |
|---|---|---|---|
| Lint after every successful file write | `post_tool` | `write_file` | `python -X utf8 -c "import sys, json; print(json.loads(sys.stdin.read())['tool_args']['path'])" \| xargs eslint --fix` |
| Slack me when a turn finishes | `turn_done` | — | `curl -X POST $SLACK_WEBHOOK -d @-` |
| Run pytest after every model edit | `post_tool` | `edit_file` | `cd "$(python -X utf8 -c "import sys, json; print(json.loads(sys.stdin.read())['cwd'])")" && pytest -m smoke -q 2>&1 \| tail -20` |
| Block tool calls that touch a sensitive path | `pre_tool` | — | `python -X utf8 -c "import sys, json; d=json.loads(sys.stdin.read()); sys.exit(1 if '/secrets/' in (d.get('tool_args', {}).get('path') or '') else 0)"` |

Each row is one entry in Settings → **Hooks**. No parallel "Workflows" subsystem to learn — the hooks panel IS the workflow builder.

---

## Working directory

Each conversation has a `cwd` that all commands run from. The chat-header dialog has a **Browse…** button that opens the native OS folder picker (tkinter on Win/macOS/Linux); the chosen path is validated server-side. Once set, `cwd` is immutable — every tool call, checkpoint, and codebase-index row is keyed by it.

**`AGENTS.md` / `CLAUDE.md` auto-injection** — on every turn the backend walks from `cwd` up to the filesystem root and concatenates every `AGENTS.md` and `CLAUDE.md` it finds into the system prompt (outermost first, innermost last — nearer-in instructions win). Both names are treated equally so a repo that ships only one still works; nested sub-projects can override parent rules.

**File checkpoints** — every `write_file` / `edit_file` snapshots the prior contents under `data/checkpoints/<conv_id>/<stamp>/<hash>.bin` and exposes a one-click restore.

**Crash resilience** — every conversation carries a `state` column (`idle` / `running` / `error`). `run_turn` flips to `running` on entry and back to `idle` in a `finally` block. On startup, a 3-second-delayed resumer polls `list_conversations_by_state('running')`, drops a system-note breadcrumb (`[crash-resilience] The previous run was interrupted…`), and either auto-resumes the turn (when the last visible message is user-role or queued inputs are waiting) or flips state back to `idle`. Scheduled tasks are persisted in SQLite; the polling daemon restarts with the backend, so a missed recurrence fires on the next tick.

---

## Requirements

- **Windows 10/11** for the launcher `.bat` scripts; Python/Node code is cross-platform.
- **Python 3.12+**
- **Node 20+**
- **Ollama** running locally on `http://localhost:11434`.
- **At least one function-calling Ollama model**, e.g. `ollama pull gemma4:e4b`.

---

## Setup

From the project root (`Gigachat/`):

```powershell
python -m pip install -r backend/requirements.txt
cd frontend
npm install
cd ..
```

---

## Running

### Development (hot reload)

```
.\dev.bat
```

Two console windows. Visit **http://localhost:5173** — Vite dev server proxies `/api/*` to the backend on :8000.

### Production (one server)

```
.\build.bat        # one-time frontend build
.\start.bat        # FastAPI at http://localhost:8000
```

`start.bat` invokes `python -m backend.server`, a thin uvicorn wrapper that reads the auth config (see **Remote access**) and binds to the configured host. Default host is `127.0.0.1` so nothing on your LAN can reach it.

---

## Remote access

Three bind modes — pick by who needs to reach it:

| Mode | Reachable from | Client install? | Use case |
| --- | --- | --- | --- |
| **loopback** *(default)* | Host machine only | — | Day-to-day solo. Zero config. |
| **`tailscale`** | Any device in your tailnet | [Tailscale](https://tailscale.com/download) on each client | Your own phone/laptop only. End-to-end encrypted. |
| **`public`** | Anyone with the URL | None — a normal browser | Sharing with others / untrusted networks via reverse proxy. |

Raw LAN IPs and `0.0.0.0` binds are intentionally unsupported — the three modes above cover every real-world access pattern without dynamic-IP / port-forwarding footguns.

### Tailscale mode

Install Tailscale on the host **and** every device you want to reach it from, sign them into the same account, run `tailscale up`.

Hash a password (PBKDF2-SHA256, 200 000 iterations, 16-byte salt):

```powershell
python -c "from backend.auth import hash_password; print(hash_password('your-password-here'))"
```

Plaintext is also accepted for dev convenience but the hash is the canonical form.

Write `data/auth.json`:

```json
{
  "host": "tailscale",
  "password": "a1b2c3…:d4e5f6…"
}
```

Or use env vars (they win over the file): `GIGACHAT_HOST=tailscale` and `GIGACHAT_PASSWORD=…`. Then `.\start.bat`.

The launcher binds to `0.0.0.0` but the access-control middleware admits only loopback and Tailscale CGNAT ranges (`100.64.0.0/10` IPv4, `fd7a:115c:a1e0::/48` IPv6) — anything else gets a flat 403. The banner prints two URLs: `http://localhost:8000` for the host machine, `http://<your-tailscale-ip>:8000` for tailnet peers. MagicDNS handles hostname lookup. Login stores an HMAC-signed session cookie (httponly, SameSite=Lax, 30-day TTL). Loopback requests from the host are still implicitly authenticated. If Tailscale isn't running, startup falls back to pure loopback with a warning.

### Public mode (reverse proxy)

For when clients won't install Tailscale (a friend logging in, library PC, demo). Backend still binds to `127.0.0.1`; you run a TLS-terminating reverse proxy on the same host that forwards public HTTPS traffic onto loopback. **Auth is mandatory for every request, including loopback** — the proxy delivers all public traffic as `127.0.0.1`, so auto-trusting it would be a complete bypass.

Same auth setup but `"host": "public"`:

```json
{
  "host": "public",
  "password": "a1b2c3…:d4e5f6…"
}
```

**Cloudflare Tunnel** is the easiest. One-time setup:

```powershell
winget install --id Cloudflare.cloudflared
cloudflared tunnel login
cloudflared tunnel create gigachat
cloudflared tunnel route dns gigachat gigachat.yourdomain.com
```

`start.bat` detects public mode, launches the backend, and runs the `gigachat` tunnel for you. Any HTTPS-terminating proxy works (Caddy `reverse_proxy 127.0.0.1:8000`, nginx, Traefik, …) as long as it terminates TLS and forwards to `http://127.0.0.1:8000`. The backend doesn't trust proxy headers (`X-Forwarded-For`, `X-Real-IP`) for auth decisions, and the session cookie is marked `Secure` so it only ever travels over the HTTPS hop.

**Per-conversation workspaces.** New chats in public mode default `cwd` to an auto-created `data/workspaces/<conv-id>/` instead of the Gigachat project root, so a remote session won't land inside the source tree. You can still retarget `cwd` from the chat header to any path on the host. Folders are never auto-deleted on chat delete.

### Signing in and out

LoginView shows the configured host so you can confirm you're at the right Gigachat before typing. The sidebar footer has a **Sign out** button when auth is enabled. Mid-session cookie expiry is detected — any API call that 401s dispatches a `gigachat:unauthorized` event and the frontend swaps back to login.

---

## Tests

A pytest suite covers the database layer, agent input queue, upload-name traversal guard, element-ID cache, `ui_wait` dispatcher, structured logging, retention janitor, the `_resolve` ↔ bash_cwd contract for file tools, and the text-format tool-call recovery parser.

Three markers (`pytest.ini`):

- **`smoke`** — fast, offline, platform-agnostic. Runs on every push in CI (`.github/workflows/ci.yml`).
- **`deep`** — slower or needs live services (Ollama / real HTTP). Run locally or nightly.
- **`windows`** — needs Windows UIA / pyautogui; auto-skipped elsewhere.

```
python -m pip install -r backend/requirements.txt
python -m pytest -m smoke         # fast tier, ~15 s, 95 tests
python -m pytest                  # everything (drops Windows-only on Linux)
```

The `isolated_db` fixture rewires `db.DB_PATH` to a tmp file per test, so the suite never touches `data/app.db`.

---

## Safety & security

- **Default bind is 127.0.0.1.** Nothing on your LAN reaches it until you opt in via `GIGACHAT_HOST` / `data/auth.json`.
- **Password gate on every remote request.** Tailscale mode: non-loopback requests need a session cookie or `Bearer` token. Public mode: every request — **including loopback** — needs auth, because the reverse proxy delivers public traffic as 127.0.0.1.
- **PBKDF2-SHA256 password hashes** (200 000 iterations, 16-byte salt). Plaintext is accepted for dev convenience; `hash_password()` is canonical. Session tokens are HMAC-SHA256 signed against `data/auth_secret.key` (0600, auto-generated). 30-day TTL. Rotating the secret file invalidates every existing session — a one-step "log everyone out" lever.
- **Public-mode hardening.** Session cookie is `Secure` so it only ever travels HTTPS. The login endpoint rate-limits to **10 failed attempts per 60 seconds** to blunt credential-stuffing.
- **Use a strong random password** (`python -c "import secrets; print(secrets.token_urlsafe(24))"`) and consider layering Cloudflare Access / Zero Trust if the machine is a target.
- **Parameterized SQL end-to-end.** No string concat into SQL.
- **Per-tool runtime caps.** 120-second default timeout, 20 000-character output cap.
- **Approve edits is the safe default** for new conversations. Read-only is great for "let the model poke around but don't let it touch anything." Allow everything is for watched sessions or scheduled jobs only.

⚠ The whole point of the app is that a local LLM can run commands on your PC. Treat it like any other agentic tool: review before approving destructive actions (delete, overwrite, `rm -rf`, package installs).

### Known risks

- **Prompt injection via tool output.** A file or command's output is fed back to the model, so a hostile file could try to trick the model with "ignore prior instructions" content. Keep the permission mode on **Approve edits** (or **Read-only**) for any conversation that touches untrusted data (email, downloads, clipboard, web scrapes, images). You'll see every proposed follow-up before it runs.
- **No path sandbox.** The agent can read and write anywhere your user account can. Point `cwd` at the narrowest folder that makes sense. `edit_file` / `write_file` checkpoint prior contents so you can restore after a bad edit.
- **Image and file uploads** — streaming size cap (10 MB), content-type allowlist (`image/png|jpeg|webp|gif`), random-hex filenames so a caller can't overwrite arbitrary files by picking a name.
- **Background shells (`bash_bg`)** keep running until the conversation is deleted or you call `kill_shell`. They inherit the same env and FS access as foreground bash — treat them like any other shell you left open.
- **Computer use controls your real desktop.** Screenshots include every visible window. Mouse/keyboard events are issued as your actual logged-in user — the agent can click "OK" on system dialogs, drag files into the trash, type into password fields.
  - Keep permission mode on **Approve edits** when first enabling — every click, keypress, and scroll pauses for confirmation with a thumbnail of the screen the model is reacting to.
  - Close private windows (banking, messages, password manager) before handing the mouse over.
  - Don't ask the agent to enter passwords, PINs, or 2FA codes; type those yourself.
  - Move the mouse into a screen corner for ~1 s and pyautogui's failsafe aborts the next action.
- **`computer_batch` is allowlisted, not a generic eval.** Only desktop primitives can appear (move/click/drag/type/key/scroll/wait_ms/focus/window/click_element/click_element_id/open_app/ocr) — `bash`, `read_file`, `write_file`, `browser_*`, `delegate`, `schedule_task`, etc. are explicitly rejected. Caps: 20 steps per call, 5 s max per `wait_ms`, 100 ms inter-step settle. Each step still respects the failsafe.
- **`click_element_id` IDs are process-scoped.** Minted by `inspect_window` or `screenshot(with_elements=true)`, kept in an in-memory `OrderedDict` (max 5000, **LRU-evicted**) guarded by a lock so concurrent subagents can't race the counter. IDs do not survive a backend restart and do not survive UI movement — re-`inspect_window` to mint fresh ones. A miss returns a clear error distinguishing **"bad format"** (typo), **"not minted"** (id past high-water mark), and **"evicted"** (real id aged out).
- **`ui_wait` is bounded.** Max 30 s, ~250 ms poll. Six kinds: `window` / `window_gone`, `element` / `element_enabled`, `text` (OCR), `pixel_change`. Pixel mode compares against the same baseline (downsampled 64×36) with a 5%-of-grid threshold tuned to ignore JPEG/compression noise but catch dialog popups. Prefer `window_gone` / `element_enabled` over `pixel_change` when a deterministic signal is available.
- **`type_into_element` is a click+type combo, not a new capability.** Same UIA tree walk as `click_element` to focus, same keystrokes as `computer_type`. 200-char name cap, 10 000-char text cap. A failed focus click bails before any keys are pressed. The `clear:true` option sends Ctrl+A + Delete first; it is intentionally NOT a delete-file path — it only operates on whatever input the focus click landed on.
- **Status-context tag** (`[ctx: foreground='...'; focused='...'; cursor=(x,y)]`) is a read-only UIA + cursor snapshot taken on every screenshot. Window titles and accessible names come from untrusted sources, so they're length-capped (80 chars) and rendered through Python's `!r` repr — embedded newlines / quotes are escaped, so a hostile aria-label can't smuggle a fake instruction line.
- **Focus-drift warnings** (`[focus drifted: 'X' → 'Y']`) on `computer_type` / `computer_key` use the same read-only UIA query before and after each action. Surface focus theft (a popup grabbed the caret mid-typing) so the model retries instead of trusting a silent miss.
- **Screenshot change-feedback is informational.** Each screenshot result includes a one-line diff (`[Δ: Δ 8% pixels; new: 'Save As']`) from a 64×36 grayscale signature + UIA window-title set. Signatures live in process-local globals, are overwritten per screenshot, never persisted. Pixel comparisons happen BEFORE the coordinate-grid / click-marker overlays are drawn.
- **The "last click" red marker is a one-shot UI hint.** Click position stored in a single-tuple global; the next screenshot draws a red dot + crosshair and clears the global. Image-space arithmetic so it appears at the correct spot regardless of monitor scaling.
- **`screenshot_window` and `list_windows` are read-only.** Both walk the UIA tree; neither modifies state, neither sends input. `screenshot_window` clips the capture rect to virtual-screen bounds and refuses minimized windows with a clear message. `list_windows` filters out zero-area / nameless windows. Output is capped at 100 windows.
- **`inspect_window` Set-of-Mark overlay is best-effort.** When the window is visible, returns an annotated PNG with each `[elN]` badge painted at the matching control's anchor; collisions resolved by shifting badges down/right (bounded retries), capped at 80 badges. If the window is fully off-screen / mss fails / PIL is missing a TTF, the renderer returns `None` and the inspect call still succeeds with the text dump. Pass `overlay: false` to skip rendering when only IDs are needed.
- **Web access pulls untrusted content into the conversation.** Pages from `fetch_url` are treated like any other tool output — a hostile page could try to prompt-inject the model. Mitigations:
  - `fetch_url` rejects non-http(s) schemes and any URL whose host is loopback / private / link-local / multicast — including DNS-resolved hostnames.
  - HTML capped at 2 MB on the wire, extracted prose at ~15 000 chars.
  - System prompt explicitly tells the model to treat fetched content as untrusted.
  - Manual approval is the real defense — you see the URL *and* a preview of the extracted text.
- **Browser-automation tools drive a real Chrome tab.** A hostile page reached via `browser_goto` can prompt-inject through `browser_text`. `browser_eval` runs arbitrary JS in the page context — it can read cookies, localStorage, and DOM of whatever site the tab is on. Keep the CDP browser pointed at throwaway / agent-only sessions, not the profile where your bank is logged in. `browser_goto` enforces http/https schemes (rejects `javascript:` / `file://`) but that's a guardrail on URL structure, not on page content — review every tool call.
- **Scheduled tasks run unattended in Allow-everything mode.** `schedule_task` opens a brand-new conversation in **Allow everything** by design (nobody's watching at 3 AM). Treat the prompt like a cron job: be specific, avoid telling the model to "do whatever" based on fetched web content, never schedule a prompt itself pulled from an untrusted source. `cancel_scheduled_task` / `list_scheduled_tasks`.
- **Autonomous loops (`start_loop`) inherit the conversation's permission mode.** Unlike `schedule_task` (which spawns a new chat in Allow-everything), a loop fires the rolling `goal` back into the *existing* conversation. An **Allow everything** chat with an active loop will click through every write call unattended every `interval_seconds` until you click **Stop loop**. Start loops only on chats you're comfortable letting run autonomously; prefer **Approve edits** if you're stepping away. Intervals clamped 60 s – 1 h, goals capped at 4 000 chars.
- **Codebase index walks the entire cwd.** `codebase_search` is read-only but the *builder* opens every matching file (up to 1 500 files, 2 MB each, allowlisted extensions) to embed it locally. Two consequences: (1) chunks land in `data/app.db` — if you indexed a cwd containing a `.env` / credential file matching the allowlist, the secret is now duplicated there. The git-aware walker is the main defence (`git ls-files --exclude-standard` respects `.gitignore`); (2) on a non-git cwd the fallback walker follows symlinks via `rglob`, so a symlink to your home directory could pull files outside the cwd into the index. Point `cwd` at the narrowest folder.
- **`read_doc` runs real parsers** (pymupdf / python-docx / openpyxl). They don't execute embedded macros / JavaScript, but extracted text from a hostile doc can still attempt prompt injection like any other tool output.
- **`doc_index` / `doc_search` store raw file contents in SQLite.** Chunks are kept verbatim (so retrieved context is readable). Any secret inside a file you indexed — API keys in a `.env`, tokens in a config — is now duplicated in `data/app.db`.
- **Lifecycle hooks run arbitrary shell commands.** Each is a shell string you entered via the UI, run with your full login shell privileges on every matching event. CRUD endpoints are bound to localhost, no CORS headers (browser preflight from other origins fails), JSON payload passed on stdin (not interpolated into the command). The command itself is trusted input by design — only register hooks you wrote yourself.
- **Global memories are injected into every system prompt** — including subagents. Two consequences: (1) avoid storing secrets, entries are not encrypted; (2) the agent can extend its own behaviour across chats, so review the panel periodically. Length caps: 8 KB per entry, 80-char topic. `forget(scope="global", pattern="")` is refused so a typo can't wipe the table.
- **`monitor` is read-only but probes the network.** `url:` reuses the SSRF guard (rejects loopback / RFC1918 / link-local / reserved, including DNS resolution). `bash:` inherits `run_bash`'s 30-second per-tick cap. Total wait time clamped to 30 minutes.
- **`http_request` calls arbitrary APIs with your credentials.** Write-class regardless of method (GET included). Same SSRF guard as `fetch_url`; `allow_private: true` opts into LAN. Method allowlist `GET/POST/PUT/PATCH/DELETE/HEAD/OPTIONS`. Response capped at 2 MB on the wire, 20 000 chars in tool output. `Authorization` / `Set-Cookie` / `X-API-Key` / `Cookie` headers are masked in echoed request + response summaries.
- **Secrets live in SQLite in plaintext.** Threat model is single-user local: anyone with read access to `data/app.db` already has read access to everything under your user profile. Values never reach the model — agent references them as `{{secret:NAME}}`; backend substitutes just before the wire. Names regex `^[A-Za-z_][A-Za-z0-9_]{0,63}$`; values capped at 16 000 chars; descriptions at 400; `UNIQUE(name)` prevents silent overrides. **Defence-in-depth:** any substituted value is scrubbed from the response body before the tool result is stored — even a misconfigured server echoing `Authorization` back doesn't land the credential in the transcript. Tiny values (<4 chars) are not scrubbed (false-positive rate too high on random 4-byte substrings).
- **`delegate_parallel` concurrency is capped.** Max 6 subagents per call, each bounded by `max_iterations` (default 10, max 20). Each gets the trimmed tool set — no nested delegation, no desktop / browser / scheduling.
- **User-defined tools run arbitrary Python in a shared venv.** Only the user can create them — the LLM has no self-extension route. Code is `ast.parse`-validated (must define `def run(args)`, must parse cleanly) but **NOT sandboxed beyond that**. Layers:
  1. Creation gated behind the Settings UI — you review code and dep list before first install.
  2. Name regex `^[a-z][a-z0-9_]{0,47}$` blocks collisions with built-ins / MCP / SQLite tricks.
  3. Dep-spec regex matches a PEP 508 subset (name + extras + version comparators only — **no URLs, no VCS URIs, no file paths**).
  4. Blocklist refuses `pip` / `setuptools` / `wheel` / `distribute`.
  5. Pip runs `--disable-pip-version-check --no-input` in a 300 s subprocess that can't read stdin.
  6. Wrapper runs the tool with `python -I` (isolated mode — ignores `PYTHONPATH` / user site-packages / startup scripts), args via stdin JSON, stdout parsed at a sentinel line.
  7. Each tool stores its own `timeout_seconds` (1-600 s) and `category` (read/write) — model can't override at call time.
  8. **Kill switch:** `GIGACHAT_DISABLE_USER_TOOLS=1` skips schema registration for existing rows and refuses execution.

---

## Layout

```
Gigachat/
├── backend/
│   ├── server.py         thin uvicorn launcher — reads auth config, resolves
│   │                     `tailscale` → `tailscale ip -4` / `public` → `127.0.0.1`,
│   │                     prints a banner, warns if remote without a password
│   ├── auth.py           PBKDF2-SHA256 password hashing + HMAC session tokens +
│   │                     loopback / Tailscale / public host resolution
│   ├── app.py            FastAPI routes + SSE + AuthMiddleware (loopback-or-tailnet
│   │                     IP allowlist + session cookie) + startup resumer for
│   │                     interrupted turns
│   ├── agent.py          run_turn() — Ollama ↔ tool loop, approvals,
│   │                     auto-compaction, subagent runner, image handling,
│   │                     text-format tool-call recovery parser
│   ├── tools.py          every tool implementation + dispatcher (~10k lines)
│   ├── tool_prompt_adapter.py   prompt-space tool-calling fallback for
│   │                            stub-template models + JSON-tag parser
│   ├── user_tools_runtime.py    shared venv at `data/tools_venv/` +
│   │                            PEP 508 dep-spec validation + isolated-mode
│   │                            subprocess wrapper for user-defined tools
│   ├── db.py             SQLite schema + CRUD; all persistent state
│   ├── prompts.py        system prompt assembly (+ AGENTS.md / CLAUDE.md walked
│   │                     up from cwd + per-conv memory + global memory) +
│   │                     Ollama tool schemas
│   ├── ollama_runtime.py Ollama HTTP client + model registry
│   ├── mcp.py            MCP server integration (external tool backends)
│   ├── push.py           Web Push notifications
│   ├── sysdetect.py      RAM / VRAM / GPU probe used to pick num_ctx at startup
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx       auth gate (LoginView vs main app) + top-level layout
│   │   ├── components/   ChatView, Sidebar, Message, ToolCall, ActivityPanel,
│   │   │                  HooksPanel, MemoriesPanel, SecretsPanel,
│   │   │                  SchedulesPanel, MCPServersPanel, UserToolsPanel,
│   │   │                  SettingsPanel, LoginView, …
│   │   ├── components/ui/ shadcn primitives (Button, Switch, Dialog, …)
│   │   ├── lib/          api client + SSE reader + cn helper
│   │   └── index.scss
│   ├── package.json
│   └── vite.config.js    proxies /api → :8000 in dev
├── backend/tests/        pytest suite (95 smoke tests; offline + platform-agnostic)
├── scripts/              QA helpers: drive_real_conversation.py runs a single
│                         agent turn end-to-end against a real Ollama model;
│                         review_conversation.py audits a finished conversation
│                         for the bash-cwd / file-tool / bash-error-rate metrics
├── data/                 runtime state (SQLite + screenshots + per-conv
│                         workspaces / memory / checkpoints), all git-ignored
├── dev.bat / build.bat / start.bat
├── AGENTS.md             agent-facing project notes
└── CLAUDE.md             project rules
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Ollama not reachable" toast | Run `ollama serve` in a terminal. |
| Model picker is empty | `ollama list` to confirm; `ollama pull gemma4:e4b`. |
| Responses very slow | Likely swapping model weights to RAM. Try `gemma4:e2b`. |
| Approval click does nothing | Check the backend console for errors; the `/approve` endpoint may have failed. |
| Dev server port 5173 in use | Kill the other Vite process or change the port in `frontend/vite.config.js`. |
| `web_search` rate-limited | DuckDuckGo occasionally rate-limits. Wait a minute; persistent? `pip install -U ddgs`. |
| `fetch_url`: "could not extract readable content" | Site is JS-rendered (SPA) or blocks bots. Try a different `web_search` result, or drive the browser via computer-use tools. |
| `browser_*`: "no Chrome tabs visible on CDP port 9222" | Chrome wasn't launched with remote debugging. Restart via `open_app({"name": "chrome", "args": ["--remote-debugging-port=9222"]})`, or close all existing Chrome windows first. |
| `doc_index` / `doc_search`: "ollama /api/embeddings returned no vector" | Embedding model not installed. `ollama pull nomic-embed-text` (or pass a different `model` you have). |
| Scheduled tasks never fire | Confirm the backend is running (the daemon lives inside the FastAPI startup event). Use `list_scheduled_tasks` to verify `next_run_at` — if it's in the far future, ISO parsing picked up the wrong timezone; recreate with an explicit offset like `2026-05-01T09:00:00+00:00`. |
| OCR: "no OCR language pack installed" | Windows Settings → Time & Language → Language → add a language with OCR support, or install Tesseract and `pip install pytesseract` for the cross-platform fallback. |
| Lifecycle hook never fires | Settings → Hooks; confirm enabled and matcher (case-insensitive substring) actually matches the tool name. Empty matcher fires for every tool. |
| `monitor` times out immediately | Double-check the `target` prefix (`file:` / `url:` / `bash:`). For `url:` the SSRF guard refuses loopback / RFC1918; use a public URL or a `bash:` target. |
| `click_element_id`: "unknown element id" | IDs live only for the current backend process (LRU at 5000). The error tells you why: **"bad format"** (`[elN]` typo), **"not minted"** (re-inspect — id was never issued), **"evicted"** (re-inspect — aged out). |
| `ui_wait` times out on `pixel_change` even though the screen visibly changed | Change was below the 5% threshold (tuned to ignore JPEG/compression flicker). Narrow with `region: {x, y, w, h}` so the changed area is a larger fraction of the box, or switch to `window` / `element` / `text` / `*_gone` / `*_enabled` for a deterministic signal. |
| `computer_batch` rejects an action | Allowlist is intentionally small — only desktop primitives. `bash`, `read_file`, `browser_*`, `delegate`, `schedule_task` etc. must be called as their own tool so they go through normal approval. |
| `screenshot_window`: "window has zero area (it's probably minimized)" | Window minimized to taskbar — bounding rect is degenerate. Call `window_action({"name": "...", "action": "restore"})` first. |
| `inspect_window` returns no annotated screenshot | Window fully off-screen, mss couldn't capture (rare on Windows), or font issue. Text dump still has every `[elN]` so you can call `click_element_id` blind. Pass `overlay: false` to suppress the attempt entirely. |
| Chat header "Browse…" toasts "Folder picker unavailable" | The picker uses `tkinter`; on headless Linux installs `python3-tk` is often missing (`sudo apt install python3-tk`). Or hand-type the absolute path. |
| "[crash-resilience] The previous run was interrupted…" note | Backend exited mid-turn last time. Resumer detected a `state='running'` conversation and dropped the breadcrumb. Either auto-resumed (last message was user-role) or flipped state back to idle. Safe to ignore. |
| User-tool create fails with "dep install failed" | Read the install_log from the toast — it's pip's raw output. Usual causes: dep misspelled, version spec stricter than anything available, network can't reach PyPI. The dep-spec regex refuses URLs / VCS URIs / file paths by design — rewrite as a plain `name>=ver` spec. Kill switch: `GIGACHAT_DISABLE_USER_TOOLS=1`. |
| User tool: "user tools are disabled" | `GIGACHAT_DISABLE_USER_TOOLS=1` is set in the backend env. Unset and restart, or remove from `data/auth.json`'s env block. Rows stay in SQLite; nothing was lost. |
| Public mode: every request 401s even on localhost | Expected. Loopback is not auto-trusted because the reverse proxy delivers public traffic as 127.0.0.1. Log in at the public URL (or `http://127.0.0.1:8000` locally) to get a session cookie. |
