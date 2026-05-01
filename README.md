<p align="center">
  <img src="frontend/src/assets/gigachat-logo.jpg" alt="Gigachat logo" width="180" />
</p>

# Gigachat

A self-hosted web app that turns **any locally-running Ollama model** (Gemma, Llama, Qwen, DeepSeek, Mistral — anything with function-calling) into a Claude-Code-style assistant: chat with conversation history, run shell commands, edit files, drive your desktop, search the web, query APIs — with a per-conversation permission gate on every tool call.

```
┌────────────────┐          ┌─────────────────┐          ┌──────────────┐
│  React + Vite  │  SSE     │  FastAPI        │  chat    │   Ollama     │
│  + shadcn/ui   │◄────────►│  agent loop     │◄────────►│   any model  │
│  (port 5173)   │  /api/*  │  (port 8000)    │  :11434  │              │
└────────────────┘          └─────────────────┘          └──────────────┘
```

> ⚠ **Capability tracks the model you run.** A bigger / better model gives you sharper reasoning, more reliable tool use, and longer plans; a small model will sometimes drop tool calls or hallucinate paths. If something feels broken, try a larger model first.

---

## Quickstart

```powershell
# 1. Install Ollama and a function-calling model.
ollama pull gemma4:e4b

# 2. One-shot setup (creates an isolated .venv\ and installs every dep).
.\setup.bat

# 3. Run the dev servers (two console windows). Visit http://localhost:5173.
.\dev.bat
```

That's it for solo loopback use. **Production** build is `.\build.bat` then `.\start.bat` (FastAPI serves both the API + the built frontend on http://localhost:8000). All three launchers auto-detect `.venv\` and use it when present.

**Manual setup** (if you'd rather not use `setup.bat`):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r backend\requirements.txt
cd frontend && npm install && cd ..
```

The venv is strongly recommended on Windows — installing into the system + user site-packages dirs mixes locations and runs into `PermissionError` issues when antivirus / OneDrive / file ACLs touch one of the half-installed files. See the [Troubleshooting](#troubleshooting) row for that error if you hit it.

**Requirements**: Windows 10/11 (for the launcher `.bat` scripts; Python/Node code is cross-platform), Python 3.12+, Node 20+, Ollama running locally on `http://localhost:11434`, at least one function-calling Ollama model.

---

## What it can do

- **Chat with full history**, persisted to SQLite — search, pin, tag, group by project, edit-and-regenerate the last user message.
- **Run real tools** — shell commands, file read/write/edit, screenshot + click, Chrome browser automation, web search, OpenAPI calls, sandboxed Docker, SSH, email, Home Assistant, audio transcription. Full catalog: [docs/TOOLS.md](./docs/TOOLS.md).
- **Computer use** — the model can see your desktop and drive your mouse/keyboard (multimodal model required). Coordinate-grid screenshots, accessibility-tree clicks, bounded waits, batched primitives.
- **Per-message permission gate** — every write-class tool call pauses with a diff or command preview until you approve. Read-only / Plan / Approve edits / Allow everything modes.
- **Watch the reasoning** — desktop side strip shows the active tool, its args, a pulsing "Thinking" card. No need to scroll the transcript to see what's running.
- **Quality modes** — same model, more compute. Refine (self-critique + revise), Consensus (sample + synthesize), Personas (diverse reasoning overlays), Auto. Closes the gap to GPT-4 / Claude class on small-to-mid models.
- **Compute pool** — pair other Gigachat installs on your LAN over a 6-digit PIN (Bluetooth-style). Big models that don't fit one machine layer-split across the pool via llama.cpp `--rpc`. Speculative decoding recruits idle peers.
- **Public pool** — opt-in global swarm. Use other peers' GPUs for models you don't have locally; donate idle compute back. End-to-end encrypted (X25519 + ChaCha20-Poly1305). See [docs/P2P.md](./docs/P2P.md).
- **Long-running tasks** — schedule prompts at an ISO datetime / recurring interval, set a chat into autonomous loop mode, monitor a file/URL until a condition flips.
- **Memory + skills** — long-term facts (per-conversation OR global), procedural skills the agent can save and recall, lifecycle hooks that fire on agent events.
- **Survive crashes** — every conversation has an `idle/running/error` state; the startup resumer either re-enters interrupted turns or flips state back to idle.
- **Stream tokens live** over Server-Sent Events; queue follow-up messages while a turn is in flight without locking the composer.

---

## Picking a model

| Model | Size | Notes |
|---|---|---|
| `gemma4:e2b` | 7.2 GB | fastest, fits in 8 GB VRAM |
| `gemma4:e4b` / `gemma4:latest` | 9.6 GB | **recommended default** — best quality on 16 GB RAM + 8 GB VRAM |
| `gemma4:26b` | 18 GB | usually too big for ≤16 GB RAM |
| `gemma4:31b` | 20 GB | requires a workstation |
| `llama3.1:8b`, `qwen2.5:7b`, `mistral-nemo` | 4-5 GB | good chat alternatives; desktop-use needs a multimodal variant |
| `llava`, `qwen2.5-vl`, `gemma4:*` | varies | pick one for computer-use / screenshot tools |

The **model picker** at the top of every chat lets you switch per-conversation. Default filter shows only models whose Ollama `capabilities` list includes `tools`; flip the wrench-icon footer toggle to **Show all models** when you want to try one without that flag (Gigachat then auto-falls-back to prompt-space tool calling).

**Auto-tuned default**: at first run the backend probes RAM / VRAM / GPU kind and picks the largest recommended Gemma 4 variant that fits. Override via Settings → **General**.

---

## Permission modes

A header dropdown picks how tool calls are gated, per conversation:

| Mode | Icon | Read tools | Write tools |
|---|---|---|---|
| **Read-only** | 👁 | run silently | refused before approval card |
| **Plan mode** | 📋 | run silently | refused; agent must end with `[PLAN READY]` to unlock the **Execute plan** button |
| **Approve edits** *(default)* | 🛡 | run silently | pause with diff/command/reason card, wait for click |
| **Allow everything** | ⚡ | run silently | run silently |

Approval cards show the *full* command (bash), the *unified diff* (write/edit), and the model's `reason` field. Side-by-side diff toggle is one click.

⚠ **Use Allow everything only when actively watching.** A hostile tool output can try to prompt-inject the model into firing destructive tools. The default Approve edits is the safe choice. Full threat model: [docs/SECURITY.md](./docs/SECURITY.md).

---

## Quality modes

A second header dropdown picks a per-conversation quality mode. Every mode uses **only the chat model the user picked** — small models close the gap to GPT-4 / Claude class by spending more compute on the same model, not by routing to a stronger judge.

| Mode | Compute | Best for |
|---|---|---|
| **Standard** *(default)* | 1× | Cheap chat, low latency. |
| **Refine** | ~2× | Code, writing, reasoning. Same model critiques its own answer (under JSON-schema-constrained decoding) and revises if needed. |
| **Consensus** | ~3-4× | Math and logic. Sample additional candidates at varied temperatures, synthesize the best answer. |
| **Personas** | ~4× | Hard, open-ended questions. Same model, different reasoning-style overlays per sample (analyst / pragmatist / skeptic), synthesize. |
| **Auto** | adaptive | Best default for varied chat. Difficulty heuristic picks refine / consensus / personas — or skips on trivial turns. |

---

## Settings drawer

One sidebar footer button (⚙ Settings) hosts eight tabs:

- **General** — default chat model, hardware summary, auto-pull status.
- **Compute pool** — identity, public-pool toggle, LAN discovery, paired devices with live status + per-workload routing toggles. Single source of truth for "other devices doing work for me."
- **Memories** — global memory CRUD (one entry per row, optional `topic` for grouping; edits propagate immediately, no save button).
- **Secrets** — named API tokens / credentials referenced via `{{secret:NAME}}`. Values hidden by default; click reveal to show one.
- **Schedules** — every queued prompt with next-run / interval / cwd. Add / delete from the UI; rows back the agent's `schedule_task` tool too.
- **Tools** — user-defined Python tools (review, pause, delete, or add new ones with code + schema + deps form).
- **Hooks** — register shell commands at agent lifecycle points (`user_prompt_submit`, `pre_tool` / `post_tool`, `turn_done`). Each receives a structured JSON payload on stdin; stdout is injected back as a system-note. Full guide: [docs/HOOKS.md](./docs/HOOKS.md).
- **Docs** — URL-indexed documentation sites for `docs_search`. Live status chip per site; reindex button.
- **MCP** — external Model Context Protocol servers.

---

## Compute pool & P2P

Pair other Gigachat installs in **Settings → Compute pool** and the host automatically uses their CPU + RAM + GPU + VRAM alongside its own. The router decides per-request whether to keep things on host (fast, no LAN hop), dispatch to a paired peer (LAN-encrypted), or — for models too big to fit one machine — layer-split across the pool via llama.cpp `--rpc`. Speculative decoding recruits idle peers as draft-model accelerators.

**Public pool** (default ON) joins the global Gigachat swarm via a stateless rendezvous service:

- **You donate** spare GPU/CPU cycles when idle.
- **You can use** other peers' GPUs when a model isn't on your local devices. The router prefers local always; the swarm is the fallback.
- **All compute traffic is end-to-end encrypted** (X25519 + ChaCha20-Poly1305 envelopes, sender-ephemeral forward secrecy).
- **Model bytes never flow peer-to-peer** — when nobody local has the model, it auto-pulls from the OFFICIAL Ollama registry.

The project ships pointing at a public Cloud Run rendezvous so a fresh install joins the swarm the moment Public Pool toggles on. Self-host your own from `rendezvous/` if you want full control.

Deep dive: [docs/P2P.md](./docs/P2P.md) (encryption, rendezvous, TURN relay, TLS pinning, fairness scheduler) and [docs/COMPUTE_POOL.md](./docs/COMPUTE_POOL.md) (routing internals, llama.cpp flags, speculative decoding).

---

## LAN access (other devices on your home network)

Two bind modes — pick by who needs to reach the web UI:

| Mode | Reachable from | Use case |
| --- | --- | --- |
| **loopback** *(default)* | Host machine only | Day-to-day solo. Zero config. |
| **`lan`** | Loopback + any device on the same Wi-Fi/Ethernet (RFC1918 ranges) | Phone / tablet / second laptop on your home network. |

Public-internet exposure and Tailscale-overlay access for the web UI are intentionally unsupported. The app is designed to live on a single physical network.

### Enabling LAN mode

Hash a password (PBKDF2-SHA256, 200 000 iterations, 16-byte salt):

```powershell
python -c "from backend.auth import hash_password; print(hash_password('your-password-here'))"
```

Write `data/auth.json`:

```json
{ "host": "lan", "password": "a1b2c3…:d4e5f6…" }
```

Or use env vars (they win over the file): `GIGACHAT_HOST=lan` and `GIGACHAT_PASSWORD=…`. Then `.\start.bat`.

The launcher binds to `0.0.0.0` but the access-control middleware admits only loopback and **private (RFC1918) LAN sources**. Public IPs and Tailscale CGNAT (`100.64.0.0/10`) get a flat 403. Login stores an HMAC-signed session cookie (httponly, SameSite=Lax, 30-day TTL).

---

## Working directory

Each conversation has a `cwd` that all commands run from. The chat-header dialog has a **Browse…** button that opens the native OS folder picker; the chosen path is validated server-side. Once set, `cwd` is immutable.

**`AGENTS.md` / `CLAUDE.md` auto-injection** — on every turn the backend walks from `cwd` up to the filesystem root and concatenates every `AGENTS.md` and `CLAUDE.md` it finds into the system prompt (outermost first, innermost last — nearer-in instructions win). Both names are treated equally so a repo that ships only one still works; nested sub-projects can override parent rules.

**File checkpoints** — every `write_file` / `edit_file` snapshots the prior contents under `data/checkpoints/<conv_id>/<stamp>/<hash>.bin` and exposes a one-click restore.

---

## Safety basics

- **Default bind is 127.0.0.1.** Nothing on your LAN reaches it until you opt in.
- **Approve edits is the safe default.** A hostile tool output can try to prompt-inject the model.
- **Use a strong random password** in LAN mode (`python -c "import secrets; print(secrets.token_urlsafe(24))"`).
- **Computer use controls your real desktop.** Close private windows before handing the mouse over. Don't ask the agent to type passwords or 2FA codes.
- **Scheduled tasks run unattended in Allow-everything mode.** Be specific in the prompt.

Full threat model + risk catalog: [docs/SECURITY.md](./docs/SECURITY.md).

---

## Tests

```
python -m pytest -m smoke         # fast tier, ~70 s, 488 tests
python -m pytest                  # everything (Windows-only tests skipped on Linux)

# One-time setup so `git push` runs the smoke tier automatically:
git config core.hooksPath .githooks
```

The `isolated_db` fixture rewires `db.DB_PATH` to a tmp file per test, so the suite never touches `data/app.db`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Ollama not reachable" toast | Run `ollama serve` in a terminal. |
| Model picker is empty | `ollama list` to confirm; `ollama pull gemma4:e4b`. |
| Responses very slow | Likely swapping model weights to RAM. Try `gemma4:e2b`. |
| Approval click does nothing | Check the backend console for errors. |
| Dev server port 5173 in use | Kill the other Vite process or change the port in `frontend/vite.config.js`. |
| `web_search` rate-limited | DuckDuckGo occasionally rate-limits. Wait a minute; persistent? `pip install -U ddgs`. |
| `doc_index` / `doc_search`: "no vector" | `ollama pull nomic-embed-text`. |
| Settings → Compute pool: rendezvous "Disconnected" / "Not configured" | Confirm Public Pool toggle is on. The default Cloud Run URL ships with the app; override or self-host via the URL editor. |
| `PermissionError: [Errno 13] Permission denied: '...\\AppData\\Roaming\\Python\\Python3xx\\site-packages\\typing_extensions.py'` on backend startup | Mixed system + user site-packages install. Cleanest fix: `.\setup.bat` (creates `.venv\` and installs every dep there; `dev.bat` / `start.bat` auto-detect it). Quick patch without venv: `del "%APPDATA%\Python\Python3xx\site-packages\typing_extensions.py"` then `python -m pip install --user typing_extensions`. |
| `pip uninstall typing-extensions` fails with `uninstall-no-record-file` | The existing copy was put there manually / by a partial install — pip can't safely remove it. Either delete the file by hand (`del "%APPDATA%\Python\Python3xx\site-packages\typing_extensions.py"`) or just run `.\setup.bat` and use the venv-aware launchers, which bypass the global install entirely. |
| LAN mode: another device gets a 403 | Its source IP isn't in an RFC1918 range. Confirm both devices are on the same physical network. Tailscale CGNAT (`100.64.0.0/10`) is intentionally refused. |

More: see [docs/SECURITY.md](./docs/SECURITY.md) for risk-specific knobs and [docs/COMPUTE_POOL.md](./docs/COMPUTE_POOL.md) for pool-routing diagnostics.

---

## Documentation index

- [docs/TOOLS.md](./docs/TOOLS.md) — full catalog of every tool the agent can call.
- [docs/COMPUTE_POOL.md](./docs/COMPUTE_POOL.md) — routing internals, llama.cpp flags, speculative decoding, override-file mechanism.
- [docs/P2P.md](./docs/P2P.md) — P2P encryption, rendezvous, TURN relay, TLS pinning, fairness scheduler, API surface.
- [docs/SECURITY.md](./docs/SECURITY.md) — full threat model + risk catalog.
- [docs/HOOKS.md](./docs/HOOKS.md) — lifecycle hooks deep dive + recipes.
- [ARCHITECTURE.md](./ARCHITECTURE.md) — for contributors: turn flow, load-bearing invariants, where to change what.
- [rendezvous/README.md](./rendezvous/README.md) — deploying the rendezvous service to Cloud Run / a VPS.
