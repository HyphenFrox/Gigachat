# Safety & security

This document is the full threat model. The README has the headline rules; this is the long form for anyone exposing Gigachat to a network or running it unattended.

For the P2P-specific crypto + envelope protocol see [P2P.md](./P2P.md).

---

## Network exposure

- **Default bind is `0.0.0.0`** so other devices on the same physical network can reach the **P2P endpoints** (`/api/p2p/secure/*`, pair handshake, identity / discover). Their X25519 + Ed25519 envelope crypto IS the auth — no password layer is needed.
- **Chat UI is loopback-only.** The `AuthMiddleware` in `app.py` returns a 403 with a clear "loopback only — install Gigachat on the other device and pair via Compute pool" for any non-loopback request that isn't a P2P endpoint or static asset. Each device runs its own local UI; cross-device chat from another device's browser isn't a supported use case.
- **Public IPs / Tailscale CGNAT** (`100.64.0.0/10`) — flat 403. The app stays on the user's own physical network.
- **Hard isolation** — set `GIGACHAT_HOST=127.0.0.1` and the backend binds loopback-only. Nothing on the LAN reaches anything, not even the P2P endpoints. Use on untrusted public Wi-Fi.
- **No password feature.** Earlier versions had an opt-in LAN-web-UI mode with a PBKDF2 password gate; that was removed. The only reason to access the chat UI from another device on your LAN was historical, and the cleaner answer is "install Gigachat there too and pair the two via Compute pool."

## Data storage

- **Parameterized SQL end-to-end.** No string concat into SQL.
- **Sensitive SQLite columns encrypted at rest** — see [P2P.md → At-rest encryption](./P2P.md#at-rest-encryption-sensitive-sqlite-columns).
- **Per-tool runtime caps.** 120-second default timeout, 20 000-character output cap.

## Permission model

- **Approve edits is the safe default** for new conversations. Read-only is great for "let the model poke around but don't let it touch anything." Allow everything is for watched sessions or scheduled jobs only.

⚠ The whole point of the app is that a local LLM can run commands on your PC. Treat it like any other agentic tool: review before approving destructive actions (delete, overwrite, `rm -rf`, package installs).

---

## Known risks

### Prompt injection via tool output

A file or command's output is fed back to the model, so a hostile file could try to trick the model with "ignore prior instructions" content. Keep the permission mode on **Approve edits** (or **Read-only**) for any conversation that touches untrusted data (email, downloads, clipboard, web scrapes, images). You'll see every proposed follow-up before it runs.

### Filesystem access

- **No path sandbox.** The agent can read and write anywhere your user account can. Point `cwd` at the narrowest folder that makes sense. `edit_file` / `write_file` checkpoint prior contents so you can restore after a bad edit.
- **Image and file uploads** — streaming size cap (10 MB), content-type allowlist (`image/png|jpeg|webp|gif`), random-hex filenames so a caller can't overwrite arbitrary files by picking a name.
- **Background shells (`bash_bg`)** keep running until the conversation is deleted or you call `kill_shell`. They inherit the same env and FS access as foreground bash — treat them like any other shell you left open.

### Computer use

Computer use controls your real desktop. Screenshots include every visible window. Mouse/keyboard events are issued as your actual logged-in user — the agent can click "OK" on system dialogs, drag files into the trash, type into password fields.

- Keep permission mode on **Approve edits** when first enabling — every click, keypress, and scroll pauses for confirmation with a thumbnail of the screen the model is reacting to.
- Close private windows (banking, messages, password manager) before handing the mouse over.
- Don't ask the agent to enter passwords, PINs, or 2FA codes; type those yourself.
- Move the mouse into a screen corner for ~1 s and pyautogui's failsafe aborts the next action.
- **`computer_batch` is allowlisted, not a generic eval.** Only desktop primitives can appear (move/click/drag/type/key/scroll/wait_ms/focus/window/click_element/click_element_id/open_app/ocr) — `bash`, `read_file`, `write_file`, `browser_*`, `delegate`, `schedule_task`, etc. are explicitly rejected. Caps: 20 steps per call, 5 s max per `wait_ms`, 100 ms inter-step settle.
- **`click_element_id` IDs are process-scoped.** Minted by `inspect_window` or `screenshot(with_elements=true)`, kept in an in-memory `OrderedDict` (max 5000, LRU-evicted) guarded by a lock so concurrent subagents can't race the counter. IDs do not survive a backend restart and do not survive UI movement — re-`inspect_window` to mint fresh ones.
- **`ui_wait` is bounded.** Max 30 s, ~250 ms poll. Six kinds: `window` / `window_gone`, `element` / `element_enabled`, `text` (OCR), `pixel_change`. Prefer `window_gone` / `element_enabled` over `pixel_change` when a deterministic signal is available.
- **Status-context tag** (`[ctx: foreground='...'; focused='...'; cursor=(x,y)]`) is a read-only UIA + cursor snapshot taken on every screenshot. Window titles and accessible names come from untrusted sources, so they're length-capped (80 chars) and rendered through Python's `!r` repr — embedded newlines / quotes are escaped, so a hostile aria-label can't smuggle a fake instruction line.
- **Focus-drift warnings** on `computer_type` / `computer_key` use the same read-only UIA query before and after each action. Surface focus theft (a popup grabbed the caret mid-typing) so the model retries instead of trusting a silent miss.

### Web access

Web access pulls untrusted content into the conversation. Pages from `fetch_url` are treated like any other tool output — a hostile page could try to prompt-inject the model. Mitigations:

- `fetch_url` rejects non-http(s) schemes and any URL whose host is loopback / private / link-local / multicast — including DNS-resolved hostnames.
- HTML capped at 2 MB on the wire, extracted prose at ~15 000 chars.
- System prompt explicitly tells the model to treat fetched content as untrusted.
- Manual approval is the real defense — you see the URL *and* a preview of the extracted text.

**Browser-automation tools drive a real Chrome tab.** A hostile page reached via `browser_goto` can prompt-inject through `browser_text`. `browser_eval` runs arbitrary JS in the page context — it can read cookies, localStorage, and DOM of whatever site the tab is on. Keep the CDP browser pointed at throwaway / agent-only sessions, not the profile where your bank is logged in.

### Autonomous execution

- **Scheduled tasks run unattended in Allow-everything mode.** `schedule_task` opens a brand-new conversation in **Allow everything** by design (nobody's watching at 3 AM). Treat the prompt like a cron job: be specific, avoid telling the model to "do whatever" based on fetched web content, never schedule a prompt itself pulled from an untrusted source.
- **Autonomous loops (`start_loop`) inherit the conversation's permission mode.** A loop fires the rolling `goal` back into the *existing* conversation. An **Allow everything** chat with an active loop will click through every write call unattended every `interval_seconds` until you click **Stop loop**. Start loops only on chats you're comfortable letting run autonomously; prefer **Approve edits** if you're stepping away. Intervals clamped 60 s – 1 h, goals capped at 4 000 chars.

### Storage of user data

- **Codebase index walks the entire cwd.** `codebase_search` is read-only but the *builder* opens every matching file (up to 1 500 files, 2 MB each, allowlisted extensions) to embed it locally. Two consequences: (1) chunks land in `data/app.db` — if you indexed a cwd containing a `.env` / credential file matching the allowlist, the secret is now duplicated there. The git-aware walker is the main defence (`git ls-files --exclude-standard` respects `.gitignore`); (2) on a non-git cwd the fallback walker follows symlinks via `rglob`, so a symlink to your home directory could pull files outside the cwd into the index. Point `cwd` at the narrowest folder.
- **`doc_index` / `doc_search` store raw file contents in SQLite.** Chunks are kept verbatim (so retrieved context is readable). Any secret inside a file you indexed is now duplicated in `data/app.db`.
- **Global memories are injected into every system prompt** — including subagents. Two consequences: (1) avoid storing secrets, entries are not encrypted; (2) the agent can extend its own behaviour across chats, so review the panel periodically. Length caps: 8 KB per entry, 80-char topic. `forget(scope="global", pattern="")` is refused so a typo can't wipe the table.

### Hooks & user tools

- **Lifecycle hooks run arbitrary shell commands.** Each is a shell string you entered via the UI, run with your full login shell privileges on every matching event. CRUD endpoints are bound to localhost, no CORS headers, JSON payload passed on stdin (not interpolated into the command). The command itself is trusted input by design — only register hooks you wrote yourself.
- **User-defined tools run arbitrary Python in a shared venv.** Only the user can create them — the LLM has no self-extension route. Code is `ast.parse`-validated (must define `def run(args)`, must parse cleanly) but **NOT sandboxed beyond that**. Layers:
  1. Creation gated behind the Settings UI — you review code and dep list before first install.
  2. Name regex `^[a-z][a-z0-9_]{0,47}$` blocks collisions with built-ins / MCP / SQLite tricks.
  3. Dep-spec regex matches a PEP 508 subset (name + extras + version comparators only — **no URLs, no VCS URIs, no file paths**).
  4. Blocklist refuses `pip` / `setuptools` / `wheel` / `distribute`.
  5. Pip runs `--disable-pip-version-check --no-input` in a 300 s subprocess that can't read stdin.
  6. Wrapper runs the tool with `python -I` (isolated mode — ignores `PYTHONPATH` / user site-packages / startup scripts), args via stdin JSON, stdout parsed at a sentinel line.
  7. Each tool stores its own `timeout_seconds` (1-600 s) and `category` (read/write) — model can't override at call time.
  8. **Kill switch:** `GIGACHAT_DISABLE_USER_TOOLS=1` skips schema registration for existing rows and refuses execution.

### Other

- **`monitor` is read-only but probes the network.** `url:` reuses the SSRF guard (rejects loopback / RFC1918 / link-local / reserved, including DNS resolution). `bash:` inherits `run_bash`'s 30-second per-tick cap. Total wait time clamped to 30 minutes.
- **`http_request` calls arbitrary APIs with your credentials.** Write-class regardless of method (GET included). Same SSRF guard as `fetch_url`; `allow_private: true` opts into LAN. Method allowlist `GET/POST/PUT/PATCH/DELETE/HEAD/OPTIONS`. Response capped at 2 MB on the wire, 20 000 chars in tool output. `Authorization` / `Set-Cookie` / `X-API-Key` / `Cookie` headers are masked in echoed request + response summaries.
- **Secrets live in SQLite in plaintext.** Threat model is single-user local: anyone with read access to `data/app.db` already has read access to everything under your user profile. Values never reach the model — agent references them as `{{secret:NAME}}`; backend substitutes just before the wire. Names regex `^[A-Za-z_][A-Za-z0-9_]{0,63}$`; values capped at 16 000 chars; descriptions at 400; `UNIQUE(name)` prevents silent overrides. **Defence-in-depth:** any substituted value is scrubbed from the response body before the tool result is stored — even a misconfigured server echoing `Authorization` back doesn't land the credential in the transcript. Tiny values (<4 chars) are not scrubbed (false-positive rate too high on random 4-byte substrings).
- **`delegate_parallel` concurrency is capped.** Max 6 subagents per call, each bounded by `max_iterations` (default 10, max 20). Each gets the trimmed tool set — no nested delegation, no desktop / browser / scheduling.

---

## P2P-specific risks

For the full P2P security model see [P2P.md](./P2P.md). Headline:

- All paired-peer traffic is end-to-end encrypted (X25519 + ChaCha20-Poly1305).
- Forward secrecy on the sender side (per-envelope ephemeral X25519).
- Public-pool consumer path: when you pick a model your local pool doesn't have, your prompt CAN dispatch to a peer in the swarm. The prompt is encrypted on the wire but it does cross the internet. Toggle Public Pool off to disable the consumer path entirely.
- The rendezvous server sees ONLY identity + STUN endpoints — no prompts, no model lists, no chat data.
