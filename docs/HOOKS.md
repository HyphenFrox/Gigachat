# Workflows (lifecycle hooks)

Settings → **Hooks** lets you register shell commands that fire at well-known points in the agent loop. Each hook receives a structured JSON payload on stdin; whatever it prints to stdout gets injected back into the conversation as a system-role message so the model sees it on the next turn. That's all — but it composes into surprisingly powerful workflows.

## Trigger events

| Event | Fires when |
|---|---|
| `user_prompt_submit` | Once per user turn, before the model runs. |
| `pre_tool` / `post_tool` | Around each tool call (optional tool-name substring matcher). |
| `tool_error` | A tool call returned `ok=False`. Cleaner than filtering inside a `post_tool` hook. |
| `consecutive_failures` | After **N** back-to-back failures of the same tool in the same conversation (N is per-hook via the `error_threshold` field, default 1). The "model is looping on the same broken call" signal. |
| `turn_done` | When the agent produces a final answer. |

## Per-hook settings

- **Tool matcher** — case-insensitive substring against the tool name. Empty = match every tool.
- **Timeout** — hard cap on the shell command. 120 s for the original four events, **900 s for `tool_error` / `consecutive_failures`** so a long-running diagnosis (Claude CLI, full pytest run) has time to finish.
- **Max fires per conversation** — cap on how often the hook can fire in one chat. Persisted in `hook_fires`, so a backend restart can't reset the counter and re-open a runaway loop. Empty = unlimited; **set this to 5-10 for any hook that calls a paid API**.

⚠ **Hooks run with your full login shell privileges.** Each is a shell string you entered via the UI. Only register hooks you wrote yourself. See [SECURITY.md](./SECURITY.md#hooks--user-tools) for the full sandboxing story (there isn't one — they're trusted input by design).

---

## Example: Claude self-fixer

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

The script — already shipped at [scripts/claude-fixer.sh](../scripts/claude-fixer.sh) — uses python (already on the box for the Gigachat backend) for JSON parsing, so it works on a vanilla Windows + Git Bash install with no extra binaries:

```bash
#!/usr/bin/env bash
set -euo pipefail

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

---

## Other useful workflow recipes

| Goal | Event | Matcher | Command (one-liner) |
|---|---|---|---|
| Lint after every successful file write | `post_tool` | `write_file` | `python -X utf8 -c "import sys, json; print(json.loads(sys.stdin.read())['tool_args']['path'])" \| xargs eslint --fix` |
| Slack me when a turn finishes | `turn_done` | — | `curl -X POST $SLACK_WEBHOOK -d @-` |
| Run pytest after every model edit | `post_tool` | `edit_file` | `cd "$(python -X utf8 -c "import sys, json; print(json.loads(sys.stdin.read())['cwd'])")" && pytest -m smoke -q 2>&1 \| tail -20` |
| Block tool calls that touch a sensitive path | `pre_tool` | — | `python -X utf8 -c "import sys, json; d=json.loads(sys.stdin.read()); sys.exit(1 if '/secrets/' in (d.get('tool_args', {}).get('path') or '') else 0)"` |

Each row is one entry in Settings → **Hooks**. No parallel "Workflows" subsystem to learn — the hooks panel IS the workflow builder.
