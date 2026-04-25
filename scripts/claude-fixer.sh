#!/usr/bin/env bash
# Reads Gigachat's payload on stdin, hands it to Claude Code, returns
# the verdict on stdout. Whatever this prints lands as a system-note
# in the conversation that triggered it, so the agent sees it.
#
# Uses python (already on the box for the Gigachat backend) for JSON
# parsing instead of jq, so this works on a vanilla Windows + Git Bash
# install without the extra dependency.

set -euo pipefail

# Build the diagnosis prompt entirely in python — including the
# multiline template — and pipe it straight into `claude --print`. We
# previously tried to extract individual fields back to bash via
# `\0`-separated output, but bash's `$(...)` command substitution
# silently strips embedded NUL bytes, which collapsed the fields into
# one string. Doing all the formatting in python sidesteps that
# entirely; bash just hands stdin → python, then python's stdout →
# claude.

prompt=$(python -X utf8 -c '
import sys, json
d = json.loads(sys.stdin.read())
conv_id = d.get("conversation_id", "")
tool_name = d.get("tool_name", "")
streak = d.get("consecutive_count", 0)
err = (d.get("error") or d.get("output") or "")[:2000]
sys.stdout.write(f"""\
A Gigachat conversation is wedged: tool {tool_name!r} just failed
{streak} times in a row in conversation {conv_id}. Last error:

{err}

Decide: is this a real Gigachat-source bug or a model-side mistake?

  - If real bug: read the relevant files under \
C:/Users/gauta/Downloads/Gigachat/backend, fix it, run \
`python -m pytest -m smoke`, and commit. Reply in ONE PARAGRAPH \
summarizing what you fixed.
  - If model mistake: reply in ONE PARAGRAPH explaining what the agent \
should try differently. No code changes.

Either way: keep your reply under 600 chars and end with a clear \
next-step the model should take.
""")
')

# `claude --print` (or `-p`) is the non-interactive mode — print the
# assistant's response to stdout and exit. The 540 s cap leaves
# headroom under the hook's 600 s ceiling so the timeout has time to
# tear down the subprocess cleanly.
timeout 540 claude --print "$prompt" 2>&1 | tail -c 1500
