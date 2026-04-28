"""Agent loop: talks to Ollama, streams tokens, runs tool calls.

Emits SSE-ready events over an async generator. Upstream (FastAPI) turns these
into Server-Sent Events and forwards to the browser.

Event shape (dict with 'type' + payload fields):
  {type: "delta", text: "..."}                  # assistant prose token(s)
  {type: "thinking", text: "..."}               # optional reasoning tokens
  {type: "tool_call", id, name, args, label}    # a tool was requested
  {type: "await_approval", id, name, args,
       label, reason, preview}                  # manual mode: waiting on user
                                                # preview is populated for
                                                # write_file/edit_file to give
                                                # the UI a diff to render.
  {type: "tool_result", id, ok, output, error, image_path}
                                                # tool finished — image_path
                                                # set only for computer-use
                                                # tools (screenshot etc.)
  {type: "assistant_message", id, content, tool_calls}
                                                # persisted row for UI
  {type: "todos_updated", todos}                # emitted whenever the model
                                                # calls todo_write — frontend
                                                # re-renders its task panel
  {type: "turn_done"}
  {type: "error", message}
"""

from __future__ import annotations

import asyncio
import base64
import difflib
import json
import math
import re
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx

from . import compute_pool, db, mcp, sysdetect, tool_prompt_adapter, tools
from .prompts import TOOL_SCHEMAS, build_system_prompt

OLLAMA_URL = "http://localhost:11434"
MAX_TOOL_ITERATIONS = 25

# In-memory registry of conversation ids that currently have an active
# `run_turn` async generator iterating in this worker process. Registered
# at the top of the wrapper and discarded in its `finally` block.
#
# A conversation with `state='running'` in the DB whose id is NOT in this
# set is definitionally stuck — the wrapper's finally never fired. This
# happens when uvicorn's --reload kills the worker mid-turn, or when the
# event loop is torn down before an async generator can clean up. The
# stale-turn watchdog in `app.py` reads this set to distinguish genuine
# in-flight turns from abandoned ones.
_ACTIVE_TURN_IDS: set[str] = set()


def is_turn_active(conversation_id: str) -> bool:
    """Return True if a turn is currently being iterated for this conversation
    in this worker process. Used by the stale-turn watchdog to decide
    whether a `state='running'` DB row is live or abandoned.
    """
    return conversation_id in _ACTIVE_TURN_IDS

# Context window the model runs with. Auto-tuned at startup by `sysdetect`
# so a weaker laptop and a beefy desktop both get a value that fits without
# forcing Ollama to spill the KV cache into system RAM (which tanks throughput
# by ~3x). The same value feeds the compaction threshold below AND gets
# returned by /api/system/config so the frontend's token meter uses the
# right denominator without hardcoding it twice.
#
# Override with the MM_NUM_CTX env var when you want to force a specific
# value (testing, specialty hardware that the auto-tune doesn't know about).
NUM_CTX = sysdetect.recommend_num_ctx()
COMPACTION_THRESHOLD = 0.75  # compact when prompt + history ≥ 75% of num_ctx
# Rough chars-per-token heuristic. English-ish text averages ~4 chars/token on
# most tokenizers; we're not trying to be precise, just to react before Ollama
# starts truncating from the front silently.
CHARS_PER_TOKEN = 4

# How many of the most-recent screenshot results keep their image attached
# when we replay history to Ollama. Older ones become text-only descriptors
# ("[screenshot from earlier turn — image elided to save context]"). Each
# attached PNG costs hundreds of vision tokens — keeping all of them through
# a 50-turn session is the dominant token cost in a screenshot-heavy
# session. Five recent frames is enough for the model to compare "what
# changed" while not paying for stale views the next click invalidates
# anyway. The most-recent screenshot is ALWAYS kept, regardless of this
# value, so the model can always see the current state.
KEEP_RECENT_SCREENSHOT_IMAGES = 5

# When a tool row's text body grows past this many chars, the auto-compactor
# treats it as "bulky" and is more aggressive about replacing it with a
# head+tail preview. A 2 KB threshold covers any meaningful bash dump,
# grep result, or file read while leaving short status lines (`done`,
# `pressed: enter`) untouched.
BULKY_TOOL_OUTPUT_CHARS = 2000

# Tool rows older than this many positions (counting from the tail) get
# proactively shrunk to head+tail snippets, even when we're nowhere near the
# context budget. Without this, a session that has a giant bash output but
# stays under 75% of num_ctx will keep paying that cost on every turn.
PROACTIVE_TOOL_AGE_TURNS = 25

# Local embedding model for semantic recall (RAG over earlier messages).
# Runs on the same Ollama instance. Pull once with `ollama pull nomic-embed-text`.
# If not installed, recall silently degrades to a no-op — the chat still works,
# just without long-range memory boost.
EMBED_MODEL = "nomic-embed-text"
RECALL_TOP_K = 5          # at most this many recalled messages injected per turn
RECALL_MIN_SCORE = 0.45   # cosine threshold below which a hit is ignored
RECALL_EXCLUDE_TAIL = 15  # don't recall messages already present in the recent tail


# In-memory pending approvals: approval_id -> (future, loop)
_pending_approvals: dict[str, tuple[asyncio.Future, asyncio.AbstractEventLoop]] = {}


# Pending AskUserQuestion prompts — keyed by tool_call_id. The agent yields an
# `await_user_answer` SSE event, the frontend renders the buttons; clicking
# one posts to /api/conversations/<conv>/answer which calls resolve_answer()
# to set the stored future's result, unblocking the agent loop.
_pending_answers: dict[str, tuple[asyncio.Future, asyncio.AbstractEventLoop]] = {}


def resolve_answer(answer_id: str, value: str) -> bool:
    """Called by the HTTP endpoint when the user clicks one of the option
    buttons. Returns True if a waiting future was resolved, False if the id
    is unknown (already resolved, expired, or never existed).
    """
    entry = _pending_answers.get(answer_id)
    if not entry:
        return False
    fut, loop = entry
    if not fut.done():
        loop.call_soon_threadsafe(fut.set_result, value)
    return True


# Conversation ids the user has asked to stop. Relying solely on the client
# closing the SSE connection is unreliable: the agent loop may be blocked on a
# long Ollama round-trip or a slow tool when the disconnect arrives, and by
# the time it yields again the browser has moved on. An explicit flag that
# the loop polls between chunks / tools lets the Stop button feel instant
# regardless of where in the pipeline the turn is currently stuck.
_stop_requests: set[str] = set()


# --- Subagent progress bus ------------------------------------------------
#
# Subagents (run_subagent / run_subagents_parallel) execute as nested loops
# inside a single `delegate` / `delegate_parallel` tool call. From the parent
# turn's perspective that's one long-running blocking tool; the UI otherwise
# sees no activity until the subagent returns. That's a bad user experience
# for delegations that fan out five parallel explorers for 30s each.
#
# The bus gives subagents a non-DB, in-process channel to publish progress
# events that the parent turn forwards onto its SSE stream. Keyed by the
# PARENT conversation id (subagents have no conv_id of their own), each
# entry is an asyncio.Queue the parent drains concurrently with its own
# tool_event_queue. When no parent is registered (e.g. a subagent invoked
# from a test or the CLI), publishes are dropped on the floor — they're
# observational.
_SUBAGENT_PROGRESS_BUS: dict[str, asyncio.Queue] = {}

# Per-conversation consecutive-failure tracker for the
# `consecutive_failures` hook event. Maps conv_id → {tool_name → count}.
# Process-local: resets on backend restart, which is fine because the
# user-facing cap (`hooks.max_fires_per_conv`) is what bounds runaway
# loops; this tracker is just the trigger.
_CONSEC_FAILURES: dict[str, dict[str, int]] = {}


def _bump_consec_failures(conv_id: str, tool_name: str, ok: bool) -> int:
    """Update the consecutive-failure tally for (conv, tool). Returns the
    NEW count after the update. Resets to 0 on success."""
    bucket = _CONSEC_FAILURES.setdefault(conv_id, {})
    if ok:
        bucket[tool_name] = 0
        return 0
    bucket[tool_name] = bucket.get(tool_name, 0) + 1
    return bucket[tool_name]


def _register_subagent_bus(parent_conv_id: str) -> asyncio.Queue:
    """Create (or reuse) the progress queue for this parent conversation.
    Called from `_run_turn_impl` at turn start.
    """
    q = _SUBAGENT_PROGRESS_BUS.get(parent_conv_id)
    if q is None:
        q = asyncio.Queue()
        _SUBAGENT_PROGRESS_BUS[parent_conv_id] = q
    return q


def _unregister_subagent_bus(parent_conv_id: str) -> None:
    """Drop the queue on turn end so a later turn starts fresh."""
    _SUBAGENT_PROGRESS_BUS.pop(parent_conv_id, None)


def _publish_subagent_event(parent_conv_id: str | None, event: dict) -> None:
    """Non-blocking publish. Safe to call from inside a subagent running on
    the same event loop as the parent turn — we use `put_nowait` and swallow
    the QueueFull error on the (extremely unlikely) case the queue is
    saturated, because losing a progress chip is better than stalling a
    subagent waiting to enqueue it.
    """
    if not parent_conv_id:
        return
    q = _SUBAGENT_PROGRESS_BUS.get(parent_conv_id)
    if q is None:
        return
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        pass


def request_stop(conversation_id: str) -> None:
    """Mark a conversation as 'please stop after the next checkpoint'.

    Idempotent: if the turn has already finished by the time the request
    lands, the flag is cleared on the next run_turn call so a subsequent
    turn doesn't stop immediately.
    """
    _stop_requests.add(conversation_id)


def is_stop_requested(conversation_id: str) -> bool:
    """True if a stop was requested for this conversation since the last
    run_turn start."""
    return conversation_id in _stop_requests


def _clear_stop(conversation_id: str) -> None:
    _stop_requests.discard(conversation_id)


# --- Empty-response retry -------------------------------------------------
#
# Thinking-capable models (gemma4:26b, DeepSeek R1, etc.) occasionally
# emit only reasoning tokens and close the stream before any `content`
# lands, leaving an empty assistant bubble. That's a generation-layer
# quirk — not a model-semantic refusal — so we nudge the model to emit
# something visible and retry once.
#
# NOTE: we intentionally do NOT try to detect or retry refusals /
# apologies / placeholder greetings. Those are root-cause problems with
# the underlying model (Gemma's RLHF refusal reflex, long-prompt
# attention drift, etc.) and trying to paper over them with prompt
# patches turned into whack-a-mole. Fix those at the model / prompt
# layer instead (switch default model, shrink system prompt, force
# tool_choice).
_MAX_EMPTY_RETRIES = 1


_EMPTY_NUDGE = (
    "Your previous reply was empty — you produced reasoning but no visible "
    "answer and no tool call. The user sees a blank bubble. Reasoning mode "
    "has now been DISABLED. Emit your final answer immediately as plain "
    "text, or call the appropriate tool. Do NOT think, reflect, or plan — "
    "respond directly with the result."
)

# Retry nudge when the model started emitting a <tool_call> block but the
# stream ended before </tool_call> (output-token limit reached mid-JSON).
# The user sees a wall of broken JSON; we clear it and ask the model to
# retry with a smaller payload. Common trigger: write_file called with a
# huge `content` arg — the model should prefer edit_file in those cases.
_TRUNCATED_TOOL_CALL_NUDGE = (
    "Your previous <tool_call> was CUT OFF before </tool_call> — the output "
    "window filled up mid-generation, so the tool never ran and the user "
    "saw broken JSON. Retry with a SMALLER payload:\n"
    "- For editing existing files, use `edit_file` with a precise "
    "`old_string` (read the file first with `read_file` so you know the "
    "exact bytes to match).\n"
    "- For writing brand-new files whose total size fits, `write_file` is "
    "fine — but if the content is large, split the work across multiple "
    "sequential `edit_file` calls.\n"
    "Emit the retry now. Do NOT narrate the failure first."
)


# XML-style tag names that carry adapter-mode tool invocations. Kept in sync
# with `tool_prompt_adapter._TOOL_CALL_TAGS` — any model flavour that wants to
# invoke a tool emits one of these wrapping a JSON body. Used both by the
# post-stream parser (so the call actually runs) and by `_StreamTagFilter`
# below (so the raw JSON doesn't leak into the UI mid-stream).
_TOOL_CALL_TAGS = ("tool_call", "execute_tool", "tool_code", "function_call")


class _StreamTagFilter:
    """Chunk-by-chunk filter that suppresses `<tool_call>…</tool_call>` and
    sibling blocks from a streaming text channel.

    Why this exists: in adapter mode the model emits tool invocations as
    inline XML inside its `content` (and, on some models, `thinking`) stream.
    The post-stream adapter strips those tags from the persisted message, but
    the raw JSON still flashes into the UI chunk-by-chunk while the stream is
    live — the user sees a wall of `{"name": "edit_file", "args": {"path": "…",
    "new_content": "…\\n…\\n…"}}` and a horizontal scrollbar from the escaped
    newlines.

    This filter wraps the delta/thinking channel so the UI only ever sees the
    prose. `accumulated_text` / `accumulated_thinking` still receive the full
    raw stream so the existing parser + truncation detector keep working.

    Implementation notes:
    * Handles openers/closers that split across chunks — we hold a short tail
      in the buffer that could be a partial opener prefix, or a partial close
      tag when we're inside a block.
    * Drops the body of any block that's still open at `flush()` time. The
      existing truncation-detector in `run_turn` notices unclosed tags in
      `accumulated_text` and triggers a retry, so we don't lose the user's
      answer — we just don't flicker broken JSON at them.
    * Stateless across turns; caller creates a fresh instance per stream
      iteration of the retry loop so leftover buffer from a retried request
      can't bleed into the replacement.
    """

    _OPEN_PATTERNS = tuple(f"<{t}>" for t in _TOOL_CALL_TAGS)
    _MAX_OPEN_LEN = max(len(p) for p in _OPEN_PATTERNS)

    def __init__(self) -> None:
        # Text not yet emitted. When we're inside a block this holds a short
        # tail kept in case a close tag splits across chunks; when we're not
        # inside a block it holds a short tail that could be a partial opener.
        self._buf = ""
        # Name of the currently-open tag (e.g. `tool_call`), or None when we
        # are in normal emit-everything mode.
        self._in_tag: str | None = None

    def feed(self, chunk: str) -> str:
        """Consume `chunk` and return whatever text is now safe to emit."""
        if not chunk:
            return ""
        self._buf += chunk
        out_parts: list[str] = []

        # Loop so one chunk can both close a tag AND open a new one.
        while True:
            if self._in_tag is not None:
                close = f"</{self._in_tag}>"
                idx = self._buf.find(close)
                if idx < 0:
                    # No closer yet — drop everything buffered so far except
                    # a tail short enough to be the start of a close tag
                    # continuing in the next chunk.
                    hold = len(close) - 1
                    if len(self._buf) > hold:
                        self._buf = self._buf[-hold:]
                    return "".join(out_parts)
                # Found a close tag: drop the body + tag, re-enter normal state.
                self._buf = self._buf[idx + len(close):]
                self._in_tag = None
                continue

            # Normal mode — look for the earliest full opener in buffer.
            opener_idx = -1
            opener_tag: str | None = None
            for tag, pat in zip(_TOOL_CALL_TAGS, self._OPEN_PATTERNS):
                i = self._buf.find(pat)
                if i >= 0 and (opener_idx < 0 or i < opener_idx):
                    opener_idx = i
                    opener_tag = tag

            if opener_idx < 0:
                # No complete opener — emit everything except a trailing
                # prefix that could still grow into an opener.
                hold = 0
                max_tail = min(len(self._buf), self._MAX_OPEN_LEN - 1)
                for h in range(1, max_tail + 1):
                    tail = self._buf[-h:]
                    if any(op.startswith(tail) for op in self._OPEN_PATTERNS):
                        hold = h
                if hold:
                    out_parts.append(self._buf[:-hold])
                    self._buf = self._buf[-hold:]
                else:
                    out_parts.append(self._buf)
                    self._buf = ""
                return "".join(out_parts)

            # Opener found: emit everything before it, then enter the tag.
            if opener_idx > 0:
                out_parts.append(self._buf[:opener_idx])
            assert opener_tag is not None
            self._buf = self._buf[opener_idx + len(f"<{opener_tag}>"):]
            self._in_tag = opener_tag
            # fall through into the in-tag branch on the next loop turn

    def flush(self) -> str:
        """Emit any safely-held tail at end of stream.

        If we're still inside an unclosed tag the body is dropped; the
        truncation detector in `run_turn` handles that case via retry.
        Otherwise the held tail couldn't grow into an opener anymore, so
        we release it as literal text.
        """
        if self._in_tag is not None:
            self._buf = ""
            return ""
        out, self._buf = self._buf, ""
        return out


# Queue of user inputs submitted while a turn is still running.
# DB-backed (see db.enqueue_user_input / db.drain_queued_inputs) so queued
# messages survive a server crash. The active `run_turn` loop drains the
# queue between iterations and persists each entry as a new user message
# before the next Ollama call. The startup resumer drains the same queue
# when it restarts an interrupted turn — so anything the user typed just
# before the crash gets replayed, not dropped.
def enqueue_user_input(
    conversation_id: str,
    text: str,
    images: list[str] | None = None,
) -> bool:
    """Append a user message to the queue for `conversation_id`.

    The next iteration of `run_turn` for this conversation will pick it up,
    persist it as a user row, and feed it to the model. Returns True if
    something was actually enqueued (text or images present), False if the
    payload was empty.
    """
    text = (text or "").strip()
    images = [i for i in (images or []) if i]
    if not text and not images:
        return False
    db.enqueue_user_input(conversation_id, text, images or None)
    return True


def _drain_queued_input(conversation_id: str) -> list[dict]:
    """Pop every queued input for `conversation_id` and return them in order.

    Called at the top of each agent iteration so the next Ollama call sees
    any messages the user typed while the previous iteration was running.
    """
    return db.drain_queued_inputs(conversation_id)


def _repair_orphan_tool_calls(conversation_id: str) -> None:
    """Ensure every assistant tool_call has a matching tool result message.

    Called at the top of a new turn. For any unresolved tool_call we append a
    synthetic tool row so the model sees a well-formed history.
    """
    history = db.list_messages(conversation_id)
    resolved_ids = {
        m["tool_calls"][0]["id"]
        for m in history
        if m["role"] == "tool" and m.get("tool_calls")
    }
    for msg in history:
        if msg["role"] != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if tc["id"] in resolved_ids:
                continue
            db.add_message(
                conversation_id,
                "tool",
                json.dumps(
                    {
                        "ok": False,
                        "output": "",
                        "error": "interrupted — tool call did not complete",
                    }
                ),
                tool_calls=[{"id": tc["id"], "name": tc.get("name", "")}],
            )
            resolved_ids.add(tc["id"])


def submit_approval_decision(approval_id: str, approved: bool) -> bool:
    """Called from the sync FastAPI route to resolve a pending approval.

    Uses call_soon_threadsafe because FastAPI sync handlers run on the
    threadpool, not the event loop that's awaiting the future.
    """
    entry = _pending_approvals.get(approval_id)
    if not entry:
        return False
    fut, loop = entry
    if fut.done():
        return False
    loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(approved))
    return True


def _load_image_b64(name: str, directory: Path) -> str | None:
    """Load a saved image (screenshot or user upload) and return base64, or None.

    The model sees images via Ollama's `images` field on user messages,
    which expects base64-encoded PNG/JPEG bytes (no data: prefix).
    """
    try:
        path = directory / name
        if not path.is_file():
            return None
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return None


def _load_screenshot_b64(name: str) -> str | None:
    """Backwards-compat wrapper: load from SCREENSHOT_DIR."""
    return _load_image_b64(name, tools.SCREENSHOT_DIR)


# Cap the before/after text we ship to the browser for side-by-side rendering.
# react-diff-viewer handles a few thousand lines comfortably, but multi-MB
# files would bog down the UI and bloat the SSE payload for no practical gain.
PREVIEW_MAX_CHARS = 200_000


def _truncate_for_preview(text: str) -> tuple[str, bool]:
    """Return (possibly-truncated text, was_truncated)."""
    if len(text) <= PREVIEW_MAX_CHARS:
        return text, False
    return text[:PREVIEW_MAX_CHARS], True


def _preview_for_write(conv_cwd: str, call_name: str, args: dict) -> dict | None:
    """Build an approval-time preview payload for file-writing tools.

    For `write_file` and `edit_file` we compute both the unified diff AND the
    full before/after text (capped at PREVIEW_MAX_CHARS) so the UI can render
    either a unified or side-by-side diff without a round-trip.

    Returns a dict like:
      {"kind": "diff", "path": str, "diff": "...",
       "before": "...", "after": "...",
       "before_size": int, "after_size": int,
       "truncated": bool, "language": str}
    or None when the tool isn't one we preview for.
    """
    try:
        if call_name == "write_file":
            path = args.get("path") or ""
            new_content = args.get("content", "")
            p = tools._resolve(conv_cwd, path)
            before = ""
            if p.is_file():
                try:
                    before = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    before = ""
            diff = _unified_diff(before, new_content, str(p))
            before_trunc, before_cut = _truncate_for_preview(before)
            after_trunc, after_cut = _truncate_for_preview(new_content)
            return {
                "kind": "diff",
                "path": str(p),
                "diff": diff,
                "before": before_trunc,
                "after": after_trunc,
                "before_size": len(before),
                "after_size": len(new_content),
                "truncated": before_cut or after_cut,
                "language": _guess_language(path),
            }
        if call_name == "edit_file":
            path = args.get("path") or ""
            old_s = args.get("old_string", "")
            new_s = args.get("new_string", "")
            p = tools._resolve(conv_cwd, path)
            if not p.is_file():
                return {
                    "kind": "diff",
                    "path": str(p),
                    "diff": "(file does not exist yet — edit_file will fail)",
                    "before": "",
                    "after": "",
                    "before_size": 0,
                    "after_size": 0,
                    "truncated": False,
                    "language": _guess_language(path),
                    "note": "file does not exist yet — edit_file will fail",
                }
            try:
                before = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                before = ""
            replace_all = bool(args.get("replace_all", False))
            count = before.count(old_s) if old_s else 0
            if count == 0:
                after = before
                note = "(old_string not found — edit_file will fail)"
            elif count > 1 and not replace_all:
                after = before
                note = f"(old_string appears {count} times; not unique — edit_file will fail unless replace_all=true)"
            else:
                after = (
                    before.replace(old_s, new_s)
                    if replace_all
                    else before.replace(old_s, new_s, 1)
                )
                note = ""
            diff = _unified_diff(before, after, str(p))
            if note:
                diff = note + "\n" + diff
            before_trunc, before_cut = _truncate_for_preview(before)
            after_trunc, after_cut = _truncate_for_preview(after)
            return {
                "kind": "diff",
                "path": str(p),
                "diff": diff,
                "before": before_trunc,
                "after": after_trunc,
                "before_size": len(before),
                "after_size": len(after),
                "truncated": before_cut or after_cut,
                "language": _guess_language(path),
                "note": note or None,
            }
    except Exception:
        return None
    return None


# Map a file extension to the language identifier react-diff-viewer-continued
# expects for Prism syntax highlighting. Kept small and curated — missing
# languages just render as plain text, which is fine.
_LANG_BY_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".json": "json",
    ".md": "markdown",
    ".html": "markup",
    ".htm": "markup",
    ".xml": "markup",
    ".css": "css",
    ".scss": "scss",
    ".sass": "scss",
    ".sh": "bash",
    ".bash": "bash",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".sql": "sql",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".rb": "ruby",
    ".php": "php",
}


def _guess_language(path: str) -> str:
    """Best-effort filename → Prism language hint for the diff viewer."""
    try:
        return _LANG_BY_EXT.get(Path(path).suffix.lower(), "")
    except Exception:
        return ""


def _unified_diff(old: str, new: str, path: str, max_lines: int = 200) -> str:
    """Render a unified diff capped at `max_lines` lines."""
    it = difflib.unified_diff(
        old.splitlines(keepends=False),
        new.splitlines(keepends=False),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=3,
    )
    lines = list(it)
    if not lines:
        return "(no changes)"
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... [truncated, {len(lines) - max_lines} more diff lines]"]
    return "\n".join(lines)


# -------- @-mention expansion ------------------------------------------
# Users can type `@path/to/file` in the composer to pin specific files
# into the conversation context. The token is preserved in the stored
# message so the UI stays readable; we append structured <file> blocks
# only when building the Ollama payload, so history replay uses the
# current contents of each file.
#
# Pattern: an `@` preceded by start-of-string or non-identifier char,
# followed by a path-like run. Rejects emails (`foo@bar.com` — the `@`
# is preceded by an identifier char) and inline code because code blocks
# are preceded by backticks/quotes which also aren't in the negative
# lookbehind set.
_MENTION_RE = __import__("re").compile(
    r"(?<![A-Za-z0-9_/.])@([A-Za-z0-9_./\\-]+)"
)

# Per-file and total caps for @-mention expansion — a runaway user (or
# a deliberate abuse) can't blow the context window by pinning a folder
# of huge files. 40 KB per file is enough for a typical source file;
# 120 KB total is ~30k tokens which is a sensible ceiling.
_MENTION_FILE_MAX = 40_000
_MENTION_TOTAL_MAX = 120_000


def _expand_file_mentions(text: str, cwd: str | None) -> str:
    """Append inline `<file>` blocks for every valid `@path` mention in `text`.

    Files are resolved under `cwd`; any mention whose path doesn't exist,
    escapes the cwd, or can't be read is silently skipped (leaves the
    `@token` in place). Total expansion is capped to avoid blowing the
    prompt budget — once the budget is hit, remaining mentions are
    ignored. Tokens are NOT substituted in-place so the reader (both
    the model and future inspectors of the prompt) can tell where the
    user put the reference vs. the injected body.
    """
    if not text or "@" not in text or not cwd:
        return text
    from pathlib import Path as _Path
    try:
        root = _Path(cwd).expanduser().resolve()
    except Exception:
        return text
    if not root.is_dir():
        return text
    matches = _MENTION_RE.findall(text)
    if not matches:
        return text
    seen: set[str] = set()
    blocks: list[str] = []
    remaining = _MENTION_TOTAL_MAX
    for rel in matches:
        if rel in seen:
            continue
        seen.add(rel)
        try:
            p = (root / rel).resolve()
            if not p.is_file() or not p.is_relative_to(root):
                continue
            body = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        truncated = False
        if len(body) > _MENTION_FILE_MAX:
            body = body[:_MENTION_FILE_MAX]
            truncated = True
        rel_norm = str(p.relative_to(root)).replace("\\", "/")
        trailer = "\n...[truncated]" if truncated else ""
        block = f"\n\n<file path=\"{rel_norm}\">\n{body}{trailer}\n</file>"
        if len(block) > remaining:
            break
        blocks.append(block)
        remaining -= len(block)
    if not blocks:
        return text
    return text + "".join(blocks)


def _index_recent_screenshot_msgs(
    history: list[dict],
    keep: int,
) -> set[str]:
    """Return the ids of the N most-recent tool-rows that carry a screenshot.

    Used by `_to_ollama_messages` to decide which screenshot images get
    attached at full fidelity vs. replaced by a one-line text descriptor.
    Vision-token cost dominates long screenshot-heavy sessions — keeping
    every PNG forward forever is wasteful, and the model only ever needs
    to look back 1-2 frames to confirm a state change.

    Walks the history in reverse, picks the first `keep` tool rows whose
    `tool_calls[0].image_path` is non-empty, and returns their message ids
    as a set. The caller then attaches images only for rows in this set.
    """
    if keep <= 0:
        return set()
    keepers: set[str] = set()
    for m in reversed(history):
        if m.get("role") != "tool":
            continue
        tc = (m.get("tool_calls") or [{}])[0]
        if not tc.get("image_path"):
            continue
        keepers.add(m["id"])
        if len(keepers) >= keep:
            break
    return keepers


def _to_ollama_messages(
    system: str, history: list[dict], cwd: str | None = None,
) -> list[dict]:
    """Convert our DB rows into Ollama-chat messages.

    DB rows look like:
      {role: user|assistant|tool, content: str, tool_calls: [...], images: [...]}
    For tool messages we also stored tool_call_id / name / image_path inside
    tool_calls[0].

    When a tool row has an image_path (computer-use screenshots), we emit
    the tool message as usual, then append a synthetic user message with the
    PNG bytes attached. This is the most portable way to get vision into a
    multimodal chat model via Ollama, since the `tool` role's schema
    doesn't officially accept images.

    Screenshot images are aged out of the prompt: only the most recent
    `KEEP_RECENT_SCREENSHOT_IMAGES` are attached as actual PNGs; older
    ones are replaced with a short text descriptor that points the model
    at the structured `[ctx:]` and `[Δ:]` summary already in the tool's
    text body. Vision tokens are the dominant cost in screenshot-heavy
    sessions and the model rarely needs to re-examine 10-turn-old frames.

    User rows can also carry an `images` list (filenames under UPLOAD_DIR)
    for images the human pasted/dragged into the composer — these are
    NEVER aged out (the user explicitly attached them, presumably because
    they remain relevant).
    """
    keep_image_ids = _index_recent_screenshot_msgs(history, KEEP_RECENT_SCREENSHOT_IMAGES)
    msgs: list[dict] = [{"role": "system", "content": system}]
    for m in history:
        role = m["role"]
        if role == "user":
            # Expand any @path mentions the user embedded in the message
            # into inline <file> blocks — the stored row keeps the raw
            # tokens so the UI bubble stays compact.
            content = _expand_file_mentions(m["content"], cwd)
            msg: dict = {"role": "user", "content": content}
            # Attach any user-uploaded images. Unlike screenshots, these
            # are deliberately attached by the human and aren't aged out.
            images = m.get("images") or []
            if images:
                b64s = []
                for name in images:
                    b = _load_image_b64(name, tools.UPLOAD_DIR)
                    if b:
                        b64s.append(b)
                if b64s:
                    msg["images"] = b64s
            msgs.append(msg)
        elif role == "assistant":
            out: dict = {"role": "assistant", "content": m["content"] or ""}
            if m.get("tool_calls"):
                out["tool_calls"] = [
                    {
                        "function": {
                            "name": tc["name"],
                            "arguments": tc.get("args", {}),
                        }
                    }
                    for tc in m["tool_calls"]
                ]
            msgs.append(out)
        elif role == "tool":
            tc = (m.get("tool_calls") or [{}])[0]
            msgs.append(
                {
                    "role": "tool",
                    "content": m["content"],
                    "tool_name": tc.get("name", ""),
                }
            )
            image_name = tc.get("image_path")
            if image_name:
                # Only attach the image if this row is in the recent-keepers
                # set; older screenshots get a text descriptor instead so we
                # don't pay vision-token cost on stale views the model can't
                # act on anyway.
                if m["id"] in keep_image_ids:
                    b64 = _load_screenshot_b64(image_name)
                    if b64:
                        msgs.append(
                            {
                                "role": "user",
                                "content": (
                                    "[Screenshot attached from the previous "
                                    f"`{tc.get('name', 'computer-use')}` tool call. "
                                    "Use it to decide the next step.]"
                                ),
                                "images": [b64],
                            }
                        )
                else:
                    # Older frames — text-only descriptor. The tool result's
                    # own content already contains the `[ctx:]` and `[Δ:]`
                    # summary, so the model can still reason about state
                    # changes without re-loading the bytes.
                    msgs.append(
                        {
                            "role": "user",
                            "content": (
                                f"[Older screenshot from `{tc.get('name', 'computer-use')}` "
                                "elided to save context — see the tool result "
                                "above for the [ctx:] and [Δ:] summary. "
                                "Take a fresh `screenshot` if you need to look at the screen now.]"
                            ),
                        }
                    )
        elif role == "system":
            # Synthetic system rows are used by the auto-compactor to insert
            # a running summary mid-history. Pass them through so the model
            # treats them as authoritative context.
            msgs.append({"role": "system", "content": m["content"]})
    return msgs


def _estimate_prompt_chars(messages: list[dict]) -> int:
    """Return a rough char-count of the prompt we'd send to Ollama.

    Used by the auto-compactor to decide whether compaction is warranted.
    Images are estimated at ~1000 "equivalent chars" each to keep the model
    from OOMing on a long screenshot-heavy thread.
    """
    total = 0
    for m in messages:
        c = m.get("content") or ""
        total += len(c)
        imgs = m.get("images") or []
        total += 1000 * len(imgs)
    return total


async def _summarize_block_with_ollama(model: str, history_block: list[dict]) -> str:
    """Ask the model to compress an older slice of history into a short summary.

    Returns just the compressed string. Non-streaming request — we don't need
    to surface these tokens to the UI; they're internal bookkeeping.

    Pool-distributed: when a worker has the chat model installed,
    compaction routes there via `compute_pool.pick_compaction_target`
    so the host's KV cache stays warm for the active chat. Falls back
    to host Ollama when no worker is eligible.
    """
    ask = (
        "You are compressing an older portion of a chat history to save "
        "context space. Produce a concise (≤400 word) summary of the block "
        "below that preserves: (a) what the user asked for, (b) what the "
        "assistant decided / did, (c) any files, paths, flags, or values "
        "that were mentioned. Do NOT include any tool-call JSON. Respond "
        "with the summary text only.\n\n--- BEGIN HISTORY BLOCK ---\n"
    )
    rendered: list[str] = []
    for m in history_block:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if role == "tool":
            content = content[:500]  # tool outputs can be huge; cap hard
        rendered.append(f"[{role}]\n{content}")
    prompt = ask + "\n\n".join(rendered) + "\n--- END HISTORY BLOCK ---"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": NUM_CTX},
    }
    # Route compaction to a worker so it doesn't compete with the
    # active chat for host GPU. Falls back to host when no worker
    # has the model.
    base_url = OLLAMA_URL
    headers: dict[str, str] = {}
    try:
        from . import compute_pool
        target = compute_pool.pick_compaction_target(model)
        if target is not None:
            base_url, token = target
            if token:
                headers["Authorization"] = f"Bearer {token}"
    except Exception:
        # Defensive: any picker failure must not break compaction itself.
        base_url = OLLAMA_URL
        headers = {}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{base_url}/api/chat", json=payload, headers=headers,
            )
            r.raise_for_status()
            data = r.json()
        return (data.get("message") or {}).get("content") or ""
    except Exception as e:
        return f"[summary failed: {type(e).__name__}: {e}]"


def _proactively_age_tool_outputs(conversation_id: str, history: list[dict]) -> bool:
    """Compress old, bulky tool rows even when we're not over budget yet.

    The classic compaction pass only fires at 75% of `num_ctx` — for a long
    session that stays under that budget it never runs, so a 20 KB bash
    dump from turn 5 keeps costing ~5000 tokens on every Ollama call all
    the way through turn 100.

    This pass shrinks any non-pinned tool row that is BOTH:
      - older than `PROACTIVE_TOOL_AGE_TURNS` positions from the tail, AND
      - has a body bigger than `BULKY_TOOL_OUTPUT_CHARS`,
    AND is not already compressed (idempotent).

    Returns True when at least one row was touched, so the caller knows the
    history has changed and may want to re-read it.
    """
    candidates: list[str] = []
    cutoff = max(0, len(history) - PROACTIVE_TOOL_AGE_TURNS)
    for m in history[:cutoff]:
        if m["role"] != "tool" or m.get("pinned"):
            continue
        body = m.get("content") or ""
        if len(body) <= BULKY_TOOL_OUTPUT_CHARS:
            continue
        if db.is_compressed_tool_output(body):
            continue
        candidates.append(m["id"])
    if not candidates:
        return False
    db.compress_tool_outputs(candidates)
    return True


async def _maybe_compact(
    conversation_id: str, system_prompt: str, model: str, cwd: str | None = None,
) -> None:
    """Auto-compaction: progressively reduce history when it gets too large.

    Phases, run in order, each stops early if it gets back under budget:

      0. **Proactive aging.** Independent of the budget check, shrink old
         bulky tool rows so a session that hovers under the threshold
         doesn't pay for stale 20 KB outputs forever. Cheap; runs first.
      1. **Compress old tool outputs (under-budget pass).** Tool rows
         (bash output, screenshots, grep hits) are the bulk of most
         conversations and are rarely useful once the agent has acted on
         them. Replace bodies with head+tail snippets; keep the row so
         the paired assistant tool_call still has a matching result.
      2. **Summarize older non-pinned user/assistant rows.** The classic
         compaction pass — hand a block of history to the model, replace
         it with a synthetic system summary. Pinned rows are excluded from
         this block and keep their full fidelity.
      3. (If we're still over budget after both phases, we do nothing more
         — the next Ollama call will truncate the prompt itself. This is
         rare in practice.)
    """
    history = db.list_messages(conversation_id)
    if len(history) < 20:
        return

    # --- Phase 0: proactive aging --------------------------------------------
    # Always cheap; never asks the model to do anything. If it shrinks any
    # row we re-read history before the budget check below.
    if _proactively_age_tool_outputs(conversation_id, history):
        history = db.list_messages(conversation_id)

    ollama_msgs = _to_ollama_messages(system_prompt, history, cwd=cwd)
    total = _estimate_prompt_chars(ollama_msgs)
    budget = int(NUM_CTX * CHARS_PER_TOKEN * COMPACTION_THRESHOLD)
    if total < budget:
        return

    # --- Phase 1: shrink old tool outputs ------------------------------------
    # Keep the most recent 8 tool rows at full fidelity; compress the rest.
    # Skip rows that are already compressed (idempotency — re-compressing
    # would degrade the head+tail snippet further on each pass).
    tool_rows = [m for m in history if m["role"] == "tool" and not m.get("pinned")]
    if len(tool_rows) > 8:
        compress = [
            m["id"] for m in tool_rows[:-8]
            if not db.is_compressed_tool_output(m.get("content") or "")
        ]
        if compress:
            db.compress_tool_outputs(compress)
            history = db.list_messages(conversation_id)
            ollama_msgs = _to_ollama_messages(system_prompt, history, cwd=cwd)
            total = _estimate_prompt_chars(ollama_msgs)
            if total < budget:
                return

    # --- Phase 2: summarize the older user/assistant middle ------------------
    keep_tail = 12
    # Keep the first user message verbatim so the model always sees the
    # original goal statement. Find its index.
    first_user_idx: int | None = None
    for i, m in enumerate(history):
        if m["role"] == "user":
            first_user_idx = i
            break
    if first_user_idx is None:
        return
    tail_start = max(first_user_idx + 1, len(history) - keep_tail)
    middle = history[first_user_idx + 1 : tail_start]
    # Never compress pinned rows — they're the user's "do not forget" anchors.
    compressible = [m for m in middle if not m.get("pinned")]
    if len(compressible) < 4:
        return
    # Idempotency: if the middle is already a single synthetic summary, bail.
    if len(compressible) == 1 and compressible[0]["role"] == "system":
        return

    summary = await _summarize_block_with_ollama(model, compressible)
    db.delete_messages([m["id"] for m in compressible])
    db.add_system_summary(
        conversation_id,
        f"[Auto-compacted summary of earlier conversation]\n\n{summary}",
    )


# ---------------------------------------------------------------------------
# Semantic recall (RAG over conversation history)
#
# When the user asks a question that's relevant to something said earlier —
# but long enough ago that compaction already dropped it — we want to
# "remember" on demand. The implementation:
#
#   1. Every new user / assistant message is embedded with a small local
#      model (nomic-embed-text via Ollama, ~300 MB) and the vector is stored
#      in the message_embeddings table.
#   2. On every turn, we embed the current user input, dot-product against
#      the stored vectors for this conversation, pick the top-K over a
#      similarity threshold, and inject their content as a synthetic
#      system message right before the current user turn.
#
# Embedding failures are non-fatal. If the user hasn't pulled
# nomic-embed-text, recall silently degrades to a no-op.
# ---------------------------------------------------------------------------
async def _embed_via(base_url: str, auth_token: str | None, text: str) -> list[float] | None:
    """POST one embed request and return the normalized vector.

    Pulled out of `_embed_text` so we can hit either the host's local
    Ollama or a registered compute worker through the same code path.
    Returns None on any failure — the caller decides whether to fall back.
    """
    headers: dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(
            f"{base_url}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text[:8000]},
            headers=headers,
        )
        r.raise_for_status()
        data = r.json()
    vec = data.get("embedding") or []
    if not vec:
        return None
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


async def _embed_text(text: str) -> list[float] | None:
    """Ollama /api/embeddings → normalized float list, or None on failure.

    Routing: we try a registered compute worker first (one with
    `use_for_embeddings=True` and the embed model installed), then fall
    back to the host's local Ollama. The fallback is silent — a worker
    that's intermittently unreachable shouldn't break recall, and the
    periodic probe will mark it `last_error` so the picker skips it on
    the next call without us blacklisting state here.
    """
    if not text or not text.strip():
        return None

    target = None
    try:
        target = compute_pool.pick_embed_target(EMBED_MODEL)
    except Exception:
        # Picker can't fail in practice (read-only DB query) but if the
        # compute_pool module ever raises here we don't want to break
        # the embedding path — silently route to host.
        target = None

    if target is not None:
        base, token = target
        try:
            vec = await _embed_via(base, token, text)
            if vec:
                return vec
            # Empty payload from worker — fall through to host.
        except Exception:
            # Worker unreachable / 5xx — fall through.
            pass

    # Host fallback. Same code path; auth_token is None on loopback Ollama.
    try:
        return await _embed_via(OLLAMA_URL, None, text)
    except Exception:
        return None


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product. Both vectors expected to be pre-normalized (unit length)."""
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


async def _schedule_embedding(message_id: str, conversation_id: str, text: str) -> None:
    """Fire-and-forget embedding so we don't block the user's turn.

    Silently swallows any failure — embedding is an enhancement, not a
    requirement. If the embed model isn't pulled, `_embed_text` returns
    None and this function is a no-op.
    """
    try:
        vec = await _embed_text(text)
        if vec:
            db.save_embedding(message_id, conversation_id, vec)
    except Exception:
        pass


async def _semantic_recall(
    conversation_id: str,
    query: str,
    exclude_ids: set[str],
) -> list[dict]:
    """Find older messages semantically similar to `query`.

    Returns a small list of message rows (not including any in exclude_ids,
    which should be the ids already present in the recent tail). Empty on
    embedding failure or when nothing crosses the similarity threshold.
    """
    if not query or not query.strip():
        return []
    q_vec = await _embed_text(query)
    if not q_vec:
        return []
    rows = db.list_embeddings_for_conv(conversation_id)
    scored: list[tuple[str, float]] = []
    for mid, vec in rows:
        if mid in exclude_ids or len(vec) != len(q_vec):
            continue
        score = _dot(q_vec, vec)
        if score >= RECALL_MIN_SCORE:
            scored.append((mid, score))
    if not scored:
        return []
    scored.sort(key=lambda t: t[1], reverse=True)
    top_ids = [mid for mid, _ in scored[:RECALL_TOP_K]]
    return db.get_messages_by_ids(top_ids)


def _format_recall(hits: list[dict]) -> str:
    """Render recalled messages as a compact system-prompt-friendly block."""
    lines = [
        "[Relevant context retrieved from earlier in this conversation. "
        "These messages were not in the recent window but may be useful "
        "for the current question.]",
        "",
    ]
    for m in hits:
        role = m.get("role", "?")
        raw = (m.get("content") or "").strip().replace("\n", " ")
        if len(raw) > 400:
            raw = raw[:400] + "…"
        lines.append(f"- [{role}] {raw}")
    return "\n".join(lines)


# Per-model cache of Ollama capabilities list. Cheap to query but we call
# /api/chat every turn — a TTL-less in-proc cache is fine because model caps
# only change when the user re-pulls a model, which restarts the server via
# the Ollama host anyway. Value is the raw list from /api/show.
_MODEL_CAPS_CACHE: dict[tuple[str, str], list[str]] = {}


async def _model_capabilities(
    client: httpx.AsyncClient,
    model: str,
    *,
    base_url: str = OLLAMA_URL,
    auth_token: str | None = None,
) -> list[str]:
    """Fetch + cache the Ollama capabilities list for ``model``.

    Returns an empty list on any error so callers can fall back to default
    behaviour (don't pass opt-in flags the model might reject).

    `base_url` defaults to the host's local Ollama; pass a worker URL when
    routing a subagent / chat call to a registered worker. The cache key is
    `(base_url, model)` so two Ollama instances reporting different caps for
    the same model name don't pollute each other.
    """
    cache_key = (base_url, model)
    cached = _MODEL_CAPS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    headers: dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    try:
        r = await client.post(
            f"{base_url}/api/show",
            json={"name": model},
            headers=headers,
            timeout=5.0,
        )
        r.raise_for_status()
        caps = r.json().get("capabilities") or []
    except Exception:
        caps = []
    _MODEL_CAPS_CACHE[cache_key] = caps
    return caps


async def _stream_ollama_chat(
    model: str,
    messages: list[dict],
    tool_schemas: list[dict],
    *,
    adapter_mode: bool = False,
    disable_thinking: bool = False,
    base_url: str = OLLAMA_URL,
    auth_token: str | None = None,
) -> AsyncGenerator[dict, None]:
    """Yield raw chunks from Ollama /api/chat with stream=true.

    Automatically sets ``think: true`` when the model advertises the
    ``thinking`` capability so reasoning tokens stream back in the
    ``message.thinking`` field. Models that don't support thinking would
    reject the flag with HTTP 400, so it's opt-in by capability.

    When ``disable_thinking`` is True we force ``think: false`` even on
    thinking-capable models — used by the empty-response retry path so a
    model that got lost in reasoning is nudged to answer directly.

    When ``adapter_mode`` is True the caller has already inlined the tool
    list into the system prompt (via ``tool_prompt_adapter``) and
    rewritten any tool-role / ``tool_calls`` history into text form. In
    that mode we omit the ``tools`` field from the payload entirely —
    sending it would be harmless for a stub template (it's ignored
    anyway) but could confuse a custom template that renders ``.Tools``
    AND also sees the inline text block, producing duplicated
    instructions.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        # Low temperature + reasonable context window.
        # Small models (Gemma 4 e4b) need a low temperature for reliable
        # tool-call emission — at 0.7 they sometimes refuse to call tools even
        # when the system prompt says to.
        # `num_predict=-1` lets the model generate until EOS — no
        # token cap. This used to be capped at 8 K as a safety fuse
        # against runaway abliterated models locking into repetition
        # loops, but the cap was also truncating legitimate long
        # responses (full-file rewrites, edit_file payloads with the
        # entire new file contents, multi-step plans). Two backstops
        # remain when generation goes off the rails: the 600 s read
        # timeout below closes the stream cleanly, and the Stop button
        # in the UI flips a flag the agent loop polls between Ollama
        # chunks.
        "options": {
            "temperature": 0.3,
            "num_ctx": NUM_CTX,
            "num_predict": -1,
        },
    }
    if not adapter_mode:
        payload["tools"] = tool_schemas
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)
    headers: dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        caps = await _model_capabilities(
            client, model, base_url=base_url, auth_token=auth_token,
        )
        if "thinking" in caps:
            payload["think"] = False if disable_thinking else True
        async with client.stream(
            "POST", f"{base_url}/api/chat", json=payload, headers=headers,
        ) as r:
            # Ollama returns a JSON-encoded ``{"error": "..."}`` body on 4xx/5xx.
            # ``raise_for_status`` alone just surfaces the generic status line
            # ("Client error '400 Bad Request' for url ...") which hides the
            # real reason — model doesn't support tools, model not pulled,
            # context window too large, etc. Read the body first so the user
            # sees Ollama's actual complaint.
            if r.status_code >= 400:
                body = await r.aread()
                detail = ""
                try:
                    detail = json.loads(body.decode("utf-8", errors="replace")).get("error", "")
                except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                    detail = body.decode("utf-8", errors="replace")[:400]
                msg = (detail or f"HTTP {r.status_code}").strip()
                # Friendly hint for the single most common cause: sending
                # `tools` to a model that doesn't implement function calling.
                if "does not support tools" in msg.lower():
                    msg = (
                        f"The selected model ({model!r}) does not support tool "
                        "calling. Switch to a function-calling model like "
                        "`gemma4:e4b`, `llama3.1:8b`, `qwen2.5:7b`, or "
                        "`mistral-nemo` from Settings → Default model."
                    )
                raise httpx.HTTPStatusError(msg, request=r.request, response=r)
            async for line in r.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# ---------------------------------------------------------------------------
# Split-model chat: talk to llama-server (OpenAI-compatible) and translate
# its SSE chunks into the Ollama chunk shape the rest of agent.py expects.
#
# llama-server's /v1/chat/completions speaks SSE with payloads like
#   data: {"choices":[{"delta":{"role":"assistant","content":"hi"}}]}
#   data: [DONE]
# We yield Ollama-shaped dicts so callers don't need a code branch:
#   {"message": {"role":"assistant","content":"hi"}, "done": False}
#
# Tool calling: split models go through the prompt-space adapter
# (`tool_prompt_adapter.needs_adapter` returns True for split: prefixes
# in `_resolve_chat_target`), so tool calls arrive as plaintext
# <tool_call>…</tool_call> blocks in `delta.content` rather than via
# OpenAI's structured `tool_calls` field. The agent loop's existing
# adapter parsing then handles them. This keeps split-model and
# Ollama-model code paths uniform.
# ---------------------------------------------------------------------------
async def _stream_llama_server_chat(
    model: str,
    messages: list[dict],
    *,
    base_url: str,
    auth_token: str | None = None,
) -> AsyncGenerator[dict, None]:
    """Yield Ollama-shape chunks from a llama-server `/v1/chat/completions`
    streaming response.

    `model` is whatever label llama-server should echo back — the server
    has only one model loaded so this is mostly cosmetic.

    Tools are deliberately NOT passed in the OpenAI shape — the caller
    is expected to be in adapter mode (system prompt has the tool
    block inlined) so we send only `messages`.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0.3,
        # No max_tokens cap — llama-server defaults to its own context
        # ceiling. Adding -1 here would 400 (OpenAI rejects negatives).
    }
    headers: dict[str, str] = {"Accept": "text/event-stream"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
        ) as r:
            if r.status_code >= 400:
                body = await r.aread()
                detail = body.decode("utf-8", errors="replace")[:400]
                msg = (detail or f"HTTP {r.status_code}").strip()
                raise httpx.HTTPStatusError(msg, request=r.request, response=r)

            # SSE framing: lines prefixed with "data: " carry the JSON.
            # Empty lines separate events. The terminal sentinel is
            # "data: [DONE]" — we yield a final {"done": True} when we
            # see it so the agent loop's "if chunk.get('done'): break"
            # check fires the same way as for Ollama's `done` flag.
            async for line in r.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                payload_str = line[len("data:"):].strip()
                if payload_str == "[DONE]":
                    yield {"message": {"role": "assistant", "content": ""}, "done": True}
                    break
                try:
                    obj = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                # OpenAI streaming: delta lives at choices[0].delta.
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0] or {}).get("delta") or {}
                # Translate to Ollama shape. We collect role + content
                # into `message` and surface `done` only on the [DONE]
                # sentinel above (OpenAI's per-chunk `finish_reason`
                # isn't a perfect mirror — better to wait for the
                # explicit terminator).
                yield {
                    "message": {
                        "role": delta.get("role") or "assistant",
                        "content": delta.get("content") or "",
                    },
                    "done": False,
                }


async def run_turn(
    conversation_id: str,
    user_text: str | None = None,
    user_images: list[str] | None = None,
    persist_user: bool = True,
) -> AsyncGenerator[dict, None]:
    """Run one turn of the agent with crash-resilience state management.

    Thin wrapper over `_run_turn_impl`: marks the conversation as 'running'
    before the first yield, then transitions back to 'idle' (or 'error') in
    a finally block so the startup resumer can tell which conversations
    crashed mid-turn.

    See `_run_turn_impl` for the turn-internal event contract.
    """
    # Short-circuit BEFORE entering the try block so a missing conversation
    # doesn't accidentally flip state on some other row.
    conv = db.get_conversation(conversation_id)
    if not conv:
        yield {"type": "error", "message": "conversation not found"}
        return

    # Per-conversation soft budget gate. `budget_turns` and `budget_tokens`
    # are both nullable — absence means "no cap". We check BEFORE flipping
    # state to 'running' so a refused turn doesn't leave the conversation
    # looking busy in the sidebar.
    #
    # Token estimation reuses the char-count proxy the frontend gauge uses
    # (one token per _CHARS_PER_TOKEN chars). It's rough, but budget refusal
    # is a UX affordance, not a billing decision, and it's the same number
    # the user sees in the header so there are no surprises.
    try:
        btx = conv.get("budget_tokens")
        btn = conv.get("budget_turns")
        if btx or btn:
            msg = _check_budget(conversation_id, btn, btx)
            if msg:
                yield {"type": "error", "message": msg}
                return
    except Exception:
        # Budget check is best-effort — don't let a query failure block a
        # turn the user explicitly asked for.
        pass
    # Reset the stop flag at the top of a fresh turn. A stop request that
    # arrived while idle (between turns) would otherwise make the new turn
    # abort before producing anything — annoying if the user clicked Stop
    # in the previous turn and then sent a new message.
    _clear_stop(conversation_id)
    # Register as active BEFORE flipping DB state so the stale-turn watchdog
    # never observes a `state='running'` row without a matching active-id
    # entry in the short window between the state write and the set.add.
    _ACTIVE_TURN_IDS.add(conversation_id)
    try:
        db.set_conversation_state(conversation_id, "running")
    except Exception:
        # DB briefly locked — non-fatal. Resumer loses detection of this
        # crash if it happens, but refusing to start would be worse UX.
        pass
    try:
        async for ev in _run_turn_impl(
            conversation_id,
            user_text=user_text,
            user_images=user_images,
            persist_user=persist_user,
        ):
            yield ev
    except Exception:
        # The turn blew up somewhere the inner handlers didn't swallow.
        # Mark errored so the UI / resumer can flag it for the user.
        try:
            db.set_conversation_state(conversation_id, "error")
        except Exception:
            pass
        raise
    finally:
        # Happy path: mark idle only if we weren't already bumped to error
        # in the except above. We re-read state to check.
        try:
            cur = db.get_conversation(conversation_id)
            if cur and cur.get("state") != "error":
                db.set_conversation_state(conversation_id, "idle")
        except Exception:
            pass
        # Clear the stop flag no matter how the turn ended — it only applies
        # to the turn that was running when it was set, not future ones.
        _clear_stop(conversation_id)
        # Decrement the active-turn count on whichever node served this
        # conversation. The affinity record itself stays so the next
        # turn can re-land on the same node (KV cache stays warm).
        try:
            from . import compute_pool as _cp
            sticky = _cp._CONV_AFFINITY.get(conversation_id)
            if sticky:
                _cp.register_turn_end(conversation_id, sticky)
        except Exception:
            pass
        # Remove from the active set LAST so the watchdog doesn't see a
        # brief "running-but-not-active" window on the way out.
        _ACTIVE_TURN_IDS.discard(conversation_id)


def _check_budget(
    conversation_id: str,
    budget_turns: int | None,
    budget_tokens: int | None,
) -> str | None:
    """Return a human-readable refusal string if either budget is exhausted,
    else None. Intentionally stateless — the caller decides how to surface
    the refusal (we yield an 'error' event, but a future caller could do
    something gentler).

    - `budget_turns` compares against `db.count_assistant_turns` (completed
      assistant replies) — a user hammering Enter while the agent churns
      doesn't consume extra turns.
    - `budget_tokens` compares against a char-count proxy
      (`conversation_content_chars / CHARS_PER_TOKEN`). Matches the gauge in
      the header so the UX is coherent.
    """
    if budget_turns and budget_turns > 0:
        used = db.count_assistant_turns(conversation_id)
        if used >= int(budget_turns):
            return (
                f"Budget exhausted: {used}/{int(budget_turns)} assistant turns "
                "used. Raise the limit in the conversation's budget settings "
                "(or clear it) to keep chatting."
            )
    if budget_tokens and budget_tokens > 0:
        chars = db.conversation_content_chars(conversation_id)
        used = chars // max(1, CHARS_PER_TOKEN)
        if used >= int(budget_tokens):
            return (
                f"Budget exhausted: ~{used:,} of {int(budget_tokens):,} "
                "tokens used (rough estimate). Raise the limit in the "
                "conversation's budget settings (or clear it) to keep chatting."
            )
    return None


async def _run_turn_impl(
    conversation_id: str,
    user_text: str | None = None,
    user_images: list[str] | None = None,
    persist_user: bool = True,
) -> AsyncGenerator[dict, None]:
    """Run one turn of the agent. If user_text is given, it's appended first.

    Yields events as described in the module docstring. The caller forwards them
    as SSE to the browser.

    `user_images` is an optional list of filenames inside tools.UPLOAD_DIR that
    the user attached to this turn's input (e.g. pasted screenshots). They're
    attached to the new user row so future replays of history include them.

    `persist_user` controls whether the supplied `user_text` / `user_images`
    are appended as a new user-role row at the top of the turn. The
    edit-and-regenerate flow sets this False because it has already updated
    an existing user row in place — adding a second copy would duplicate the
    prompt in history.
    """
    conv = db.get_conversation(conversation_id)
    if not conv:
        yield {"type": "error", "message": "conversation not found"}
        return

    # NOTE: unlike the old in-memory queue we do NOT drop pre-existing queued
    # inputs here. With DB-backed queuing + state-machine transitions, the
    # only way queue rows survive into a new turn is if the previous turn
    # was interrupted — and we specifically want to replay those now.

    new_user_msg_id: str | None = None
    if persist_user and (user_text is not None or user_images):
        new_user_row = db.add_message(
            conversation_id,
            "user",
            user_text or "",
            images=user_images or None,
        )
        new_user_msg_id = new_user_row["id"]
        # Embed the new user message in the background so semantic recall on
        # future turns can surface it. Fire-and-forget — failure is silent.
        if user_text and user_text.strip():
            asyncio.create_task(
                _schedule_embedding(new_user_msg_id, conversation_id, user_text)
            )

    # -------- Lifecycle hook: user_prompt_submit ------------------------
    # Fires once per turn when the user submits something new. Hook output
    # is injected back into the conversation as a system-note so the agent
    # can see linter results, reminders, env checks, etc. Failures never
    # block the turn — we surface them via SSE and move on.
    hook_prompt_outputs: list[str] = []
    if user_text:
        try:
            prompt_hook_results = await tools.run_hooks(
                "user_prompt_submit",
                {
                    "event": "user_prompt_submit",
                    "conversation_id": conversation_id,
                    "cwd": conv["cwd"],
                    "user_text": user_text,
                },
                conv_id=conversation_id,
            )
        except Exception:
            prompt_hook_results = []
        for h in prompt_hook_results:
            yield {
                "type": "hook_ran",
                "event": "user_prompt_submit",
                "hook_id": h.get("hook_id"),
                "command": h.get("command"),
                "ok": h.get("ok"),
                "output": h.get("output", ""),
                "error": h.get("error"),
            }
            out = (h.get("output") or "").strip()
            if out:
                hook_prompt_outputs.append(
                    f"[user_prompt_submit hook: `{h.get('command', '')}`]\n{out}"
                )
        if hook_prompt_outputs:
            # Persist as a system-role message so it shows up in history
            # replays AND makes it into the next Ollama call. Keeps hooks
            # auditable and survives auto-compaction.
            db.add_system_summary(
                conversation_id,
                "\n\n".join(hook_prompt_outputs),
            )

    # Repair hook: if the previous turn was interrupted mid-tool-call (server
    # restart, browser close, etc.), the history may have an assistant message
    # with tool_calls but no matching `tool` result rows. Feeding that to
    # Ollama as-is makes the model reject the conversation with an unpaired
    # tool_call error. Backfill a synthetic "interrupted" result so the chat
    # can continue cleanly.
    _repair_orphan_tool_calls(conversation_id)

    # System prompt picks up AGENTS.md + the per-conversation memory file +
    # the optional per-conversation persona override (free-text extension
    # set via ChatHeader → Persona). When the conversation is in `plan`
    # permission mode we pass that through so prompts.py appends the
    # plan-mode instruction block (no writes, produce an approvable plan).
    _pm = conv.get("permission_mode") or (
        "allow_all" if conv.get("auto_approve") else "approve_edits"
    )
    system_prompt = build_system_prompt(
        conv["cwd"],
        conversation_id,
        persona=conv.get("persona"),
        permission_mode=_pm,
    )

    # Auto-compaction — done before the Ollama call so token budget is tidy
    # even when the user resumes a long conversation after a break.
    try:
        await _maybe_compact(
            conversation_id, system_prompt, conv["model"], cwd=conv.get("cwd"),
        )
    except Exception:
        # Compaction is a nice-to-have; never block the turn if it fails.
        pass

    # Semantic recall runs once per turn (not per tool iteration) — we want
    # to retrieve based on the user's actual question, not the intermediate
    # tool-result chatter. Run it after compaction so we don't accidentally
    # "recall" a message we just summarized away in this same turn.
    recall_block: str | None = None
    if user_text and user_text.strip():
        try:
            recent_ids = {
                m["id"] for m in db.list_messages(conversation_id)[-RECALL_EXCLUDE_TAIL:]
            }
            if new_user_msg_id:
                recent_ids.add(new_user_msg_id)
            hits = await _semantic_recall(conversation_id, user_text, recent_ids)
            if hits:
                recall_block = _format_recall(hits)
        except Exception:
            recall_block = None

    for _ in range(MAX_TOOL_ITERATIONS):
        # Stop checkpoint — honours a Stop button click that arrived while
        # the previous iteration was executing tools or after an approval.
        # Breaking out here ends the turn cleanly without starting a new
        # Ollama round-trip the user doesn't want.
        if is_stop_requested(conversation_id):
            yield {"type": "error", "message": "stopped by user"}
            yield {"type": "turn_done"}
            return
        # Drain any user input that arrived while the previous iteration was
        # streaming or executing tools. Each queued entry is persisted as a
        # real user-role row before the next Ollama call, so the model sees
        # them as ordinary follow-up messages — no special prompting needed.
        for queued in _drain_queued_input(conversation_id):
            qrow = db.add_message(
                conversation_id,
                "user",
                queued.get("text") or "",
                images=queued.get("images") or None,
            )
            qtext = (queued.get("text") or "").strip()
            if qtext:
                # Mirror the embedding behaviour of the initial prompt so
                # semantic recall surfaces follow-ups too.
                asyncio.create_task(
                    _schedule_embedding(qrow["id"], conversation_id, qtext)
                )
            # Tell the UI the queued message has landed in history so the
            # composer can clear its "queued (n)" indicator and the
            # transcript can echo it inline.
            yield {
                "type": "user_message_added",
                "id": qrow["id"],
                "content": qrow["content"],
                "images": qrow.get("images") or [],
                "created_at": qrow["created_at"],
            }

        history = db.list_messages(conversation_id)
        ollama_msgs = _to_ollama_messages(
            system_prompt, history, cwd=conv.get("cwd"),
        )
        # Inject recalled context just before the last message so it feels
        # like "here's some relevant background, now answer the question".
        if recall_block and len(ollama_msgs) >= 2:
            ollama_msgs.insert(
                len(ollama_msgs) - 1,
                {"role": "system", "content": recall_block},
            )

        # Merge live MCP tool schemas + user-defined tool schemas onto
        # the built-in set. MCP names are namespaced (`mcp__<server>__
        # <tool>`) so they can't collide with built-ins; user tools are
        # guaranteed collision-free at creation time (create_tool
        # refuses colliding names).
        merged_schemas = (
            TOOL_SCHEMAS
            + mcp.tool_schemas_for_agent()
            + tools.user_tool_schemas()
        )

        # Lazy tool loading: cut the schemas list down to (always-loaded
        # meta-tools + tools the model has explicitly loaded for this
        # conversation via `tool_load`). The system prompt's manifest
        # section already lists every tool's name + 1-liner so the model
        # knows what's available; loading is what brings the full schema
        # into the `tools=[...]` payload (or the adapter's prompt-space
        # block). Big win on small-context local models — the default
        # full payload is ~18 K tokens of schema, and most conversations
        # only ever touch a handful of tools.
        try:
            loaded_set = set(db.get_loaded_tools(conversation_id))
        except Exception:
            loaded_set = set()
        # Always include the meta-tools so the model can never lock
        # itself out of further tool loads.
        loaded_set.update(tools.ALWAYS_LOADED_TOOLS)
        merged_schemas = [
            s for s in merged_schemas
            if (((s.get("function") or {}).get("name")) in loaded_set)
        ]

        # Phase 2 auto-router: decide whether to route this turn to a
        # transparent layer-split llama-server or to Ollama. The router
        # checks the model's GGUF size against host VRAM and the
        # connected workers' rpc-server availability; it auto-spawns
        # llama-server with --rpc when the model exceeds host budget,
        # auto-stops it when it doesn't. Everything is invisible to the
        # user — they just pick a model from the picker.
        split_target: tuple[str, str] | None = None
        try:
            decision = await compute_pool.route_chat_for(conv["model"])
        except compute_pool.RouteChatError as e:
            # Two failure modes:
            # 1. Acquisition in progress (Scope B): override / mmproj
            #    files are being prepared — phase / progress in
            #    `e.status`. Persist a friendly "preparing the model"
            #    message that the user can act on (wait + retry) and
            #    yield a structured `preparing_model` event so the UI
            #    can render a progress bar instead of the generic
            #    error toast.
            # 2. Hard fail: model can't be served at all (doesn't fit
            #    combined pool, missing GGUF, etc.). Persist a clear
            #    error message so the user isn't stuck on a spinner.
            status = getattr(e, "status", {}) or {}
            preparing_phases = (
                "starting", "running", "init",
                "surgery", "downloading-main", "downloading-mmproj",
            )
            if status.get("status") in preparing_phases or status.get("phase") in preparing_phases:
                msg = (
                    f"[compute pool] Preparing {conv['model']!r} for the pool: "
                    f"{status.get('phase', 'starting')} "
                    f"({status.get('progress_pct', 0):.0f}%). "
                    f"~{status.get('estimated_total_gb', 0):.1f} GB to fetch in total. "
                    "Please retry in a few minutes."
                )
                db.add_message(conversation_id, "assistant", msg)
                yield {
                    "type": "preparing_model",
                    "model": conv["model"],
                    "status": status,
                    "message": msg,
                }
                return
            db.add_message(
                conversation_id, "assistant",
                f"[compute pool] Could not run {conv['model']!r}: {e}",
            )
            yield {"type": "error", "error": str(e)}
            return
        except Exception as e:
            log.warning("compute_pool: route_chat_for failed: %s", e)
            decision = {"engine": "ollama"}

        if decision.get("engine") == "llama_server":
            split_target = (decision["base_url"], decision["label"])

        # Compute-pool Phase 1 routing for the parent chat turn (only
        # when the auto-router didn't pick a split target). Same picker
        # rules as embeddings/subagents — pick a worker iff one is
        # enabled, `use_for_chat=True`, freshly probed, and has the
        # model installed. If no worker is eligible we stay on the host
        # (the common case before the user adds devices). Pick once per
        # turn so the entire tool-loop hits the same Ollama: KV-cache
        # warmth survives across streamed tool-call iterations.
        chat_base_url = OLLAMA_URL
        chat_auth_token: str | None = None
        if split_target is None:
            try:
                # Pass conv_id so the picker honors per-conversation
                # affinity: a chat that landed on worker-A on its
                # previous turn keeps landing there as long as A is
                # eligible — KV cache survives across follow-up
                # turns. Concurrent conversations naturally spread
                # across the pool because the picker breaks ties on
                # active-turn count.
                chat_target = compute_pool.pick_chat_target(
                    conv["model"], conv_id=conversation_id,
                )
            except Exception:
                chat_target = None
            if chat_target is not None:
                chat_base_url, chat_auth_token = chat_target
            # Register this turn against the chosen node so future
            # picks see accurate load. The matching `register_turn_end`
            # runs in `run_turn`'s finally block — we read the node id
            # back from `_CONV_AFFINITY[conversation_id]` there.
            try:
                _node = compute_pool._node_id_for_target(chat_target)
                compute_pool.register_turn_start(conversation_id, _node)
            except Exception:
                pass

        # Prompt-space tool adapter: some Ollama models advertise the
        # `tools` capability but ship a passthrough chat template
        # (`{{ .Prompt }}`) that never renders `.Tools`. The native
        # function-calling path silently fails on those — the model sees
        # no tool list and refuses to act. We detect that once per
        # `(model, ollama_url)` (templates can technically differ between
        # host and a worker running a different Ollama version) and when
        # the adapter is needed we:
        #   * append an XML-tagged tool block to the system prompt,
        #   * rewrite historical `role=tool` rows and assistant
        #     `tool_calls` into inline <tool_call> / <tool_result> text,
        #   * drop `tools=[...]` from the Ollama payload (see
        #     `_stream_ollama_chat`),
        #   * parse any <tool_call> blocks out of the streamed text after
        #     the response completes (below).
        # Split-model chat ALWAYS uses adapter mode. llama-server's
        # /v1/chat/completions speaks OpenAI tool format, not Ollama's,
        # and we don't translate that here — the prompt-space adapter
        # inlines tool definitions in the system prompt and parses
        # <tool_call>…</tool_call> out of plain assistant text. This
        # keeps the agent loop's tool plumbing identical for both paths.
        if split_target is not None:
            adapter_mode = True
        else:
            adapter_mode = await tool_prompt_adapter.needs_adapter(
                conv["model"], chat_base_url, auth_token=chat_auth_token,
            )
        if adapter_mode:
            ollama_msgs = tool_prompt_adapter.inject_tools_block_into_system(
                ollama_msgs, merged_schemas,
            )
            ollama_msgs = tool_prompt_adapter.rewrite_messages_for_adapter(
                ollama_msgs,
            )

        # Copy of ollama_msgs that we may append a nudge to on retry.
        # Keep the original `ollama_msgs` untouched so the outer iteration
        # loop's persisted history doesn't carry a synthetic system note.
        stream_msgs = list(ollama_msgs)
        accumulated_text = ""
        accumulated_thinking = ""
        tool_calls_buf: list[dict] = []
        empty_retries = 0
        # Set when the last stream ended with an unclosed <tool_call>. Drives
        # the retry path to use `_TRUNCATED_TOOL_CALL_NUDGE` instead of the
        # plain empty-response nudge.
        truncation_retry_pending = False

        # Defensive partial-persist helper. Used both at the natural end
        # of the streaming loop AND inside the CancelledError branch so
        # a Stop / disconnect / network drop saves whatever the model
        # had streamed so far. Idempotent: a second call after a
        # successful persist is a no-op (gated on `_state["persisted_id"]`).
        _state = {"persisted_id": None}

        def _persist_partial(*, partial: bool = False) -> dict | None:
            """Write `accumulated_text` (and any `tool_calls_buf`) to the DB.

            With `partial=True` the body is annotated with a
            `[stopped mid-response]` marker so the user can tell the
            transcript holds the in-flight prefix, not a finished
            answer. Pure-thinking outputs (model wedged in reasoning,
            no answer yet) are surfaced too — strictly better than
            losing thousands of tokens of work.
            """
            nonlocal accumulated_text
            if _state["persisted_id"]:
                return None
            body = accumulated_text
            if partial:
                if not body.strip() and accumulated_thinking.strip():
                    body = (
                        "(stopped before producing an answer; "
                        "partial reasoning shown below)\n\n"
                        + accumulated_thinking.strip()
                    )
                if body and not body.rstrip().endswith("[stopped mid-response]"):
                    body = body.rstrip() + "\n\n[stopped mid-response]"
                elif not body and tool_calls_buf:
                    body = "[stopped mid-response]"
            if not body and not tool_calls_buf:
                return None
            row = db.add_message(
                conversation_id,
                "assistant",
                body,
                tool_calls=tool_calls_buf or None,
            )
            _state["persisted_id"] = row["id"]
            accumulated_text = body
            return row

        while True:
            accumulated_text = ""
            accumulated_thinking = ""
            tool_calls_buf = []
            # Fresh filters per retry iteration — carrying state across a
            # retried request would leave a half-open tag in the buffer and
            # silently swallow the entire replacement response.
            delta_filter = _StreamTagFilter()
            thinking_filter = _StreamTagFilter()
            # Pick the right streamer for the target. For split models we
            # talk OpenAI shape on llama-server; for everything else we
            # talk Ollama shape. Both yield `{message: {role, content,
            # tool_calls}, done: bool}` so the rest of the loop is
            # streamer-agnostic.
            if split_target is not None:
                split_base, split_label = split_target
                stream_iter = _stream_llama_server_chat(
                    split_label, stream_msgs,
                    base_url=split_base,
                    auth_token=None,    # llama-server is loopback, no auth
                )
            else:
                stream_iter = _stream_ollama_chat(
                    conv["model"],
                    stream_msgs,
                    merged_schemas,
                    adapter_mode=adapter_mode,
                    # On retry we shut thinking off entirely so a model that
                    # got lost in reasoning is forced to produce the answer
                    # (or a tool call) directly.
                    disable_thinking=empty_retries > 0,
                    base_url=chat_base_url,
                    auth_token=chat_auth_token,
                )
            try:
                async for chunk in stream_iter:
                    # Honour Stop button mid-stream. Breaking here closes the
                    # httpx streaming context (via the async-generator protocol),
                    # which aborts the request to Ollama — so the local model
                    # stops generating as soon as the break lands instead of
                    # running to the end of the response that'll never be shown.
                    if is_stop_requested(conversation_id):
                        break
                    msg = chunk.get("message") or {}
                    delta = msg.get("content") or ""
                    thinking = msg.get("thinking") or ""
                    tcs = msg.get("tool_calls") or []

                    if thinking:
                        # Accumulate the RAW text so adapter post-parse and
                        # the "thinking-but-no-answer" fallback still see it;
                        # only the UI stream goes through the filter.
                        accumulated_thinking += thinking
                        safe_thinking = thinking_filter.feed(thinking)
                        if safe_thinking:
                            yield {"type": "thinking", "text": safe_thinking}
                    if delta:
                        accumulated_text += delta
                        safe_delta = delta_filter.feed(delta)
                        if safe_delta:
                            yield {"type": "delta", "text": safe_delta}
                    if tcs:
                        for tc in tcs:
                            fn = tc.get("function") or {}
                            raw_name = fn.get("name", "")
                            raw_args = fn.get("arguments", {}) or {}
                            # Silently redirect invented tool names (e.g. the
                            # model hallucinates `google:search`) to their
                            # canonical equivalent BEFORE the UI sees the
                            # tool_call event — so the approval card and tool
                            # result show the real tool that ran, not a
                            # fantasy one that would have errored.
                            canonical_name, canonical_args = tools.resolve_tool_alias(
                                raw_name, raw_args
                            )
                            tool_calls_buf.append(
                                {
                                    "id": tc.get("id") or f"call_{uuid.uuid4().hex[:10]}",
                                    "name": canonical_name,
                                    "args": canonical_args,
                                }
                            )
                    if chunk.get("done"):
                        break
                # Stream ended — release any literal tail that the filter was
                # holding back in case it was a partial opener. Unclosed tags
                # still drop their body here; the truncation detector below
                # looks at `accumulated_text` (which is unfiltered) and drives
                # the retry.
                tail_thinking = thinking_filter.flush()
                if tail_thinking:
                    yield {"type": "thinking", "text": tail_thinking}
                tail_delta = delta_filter.flush()
                if tail_delta:
                    yield {"type": "delta", "text": tail_delta}
            except httpx.HTTPError as e:
                yield {"type": "error", "message": f"ollama error: {e}"}
                return
            except asyncio.CancelledError:
                # The Stop button (frontend aborts the SSE fetch) and a
                # client disconnect both surface here as cancellation
                # raised at whatever Ollama-chunk await was current.
                # Without a defensive persist, the post-stream
                # `_persist_partial()` below never runs and thousands
                # of tokens of streamed answer (or pure thinking) are
                # silently lost. Save what we have with a "[stopped
                # mid-response]" marker, then re-raise so run_turn()'s
                # outer try/finally still flips state to idle. The DB
                # write is sync (sqlite) so it can't itself be
                # cancelled.
                _persist_partial(partial=True)
                raise

            # Post-parse fallback. In adapter mode the model is *expected*
            # to emit tool calls as <tool_call>...</tool_call> tags inside
            # its streamed text. But some models that advertise native
            # tool-calling support (gemma4:e4b, several smaller Qwens)
            # still slip into the text-format every now and then —
            # announcing the call in prose, then dumping JSON between
            # `<tool_call>` tags, while leaving the structured channel
            # empty. Without this fallback the user sees the announcement
            # and then silence: the model "emits a tool but doesn't
            # actually run it." Run the parser whenever the structured
            # channel came up empty; it's a no-op (returns the text
            # unchanged, no calls) when no recognised tags are present,
            # so we don't pay a penalty on well-behaved models.
            #
            # Malformed JSON inside a block is left in-place by the
            # parser — the model sees its own mistake on the next turn
            # and can fix it, rather than silently losing the call.
            if not tool_calls_buf and accumulated_text:
                cleaned, parsed_calls = (
                    tool_prompt_adapter.parse_tool_calls_from_text(
                        accumulated_text,
                    )
                )
                if parsed_calls:
                    accumulated_text = cleaned
                    for pc in parsed_calls:
                        canonical_name, canonical_args = tools.resolve_tool_alias(
                            pc["name"], pc["args"],
                        )
                        tool_calls_buf.append(
                            {
                                "id": f"call_{uuid.uuid4().hex[:10]}",
                                "name": canonical_name,
                                "args": canonical_args,
                            }
                        )
                else:
                    # No call parsed — the model may have started emitting
                    # a <tool_call> block but hit EOS before the closer.
                    # Detect that case so we can clear the broken JSON from
                    # the user-visible bubble and retry with a nudge.
                    lt = accumulated_text.lower()
                    _tags = ("tool_call", "execute_tool", "tool_code", "function_call")
                    has_opener = any(f"<{t}>" in lt for t in _tags)
                    has_closer = any(f"</{t}>" in lt for t in _tags)
                    if has_opener and not has_closer:
                        log.warning(
                            "truncated tool_call detected (len=%d) — "
                            "triggering retry with shorter-payload nudge.",
                            len(accumulated_text),
                        )
                        accumulated_text = ""
                        truncation_retry_pending = True

            # Empty-response retry. Thinking-capable models can emit only
            # reasoning tokens and close the stream without ever producing
            # `content` or a tool call — the user sees a blank bubble. Also
            # fires for truncated `<tool_call>` blocks cleared above. One
            # forceful nudge gets the model to actually answer. Skip the
            # retry when the user pressed Stop mid-stream (empty is expected).
            if (
                not tool_calls_buf
                and not accumulated_text.strip()
                and empty_retries < _MAX_EMPTY_RETRIES
                and not is_stop_requested(conversation_id)
            ):
                empty_retries += 1
                nudge = (
                    _TRUNCATED_TOOL_CALL_NUDGE
                    if truncation_retry_pending
                    else _EMPTY_NUDGE
                )
                retry_reason = (
                    "truncated_tool_call" if truncation_retry_pending else "empty"
                )
                truncation_retry_pending = False
                yield {"type": "stream_retry", "reason": retry_reason}
                stream_msgs = stream_msgs + [
                    {"role": "system", "content": nudge}
                ]
                continue

            break

        # Absolute-last-resort fallback: if the retry loop above still left
        # us with nothing to show AND we have thinking tokens, expose them as
        # a parenthetical so the user sees some reasoning instead of a blank
        # bubble. This is purely a user-experience safety net — the retry
        # should have handled the common case.
        if (
            not tool_calls_buf
            and not accumulated_text.strip()
            and accumulated_thinking.strip()
            and not is_stop_requested(conversation_id)
        ):
            accumulated_text = (
                "(The model produced reasoning but no final answer. "
                "Try rephrasing, switching to a model without `thinking` "
                "capability, or lowering context size.)"
            )

        # Persist the assistant turn (clean exit). The CancelledError
        # branch around the streaming loop calls the same helper with
        # `partial=True`; this branch finishes a complete turn.
        assistant_row = _persist_partial()
        if assistant_row is None:
            # The model produced nothing AND emitted no tool calls AND
            # we have no thinking to surface. Yield a synthetic event
            # so downstream code that expects an `assistant_message`
            # tick (UI, tests) doesn't get confused.
            yield {
                "type": "assistant_message",
                "id": None,
                "content": "",
                "tool_calls": [],
            }
        else:
            # Embed prose replies too so semantic recall can surface
            # relevant past answers. Tool-call-only rows are skipped —
            # they carry no searchable text.
            if accumulated_text and accumulated_text.strip():
                asyncio.create_task(
                    _schedule_embedding(
                        assistant_row["id"], conversation_id, accumulated_text
                    )
                )
            yield {
                "type": "assistant_message",
                "id": assistant_row["id"],
                "content": accumulated_text,
                "tool_calls": tool_calls_buf,
            }

        # If the Stop button was pressed during streaming, the partial
        # assistant message has been persisted (so the transcript isn't
        # empty); bail out before running any tool calls the model asked
        # for — the user clearly doesn't want those side effects.
        if is_stop_requested(conversation_id):
            yield {"type": "error", "message": "stopped by user"}
            yield {"type": "turn_done"}
            return

        if not tool_calls_buf:
            # -------- Lifecycle hook: turn_done ---------------------------
            # Fires once when the agent produces a final answer (no more
            # tool calls). Hook output is appended to the final assistant
            # turn as a system note for continuity. Failures are reported
            # via SSE but don't prevent the turn from closing cleanly.
            try:
                done_hook_results = await tools.run_hooks(
                    "turn_done",
                    {
                        "event": "turn_done",
                        "conversation_id": conversation_id,
                        "cwd": conv["cwd"],
                        "final_text": accumulated_text,
                    },
                    conv_id=conversation_id,
                )
            except Exception:
                done_hook_results = []
            hook_done_outputs: list[str] = []
            for h in done_hook_results:
                yield {
                    "type": "hook_ran",
                    "event": "turn_done",
                    "hook_id": h.get("hook_id"),
                    "command": h.get("command"),
                    "ok": h.get("ok"),
                    "output": h.get("output", ""),
                    "error": h.get("error"),
                }
                out = (h.get("output") or "").strip()
                if out:
                    hook_done_outputs.append(
                        f"[turn_done hook `{h.get('command', '')}`]\n{out}"
                    )
            if hook_done_outputs:
                # Persist so the next turn's context still sees the hook note.
                db.add_system_summary(
                    conversation_id,
                    "\n\n".join(hook_done_outputs),
                )
            yield {"type": "turn_done"}
            # Follow-ups that landed in the input queue while we were
            # working: loop back so the top-of-iteration drain picks them
            # up and the model replies on the SAME SSE stream. Without
            # this we'd return here and the queued message would sit in
            # the DB until the user pressed Send again.
            if db.has_queued_inputs(conversation_id):
                continue
            return

        # Execute tool calls — fan out in parallel across every call the
        # model emitted in this turn (reads AND writes). Each call runs as
        # its own asyncio.Task; their SSE events stream through a shared
        # asyncio.Queue so the UI sees `tool_call` / `tool_result` / hook
        # events in real time as each tool progresses. The top-level await
        # loop below drains the queue until a sentinel signals "all tasks
        # finished", then the outer while-loop asks the model for its next
        # turn with the full batch of tool results in history.
        conv = db.get_conversation(conversation_id) or conv
        permission_mode = conv.get("permission_mode") or (
            "allow_all" if conv.get("auto_approve") else "approve_edits"
        )
        _TOOL_QUEUE_SENTINEL: object = object()
        tool_event_queue: asyncio.Queue = asyncio.Queue()
        # Subagent progress bus — subagents (running inside a `delegate` /
        # `delegate_parallel` tool) publish progress events here. A small
        # drain task below forwards them onto tool_event_queue so they
        # interleave with parent tool events on the SSE stream.
        subagent_bus = _register_subagent_bus(conversation_id)

        async def _drain_subagent_bus() -> None:
            """Forward subagent progress events onto the main queue until
            cancelled. Cancelled on turn end — see the `finally` below.
            """
            try:
                while True:
                    evt = await subagent_bus.get()
                    await tool_event_queue.put(evt)
            except asyncio.CancelledError:
                return

        subagent_drain_task = asyncio.create_task(_drain_subagent_bus())
        # Serialize shell calls within a single turn so `cd` (and any other
        # cwd-sensitive step) in one bash call persists to the next. Without
        # this, a model that emits `mkdir foo` + `cd foo` + `npm install` as
        # parallel calls would have all three start from the same cwd —
        # `npm install` races `cd` and almost always wins, blowing up with
        # ENOENT. Non-bash tools still run fully in parallel.
        _bash_serializer: asyncio.Lock = asyncio.Lock()

        async def _run_one_tool_call(tc: dict) -> None:
            """Execute a single tool call and push every SSE event onto the
            shared queue. Mirrors the old sequential body 1:1 — every `yield`
            became `await tool_event_queue.put(...)` so we can drive many
            calls concurrently via asyncio.gather."""
            label = tools.describe_tool_call(tc["name"], tc["args"])
            # `reason` is a display-only field — the tool body ignores it,
            # but the UI surfaces it on the approval card.
            reason = str(tc["args"].get("reason") or "").strip()
            await tool_event_queue.put({
                "type": "tool_call",
                "id": tc["id"],
                "name": tc["name"],
                "args": tc["args"],
                "label": label,
                "reason": reason,
            })

            # Permission gate — same four-mode matrix as before:
            #   read_only / plan  — writes refused outright (no approval UI).
            #   approve_edits     — reads silent; writes pause per-tool.
            #   allow_all         — nothing pauses.
            category = tools.classify_tool(tc["name"])
            approved = True

            if permission_mode in ("read_only", "plan") and category == "write":
                mode_label = "plan" if permission_mode == "plan" else "read-only"
                if permission_mode == "plan":
                    err = (
                        f"plan mode is active: `{tc['name']}` is a write-"
                        "category tool and cannot run here. Finish your plan "
                        "first — the user will review it, then click 'Execute "
                        "plan' to switch the conversation out of plan mode and "
                        "replay the plan for execution."
                    )
                else:
                    err = (
                        f"permission denied: this conversation is in read-only "
                        f"mode, and `{tc['name']}` is a write-category tool. "
                        "Switch the conversation to 'approve_edits' or "
                        "'allow_all' to run it."
                    )
                db.add_message(
                    conversation_id,
                    "tool",
                    f"[permission denied — {mode_label} mode: {tc['name']}]",
                    tool_calls=[{"id": tc["id"], "name": tc["name"]}],
                )
                await tool_event_queue.put({
                    "type": "tool_result",
                    "id": tc["id"],
                    "ok": False,
                    "output": "",
                    "error": err,
                })
                return

            if permission_mode == "approve_edits" and category == "write":
                approval_id = tc["id"]
                loop = asyncio.get_running_loop()
                fut: asyncio.Future = loop.create_future()
                _pending_approvals[approval_id] = (fut, loop)
                preview = _preview_for_write(
                    conv["cwd"], tc["name"], tc["args"]
                )
                await tool_event_queue.put({
                    "type": "await_approval",
                    "id": approval_id,
                    "name": tc["name"],
                    "args": tc["args"],
                    "label": label,
                    "reason": reason,
                    "preview": preview,
                })
                try:
                    approved = await fut
                finally:
                    _pending_approvals.pop(approval_id, None)

            if not approved:
                result_text = "[User rejected the tool call. Do not retry it without explicit permission.]"
                db.add_message(
                    conversation_id,
                    "tool",
                    result_text,
                    tool_calls=[{"id": tc["id"], "name": tc["name"]}],
                )
                await tool_event_queue.put({
                    "type": "tool_result",
                    "id": tc["id"],
                    "ok": False,
                    "output": "",
                    "error": "rejected by user",
                })
                return

            # -------- Lifecycle hook: pre_tool ------------------------------
            # Observational: hook failures are logged but never block the
            # tool. Matcher on the hook row can scope it to specific names.
            try:
                pre_hook_results = await tools.run_hooks(
                    "pre_tool",
                    {
                        "event": "pre_tool",
                        "conversation_id": conversation_id,
                        "cwd": conv["cwd"],
                        "tool_name": tc["name"],
                        "tool_args": tc["args"],
                    },
                    tool_name=tc["name"],
                    conv_id=conversation_id,
                )
            except Exception:
                pre_hook_results = []
            for h in pre_hook_results:
                await tool_event_queue.put({
                    "type": "hook_ran",
                    "event": "pre_tool",
                    "hook_id": h.get("hook_id"),
                    "command": h.get("command"),
                    "ok": h.get("ok"),
                    "output": h.get("output", ""),
                    "error": h.get("error"),
                })

            # ask_user_question is intercepted BEFORE dispatch so we can emit
            # the SSE event and await the user clicking a rendered button.
            # The tool's Python body is a stub; we bypass it entirely here.
            if tc["name"] == "ask_user_question":
                q = str(tc["args"].get("question") or "").strip()
                raw_options = tc["args"].get("options") or []
                err = tools._validate_ask_question_args(q, raw_options)
                if err:
                    result = {"ok": False, "output": "", "error": err}
                else:
                    answer_id = tc["id"]
                    loop = asyncio.get_running_loop()
                    fut2: asyncio.Future = loop.create_future()
                    _pending_answers[answer_id] = (fut2, loop)
                    # Normalize options to {label, value} dicts for the UI —
                    # the validator already confirmed the shape.
                    norm_options = [
                        {"label": o["label"], "value": o["value"]}
                        for o in raw_options
                    ]
                    await tool_event_queue.put({
                        "type": "await_user_answer",
                        "id": answer_id,
                        "question": q,
                        "options": norm_options,
                    })
                    try:
                        answer_value = await fut2
                        result = {
                            "ok": True,
                            "output": f"user chose: {answer_value}",
                            "answer": answer_value,
                        }
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        result = {
                            "ok": False, "output": "",
                            "error": f"ask_user_question failed: {e}",
                        }
                    finally:
                        _pending_answers.pop(answer_id, None)
            else:
                # Serialize bash/bash_bg so cwd (and any shell side-effects)
                # from earlier calls in the batch are visible to later ones.
                if tc["name"] in ("bash", "bash_bg"):
                    async with _bash_serializer:
                        result = await tools.dispatch(
                            tc["name"],
                            tc["args"],
                            conv["cwd"],
                            conv_id=conversation_id,
                            model=conv["model"],
                            tool_call_id=tc["id"],
                        )
                else:
                    result = await tools.dispatch(
                        tc["name"],
                        tc["args"],
                        conv["cwd"],
                        conv_id=conversation_id,
                        model=conv["model"],
                        tool_call_id=tc["id"],
                    )

            # -------- Lifecycle hook: post_tool -----------------------------
            # Output is appended to the tool result text the model sees, so a
            # hook can annotate (e.g. 'write_file changed 3 files').
            tool_ok = bool(result.get("ok"))
            tool_payload = {
                "event": "post_tool",
                "conversation_id": conversation_id,
                "cwd": conv["cwd"],
                "tool_name": tc["name"],
                "tool_args": tc["args"],
                "ok": tool_ok,
                "output": result.get("output", ""),
                "error": result.get("error"),
            }
            try:
                post_hook_results = await tools.run_hooks(
                    "post_tool",
                    tool_payload,
                    tool_name=tc["name"],
                    conv_id=conversation_id,
                )
            except Exception:
                post_hook_results = []
            hook_appendix_parts: list[str] = []
            for h in post_hook_results:
                await tool_event_queue.put({
                    "type": "hook_ran",
                    "event": "post_tool",
                    "hook_id": h.get("hook_id"),
                    "command": h.get("command"),
                    "ok": h.get("ok"),
                    "output": h.get("output", ""),
                    "error": h.get("error"),
                })
                out = (h.get("output") or "").strip()
                if out:
                    hook_appendix_parts.append(
                        f"[post_tool hook `{h.get('command', '')}`]\n{out}"
                    )
            if hook_appendix_parts:
                # Mutate the result so the persisted tool row carries the
                # hook notes — that way history replays stay consistent.
                appended = (result.get("output") or "").rstrip()
                separator = "\n\n" if appended else ""
                result["output"] = appended + separator + "\n\n".join(hook_appendix_parts)

            # -------- Workflow triggers: tool_error + consecutive_failures ---
            # These fire AFTER post_tool because they're diagnostic-tier:
            # the user wires them up to spot patterns the model can't fix
            # itself ("agent looped on the same broken call 3 times → ask
            # Claude to step in"). Their output is injected as a SEPARATE
            # system-role message rather than appended to the tool result,
            # so it lands in the next Ollama turn with higher prominence
            # — the model knows this is a workflow signal, not just more
            # tool output.
            workflow_notes: list[str] = []
            new_count = _bump_consec_failures(
                conversation_id, tc["name"], tool_ok,
            )
            if not tool_ok:
                # tool_error — fires on every failure (matcher applies).
                try:
                    err_hook_results = await tools.run_hooks(
                        "tool_error",
                        tool_payload,
                        tool_name=tc["name"],
                        conv_id=conversation_id,
                    )
                except Exception:
                    err_hook_results = []
                # consecutive_failures — fires when the streak hits the
                # per-hook `error_threshold`. Multiple hooks can register
                # on this event with different thresholds; we fire each
                # one whose threshold is now met.
                try:
                    candidates = db.get_hooks_for_event("consecutive_failures")
                except Exception:
                    candidates = []
                consec_hook_results: list[dict] = []
                for hook_row in candidates:
                    threshold = int(hook_row.get("error_threshold") or 1)
                    if new_count < threshold:
                        continue
                    # Reuse the run_hooks cap + tracking machinery for
                    # this single hook by calling it directly with a
                    # narrowed event set. Easier: just enforce the cap
                    # inline and run the single hook.
                    cap = hook_row.get("max_fires_per_conv")
                    if cap is not None:
                        try:
                            seen = db.get_hook_fire_count(
                                hook_row["id"], conversation_id,
                            )
                        except Exception:
                            seen = 0
                        if seen >= cap:
                            consec_hook_results.append({
                                "hook_id": hook_row["id"],
                                "command": hook_row["command"],
                                "event": "consecutive_failures",
                                "ok": False,
                                "output": "",
                                "error": (
                                    f"hook hit its per-conversation cap "
                                    f"of {cap} fires"
                                ),
                                "capped": True,
                            })
                            continue
                    if not tools._hook_matches(hook_row, tc["name"]):
                        continue
                    payload = {
                        **tool_payload,
                        "event": "consecutive_failures",
                        "consecutive_count": new_count,
                        "threshold": threshold,
                    }
                    one = await tools._run_single_hook(hook_row, payload)
                    try:
                        db.incr_hook_fire(hook_row["id"], conversation_id)
                    except Exception:
                        pass
                    consec_hook_results.append({
                        "hook_id": hook_row["id"],
                        "command": hook_row["command"],
                        "event": "consecutive_failures",
                        "ok": bool(one.get("ok")),
                        "output": one.get("output", ""),
                        "error": one.get("error"),
                        "capped": False,
                    })
                for h in (err_hook_results + consec_hook_results):
                    await tool_event_queue.put({
                        "type": "hook_ran",
                        "event": h.get("event"),
                        "hook_id": h.get("hook_id"),
                        "command": h.get("command"),
                        "ok": h.get("ok"),
                        "output": h.get("output", ""),
                        "error": h.get("error"),
                        "capped": h.get("capped"),
                    })
                    out = (h.get("output") or "").strip()
                    if out:
                        evt_label = h.get("event") or "workflow"
                        workflow_notes.append(
                            f"[{evt_label} hook `{h.get('command', '')}`]\n{out}"
                        )
            if workflow_notes:
                # Persist as a system-role row so the model sees it on
                # the next Ollama call and the note survives compaction.
                # Higher prominence than post_tool appendix because
                # diagnostic hooks (e.g. Claude-fixer) speak ABOVE the
                # tool layer.
                db.add_system_summary(
                    conversation_id,
                    "\n\n".join(workflow_notes),
                )
            result_text = json.dumps(
                {
                    "ok": result.get("ok"),
                    "output": result.get("output", ""),
                    "error": result.get("error"),
                    "exit_code": result.get("exit_code"),
                },
                ensure_ascii=False,
            )
            # Persist the tool row. Carry `image_path` (screenshot filename)
            # on the tool_calls metadata so future turns can re-attach the
            # image when replaying history to Ollama, and so the UI can
            # re-render the thumbnail after a reload.
            tool_row_calls: list[dict[str, Any]] = [
                {"id": tc["id"], "name": tc["name"]}
            ]
            image_path = result.get("image_path")
            if image_path:
                tool_row_calls[0]["image_path"] = image_path
            db.add_message(
                conversation_id,
                "tool",
                result_text,
                tool_calls=tool_row_calls,
            )
            await tool_event_queue.put({
                "type": "tool_result",
                "id": tc["id"],
                "ok": bool(result.get("ok")),
                "output": result.get("output", ""),
                "error": result.get("error"),
                "image_path": image_path,
            })

            # Per-tool side-channel events (UI refresh triggers).
            if tc["name"] == "todo_write" and result.get("ok") and isinstance(result.get("todos"), list):
                await tool_event_queue.put({
                    "type": "todos_updated", "todos": result["todos"]
                })
            if tc["name"] == "spawn_task" and result.get("ok") and isinstance(result.get("side_task"), dict):
                await tool_event_queue.put({
                    "type": "side_task_flagged",
                    "side_task": result["side_task"],
                })
            if tc["name"] == "schedule_wakeup" and result.get("ok"):
                await tool_event_queue.put({
                    "type": "wakeup_scheduled",
                    "id": result.get("id"),
                    "output": result.get("output", ""),
                })
            if tc["name"] in ("create_worktree", "remove_worktree") and result.get("ok"):
                await tool_event_queue.put({"type": "worktrees_changed"})

        # Spawn every tool call concurrently. asyncio.gather lets writes and
        # reads run in parallel — the user explicitly opted into this
        # (they'll approve each write card in whichever order they prefer).
        tool_tasks = [
            asyncio.create_task(_run_one_tool_call(tc))
            for tc in tool_calls_buf
        ]

        async def _signal_when_all_done() -> None:
            try:
                await asyncio.gather(*tool_tasks, return_exceptions=True)
                # Shut the subagent drain down cleanly FIRST, then flush
                # anything still in the bus. If we pushed the sentinel
                # without draining, the main loop could break before a
                # late-arriving subagent event was forwarded.
                subagent_drain_task.cancel()
                try:
                    await subagent_drain_task
                except asyncio.CancelledError:
                    pass
                while True:
                    try:
                        evt = subagent_bus.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    await tool_event_queue.put(evt)
            finally:
                await tool_event_queue.put(_TOOL_QUEUE_SENTINEL)

        done_watcher = asyncio.create_task(_signal_when_all_done())

        try:
            while True:
                evt = await tool_event_queue.get()
                if evt is _TOOL_QUEUE_SENTINEL:
                    break
                yield evt
        finally:
            # Best-effort cleanup if the outer generator is closed mid-stream
            # (e.g. client disconnected). Cancel any stragglers so their
            # file handles / subprocesses don't leak.
            for t in (*tool_tasks, done_watcher, subagent_drain_task):
                if not t.done():
                    t.cancel()
            # Stop forwarding subagent events and drop the bus — a later
            # turn on the same conversation will register a fresh queue.
            _unregister_subagent_bus(conversation_id)
        # Loop back: feed tool results (and any attached screenshots) to the
        # model for the next iteration.

    yield {
        "type": "error",
        "message": f"max tool iterations ({MAX_TOOL_ITERATIONS}) exceeded",
    }


# ---------------------------------------------------------------------------
# Subagent (used by the `delegate` tool)
#
# Runs a focused, ephemeral agent loop without persisting anything to the DB.
# Returns a compact summary string the parent agent can incorporate. This is
# deliberately stripped-down: no approval flow (the parent already had the
# user approve the delegation), no streaming, no screenshots, no nested
# delegation. If we need those later we can layer them in.
# ---------------------------------------------------------------------------
SUBAGENT_TYPES = {
    "general": {
        "label": "general",
        "description": "Balanced reader/writer for open-ended delegated work.",
        "prompt": "",  # no extra prompt — use the base subagent description
        "forbid_writes": False,
    },
    "explorer": {
        "label": "explorer",
        "description": (
            "Read-only researcher for codebase/file-system exploration. Cannot "
            "run write tools (edit, write, bash, python_exec, docker, http)."
        ),
        "prompt": (
            "\n\nYou are an EXPLORER subagent. Focus on fast, broad discovery: "
            "search the tree with `glob` / `grep`, sample files with `read_file`, "
            "check directory layouts with `list_dir`. Do NOT attempt to modify "
            "anything — writes are refused. End with a concise summary of what "
            "you found (file paths, line numbers, patterns) so the parent agent "
            "can act on it."
        ),
        "forbid_writes": True,
    },
    "architect": {
        "label": "architect",
        "description": (
            "Read-only planner that produces a step-by-step implementation "
            "plan. Allowed to read and search but not to write."
        ),
        "prompt": (
            "\n\nYou are an ARCHITECT subagent. Your job is to produce a "
            "concrete, step-by-step implementation plan for the delegated "
            "task. Use read tools (`glob`, `grep`, `read_file`, `list_dir`) "
            "to ground the plan in the actual codebase. Do NOT attempt to "
            "edit or run anything. End with a numbered plan the parent can "
            "execute — include file paths, function names, and what changes "
            "where. Call out non-obvious trade-offs briefly."
        ),
        "forbid_writes": True,
    },
    "reviewer": {
        "label": "reviewer",
        "description": (
            "Read-only code reviewer. Inspects a diff or set of files and "
            "surfaces bugs, security issues, and inconsistencies."
        ),
        "prompt": (
            "\n\nYou are a REVIEWER subagent. Read the target files or diff, "
            "look for bugs, security issues, dead code, and inconsistencies "
            "with the surrounding codebase. Do NOT modify anything — writes "
            "are refused. End with a short list: 🔴 critical, 🟡 concerns, "
            "🟢 nits. If nothing is wrong, say so clearly rather than "
            "padding with trivia."
        ),
        "forbid_writes": True,
    },
}


# Write-category tool names a read-only subagent type (explorer / architect /
# reviewer) must not be allowed to call. Mirrors tools.TOOL_CATEGORIES for the
# common write tools; built dynamically at first use to stay in sync.
_SUBAGENT_WRITE_FORBIDDEN: set[str] | None = None


def _get_subagent_write_forbidden() -> set[str]:
    global _SUBAGENT_WRITE_FORBIDDEN
    if _SUBAGENT_WRITE_FORBIDDEN is None:
        _SUBAGENT_WRITE_FORBIDDEN = {
            name for name, cat in tools.TOOL_CATEGORIES.items() if cat == "write"
        }
    return _SUBAGENT_WRITE_FORBIDDEN


async def run_subagent(
    task: str,
    cwd: str,
    model: str,
    max_iterations: int = 10,
    subagent_type: str = "general",
    parent_conv_id: str | None = None,
    parent_tool_call_id: str | None = None,
    branch_id: str | None = None,
    *,
    base_url: str = OLLAMA_URL,
    auth_token: str | None = None,
) -> dict:
    """Execute `task` in an isolated agent loop. Returns the same
    {ok, output, error} shape as a normal tool result.

    The subagent cannot delegate further (to avoid runaway recursion).
    `subagent_type` picks one of SUBAGENT_TYPES — explorer / architect /
    reviewer are read-only variants with a tailored system prompt, general
    is the default balanced variant.

    `parent_conv_id` — when set, the subagent publishes progress events
    (`subagent_started`, `subagent_tool_call`, `subagent_tool_result`,
    `subagent_done`) onto the parent turn's progress bus so the UI can
    render nested activity under the `delegate` tool card in real time.
    `parent_tool_call_id` — the id of the delegate/delegate_parallel tool
    call that spawned this subagent; included in every event so the UI
    can attach it to the right card. `branch_id` disambiguates fan-out:
    set by `run_subagents_parallel` so the UI can group events per branch.
    """
    if not task or not task.strip():
        return {"ok": False, "output": "", "error": "empty task"}
    max_iterations = max(1, min(int(max_iterations or 10), 20))

    spec = SUBAGENT_TYPES.get(subagent_type) or SUBAGENT_TYPES["general"]

    # Stable identifier for this subagent run — the UI uses it to correlate
    # all events from one subagent (start → tool_call → tool_result → done).
    # Fall back to a fresh uuid when the caller didn't assign a branch id.
    subagent_id = branch_id or f"sub_{uuid.uuid4().hex[:8]}"

    _publish_subagent_event(parent_conv_id, {
        "type": "subagent_started",
        "parent_tool_call_id": parent_tool_call_id,
        "subagent_id": subagent_id,
        "subagent_type": spec.get("label") or subagent_type,
        "task": task.strip(),
    })

    # Subagent runs without the parent's memory file — it's a fresh scratch
    # context and shouldn't read or write the parent conversation's notes.
    system = build_system_prompt(cwd, conv_id=None) + (
        "\n\n## You are a SUBAGENT\n"
        "A parent agent delegated a focused task to you. Do the task using "
        "your tools, then reply with a concise summary the parent can act "
        "on. Do NOT call `delegate` or `delegate_parallel` (no nested "
        "subagents). Do NOT call desktop-control tools (`screenshot`, "
        "`computer_*`, `click_element`, `type_into_element`, `focus_window`, "
        "`open_app`, `window_*`, `inspect_window`, `ocr_screenshot`) or "
        "browser-control tools (`browser_*`) — the parent owns the desktop "
        "and browser. Do NOT call `schedule_task` / `list_scheduled_tasks` "
        "/ `cancel_scheduled_task` — scheduling is a parent-level concern. "
        "Do NOT call `schedule_wakeup`, `spawn_task`, or `ask_user_question` "
        "— those are parent-loop tools that need direct user interaction. "
        "Do NOT call `remember` / `forget` — you have no conversation "
        "context to save to. The `monitor` tool IS available — use it if "
        "you need to wait for a condition to flip."
    ) + (spec.get("prompt") or "")
    # Trim the tool schemas that subagents aren't allowed to use.
    # Desktop + browser + scheduling tools are parent-owned: screenshots and
    # clicks belong to whoever is at the keyboard, scheduled tasks spawn
    # standalone conversations (no nested scheduling from a subagent), and
    # CDP browser tools share the same desktop Chrome as the parent.
    forbidden = {
        "delegate",
        "delegate_parallel",
        "screenshot",
        "list_monitors",
        "computer_click",
        "computer_drag",
        "computer_type",
        "computer_key",
        "computer_scroll",
        "computer_mouse_move",
        "click_element",
        "click_element_id",
        "type_into_element",
        "focus_window",
        "open_app",
        "window_action",
        "window_bounds",
        "inspect_window",
        "ocr_screenshot",
        "ui_wait",
        "computer_batch",
        "screenshot_window",
        "list_windows",
        "browser_tabs",
        "browser_goto",
        "browser_click",
        "browser_type",
        "browser_text",
        "browser_eval",
        "schedule_task",
        "list_scheduled_tasks",
        "cancel_scheduled_task",
        # Parent-loop only — subagents have no chat to re-enter, no user to
        # ask, and no conversation to spawn side tasks from.
        "schedule_wakeup",
        "spawn_task",
        "ask_user_question",
        "remember",
        "forget",
    }
    # Specialized subagent types (explorer / architect / reviewer) are
    # read-only — merge every write-category tool into the forbidden set so
    # the model can't accidentally fall through to a write call.
    if spec.get("forbid_writes"):
        forbidden = forbidden | _get_subagent_write_forbidden()
    # Strip the lazy-load meta-tools from the subagent palette: subagents
    # are short-lived and have no per-invocation persisted loaded-set, so
    # `tool_search` / `tool_load` would just be dead schema noise. They
    # always get their full palette upfront — fewer round-trips matters
    # more than schema-prompt size for a 10-iteration sub-task.
    forbidden = forbidden | set(tools.ALWAYS_LOADED_TOOLS)
    schemas = [s for s in TOOL_SCHEMAS if s.get("function", {}).get("name") not in forbidden]
    # Subagents inherit MCP + user-defined tools — they're how a delegate
    # task can e.g. query an external database or call a utility the parent
    # agent registered earlier in the conversation.
    schemas = schemas + mcp.tool_schemas_for_agent() + tools.user_tool_schemas()

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]

    collected_text = ""
    steps_log: list[str] = []

    for _ in range(max_iterations):
        acc_text = ""
        tool_calls: list[dict] = []
        try:
            async for chunk in _stream_ollama_chat(
                model, messages, schemas,
                base_url=base_url, auth_token=auth_token,
            ):
                msg = chunk.get("message") or {}
                if msg.get("content"):
                    acc_text += msg["content"]
                for tc in (msg.get("tool_calls") or []):
                    fn = tc.get("function") or {}
                    sub_name, sub_args = tools.resolve_tool_alias(
                        fn.get("name", ""), fn.get("arguments", {}) or {}
                    )
                    tool_calls.append(
                        {
                            "id": tc.get("id") or f"sub_{uuid.uuid4().hex[:8]}",
                            "name": sub_name,
                            "args": sub_args,
                        }
                    )
                if chunk.get("done"):
                    break
        except httpx.HTTPError as e:
            err = f"subagent ollama error: {e}"
            _publish_subagent_event(parent_conv_id, {
                "type": "subagent_done",
                "parent_tool_call_id": parent_tool_call_id,
                "subagent_id": subagent_id,
                "ok": False,
                "steps": len(steps_log),
                "error": err,
            })
            return {"ok": False, "output": "", "error": err}

        assistant_msg: dict = {"role": "assistant", "content": acc_text}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {"function": {"name": tc["name"], "arguments": tc["args"]}}
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

        collected_text = acc_text or collected_text
        if not tool_calls:
            # Final answer — subagent is done.
            break

        for tc in tool_calls:
            step_label = tools.describe_tool_call(tc["name"], tc["args"])
            steps_log.append(step_label)
            _publish_subagent_event(parent_conv_id, {
                "type": "subagent_tool_call",
                "parent_tool_call_id": parent_tool_call_id,
                "subagent_id": subagent_id,
                "tool_call_id": tc["id"],
                "name": tc["name"],
                "args": tc["args"],
                "label": step_label,
            })
            # Guard against an "off-list" call the model hallucinated.
            if tc["name"] in forbidden:
                result = {"ok": False, "output": "", "error": f"tool not available to subagent: {tc['name']}"}
            elif (
                tc["name"] not in tools.TOOL_REGISTRY
                and tc["name"] != "delegate"
                and not mcp.is_mcp_tool(tc["name"])
                and not tools.classify_user_tool(tc["name"])
            ):
                result = {"ok": False, "output": "", "error": f"tool not available to subagent: {tc['name']}"}
            else:
                result = await tools.dispatch(
                    tc["name"],
                    tc["args"],
                    cwd,
                    conv_id=None,       # no DB / checkpoint context
                    model=model,
                )
            _publish_subagent_event(parent_conv_id, {
                "type": "subagent_tool_result",
                "parent_tool_call_id": parent_tool_call_id,
                "subagent_id": subagent_id,
                "tool_call_id": tc["id"],
                "name": tc["name"],
                "ok": bool(result.get("ok")),
                # Truncate the output for the progress chip — the full text
                # still gets embedded in the subagent's final summary that
                # the parent agent consumes.
                "output": (result.get("output") or "")[:500],
                "error": result.get("error"),
            })
            messages.append(
                {
                    "role": "tool",
                    "tool_name": tc["name"],
                    "content": json.dumps(
                        {
                            "ok": result.get("ok"),
                            "output": result.get("output", ""),
                            "error": result.get("error"),
                        },
                        ensure_ascii=False,
                    ),
                }
            )

    summary_lines = [
        "Subagent summary:",
        collected_text.strip() or "(no final answer produced)",
    ]
    if steps_log:
        summary_lines.append("")
        summary_lines.append("Steps taken:")
        for s in steps_log[:25]:
            summary_lines.append(f"- {s}")
        if len(steps_log) > 25:
            summary_lines.append(f"- ...and {len(steps_log) - 25} more")
    final_output = "\n".join(summary_lines)
    _publish_subagent_event(parent_conv_id, {
        "type": "subagent_done",
        "parent_tool_call_id": parent_tool_call_id,
        "subagent_id": subagent_id,
        "ok": True,
        "steps": len(steps_log),
        "summary": (collected_text.strip() or "")[:500],
    })
    return {"ok": True, "output": final_output}


# ---------------------------------------------------------------------------
# Parallel subagents (used by the `delegate_parallel` tool)
#
# Runs N independent subagents concurrently with asyncio.gather so the parent
# agent can fan out research / code-search / summarisation tasks in one tool
# call. Each subagent is fully independent — they don't see each other's
# context and can't coordinate. Failures in one don't affect the others.
#
# Bounded fan-out (max 6) keeps a pathological call from saturating the local
# Ollama server, which has finite per-request concurrency.
# ---------------------------------------------------------------------------
_MAX_PARALLEL_SUBAGENTS = 6


async def run_subagents_parallel(
    tasks: list[str],
    cwd: str,
    model: str,
    max_iterations: int = 10,
    subagent_type: str = "general",
    parent_conv_id: str | None = None,
    parent_tool_call_id: str | None = None,
) -> dict:
    """Spawn `tasks` subagents concurrently; return combined results.

    Each task gets its own fresh scratch context and runs in parallel via
    `asyncio.gather`. Partial failures are surfaced in the output string
    (one subagent failing doesn't fail the whole call) so the parent agent
    can see exactly which branch worked. `subagent_type` applies to every
    fanned-out subagent — all tasks run with the same specialization.

    `parent_conv_id` / `parent_tool_call_id` — when set, every fanned-out
    subagent publishes progress events to the parent turn's bus, tagged
    with the originating tool call id and a unique per-branch id so the UI
    can render each branch's activity separately under the right card.
    """
    if not isinstance(tasks, list) or not tasks:
        return {"ok": False, "output": "", "error": "tasks must be a non-empty list of strings"}
    # Clean + validate inputs up front.
    cleaned: list[str] = []
    for i, t in enumerate(tasks):
        if not isinstance(t, str):
            return {"ok": False, "output": "", "error": f"tasks[{i}] must be a string"}
        t2 = t.strip()
        if not t2:
            return {"ok": False, "output": "", "error": f"tasks[{i}] is empty"}
        cleaned.append(t2)
    if len(cleaned) > _MAX_PARALLEL_SUBAGENTS:
        return {
            "ok": False, "output": "",
            "error": f"too many parallel tasks ({len(cleaned)}, max {_MAX_PARALLEL_SUBAGENTS})",
        }
    # Pre-assign a stable branch id per task so the UI can differentiate
    # sibling subagents in a delegate_parallel fan-out.
    branch_ids = [f"sub_{uuid.uuid4().hex[:8]}" for _ in cleaned]

    # Distribute across host + every eligible compute worker so the fan-out
    # actually parallelizes on hardware: 6 tasks across 1 host + 2 workers
    # schedules ~2 per machine. Fall through to host-only when no workers
    # are registered or eligible (the common case before the user adds
    # devices). Host is always slot 0 — gives the host first dibs on the
    # smallest fan-outs (1-2 tasks) since its loopback Ollama is the
    # cheapest hop and we don't want to wake up a sleeping laptop for a
    # single subagent task.
    targets: list[tuple[str, str | None]] = [(OLLAMA_URL, None)]
    try:
        targets.extend(compute_pool.list_subagent_workers(model))
    except Exception:
        # compute_pool can't fail here in practice (pure DB read), but
        # belt-and-braces: a routing layer hiccup must never break the
        # core delegate_parallel call. Stay host-only.
        pass

    # Fan out. return_exceptions=True turns a single subagent crash into a
    # value in the results list rather than aborting the gather.
    coros = []
    for i, (t, bid) in enumerate(zip(cleaned, branch_ids)):
        base_url, token = targets[i % len(targets)]
        coros.append(
            run_subagent(
                t, cwd, model, max_iterations,
                subagent_type=subagent_type,
                parent_conv_id=parent_conv_id,
                parent_tool_call_id=parent_tool_call_id,
                branch_id=bid,
                base_url=base_url,
                auth_token=token,
            )
        )
    results = await asyncio.gather(*coros, return_exceptions=True)

    # Format each result into its own labelled block so the parent agent can
    # see all branches in order.
    blocks: list[str] = []
    any_ok = False
    for i, (task, res) in enumerate(zip(cleaned, results), start=1):
        header = f"=== task {i}/{len(cleaned)} ==="
        if isinstance(res, BaseException):
            blocks.append(f"{header}\ntask: {task}\n\n[crashed: {type(res).__name__}: {res}]")
            continue
        if res.get("ok"):
            any_ok = True
            blocks.append(f"{header}\ntask: {task}\n\n{res.get('output', '').strip()}")
        else:
            blocks.append(
                f"{header}\ntask: {task}\n\n[failed: {res.get('error') or 'unknown error'}]"
            )
    return {
        "ok": True,   # Always True — partial failures are visible in the output.
        "output": "\n\n".join(blocks),
        "any_ok": any_ok,
        "count": len(cleaned),
    }
