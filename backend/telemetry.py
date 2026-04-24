"""Structured logging + per-tool timing.

Why this exists
---------------
Long agent sessions fail in subtle ways — a tool silently stalls, a
particular Ollama model takes 20 s on what should be a 2 s call, an MCP
server starts returning errors under load. Without structured timing data
we can only eyeball the SSE stream in the browser and guess.

This module gives us:
  * `get_logger(name)` — a `logging.Logger` configured once per process
    with a JSON-per-line formatter, so log lines are grep-friendly AND
    machine-parseable.
  * `log_tool_call(...)` — a helper that emits a single `tool_call`
    event with the fields we care about (tool name, duration, ok/fail,
    conv_id, argument size, error class if any).
  * `@timed_tool` — an async decorator that wraps the top-level tool
    dispatcher. Every call produces exactly one structured log line on
    exit, even if the dispatcher raises.

Design choices
--------------
  * Stdlib `logging` only — no structlog / no new dep. FastAPI + uvicorn
    already configure the root logger; we just add a formatter.
  * JSON-per-line (newline-delimited JSON, "NDJSON"), because every log
    aggregator on earth can parse it and `grep duration_ms logs.txt |
    jq` is a one-liner.
  * Arguments are summarized, not dumped — LLMs sometimes pass 50 KB of
    text through `write_file`, and we don't want that in every log line.
    We record `arg_size_bytes` (for volume tracking) and truncate the
    repr to 200 chars (for quick eyeballing).
  * **Secrets caveat.** The truncated arg preview may include sensitive
    values if the user explicitly passes them through a tool (e.g. a
    `bash` command with an inline API key). The backend is a local
    single-user process so logs are never shipped anywhere by default,
    but if a future deployment adds log forwarding, add a redaction
    pass here — search for patterns like `--password=`, `api_key=`,
    `Authorization: Bearer`, etc. For now the invariant is: arg_preview
    is for local debugging only.
  * Timing uses `time.monotonic()` — wall-clock math is unsafe across
    NTP adjustments and monotonic is accurate to the microsecond on
    every OS we care about.

The logger name `gigachat.tools` is stable; downstream tooling can filter
on it.
"""
from __future__ import annotations

import functools
import json
import logging
import sys
import time
from typing import Any, Awaitable, Callable

# --- Log record shape ------------------------------------------------------

_MAX_ARG_REPR = 200  # chars
_MAX_ERROR_REPR = 300  # chars
_LOGGER_NAME_TOOLS = "gigachat.tools"
_LOGGER_NAME_APP = "gigachat.app"


class _JsonFormatter(logging.Formatter):
    """Render a LogRecord as one JSON object per line.

    We include the standard envelope (timestamp, level, logger, message)
    plus every extra attribute attached to the record. The formatter is
    permissive — any non-serializable extra gets stringified rather than
    raising, because a broken log statement must never crash the request
    it's logging.
    """

    # Attributes that come with every LogRecord and aren't user-supplied
    # extras. We skip these so the JSON stays compact.
    _STANDARD_ATTRS = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Surface every user-supplied field on the record (added via the
        # `extra=` kwarg on logger calls). This is where tool_call data
        # lands.
        for key, value in record.__dict__.items():
            if key in self._STANDARD_ATTRS or key.startswith("_"):
                continue
            try:
                json.dumps(value)  # probe serializability
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


# --- One-shot setup --------------------------------------------------------

_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """Install the JSON formatter on stderr exactly once per process.

    Idempotent: safe to call from app startup AND from test fixtures.
    We attach the handler to the root logger so library-emitted records
    (httpx, uvicorn's access log, etc.) get the same treatment and we
    don't end up with interleaved text-and-JSON.

    Called once from `app.py`'s startup event. Tests that want log
    assertions can call this themselves.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    # Only replace existing handlers when we're definitely the first
    # configurer — uvicorn sets up its own text formatter and the mix
    # would be ugly. Keeping `hasHandlers` check defensive for reloads.
    if root.hasHandlers():
        for h in list(root.handlers):
            root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str = _LOGGER_NAME_APP) -> logging.Logger:
    """Return a logger under the `gigachat.*` namespace.

    Callers can pass a dotted suffix (`tools.dispatch`, `agent.run_turn`)
    to get a child logger; if no name is given we default to the generic
    app logger. The JSON formatter is attached to the root, so any child
    logger benefits.
    """
    # Allow fully-qualified names to pass through so callers can do
    # `get_logger(__name__)` if they want their module path in the record.
    if name.startswith("gigachat."):
        return logging.getLogger(name)
    return logging.getLogger(f"gigachat.{name}")


# --- Tool-call helper ------------------------------------------------------

def _summarize_args(args: dict | None) -> tuple[int, str]:
    """Return (byte_size, short_repr) for a tool-call args dict.

    Models sometimes pass megabytes through `write_file`. We track size
    so the log has volumetric data but avoid pasting the raw content.
    """
    if not args:
        return 0, "{}"
    try:
        serialized = json.dumps(args, default=str, ensure_ascii=False)
    except Exception:
        serialized = repr(args)
    size = len(serialized.encode("utf-8", errors="replace"))
    if len(serialized) > _MAX_ARG_REPR:
        short = serialized[: _MAX_ARG_REPR - 3] + "..."
    else:
        short = serialized
    return size, short


def log_tool_call(
    *,
    tool: str,
    duration_ms: float,
    ok: bool,
    conv_id: str | None,
    args: dict | None,
    error: str | None = None,
    error_kind: str | None = None,
) -> None:
    """Emit one `tool_call` event at INFO (or WARNING if failed).

    Kept separate from the decorator so non-decorated paths (a future
    retry/fallback layer, for instance) can still produce comparable
    records.
    """
    arg_size, arg_preview = _summarize_args(args)
    log = get_logger(_LOGGER_NAME_TOOLS)
    level = logging.INFO if ok else logging.WARNING
    extra: dict[str, Any] = {
        "event": "tool_call",
        "tool": tool,
        "duration_ms": round(duration_ms, 1),
        "ok": bool(ok),
        "conv_id": conv_id,
        "arg_size_bytes": arg_size,
        "arg_preview": arg_preview,
    }
    if not ok:
        # Truncate error repr — some tool errors stringify to 5 KB tracebacks.
        if error and len(error) > _MAX_ERROR_REPR:
            error = error[: _MAX_ERROR_REPR - 3] + "..."
        extra["error"] = error
        extra["error_kind"] = error_kind
    log.log(level, f"tool_call {tool} ok={ok} dur={duration_ms:.0f}ms", extra=extra)


# --- Decorator -------------------------------------------------------------

# Type of the wrapped dispatch function. Kept loose so we don't pin the
# signature — callers can add kwargs without breaking this module.
_Dispatch = Callable[..., Awaitable[dict]]


def timed_tool(fn: _Dispatch) -> _Dispatch:
    """Wrap an async tool dispatcher to emit one `tool_call` event per call.

    Extracts `name`, `args`, and `conv_id` from the call (by position OR
    keyword, matching the real `dispatch` signature) so it stays a pure
    decorator — no changes required at the call site.

    Behavior:
      * On normal return, reads `result.get("ok")` + `result.get("error")`
        to classify the outcome. A missing `ok` key is treated as True
        (some tools return a plain dict without the flag; we shouldn't
        mark them failed just for that).
      * On exception, logs as ok=False with the exception class as
        `error_kind`, then re-raises. The caller still gets its exception;
        we just observe.
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> dict:
        # Positional signature of dispatch: (name, args, cwd, conv_id, model).
        # Support kwargs too so the decorator doesn't care how dispatch is
        # called.
        name = args[0] if args else kwargs.get("name", "?")
        tool_args = (
            args[1] if len(args) > 1 else kwargs.get("args") or {}
        )
        conv_id = (
            args[3] if len(args) > 3 else kwargs.get("conv_id")
        )
        t0 = time.monotonic()
        try:
            result = await fn(*args, **kwargs)
        except Exception as exc:
            # Log, then propagate. The caller controls the user-visible
            # error formatting; our job is to keep the record.
            duration_ms = (time.monotonic() - t0) * 1000.0
            log_tool_call(
                tool=str(name),
                duration_ms=duration_ms,
                ok=False,
                conv_id=conv_id,
                args=tool_args if isinstance(tool_args, dict) else None,
                error=f"{type(exc).__name__}: {exc}",
                error_kind=type(exc).__name__,
            )
            raise
        duration_ms = (time.monotonic() - t0) * 1000.0
        # `ok` missing is treated as success; explicit False is a failure.
        ok_flag = True if not isinstance(result, dict) else result.get("ok", True)
        err_msg = None
        err_kind = None
        if isinstance(result, dict) and ok_flag is False:
            err_msg = result.get("error") or ""
            # Pluck a coarse classification from the error prefix so
            # downstream metric aggregation can group failures without
            # re-tokenizing. Example: "unknown tool: 'foo'" → unknown_tool.
            low = err_msg.lower()
            if "unknown tool" in low:
                err_kind = "unknown_tool"
            elif "timed out" in low or "timeout" in low:
                err_kind = "timeout"
            elif "not minted" in low or "evicted" in low or "bad format" in low:
                err_kind = "element_cache_miss"
            elif "permission_denied" in low or "permission denied" in low:
                err_kind = "permission_denied"
            elif low:
                err_kind = "other"
        log_tool_call(
            tool=str(name),
            duration_ms=duration_ms,
            ok=bool(ok_flag),
            conv_id=conv_id,
            args=tool_args if isinstance(tool_args, dict) else None,
            error=err_msg,
            error_kind=err_kind,
        )
        return result

    return wrapper
