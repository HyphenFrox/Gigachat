"""Tests for the structured-logging + per-tool timing module.

We check three things:
  * The JSON formatter emits one parseable object per record, with the
    user-supplied `extra` fields surfaced alongside the standard envelope.
  * `log_tool_call` writes at INFO on success and WARNING on failure.
  * `@timed_tool` emits exactly one log record per dispatch, correctly
    classifying ok/fail and computing a non-zero duration for slow calls.

All of this is pure-Python — no I/O, no network — so the whole module
runs in the smoke tier.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import time

import pytest

from backend import telemetry

pytestmark = pytest.mark.smoke


# --- JSON formatter --------------------------------------------------------


def _capture_log(level: int = logging.INFO) -> tuple[logging.Handler, io.StringIO]:
    """Return a handler+buffer attached to the gigachat root logger.

    Yields a StreamHandler writing into a StringIO so tests can read
    exactly what the formatter produced. The caller is responsible for
    removing the handler when done.
    """
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(telemetry._JsonFormatter())
    handler.setLevel(level)
    logger = logging.getLogger("gigachat")
    logger.addHandler(handler)
    logger.setLevel(level)
    return handler, buf


def test_json_formatter_emits_parseable_object():
    """Each record becomes one NDJSON line with the expected envelope."""
    handler, buf = _capture_log()
    try:
        log = telemetry.get_logger("test")
        log.info("hello", extra={"foo": 1, "bar": "baz"})
        line = buf.getvalue().strip()
        obj = json.loads(line)
        assert obj["level"] == "INFO"
        assert obj["logger"] == "gigachat.test"
        assert obj["message"] == "hello"
        # User-supplied extras are surfaced at the top level.
        assert obj["foo"] == 1
        assert obj["bar"] == "baz"
        # Timestamp is present and looks ISO-ish.
        assert "T" in obj["ts"]
    finally:
        logging.getLogger("gigachat").removeHandler(handler)


def test_json_formatter_stringifies_non_serializable_extras():
    """A non-JSONable extra must not crash the log call — repr() fallback."""
    handler, buf = _capture_log()
    try:
        log = telemetry.get_logger("test")

        class _Opaque:
            def __repr__(self) -> str:
                return "<opaque>"

        log.info("check", extra={"weird": _Opaque()})
        obj = json.loads(buf.getvalue().strip())
        assert obj["weird"] == "<opaque>"
    finally:
        logging.getLogger("gigachat").removeHandler(handler)


# --- log_tool_call ---------------------------------------------------------


def test_log_tool_call_success_is_info():
    """A successful tool call is logged at INFO with ok=True."""
    handler, buf = _capture_log()
    try:
        telemetry.log_tool_call(
            tool="list_dir",
            duration_ms=12.3,
            ok=True,
            conv_id="c1",
            args={"path": "."},
        )
        obj = json.loads(buf.getvalue().strip())
        assert obj["level"] == "INFO"
        assert obj["event"] == "tool_call"
        assert obj["tool"] == "list_dir"
        assert obj["ok"] is True
        assert obj["duration_ms"] == 12.3
        assert obj["conv_id"] == "c1"
        # Args summarized — preview round-trips and size is bytes.
        assert '"path"' in obj["arg_preview"]
        assert obj["arg_size_bytes"] > 0
        # No error fields on success.
        assert "error" not in obj
    finally:
        logging.getLogger("gigachat").removeHandler(handler)


def test_log_tool_call_failure_is_warning_with_kind():
    """A failed call logs WARNING and truncates oversize error text."""
    handler, buf = _capture_log(logging.WARNING)
    try:
        # Very long error — the formatter must truncate to _MAX_ERROR_REPR.
        long_err = "boom " * 500
        telemetry.log_tool_call(
            tool="thing",
            duration_ms=5.0,
            ok=False,
            conv_id=None,
            args=None,
            error=long_err,
            error_kind="timeout",
        )
        obj = json.loads(buf.getvalue().strip())
        assert obj["level"] == "WARNING"
        assert obj["ok"] is False
        assert obj["error_kind"] == "timeout"
        # Truncated AND marked with the ellipsis sentinel.
        assert obj["error"].endswith("...")
        assert len(obj["error"]) <= telemetry._MAX_ERROR_REPR
    finally:
        logging.getLogger("gigachat").removeHandler(handler)


def test_log_tool_call_truncates_arg_preview():
    """Args serialized above the preview cap are truncated, but size reflects
    the untruncated byte length."""
    handler, buf = _capture_log()
    try:
        big = {"blob": "x" * 5000}
        telemetry.log_tool_call(
            tool="write_file",
            duration_ms=1.0,
            ok=True,
            conv_id=None,
            args=big,
        )
        obj = json.loads(buf.getvalue().strip())
        assert obj["arg_size_bytes"] > 4000  # real size preserved
        assert len(obj["arg_preview"]) <= telemetry._MAX_ARG_REPR
        assert obj["arg_preview"].endswith("...")
    finally:
        logging.getLogger("gigachat").removeHandler(handler)


# --- @timed_tool decorator -------------------------------------------------


def test_timed_tool_logs_once_on_success():
    """One record per call with ok=True and a measurable duration."""
    handler, buf = _capture_log()
    try:
        @telemetry.timed_tool
        async def fake_dispatch(name, args, cwd, conv_id=None, model=None):
            # Sleep enough that duration_ms is clearly non-zero but
            # stays fast so the smoke suite isn't slow.
            await asyncio.sleep(0.02)
            return {"ok": True, "output": "ok"}

        asyncio.run(fake_dispatch("t", {"k": "v"}, cwd=".", conv_id="c"))
        lines = [l for l in buf.getvalue().splitlines() if l.strip()]
        assert len(lines) == 1, f"expected 1 log line, got {len(lines)}"
        obj = json.loads(lines[0])
        assert obj["tool"] == "t"
        assert obj["ok"] is True
        assert obj["conv_id"] == "c"
        assert obj["duration_ms"] >= 15  # slept at least 20 ms
    finally:
        logging.getLogger("gigachat").removeHandler(handler)


def test_timed_tool_classifies_error_kinds():
    """The error-kind heuristic picks recognizable buckets from error text."""
    handler, buf = _capture_log(logging.WARNING)
    try:
        @telemetry.timed_tool
        async def fake(name, args, cwd, conv_id=None, model=None):
            return {"ok": False, "error": "unknown tool: 'foo'"}

        asyncio.run(fake("foo", {}, cwd="."))
        obj = json.loads(buf.getvalue().strip())
        assert obj["error_kind"] == "unknown_tool"
    finally:
        logging.getLogger("gigachat").removeHandler(handler)


def test_timed_tool_reraises_but_still_logs():
    """If the wrapped function raises, we log ok=False and re-raise."""
    handler, buf = _capture_log(logging.WARNING)
    try:
        @telemetry.timed_tool
        async def explodes(name, args, cwd, conv_id=None, model=None):
            raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError, match="kaboom"):
            asyncio.run(explodes("x", {}, cwd="."))
        obj = json.loads(buf.getvalue().strip())
        assert obj["ok"] is False
        assert obj["error_kind"] == "RuntimeError"
        assert "kaboom" in obj["error"]
    finally:
        logging.getLogger("gigachat").removeHandler(handler)
