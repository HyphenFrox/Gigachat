"""JSON encode/decode helpers backed by orjson when available.

orjson is a Rust-based JSON library typically 3-5× faster than the
stdlib `json` module on both parse and dump paths. The chat backend
hits JSON on every SSE chunk (Ollama / llama-server stream parsing),
every DB row hydration, every tool-call serialization. The cumulative
saving across a busy session is real but only gets realized when
every hot path uses the fast helpers — hence centralizing here so
the dependency stays optional and the call sites stay terse.

Falls back to stdlib `json` when orjson isn't installed. Behaviour
is identical for the payloads we serialize (UTF-8 output,
JSON-spec-compliant parse) — orjson's `JSONDecodeError` inherits
from `json.JSONDecodeError`, so existing `except json.JSONDecodeError`
clauses catch orjson errors too.

Public surface:
    `loads(text_or_bytes)` — parse JSON to Python object.
    `dumps(obj)`           — serialize to UTF-8 str (no trailing whitespace).
    `HAVE_ORJSON`          — True when orjson successfully imported.
"""
from __future__ import annotations

import json as _stdlib_json
from typing import Any

try:
    import orjson as _orjson  # type: ignore
    HAVE_ORJSON = True
except ImportError:
    _orjson = None  # type: ignore
    HAVE_ORJSON = False


if HAVE_ORJSON:
    def loads(data: Any) -> Any:
        """Parse JSON. Accepts str, bytes, or bytearray."""
        return _orjson.loads(data)

    def dumps(obj: Any) -> str:
        """Serialize to UTF-8 str. orjson always emits compact UTF-8;
        we decode the bytes to str for callers that want the str form
        (the most common case for SQLite TEXT columns and HTTP body
        strings). The decode is sub-microsecond — net is still 3-5×
        faster than `json.dumps`.
        """
        return _orjson.dumps(obj).decode("utf-8")
else:
    def loads(data: Any) -> Any:
        # stdlib json.loads accepts str only; convert bytes for parity
        # with orjson's broader input acceptance.
        if isinstance(data, (bytes, bytearray, memoryview)):
            data = bytes(data).decode("utf-8")
        return _stdlib_json.loads(data)

    def dumps(obj: Any) -> str:
        # ensure_ascii=False matches orjson's UTF-8 output — saves
        # bytes on non-ASCII content and avoids \uXXXX escape bloat.
        # separators=(",", ":") matches orjson's compact format so
        # call sites that put the output into a size-constrained
        # context (HTTP body, prompt embedding) get the same byte
        # count regardless of which backend is active.
        return _stdlib_json.dumps(
            obj, ensure_ascii=False, separators=(",", ":"),
        )
