"""Regression: `parse_tool_calls_from_text` must recover tool calls from
text-format payloads even on models that *also* support native tool calls.

Background: gemma4:e4b (and a couple of small Qwens) advertise tool-calling
support in Ollama, so the agent runs them with `adapter_mode=False` and
expects calls to come back via the structured `tool_calls` field. In
practice these models will sometimes announce the call in prose and then
emit JSON wrapped in `<tool_call>...</tool_call>` tags, leaving the
structured channel empty. Without this parser kicking in as a fallback the
user sees the announcement and then silence — "it keeps emitting tools but
doesn't actually run them."

These tests pin the parser's behaviour on payloads shaped like the ones
that triggered that report, so a future tightening of the regex can't
silently re-introduce the dead-tool-call symptom.
"""
from __future__ import annotations

import pytest

from backend import tool_prompt_adapter

pytestmark = pytest.mark.smoke


GEMMA_LIKE = """\
I will now update `src/App.jsx` to implement the stock tracker functionality.

<tool_call>
{"name": "edit_file", "args": {"path": "src/App.jsx", "old_string": "old", "new_string": "new", "reason": "swap boilerplate for tracker"}}
</tool_call>
"""


def test_parser_recovers_gemma_style_tool_call():
    cleaned, calls = tool_prompt_adapter.parse_tool_calls_from_text(GEMMA_LIKE)
    assert len(calls) == 1
    c = calls[0]
    assert c["name"] == "edit_file"
    assert c["args"]["path"] == "src/App.jsx"
    assert c["args"]["new_string"] == "new"
    # The user-visible prose must survive; only the tag block is stripped.
    assert "I will now update" in cleaned
    assert "<tool_call>" not in cleaned.lower()


def test_parser_no_op_on_plain_prose():
    """Calling the parser on a normal assistant reply must be a no-op so
    the post-parse fallback doesn't penalise well-behaved models."""
    text = "Sure, here's a summary of what I did. The tests are green."
    cleaned, calls = tool_prompt_adapter.parse_tool_calls_from_text(text)
    assert calls == []
    assert cleaned == text


def test_parser_extracts_multiple_blocks_in_order():
    text = (
        "First I'll read the file:\n"
        "<tool_call>\n"
        '{"name": "read_file", "args": {"path": "a.txt"}}\n'
        "</tool_call>\n"
        "Then I'll write a new one:\n"
        "<tool_call>\n"
        '{"name": "write_file", "args": {"path": "b.txt", "content": "hi"}}\n'
        "</tool_call>\n"
    )
    cleaned, calls = tool_prompt_adapter.parse_tool_calls_from_text(text)
    assert [c["name"] for c in calls] == ["read_file", "write_file"]
    assert calls[1]["args"]["content"] == "hi"
    # Both blocks stripped; both prose lines survive.
    assert "<tool_call>" not in cleaned.lower()
    assert "First I'll read" in cleaned
    assert "Then I'll write" in cleaned


def test_parser_leaves_malformed_block_in_place():
    """A block with broken JSON should NOT be silently dropped — the model
    needs to see its own mistake echoed on the next turn so it can retry."""
    text = (
        "Trying a call:\n"
        "<tool_call>\n"
        "{not valid json}\n"
        "</tool_call>\n"
    )
    cleaned, calls = tool_prompt_adapter.parse_tool_calls_from_text(text)
    assert calls == []
    assert "<tool_call>" in cleaned.lower()
    assert "{not valid json}" in cleaned
