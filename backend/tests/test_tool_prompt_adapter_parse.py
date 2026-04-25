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


def test_parser_repairs_missing_outer_brace():
    """Real failure mode from a gemma4:e4b conversation: the model
    streamed a long `edit_file` payload and dropped the final `}`. The
    body is missing one closing brace; everything else parses. Auto-
    repair should accept it so the call reaches dispatch instead of
    vanishing as a "Trying a call: <unparseable>" assistant message."""
    # Body is missing the outer `}` — args closes but the wrapping
    # object never does. Ends with `"}` (the new_string close + args
    # close), no final `}` on the outer object.
    text = (
        "I will now edit App.jsx.\n"
        "<tool_call>\n"
        '{"name": "edit_file", "args": {"path": "src/App.jsx", '
        '"old_string": "old", "new_string": "new"}\n'
        "</tool_call>\n"
    )
    cleaned, calls = tool_prompt_adapter.parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "edit_file"
    assert calls[0]["args"]["path"] == "src/App.jsx"
    assert calls[0]["args"]["new_string"] == "new"
    # The repaired block was still stripped from the user-visible prose.
    assert "<tool_call>" not in cleaned.lower()
    assert "I will now edit App.jsx" in cleaned


def test_parser_repairs_missing_both_braces():
    """Worst-case: model dropped BOTH the args close and the outer
    close. Two `}` short. Repair tries up to two trailing closers."""
    text = (
        "<tool_call>\n"
        '{"name": "write_file", "args": {"path": "a.txt", "content": "hi"\n'
        "</tool_call>\n"
    )
    cleaned, calls = tool_prompt_adapter.parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "write_file"
    assert calls[0]["args"]["content"] == "hi"


def test_parser_unwraps_double_wrapped_call():
    """Real failure mode from gemma4:e4b: model emitted

        {"name": "", "args": {"name": "read_file",
                              "args": {"path": "tailwind.config.js"}}}

    instead of just `{"name": "read_file", "args": {...}}`. The outer
    object's `name` is empty so dispatch errors with `unknown tool: ''`
    and the inner call vanishes. Unwrap one level when the outer name
    is empty AND the inner args dict looks like a tool call."""
    text = (
        "<tool_call>\n"
        '{"name": "", "args": {"name": "read_file", '
        '"args": {"path": "tailwind.config.js"}}}\n'
        "</tool_call>\n"
    )
    _cleaned, calls = tool_prompt_adapter.parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "read_file"
    assert calls[0]["args"]["path"] == "tailwind.config.js"


def test_parser_does_not_unwrap_genuine_args_with_name_field():
    """Don't be too eager unwrapping: if the OUTER name is filled
    AND the inner args happens to have a `name` field (e.g. a real
    tool call where `args.name` is meaningful), leave it alone."""
    text = (
        "<tool_call>\n"
        '{"name": "click_element", "args": {"name": "Submit"}}\n'
        "</tool_call>\n"
    )
    _cleaned, calls = tool_prompt_adapter.parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "click_element"
    # `args.name` survived — no double-unwrap.
    assert calls[0]["args"] == {"name": "Submit"}


def test_parser_unwraps_args_inside_args():
    """Real failure mode from the latest test conversation: model
    emitted

        {"name": "bash", "args": {"args": {"command": "cd ...",
                                           "reason": "..."}}}

    Outer name is filled (`bash`) but the actual fields are nested
    one level too deep under another `args` key. Dispatch sees
    `args.command = undefined` and returns "empty command", which
    cascades into npm running in the wrong directory and creating
    a stray package-lock.json at the workspace root. Unwrap when
    the inner `args` dict has a single arg-aliased key."""
    text = (
        "<tool_call>\n"
        '{"name": "bash", "args": {"args": {"command": "cd stock-dashboard", '
        '"reason": "Change into the project dir."}}}\n'
        "</tool_call>\n"
    )
    _cleaned, calls = tool_prompt_adapter.parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "bash"
    assert calls[0]["args"]["command"] == "cd stock-dashboard"
    assert "args" not in calls[0]["args"], "should have unwrapped one level"


def test_parser_unwraps_args_inside_arguments_alias():
    """Same shape but the inner key is `arguments` (the OpenAI-style
    alias). Same unwrap should apply."""
    text = (
        '<tool_call>{"name": "write_file", "args": {"arguments": '
        '{"path": "x.txt", "content": "hi"}}}</tool_call>'
    )
    _cleaned, calls = tool_prompt_adapter.parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "write_file"
    assert calls[0]["args"]["path"] == "x.txt"
    assert calls[0]["args"]["content"] == "hi"


def test_parser_does_not_unwrap_when_inner_args_is_two_keys():
    """The unwrap is gated on EXACTLY one key inside the wrapper —
    so a legitimate call where the user-tool happens to define an
    `args` parameter alongside other params stays intact."""
    text = (
        '<tool_call>{"name": "user_tool", "args": '
        '{"args": "literal-string", "other": "val"}}</tool_call>'
    )
    _cleaned, calls = tool_prompt_adapter.parse_tool_calls_from_text(text)
    assert len(calls) == 1
    # Both keys preserved — no false unwrap.
    assert calls[0]["args"] == {"args": "literal-string", "other": "val"}
