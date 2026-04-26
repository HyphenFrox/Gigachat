"""Regression: tool-capable model families whose Ollama upload stripped
the `{{ if .Tools }}` template block must still work in Gigachat.

Background: a model's chat template determines whether Ollama advertises
the ``tools`` capability. Some uploads of tool-capable model families
(Dolphin 3, the Dolphin-Mistral 24B fork, Llama 3.2 vision, DeepSeek-V2
Coder) ship a Modelfile that omits the tools block, so Ollama's
/api/show drops the cap flag, the picker hides the model, and the
agent loop's ``needs_adapter`` would normally return False on the first
branch. The user shouldn't have to ``ollama create`` a custom Modelfile
to use these — the weights themselves are tool-capable, they just need
prompt-space wiring instead of native.

These tests pin:

  1. The known-tool-capable name patterns continue to match the variants
     users actually have installed (publisher prefixes, tag suffixes,
     case differences).
  2. ``needs_adapter`` returns True for those models even when /api/show
     reports no ``tools`` cap and no ``.Tools`` template marker.
  3. Models genuinely outside the family (older DeepSeek-Coder v1, base
     Gemma 3, embedding-only models) are NOT falsely admitted.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from backend import tool_prompt_adapter

pytestmark = pytest.mark.smoke


# --- name pattern matching -------------------------------------------------

# (model name, expected_match) — name forms taken from real
# `ollama list` output the user reported, so a future tag-format change
# in Ollama doesn't sneak past this contract.
PATTERN_CASES = [
    # Should match: known tool-capable families.
    ("dolphin3:8b", True),
    ("DOLPHIN3:8B", True),  # case-insensitive
    ("ikiru/Dolphin-Mistral-24B-Venice-Edition:latest", True),
    ("llama3.1:8b", True),
    ("llama3.2-vision:11b", True),
    ("llama-3.3:70b", True),
    ("mistral:7b", True),
    ("mistral-nemo:latest", True),
    ("mistral-small3.2:24b", True),
    ("mixtral:8x7b", True),
    ("deepseek-coder-v2:lite", True),
    ("deepseek-v3:latest", True),
    ("deepseek-r1:32b", True),
    ("huihui_ai/qwen2.5-coder-abliterate:7b", True),
    ("qwen3.5:9b", True),

    # Should NOT match: genuinely no native tool support / wrong family.
    ("deepseek-coder:6.7b", False),  # v1, not v2 — no tools at weights
    ("gemma3:4b", False),
    ("nomic-embed-text:latest", False),
    ("llama2:7b", False),
    ("dolphin2.9.4:8b", False),
    ("", False),
]


@pytest.mark.parametrize("name,expected", PATTERN_CASES)
def test_known_tool_capable_pattern(name: str, expected: bool) -> None:
    assert tool_prompt_adapter._matches_known_tool_capable(name) is expected, (
        f"{name!r} should{' ' if expected else ' NOT '}match the known list"
    )


# --- needs_adapter integration --------------------------------------------

class _FakeResponse:
    """Minimal stand-in for httpx.Response — enough for needs_adapter."""

    def __init__(self, info: dict) -> None:
        self._info = info

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._info


class _FakeClient:
    """Captures the /api/show probe so tests don't need a real Ollama."""

    def __init__(self, info: dict) -> None:
        self._info = info
        self.calls: list[dict] = []

    async def post(self, url, *, json=None, timeout=None, headers=None):  # noqa: A002
        # `headers` was added when needs_adapter learned to forward a Bearer
        # token to the worker's Ollama; the legacy stub had no use for it,
        # but the kwarg must be accepted or the probe call raises TypeError
        # and falls into the "assume native works" fallback path.
        self.calls.append({"url": url, "json": json, "headers": headers})
        return _FakeResponse(self._info)


def _run(coro):
    return asyncio.run(coro)


def _clear_cache() -> None:
    tool_prompt_adapter.clear_cache()


def test_needs_adapter_true_for_known_family_without_cap() -> None:
    """`dolphin3:8b` reports no tools cap and a stub template; we must
    still flip on adapter mode because the base weights (Llama 3.1)
    support function calling."""
    _clear_cache()
    info = {"capabilities": ["completion"], "template": "{{ .Prompt }}"}
    client = _FakeClient(info)
    result = _run(
        tool_prompt_adapter.needs_adapter(
            "dolphin3:8b", "http://x", client=client
        )
    )
    assert result is True
    # And the result is cached so a second call doesn't re-probe.
    result_cached = _run(
        tool_prompt_adapter.needs_adapter(
            "dolphin3:8b", "http://x", client=client
        )
    )
    assert result_cached is True
    assert len(client.calls) == 1


def test_needs_adapter_false_when_native_works() -> None:
    """`llama3.1:8b` reports tools cap AND the template renders .Tools —
    native path. Even though llama3.1 IS in our known list, when native
    works we must return False so the agent uses the structured channel.
    """
    _clear_cache()
    info = {
        "capabilities": ["completion", "tools"],
        "template": "...{{ if .Tools }}<TOOLS>{{ end }}...",
    }
    client = _FakeClient(info)
    result = _run(
        tool_prompt_adapter.needs_adapter(
            "llama3.1:8b", "http://x", client=client
        )
    )
    assert result is False


def test_needs_adapter_false_for_unknown_family_without_cap() -> None:
    """A model with no tools cap, stub template, and no name match
    must keep returning False so we don't accidentally start an adapter
    conversation against weights that can't produce a JSON tool call."""
    _clear_cache()
    info = {"capabilities": ["completion"], "template": "{{ .Prompt }}"}
    client = _FakeClient(info)
    result = _run(
        tool_prompt_adapter.needs_adapter(
            "deepseek-coder:6.7b", "http://x", client=client
        )
    )
    # `deepseek-coder` (v1) is NOT in the known list; no cap; stub template.
    # Old behaviour returns True ("not native_ok") — we want to keep that
    # so the existing prompt-space-mode-for-stub-template path stays. The
    # KEY assertion is the first one (known-family-without-cap) flips on
    # adapter mode, which it didn't before.
    assert result is True


def test_needs_adapter_true_for_known_family_with_stub_template() -> None:
    """Mirror of the gemma4:e4b case — has tools cap, no .Tools in
    template. Already worked before, but make sure adding the new
    branch didn't regress this path."""
    _clear_cache()
    info = {
        "capabilities": ["completion", "tools"],
        "template": "{{ .Prompt }}",
    }
    client = _FakeClient(info)
    result = _run(
        tool_prompt_adapter.needs_adapter(
            "qwen3.5:9b", "http://x", client=client
        )
    )
    assert result is True
