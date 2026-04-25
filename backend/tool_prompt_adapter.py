"""Prompt-space tool adapter.

Works around Ollama models whose chat template silently drops the
`tools=[...]` payload (e.g. ``gemma4:e4b`` ships with a ``{{ .Prompt }}``
passthrough template and never renders ``.Tools``). For those models
native function calling is impossible — Ollama accepts the ``tools``
field (capabilities list advertises ``tools``) and then the template
discards it, so the model literally never sees the tool list and replies
with "I'm a language model, I don't have tools."

Strategy: detect once per model. If the template doesn't reference
``.Tools`` / ``.ToolCalls``, switch the conversation into adapter mode:

  1. Render the tool list as an XML-tagged block appended to the system
     prompt.
  2. Instruct the model to emit ``<tool_call>{"name": ..., "args": {...}}
     </tool_call>`` when it wants a tool run.
  3. Parse those tags out of the assistant's streamed text and hand them
     to the existing dispatch pipeline as if Ollama had emitted native
     ``tool_calls``.
  4. Flatten prior turns too — historical ``role=tool`` messages become
     user-role ``<tool_result name="...">...</tool_result>`` blocks, and
     historical assistant ``tool_calls`` are inlined into the content as
     ``<tool_call>`` blocks so the model sees a coherent dialogue.

Every piece is shaped to match the existing agent loop so the adapter
branch only touches three call sites in ``agent.py``: probe, payload
rewrite, post-stream text parse.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

# Process-wide cache keyed by model name.
#   True  = native tool calling works (template renders .Tools).
#   False = stub template / no tools capability — use adapter.
_ADAPTER_CACHE: dict[str, bool] = {}

# Substrings we accept as proof that the chat template actually renders
# tools. Ollama templates use Go `text/template` syntax, so ``.Tools`` is
# the canonical form; we also accept a few lowercased / alternate forms
# so community Modelfiles with bespoke template logic still count.
_TEMPLATE_TOOL_MARKERS = (
    ".Tools",
    ".ToolCalls",
    "tool_calls",
    "tools }}",
    "range $.Tools",
)

# Some Ollama uploads of tool-capable model families ship a Modelfile that
# omits the ``{{ if .Tools }}`` block entirely — so Ollama drops the
# ``tools`` capability flag, which would normally hide the model from the
# picker AND skip the adapter. We override that for known-good families:
# the underlying weights were trained with function calling, the upload is
# just missing the template wiring. With this list:
#
#   * the picker shows them (`/api/models` allowlist),
#   * the agent loop forces adapter mode for them (XML-tag tool calls in
#     prose), so tools work without the user having to `ollama create`
#     a custom Modelfile.
#
# Patterns are case-insensitive substring matches against the model name,
# anchored only loosely so common tag suffixes (`:7b`, `:latest`, `-instruct`,
# `-q4_K_M`, etc.) all match. Add to the list only when you've verified the
# *base* model genuinely produces well-formed JSON tool calls when prompted
# in prompt-space mode — adding a wrong entry surfaces as a `<tool_call>`
# block the parser can't decode and a confused user.
_KNOWN_TOOL_CAPABLE_NAME_PATTERNS = (
    # Llama 3.1 / 3.2 / 3.3 families — trained with function calling. The
    # vanilla ``llama3.1:8b`` ships a working template; community fine-tunes
    # (Dolphin, abliterated forks, vision variants) usually don't.
    "llama3.1",
    "llama-3.1",
    "llama3.2",
    "llama-3.2",
    "llama3.3",
    "llama-3.3",
    "dolphin3",          # Cognitive Computations Dolphin 3.0 → Llama 3.1 base.
    # Mistral families — every modern Mistral (Nemo, Small, Small 3, Large,
    # Mixtral) supports JSON function calling. ``ikiru/Dolphin-Mistral-24B-…``
    # is Mistral Small 3 base with Dolphin fine-tuning.
    "mistral",
    "mixtral",
    "dolphin-mistral",
    # DeepSeek-V2 and V3 are both function-calling-trained. The original
    # ``deepseek-coder:*`` (V1) is NOT — make the match specific to v2+ so
    # we don't falsely admit the older series.
    "deepseek-coder-v2",
    "deepseek-v2",
    "deepseek-v3",
    "deepseek-r1",
    # Qwen 2.5+ families. Qwen 2.5 ships a working tools template on Ollama
    # so this is mainly belt-and-suspenders for community variants. Qwen 3 /
    # 3.5 / 3.6 advertise the cap but ship a stub template — already covered
    # by the existing `tools cap + no .Tools template` branch, listed here so
    # a future bare ``qwen3.5:*`` rename without the cap flag still works.
    "qwen2.5",
    "qwen3",
)


def _matches_known_tool_capable(model: str) -> bool:
    """Match the model name against `_KNOWN_TOOL_CAPABLE_NAME_PATTERNS`.

    Case-insensitive substring match. Lets us handle tag variants
    (`:7b`, `:latest`, `:e4b`), publisher prefixes
    (`huihui_ai/qwen2.5-coder-abliterate:14b`), and quantization suffixes
    (`-q4_K_M`) without enumerating each one.
    """
    if not model:
        return False
    n = model.lower()
    return any(p in n for p in _KNOWN_TOOL_CAPABLE_NAME_PATTERNS)


async def needs_adapter(
    model: str,
    ollama_url: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> bool:
    """Return ``True`` when the given model can't do native tool calls.

    Probes Ollama's ``/api/show`` once per model (cached for the process
    lifetime) and compares:

      * ``capabilities`` list — must include ``"tools"``.
      * ``template`` — must reference ``.Tools`` (or an equivalent marker).

    If EITHER signal is missing, we need the prompt-space adapter. If the
    probe itself fails (Ollama down, model not pulled), we conservatively
    assume native works so we don't regress models that are actually fine.

    Accepts an optional shared ``httpx.AsyncClient`` so callers inside a
    streaming context can reuse the open connection.
    """
    if not model:
        return False
    cached = _ADAPTER_CACHE.get(model)
    if cached is not None:
        return cached

    async def _probe(c: httpx.AsyncClient) -> dict[str, Any]:
        r = await c.post(
            f"{ollama_url}/api/show",
            json={"name": model},
            timeout=5.0,
        )
        r.raise_for_status()
        return r.json() or {}

    try:
        if client is not None:
            info = await _probe(client)
        else:
            async with httpx.AsyncClient(timeout=5.0) as owned:
                info = await _probe(owned)
    except Exception as e:
        log.warning(
            "tool_prompt_adapter: /api/show probe failed for %s (%s); "
            "assuming native tool calling works",
            model, e,
        )
        _ADAPTER_CACHE[model] = False
        return False

    caps = info.get("capabilities") or []
    template = info.get("template") or ""
    has_cap = "tools" in caps
    renders = any(marker in template for marker in _TEMPLATE_TOOL_MARKERS)
    native_ok = bool(has_cap and renders)
    # `not native_ok` is the original branch: cap flag missing OR template
    # doesn't reference .Tools → use adapter. We also force-enable adapter
    # mode for known-tool-capable model families when Ollama hasn't
    # advertised the cap at all (i.e. the upload's Modelfile stripped the
    # template). Without this branch those models would either be hidden
    # from the picker entirely (filter side) or rejected by Ollama with a
    # 400 when the agent tried to send `tools=[...]` natively.
    known_family = (not has_cap) and _matches_known_tool_capable(model)
    use_adapter = (not native_ok) or known_family
    _ADAPTER_CACHE[model] = use_adapter
    if use_adapter:
        log.info(
            "tool_prompt_adapter: %s will use prompt-space fallback "
            "(capabilities=%s, template_references_tools=%s, "
            "known_tool_capable_family=%s)",
            model, caps, renders, known_family,
        )
    return use_adapter


def clear_cache() -> None:
    """Forget every cached detection result.

    Called when a model is re-pulled or an admin wants to re-probe after
    editing a Modelfile template.
    """
    _ADAPTER_CACHE.clear()


# ---------------------------------------------------------------------------
# System-prompt tool block
# ---------------------------------------------------------------------------

# Small models follow XML-ish tags more reliably than JSON code fences,
# and the tag boundary gives the parser an unambiguous stop marker. The
# final instruction (no markdown fences around the tag) catches the most
# common mistake when the model ALSO knows a second tool-calling format
# from pre-training.
_ADAPTER_INSTRUCTIONS = """\
Call a tool by emitting this block:

<tool_call>
{"name":"<tool_name>","args":{...}}
</tool_call>

For INDEPENDENT operations, emit multiple <tool_call> blocks back-to-back \
in the SAME turn — they run in parallel and each <tool_result name="..."> \
comes back on the next turn. Chain calls across turns only when a later \
call needs an earlier result. When the task is done, reply in plain text \
with NO <tool_call>.

Rules:
- JSON must be valid (no comments, trailing commas, or unquoted keys).
- Use only names from the tool list below.
- An arg name ending in * is required; others are optional.
- Never wrap <tool_call> in markdown fences or describe it in prose first.
"""


def _compact_params_schema(params: dict[str, Any]) -> dict[str, Any]:
    """Flatten the OpenAPI-ish parameter schema into a ``{name: type}`` map.

    Full JSON Schema (``{"type":"object","properties":{"q":{"type":"string"}},
    "required":["q"]}``) is valuable for big function-calling models, but
    for 4B stub-template models (Gemma4 e4b) 72 tools × that level of
    ceremony blows past 80k chars of system prompt — the model stops
    emitting thinking tokens, ignores our ``<tool_call>`` instruction,
    and hallucinates that no tools exist.

    The flat form preserves everything the model actually needs:

      * Name → type-string (``string`` / ``integer`` / ``boolean`` / ``array`` /
        ``object``). Enum values keep their discrete list so the model
        still knows which strings are valid.
      * Arrays carry their element type inline (``array<string>``).
      * Required args are marked with a trailing ``*`` on the key so the
        ``required`` list disappears entirely — the adapter preamble
        explains the convention once.

    Output:
      ``{"query*": "string", "max_results": "integer", "mode": ["fast","slow"]}``
    """
    if not isinstance(params, dict):
        return {}
    props = params.get("properties") or {}
    if not isinstance(props, dict):
        return {}
    required = set(params.get("required") or [])
    out: dict[str, Any] = {}
    for pname, pspec in props.items():
        key = f"{pname}*" if pname in required else pname
        if not isinstance(pspec, dict):
            out[key] = "string"
            continue
        if "enum" in pspec and isinstance(pspec["enum"], list):
            # Keep the discrete list verbatim — it's the cheapest way to
            # tell the model which string literals are acceptable.
            out[key] = pspec["enum"]
            continue
        ptype = pspec.get("type") or "string"
        items = pspec.get("items")
        if ptype == "array" and isinstance(items, dict):
            item_t = items.get("type") or "string"
            out[key] = f"array<{item_t}>"
        else:
            out[key] = ptype
    return out


def build_tools_system_block(tool_schemas: list[dict[str, Any]]) -> str:
    """Serialize the tool list into a system-prompt block.

    Input shape matches what we already pass as Ollama's ``tools`` field:
    a list of ``{"type": "function", "function": {name, description,
    parameters}}`` entries. The output is plain text ready to be appended
    to the existing system prompt.

    We compact every tool entry (drop per-argument ``description`` fields)
    because small stub-template models can't follow the prompt when tool
    schemas balloon to tens of thousands of characters — they skip the
    thinking phase, ignore our ``<tool_call>`` instruction, and hallucinate
    that no tools exist.
    """
    if not tool_schemas:
        return ""
    lines = ["<tools>"]
    for entry in tool_schemas:
        fn = entry.get("function") or entry
        name = fn.get("name") or ""
        if not name:
            continue
        desc = (fn.get("description") or "").strip()
        # Take only the first sentence of the description — that's the
        # essential "what does this do" hook. Rich usage prose confuses
        # small models more than it helps.
        if desc:
            first_stop = desc.find(". ")
            if first_stop != -1:
                desc = desc[: first_stop + 1]
        params = _compact_params_schema(fn.get("parameters") or {})
        lines.append(f'<tool name="{_xml_attr(name)}">')
        if desc:
            lines.append(f"<description>{_xml_text(desc)}</description>")
        # Skip the params tag entirely when the tool takes no arguments —
        # ``<parameters>{}</parameters>`` is pure noise for zero-arg tools
        # like ``list_models`` or ``get_cwd``.
        if params:
            try:
                params_json = json.dumps(
                    params, ensure_ascii=False, separators=(",", ":"),
                )
            except (TypeError, ValueError):
                params_json = "{}"
            lines.append(f"<parameters>{params_json}</parameters>")
        lines.append("</tool>")
    lines.append("</tools>")
    lines.append("")
    lines.append(_ADAPTER_INSTRUCTIONS)
    return "\n".join(lines)


def _xml_attr(s: str) -> str:
    """Minimal escaping for an XML attribute value.

    Tool names are constrained by our schema builder to ``[A-Za-z0-9_]``,
    but we still escape defensively in case a user-defined tool sneaks in
    a funky name before validation catches up.
    """
    return (
        s.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _xml_text(s: str) -> str:
    """Escape text appearing between XML tags.

    We don't need to be strict — the model is going to read this, not an
    XML parser — but ``<`` / ``>`` in descriptions would confuse the
    model's own tag-tracking so we escape them.
    """
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Parser: extract <tool_call> blocks from streamed text
# ---------------------------------------------------------------------------

# Accept any of the tag shapes small models commonly fall back to from
# pre-training. All of them wrap a single tool invocation; the parser
# normalises them into the canonical ``{"name":..., "args":{...}}`` form.
#   <tool_call>      — what our system prompt asks for (JSON body).
#   <execute_tool>   — Gemini's prompt-space default (JSON OR fn-call body).
#   <tool_code>      — another Gemini variant (usually fn-call body).
#   <function_call>  — generic fallback some models emit.
_TOOL_CALL_RE = re.compile(
    r"<(?P<tag>tool_call|execute_tool|tool_code|function_call)>\s*(?P<body>.*?)\s*</(?P=tag)>",
    re.DOTALL | re.IGNORECASE,
)

# Some models like to wrap the JSON in a markdown fence anyway — strip a
# leading ```json or ``` prefix and the trailing ``` before parsing.
_FENCE_PREFIX_RE = re.compile(r"^`{1,3}[A-Za-z0-9_+-]*\n")
# Match 1-3 trailing backticks. Models occasionally start to wrap the body
# in a code fence (` ``` `) but emit only one or two — strip whatever they
# left behind so JSON parsing isn't derailed by stray ticks.
_FENCE_SUFFIX_RE = re.compile(r"\s*`{1,3}\s*$")

# Python-style function call e.g. ``web_search(query="foo", top_k=5)``
# that some models emit inside <execute_tool> blocks.
_FN_CALL_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<args>.*)\)\s*;?\s*$",
    re.DOTALL,
)

# Matches a single ``\`` that is NOT followed by one of the JSON-valid
# escape characters (``\``, ``"``, ``/``, ``b``, ``f``, ``n``, ``r``,
# ``t``, ``u``). Used to repair Windows paths like ``C:\Users\gauta``
# that the model emits raw inside JSON string values.
_BAD_BACKSLASH_RE = re.compile(r'\\(?!["\\/bfnrtu])')


def _try_parse_tool_body(body: str) -> dict[str, Any] | None:
    """Parse a tool-call body into ``{"name": ..., "args": {...}}``.

    Handles both shapes we see in the wild:
      * JSON: ``{"name": "tool", "args": {"x": 1}}`` or ``{"tool_name": ...}``
      * Function-call: ``tool(x=1, y="s")`` — common for Gemini-trained
        models that emit ``<execute_tool>`` from their pre-training.
    """
    body = body.strip()
    body = _FENCE_PREFIX_RE.sub("", body)
    body = _FENCE_SUFFIX_RE.sub("", body)
    if not body:
        return None
    # Try JSON first — cheap and precise.
    try:
        parsed = json.loads(body)
    except Exception:
        parsed = None
    # Repair pass: Windows paths are the #1 cause of JSON parse failure
    # here. Models emit ``"path":"C:\Users\gauta\..."`` literally — those
    # backslashes are invalid JSON escapes (`\U`, `\g` are not defined),
    # so ``json.loads`` rejects the whole body. Double every backslash
    # that isn't already part of a valid escape sequence and retry.
    if parsed is None:
        repaired = _BAD_BACKSLASH_RE.sub(r"\\\\", body)
        try:
            parsed = json.loads(repaired)
        except Exception:
            parsed = None
    # Repair pass #2: missing trailing braces. Small models stream long
    # `edit_file` / `write_file` payloads that span thousands of tokens
    # and sometimes drop the final closing `}` (or both — outer object
    # AND `args` object). The body looks valid up until the very end:
    # ``{"name": "edit_file", "args": {"path": "...", "new_string":
    # "...export default App;"}`` — that's one `{` ahead of `}`. Try
    # appending up to two closers; cheap to try, and the model's intent
    # is unambiguous when everything else parses cleanly. We also gate
    # by an opening-vs-closing brace count so we don't try to "repair"
    # genuinely garbage bodies.
    if parsed is None:
        opens = body.count("{")
        closes = body.count("}")
        deficit = opens - closes
        if 1 <= deficit <= 2:
            try:
                candidate = json.loads(body + ("}" * deficit))
                if isinstance(candidate, dict):
                    parsed = candidate
            except Exception:
                parsed = None
    # Repair pass #3: extra trailing junk. Real failure mode: model
    # closed the outer object too early — wrote `"}}` (closing args +
    # outer) when it meant just `"}` (closing args only), then dangled
    # the `reason` field outside as `, "reason": "..."}`. `json.loads`
    # rejects with "Extra data" but the FIRST valid JSON object is
    # exactly the tool call we want; the discarded tail is just the
    # misplaced display-only `reason` field. Use `raw_decode` to take
    # that first object and ignore everything after.
    if parsed is None:
        try:
            candidate, _end = json.JSONDecoder().raw_decode(body)
            if isinstance(candidate, dict):
                parsed = candidate
        except Exception:
            parsed = None
    if isinstance(parsed, dict):
        # Unwrap shape #1 — full double-wrap with empty outer name:
        #   {"name": "", "args": {"name": "read_file", "args": {...}}}
        # Outer name is empty so dispatch errors `unknown tool: ''`.
        if (
            not (parsed.get("name") or parsed.get("tool") or parsed.get("tool_name"))
            and isinstance(parsed.get("args"), dict)
        ):
            inner = parsed["args"]
            if (
                inner.get("name")
                and ("args" in inner or "arguments" in inner or "parameters" in inner)
            ):
                parsed = inner
        name = (
            parsed.get("name")
            or parsed.get("tool")
            or parsed.get("tool_name")
            or ""
        )
        args = parsed.get("args")
        if args is None:
            args = (
                parsed.get("arguments")
                or parsed.get("parameters")
                or parsed.get("input")
                or {}
            )
        # Unwrap shape #2 — partial double-wrap with the outer name
        # filled but `args` itself nested under another `args` key:
        #   {"name": "bash", "args": {"args": {"command": "..."}}}
        # Here dispatch sees `name=bash` but `args.command` is missing
        # (it's actually under `args.args.command`), so bash returns
        # "empty command" and the call vanishes. We only unwrap when
        # the inner `args` dict is a SINGLE-KEY arg-shaped wrapper —
        # this avoids touching a legitimate call where `args.args`
        # happens to be a real parameter name (no built-in does that
        # but a future user-tool might). The check is conservative:
        # the inner key must be exactly one of the arg-aliases.
        if isinstance(args, dict) and len(args) == 1:
            (only_key,) = args.keys()
            if only_key in ("args", "arguments", "parameters", "input"):
                inner_args = args[only_key]
                if isinstance(inner_args, dict):
                    args = inner_args
                elif isinstance(inner_args, str):
                    try:
                        candidate = json.loads(inner_args)
                        if isinstance(candidate, dict):
                            args = candidate
                    except Exception:
                        pass
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}
        if name:
            return {"name": str(name), "args": args}
    # Fall back to function-call syntax.
    m = _FN_CALL_RE.match(body)
    if not m:
        return None
    name = m.group("name")
    raw_args = (m.group("args") or "").strip()
    args = _parse_fn_call_args(raw_args)
    return {"name": name, "args": args}


def _parse_fn_call_args(raw: str) -> dict[str, Any]:
    """Parse ``key="value", other=123`` into a dict.

    We don't implement a real Python parser — we only handle what small
    models actually emit: comma-separated ``key=value`` pairs where
    ``value`` is a string literal (either kind of quote), a number, a
    bool, or ``None``. Anything weirder falls back to ``{}`` so the
    model sees an empty-args tool call and can correct itself.
    """
    if not raw:
        return {}
    # Wrap in braces and convert to a JSON-ish form. Easier: split pairs by
    # top-level commas (ignoring commas inside quotes or brackets).
    pairs: list[str] = []
    buf = ""
    depth = 0
    quote: str | None = None
    for ch in raw:
        if quote:
            buf += ch
            if ch == quote and not buf.endswith("\\" + quote):
                quote = None
            continue
        if ch in ('"', "'"):
            quote = ch
            buf += ch
            continue
        if ch in "[{(":
            depth += 1
            buf += ch
            continue
        if ch in "]})":
            depth -= 1
            buf += ch
            continue
        if ch == "," and depth == 0:
            pairs.append(buf)
            buf = ""
            continue
        buf += ch
    if buf.strip():
        pairs.append(buf)
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            continue
        k, _, v = pair.partition("=")
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        # Try JSON first so "true"/"false"/numbers/quoted strings parse right.
        # Python-style single quotes need to become double for json.loads.
        v_json = v
        if v_json.startswith("'") and v_json.endswith("'"):
            v_json = '"' + v_json[1:-1].replace('"', '\\"') + '"'
        try:
            out[k] = json.loads(v_json)
        except Exception:
            out[k] = v.strip("'\"")
    return out


def parse_tool_calls_from_text(
    text: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Pull every tool-call tag out of ``text``.

    Returns ``(stripped_text, calls)`` where ``stripped_text`` is the
    original text with every recognised block removed (so the user sees
    only the prose) and ``calls`` is a list of ``{"name": str, "args":
    dict}`` ready to drop into the existing ``tool_calls_buf`` in the
    agent loop.

    Malformed bodies inside a recognised tag are logged and left in-place
    so the model sees its own mistake echoed on the next turn and can
    retry.
    """
    if not text:
        return text, []
    lower = text.lower()
    if not any(
        f"<{t}>" in lower
        for t in ("tool_call", "execute_tool", "tool_code", "function_call")
    ):
        return text, []

    calls: list[dict[str, Any]] = []
    # We iterate matches but strip only the successful ones, so we can't
    # use ``re.sub`` directly. Walk from the end so earlier spans stay
    # valid as we slice.
    spans_to_remove: list[tuple[int, int]] = []
    for match in _TOOL_CALL_RE.finditer(text):
        body = match.group("body") or ""
        parsed = _try_parse_tool_body(body)
        if not parsed:
            log.warning(
                "tool_prompt_adapter: unparseable <%s> body: %r",
                match.group("tag"), body[:200],
            )
            continue
        calls.append(parsed)
        spans_to_remove.append(match.span())

    if not spans_to_remove:
        return text, []

    stripped = text
    for start, end in reversed(spans_to_remove):
        stripped = stripped[:start] + stripped[end:]
    # Collapse any double-blank-lines left behind by stripping.
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()
    return stripped, calls


# ---------------------------------------------------------------------------
# Message-list rewrite for adapter mode
# ---------------------------------------------------------------------------


def rewrite_messages_for_adapter(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten native tool-role / tool_calls structures into plain text.

    Stub-template models have no ``tool`` role in their chat template and
    can't consume ``message.tool_calls`` structurally. We rewrite:

      * ``role == "tool"`` → user-role message wrapped in
        ``<tool_result name="...">...</tool_result>``. If the original
        row had an ``images`` field (computer-use screenshot), we
        preserve it so vision-capable stub templates (Gemma4 e4b *does*
        have vision) can still see the image.
      * ``role == "assistant"`` with ``tool_calls`` → plain assistant
        message whose content has ``<tool_call>{...}</tool_call>``
        blocks appended after any existing prose.

    Everything else (system, plain user, plain assistant) passes through
    untouched — except we drop any lingering ``tool_calls`` key on the
    assistant rewrite path to keep Ollama's schema validator happy.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            name = msg.get("tool_name") or msg.get("name") or ""
            body = msg.get("content") or ""
            wrapped = {
                "role": "user",
                "content": (
                    f'<tool_result name="{_xml_attr(str(name))}">\n'
                    f"{body}\n"
                    f"</tool_result>"
                ),
            }
            # Preserve images so vision-in-adapter-mode keeps working.
            if msg.get("images"):
                wrapped["images"] = msg["images"]
            out.append(wrapped)
            continue

        if role == "assistant" and msg.get("tool_calls"):
            base = msg.get("content") or ""
            parts: list[str] = []
            if base.strip():
                parts.append(base)
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or tc
                tc_name = fn.get("name") or ""
                raw_args = fn.get("arguments")
                if raw_args is None:
                    raw_args = fn.get("args") or {}
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except Exception:
                        raw_args = {}
                if not isinstance(raw_args, dict):
                    raw_args = {}
                try:
                    payload_json = json.dumps(
                        {"name": tc_name, "args": raw_args},
                        ensure_ascii=False,
                    )
                except Exception:
                    payload_json = json.dumps(
                        {"name": tc_name, "args": {}}
                    )
                parts.append(f"<tool_call>\n{payload_json}\n</tool_call>")
            out.append({"role": "assistant", "content": "\n\n".join(parts)})
            continue

        # Pass through. Strip `tool_calls` if present defensively — only
        # the branch above should carry it.
        cleaned = {k: v for k, v in msg.items() if k != "tool_calls"}
        out.append(cleaned)
    return out


def inject_tools_block_into_system(
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append the tools block to the FIRST system message in ``messages``.

    If no system message exists we prepend a new one; this matches the
    existing ``_to_ollama_messages`` output which always opens with a
    single system turn, but we handle the edge case so a future refactor
    that drops the system turn doesn't silently break adapter mode.
    """
    block = build_tools_system_block(tool_schemas)
    if not block:
        return messages
    out = list(messages)
    for i, msg in enumerate(out):
        if msg.get("role") == "system":
            base = (msg.get("content") or "").rstrip()
            sep = "\n\n" if base else ""
            out[i] = {"role": "system", "content": f"{base}{sep}{block}"}
            return out
    # No system message found — inject one at the front.
    return [{"role": "system", "content": block}, *out]
