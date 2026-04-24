"""MCP (Model Context Protocol) stdio client.

Spawns external MCP servers as long-lived child processes, negotiates the
protocol handshake, fetches their advertised tools, and exposes a dispatch
entry point so the agent can invoke those tools like any native one.

Design constraints:
  - Pure stdlib + `asyncio` so we don't pull in the official mcp-python SDK
    (keeps the server dependency surface small; the protocol is small enough
    to hand-roll correctly).
  - One session per configured server (config rows are persisted in
    `mcp_servers`). Sessions are started lazily on first tool call or
    explicit refresh, and re-used across the whole process lifetime.
  - Tool names are namespaced `mcp__<server_name>__<tool>` so they can't
    collide with built-in tools or with tools from another MCP server.
  - Timeouts everywhere — a broken MCP server must NEVER hang the agent loop.

Wire protocol (JSON-RPC 2.0 over newline-delimited JSON on stdin/stdout):
  → {"jsonrpc":"2.0","id":1,"method":"initialize",
     "params":{"protocolVersion":"2024-11-05",
               "capabilities":{},"clientInfo":{"name":"gigachat","version":"1"}}}
  ← {"jsonrpc":"2.0","id":1,"result":{...}}
  → {"jsonrpc":"2.0","method":"notifications/initialized"}
  → {"jsonrpc":"2.0","id":2,"method":"tools/list"}
  ← {"jsonrpc":"2.0","id":2,"result":{"tools":[...]}}
  → {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"...","arguments":{...}}}
  ← {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"..."}],"isError":false}}

Any server that can't complete the handshake inside HANDSHAKE_TIMEOUT_SEC is
considered broken and its session torn down; the row stays in the DB so the
user can edit command/args and retry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from typing import Any

from . import db

log = logging.getLogger(__name__)

# Per-server handshake timeout. MCP servers that take longer than this to
# finish `initialize` are almost always stuck — npm cold-install, python
# module import failures, etc. Kill them fast so the UI gets feedback.
HANDSHAKE_TIMEOUT_SEC = 20.0

# Per-call timeout for `tools/list` and `tools/call`. Individual tools may
# legitimately be slow (web fetches, large file reads), so this is generous.
CALL_TIMEOUT_SEC = 120.0

# Max size of a JSON-RPC line we'll accept from a server. MCP servers
# occasionally stream multi-megabyte outputs (documentation scrapers, full
# file reads); 8 MB is a reasonable upper bound before we consider the
# server hostile or buggy.
MAX_LINE_BYTES = 8 * 1024 * 1024

# Tool-name prefix so MCP tools never collide with built-ins. Kept short to
# fit inside any tokenizer's tool-name length heuristics.
NAMESPACE_PREFIX = "mcp__"


def _tool_full_name(server_name: str, tool_name: str) -> str:
    """Join server + tool name into the form the model sees."""
    return f"{NAMESPACE_PREFIX}{server_name}__{tool_name}"


def _split_full_name(full: str) -> tuple[str, str] | None:
    """Reverse of _tool_full_name. Returns (server_name, tool_name) or None."""
    if not full.startswith(NAMESPACE_PREFIX):
        return None
    rest = full[len(NAMESPACE_PREFIX) :]
    # First `__` separator — server names are user-supplied so could in theory
    # contain double underscores, but we validate at config time to forbid it
    # (see validate_server_name below) so a simple split is safe.
    sep = rest.find("__")
    if sep < 0:
        return None
    return rest[:sep], rest[sep + 2 :]


def validate_server_name(name: str) -> str:
    """Normalise + validate a user-supplied server name.

    Rules:
      - 1–40 chars
      - letters, digits, hyphen, single underscore
      - no leading/trailing separators, no double underscores (so our
        namespace split stays unambiguous)
    """
    n = (name or "").strip()
    if not (1 <= len(n) <= 40):
        raise ValueError("server name must be 1-40 characters")
    for ch in n:
        if not (ch.isalnum() or ch in "-_"):
            raise ValueError("server name may only contain letters, digits, '-', '_'")
    if "__" in n:
        raise ValueError("server name must not contain consecutive underscores")
    if n.startswith(("-", "_")) or n.endswith(("-", "_")):
        raise ValueError("server name must not start or end with '-' or '_'")
    return n


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
class MCPSession:
    """One running MCP subprocess plus its JSON-RPC plumbing.

    A session is disposable — on any fatal protocol error we tear it down,
    log, and let the next call spawn a fresh one. Callers should NOT cache
    a reference; always go through the module-level `sessions` dict.
    """

    def __init__(self, server: dict) -> None:
        self._server = server
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._tools: list[dict] = []
        self._started = False
        self._lock = asyncio.Lock()
        self._stderr_tail: list[str] = []

    @property
    def name(self) -> str:
        return self._server["name"]

    @property
    def tools(self) -> list[dict]:
        """List of tool metadata dicts: [{name, description, inputSchema}]"""
        return list(self._tools)

    @property
    def running(self) -> bool:
        return bool(self._proc and self._proc.returncode is None)

    def stderr_tail(self) -> str:
        """Last ~2KB of stderr for diagnostic display in the settings UI."""
        return "".join(self._stderr_tail)[-2000:]

    async def start(self) -> None:
        """Spawn the subprocess and run the protocol handshake.

        Safe to call multiple times; the inner lock ensures only one
        handshake runs concurrently and subsequent calls are no-ops once
        the session is alive.
        """
        async with self._lock:
            if self._started and self.running:
                return
            await self._start_locked()

    async def _start_locked(self) -> None:
        cmd = (self._server.get("command") or "").strip()
        if not cmd:
            raise RuntimeError("MCP server has no command configured")
        # Refuse relative shell fragments that could trigger PATH surprises
        # on Windows. Resolve to an absolute path if possible; if `shutil.which`
        # can't find it, fall back to the literal (some users give full paths
        # with spaces that Windows accepts only verbatim).
        resolved = shutil.which(cmd) or cmd
        args = [str(a) for a in self._server.get("args") or []]
        # Start from the user's environment so PATH, HOME, Node globals, etc.
        # all work without the user re-declaring them. Layer explicit overrides
        # on top — those are the per-server secrets/flags from the settings UI.
        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in (self._server.get("env") or {}).items()})

        log.info("mcp: starting server %r: %s %s", self.name, resolved, args)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                resolved,
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"MCP command not found: {cmd}") from e
        except Exception as e:
            raise RuntimeError(f"failed to launch MCP server: {e}") from e

        self._reader_task = asyncio.create_task(self._read_loop())
        asyncio.create_task(self._drain_stderr())

        # Handshake: initialize → initialized notification → tools/list.
        try:
            await asyncio.wait_for(self._handshake(), timeout=HANDSHAKE_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            await self.stop()
            tail = self.stderr_tail().strip()
            hint = f"\nstderr: {tail}" if tail else ""
            raise RuntimeError(
                f"MCP server {self.name!r} did not complete the handshake in "
                f"{HANDSHAKE_TIMEOUT_SEC:.0f}s.{hint}"
            )
        except Exception:
            await self.stop()
            raise
        self._started = True

    async def _handshake(self) -> None:
        init_res = await self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "gigachat", "version": "1.0"},
            },
        )
        # Some servers enforce a post-initialize notification before any
        # subsequent calls are accepted.
        await self._notify("notifications/initialized", {})
        caps = (init_res or {}).get("capabilities") or {}
        if "tools" not in caps:
            # Not every server advertises tools; that's fine, we just end
            # up with an empty tool list which the agent harmlessly ignores.
            log.info("mcp: server %r does not advertise tools capability", self.name)
        tools_res = await self._request("tools/list", {})
        raw = (tools_res or {}).get("tools") or []
        cleaned: list[dict] = []
        for t in raw:
            if not isinstance(t, dict):
                continue
            name = t.get("name")
            if not isinstance(name, str) or not name:
                continue
            cleaned.append(
                {
                    "name": name,
                    "description": str(t.get("description") or "")[:1000],
                    "inputSchema": t.get("inputSchema") or {"type": "object"},
                }
            )
        self._tools = cleaned
        log.info("mcp: server %r advertises %d tool(s)", self.name, len(cleaned))

    async def stop(self) -> None:
        """Terminate the subprocess and cancel pending RPCs."""
        async with self._lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        self._started = False
        # Fail any in-flight futures so callers don't hang forever.
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(RuntimeError("MCP session stopped"))
        self._pending.clear()
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
        self._reader_task = None
        proc = self._proc
        self._proc = None
        if proc and proc.returncode is None:
            try:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    proc.kill()
            except Exception:
                pass

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Invoke one of the server's advertised tools.

        Returns the agent-shaped result dict ({ok, output, error?}) so the
        existing tool dispatch pipeline does not need to special-case MCP.
        """
        if not self.running:
            await self.start()
        try:
            raw = await asyncio.wait_for(
                self._request(
                    "tools/call",
                    {"name": tool_name, "arguments": arguments or {}},
                ),
                timeout=CALL_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            return {
                "ok": False,
                "output": "",
                "error": f"MCP tool {tool_name!r} timed out after {CALL_TIMEOUT_SEC:.0f}s",
            }
        except Exception as e:
            return {
                "ok": False,
                "output": "",
                "error": f"MCP call failed: {type(e).__name__}: {e}",
            }
        return _coerce_mcp_result(raw)

    # --- Internal JSON-RPC plumbing --------------------------------------
    async def _request(self, method: str, params: dict) -> dict:
        if not self._proc or self._proc.returncode is not None:
            raise RuntimeError("MCP session is not running")
        req_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        await self._write_line(payload)
        try:
            return await fut
        finally:
            self._pending.pop(req_id, None)

    async def _notify(self, method: str, params: dict) -> None:
        """Fire-and-forget notification (no id, no response expected)."""
        await self._write_line({"jsonrpc": "2.0", "method": method, "params": params})

    async def _write_line(self, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
        stdin = self._proc.stdin if self._proc else None
        if stdin is None:
            raise RuntimeError("MCP session stdin not available")
        stdin.write(data)
        await stdin.drain()

    async def _read_loop(self) -> None:
        """Consumer for stdout — parses lines and fulfills pending futures."""
        assert self._proc and self._proc.stdout
        stdout = self._proc.stdout
        while True:
            try:
                line = await stdout.readline()
            except asyncio.LimitOverrunError:
                # Drop the pathological line — keep the session alive.
                log.warning("mcp: %r produced an oversize line; skipping", self.name)
                continue
            except Exception as e:
                log.info("mcp: %r reader error: %s", self.name, e)
                break
            if not line:
                break  # EOF
            if len(line) > MAX_LINE_BYTES:
                log.warning("mcp: %r line too large (%d bytes); skipping", self.name, len(line))
                continue
            try:
                msg = json.loads(line.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                # MCP servers sometimes spill non-JSON logs to stdout. Ignore
                # instead of crashing — they'll also appear on stderr.
                continue
            self._handle_message(msg)
        # Drain any outstanding futures so callers unblock.
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(RuntimeError("MCP session stdout closed"))
        self._pending.clear()

    def _handle_message(self, msg: dict) -> None:
        # Responses carry an `id`; notifications/requests from the server
        # carry a `method`. We currently ignore server-initiated requests
        # (resource updates, sampling, etc.) — supporting them can be
        # added later without changing the tool surface.
        if "id" in msg and ("result" in msg or "error" in msg):
            rid = msg.get("id")
            fut = self._pending.pop(rid, None) if isinstance(rid, int) else None
            if fut is None or fut.done():
                return
            if "error" in msg:
                err = msg["error"] or {}
                fut.set_exception(
                    RuntimeError(
                        f"MCP error {err.get('code', '?')}: {err.get('message', '')}"
                    )
                )
            else:
                fut.set_result(msg.get("result") or {})

    async def _drain_stderr(self) -> None:
        """Capture stderr into a rolling buffer for error diagnostics."""
        assert self._proc and self._proc.stderr
        while True:
            try:
                line = await self._proc.stderr.readline()
            except Exception:
                break
            if not line:
                break
            try:
                text = line.decode("utf-8", errors="replace")
            except Exception:
                continue
            self._stderr_tail.append(text)
            # Keep the buffer bounded (~50 lines).
            if len(self._stderr_tail) > 50:
                del self._stderr_tail[:-50]


def _coerce_mcp_result(raw: dict) -> dict:
    """Translate an MCP `tools/call` result into {ok, output, error} shape.

    MCP responses come back as `{content: [{type, text|image|...}], isError}`.
    We join all text parts into a single string (that's what the model will
    see in its tool-result row) and surface `isError` as ok=false so the
    agent can react like it would for any other failed tool.
    """
    if not isinstance(raw, dict):
        return {"ok": False, "output": "", "error": "MCP returned a non-object result"}
    content = raw.get("content") or []
    is_error = bool(raw.get("isError"))
    text_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "text":
            txt = item.get("text")
            if isinstance(txt, str):
                text_parts.append(txt)
        elif t == "image":
            # We don't have an attachment-rendering path for MCP images yet;
            # record the mime-type so the model at least knows an image was
            # produced. Extending this to surface the PNG through the
            # screenshot pipeline is a follow-up.
            mime = item.get("mimeType") or "image/*"
            text_parts.append(f"[mcp image attachment: {mime}]")
        else:
            # Unknown content types become a short marker so the model is
            # not silently lied to about the response shape.
            text_parts.append(f"[mcp content type {t!r}]")
    output = "\n".join(p for p in text_parts if p)
    if is_error:
        return {"ok": False, "output": output, "error": output or "MCP tool reported an error"}
    return {"ok": True, "output": output}


# ---------------------------------------------------------------------------
# Module-level session registry
# ---------------------------------------------------------------------------
# server name -> session. Built lazily; restarted via `refresh()` when a row
# is added/edited/deleted through the settings UI.
_sessions: dict[str, MCPSession] = {}
_sessions_lock = asyncio.Lock()


async def get_session(name: str) -> MCPSession | None:
    """Fetch an already-running session (does not spawn new ones)."""
    async with _sessions_lock:
        return _sessions.get(name)


async def ensure_started(server: dict) -> MCPSession:
    """Start (or reuse) a session for the given config row."""
    async with _sessions_lock:
        existing = _sessions.get(server["name"])
        if existing and existing.running:
            return existing
        sess = MCPSession(server)
    # start outside the registry lock so a slow handshake doesn't block
    # concurrent calls targeting other servers.
    await sess.start()
    async with _sessions_lock:
        old = _sessions.get(server["name"])
        if old and old is not sess:
            await old.stop()
        _sessions[server["name"]] = sess
    return sess


async def stop_session(name: str) -> None:
    """Terminate one session (used when a row is disabled or deleted)."""
    async with _sessions_lock:
        sess = _sessions.pop(name, None)
    if sess:
        await sess.stop()


async def stop_all() -> None:
    """Shut down every running session (used at app teardown)."""
    async with _sessions_lock:
        items = list(_sessions.items())
        _sessions.clear()
    for _, sess in items:
        try:
            await sess.stop()
        except Exception:
            pass


async def refresh_all() -> dict:
    """Reconcile the session map with the DB rows.

    - Stops sessions whose config row vanished or is now disabled.
    - Starts sessions for newly-enabled rows.
    - Restarts sessions whose command/args/env changed (detected by diffing
      the stored tuple against the live one).

    Returns a dict {name: {running, tools, error?}} the UI can render.
    """
    desired = {s["name"]: s for s in db.list_mcp_servers() if s["enabled"]}
    report: dict[str, dict] = {}

    # Stop sessions no longer desired / whose fingerprint changed.
    async with _sessions_lock:
        alive = dict(_sessions)
    for name, sess in alive.items():
        target = desired.get(name)
        if target is None or _fingerprint(target) != _fingerprint(sess._server):
            await stop_session(name)

    # Start / restart.
    for name, server in desired.items():
        try:
            sess = await ensure_started(server)
            report[name] = {
                "running": sess.running,
                "tools": [t["name"] for t in sess.tools],
            }
        except Exception as e:
            report[name] = {"running": False, "tools": [], "error": str(e)}
    return report


def _fingerprint(server: dict) -> tuple:
    """Tuple that changes whenever a server's launch config changes.

    Used by refresh_all to decide whether to restart an already-running
    session. env order is sorted so a dict rebuild with identical content
    does not trigger a spurious restart.
    """
    return (
        server.get("command"),
        tuple(server.get("args") or []),
        tuple(sorted((server.get("env") or {}).items())),
    )


# ---------------------------------------------------------------------------
# Integration helpers consumed by agent.py / tools.py / app.py
# ---------------------------------------------------------------------------
def tool_schemas_for_agent() -> list[dict]:
    """Return Ollama-compatible tool schemas for every live MCP tool.

    The agent appends this list to its built-in TOOL_SCHEMAS before each
    Ollama request, which is how the model discovers MCP tools without any
    special prompting.
    """
    schemas: list[dict] = []
    # Snapshot outside async so we can call this from the synchronous
    # per-turn schema assembly in agent.py.
    for name, sess in list(_sessions.items()):
        if not sess.running:
            continue
        for t in sess.tools:
            full = _tool_full_name(name, t["name"])
            params = t.get("inputSchema") or {"type": "object"}
            # MCP schemas are JSON Schema, which Ollama passes through to the
            # model as-is. Guard against servers that emit non-object
            # top-levels by forcing an object shell.
            if not isinstance(params, dict) or params.get("type") != "object":
                params = {"type": "object"}
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": full,
                        "description": (t.get("description") or f"{full} (MCP tool)")[:1000],
                        "parameters": params,
                    },
                }
            )
    return schemas


def is_mcp_tool(name: str) -> bool:
    """True if the given tool name is namespaced to an MCP server."""
    return _split_full_name(name) is not None


async def dispatch_tool(full_name: str, args: dict) -> dict:
    """Route a namespaced MCP tool call to the correct session."""
    parts = _split_full_name(full_name)
    if not parts:
        return {"ok": False, "output": "", "error": f"not an MCP tool: {full_name}"}
    server_name, tool_name = parts
    sess = await get_session(server_name)
    if sess is None:
        # Session may have been torn down by a restart; try to revive from
        # the DB row rather than failing the turn.
        row = next((s for s in db.list_mcp_servers() if s["name"] == server_name and s["enabled"]), None)
        if row is None:
            return {
                "ok": False,
                "output": "",
                "error": f"MCP server {server_name!r} is not enabled",
            }
        try:
            sess = await ensure_started(row)
        except Exception as e:
            return {"ok": False, "output": "", "error": str(e)}
    return await sess.call_tool(tool_name, args or {})


async def startup() -> None:
    """Called from FastAPI's lifespan hook — spin up every enabled server."""
    try:
        await refresh_all()
    except Exception as e:
        log.warning("mcp: startup refresh failed: %s", e)


async def shutdown() -> None:
    """Called from FastAPI's lifespan hook — tear everything down cleanly."""
    await stop_all()
