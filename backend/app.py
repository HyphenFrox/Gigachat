"""FastAPI app serving the chat UI and agent over SSE.

Routes:
  GET  /                       - static index.html
  GET  /static/*               - other static assets
  GET  /api/models             - list installed Ollama models
  GET  /api/conversations      - list conversations
  POST /api/conversations      - create
  GET  /api/conversations/{id} - fetch meta + messages
  PATCH /api/conversations/{id} - update title/model/auto_approve/cwd
  DELETE /api/conversations/{id}
  POST /api/conversations/{id}/messages  - SSE stream of the agent turn
                                           Optional `images` list in body is
                                           attached to the user message as
                                           multimodal input.
  POST /api/conversations/{id}/approve   - approve/reject a pending tool call
                                           (the original /messages stream is
                                           still blocked and resumes on its
                                           own once the approval is submitted)
  POST /api/conversations/{id}/uploads   - accept a pasted image (multipart)
                                           and return the saved filename so
                                           the frontend can reference it in
                                           the next /messages request
  POST /api/conversations/{id}/restore/{stamp}
                                         - roll files back to a checkpoint
                                           snapshot taken before a write/edit
  GET  /api/screenshots/{name}           - serve a computer-use screenshot
  GET  /api/uploads/{name}               - serve a user-uploaded image
  GET  /api/memories                     - list global (cross-conversation) memories
  POST /api/memories                     - add a new global memory
  PATCH /api/memories/{mid}              - edit content/topic of a global memory
  DELETE /api/memories/{mid}             - remove a global memory
  GET  /api/secrets                      - list stored secrets (metadata only)
  GET  /api/secrets/{sid}                - reveal one secret (value + metadata)
  POST /api/secrets                      - store a new secret
  PATCH /api/secrets/{sid}               - update a secret's value/name/description
  DELETE /api/secrets/{sid}              - delete a secret
  GET  /api/conversations/{cid}/memory   - read the per-conversation memory file
  PUT  /api/conversations/{cid}/memory   - replace the per-conversation memory file
  DELETE /api/conversations/{cid}/memory - clear the per-conversation memory file
  GET  /api/conversations/{cid}/pinned   - list pinned messages in a conversation
  GET  /api/conversations/{cid}/usage    - cumulative-usage breakdown + budget
  DELETE /api/conversations/{cid}/messages/{mid} - delete one message
  GET  /api/auth/status                  - is a password required / am I logged in
  POST /api/auth/login                   - exchange password for a session cookie
  POST /api/auth/logout                  - clear the session cookie
"""

from __future__ import annotations

import asyncio
import base64
import json
import secrets
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Force the Windows Proactor event loop BEFORE anything else imports asyncio.
#
# On Windows, Python 3.8+ defaults to ProactorEventLoop which supports
# asyncio subprocesses (what our `bash` / `bash_bg` tools need). But some
# versions/configurations of uvicorn call WindowsSelectorEventLoopPolicy on
# startup, which leaves `asyncio.create_subprocess_shell()` raising a bare
# NotImplementedError — breaking every tool that shells out. Setting the
# Proactor policy explicitly here guarantees it, regardless of how the
# server was launched.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        # If the platform exposes an unexpected policy name (e.g. on some
        # Python builds) fall back to the default rather than crashing.
        pass
else:
    # uvloop on Linux/macOS: drop-in asyncio replacement that's typically
    # 10-20% faster for I/O-heavy workloads (every chat turn streams SSE,
    # every indexer fans out HTTP). Falls through silently when uvloop
    # isn't installed — it's an optional `pip install uvloop` not a hard
    # dependency, so unsupported environments (Windows; some niche
    # platforms) keep working with the stdlib loop.
    try:
        import uvloop  # type: ignore
        uvloop.install()
    except Exception:
        # Module missing or install rejected — stdlib loop is fine.
        pass

import httpx
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from . import agent, auth, compute_pool, db, mcp, model_sync, ollama_runtime, push, split_lifecycle, split_runtime, sysdetect, tools

ROOT = Path(__file__).resolve().parent.parent
# In production, the frontend is built to `frontend/dist` by Vite.
# In development, `npm run dev` serves the frontend itself on :5173 and
# proxies /api/* to us — the static-serving routes below are a no-op.
FRONTEND_DIST = ROOT / "frontend" / "dist"
FRONTEND_SRC = ROOT / "frontend"
FRONTEND = FRONTEND_DIST if FRONTEND_DIST.exists() else FRONTEND_SRC

# Hard cap on a single pasted image: 10 MB is plenty for a full-screen PNG
# at 1440p and keeps a stray multi-hundred-MB paste from blowing up RAM.
UPLOAD_MAX_BYTES = 10 * 1024 * 1024
ALLOWED_UPLOAD_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
# Document uploads — we extract plain text on the server and hand it back
# to the UI, which prepends it to the next user message. No binary is sent
# to the model, so these work with text-only chat models too.
ALLOWED_DOCUMENT_TYPES = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/csv": ".csv",
}
# Hard cap on extracted text inserted into a single message — keeps a huge
# PDF from nuking the context window.
DOCUMENT_EXTRACT_MAX_CHARS = 40000

# ---------------------------------------------------------------------------
# Lifespan event handler
#
# Replaces the legacy `@app.on_event("startup")` / `@app.on_event("shutdown")`
# decorators that FastAPI deprecated in favour of an async context manager.
# Everything that used to live behind a per-event decorator is invoked here in
# a single explicit sequence: setup runs before `yield`, teardown runs after.
#
# The handler functions referenced below (`_configure_structured_logging`,
# `_start_scheduler`, etc.) are defined further down in this module. Python
# resolves those names at call time, not at function-definition time, so
# forward references work as long as the lifespan body never runs before the
# module finishes importing — which is exactly the case (uvicorn invokes the
# context manager when the server actually boots).
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # ----- startup ---------------------------------------------------------
    # Structured logging first so subsequent hooks emit JSON-formatted lines.
    await _configure_structured_logging()
    # Capture the running event loop so threadpool endpoints can schedule
    # background work via `run_coroutine_threadsafe`.
    await _capture_main_loop()
    # Background daemons that watchdog state and reclaim resources.
    await _start_scheduler()
    await _start_retention_sweeper()
    await _start_stale_watchdog()
    # Crash-resilience: re-fire any conversation that was mid-turn last
    # process. Runs as its own task with a 3-second delay so Ollama has a
    # chance to come up first.
    await _start_resumer()
    # Bring Ollama up before MCP, which may have servers that depend on it.
    await _auto_start_ollama()
    # Pull the auto-tuned default chat + embedding models in the background;
    # the UI stays responsive while the multi-GB download streams.
    await _auto_tune_ollama_models()
    # Compute-pool liveness sweep (5-min cadence) and split-model boot
    # reconcile (clears stale `running` rows from the previous process).
    await _start_compute_pool_probe()
    await _reconcile_split_models()
    # MCP servers last so they can talk to Ollama + the compute pool when
    # they need to.
    await _start_mcp()
    # Event runtime — file-watcher polling daemon. Webhook firing is
    # inline in the route handler, but the watcher needs its own loop.
    # Kept after MCP so any tool-system bootstrap finishes first.
    from . import event_runtime as _evrt
    await _evrt.start_event_runtime()
    # P2P LAN discovery (mDNS). Advertises this install on
    # `_gigachat._tcp.local.` and listens for peer ads — the Bluetooth-
    # style pairing UX rides on top. Best-effort: failure (e.g. multicast
    # blocked by the OS firewall) logs a warning and the rest of the app
    # boots normally.
    try:
        from . import p2p_discovery as _p2pd
        # Pick the same port FastAPI is serving on so peers know
        # exactly where to reach us. uvicorn sets PORT in env.
        adv_port = int(os.environ.get("PORT", "8000"))
        await _p2pd.start(advertise_port=adv_port)
    except Exception as e:
        log.warning("p2p_discovery startup failed: %s", e)
    # Rendezvous client — registers this install with the GCP Cloud
    # Run rendezvous so other peers across the internet can find us.
    # No-op when GIGACHAT_RENDEZVOUS_URL is unset OR when the user
    # has Public Pool toggled off. Privacy: rendezvous only sees
    # identity + STUN endpoints, NEVER prompts and (per the new P2P
    # architecture) NEVER our model inventory either.
    try:
        from . import p2p_rendezvous as _rdv
        await _rdv.start()
    except Exception as e:
        log.warning("p2p_rendezvous startup failed: %s", e)
    # Pool inventory loop — keeps a local cache of "what models does
    # each peer in the swarm have" by directly querying each peer's
    # encrypted /api/tags endpoint. The model picker + smart routing
    # read from this cache; the rendezvous never sees model data.
    try:
        from . import p2p_pool_inventory as _inv
        await _inv.start()
    except Exception as e:
        log.warning("p2p_pool_inventory startup failed: %s", e)
    # Relay inbox poll loop — TURN-style fallback for symmetric-NAT
    # peers that can't be reached directly. The loop long-polls the
    # rendezvous's /relay/inbox/{device_id} for envelopes addressed
    # to us, dispatching each through the secure-proxy verify+forward
    # path. Relay sees only ciphertext.
    try:
        from . import p2p_relay as _relay
        await _relay.start()
    except Exception as e:
        log.warning("p2p_relay startup failed: %s", e)

    yield

    # ----- shutdown --------------------------------------------------------
    # Reverse-ish order: stop the things that depend on Ollama / split
    # processes first, then the daemons. Each handler is independently
    # robust — failures during shutdown are swallowed inside each helper so
    # uvicorn always exits cleanly.
    try:
        from . import p2p_relay as _relay
        await _relay.stop()
    except Exception as e:
        log.warning("p2p_relay shutdown failed: %s", e)
    try:
        from . import p2p_pool_inventory as _inv
        await _inv.stop()
    except Exception as e:
        log.warning("p2p_pool_inventory shutdown failed: %s", e)
    try:
        from . import p2p_rendezvous as _rdv
        await _rdv.stop()
    except Exception as e:
        log.warning("p2p_rendezvous shutdown failed: %s", e)
    try:
        from . import p2p_discovery as _p2pd
        await _p2pd.stop()
    except Exception as e:
        log.warning("p2p_discovery shutdown failed: %s", e)
    from . import event_runtime as _evrt
    await _evrt.stop_event_runtime()
    await _stop_mcp()
    await _stop_split_models()
    await _stop_compute_pool_probe()
    await _stop_stale_watchdog()
    await _stop_scheduler()
    await _close_shared_http_client()


# orjson is a Rust-based JSON encoder ~3-5× faster than the stdlib
# `json` module that FastAPI uses by default. Setting the default
# response class to ORJSONResponse propagates that to every endpoint
# that returns a dict (`return {"ok": True, ...}`). Streaming endpoints
# (chat SSE, model-pull progress) explicitly use StreamingResponse and
# aren't affected.
#
# `orjson` is an optional dep: it ships in `requirements.txt` but if
# the install was bypassed (developer setup, edge environment),
# `ORJSONResponse` would raise ImportError at import time. We probe
# once and fall back to the default class so the server starts cleanly
# either way.
try:
    from fastapi.responses import ORJSONResponse  # type: ignore
    _DEFAULT_RESPONSE_CLASS = ORJSONResponse
except ImportError:
    _DEFAULT_RESPONSE_CLASS = JSONResponse


app = FastAPI(
    title="Gigachat",
    lifespan=lifespan,
    default_response_class=_DEFAULT_RESPONSE_CLASS,
)
db.init()


# ---------------------------------------------------------------------------
# Access-control middleware
#
# In LAN mode the server binds to 0.0.0.0 so the host machine can hit itself
# via ``localhost`` *and* other devices on the same Wi-Fi/Ethernet can hit it
# via the LAN IP. That's a wider listener than we want to trust unconditionally
# — a public IP could reach the port if the user is on a network with a
# misconfigured firewall, and we never want Tailscale CGNAT clients to drain
# the LLM either. This middleware closes both gaps: it inspects
# ``request.client.host`` and admits only loopback (this machine) or RFC1918
# LAN sources. Everything else — public IPs, Tailscale CGNAT, anything weird —
# gets a flat 403.
#
# On top of that, every non-loopback request must carry a valid session cookie
# (or Bearer token). Loopback is implicitly trusted because if someone can
# already execute code on the box, there's no boundary left to enforce.
#
# The login endpoint and static assets are public so the frontend can render
# its password form before authenticating.
# ---------------------------------------------------------------------------
_AUTH_EXEMPT_PREFIXES = (
    "/api/auth/",
    "/static/",
    "/assets/",
    "/favicon",
    "/manifest",
    "/icons/",
    "/sw.js",
)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cfg = auth.get_config()
        # Pure loopback deployment — no auth, no IP filtering. The OS
        # already ensured only local processes can reach the socket.
        if not auth.requires_password(cfg):
            return await call_next(request)

        client_host = request.client.host if request.client else None
        is_loopback = auth.is_loopback(client_host)
        is_lan = auth.is_lan_client(client_host)

        # LAN mode: reject anything that didn't come in over the loopback
        # interface or a private RFC1918 / IPv6 ULA range. This is defence
        # in depth — the port is physically listening on 0.0.0.0 so a
        # public client *could* TCP-connect if the firewall is wide open,
        # but they'll get a 403 before any handler runs. Tailscale CGNAT
        # (100.64.0.0/10) is intentionally NOT in the allowlist: this app
        # stays on the user's own LAN.
        if not (is_loopback or is_lan):
            return JSONResponse(
                {"error": "forbidden"},
                status_code=403,
            )

        path = request.url.path
        # The login page itself plus the static assets needed to render it
        # are always public for allowed clients. Everything else under
        # /api/* is gated, and so is the SPA entrypoint (the entrypoint is
        # ``/``, and that's also the login page in the unauthenticated
        # case, so we let it through — the frontend does the redirect).
        if path == "/" or any(path.startswith(p) for p in _AUTH_EXEMPT_PREFIXES):
            return await call_next(request)
        # Loopback on-host is trusted (the operator themselves or a local
        # script). Other LAN clients still need to authenticate.
        if is_loopback:
            return await call_next(request)
        token = request.cookies.get(auth.SESSION_COOKIE)
        if not token:
            # Also accept Bearer header for programmatic clients (curl,
            # mobile app, scripted automation). Keeps the cookie path as
            # the primary flow but doesn't force it.
            header = request.headers.get("authorization", "")
            if header.lower().startswith("bearer "):
                token = header[7:].strip()
        if auth.verify_token(token):
            return await call_next(request)
        return JSONResponse(
            {"error": "authentication required", "requires_password": True},
            status_code=401,
        )


app.add_middleware(AuthMiddleware)


# Gzip compression for large JSON responses. The threshold is generous
# (1 KiB) so small endpoint replies don't pay the encoding cost; only
# the chunky ones (codebase index lists, model inventories, settings
# dumps) get compressed. Streaming responses (chat SSE, model-pull
# progress) are explicitly StreamingResponse and bypass middleware,
# preserving the streaming behaviour.
#
# gzip is universal across browsers and saves ~70-90% on JSON. CPU
# cost (~1-2 ms per response at level 5) is dwarfed by network
# transfer time on any non-localhost client.
from starlette.middleware.gzip import GZipMiddleware  # noqa: E402

app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)


# ---------------------------------------------------------------------------
# Scheduled-tasks background daemon
#
# The `schedule_task` tool writes rows into the `scheduled_tasks` table with
# a `next_run_at` timestamp. This coroutine, started at FastAPI startup and
# run for the lifetime of the server, polls that table every SCHED_POLL_SEC
# seconds and fires any due rows as standalone agent turns inside a fresh
# conversation. One-shots are deleted after firing; recurring tasks have
# their next_run_at bumped forward by `interval_seconds`.
#
# All work happens on the event loop — no threads — because the agent
# already runs async end-to-end. A single failure (Ollama down, bad prompt)
# is isolated to its own try/except so one broken task can't stall the
# whole scheduler.
# ---------------------------------------------------------------------------
SCHED_POLL_SEC = 30
_SCHED_TASK: asyncio.Task | None = None

# Track cwds currently being indexed so two conversations pointed at the same
# directory can share the single in-flight run instead of duplicating work.
# Each entry is the absolute, resolved cwd; it's removed when the task
# finishes. The dedup is advisory only — if we ever race and miss, the
# per-file delete-before-insert in the indexer keeps the chunks clean.
_CODEBASE_INDEX_INFLIGHT: set[str] = set()

# Reference to the main event loop, captured at startup. Sync FastAPI
# endpoints run on the threadpool — `asyncio.get_running_loop()` raises
# there — so background work triggered by sync endpoints (like auto-index
# on new-chat creation) needs this explicit handle to schedule onto the
# real event loop via `run_coroutine_threadsafe`.
_MAIN_LOOP: asyncio.AbstractEventLoop | None = None


def _kick_codebase_index(cwd: str | None) -> None:
    """Fire-and-forget: schedule a background index of `cwd` if one isn't
    already running and the current state is stale (never indexed or errored).

    Called whenever a conversation's cwd is set or changed. Safe to call
    liberally — the function is cheap when the index is already fresh, and
    the inflight set prevents duplicate workers.

    Swallows all exceptions: this is a best-effort UX feature, not a critical
    path. Works from both async endpoints (running loop available) and sync
    endpoints executed on FastAPI's threadpool (we fall back to the main
    loop handle captured at startup). If neither is available — e.g. called
    at import time before startup has fired — we silently drop.
    """
    if not cwd:
        return
    try:
        root = str(Path(cwd).expanduser().resolve())
    except Exception:
        return
    if root in _CODEBASE_INDEX_INFLIGHT:
        return
    existing = db.get_codebase_index(root)
    # Only (re)index when we'd actually change something — skip if the row is
    # fresh ('ready') to avoid re-embedding every time the user opens a chat.
    if existing and existing["status"] in ("indexing", "ready"):
        return

    async def _run():
        try:
            from . import tools as _tools
            await _tools._codebase_index_cwd_impl(root)
        except Exception as e:
            print(f"[codebase-index] {root!r} failed: {e}", file=sys.stderr)
        finally:
            _CODEBASE_INDEX_INFLIGHT.discard(root)

    # Prefer the loop in the current thread (async endpoints); fall back to
    # the main loop captured at startup for sync endpoints running on the
    # threadpool. Without this fallback, sync endpoints like POST
    # /api/conversations silently drop the index request.
    try:
        loop = asyncio.get_running_loop()
        in_thread = False
    except RuntimeError:
        loop = _MAIN_LOOP
        in_thread = True
    if loop is None or loop.is_closed():
        return

    _CODEBASE_INDEX_INFLIGHT.add(root)
    db.upsert_codebase_index(root, status="pending")

    if in_thread:
        asyncio.run_coroutine_threadsafe(_run(), loop)
    else:
        loop.create_task(_run())


def _kick_docs_url_crawl(did: str | None) -> None:
    """Fire-and-forget: schedule a BFS crawl + embed of `did`'s URL seed.

    Mirrors `_kick_codebase_index` shape (loop-capture fallback for sync
    endpoints; exception-swallowing; best-effort) but dispatches to
    `_docs_url_crawl_impl` instead. The inflight set lives inside tools.py
    so we only need to start the worker here.
    """
    if not did:
        return
    # Flip status to 'pending' immediately so the UI shows progress before
    # the worker actually starts. The crawler itself will transition to
    # 'crawling' once it begins.
    try:
        existing = db.get_doc_url(did)
        if not existing:
            return
        if existing["status"] == "crawling":
            # Let the inflight one finish; reindex endpoint can force a rerun.
            return
        db.update_doc_url(did, status="pending", error="")
    except Exception:
        return

    async def _run():
        try:
            await tools._docs_url_crawl_impl(did)
        except Exception as e:
            print(f"[docs-url] {did!r} failed: {e}", file=sys.stderr)

    try:
        loop = asyncio.get_running_loop()
        in_thread = False
    except RuntimeError:
        loop = _MAIN_LOOP
        in_thread = True
    if loop is None or loop.is_closed():
        return
    if in_thread:
        asyncio.run_coroutine_threadsafe(_run(), loop)
    else:
        loop.create_task(_run())


# Hard fallback when neither the user nor the auto-tuner has picked a model
# yet — used at the very first startup before `auto_tune_models()` finishes.
# Gemma 4 e4b is the mid-tier variant that fits on 8 GB+ VRAM hosts and is
# our reference target for prompt/tool-call formatting.
DEFAULT_CHAT_MODEL_FALLBACK = "gemma4:e4b"


def _resolve_default_chat_model() -> str:
    """Decide which model a new conversation should be created with.

    Precedence:
      1. User-chosen default from the settings table (Settings → Default model).
      2. The auto-tuner's pick for this hardware (see ollama_runtime).
      3. Hard fallback constant so we never hand back an empty string.
    """
    chosen = db.get_setting("default_chat_model")
    if isinstance(chosen, str) and chosen.strip():
        return chosen.strip()
    rec = ollama_runtime.get_recommendation().get("chat_model")
    if isinstance(rec, str) and rec.strip():
        return rec.strip()
    return DEFAULT_CHAT_MODEL_FALLBACK


def _snippet(text: str, limit: int = 140) -> str:
    """Collapse whitespace and clip to `limit` chars for a push preview."""
    s = " ".join((text or "").split())
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip() + "…"


async def _safe_push(payload: dict) -> None:
    """Fire a push fan-out without letting an error take down the caller.

    Runs the blocking pywebpush calls on a worker thread so the event loop
    isn't held up by a slow push-service round-trip.
    """
    try:
        await asyncio.to_thread(push.send_to_all, payload)
    except Exception as e:
        # Don't propagate — push is best-effort.
        print(f"[push] send_to_all failed: {e!r}", file=sys.stderr)


async def _run_scheduled_prompt(task: dict) -> None:
    """Execute one scheduled row: drive the agent for one turn.

    Two firing modes:

    * Target conversation set (`task["target_conversation_id"]`) — this is a
      `schedule_wakeup` row. We RESUME that existing conversation, append
      the prompt as a user turn, and drive one agent iteration. The chat
      stays in the sidebar where the user expects it.

    * Target unset — classic `schedule_task` path. We create a new
      conversation titled `Scheduled: <name>` so the unattended run shows
      up as its own transcript.

    When push notifications are enabled we fire one after the turn finishes
    so the user can see the scheduled job landed even if the browser is closed.
    """
    conv_id: str | None = None
    final_text = ""
    kind = task.get("kind") or "task"
    is_resume = bool(task.get("target_conversation_id"))  # wakeup or loop
    try:
        if is_resume:
            # Resume mode — reuse the target conversation if it still exists.
            # If the user deleted it in the meantime, drop silently (the row
            # has already been removed by the caller, so no retry).
            target = db.get_conversation(task["target_conversation_id"])
            if not target:
                print(
                    f"[scheduler] {kind} target {task['target_conversation_id'][:8]} "
                    "no longer exists; skipping",
                    file=sys.stderr,
                )
                # Stop a loop that lost its target — otherwise we burn a push
                # notification every tick forever.
                if kind == "loop":
                    db.cancel_loops_for_conversation(task["target_conversation_id"])
                return
            conv_id = target["id"]
        else:
            conv = db.create_conversation(
                title=f"Scheduled: {task['name']}",
                model=_resolve_default_chat_model(),
                cwd=task["cwd"],
                auto_approve=True,  # scheduled tasks fire unattended — no UI to click Approve
            )
            conv_id = conv["id"]
        # Drain the agent turn. We sniff the last non-empty assistant text so
        # the push preview shows something meaningful instead of just "done".
        async for ev in agent.run_turn(conv_id, user_text=task["prompt"]):
            if ev.get("type") == "assistant_message":
                text = (ev.get("content") or "").strip()
                if text:
                    final_text = text
    except Exception as e:
        # Log to stderr so a crash is visible in the dev console but the
        # loop keeps going.
        print(f"[scheduler] task {task.get('name')!r} failed: {e}", file=sys.stderr)

    # Fire a push notification regardless of task outcome — a silent failure
    # is worse than a "task finished" ping that leads the user to investigate.
    title_prefix = {
        "loop": "loop tick",
        "wakeup": "wakeup",
        "task": "scheduled",
    }.get(kind, "scheduled")
    await _safe_push(
        {
            "title": f"Gigachat: {title_prefix} — {task.get('name') or 'continued'}",
            "body": _snippet(final_text or "Scheduled run completed."),
            "tag": f"sched-{task.get('id', '')}",
            "conversation_id": conv_id,
            "kind": kind,
        }
    )


async def _scheduled_tasks_daemon() -> None:
    """Forever-loop that fires due scheduled tasks. Started at app startup."""
    import time
    while True:
        try:
            now = time.time()
            due = db.get_due_scheduled_tasks(now)
            for task in due:
                # Fire-and-forget: we don't await the agent here because a
                # long-running scheduled job would block the scheduler for
                # every subsequent task. Each run is its own background task.
                asyncio.create_task(_run_scheduled_prompt(task))
                if task.get("interval_seconds"):
                    # Recurring — bump next_run_at; never lets next_run drift
                    # into the past by more than one interval.
                    nxt = now + int(task["interval_seconds"])
                    db.update_scheduled_task_next_run(task["id"], nxt)
                else:
                    # One-shot — delete the row so it doesn't re-fire.
                    db.delete_scheduled_task(task["id"])
        except Exception as e:
            print(f"[scheduler] daemon tick failed: {e}", file=sys.stderr)
        await asyncio.sleep(SCHED_POLL_SEC)


async def _configure_structured_logging() -> None:
    """Install the JSON log formatter before any other startup hook runs.

    Every hook below may emit log lines — we want them in the same
    newline-delimited JSON shape as the `tool_call` events emitted by
    `telemetry.timed_tool`, so aggregators (jq, grep, any log shipper)
    see one format across the whole process.
    """
    from .telemetry import configure_logging
    configure_logging()


async def _capture_main_loop() -> None:
    """Stash the main event loop so sync endpoints on the threadpool can
    schedule background work via `run_coroutine_threadsafe`. Must run before
    any sync endpoint might fire — `on_event("startup")` handlers execute
    in the order registered, so declaring this here (early in the module)
    is sufficient.
    """
    global _MAIN_LOOP
    _MAIN_LOOP = asyncio.get_running_loop()


async def _start_scheduler() -> None:
    """Kick off the background scheduler once the event loop is running."""
    global _SCHED_TASK
    if _SCHED_TASK is None or _SCHED_TASK.done():
        _SCHED_TASK = asyncio.create_task(_scheduled_tasks_daemon())


_RETENTION_TASK: asyncio.Task | None = None


async def _start_retention_sweeper() -> None:
    """Kick off the disk-retention daemon.

    Deletes expired checkpoints + orphaned memory files at a slow cadence
    (see `retention.SWEEP_INTERVAL_SECONDS`). Started here so the first
    sweep reclaims whatever leaked across the last process's lifetime.
    The daemon's own exception handler keeps it alive across transient
    filesystem errors.
    """
    global _RETENTION_TASK
    if _RETENTION_TASK is not None and not _RETENTION_TASK.done():
        return
    from . import retention
    from .tools import CHECKPOINT_DIR, MEMORY_DIR, SCREENSHOT_DIR, UPLOAD_DIR
    _RETENTION_TASK = asyncio.create_task(
        retention.sweep_daemon(
            CHECKPOINT_DIR, MEMORY_DIR, UPLOAD_DIR, SCREENSHOT_DIR,
        )
    )


# ---------------------------------------------------------------------------
# Stale-turn watchdog
#
# The startup resumer (see `_resume_interrupted_conversations` below) handles
# the server-restart case — rows left at `state='running'` when the worker
# was killed. This watchdog handles the mid-session case: an async-generator
# cancellation path (client disconnect, uvicorn --reload tearing down the
# worker without a clean shutdown hook, unhandled exception upstream of
# `run_turn`'s finally) that bypasses the state-reset code. Without it, the
# UI shows a perpetual "working..." spinner and the user can't send new
# messages for that conversation.
#
# How it works:
#   - `agent._ACTIVE_TURN_IDS` is a set of conv_ids currently being iterated
#     in this worker's event loop. The `run_turn` wrapper registers on entry
#     and removes on exit (in its finally).
#   - This daemon periodically scans DB rows in `state='running'`. Any row
#     whose id is NOT in the set — and whose last update is older than the
#     grace period — is abandoned. We flip it to 'idle' so the user can
#     continue.
#   - The grace period avoids racing a turn that just set `state='running'`
#     in the same tick the watchdog polled.
# ---------------------------------------------------------------------------
_STALE_WATCHDOG_TASK: asyncio.Task | None = None
_STALE_POLL_SEC = 30
_STALE_GRACE_SEC = 10  # leave fresh turns alone for at least this long


async def _stale_turn_watchdog() -> None:
    """Flip `state='running'` rows with no live turn back to `state='idle'`."""
    while True:
        try:
            running = db.list_conversations_by_state("running")
            now = time.time()
            for conv in running:
                cid = conv.get("id")
                if not cid:
                    continue
                age = now - float(conv.get("updated_at") or 0)
                # Grace period: the wrapper sets state='running' a hair before
                # the caller starts iterating. Don't second-guess fresh turns.
                if age < _STALE_GRACE_SEC:
                    continue
                if agent.is_turn_active(cid):
                    continue
                # Row says running, but no async generator in this worker is
                # iterating for it — the turn is abandoned. Reset so the UI
                # unblocks.
                try:
                    db.set_conversation_state(cid, "idle")
                    print(
                        f"[stale-watchdog] reset {cid!r} — DB said running "
                        f"but no active turn (age={age:.0f}s)",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(
                        f"[stale-watchdog] failed to reset {cid!r}: {e}",
                        file=sys.stderr,
                    )
        except Exception as e:
            # Don't let a transient DB hiccup kill the daemon.
            print(f"[stale-watchdog] tick failed: {e}", file=sys.stderr)
        await asyncio.sleep(_STALE_POLL_SEC)


async def _start_stale_watchdog() -> None:
    """Kick off the stale-turn watchdog."""
    global _STALE_WATCHDOG_TASK
    if _STALE_WATCHDOG_TASK is None or _STALE_WATCHDOG_TASK.done():
        _STALE_WATCHDOG_TASK = asyncio.create_task(_stale_turn_watchdog())


# ---------------------------------------------------------------------------
# Crash-resilience: resume interrupted conversations on startup
#
# If the process died mid-turn (panic, kill -9, power loss, whatever), the
# affected conversation rows are left with state='running' and any user
# messages the composer queued are still sitting in `queued_inputs`. This
# startup task finds those rows and fires a fresh `run_turn` for each one
# so the agent picks up where it left off without the user having to re-send.
#
# Safety:
#   - We only auto-resume conversations that look continuable (last visible
#     message is a user message, OR there's at least one queued_inputs row).
#     Conversations whose last message was already an assistant reply get
#     reset to 'idle' without a new turn — nothing to resume.
#   - We drop a system-note into history first so the user knows the turn
#     was interrupted and is being retried.
#   - Each resume runs as its own asyncio.Task so one broken conversation
#     can't stall the others.
# ---------------------------------------------------------------------------
async def _resume_interrupted_conversations() -> None:
    """Scan for conversations left mid-turn and relaunch their run_turn loops."""
    try:
        running = db.list_conversations_by_state("running")
    except Exception as e:
        print(f"[resume] list failed: {e}", file=sys.stderr)
        return
    if not running:
        return
    for conv in running:
        try:
            await _maybe_resume_one(conv)
        except Exception as e:
            print(
                f"[resume] conversation {conv.get('id')!r} resume failed: {e}",
                file=sys.stderr,
            )


# A turn whose most recent activity is older than this is considered stale —
# the process may have died ages ago, the user moved on, or the turn was
# abandoned during development reloads. Auto-resuming at that point just
# revives conversations the user isn't thinking about anymore.
_RESUME_STALENESS_SEC = 120


async def _maybe_resume_one(conv: dict) -> None:
    """Decide what to do with a single interrupted conversation.

    Branches:
      - Last visible message is assistant AND no queued inputs → nothing to
        do; the turn actually completed, we just failed to mark state=idle.
        Flip it to idle and move on.
      - Activity is older than `_RESUME_STALENESS_SEC` → the turn wasn't
        genuinely mid-flight at shutdown (or the user has long moved on).
        Flip to idle without spawning a new turn.
      - Last visible message is user, OR queued inputs exist, AND activity
        is recent → resume with a system-note explaining the interruption.
      - No messages at all → just flip to idle.
    """
    cid = conv["id"]
    messages = db.list_messages(cid)
    has_queued = db.has_queued_inputs(cid)

    # Find the most recent user/assistant row — tool rows don't count for
    # "was the user waiting for a reply".
    last_visible = None
    for m in reversed(messages):
        if m.get("role") in ("user", "assistant"):
            last_visible = m
            break

    needs_resume = has_queued or (last_visible and last_visible["role"] == "user")

    if not needs_resume:
        # Stale 'running' marker with nothing left to do. Reset quietly.
        try:
            db.set_conversation_state(cid, "idle")
        except Exception:
            pass
        return

    # Staleness gate: only resume if the turn was actually in-flight recently.
    # We look at both the last message timestamp AND the conversation's
    # updated_at so a queued-input row or a stale state flip doesn't skew it.
    import time
    now = time.time()
    last_activity = 0.0
    if last_visible:
        last_activity = max(last_activity, float(last_visible.get("created_at") or 0))
    if messages:
        last_activity = max(
            last_activity, float(messages[-1].get("created_at") or 0)
        )
    last_activity = max(last_activity, float(conv.get("updated_at") or 0))

    if last_activity and (now - last_activity) > _RESUME_STALENESS_SEC:
        # Running marker is stale — server was killed / reloaded long after
        # the turn stopped being interesting. Don't auto-resume; the user
        # can re-send if they actually want to continue.
        try:
            db.set_conversation_state(cid, "idle")
        except Exception:
            pass
        return

    # Post a breadcrumb so the user can tell the previous turn was interrupted.
    # Persisting as a system summary keeps it in the model's context too.
    try:
        db.add_system_summary(
            cid,
            "[crash-resilience] The previous run was interrupted "
            "(likely a server restart). Resuming the turn now — any queued "
            "messages you sent just before the crash have been preserved.",
        )
    except Exception:
        pass

    # Spawn the resume as its own task so the startup path doesn't block.
    asyncio.create_task(_drive_resumed_turn(cid))


async def _drive_resumed_turn(cid: str) -> None:
    """Run agent.run_turn() to completion for a resumed conversation.

    Events are discarded — there's no live SSE consumer at server-restart
    time. The work still matters: the model writes its reply to the DB, so
    when the user reopens the conversation they see a completed turn.
    """
    try:
        async for _ in agent.run_turn(cid, user_text=None, persist_user=False):
            pass
    except Exception as e:
        print(f"[resume] turn for {cid!r} crashed again: {e}", file=sys.stderr)


async def _start_resumer() -> None:
    """Fire the interrupted-conversation resumer once after startup settles.

    We delay by a couple seconds so Ollama auto-start has a chance to come
    up first — a resumed turn that hits "model not reachable" is annoying.
    """
    async def _wait_and_resume():
        await asyncio.sleep(3)
        await _resume_interrupted_conversations()
    asyncio.create_task(_wait_and_resume())


async def _auto_start_ollama() -> None:
    """Bring Ollama up automatically if the user hasn't already.

    Best-effort: if Ollama isn't installed we log a hint and move on —
    the existing "Ollama not reachable" toast in the UI is the right place
    for the end-user notice. We do this BEFORE starting MCP so any MCP
    servers that depend on Ollama (embeddings, small helpers) find it up.
    """
    try:
        result = await ollama_runtime.ensure_running()
        if not result.get("ok"):
            print(
                f"[ollama] auto-start skipped: {result.get('reason')}"
                + (f" — {result.get('hint')}" if result.get("hint") else ""),
                file=sys.stderr,
            )
    except Exception as e:
        print(f"[ollama] auto-start crashed: {e}", file=sys.stderr)


async def _auto_tune_ollama_models() -> None:
    """Pick the best Gemma 4 variant for this hardware and pull it + the
    embedding model if they're missing.

    Spawned as a background task so uvicorn finishes startup immediately —
    the multi-GB first-time download streams in the background while the UI
    stays responsive. Progress is readable via /api/system/config.
    """
    asyncio.create_task(ollama_runtime.startup_autotune_background())


async def _start_compute_pool_probe() -> None:
    """Kick off the background liveness/capability sweep for compute workers.

    The first sweep fires immediately so the Settings UI doesn't show stale
    "never seen" state on a fresh boot; the loop then re-runs every 5 min.
    Cheap: two GETs per enabled worker. No-op when no workers registered.

    Also reaps stale SSH ControlMaster sockets from previous runs — a
    hard-killed previous process leaves orphan sockets in /tmp that
    accumulate forever otherwise. Each reaped socket forces one fresh
    handshake on the next dispatch, identical to the no-ControlMaster
    cost; the alternative is the orphans piling up across reboots.
    """
    try:
        compute_pool.reap_stale_ssh_control_sockets()
    except Exception as e:
        print(f"[compute_pool] SSH socket reap failed: {e}", file=sys.stderr)
    try:
        compute_pool.start_periodic_probe()
    except Exception as e:
        print(f"[compute_pool] periodic probe failed to start: {e}", file=sys.stderr)


async def _reconcile_split_models() -> None:
    """Reset stale `running`/`loading` rows on boot.

    If the previous app process was killed mid-flight, llama-server
    children went with it but the DB still says `running`. Without
    reconcile the UI shows phantom green pills and chat routing tries
    to talk to a port nothing is bound to."""
    try:
        n = split_lifecycle.reconcile_on_boot()
        if n:
            print(f"[split] reconciled {n} stale split-model row(s) to stopped", file=sys.stderr)
    except Exception as e:
        print(f"[split] boot reconcile failed: {e}", file=sys.stderr)


async def _start_mcp() -> None:
    """Spawn every enabled MCP server once the event loop is running.

    Failures here never crash the app — a broken MCP server just means its
    tools are missing from this session; the user can re-enable it after
    fixing the config from the settings panel.
    """
    try:
        await mcp.startup()
    except Exception as e:
        print(f"[mcp] startup failed: {e}", file=sys.stderr)


async def _stop_scheduler() -> None:
    """Cancel the scheduler so uvicorn can exit cleanly."""
    global _SCHED_TASK
    if _SCHED_TASK and not _SCHED_TASK.done():
        _SCHED_TASK.cancel()
        try:
            await _SCHED_TASK
        except (asyncio.CancelledError, Exception):
            pass


async def _stop_stale_watchdog() -> None:
    """Cancel the stale-turn watchdog so uvicorn can exit cleanly."""
    global _STALE_WATCHDOG_TASK
    if _STALE_WATCHDOG_TASK and not _STALE_WATCHDOG_TASK.done():
        _STALE_WATCHDOG_TASK.cancel()
        try:
            await _STALE_WATCHDOG_TASK
        except (asyncio.CancelledError, Exception):
            pass


async def _stop_compute_pool_probe() -> None:
    """Cancel the compute-pool sweep so uvicorn exits cleanly and the test
    runner doesn't see a stranded task."""
    try:
        compute_pool.stop_periodic_probe()
    except Exception:
        pass


async def _stop_split_models() -> None:
    """Terminate every running llama-server child on app shutdown so
    GPU memory is reclaimed cleanly and uvicorn doesn't leave orphan
    processes."""
    try:
        await split_lifecycle.stop_all()
    except Exception:
        pass


async def _stop_mcp() -> None:
    """Terminate every MCP subprocess before uvicorn exits."""
    try:
        await mcp.shutdown()
    except Exception:
        pass


async def _close_shared_http_client() -> None:
    """Close the process-wide shared `httpx.AsyncClient`. Idempotent —
    safe to call even if the client was never lazily-created.
    """
    try:
        from . import http_client
        await http_client.aclose_shared_client()
    except Exception:
        pass


class CreateConversation(BaseModel):
    """Body for POST /api/conversations.

    `model` defaults to an empty string; the route handler substitutes the
    resolved default (user preference → auto-tune recommendation → hardcoded
    fallback) so a change in Settings takes effect immediately without
    restarting any client that cached the old default.

    `cwd` defaults to ``None`` so the handler can tell "client didn't
    specify" from "client explicitly passed the project root". An
    unspecified cwd inherits the Gigachat project root.
    """

    title: str = "New chat"
    model: str = ""
    cwd: str | None = None
    auto_approve: bool = False
    # Three-value permission mode: read_only | approve_edits | allow_all.
    # When None, derived from auto_approve for backward compatibility with
    # older API clients that still pass the bool.
    permission_mode: str | None = None
    # Free-text project label for sidebar grouping. None / empty means
    # "ungrouped" (renders under the "No project" section). Normalized
    # server-side to trim whitespace and cap at 80 chars.
    project: str | None = None


class UpdateConversation(BaseModel):
    """Body for PATCH /api/conversations/{id}. All fields optional."""

    title: str | None = None
    model: str | None = None
    cwd: str | None = None
    auto_approve: bool | None = None
    # Three-value permission mode. Validated in db.update_conversation — an
    # unknown string is silently dropped so a hostile body can't smuggle
    # arbitrary text into the column.
    permission_mode: str | None = None
    # Conversation-level pin — pinned conversations float to the top of the
    # sidebar regardless of last-touched time. Independent of message-level
    # pinning (which protects individual rows from auto-compaction).
    pinned: bool | None = None
    # Free-form labels stored as a JSON list. Accepts a list/tuple of
    # strings; empty/whitespace-only entries are dropped at the DB layer.
    tags: list[str] | None = None
    # Free-text persona / system-prompt extension. An empty or
    # whitespace-only string clears the override (stored as NULL); None
    # means "don't touch". Capped at 4 KB in prompts.build_system_prompt
    # so a runaway paste can't blow out the context window.
    persona: str | None = None
    # Soft budgets. 0 clears the cap (stored as NULL); None means "don't
    # touch"; positive ints set a new ceiling. budget_turns counts
    # assistant replies; budget_tokens is a rough char-count estimate that
    # matches the header gauge.
    budget_turns: int | None = None
    budget_tokens: int | None = None
    # Free-text project label for sidebar grouping. Empty/whitespace string
    # clears the grouping (stored as NULL); None means "don't touch". The DB
    # layer normalizes whitespace and caps length at 80 chars.
    project: str | None = None
    # Quality-mode toggle for the chat path. One of:
    #   * "standard"  — single-pass chat (default).
    #   * "refine"    — generate, then critique + revise with the same
    #                   model. ~2× compute, big lift on small models.
    #   * "consensus" — sample N parallel responses with the same model
    #                   then synthesize the best one. ~3-5× compute,
    #                   biggest lift on math / logic.
    #   * "personas"  — fan out across diverse "reasoning style" overlays
    #                   on the same model, then synthesize. MoA-style
    #                   without using a second model. ~4× compute.
    #   * "auto"      — pick refine / consensus / personas (or skip
    #                   entirely) per turn based on a difficulty
    #                   heuristic over the user's prompt. Cheapest mode
    #                   that still gives a measurable lift on most chats.
    # All modes use ONLY the user's selected model — never a different
    # one — so the user's "I picked this model" intent is preserved.
    # Validated at the DB layer (unknown values silently dropped).
    quality_mode: str | None = None


class SendMessage(BaseModel):
    """Body for POST /api/conversations/{id}/messages.

    `images` — optional list of filenames previously returned by /uploads.
    Only filenames (no path components) are accepted; the upload endpoint
    enforces the filesystem shape.
    """

    content: str
    images: list[str] | None = None


class ApprovalDecision(BaseModel):
    """Body for POST /api/conversations/{id}/approve."""

    approval_id: str
    approved: bool


class QuestionAnswer(BaseModel):
    """Body for POST /api/conversations/{id}/answer.

    Resolves an AskUserQuestion pending prompt by carrying which option value
    the user clicked. The frontend validates `value` matches one of the
    emitted options before POSTing; the backend re-validates defensively.
    """

    answer_id: str
    value: str


class SpawnTaskOpen(BaseModel):
    """Body for POST /api/side-tasks/{id}/open.

    Optional — we can spin the new conversation off with the stored prompt
    alone. The client may also override cwd/model here.
    """

    cwd: str | None = None
    model: str | None = None


class LoginBody(BaseModel):
    """Body for POST /api/auth/login. The frontend sends the typed password."""

    password: str


@app.get("/api/auth/status")
def api_auth_status(request: Request) -> dict:
    """Tell the frontend whether it needs to show the login page.

    Returns ``{requires_password, authenticated, host}``. ``host`` is the
    *configured* bind mode (``127.0.0.1`` or ``lan``) so the UI can
    surface a "you're reachable from other LAN devices" indicator
    without probing the network from the client.

    Loopback clients are implicitly authenticated — the middleware waives
    the gate for them, and reporting ``authenticated: false`` here would
    make the frontend show a login page the user can never need (every
    subsequent API call would pass without the cookie anyway).
    """
    cfg = auth.get_config()
    requires = auth.requires_password(cfg)
    host_str = cfg.get("host", "127.0.0.1")
    if not requires:
        return {"requires_password": False, "authenticated": True, "host": host_str}
    client_host = request.client.host if request.client else None
    # Loopback callers (curl on the host, the desktop browser) are always
    # trusted — they're already on the box. LAN clients still have to
    # present a valid session cookie or Bearer token.
    if auth.is_loopback(client_host):
        return {"requires_password": True, "authenticated": True, "host": host_str}
    token = request.cookies.get(auth.SESSION_COOKIE)
    header = request.headers.get("authorization", "")
    if not token and header.lower().startswith("bearer "):
        token = header[7:].strip()
    authed = auth.verify_token(token)
    return {
        "requires_password": True,
        "authenticated": authed,
        "host": host_str,
    }


# Simple in-memory login rate limiter — one global counter (we don't bother
# keying per-IP because LAN mode admits at most a handful of devices and a
# brute-force attempt from any of them is equally interesting). At PBKDF2-
# 200k's ~100 ms per attempt the password check itself already caps
# throughput; this is belt-and-braces against a misbehaving script. The
# count resets on a successful login so the common "wrong password, retype"
# case doesn't lock the real user out.
_LOGIN_FAIL_COUNT = 0
_LOGIN_LOCKED_UNTIL = 0.0
_LOGIN_MAX_FAILS = 10
_LOGIN_LOCKOUT_SEC = 60


@app.post("/api/auth/login")
def api_auth_login(body: LoginBody, response: Response) -> dict:
    """Exchange a password for a signed session cookie.

    A tiny rate limit guards the endpoint: after 10 consecutive failed
    attempts further logins are refused for 60 seconds. The window is
    deliberately short — the threat model is a typo or a curious LAN
    neighbour, not a sustained credential-stuffing campaign.
    """
    global _LOGIN_FAIL_COUNT, _LOGIN_LOCKED_UNTIL
    cfg = auth.get_config()
    if not auth.requires_password(cfg):
        # Server isn't running in auth mode — login is a no-op success so
        # the frontend flow stays uniform.
        return {"ok": True, "authenticated": True}
    now = time.time()
    if now < _LOGIN_LOCKED_UNTIL:
        retry_in = int(_LOGIN_LOCKED_UNTIL - now) + 1
        raise HTTPException(
            429,
            f"too many failed logins; retry in {retry_in}s",
        )
    if not auth.check_password(body.password or ""):
        _LOGIN_FAIL_COUNT += 1
        if _LOGIN_FAIL_COUNT >= _LOGIN_MAX_FAILS:
            _LOGIN_LOCKED_UNTIL = time.time() + _LOGIN_LOCKOUT_SEC
            _LOGIN_FAIL_COUNT = 0
        raise HTTPException(401, "invalid password")
    # Success: reset the counter so a user who typo'd doesn't stay locked.
    _LOGIN_FAIL_COUNT = 0
    _LOGIN_LOCKED_UNTIL = 0.0
    token = auth.make_token()
    # HttpOnly + SameSite=Lax: the cookie is not JS-readable (no XSS to
    # cookie-exfiltration path) and isn't sent on cross-site POSTs (no
    # trivial CSRF). The Secure flag is omitted because LAN mode serves
    # plain HTTP on the local network — the cookie never traverses a
    # public network where it could be sniffed in transit.
    response.set_cookie(
        auth.SESSION_COOKIE,
        token,
        max_age=auth.SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
    )
    return {"ok": True, "authenticated": True, "token": token}


@app.post("/api/auth/logout")
def api_auth_logout(response: Response) -> dict:
    """Clear the session cookie. Always succeeds — idempotent."""
    response.delete_cookie(auth.SESSION_COOKIE, path="/")
    return {"ok": True}


# In-memory cache of "does this model support tool calling?" — keyed by
# model name. Ollama's /api/show returns the Go chat template; we look for
# the `.Tools` action, which is how Ollama itself decides whether to allow
# a `tools` field in /api/chat. Templates only change when a model is re-
# pulled, so a 10-minute TTL is plenty while still eventually refreshing
# if the user rebuilds a model locally with a tool-aware template.
_MODEL_TOOL_CACHE: dict[str, tuple[float, bool]] = {}
_MODEL_TOOL_CACHE_TTL = 600  # seconds


async def _model_supports_tools(client: httpx.AsyncClient, name: str) -> bool:
    """Return True iff this model can drive the Gigachat agent loop.

    Two paths admit a model into the picker:

      1. Ollama's /api/show ``capabilities`` list contains ``tools`` —
         function-calling requests are accepted natively.
      2. The model is in the prompt-space-adapter's known-tool-capable
         allowlist (``tool_prompt_adapter._matches_known_tool_capable``).
         These are model families whose weights were trained with
         function calling but whose Ollama upload happens to ship a
         Modelfile that omits the ``{{ if .Tools }}`` template block —
         so Ollama drops the cap flag, but the model itself can still
         emit JSON tool calls when prompted in prompt-space mode (XML
         tags in prose). Examples: ``dolphin3:*`` (Llama 3.1 base),
         ``ikiru/Dolphin-Mistral-…`` (Mistral 24B base),
         ``llama3.2-vision:11b`` (text decoder supports tools, the
         vision template just doesn't include them), ``deepseek-coder-v2``.

    Falls back to True on /api/show error so a transient hiccup doesn't
    silently hide every model — the worst case is one 400 on the next chat
    call, which now surfaces a clear "does not support tools" message.
    """
    import time as _t
    from . import tool_prompt_adapter
    now = _t.time()
    cached = _MODEL_TOOL_CACHE.get(name)
    if cached and now - cached[0] < _MODEL_TOOL_CACHE_TTL:
        return cached[1]
    try:
        r = await client.post(
            "http://localhost:11434/api/show",
            json={"name": name},
            timeout=5.0,
        )
        r.raise_for_status()
        caps = r.json().get("capabilities") or []
    except Exception:
        # Don't cache failures — we'll retry next time.
        return True
    supports = ("tools" in caps) or tool_prompt_adapter._matches_known_tool_capable(name)
    _MODEL_TOOL_CACHE[name] = (now, supports)
    return supports


@app.get("/api/models")
async def list_models(all: bool = False) -> dict:
    """List installed Ollama models.

    By default only returns models whose chat template supports tool calling
    — pick-by-default is the whole point of the filter, and the agent loop
    is useless without tools anyway. Pass ``?all=1`` to bypass the filter
    (for debugging / curiosity).
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get("http://localhost:11434/api/tags")
            r.raise_for_status()
            data = r.json()
            names = [m["name"] for m in data.get("models", [])]
            if all:
                return {"models": names}
            # Probe each model in parallel — the first call for a fresh set
            # of models pays N×~30 ms; subsequent calls hit the TTL cache
            # and are effectively free.
            results = await asyncio.gather(
                *[_model_supports_tools(c, n) for n in names],
                return_exceptions=True,
            )
            filtered = [
                n
                for n, ok in zip(names, results)
                if isinstance(ok, bool) and ok
            ]
        # Phase 2 split path is invisible from the picker — the user
        # picks any normal Ollama model and the auto-router
        # (compute_pool.route_chat_for) decides at run-turn time
        # whether to use Ollama or to spawn llama-server with --rpc.
        # No `split:<label>` prefixes here.
        return {"models": filtered}
    except Exception as e:
        return {"models": [], "error": str(e)}


@app.get("/api/models/all-sources")
async def list_models_all_sources(tools_only: bool = True) -> dict:
    """Aggregated model inventory across local + LAN + public pool.

    Response shape:
      {
        "local":  [{"name", "family", "size_bytes"}, ...],
        "lan":    [{"name", "source_device_id", "source_label",
                    "size_bytes", "family"}, ...],
        "public": [{"name", "source_device_id", "source_label", ...}, ...],
        "public_pool_enabled": bool,
      }

    LAN models come from each paired peer's last probe (cached in
    `compute_workers.capabilities_json.models` — populated by the
    periodic /api/tags probe). No live network calls here, so the
    endpoint is fast.

    Public-pool models come from the rendezvous lookup when the
    Public Pool toggle is on. If the rendezvous is unreachable or
    not configured, the `public` array is empty + `error` set.

    `tools_only` (default True) filters local models to just those
    whose Ollama template declares tool-calling support — same logic
    as `/api/models`. LAN/Public models aren't filtered because we
    don't have probe data on their templates yet.
    """
    out: dict[str, Any] = {
        "local": [],
        "lan": [],
        "public": [],
        "public_pool_enabled": False,
    }
    # Local
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get("http://localhost:11434/api/tags")
            r.raise_for_status()
            data = r.json()
            local_raw = [m for m in (data.get("models") or [])]
            if tools_only:
                results = await asyncio.gather(
                    *[_model_supports_tools(c, m["name"]) for m in local_raw],
                    return_exceptions=True,
                )
                local_raw = [
                    m for m, ok in zip(local_raw, results)
                    if isinstance(ok, bool) and ok
                ]
            for m in local_raw:
                details = m.get("details") or {}
                out["local"].append({
                    "name": m.get("name"),
                    "family": details.get("family"),
                    "parameter_size": details.get("parameter_size"),
                    "quantization_level": details.get("quantization_level"),
                    "size_bytes": m.get("size") or 0,
                })
    except Exception as e:
        out["local_error"] = str(e)

    # LAN — read from paired-peer capabilities_json (already
    # populated by the periodic worker probe). No network call here.
    try:
        for w in db.list_compute_workers(enabled_only=True):
            if not w.get("gigachat_device_id"):
                continue  # manually-added (non-paired) worker, skip
            caps = w.get("capabilities") or {}
            for m in caps.get("models") or []:
                if not m.get("name"):
                    continue
                out["lan"].append({
                    "name": m["name"],
                    "source_device_id": w.get("gigachat_device_id"),
                    "source_label": w.get("label"),
                    "family": m.get("family"),
                    "parameter_size": m.get("parameter_size"),
                    "quantization_level": m.get("quantization_level"),
                    "size_bytes": m.get("size") or 0,
                    "encrypted": bool(w.get("use_encrypted_proxy")),
                })
    except Exception as e:
        out["lan_error"] = str(e)

    # Public pool — only when toggled on. Reads from the LOCAL
    # inventory cache (`p2p_pool_inventory`) which is populated by
    # direct peer-to-peer queries of each peer's /api/tags via the
    # encrypted secure proxy. We do NOT call the rendezvous from
    # here — model inventory is fully P2P. The rendezvous's only
    # job is the bootstrap "who's online" list, polled by the
    # inventory loop on its own cadence.
    pp_val = db.get_setting("p2p_public_pool_enabled")
    public_enabled = (
        pp_val is None
        or str(pp_val).lower() in ("1", "true", "yes", "on")
    )
    out["public_pool_enabled"] = public_enabled
    if public_enabled:
        try:
            from . import p2p_pool_inventory as _inv
            # Force a refresh if the cache is stale — opening the
            # picker should show fresh data even when the periodic
            # loop hasn't fired in a while. ensure_fresh is a no-op
            # when the cache is already current, so this is cheap
            # in the common case.
            await _inv.ensure_fresh(max_age_sec=120.0)
            for entry in _inv.list_all_models():
                out["public"].append(entry)
        except Exception as e:
            out["public_error"] = str(e)

    return out


@app.get("/api/models/{name:path}/capabilities")
async def api_model_capabilities(name: str) -> dict:
    """Return the Ollama capabilities list for one installed model.

    Surfaces the raw `capabilities` array from /api/show — the frontend
    uses this to decide whether the current chat model can consume image
    attachments (`"vision"`), tools (`"tools"`), etc. Models without the
    needed capability still run; the UI just surfaces a clear warning
    before the user commits.

    Returns a small shape that doesn't leak the full /api/show payload
    — caps + a best-effort boolean per capability the UI cares about.
    """
    n = (name or "").strip()
    if not n:
        raise HTTPException(400, "model name required")
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.post(
                "http://localhost:11434/api/show",
                json={"name": n},
                timeout=5.0,
            )
            r.raise_for_status()
            caps = r.json().get("capabilities") or []
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(404, f"model {n!r} not found")
        raise HTTPException(502, f"ollama error: {e}")
    except Exception as e:
        raise HTTPException(502, f"ollama unreachable: {e}")
    caps_l = [c.lower() for c in caps if isinstance(c, str)]
    return {
        "model": n,
        "capabilities": caps,
        "vision": "vision" in caps_l,
        "tools": "tools" in caps_l,
    }


@app.get("/api/system/config")
def api_system_config() -> dict:
    """Return the auto-detected host profile + the chosen context window.

    The frontend's TokenUsage meter fetches this so its token-budget
    denominator matches whatever the backend is actually feeding Ollama —
    avoids the old bug where bumping NUM_CTX on the server silently left
    the UI showing the wrong percentage.

    Also surfaces the auto-tuner's recommendation so the settings UI can
    show "we picked gemma4:e4b for your hardware" and render a live pull
    progress indicator while the first-boot download runs.
    """
    info = sysdetect.detect_system()
    rec = ollama_runtime.get_recommendation()
    user_default = db.get_setting("default_chat_model")
    return {
        **info,
        "num_ctx": agent.NUM_CTX,
        "ollama_url": agent.OLLAMA_URL,
        "recommended_chat_model": rec.get("chat_model"),
        "recommended_embed_model": rec.get("embed_model"),
        "model_pulling": bool(rec.get("pulling")),
        "pull_status": rec.get("pull_status") or "",
        "pull_error": rec.get("pull_error") or "",
        "default_chat_model": (
            user_default if isinstance(user_default, str) and user_default else None
        ),
        "effective_chat_model": _resolve_default_chat_model(),
    }


@app.post("/api/fs/pick-directory")
async def api_pick_directory() -> dict:
    """Open a native OS "browse for folder" dialog and return the chosen path.

    Gigachat runs as a localhost app — frontend, backend, and user are all on
    the same machine — so popping a native Tk dialog on the server IS a
    native dialog from the user's perspective. This lets the workspace-folder
    picker in ChatHeader offer a real Browse… button instead of forcing the
    user to paste a path.

    Security: the endpoint takes no input and returns only the user's chosen
    path — the backend is not exposing any additional filesystem state. The
    dialog runs on a worker thread so the event loop isn't blocked while the
    user is choosing, and the existing CORS / localhost posture of the rest
    of the API applies. If the user cancels, we return path=None.

    We use Python's bundled Tkinter (no extra dependency). On a truly
    headless machine Tk will fail to initialise; in that case we fall back
    to returning {"ok": false, "error": ...} so the frontend can keep its
    text input as the primary path-entry affordance.
    """
    def _show_dialog() -> dict:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as e:
            return {"ok": False, "error": f"tkinter unavailable: {e}"}
        try:
            root = tk.Tk()
            try:
                root.withdraw()
                # Bring the dialog to the front — without this, Windows
                # sometimes hides it behind the browser.
                root.attributes("-topmost", True)
                path = filedialog.askdirectory(
                    title="Pick working directory for this conversation",
                    mustexist=True,
                )
            finally:
                try:
                    root.destroy()
                except Exception:
                    pass
            # askdirectory returns "" when the user cancels.
            return {"ok": True, "path": path or None}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    return await asyncio.to_thread(_show_dialog)


@app.get("/api/fs/default-cwd")
def api_default_cwd() -> dict:
    """Return the directory Gigachat itself is running from.

    Used by the Sidebar's new-chat dialog as the default workspace — if the
    user doesn't explicitly pick one, "the folder the app lives in" is a
    sensible zero-config choice that matches where their docs/notes are
    likely to be when they're tinkering with Gigachat itself. The frontend
    shows this as the pre-filled value; the user can overwrite or Browse for
    a different path before creating the chat.
    """
    return {"cwd": str(ROOT)}


@app.get("/api/conversations")
def api_list(
    limit: int | None = None,
    offset: int = 0,
) -> dict:
    """List conversations.

    Backward-compatible: no params → return everything (legacy shape).
    With ``limit`` set, returns one page plus a ``total`` count so
    the sidebar can render "showing 50 of 312" and offer "load more".

    The cap is [1, 500] — even paginated, more than 500 rows in one
    shot becomes a noticeable JSON marshal hitch on the threadpool.
    """
    if limit is not None:
        capped = max(1, min(int(limit), 500))
        convs, total = db.list_conversations_paginated(
            limit=capped, offset=max(0, int(offset)),
        )
        return {
            "conversations": convs,
            "total": total,
            "has_more": (max(0, int(offset)) + len(convs)) < total,
        }
    return {"conversations": db.list_conversations()}


@app.get("/api/conversations/search")
def api_search_conversations(q: str = "") -> dict:
    """Substring-search across title, tags, and message content.

    Empty query returns an empty list (so an empty search box doesn't
    accidentally re-render the whole sidebar). Frontend should fall back to
    `/api/conversations` for the default unfiltered view.
    """
    return {"conversations": db.search_conversations(q)}


# Lower threshold than the internal RAG (agent.RECALL_MIN_SCORE = 0.45) because
# the UI wants to surface MORE candidates for human inspection, not fewer.
# Users can scroll past weak matches; they can't un-miss a hidden one.
SEMANTIC_SEARCH_MIN_SCORE = 0.35
SEMANTIC_SEARCH_MAX_HITS = 30


@app.get("/api/conversations/semantic-search")
async def api_semantic_search(q: str = "") -> dict:
    """Cross-conversation semantic search over embedded messages.

    Embeds the query via the same local model used for RAG, dot-products
    against every stored message vector, and returns the top hits with their
    conversation title + a content snippet so the sidebar can render a list
    the user can click through to jump straight to the source message.
    """
    q = (q or "").strip()
    if not q:
        return {"hits": [], "indexed": 0, "total": 0}

    q_vec = await agent._embed_text(q)
    if not q_vec:
        # Embedding model isn't installed or Ollama is offline. Degrade
        # gracefully — the UI will show a toast explaining what to do.
        embedded, total = db.count_embedded_vs_total()
        return {
            "hits": [],
            "indexed": embedded,
            "total": total,
            "error": "embedding_unavailable",
        }

    rows = db.list_all_embeddings()
    scored: list[tuple[str, str, float]] = []
    for mid, cid, vec in rows:
        if len(vec) != len(q_vec):
            continue
        score = agent._dot(q_vec, vec)
        if score >= SEMANTIC_SEARCH_MIN_SCORE:
            scored.append((mid, cid, score))

    scored.sort(key=lambda t: t[2], reverse=True)
    scored = scored[:SEMANTIC_SEARCH_MAX_HITS]

    if not scored:
        embedded, total = db.count_embedded_vs_total()
        return {"hits": [], "indexed": embedded, "total": total}

    # Fetch the actual message bodies + conversation titles in one shot.
    msg_ids = [mid for mid, _, _ in scored]
    messages = {m["id"]: m for m in db.get_messages_by_ids(msg_ids)}
    conv_titles: dict[str, str] = {}
    for _, cid, _ in scored:
        if cid in conv_titles:
            continue
        conv = db.get_conversation(cid)
        conv_titles[cid] = conv["title"] if conv else "(deleted)"

    hits = []
    for mid, cid, score in scored:
        m = messages.get(mid)
        if not m:
            continue
        snippet = (m.get("content") or "").strip().replace("\n", " ")
        if len(snippet) > 280:
            snippet = snippet[:280] + "…"
        hits.append(
            {
                "message_id": mid,
                "conversation_id": cid,
                "conversation_title": conv_titles.get(cid, ""),
                "role": m.get("role", ""),
                "snippet": snippet,
                "score": round(float(score), 4),
                "created_at": m.get("created_at"),
            }
        )

    embedded, total = db.count_embedded_vs_total()
    return {"hits": hits, "indexed": embedded, "total": total}


@app.post("/api/conversations/reindex")
async def api_reindex_embeddings() -> dict:
    """Backfill embeddings for any user/assistant messages that don't have one.

    Safe to call repeatedly — rows already in `message_embeddings` are
    skipped. Processes at most 500 messages per call so a huge backlog
    doesn't hold the event loop hostage; the user can click again.
    """
    pending = db.list_unembedded_messages(limit=500)
    indexed = 0
    for m in pending:
        vec = await agent._embed_text(m["content"] or "")
        if vec:
            db.save_embedding(m["id"], m["conversation_id"], vec)
            indexed += 1
    embedded, total = db.count_embedded_vs_total()
    return {
        "indexed_now": indexed,
        "indexed": embedded,
        "total": total,
        "pending": max(0, total - embedded),
    }


@app.post("/api/conversations")
def api_create(body: CreateConversation) -> dict:
    # Empty model means "let the server pick" — resolve against the user's
    # saved default first, then the auto-tuner's recommendation, then the
    # hardcoded fallback. Explicit model passed by the client wins.
    model = (body.model or "").strip() or _resolve_default_chat_model()
    # cwd is mandatory — the UI prompts the user for a workspace on New
    # Chat and there's no way to change it later, so we refuse to create
    # a chat without one. Public-mode callers that want the isolated
    # per-conversation workspace should POST /api/fs/default-workspace
    # first to reserve a path. Validate the directory exists so a typo
    # doesn't bake in a broken path the user can never fix.
    raw_cwd = (body.cwd or "").strip()
    if not raw_cwd:
        raise HTTPException(400, "cwd is required — pick a working directory for the chat")
    try:
        resolved = Path(raw_cwd).expanduser().resolve()
    except Exception:
        raise HTTPException(400, "invalid cwd")
    if not resolved.exists() or not resolved.is_dir():
        raise HTTPException(400, f"cwd does not exist: {resolved}")
    cwd = str(resolved)
    conv = db.create_conversation(
        title=body.title,
        model=model,
        cwd=cwd,
        auto_approve=body.auto_approve,
        permission_mode=body.permission_mode,
        project=body.project,
    )
    # Kick off a background index of the cwd so codebase_search is ready by
    # the time the first turn runs. No-op if already indexed / in flight.
    _kick_codebase_index(conv.get("cwd"))
    return {"conversation": conv}


@app.get("/api/conversations/{cid}")
def api_get(
    cid: str,
    limit: int | None = None,
    before_id: str | None = None,
) -> dict:
    """Return one conversation plus its messages.

    ``limit`` and ``before_id`` are optional pagination parameters:

      * No query params (legacy behaviour) → return EVERY message
        oldest-first. Fast for typical chats; slow for ones that
        grew to thousands of messages.
      * ``?limit=200`` → return only the most-recent 200 messages
        plus a `total` field so the UI can render
        "showing N of TOTAL". The page is still oldest-first
        within itself.
      * ``?limit=200&before_id=<msg_id>`` → "load more" path —
        returns the 200 messages immediately preceding ``before_id``.
        Frontend prepends to its existing list.

    Backward-compatible: clients that don't pass `limit` see the
    same shape they always did.
    """
    conv = db.get_conversation(cid)
    if not conv:
        raise HTTPException(404, "not found")
    if limit is not None:
        capped = max(1, min(int(limit), 1000))
        messages, total = db.list_messages_paginated(
            cid, limit=capped, before_id=before_id,
        )
        return {
            "conversation": conv,
            "messages": messages,
            "total": total,
            "has_more": len(messages) < total and (
                before_id is None or bool(messages)
            ),
        }
    return {"conversation": conv, "messages": db.list_messages(cid)}


@app.patch("/api/conversations/{cid}")
def api_update(cid: str, body: UpdateConversation) -> dict:
    updates: dict[str, Any] = {k: v for k, v in body.model_dump().items() if v is not None}
    # cwd is immutable after creation — every tool call, checkpoint, and
    # codebase-index row is keyed by it, so a mid-conversation swap would
    # leave stale chunks in the index and orphan existing checkpoints.
    # Reject explicitly so an old API client sees the policy rather than
    # silently losing the field.
    if "cwd" in updates:
        raise HTTPException(400, "cwd is fixed for the life of a conversation")
    conv = db.update_conversation(cid, **updates)
    if not conv:
        raise HTTPException(404, "not found")
    return {"conversation": conv}


@app.delete("/api/conversations/{cid}")
def api_delete(cid: str) -> dict:
    db.delete_conversation(cid)
    # Drop any per-conversation in-memory state the tools module holds so
    # a new conversation reusing the same id (unlikely, uuids are random)
    # wouldn't inherit stale file-read tracking. This is tiny but avoids
    # a slow memory leak over many deletes.
    tools.clear_read_state_for_conversation(cid)
    # Drop the worker-affinity entry for this conversation so the
    # in-memory map doesn't grow forever across many deletes.
    try:
        compute_pool.forget_conv_affinity(cid)
    except Exception:
        pass
    # Drop agent-loop process-local state (failure counters, stop flags,
    # subagent buses) so long-lived backends don't accumulate one entry
    # per ever-deleted conversation.
    try:
        agent.forget_conv_state(cid)
    except Exception:
        pass
    return {"ok": True}


class PinMessage(BaseModel):
    """Body for PATCH /api/conversations/{cid}/messages/{mid}."""

    pinned: bool


@app.patch("/api/conversations/{cid}/messages/{mid}")
def api_pin_message(cid: str, mid: str, body: PinMessage) -> dict:
    """Toggle the `pinned` flag on a single message.

    Pinned messages survive the auto-compactor — use this when the user
    marks something "do not forget" in the UI. The route is idempotent.
    """
    if not db.get_conversation(cid):
        raise HTTPException(404, "conversation not found")
    msg = db.set_message_pinned(mid, body.pinned)
    if not msg or msg["conversation_id"] != cid:
        raise HTTPException(404, "message not found")
    return {"message": msg}


@app.delete("/api/conversations/{cid}/messages/{mid}")
def api_delete_message(cid: str, mid: str) -> dict:
    """Permanently delete one message from a conversation.

    Scoped to the given `cid` so a malicious client cannot delete a
    neighbour's messages by guessing ids. 404 if the message isn't there
    (or belongs to a different conversation) so the UI can distinguish
    "already gone" from "never existed".
    """
    if not db.get_conversation(cid):
        raise HTTPException(404, "conversation not found")
    if not db.delete_message(cid, mid):
        raise HTTPException(404, "message not found")
    return {"ok": True}


@app.get("/api/conversations/{cid}/pinned")
def api_list_pinned(cid: str) -> dict:
    """Return every pinned message in one conversation (oldest-first).

    Separate endpoint from the full messages list so the ChatHeader →
    Pinned viewer can refresh on demand without re-fetching the entire
    transcript when a long conversation is open.
    """
    if not db.get_conversation(cid):
        raise HTTPException(404, "conversation not found")
    return {"messages": db.list_pinned_messages(cid)}


# ---------------------------------------------------------------------------
# Per-conversation memory file
#
# Each conversation can have an optional markdown memory file at
# data/memory/<conv_id>.md. The agent writes to it via the `remember` tool;
# the user can view / edit / clear it from the ChatHeader → Memory dialog.
#
# We deliberately cap writes at MEMORY_MAX_CHARS (the same limit the
# `remember` tool enforces) to keep the system-prompt injection bounded.
# ---------------------------------------------------------------------------
class ConversationMemoryBody(BaseModel):
    """Body for PUT /api/conversations/{cid}/memory."""

    content: str


@app.get("/api/conversations/{cid}/memory")
def api_get_conv_memory(cid: str) -> dict:
    """Return the raw contents of the per-conversation memory file.

    Empty string when the file doesn't exist yet — that's a valid "no
    notes" state, not an error.
    """
    if not db.get_conversation(cid):
        raise HTTPException(404, "conversation not found")
    path = tools.MEMORY_DIR / f"{cid}.md"
    try:
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
    except Exception as e:
        raise HTTPException(500, f"could not read memory file: {e}")
    return {"content": text, "max_chars": tools.MEMORY_MAX_CHARS}


@app.put("/api/conversations/{cid}/memory")
def api_put_conv_memory(cid: str, body: ConversationMemoryBody) -> dict:
    """Overwrite the per-conversation memory file.

    Writes are capped at `tools.MEMORY_MAX_CHARS`; anything longer is
    rejected with 400 so the user sees a clear error rather than silently
    losing the tail. Directory is created on demand so a fresh install
    doesn't need a bootstrap step.
    """
    if not db.get_conversation(cid):
        raise HTTPException(404, "conversation not found")
    text = body.content or ""
    if len(text) > tools.MEMORY_MAX_CHARS:
        raise HTTPException(
            400,
            f"memory content exceeds {tools.MEMORY_MAX_CHARS} chars "
            f"(got {len(text)})",
        )
    try:
        tools.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        # Bytes, not text — Windows text mode translates `\n` → `\r\n` and
        # we want memory files byte-stable across read/write round-trips.
        (tools.MEMORY_DIR / f"{cid}.md").write_bytes(text.encode("utf-8"))
    except Exception as e:
        raise HTTPException(500, f"could not write memory file: {e}")
    return {"ok": True, "content": text}


@app.delete("/api/conversations/{cid}/memory")
def api_delete_conv_memory(cid: str) -> dict:
    """Clear the per-conversation memory file by deleting it.

    Idempotent — deleting a file that doesn't exist is a success case
    (the end-state ‘no memory’ matches what the caller wanted).
    """
    if not db.get_conversation(cid):
        raise HTTPException(404, "conversation not found")
    path = tools.MEMORY_DIR / f"{cid}.md"
    try:
        if path.is_file():
            path.unlink()
    except Exception as e:
        raise HTTPException(500, f"could not delete memory file: {e}")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Per-conversation usage / budget snapshot
# ---------------------------------------------------------------------------
@app.get("/api/conversations/{cid}/usage")
def api_conv_usage(cid: str) -> dict:
    """Return a cumulative-usage snapshot for the budget gauge.

    Fields:
      - assistant_turns: completed assistant replies so far.
      - content_chars: sum of `content` lengths across every row.
      - tokens_estimate: content_chars / CHARS_PER_TOKEN (matches the
        header gauge's rough proxy).
      - budget_turns / budget_tokens: the saved caps, or null when
        unbounded. Echoed so the frontend can render the progress ring
        without fetching the conversation meta separately.
    """
    conv = db.get_conversation(cid)
    if not conv:
        raise HTTPException(404, "conversation not found")
    content_chars = db.conversation_content_chars(cid)
    return {
        "assistant_turns": db.count_assistant_turns(cid),
        "content_chars": content_chars,
        "tokens_estimate": content_chars // max(1, agent.CHARS_PER_TOKEN),
        "budget_turns": conv.get("budget_turns"),
        "budget_tokens": conv.get("budget_tokens"),
    }


def _sse(event: dict) -> str:
    """Render a dict as one SSE `data:` record.

    Hot path — fires on every event yielded out of `agent.run_turn`,
    which is dozens-to-hundreds per chat turn (token deltas, thinking
    blocks, tool events). Routed through jsonutil so orjson speeds
    up the per-event encode by 3-5x when installed.
    """
    from . import jsonutil as _ju
    return f"data: {_ju.dumps(event)}\n\n"


def _safe_upload_name(name: str) -> str | None:
    """Strip any path components and verify the filename shape.

    Returns the basename if valid, or None if it looks traversy / empty.
    """
    base = Path(name).name
    if not base or base in {".", ".."}:
        return None
    # Only accept filenames we would have produced ourselves — uuid hex +
    # one of the allowed suffixes. This makes a traversal attack impossible.
    stem, _, suffix = base.rpartition(".")
    if not stem or not suffix:
        return None
    try:
        int(stem, 16)
    except ValueError:
        return None
    if f".{suffix.lower()}" not in ALLOWED_UPLOAD_TYPES.values():
        return None
    return base


def _filter_safe_images(raw_names: list[str] | None) -> list[str]:
    """Return only the upload filenames that pass our shape + traversal checks
    AND actually exist inside `tools.UPLOAD_DIR`.

    Centralized so every endpoint that accepts an `images` list (send, queue,
    edit-and-regenerate) gets identical validation. Skips silently rather
    than raising — a malformed/missing entry should drop the attachment, not
    fail the whole request.
    """
    safe: list[str] = []
    for raw in raw_names or []:
        name = _safe_upload_name(str(raw))
        if not name:
            continue
        path = (tools.UPLOAD_DIR / name).resolve()
        try:
            path.relative_to(tools.UPLOAD_DIR.resolve())
        except ValueError:
            continue
        if path.is_file():
            safe.append(name)
    return safe


@app.post("/api/conversations/{cid}/messages")
async def api_send(cid: str, body: SendMessage):
    """Open an SSE stream and run one agent turn.

    Validates the optional `images` list against the upload directory so a
    malicious client can't trick the backend into reading arbitrary files.
    """
    conv = db.get_conversation(cid)
    if not conv:
        raise HTTPException(404, "not found")

    safe_images = _filter_safe_images(body.images)

    async def gen():
        try:
            async for evt in agent.run_turn(
                cid,
                user_text=body.content,
                user_images=safe_images or None,
            ):
                yield _sse(evt)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/conversations/{cid}/stop")
def api_stop_turn(cid: str) -> dict:
    """Signal the active agent turn for this conversation to stop.

    Sets an in-memory flag that the agent loop polls at well-defined
    checkpoints (between Ollama stream chunks, before tool dispatch, at
    the top of each iteration). Aborting the client's SSE fetch alone
    isn't enough: if the server is mid-round-trip to Ollama when the
    disconnect arrives, the local model keeps generating until it finishes
    — which makes the Stop button feel broken. This endpoint short-circuits
    that by making the loop notice and exit cleanly.
    """
    if not db.get_conversation(cid):
        raise HTTPException(404, "not found")
    agent.request_stop(cid)
    return {"ok": True, "stopped": True}


@app.post("/api/conversations/{cid}/queue")
def api_queue_input(cid: str, body: SendMessage) -> dict:
    """Append a follow-up user message to the in-flight turn's input queue.

    The frontend hits this when the user sends another message while the
    agent is still working on the previous one — the active `run_turn` loop
    will pick it up between iterations and persist it as a real user row,
    yielding a `user_message_added` SSE event so the transcript stays in
    sync. No new SSE stream is opened — the existing one keeps the events.
    """
    if not db.get_conversation(cid):
        raise HTTPException(404, "not found")
    safe_images = _filter_safe_images(body.images)
    queued = agent.enqueue_user_input(cid, body.content or "", safe_images or None)
    if not queued:
        # Empty payload (no text, no usable images). Treat as a 400 so the
        # frontend can surface a toast rather than silently dropping it.
        raise HTTPException(400, "empty message — nothing to queue")
    return {"ok": True, "queued": True}


@app.post("/api/conversations/{cid}/messages/{mid}/edit")
async def api_edit_and_regenerate(cid: str, mid: str, body: SendMessage):
    """Rewrite a user message in place, drop everything that came after it,
    then run a fresh agent turn against the corrected history.

    Only user-role messages are editable (enforced at the DB layer). The
    response is an SSE stream identical in shape to /messages so the
    frontend can reuse the same event handler.

    Note: deletes are intentionally cascade-style — assistant replies, tool
    calls, and tool results that depended on the now-rewritten prompt no
    longer make sense, so we throw them away rather than leave the model
    looking at contradictory history.
    """
    if not db.get_conversation(cid):
        raise HTTPException(404, "not found")
    safe_images = _filter_safe_images(body.images)

    updated = db.update_user_message_content(mid, body.content or "")
    if not updated or updated["conversation_id"] != cid:
        raise HTTPException(404, "user message not found")
    db.delete_messages_after(cid, mid)

    async def gen():
        try:
            # persist_user=False because the user row already exists (we
            # just rewrote it). Passing the text along still triggers the
            # user_prompt_submit lifecycle hook so workflow integrations
            # behave the same as a fresh send.
            async for evt in agent.run_turn(
                cid,
                user_text=body.content or "",
                user_images=safe_images or None,
                persist_user=False,
            ):
                yield _sse(evt)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/conversations/{cid}/approve")
def api_approve(cid: str, body: ApprovalDecision) -> dict:
    """Resolve a pending tool-call approval. The SSE stream resumes on its own."""
    ok = agent.submit_approval_decision(body.approval_id, body.approved)
    return {"ok": ok}


@app.post("/api/conversations/{cid}/answer")
def api_answer(cid: str, body: QuestionAnswer) -> dict:
    """Resolve a pending AskUserQuestion by relaying the clicked option value.

    The agent loop yielded an `await_user_answer` event and is awaiting the
    future stored in `_pending_answers[body.answer_id]`. Calling
    `agent.resolve_answer` sets that future to `body.value`, which returns
    control to the tool-call loop with `{"ok": True, "output": "user chose: …"}`.
    """
    val = (body.value or "").strip()
    if not val:
        raise HTTPException(400, "value required")
    if len(val) > 500:
        raise HTTPException(400, "value too long")
    ok = agent.resolve_answer(body.answer_id, val)
    return {"ok": ok}


@app.post("/api/conversations/{cid}/execute-plan")
def api_execute_plan(cid: str) -> dict:
    """Switch a plan-mode conversation out of plan mode and enqueue the plan
    as the next user turn for execution.

    Flow:
      1. Must be in `permission_mode = 'plan'` — reject with 409 otherwise so
         the client knows the user double-clicked.
      2. Grab the last assistant message (it contains the plan that ended
         with `[PLAN READY]`).
      3. Flip the conversation to `approve_edits` so writes resume the
         normal approval flow.
      4. Insert the plan as a queued_inputs row — next /send call will pick
         it up as the user turn, exactly as if the user had pasted it.
    """
    conv = db.get_conversation(cid)
    if not conv:
        raise HTTPException(404, "not found")
    if (conv.get("permission_mode") or "") != "plan":
        raise HTTPException(409, "conversation is not in plan mode")
    msgs = db.list_messages(cid)
    last_assistant = next(
        (m for m in reversed(msgs) if m["role"] == "assistant" and (m.get("content") or "").strip()),
        None,
    )
    if not last_assistant:
        raise HTTPException(409, "no plan message to execute — ask the agent to produce a plan first")
    plan_body = (last_assistant["content"] or "").strip()
    # Strip the sentinel so the replay reads as a clean instruction.
    if plan_body.endswith("[PLAN READY]"):
        plan_body = plan_body[: -len("[PLAN READY]")].rstrip()
    replay = (
        "Execute the plan above. You are no longer in plan mode — writes are "
        "allowed (with approval where configured). Follow the plan step by "
        "step, stop and ask if you need to deviate.\n\n"
        "--- PLAN TO EXECUTE ---\n\n"
        f"{plan_body}"
    )
    db.update_conversation(cid, permission_mode="approve_edits")
    db.enqueue_user_input(cid, replay, None)
    # Return the updated conversation row so the UI can flip the header
    # permission-mode badge without a follow-up GET.
    return {"ok": True, "queued": True, "conversation": db.get_conversation(cid)}


# ---------------------------------------------------------------------------
# Side tasks — surfaced as chips under the streaming assistant message.
# ---------------------------------------------------------------------------
@app.get("/api/conversations/{cid}/side-tasks")
def api_list_side_tasks(cid: str) -> dict:
    """Return pending side tasks for this conversation — powers the chip row."""
    conv = db.get_conversation(cid)
    if not conv:
        raise HTTPException(404, "not found")
    return {"side_tasks": db.list_pending_side_tasks(cid)}


@app.post("/api/side-tasks/{sid}/open")
def api_open_side_task(sid: str, body: SpawnTaskOpen) -> dict:
    """User clicked Open on a side-task chip. Spin a new conversation with
    the stored prompt, mark the row as opened, and return the new id so the
    client can navigate there.
    """
    row = db.get_side_task(sid)
    if not row:
        raise HTTPException(404, "not found")
    if row["status"] != "pending":
        raise HTTPException(409, f"side task already {row['status']}")
    source_conv = db.get_conversation(row["source_conversation_id"])
    cwd = (body.cwd or (source_conv or {}).get("cwd") or str(Path.cwd()))
    model = (body.model or (source_conv or {}).get("model") or _resolve_default_chat_model())
    new_conv = db.create_conversation(
        title=row["title"][:120],
        model=model,
        cwd=cwd,
    )
    db.mark_side_task_opened(sid, new_conv["id"])
    db.enqueue_user_input(new_conv["id"], row["prompt"], None)
    # Return the full conversation row so the UI can navigate + update the
    # sidebar in one round-trip without a follow-up GET.
    return {"ok": True, "conversation": db.get_conversation(new_conv["id"])}


@app.post("/api/side-tasks/{sid}/dismiss")
def api_dismiss_side_task(sid: str) -> dict:
    """User clicked Dismiss. Transition the row to the terminal 'dismissed' state."""
    row = db.get_side_task(sid)
    if not row:
        raise HTTPException(404, "not found")
    updated = db.mark_side_task_dismissed(sid)
    if not updated:
        raise HTTPException(409, f"side task already {row['status']}")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Git worktrees — read-only listing; create/remove happens via agent tools.
# ---------------------------------------------------------------------------
@app.get("/api/conversations/{cid}/worktrees")
def api_list_worktrees(cid: str) -> dict:
    """List worktrees (active + removed, newest first) created in this conversation."""
    conv = db.get_conversation(cid)
    if not conv:
        raise HTTPException(404, "not found")
    return {"worktrees": db.list_worktrees_for_conversation(cid)}


@app.post("/api/conversations/{cid}/uploads")
async def api_upload(cid: str, file: UploadFile = File(...)) -> dict:
    """Accept a user-pasted image OR document for the next chat turn.

    Two flavours of upload are supported:

    * **Images** (png/jpeg/webp/gif) — saved verbatim and returned as a
      filename the client passes in the next message's `images` array. The
      agent loop attaches them as multimodal input if the model supports it.
    * **Documents** (pdf/txt/md/csv) — we extract UTF-8 text on the server
      and return it alongside the filename. The UI prepends that text to
      the user's next message so it reaches any model, even text-only ones.
      The original file is still stored under `data/uploads/` for
      auditability but the model never sees the binary.

    Security:
      - Content-Type must be in the image *or* document allowlist.
      - Hard size cap enforced mid-stream so a gigabyte paste can't DOS us.
      - The saved filename is `<random-hex><ext>`; the original client-
        supplied filename is ignored entirely to avoid path traversal.
    """
    if not db.get_conversation(cid):
        raise HTTPException(404, "not found")

    ctype = (file.content_type or "").lower()
    is_image = ctype in ALLOWED_UPLOAD_TYPES
    is_document = ctype in ALLOWED_DOCUMENT_TYPES
    if not (is_image or is_document):
        raise HTTPException(415, f"unsupported content type: {ctype!r}")

    tools.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ALLOWED_UPLOAD_TYPES.get(ctype) or ALLOWED_DOCUMENT_TYPES[ctype]
    # 16 bytes of hex = 32 chars, comfortably unguessable.
    name = secrets.token_hex(16) + suffix
    dest = tools.UPLOAD_DIR / name

    total = 0
    # Read in chunks so a huge paste doesn't balloon memory. UploadFile
    # exposes a SpooledTemporaryFile-backed stream.
    try:
        with dest.open("wb") as fh:
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > UPLOAD_MAX_BYTES:
                    fh.close()
                    try:
                        dest.unlink()
                    except Exception:
                        pass
                    raise HTTPException(
                        413,
                        f"upload too large (max {UPLOAD_MAX_BYTES // (1024 * 1024)} MB)",
                    )
                fh.write(chunk)
    finally:
        await file.close()

    # Best-effort original filename for the UI chip — sanitized because the
    # client controls it. We only ever echo it back, never treat it as a
    # path.
    original = (file.filename or "").strip() or None
    if original:
        # Drop anything that looks like a path separator or sneaky bytes.
        original = original.replace("\\", "/").rsplit("/", 1)[-1]
        original = "".join(ch for ch in original if ch.isprintable())[:120] or None

    payload: dict = {
        "name": name,
        "size": total,
        "content_type": ctype,
        "original_name": original,
        "kind": "image" if is_image else "document",
    }
    if is_document:
        extracted = await asyncio.to_thread(_extract_document_text, dest, ctype)
        payload.update(extracted)
    return payload


def _extract_document_text(path: Path, ctype: str) -> dict:
    """Pull plain text out of an uploaded document.

    Runs on a worker thread so PDF parsing doesn't stall the event loop.
    Returns a dict shaped for merging into the upload response:

        {
            "extracted_text": str,  # UTF-8, already clipped to the cap
            "truncated": bool,
            "page_count": int (PDF only),
            "extract_error": str (only set on failure),
        }
    """
    try:
        if ctype == "application/pdf":
            # Reuse the tools-level helper so the two code paths can't
            # diverge on how they handle PDFs.
            r = tools._read_pdf_sync(str(path), pages=None)
            if not r.get("ok"):
                return {"extract_error": r.get("error") or "pdf parse failed"}
            text = (r.get("output") or "").strip()
            return {
                "extracted_text": text[:DOCUMENT_EXTRACT_MAX_CHARS],
                "truncated": len(text) > DOCUMENT_EXTRACT_MAX_CHARS,
                "page_count": r.get("page_count"),
            }
        # text/plain, text/markdown, text/csv — read directly with a lenient
        # decoder so UTF-8-ish files with stray bytes still come through.
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        return {
            "extracted_text": text[:DOCUMENT_EXTRACT_MAX_CHARS],
            "truncated": len(text) > DOCUMENT_EXTRACT_MAX_CHARS,
        }
    except Exception as e:
        return {"extract_error": f"{type(e).__name__}: {e}"}


@app.post("/api/conversations/{cid}/restore/{stamp}")
def api_restore(cid: str, stamp: str) -> dict:
    """Restore files snapshotted under a given checkpoint stamp."""
    if not db.get_conversation(cid):
        raise HTTPException(404, "not found")
    # Defensive: the stamp we generate looks like 'YYYYMMDDTHHMMSS_xxxx';
    # refuse anything with slashes or '..' to prevent path traversal.
    if "/" in stamp or "\\" in stamp or ".." in stamp:
        raise HTTPException(400, "invalid stamp")
    result = tools.restore_checkpoint(cid, stamp)
    if not result.get("ok"):
        raise HTTPException(404, result.get("error", "checkpoint not found"))
    return result


# ---------------------------------------------------------------------------
# Lifecycle hooks CRUD
#
# Hooks are user-defined shell commands that fire at specific points in the
# agent loop (see agent.py / tools.run_hooks). We expose a tiny REST surface
# so the frontend settings page can list / create / edit / delete them.
#
# Security note: these endpoints are callable only by the local frontend
# bound to localhost (uvicorn default). The commands themselves execute with
# the user's privileges; the UI warns before accepting a new hook.
# ---------------------------------------------------------------------------
class HookBody(BaseModel):
    """POST /api/hooks body. `event` must be one of db.HOOK_EVENTS."""

    event: str
    command: str
    matcher: str | None = None
    timeout_seconds: int = 10
    enabled: bool = True
    # Used by the `consecutive_failures` event — fire after N back-to-back
    # ok=False results from the same tool. Default = 1 (= fire on first
    # failure, equivalent to `tool_error`).
    error_threshold: int | None = None
    # Per-conversation cap. NULL = unlimited.
    max_fires_per_conv: int | None = None


class HookPatchBody(BaseModel):
    """PATCH /api/hooks/{id} body. All fields optional for partial updates."""

    event: str | None = None
    command: str | None = None
    matcher: str | None = None
    timeout_seconds: int | None = None
    enabled: bool | None = None
    error_threshold: int | None = None
    max_fires_per_conv: int | None = None


@app.get("/api/hooks")
def api_list_hooks() -> dict:
    """Return every hook (enabled + disabled) newest-first.

    Also surfaces the per-event timeout caps so the frontend can clamp
    the form input before round-tripping.
    """
    return {
        "hooks": db.list_hooks(),
        "events": sorted(db.HOOK_EVENTS),
        "timeout_caps": {e: db.hook_timeout_cap(e) for e in db.HOOK_EVENTS},
    }


@app.post("/api/hooks")
def api_create_hook(body: HookBody) -> dict:
    """Register a new lifecycle hook."""
    try:
        hid = db.create_hook(
            event=body.event,
            command=body.command,
            matcher=body.matcher,
            timeout_seconds=body.timeout_seconds,
            enabled=body.enabled,
            error_threshold=body.error_threshold,
            max_fires_per_conv=body.max_fires_per_conv,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return db.get_hook(hid) or {}


@app.patch("/api/hooks/{hid}")
def api_update_hook(hid: str, body: HookPatchBody) -> dict:
    """Patch fields on an existing hook. Ignores unset fields."""
    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        updated = db.update_hook(hid, **patch)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not updated:
        raise HTTPException(404, "hook not found")
    return updated


@app.delete("/api/hooks/{hid}")
def api_delete_hook(hid: str) -> dict:
    """Remove a hook. 404 if no row was deleted."""
    n = db.delete_hook(hid)
    if not n:
        raise HTTPException(404, "hook not found")
    return {"ok": True, "deleted": hid}


# ---------------------------------------------------------------------------
# Scheduled-tasks CRUD
#
# The background daemon at the top of this file consumes `scheduled_tasks`
# rows. This REST surface lets the UI list/create/delete them directly so
# the user can manage schedules without going through the agent's
# `schedule_task` tool. The `schedule_task` tool keeps working — both write
# into the same table.
#
# The daemon fires any row whose `next_run_at` is in the past, so "run now"
# is implemented client-side by simply creating a row with `run_at` set to
# the current time.
# ---------------------------------------------------------------------------
class ScheduledTaskBody(BaseModel):
    """Body for POST /api/scheduled-tasks.

    Callers pass `run_at` as either:
      - a unix timestamp (float, seconds since epoch), or
      - an ISO-8601 string (the UI uses <input type="datetime-local">).

    `interval_seconds` omitted / 0 / null means a one-shot. The minimum
    recurrence is 60s to stop the scheduler from starving under a pathological
    `interval: 1`.
    """

    name: str
    prompt: str
    run_at: str | float
    interval_seconds: int | None = None
    cwd: str | None = None


def _parse_run_at(raw: str | float) -> float:
    """Accept unix seconds OR ISO-8601 and return a unix timestamp."""
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s:
        raise ValueError("run_at required")
    # Plain-number string falls through to unix seconds.
    try:
        return float(s)
    except ValueError:
        pass
    # datetime.fromisoformat handles '2026-04-24T14:30' and with timezone.
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc).astimezone()  # treat as local
        return dt.timestamp()
    except Exception as e:
        raise ValueError(f"invalid run_at {s!r}: {e}")


@app.get("/api/scheduled-tasks")
def api_list_scheduled_tasks() -> dict:
    """List every pending scheduled task, soonest-first."""
    return {"tasks": db.list_scheduled_tasks()}


@app.post("/api/scheduled-tasks")
def api_create_scheduled_task(body: ScheduledTaskBody) -> dict:
    """Create a new scheduled task. Body validation mirrors the `schedule_task`
    tool — see its docstring for field semantics."""
    name = (body.name or "").strip()
    prompt = (body.prompt or "").strip()
    if not name or not prompt:
        raise HTTPException(400, "name and prompt are required")
    try:
        run_at = _parse_run_at(body.run_at)
    except ValueError as e:
        raise HTTPException(400, str(e))
    interval = body.interval_seconds or None
    if interval is not None:
        if interval < 60:
            raise HTTPException(400, "interval_seconds must be >= 60")
    cwd = (body.cwd or str(ROOT)).strip() or str(ROOT)
    tid = db.create_scheduled_task(
        name=name,
        prompt=prompt,
        next_run_at=run_at,
        interval_seconds=interval,
        cwd=cwd,
    )
    # Return the full row so the client can render it without a follow-up GET.
    for row in db.list_scheduled_tasks():
        if row["id"] == tid:
            return row
    return {"id": tid}


@app.delete("/api/scheduled-tasks/{tid}")
def api_delete_scheduled_task(tid: str) -> dict:
    """Cancel a pending scheduled task. Accepts short-id prefixes."""
    n = db.cancel_scheduled_task(tid)
    if not n:
        raise HTTPException(404, "scheduled task not found")
    return {"ok": True, "deleted": tid, "count": n}


@app.get("/api/conversations/{cid}/loop")
def api_get_conversation_loop(cid: str) -> dict:
    """Return the active autonomous-loop row for this conversation, or null.

    The frontend polls this on mount + after each agent turn so the loop
    banner can appear / disappear without relying on an SSE event we'd
    otherwise need to invent.
    """
    return {"loop": db.get_active_loop_for_conversation(cid)}


@app.delete("/api/conversations/{cid}/loop")
def api_stop_conversation_loop(cid: str) -> dict:
    """Stop the autonomous loop on a conversation (user clicked Stop loop).

    Idempotent — missing / already-stopped loops return `count: 0` rather
    than 404, so the client doesn't need to distinguish "there was nothing
    to stop" from "server error". Mirrors the behaviour of the
    `stop_loop` tool the agent can call itself.
    """
    n = db.cancel_loops_for_conversation(cid)
    return {"ok": True, "count": n}


@app.get("/api/conversations/{cid}/codebase-index")
def api_get_codebase_index(cid: str) -> dict:
    """Return the codebase-index row for a conversation's cwd, or null.

    The frontend shows a small chip next to the cwd so the user knows whether
    the semantic index is ready, still building, or errored. Resolves the
    cwd server-side so the UI never needs to do the expanduser/resolve dance.
    """
    conv = db.get_conversation(cid)
    if not conv:
        raise HTTPException(404, "not found")
    try:
        root = str(Path(conv["cwd"]).expanduser().resolve())
    except Exception:
        return {"index": None, "cwd": conv.get("cwd")}
    return {"index": db.get_codebase_index(root), "cwd": root}


@app.post("/api/conversations/{cid}/codebase-index/reindex")
def api_reindex_codebase(cid: str) -> dict:
    """Force a fresh background index of this conversation's cwd.

    Flips the registry row to 'pending' (so the UI shows progress) and
    enqueues the worker via the existing kick helper. Idempotent — if a
    build is already in flight we just return its current state.
    """
    conv = db.get_conversation(cid)
    if not conv:
        raise HTTPException(404, "not found")
    if not conv.get("cwd"):
        raise HTTPException(400, "conversation has no cwd")
    try:
        root = str(Path(conv["cwd"]).expanduser().resolve())
    except Exception:
        raise HTTPException(400, "invalid cwd")
    # Force a rebuild even if the row is already 'ready' — the kick helper
    # otherwise short-circuits. We stomp the status so the next kick
    # actually schedules work.
    if root not in _CODEBASE_INDEX_INFLIGHT:
        db.upsert_codebase_index(root, status="pending")
    _kick_codebase_index(root)
    return {"ok": True, "index": db.get_codebase_index(root), "cwd": root}


@app.get("/api/conversations/{cid}/files/search")
async def api_search_conversation_files(
    cid: str, q: str = "", limit: int = 12
) -> dict:
    """Fuzzy-find files under the conversation's cwd for @-mention autocomplete.

    Strategy:
      * Always do a substring match against the set of files `_codebase_list_files`
        would index (gitignore-aware when the cwd is a git repo) — cheap and
        deterministic for the short filename queries the UI sends most often.
      * If the query is long-form (>= 3 chars that aren't typical path chars)
        AND the semantic index is ready, also fold in the top semantic hits.
      * Results are capped and deduped by absolute path; relative paths are
        returned so the UI can show compact labels.

    Security: paths are scoped to the conversation's cwd; any result whose
    absolute path escapes the cwd (symlink chicanery) is filtered out.
    """
    conv = db.get_conversation(cid)
    if not conv:
        raise HTTPException(404, "not found")
    try:
        root = Path(conv["cwd"]).expanduser().resolve()
    except Exception:
        raise HTTPException(400, "invalid cwd")
    if not root.is_dir():
        return {"files": [], "cwd": str(root)}

    q = (q or "").strip()
    lim = max(1, min(int(limit or 12), 50))

    # ---- name-match pass (always-on) --------------------------------------
    # Reuses the same file-discovery logic the indexer uses so @-mentions
    # stay in lockstep with what the model already has indexed.
    try:
        candidates = tools._codebase_list_files(root)
    except Exception:
        candidates = []
    ql = q.lower()

    def _name_score(p: Path) -> float:
        name = p.name.lower()
        rel = str(p.relative_to(root)).lower() if p.is_relative_to(root) else name
        if not ql:
            return 1.0  # empty query — surface recent/any files
        if name == ql:
            return 100.0
        if name.startswith(ql):
            return 50.0 + (10.0 / max(1, len(name)))
        if ql in name:
            return 20.0 + (5.0 / max(1, len(name)))
        if ql in rel:
            return 5.0
        return 0.0

    scored: list[tuple[float, Path]] = []
    for p in candidates:
        s = _name_score(p)
        if s > 0:
            scored.append((s, p))
    scored.sort(key=lambda x: x[0], reverse=True)

    seen: set[str] = set()
    results: list[dict] = []
    for _, p in scored:
        if len(results) >= lim:
            break
        try:
            abs_p = str(p.resolve())
            if abs_p in seen:
                continue
            # Defensive — resolve() chases symlinks; ensure we still sit
            # under the cwd after resolution.
            if not Path(abs_p).is_relative_to(root):
                continue
            rel = str(p.relative_to(root)).replace("\\", "/")
            seen.add(abs_p)
            results.append({
                "path": abs_p,
                "rel_path": rel,
                "name": p.name,
                "source": "name",
            })
        except Exception:
            continue

    # ---- semantic pass (long-form queries only) ---------------------------
    # Don't pay the embedding round-trip on 1-2 char queries — the user is
    # almost certainly still typing a filename.
    if len(q) >= 3 and not q.startswith("/") and "." not in q[-4:]:
        idx = db.get_codebase_index(str(root))
        if idx and idx.get("status") == "ready":
            try:
                sem = await tools.codebase_search(q, top_k=lim, cwd=str(root))
            except Exception:
                sem = {"ok": False}
            # codebase_search returns a text-formatted `output`; parse out
            # the paths the same way the agent does.
            if sem.get("ok") and sem.get("output"):
                import re as _re
                for line in sem["output"].splitlines():
                    m = _re.match(r"\[([\d.]+)\]\s+(\S+)\s+#\d+:\s*(.*)", line)
                    if not m:
                        continue
                    rel = m.group(2).replace("\\", "/")
                    snippet = m.group(3)[:140]
                    abs_p = str((root / rel).resolve())
                    if abs_p in seen:
                        # already surfaced by name-match; just attach a
                        # snippet so the UI can show why it matched.
                        for r in results:
                            if r["path"] == abs_p and "snippet" not in r:
                                r["snippet"] = snippet
                                break
                        continue
                    if not Path(abs_p).is_relative_to(root):
                        continue
                    seen.add(abs_p)
                    results.append({
                        "path": abs_p,
                        "rel_path": rel,
                        "name": Path(rel).name,
                        "snippet": snippet,
                        "source": "semantic",
                    })
                    if len(results) >= lim:
                        break

    return {"files": results[:lim], "cwd": str(root), "query": q}


# ---------------------------------------------------------------------------
# Docs URL indexing — crawl & embed public documentation sites so the agent
# can ground answers on them. Thin REST over `db.*_doc_url` + the crawler
# in tools._docs_url_crawl_impl. The Settings UI calls these endpoints to
# manage the registry; searches are routed through the agent's `docs_search`
# tool, not this surface.
# ---------------------------------------------------------------------------
class DocUrlBody(BaseModel):
    """Body for POST /api/docs/urls — register a new URL to index.

    `url` is required and must be http(s). The SSRF guard in the crawler
    itself rejects private/loopback targets at fetch time; we don't
    duplicate that check here so new schemes automatically pick up the
    stricter backend rules.
    """

    url: str
    title: str | None = None
    max_pages: int | None = None
    same_origin_only: bool | None = None


@app.get("/api/docs/urls")
def api_list_doc_urls() -> dict:
    """Return every registered docs URL, most-recently-touched first."""
    return {"urls": db.list_doc_urls()}


@app.get("/api/docs/urls/{did}")
def api_get_doc_url(did: str) -> dict:
    """Return one URL row — used by the UI to poll crawl progress."""
    row = db.get_doc_url(did)
    if not row:
        raise HTTPException(404, "not found")
    return {"url": row}


@app.post("/api/docs/urls")
def api_create_doc_url(body: DocUrlBody) -> dict:
    """Register a new URL and kick off the first crawl.

    Idempotent on the seed URL: re-posting the same URL returns the
    existing row unchanged (no re-crawl triggered — use the /reindex
    endpoint for that).
    """
    u = (body.url or "").strip()
    if not u:
        raise HTTPException(400, "url is required")
    # Sanity check — reject obvious bad schemes early so the UI surfaces a
    # clean 400 instead of the crawler quietly failing later.
    parsed = urlparse(u)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(400, "url must be http(s)")
    if not (parsed.hostname or "").strip():
        raise HTTPException(400, "url must include a hostname")

    existing = db.get_doc_url_by_url(u)
    if existing:
        return {"url": existing, "created": False}

    max_pages = body.max_pages if body.max_pages is not None else 20
    same_origin = True if body.same_origin_only is None else bool(body.same_origin_only)
    row = db.create_doc_url(
        url=u,
        title=body.title,
        max_pages=max_pages,
        same_origin_only=same_origin,
    )
    if not row:
        raise HTTPException(500, "failed to create url row")
    _kick_docs_url_crawl(row["id"])
    return {"url": db.get_doc_url(row["id"]) or row, "created": True}


@app.post("/api/docs/urls/{did}/reindex")
def api_reindex_doc_url(did: str) -> dict:
    """Force a fresh crawl of an existing seed.

    Overwrites any cached chunks (the crawler wipes the path-prefix first)
    so stale pages can be dropped by re-running the seed.
    """
    row = db.get_doc_url(did)
    if not row:
        raise HTTPException(404, "not found")
    _kick_docs_url_crawl(did)
    return {"url": db.get_doc_url(did) or row}


@app.delete("/api/docs/urls/{did}")
def api_delete_doc_url(did: str) -> dict:
    """Drop a URL seed and every chunk crawled beneath it."""
    row = db.get_doc_url(did)
    if not row:
        raise HTTPException(404, "not found")
    db.delete_doc_url(did)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Global memories CRUD
#
# Global memories are short, durable notes that get injected into the system
# prompt of EVERY conversation (including subagents). They live in their own
# SQLite table — separate from per-conversation memory files — so the user
# can curate "facts about me / how I like to work" once and have every chat
# benefit. The agent can also write these via the `remember(scope="global")`
# tool; this REST surface powers the Settings → Memories panel in the UI.
# ---------------------------------------------------------------------------
class MemoryBody(BaseModel):
    """Body for POST /api/memories and PATCH /api/memories/{mid}.

    For POST, `content` is required. For PATCH, both fields are optional —
    omit a field to leave it unchanged. Sending `topic: ""` (empty string)
    explicitly clears the topic.
    """

    content: str | None = None
    topic: str | None = None


@app.get("/api/memories")
def api_list_memories() -> dict:
    """Return every global memory, oldest-first.

    Oldest-first matches the order in which the entries appear in the
    injected system prompt, so what the user sees in Settings is exactly
    what the model sees.
    """
    return {"memories": db.list_global_memories()}


@app.post("/api/memories")
def api_create_memory(body: MemoryBody) -> dict:
    """Create a new global memory entry. `content` is required."""
    if body.content is None:
        raise HTTPException(400, "content is required")
    try:
        row = db.add_global_memory(body.content, body.topic)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"memory": row}


@app.patch("/api/memories/{mid}")
def api_update_memory(mid: str, body: MemoryBody) -> dict:
    """Patch a global memory's content and/or topic in place."""
    try:
        row = db.update_global_memory(mid, content=body.content, topic=body.topic)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not row:
        raise HTTPException(404, "memory not found")
    return {"memory": row}


@app.delete("/api/memories/{mid}")
def api_delete_memory(mid: str) -> dict:
    """Permanently remove one global memory entry. 404 if no row matched."""
    n = db.delete_global_memory(mid)
    if not n:
        raise HTTPException(404, "memory not found")
    return {"ok": True, "deleted": mid}


# ---------------------------------------------------------------------------
# Secrets (API tokens / credentials that the `http_request` tool can reference
# via `{{secret:NAME}}` placeholders). Values never go out to the model in a
# regular list call — the frontend asks for a single secret by id when the
# user clicks "reveal". Writing a secret requires the full value; reading
# metadata hides it.
# ---------------------------------------------------------------------------
class SecretCreate(BaseModel):
    """Body for POST /api/secrets."""

    name: str
    value: str
    description: str | None = None


class SecretUpdate(BaseModel):
    """Body for PATCH /api/secrets/{sid}. Omit a field to leave it unchanged."""

    name: str | None = None
    value: str | None = None
    description: str | None = None


@app.get("/api/secrets")
def api_list_secrets() -> dict:
    """List every stored secret's metadata — no values in the response."""
    return {"secrets": db.list_secrets()}


@app.get("/api/secrets/{sid}")
def api_reveal_secret(sid: str) -> dict:
    """Return one secret including its value. Used by the reveal button."""
    row = db.get_secret(sid, include_value=True)
    if not row:
        raise HTTPException(404, "secret not found")
    return {"secret": row}


@app.post("/api/secrets")
def api_create_secret(body: SecretCreate) -> dict:
    """Create a new secret. Name + value required."""
    try:
        row = db.create_secret(
            name=body.name,
            value=body.value,
            description=body.description,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    # Return metadata only; the caller already had the value.
    row.pop("value", None)
    return {"secret": row}


@app.patch("/api/secrets/{sid}")
def api_update_secret(sid: str, body: SecretUpdate) -> dict:
    """Update a secret in place. Returns metadata only."""
    try:
        row = db.update_secret(
            sid,
            name=body.name,
            value=body.value,
            description=body.description,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not row:
        raise HTTPException(404, "secret not found")
    row.pop("value", None)
    return {"secret": row}


@app.delete("/api/secrets/{sid}")
def api_delete_secret(sid: str) -> dict:
    """Permanently remove a stored secret."""
    n = db.delete_secret(sid)
    if not n:
        raise HTTPException(404, "secret not found")
    return {"ok": True, "deleted": sid}


# ---------------------------------------------------------------------------
# User-defined tools CRUD
#
# Backs the Settings → Tools panel. Only the user (via this HTTP surface) can
# create, edit, or delete user-defined tools — the LLM has no equivalent
# tool-call route. This is a deliberate safety boundary so the model can't
# mint new tools and extend its own privileges mid-conversation. Once created
# the model CAN call the tool like any other; see tools.user_tool_schemas().
# ---------------------------------------------------------------------------
class UserToolBody(BaseModel):
    """Body for POST /api/user-tools — maps 1:1 to tools.create_tool arguments."""

    name: str
    description: str
    code: str
    schema_: dict | None = Field(default=None, alias="schema")
    deps: list[str] | None = None
    category: str = "write"
    timeout_seconds: int = 60

    model_config = {"populate_by_name": True}


class UserToolPatchBody(BaseModel):
    """Body for PATCH /api/user-tools/{tid}. All fields optional."""

    description: str | None = None
    code: str | None = None
    schema_: dict | None = Field(default=None, alias="schema")
    deps: list[str] | None = None
    category: str | None = None
    timeout_seconds: int | None = None
    enabled: bool | None = None

    model_config = {"populate_by_name": True}


@app.get("/api/user-tools")
def api_list_user_tools() -> dict:
    """List every user-defined tool (enabled + disabled), newest-first."""
    return {
        "tools": db.list_user_tools(enabled_only=False),
        "disabled": tools._utr.is_disabled(),
    }


@app.post("/api/user-tools")
async def api_create_user_tool(body: UserToolBody) -> dict:
    """Create a new user tool.

    Returns the freshly-inserted row plus a short install log so the UI can
    show pip output (and errors) in a toast. This is the only creation path —
    the LLM cannot reach this route because there is no corresponding tool
    schema exposed to it.
    """
    if tools._utr.is_disabled():
        raise HTTPException(
            403, "user tools are disabled via GIGACHAT_DISABLE_USER_TOOLS"
        )
    result = await tools.create_tool(
        body.name,
        body.description,
        body.code,
        body.schema_ or {},
        body.deps or [],
        body.category or "write",
        int(body.timeout_seconds or 60),
    )
    if not result.get("ok"):
        raise HTTPException(400, result.get("error") or "user tool creation failed")
    # Re-fetch so the caller gets the full row (incl. timestamps) rather
    # than the partial dict the helper returns.
    row = db.get_user_tool_by_name(body.name.strip().lower()) or {}
    return {"tool": row, "install_log": result.get("output", "")}


@app.patch("/api/user-tools/{tid}")
def api_update_user_tool(tid: str, body: UserToolPatchBody) -> dict:
    """Patch fields on a user tool. Name is immutable (see db.update_user_tool).

    Mirrors the hardening of POST: any `code` update is AST-validated (must
    still define `def run(args)`) and any `deps` update passes through the
    PEP 508 subset regex + blocklist. Unchanged fields are untouched.
    """
    patch = {}
    dumped = body.model_dump(by_alias=False)
    for k, v in dumped.items():
        if v is None:
            continue
        # Pydantic rewrites `schema_` → `schema` for the DB layer.
        if k == "schema_":
            patch["schema"] = v
        else:
            patch[k] = v
    # Same validation pass that POST runs — keeps the UI surface from smuggling
    # syntactically-broken Python or hostile dep specs past the AST / PEP 508
    # guards just by using PATCH instead of POST.
    if "code" in patch:
        try:
            tools._validate_user_tool_code(patch["code"] or "")
        except ValueError as e:
            raise HTTPException(400, str(e))
    if "deps" in patch:
        try:
            patch["deps"] = tools._utr.validate_deps(patch["deps"] or [])
        except ValueError as e:
            raise HTTPException(400, str(e))
    try:
        updated = db.update_user_tool(tid, **patch)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not updated:
        raise HTTPException(404, "user tool not found")
    return updated


@app.delete("/api/user-tools/{tid}")
def api_delete_user_tool(tid: str) -> dict:
    """Permanently remove a user tool. The venv + installed deps are untouched."""
    n = db.delete_user_tool(tid)
    if not n:
        raise HTTPException(404, "user tool not found")
    return {"ok": True, "deleted": tid}


@app.get("/api/screenshots/{name}")
def api_screenshot(name: str) -> FileResponse:
    """Serve a saved computer-use screenshot to the browser.

    Path traversal is blocked by forcing `name` through Path and requiring
    the resolved file to live directly inside `tools.SCREENSHOT_DIR`.
    """
    safe_name = Path(name).name  # strip any ../ or nested paths
    target = (tools.SCREENSHOT_DIR / safe_name).resolve()
    try:
        target.relative_to(tools.SCREENSHOT_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "invalid screenshot path")
    if not target.is_file():
        raise HTTPException(404, "screenshot not found")
    return FileResponse(target, media_type="image/png")


@app.get("/api/uploads/{name}")
def api_upload_read(name: str) -> FileResponse:
    """Serve a user-uploaded image. Same hardening as /api/screenshots."""
    safe_name = _safe_upload_name(name)
    if not safe_name:
        raise HTTPException(400, "invalid upload path")
    target = (tools.UPLOAD_DIR / safe_name).resolve()
    try:
        target.relative_to(tools.UPLOAD_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "invalid upload path")
    if not target.is_file():
        raise HTTPException(404, "upload not found")
    # Guess the media type from the extension — we only ever wrote known ones.
    suffix = target.suffix.lower()
    media = next(
        (k for k, v in ALLOWED_UPLOAD_TYPES.items() if v == suffix),
        "application/octet-stream",
    )
    return FileResponse(target, media_type=media)


# ---------------------------------------------------------------------------
# Web Push subscription management.
#
# Flow:
#   1. Browser registers the service worker and calls pushManager.subscribe()
#      with the VAPID public key fetched from /api/push/vapid-key.
#   2. The resulting {endpoint, keys} blob is POSTed to /api/push/subscribe.
#   3. Later, when a scheduled task fires or a long agent turn finishes while
#      the tab is hidden, backend/push.py fan-outs an encrypted payload to
#      every saved subscription so the OS shows a native notification.
# ---------------------------------------------------------------------------
class PushKeys(BaseModel):
    """Subscription keys as emitted by the browser's PushSubscription.toJSON()."""

    p256dh: str
    auth: str


class PushSubscribeBody(BaseModel):
    """Body for POST /api/push/subscribe.

    Exactly mirrors the shape of `PushSubscription.toJSON()` in the browser,
    with a best-effort `user_agent` tagged on so the settings UI can label
    each registered device.
    """

    endpoint: str
    keys: PushKeys
    user_agent: str | None = None


class PushUnsubscribeBody(BaseModel):
    """Body for POST /api/push/unsubscribe — only the endpoint is needed."""

    endpoint: str


@app.get("/api/push/vapid-key")
def api_push_vapid_key() -> dict:
    """Public VAPID key for the browser's pushManager.subscribe() call.

    Lazily generated on first call; subsequent calls return the same key so
    previously-saved subscriptions remain valid across server restarts.
    """
    return {"public_key": push.vapid_public_key_b64url()}


@app.post("/api/push/subscribe")
def api_push_subscribe(body: PushSubscribeBody) -> dict:
    """Persist a browser's push subscription.

    Defensive:
      - We reject obvious junk endpoints (non-https, >2KB) before touching the DB.
      - Re-subscribing the same endpoint updates the keys in place — browsers
        rotate their p256dh/auth pair periodically.
    """
    endpoint = body.endpoint.strip()
    if not endpoint.startswith("https://") or len(endpoint) > 2048:
        raise HTTPException(400, "invalid push endpoint")
    db.upsert_push_subscription(
        endpoint=endpoint,
        p256dh=body.keys.p256dh,
        auth=body.keys.auth,
        user_agent=(body.user_agent or "")[:300] or None,
    )
    return {"ok": True, "count": db.count_push_subscriptions()}


@app.post("/api/push/unsubscribe")
def api_push_unsubscribe(body: PushUnsubscribeBody) -> dict:
    """Remove a browser's subscription. Idempotent — deleting something we
    don't have still returns ok=True so the UI can collapse retries."""
    n = db.delete_push_subscription(body.endpoint)
    return {"ok": True, "removed": n, "count": db.count_push_subscriptions()}


@app.get("/api/push/status")
def api_push_status() -> dict:
    """Return the number of registered browsers so the settings panel can
    render "Notifications enabled on 2 devices" without leaking endpoint URLs.
    """
    return {"count": db.count_push_subscriptions()}


@app.post("/api/push/test")
async def api_push_test() -> dict:
    """Fire a test notification to every registered browser.

    Useful for the settings panel's "Send test" button — lets the user
    confirm their device is receiving pushes before relying on the feature.
    """
    result = await asyncio.to_thread(
        push.send_to_all,
        {
            "title": "Gigachat",
            "body": "Push notifications are working.",
            "kind": "test",
        },
    )
    return {"ok": True, **result}


# ---------------------------------------------------------------------------
# MCP (Model Context Protocol) server management.
#
# Each row in the `mcp_servers` table corresponds to one external process we
# spawn over stdio. Tools advertised by those processes are merged into the
# agent's TOOL_SCHEMAS at request time, namespaced under mcp__<name>__<tool>.
#
# The UI calls these routes to CRUD server configs; each write triggers a
# `mcp.refresh_all()` so the running session map reconciles without a
# server restart.
# ---------------------------------------------------------------------------
class MCPServerBody(BaseModel):
    """Body for POST /api/mcp/servers and PATCH /api/mcp/servers/{id}.

    All fields are optional on PATCH. `command` is the executable path or
    name (resolved via PATH at spawn time); `args` is the CLI arg list;
    `env` is merged on top of the parent process env (handy for passing
    API keys to an MCP server without exporting them globally).
    """

    name: str | None = None
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    enabled: bool | None = None


def _mcp_live_status() -> dict[str, dict]:
    """Return {server_name: {running, tools, error?}} for the UI.

    Built from the in-memory session map, so a row that was configured but
    failed to start still shows running=False plus an error hint.
    """
    out: dict[str, dict] = {}
    for name, sess in list(mcp._sessions.items()):
        out[name] = {
            "running": sess.running,
            "tools": [t["name"] for t in sess.tools],
            "stderr_tail": sess.stderr_tail(),
        }
    return out


@app.get("/api/mcp/servers")
def api_mcp_list() -> dict:
    """Enumerate every configured MCP server plus its live status."""
    rows = db.list_mcp_servers()
    status = _mcp_live_status()
    for r in rows:
        r["status"] = status.get(r["name"]) or {"running": False, "tools": []}
    return {"servers": rows}


@app.post("/api/mcp/servers")
async def api_mcp_create(body: MCPServerBody) -> dict:
    """Add a new MCP server configuration and try to bring it up.

    The response includes the persisted row and the refresh report so the UI
    can show "started: 3 tools" or "failed: <error>" immediately without a
    follow-up poll.
    """
    if not body.name or not body.command:
        raise HTTPException(400, "name and command are required")
    try:
        name = mcp.validate_server_name(body.name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    cmd = body.command.strip()
    if not cmd:
        raise HTTPException(400, "command must not be empty")
    try:
        row = db.create_mcp_server(
            name=name,
            command=cmd,
            args=body.args or [],
            env=body.env or {},
            enabled=bool(body.enabled) if body.enabled is not None else True,
        )
    except Exception as e:
        raise HTTPException(400, f"failed to save: {e}")
    report = await mcp.refresh_all()
    return {"server": row, "report": report}


@app.patch("/api/mcp/servers/{sid}")
async def api_mcp_update(sid: str, body: MCPServerBody) -> dict:
    """Partial update — any subset of name/command/args/env/enabled.

    Restarts the backing session if the launch fingerprint changed (command,
    args, or env), or stops it entirely when enabled=false.
    """
    patch: dict[str, Any] = {}
    if body.name is not None:
        try:
            patch["name"] = mcp.validate_server_name(body.name)
        except ValueError as e:
            raise HTTPException(400, str(e))
    if body.command is not None:
        patch["command"] = body.command.strip()
        if not patch["command"]:
            raise HTTPException(400, "command must not be empty")
    if body.args is not None:
        patch["args"] = body.args
    if body.env is not None:
        patch["env"] = body.env
    if body.enabled is not None:
        patch["enabled"] = bool(body.enabled)
    row = db.update_mcp_server(sid, patch)
    if row is None:
        raise HTTPException(404, "MCP server not found")
    # If the name changed, the old session (keyed on the previous name) would
    # leak; stop_all + refresh_all guarantees the session map matches current DB.
    report = await mcp.refresh_all()
    return {"server": row, "report": report}


@app.delete("/api/mcp/servers/{sid}")
async def api_mcp_delete(sid: str) -> dict:
    """Remove an MCP server configuration and stop its subprocess."""
    row = db.get_mcp_server(sid)
    if row is None:
        raise HTTPException(404, "MCP server not found")
    await mcp.stop_session(row["name"])
    n = db.delete_mcp_server(sid)
    return {"ok": True, "removed": n}


@app.post("/api/mcp/refresh")
async def api_mcp_refresh() -> dict:
    """Force a reconcile pass. Useful after the user edits a server that
    wasn't starting cleanly — they fix the command, hit Refresh, and see the
    updated report without having to restart the whole backend."""
    report = await mcp.refresh_all()
    return {"ok": True, "report": report}


# ---------------------------------------------------------------------------
# User settings
#
# Small key/value store persisted in SQLite. The Settings UI uses it to keep
# user preferences across restarts (e.g. which model new conversations should
# default to). Keeping this route generic means a future preference doesn't
# need a dedicated column or endpoint — just POST/PATCH a new key.
# ---------------------------------------------------------------------------
# Keys we expose to the UI. Any other key posted to PATCH is ignored so a
# rogue request can't pollute the store with arbitrary entries.
_ALLOWED_SETTING_KEYS = {
    "default_chat_model",
    # Compute-pool: flag for engaging llama-server with `--model-draft`
    # (speculative decoding) when a fits-on-host model has a viable
    # vocab-compatible smaller chat model anywhere in the pool's
    # combined inventory. Default ON — the picker's gates handle
    # viability, so leaving it on is a no-op for setups that can't
    # benefit and a free 1.3-2× speedup for everyone else. Set to
    # "false" to force the legacy Ollama-only path.
    "compute_pool_speculative_decoding",
    # Compute-pool: per-target manual draft override, JSON-encoded
    # mapping `{"<target_model_name>": "<draft_model_name>"}`. Power-
    # user escape hatch for cross-vocab pairs the safety checks
    # reject. Misuse just produces low accept rates, so it's
    # documented in the README rather than gated.
    "compute_pool_speculative_overrides",
}


class SettingsPatch(BaseModel):
    """Body for PATCH /api/settings. All fields optional.

    Sending `default_chat_model: null` explicitly clears the user's
    preference and lets the auto-tune recommendation take over again.
    Sending an empty string is treated the same way.
    """

    default_chat_model: str | None = None


@app.get("/api/settings")
def api_settings_get() -> dict:
    """Return every stored user setting plus the currently-effective default
    model so the UI can render "Default: gemma4:e4b (auto-detected)" when the
    user hasn't overridden it yet.
    """
    stored = db.get_all_settings()
    return {
        "settings": {k: stored.get(k) for k in _ALLOWED_SETTING_KEYS},
        "effective_chat_model": _resolve_default_chat_model(),
    }


@app.patch("/api/settings")
async def api_settings_patch(body: SettingsPatch) -> dict:
    """Update one or more user settings.

    We validate `default_chat_model` against the list of currently-installed
    Ollama models so a typo can't leave the UI stuck creating conversations
    against a model that doesn't exist on this machine. Clearing the value
    (null or empty string) deletes the row and falls back to the auto-tune
    recommendation.
    """
    fields = body.model_dump(exclude_unset=True)
    if "default_chat_model" in fields:
        raw = fields["default_chat_model"]
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            db.delete_setting("default_chat_model")
        else:
            candidate = raw.strip()
            # Validate against the installed models — prevents a typo from
            # creating conversations against a model Ollama can't load.
            installed = await ollama_runtime.list_installed_models()
            # Accept either the exact tag or the tag with an implicit ":latest"
            # suffix — Ollama reports "gemma4:latest" but users often type "gemma4".
            installed_norm = {m for m in installed}
            installed_norm |= {m.split(":", 1)[0] for m in installed if ":" in m}
            if candidate not in installed_norm:
                raise HTTPException(
                    400,
                    f"model {candidate!r} is not installed — pull it first "
                    "with `ollama pull <name>`",
                )
            db.set_setting("default_chat_model", candidate)
    return {
        "ok": True,
        "settings": {
            k: db.get_all_settings().get(k) for k in _ALLOWED_SETTING_KEYS
        },
        "effective_chat_model": _resolve_default_chat_model(),
    }


# ---------------------------------------------------------------------------
# Compute pool: register other PCs as workers the host can route work to.
#
# Each row represents another machine reachable over the LAN — same Wi-Fi
# or Ethernet, no internet bandwidth used. An optional Tailscale identifier
# is stored alongside the LAN address purely as a self-repair handle: when
# DHCP rebinds the worker to a new LAN IP, the host reaches it over
# Tailscale just long enough to rediscover the new address. Capability
# probe + routing live in `compute_pool.py`; this layer is just CRUD.
# ---------------------------------------------------------------------------
class ComputeWorkerCreate(BaseModel):
    """Body for POST /api/compute-workers.

    All ongoing traffic — chat / embeddings / subagent calls — flows over
    ``address`` (a LAN hostname or RFC1918 IPv4). ``tailscale_host`` is an
    optional fallback used ONLY by the auto-repair routine: when the LAN
    address goes stale (e.g. because the worker rejoined the network and
    DHCP gave it a new lease), the backend reaches the worker over its
    Tailscale identifier just long enough to ask for the new LAN IP, then
    resumes regular traffic over LAN.
    """
    label: str
    address: str  # LAN hostname or IPv4 — used for ongoing Ollama traffic
    ollama_port: int = 11434
    auth_token: str | None = None
    # Optional SSH alias from ~/.ssh/config — used for LAN-side scp of
    # Ollama model blobs. Connects over LAN, not Tailscale.
    ssh_host: str | None = None
    # Optional Tailscale identifier (MagicDNS name like
    # ``worker.your-tailnet.ts.net``, or a CGNAT IPv4 in 100.64.0.0/10).
    # Used only by the auto-repair routine to rediscover the worker's LAN
    # IP after it changes; never used for ongoing traffic.
    tailscale_host: str | None = None
    enabled: bool = True
    use_for_chat: bool = True
    use_for_embeddings: bool = True
    use_for_subagents: bool = True


class ComputeWorkerPatch(BaseModel):
    """Body for PATCH /api/compute-workers/{wid}. All fields optional."""
    label: str | None = None
    address: str | None = None
    ollama_port: int | None = None
    auth_token: str | None = None       # send "" to clear, null to leave alone
    ssh_host: str | None = None         # send "" to clear, null to leave alone
    tailscale_host: str | None = None   # send "" to clear, null to leave alone
    enabled: bool | None = None
    use_for_chat: bool | None = None
    use_for_embeddings: bool | None = None
    use_for_subagents: bool | None = None


@app.get("/api/compute-workers")
def api_list_compute_workers() -> dict:
    """Return every registered worker.

    Auth tokens are NEVER included in the response — the row dict only
    carries an `auth_token_set` boolean so the UI can show "(set)" or
    "(none)" without ever exposing the secret.
    """
    return {"workers": db.list_compute_workers()}


@app.post("/api/compute-workers")
def api_create_compute_worker(body: ComputeWorkerCreate) -> dict:
    try:
        wid = db.create_compute_worker(
            label=body.label,
            address=body.address,
            ollama_port=body.ollama_port,
            auth_token=(body.auth_token or None),
            ssh_host=(body.ssh_host or None),
            tailscale_host=(body.tailscale_host or None),
            enabled=body.enabled,
            use_for_chat=body.use_for_chat,
            use_for_embeddings=body.use_for_embeddings,
            use_for_subagents=body.use_for_subagents,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return db.get_compute_worker(wid) or {}


@app.patch("/api/compute-workers/{wid}")
def api_update_compute_worker(wid: str, body: ComputeWorkerPatch) -> dict:
    patch = body.model_dump(exclude_unset=True)
    # Treat empty-string auth_token as "clear it" — the UI's clear
    # button submits "" rather than re-sending the secret.
    if "auth_token" in patch and patch["auth_token"] == "":
        patch["auth_token"] = None
    try:
        updated = db.update_compute_worker(wid, **patch)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not updated:
        raise HTTPException(404, "worker not found")
    return updated


@app.delete("/api/compute-workers/{wid}")
def api_delete_compute_worker(wid: str) -> dict:
    n = db.delete_compute_worker(wid)
    if not n:
        raise HTTPException(404, "worker not found")
    return {"ok": True, "deleted": wid}


@app.post("/api/compute-workers/{wid}/probe")
async def api_probe_compute_worker(wid: str) -> dict:
    """Manual "Test connection" trigger from the Settings UI.

    Probes one worker now (out-of-band of the 5-min sweep) and returns the
    fresh result so the UI can show success/failure immediately. The probe
    machinery itself persists capabilities/last_error/last_seen on the row,
    so a subsequent GET /api/compute-workers reflects the same outcome.
    """
    result = await compute_pool.probe_worker(wid)
    if result.get("error") == "worker not found":
        raise HTTPException(404, "worker not found")
    return result


@app.post("/api/compute-workers/probe-all")
async def api_probe_all_compute_workers() -> dict:
    """Probe every enabled worker now. Useful as a "refresh all" action
    from the Settings panel; returns a summary list (per-worker ok/error)
    and the heavy capability data is persisted on the rows themselves."""
    return {"results": await compute_pool.probe_all_enabled()}


@app.post("/api/compute-workers/{wid}/push-model")
async def api_push_model_to_worker(wid: str, model: str) -> dict:
    """Copy an Ollama model from this host to the named worker via SCP.

    Saves internet bandwidth: the worker doesn't pull from the public
    Ollama registry, the bytes flow over the LAN. Requires the worker
    row to have `ssh_host` set (an SSH alias the host can resolve via
    its ~/.ssh/config — typically same alias the user already uses to
    connect to that machine).
    """
    try:
        return await model_sync.sync_model(model, wid)
    except model_sync.ModelSyncError as e:
        raise HTTPException(400, str(e))


@app.get("/api/compute-workers/{wid}/push-model/plan")
def api_push_model_plan(wid: str, model: str) -> dict:
    """Preview what `push-model` would ship — number of blobs, total
    bytes, manifest path. Lets the UI confirm before kicking off a
    multi-GB transfer."""
    try:
        plan = model_sync.plan(model, wid)
    except model_sync.ModelSyncError as e:
        raise HTTPException(400, str(e))
    return {
        "manifest_path": str(plan.manifest_path),
        "manifest_dest": plan.manifest_dest,
        "blob_count": len(plan.blob_digests),
        "total_bytes": plan.total_bytes,
    }


class DedupExecuteBody(BaseModel):
    """Body for POST /api/compute-pool/dedup/execute. Optional `model`
    filter restricts execution to that model only — useful when the
    user wants to dry-run one model first before broad-stroke deletes."""
    model: str | None = None


@app.post("/api/compute-pool/dedup/execute")
async def api_compute_pool_dedup_execute(body: DedupExecuteBody) -> dict:
    """Execute the dedup advisor's recommendations.

    SSHes into each worker named by `pool_dedup_recommendations` and
    runs `ollama rm <model>` on it. Skips host (host removals are
    operator-driven by design — the operator's primary surface is
    Ollama, and we don't want a destructive API silently mutating
    the host's local state).

    Returns a per-action result list. Failures on individual workers
    don't abort the run; the caller sees which removes succeeded.
    """
    results = await compute_pool.execute_dedup_recommendations(
        model_filter=body.model,
    )
    bytes_reclaimed = sum(r.get("bytes_reclaimed", 0) for r in results if r.get("ok"))
    return {
        "results": results,
        "total_bytes_reclaimed": bytes_reclaimed,
    }


@app.get("/api/compute-pool/inventory")
def api_pool_inventory() -> dict:
    """Pool-wide model inventory + dedup recommendations.

    Body shape::

        {
          "summary": <pool_inventory_summary output>,
          "dedup_recommendations": <pool_dedup_recommendations output>,
        }

    The Settings UI's "Pool storage" panel renders this — the user can
    see how much disk each model occupies across the pool, where every
    copy lives, and which redundant copies could be safely removed.
    Read-only: the API never deletes anything; the operator triggers
    deletes through the existing Ollama CLI per node, or uses the
    `pushModelToWorker` flow to redistribute.
    """
    return {
        "summary": compute_pool.pool_inventory_summary(),
        "dedup_recommendations": compute_pool.pool_dedup_recommendations(),
    }


@app.get("/api/compute-pool/acquisition/{model_name:path}")
def api_acquisition_status(model_name: str) -> dict:
    """Return the live override-file acquisition status for `model_name`.

    Used by the UI to render a progress bar when Phase 2 split needs an
    override / mmproj file that isn't yet on disk. Body shape:
        {
          "status": "no_override_needed" | "starting" | "running" | "done" | "error",
          "phase":  "init" | "surgery" | "downloading-main" | "downloading-mmproj" | "done",
          "progress_pct": 0..100,
          "needs_main": bool, "needs_mmproj": bool,
          "estimated_total_gb": float,
          "started_at": float, "completed_at": float | None,
          "error": str | None,
        }
    Returns `{"status": "no_override_needed"}` for models that don't
    need any override (most models).
    """
    state = compute_pool.get_acquisition_status(model_name)
    if state is not None:
        return state
    # No live state — derive a one-shot view: is this model in the
    # registry at all, and are its files present?
    if model_name not in compute_pool._KNOWN_OVERRIDE_REGISTRY:
        return {"status": "no_override_needed"}
    spec = compute_pool._KNOWN_OVERRIDE_REGISTRY[model_name]
    main_path = compute_pool._override_gguf_path_for(model_name)
    mmproj_path = compute_pool._override_mmproj_path_for(model_name)
    main_present = main_path.is_file()
    mmproj_present = mmproj_path.is_file() if spec.get("needs_mmproj") else True
    if main_present and mmproj_present:
        return {"status": "ready"}
    return {
        "status": "needed",
        "needs_main": not main_present,
        "needs_mmproj": spec.get("needs_mmproj") and not mmproj_present,
        "estimated_total_gb": (
            (spec.get("main_size_gb", 0) if not main_present else 0)
            + (spec.get("mmproj_size_gb", 0) if spec.get("needs_mmproj") and not mmproj_present else 0)
        ),
        "reason": spec.get("reason"),
    }


@app.get("/api/models/lan-source")
def api_find_lan_source(model: str, exclude_worker_id: str | None = None) -> dict:
    """Where on the LAN is this model already available?

    Returns one of:
      {"kind": "host"}                      — push from host (cheap LAN copy)
      {"kind": "worker", "worker_id":...}   — a peer has it, but the host
                                              doesn't yet (peer→peer push
                                              not implemented; this is a
                                              hint that the user should pull
                                              the model on host first).
      {"kind": null}                        — no LAN source; internet pull
                                              is the only option.
    """
    src = model_sync.find_lan_source_for(model, exclude_worker_id=exclude_worker_id)
    return {"source": src}


# ---------------------------------------------------------------------------
# Split models (Phase 2): one big model whose layers fan across host +
# workers via llama.cpp's RPC mechanism. Schema/CRUD live in db.py;
# this layer wraps it for the API + adds start/stop/status endpoints
# tied to the lifecycle module.
# ---------------------------------------------------------------------------
class SplitModelCreate(BaseModel):
    """Body for POST /api/split-models."""
    label: str
    gguf_path: str
    worker_ids: list[str] = []
    llama_port: int = db.SPLIT_MODEL_DEFAULT_PORT
    enabled: bool = True


class SplitModelPatch(BaseModel):
    """Body for PATCH /api/split-models/{sid}. All fields optional."""
    label: str | None = None
    gguf_path: str | None = None
    worker_ids: list[str] | None = None
    llama_port: int | None = None
    enabled: bool | None = None


@app.get("/api/split-models")
def api_list_split_models() -> dict:
    """Return every registered split model + the host's llama.cpp install
    status so the UI can show "install llama.cpp" prompts up front."""
    install = split_runtime.get_install_status()
    return {
        "split_models": db.list_split_models(),
        "llama_cpp": {
            "installed": install.installed,
            "version": install.version,
            "install_dir": install.install_dir,
            "llama_server_path": install.llama_server_path,
            "rpc_server_path": install.rpc_server_path,
            "platform_supported": install.platform_supported,
            "platform_reason": install.platform_reason,
        },
    }


@app.post("/api/split-models")
def api_create_split_model(body: SplitModelCreate) -> dict:
    try:
        sid = db.create_split_model(
            label=body.label,
            gguf_path=body.gguf_path,
            worker_ids=body.worker_ids,
            llama_port=body.llama_port,
            enabled=body.enabled,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return db.get_split_model(sid) or {}


@app.patch("/api/split-models/{sid}")
def api_update_split_model(sid: str, body: SplitModelPatch) -> dict:
    patch = body.model_dump(exclude_unset=True)
    try:
        updated = db.update_split_model(sid, **patch)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not updated:
        raise HTTPException(404, "split_model not found")
    return updated


@app.delete("/api/split-models/{sid}")
async def api_delete_split_model(sid: str) -> dict:
    """Remove a split-model row. Stops the llama-server first so the
    deletion doesn't leave a stranded subprocess holding VRAM."""
    try:
        await split_lifecycle.stop(sid)
    except Exception:
        # Stop failures shouldn't block delete — the user wants this row
        # gone. The lifecycle module already best-effort cleans up.
        pass
    n = db.delete_split_model(sid)
    if not n:
        raise HTTPException(404, "split_model not found")
    return {"ok": True, "deleted": sid}


@app.post("/api/split-models/{sid}/start")
async def api_start_split_model(sid: str) -> dict:
    """Spawn llama-server for this row. Returns once /health says OK or
    times out. The UI shows a spinner during the wait."""
    try:
        return await split_lifecycle.start(sid)
    except split_lifecycle.SplitLifecycleError as e:
        raise HTTPException(400, str(e))


@app.post("/api/split-models/{sid}/stop")
async def api_stop_split_model(sid: str) -> dict:
    """Terminate the llama-server child. Idempotent on a stopped row."""
    return await split_lifecycle.stop(sid)


@app.get("/api/split-models/{sid}/status")
def api_status_split_model(sid: str) -> dict:
    """Read-only snapshot. Cheap; the UI polls this every few seconds
    while a row is `loading`."""
    s = split_lifecycle.status(sid)
    if not s.get("ok"):
        raise HTTPException(404, s.get("error") or "not found")
    return s


@app.get("/api/split-models/{sid}/log")
def api_split_model_log(sid: str, lines: int = 100) -> dict:
    """Return the tail of llama-server's per-split log file.

    Most start failures (port already bound, GGUF too new for this
    llama.cpp build, OOM during layer offload) surface in the log
    BEFORE the bare 'failed to start' status reaches the API. The
    Settings UI's diagnostic pane uses this to show the operator what
    really went wrong.
    """
    if not db.get_split_model(sid):
        raise HTTPException(404, "split_model not found")
    return {"id": sid, "log": split_lifecycle.read_log_tail(sid, lines=lines)}


@app.post("/api/split-models/install-llamacpp")
def api_install_llamacpp(variant: str = "host") -> dict:
    """Trigger an explicit llama.cpp download + install for the host.

    Variant defaults to `host` (CUDA build for the RTX 3060 Ti). The
    download is multi-hundred-MB and runs synchronously here — the UI
    shows a spinner and is intentionally serialized so the user
    doesn't accidentally trigger two parallel downloads. If you need
    progress streaming, switch to a streaming endpoint later.
    """
    try:
        path = split_runtime.download_llama_cpp(variant=variant)
    except Exception as e:
        raise HTTPException(500, f"install failed: {type(e).__name__}: {e}")
    install = split_runtime.get_install_status()
    return {
        "ok": True,
        "install_dir": str(path),
        "installed": install.installed,
        "llama_server_path": install.llama_server_path,
    }


# ---------------------------------------------------------------------------
# Static frontend (production mode).
#
# When `frontend/dist` exists (after `npm run build`), we serve it at "/".
# The Vite build emits index.html plus an `assets/` directory; we mount the
# whole dist folder so links like /assets/index-abc.js resolve correctly.
# In development you should run `npm run dev` on :5173 instead.
# ---------------------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    index_file = FRONTEND / "index.html"
    if not index_file.exists():
        raise HTTPException(
            503,
            "frontend not built — run `npm install && npm run build` in ./frontend, "
            "or use `npm run dev` on port 5173 during development",
        )
    return FileResponse(index_file)


# Mount /assets for Vite build artifacts (hashed JS/CSS).
if (FRONTEND / "assets").exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND / "assets")),
        name="assets",
    )


# ---------------------------------------------------------------------------
# PWA asset routes — manifest and service worker.
#
# The service worker MUST be served from the root path (or above the scope it
# intends to control), which means we can't nest it under /assets/. Vite
# copies everything under `frontend/public/` to `frontend/dist/` verbatim at
# build time, so in production these files live right next to index.html.
# In dev, Vite's own server handles /sw.js and /manifest.webmanifest from the
# `public/` folder directly — the endpoints below are a no-op on port 5173.
#
# `Service-Worker-Allowed: /` lets us register the worker with top-level
# scope even if a proxy rewrote its path. Cache-Control: no-cache on sw.js
# ensures browsers pick up a new service worker as soon as we ship one.
# ---------------------------------------------------------------------------
_PWA_ROOT_FILES: dict[str, tuple[str, dict[str, str]]] = {
    "sw.js": (
        "application/javascript",
        {"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"},
    ),
    "manifest.webmanifest": ("application/manifest+json", {}),
    "icon-192.png": ("image/png", {"Cache-Control": "public, max-age=86400"}),
    "icon-512.png": ("image/png", {"Cache-Control": "public, max-age=86400"}),
    "icon.svg": ("image/svg+xml", {"Cache-Control": "public, max-age=86400"}),
}


def _serve_pwa_file(name: str) -> FileResponse:
    """Serve a PWA-adjacent asset from the active frontend directory.

    Looks first in `FRONTEND/` (the dist folder in prod) and falls back to
    `FRONTEND/public/` (dev-mode layout). Raises 404 if the file is missing
    so the browser gets a clear signal rather than an HTML 200.
    """
    if name not in _PWA_ROOT_FILES:
        raise HTTPException(404, "not found")
    media_type, headers = _PWA_ROOT_FILES[name]
    # Try dist/ first (production), then public/ (dev).
    candidates = [FRONTEND / name, FRONTEND / "public" / name]
    for path in candidates:
        if path.is_file():
            return FileResponse(path, media_type=media_type, headers=headers)
    raise HTTPException(404, f"{name} not found")


# ----------------------------------------------------------------------
# P2P pool — mDNS discovery + PIN-based pairing + public-pool toggle.
#
# The user-visible UX:
#   1. Open Settings → Compute → "Add a device on this network"
#   2. Other devices on the LAN appear in the list as their mDNS ads land
#   3. Click one → Gigachat shows a 6-digit PIN
#   4. On the chosen device, type the PIN
#   5. Both sides verify the Ed25519 signature, store the pairing, done.
#
# After pairing, peers reconnect automatically when their IP changes —
# the trust anchor is the device's public key, not its address.
#
# Public-pool toggle is a separate concept: when ON (default), this
# install donates spare compute to the wider P2P network and benefits
# from cooperative model-weight distribution. Prompts NEVER leave the
# local pool regardless of the toggle's state. Off → fully isolated
# to local pool only.
# ----------------------------------------------------------------------
class P2PPairAcceptBody(BaseModel):
    """Body for POST /api/p2p/pair/accept (host-side claim acceptance).

    The HOST receives this from the claimant device — usually a
    direct LAN POST from the device the user is pairing. Same shape
    is used by the simulated same-host pair flow in tests.

    `claimant_x25519_public_b64` is the new X25519 (encryption)
    pubkey the claimant ships alongside its Ed25519 (signing)
    pubkey. Stored in `paired_devices.x25519_public_b64` and used
    by the encrypted-envelope module for all subsequent peer-to-peer
    traffic to this peer.
    """
    pairing_id: str
    pin: str
    claimant_device_id: str
    claimant_label: str
    claimant_public_key_b64: str
    signature_b64: str
    claimant_ip: str | None = None
    claimant_port: int | None = None
    claimant_x25519_public_b64: str | None = None


class P2PIdentityLabelBody(BaseModel):
    """Body for PATCH /api/p2p/identity (rename my own device label)."""
    label: str


class P2PPublicPoolBody(BaseModel):
    """Body for PATCH /api/p2p/public-pool — opt in or out of the
    swarm. Toggle off → instantly disconnect from rendezvous +
    close swarm sockets (when those exist; v1 just persists the
    bit so other phases can read it)."""
    enabled: bool


@app.get("/api/p2p/identity")
def api_p2p_identity() -> dict:
    """Return THIS install's public identity — what other peers see."""
    from . import identity as _ident
    me = _ident.get_identity()
    return {
        "device_id": me.device_id,
        "device_id_pretty": _ident.format_device_id(me.device_id),
        "label": me.label,
        "public_key_b64": me.public_key_b64,
    }


@app.patch("/api/p2p/identity")
def api_p2p_set_label(body: P2PIdentityLabelBody) -> dict:
    """Rename the local device. Identity (keypair) is unchanged —
    only the friendly label other peers see in their pairing UI."""
    from . import identity as _ident
    if not (body.label or "").strip():
        raise HTTPException(400, "label must not be empty")
    me = _ident.set_label(body.label)
    return {
        "device_id": me.device_id,
        "device_id_pretty": _ident.format_device_id(me.device_id),
        "label": me.label,
        "public_key_b64": me.public_key_b64,
    }


@app.get("/api/p2p/discover")
def api_p2p_discover() -> dict:
    """Snapshot of currently-discovered LAN peers (excluding self).

    The UI polls this endpoint at ~2 s cadence to populate the
    "Available devices" list. Stale entries are pruned on read so
    a peer that went offline disappears within `_DISCOVERY_TTL_SEC`.
    """
    from . import p2p_discovery as _p2pd
    return {
        "running": _p2pd.is_running(),
        "devices": _p2pd.list_discovered(),
    }


@app.post("/api/p2p/pair/start")
def api_p2p_pair_start() -> dict:
    """Generate a fresh PIN to display. The other device claims it
    by POSTing to /api/p2p/pair/accept within the TTL window."""
    from . import p2p_pairing as _pair
    return _pair.start_pairing()


@app.delete("/api/p2p/pair/{pairing_id}")
def api_p2p_pair_cancel(pairing_id: str) -> dict:
    """Drop a pending pairing offer (user closed the dialog)."""
    from . import p2p_pairing as _pair
    if not _pair.cancel_pairing(pairing_id):
        raise HTTPException(404, "pairing offer not found")
    return {"cancelled": True}


@app.get("/api/p2p/pair/pending")
def api_p2p_pair_pending() -> dict:
    """List active pairing offers — used by the UI to restore the
    "PIN displayed, waiting…" panel after a refresh."""
    from . import p2p_pairing as _pair
    return {"pending": _pair.list_pending()}


@app.post("/api/p2p/pair/accept")
def api_p2p_pair_accept(body: P2PPairAcceptBody) -> dict:
    """Verify a pairing claim and persist the trust anchor.

    Called by the OTHER device (the claimant) — the device whose
    user just typed the PIN. Body carries the claimant's
    Ed25519-signed proof of PIN knowledge.
    """
    from . import p2p_pairing as _pair
    try:
        rec = _pair.accept_pairing(
            pairing_id=body.pairing_id,
            pin=body.pin,
            claimant_device_id=body.claimant_device_id,
            claimant_label=body.claimant_label,
            claimant_public_key_b64=body.claimant_public_key_b64,
            signature_b64=body.signature_b64,
            claimant_ip=body.claimant_ip,
            claimant_port=body.claimant_port,
            claimant_x25519_public_b64=body.claimant_x25519_public_b64,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"paired": rec}


class P2PPairNotifyBody(BaseModel):
    """Body for POST /api/p2p/pair/notify (claimant-side receiver).

    The HOST sends this AFTER it accepts our PIN — it tells us "we
    paired you, here's our identity (signing + encryption keys),
    mirror the friendship on your side." Same canonical-bytes
    signature as the outbound side so a tampered or impersonated
    message is rejected.
    """
    host_device_id: str
    host_label: str
    host_public_key_b64: str
    host_x25519_public_b64: str
    claimant_device_id: str
    timestamp: float
    signature_b64: str


class P2PUnpairNotifyBody(BaseModel):
    """Body for POST /api/p2p/pair/unpair-notify.

    The friend sends this when their user removes the pairing on
    their side. We verify the signature against the peer's stored
    pubkey, then drop our matching record so the friendship is
    symmetrically removed.
    """
    initiator_device_id: str
    initiator_public_key_b64: str
    peer_device_id: str
    timestamp: float
    signature_b64: str


@app.post("/api/p2p/pair/notify")
async def api_p2p_pair_notify(request: Request) -> dict:
    """Receive a "we paired with you" notice from the host side.

    Two acceptable body shapes:
      * Encrypted envelope:  ``{"encrypted": <p2p_crypto envelope>}``
        — preferred when the sender has our X25519 key. Decrypts to
        the same fields the legacy plaintext shape carries.
      * Legacy plaintext:    ``{<host_device_id, host_label, ...>}``
        — first-pair / pre-E2E peers that don't have our X25519 key.
        Still signed (Ed25519) so the host's identity is verified.

    Verifies (after decrypt if needed):
      * signature against claimed host public key
      * claimant_device_id matches us (not a misrouted ping)
      * timestamp is fresh (replay protection)
      * claimed device_id matches the host's own pubkey

    On success: persist the host as a paired peer (X25519 key
    captured when supplied) and mirror the compute_worker row.
    """
    from . import identity as _ident
    from . import p2p_lan_client as _lan
    from . import p2p_crypto as _pc
    me = _ident.get_identity()
    raw = await request.json()
    if not isinstance(raw, dict):
        raise HTTPException(400, "body must be a JSON object")
    if "encrypted" in raw:
        # Encrypted shape — open envelope addressed to us.
        try:
            payload, _verified_sender = _pc.open_envelope_json(
                raw["encrypted"],
                # First-touch: the envelope's claimed sender is the
                # one we'll trust here because we don't yet have
                # them in paired_devices. The signature still binds
                # the envelope to whatever pubkey it claims, which
                # we cross-check against the inner host_public_key.
                expected_sender_ed25519_pub_b64=None,
            )
        except _pc.CryptoError as e:
            raise HTTPException(400, f"envelope decrypt failed: {e}")
        body_dict = payload
    else:
        body_dict = raw
    try:
        body = P2PPairNotifyBody(**body_dict)
    except Exception as e:
        raise HTTPException(400, f"malformed pair-notify body: {e}")
    if body.claimant_device_id != me.device_id:
        raise HTTPException(400, "this notify is for a different device")
    if abs(time.time() - body.timestamp) > 120.0:
        raise HTTPException(400, "notify timestamp out of window")
    # Cross-check that the host's claimed device_id derives from
    # their public key — same protection the rendezvous applies.
    try:
        derived = _ident._device_id_from_pubkey(
            base64.b64decode(body.host_public_key_b64)
        )
    except Exception:
        raise HTTPException(400, "host public key is malformed")
    if derived != body.host_device_id:
        raise HTTPException(400, "host device_id does not match host public key")
    digest = _lan._sign_pair_notify(
        host_device_id=body.host_device_id,
        host_label=body.host_label,
        host_public_key_b64=body.host_public_key_b64,
        host_x25519_public_b64=body.host_x25519_public_b64,
        claimant_device_id=body.claimant_device_id,
        timestamp=body.timestamp,
    )
    try:
        sig = base64.b64decode(body.signature_b64)
    except Exception:
        raise HTTPException(400, "signature is not valid base64")
    if not _ident.verify_signature(body.host_public_key_b64, digest, sig):
        raise HTTPException(401, "signature verification failed")
    # Trust verified — persist the host as a paired peer.
    peer_ip = ""
    if request.client:
        peer_ip = request.client.host or ""
    rec = db.upsert_paired_device(
        device_id=body.host_device_id,
        public_key_b64=body.host_public_key_b64,
        label=body.host_label,
        ip=peer_ip or None,
        port=None,  # we don't know their FastAPI port — mDNS will fill in
        role="local",
        x25519_public_b64=body.host_x25519_public_b64,
    )
    # Phase 2 mirror: also create a compute_worker row so the host
    # appears in our routing pool too. Symmetric with what the host
    # did when it accepted us. Encrypted-proxy mode by default —
    # all compute traffic to the host's machine flows via their
    # Gigachat secure proxy (X25519+ChaCha20), never plaintext.
    # We don't know the host's Gigachat port from this notify
    # (mDNS will fill it in later); fall back to the standard 8000
    # which the host's mDNS ad will refresh on the next sweep.
    try:
        existing = db.get_compute_worker_by_device_id(body.host_device_id)
        if not existing and peer_ip:
            db.create_compute_worker(
                label=body.host_label or body.host_device_id,
                address=peer_ip,
                ollama_port=8000,
                enabled=True,
                use_for_chat=True,
                use_for_embeddings=True,
                use_for_subagents=True,
                gigachat_device_id=body.host_device_id,
                use_encrypted_proxy=True,
            )
    except Exception as e:
        log.info("symmetric pair: compute_worker mirror failed: %s", e)
    return {"paired": rec}


@app.post("/api/p2p/pair/unpair-notify")
async def api_p2p_pair_unpair_notify(request: Request) -> dict:
    """Receive an unpair notice from a former friend.

    Same dual body shape as `pair_notify`: encrypted envelope OR
    legacy plaintext (for peers without our X25519 key on file).
    Verifies the inner signature against the initiator's stored
    pubkey; falls back to the message's claimed pubkey when we
    have no record (in which case the unpair is a no-op anyway).
    """
    from . import identity as _ident
    from . import p2p_lan_client as _lan
    from . import p2p_crypto as _pc
    raw = await request.json()
    if not isinstance(raw, dict):
        raise HTTPException(400, "body must be a JSON object")
    if "encrypted" in raw:
        try:
            payload, _verified = _pc.open_envelope_json(
                raw["encrypted"],
                expected_sender_ed25519_pub_b64=None,
            )
        except _pc.CryptoError as e:
            raise HTTPException(400, f"envelope decrypt failed: {e}")
        body_dict = payload
    else:
        body_dict = raw
    try:
        body = P2PUnpairNotifyBody(**body_dict)
    except Exception as e:
        raise HTTPException(400, f"malformed unpair-notify body: {e}")
    if abs(time.time() - body.timestamp) > 120.0:
        raise HTTPException(400, "notify timestamp out of window")
    me = _ident.get_identity()
    if body.peer_device_id != me.device_id:
        raise HTTPException(400, "this notify is for a different device")
    # Use the stored pubkey if we have one — that way a peer can't
    # use a fresh keypair to unpair under another peer's id. If we
    # have no record at all, accept the message's claimed pubkey
    # for verification but mark the unpair as a no-op (nothing to
    # remove).
    paired = db.get_paired_device(body.initiator_device_id)
    pubkey = (paired or {}).get("public_key_b64") or body.initiator_public_key_b64
    digest = _lan._sign_unpair_notify(
        initiator_device_id=body.initiator_device_id,
        initiator_public_key_b64=body.initiator_public_key_b64,
        peer_device_id=body.peer_device_id,
        timestamp=body.timestamp,
    )
    try:
        sig = base64.b64decode(body.signature_b64)
    except Exception:
        raise HTTPException(400, "signature is not valid base64")
    if not _ident.verify_signature(pubkey, digest, sig):
        raise HTTPException(401, "signature verification failed")
    # Verified — drop the worker first then the pairing.
    try:
        worker = db.get_compute_worker_by_device_id(body.initiator_device_id)
        if worker:
            db.delete_compute_worker(worker["id"])
    except Exception as e:
        log.info("symmetric unpair: compute_worker cleanup failed: %s", e)
    removed = db.delete_paired_device(body.initiator_device_id)
    # Use the local request IP defensively — we don't actually
    # send anything back, just used for logging.
    _ = request.client.host if request.client else "unknown"
    return {"unpaired": removed}


@app.post("/api/p2p/pair/build-claim")
def api_p2p_pair_build_claim(body: dict) -> dict:
    """Build a signed pairing claim FROM this device's identity.

    Called by the claimant's frontend after the user enters the PIN
    they read off the host's screen. The frontend then POSTs the
    returned blob to the host's `/api/p2p/pair/accept` endpoint.

    Body: {pin, nonce, host_public_key_b64}.
    """
    pin = (body.get("pin") or "").strip()
    nonce = (body.get("nonce") or "").strip()
    host_pubkey = (body.get("host_public_key_b64") or "").strip()
    if not all((pin, nonce, host_pubkey)):
        raise HTTPException(400, "pin, nonce, host_public_key_b64 are required")
    from . import p2p_pairing as _pair
    return _pair.build_claim_signature(
        pin=pin, nonce_b64=nonce, host_public_key_b64=host_pubkey,
    )


@app.get("/api/p2p/paired")
def api_p2p_list_paired() -> dict:
    """List devices we've paired with (any role)."""
    return {"paired": db.list_paired_devices()}


@app.delete("/api/p2p/paired/{device_id}")
def api_p2p_unpair(device_id: str) -> dict:
    """Drop a pairing record. The other side keeps theirs until
    they unpair on their own device."""
    if not db.delete_paired_device(device_id):
        raise HTTPException(404, "device not paired")
    return {"unpaired": True}


@app.get("/api/p2p/public-pool")
def api_p2p_public_pool_status() -> dict:
    """Read the public-pool opt-in flag.

    Default: enabled. The flag itself lives in the user_settings
    table so it survives restarts. v1 stores the bit; subsequent
    phases will gate the rendezvous client + donation worker on it.
    """
    val = db.get_setting("p2p_public_pool_enabled")
    if val is None:
        enabled = True  # default ON
    else:
        enabled = str(val).lower() in ("1", "true", "yes", "on")
    return {"enabled": enabled}


@app.patch("/api/p2p/public-pool")
async def api_p2p_public_pool_set(body: P2PPublicPoolBody) -> dict:
    """Toggle the public-pool flag. Effect:
      * ON  — register with the rendezvous, become discoverable
              across the internet, donate spare compute to the
              swarm. Prompts STILL never leave the local pool.
      * OFF — unregister from the rendezvous, close the heartbeat
              loop, fully isolate to local pool only.

    The toggle is applied IMMEDIATELY: turning ON kicks off the
    rendezvous loop and the user can be discovered within a few
    seconds; turning OFF stops the loop and the entry expires from
    the rendezvous within 60 seconds.
    """
    db.set_setting(
        "p2p_public_pool_enabled", "1" if body.enabled else "0",
    )
    # Fire-and-forget the lifecycle change so the HTTP response
    # doesn't block on a slow STUN round-trip. Errors logged inside
    # the rendezvous module.
    try:
        from . import p2p_rendezvous as _rdv
        if body.enabled:
            await _rdv.start()
        else:
            await _rdv.stop()
    except Exception as e:
        log.warning("p2p_rendezvous toggle failed: %s", e)
    # Pool-inventory loop tracks the same toggle. When the user
    # turns Public Pool OFF we also wipe the discovered-peer cache
    # — both to stop pinging peers AND to revoke their tighter-
    # whitelist access to our /api/tags via the secure proxy.
    try:
        from . import p2p_pool_inventory as _inv
        if body.enabled:
            await _inv.start()
        else:
            await _inv.stop()
            _inv.clear_cache()
    except Exception as e:
        log.warning("p2p_pool_inventory toggle failed: %s", e)
    # Relay inbox loop tracks the same toggle so we don't keep
    # long-polling the rendezvous when Public Pool is off (and so
    # peers can't reach us via relay either, mirroring the cache
    # wipe above).
    try:
        from . import p2p_relay as _relay
        if body.enabled:
            await _relay.start()
        else:
            await _relay.stop()
    except Exception as e:
        log.warning("p2p_relay toggle failed: %s", e)
    return {"enabled": bool(body.enabled)}


@app.post("/api/p2p/secure/forward")
async def api_p2p_secure_forward(envelope: dict) -> dict:
    """Inbound encrypted-proxy endpoint (one-shot variant).

    The peer's secure-client wrapped a (method, path, body) tuple in
    a `p2p_crypto` envelope addressed to us; we decrypt + verify
    + forward to local Ollama + encrypt the response back. Errors
    in any of those steps surface as HTTP 400 (envelope problem)
    or HTTP 502 (upstream Ollama problem) so the caller can
    distinguish a security failure from a compute failure.
    """
    from . import p2p_secure_proxy as _sp
    from . import p2p_crypto as _pc
    try:
        return await _sp.serve_forward_one_shot(envelope)
    except _pc.CryptoError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.warning("secure_forward upstream error: %s", e)
        raise HTTPException(502, f"upstream error: {type(e).__name__}")


@app.post("/api/p2p/secure/forward-stream")
async def api_p2p_secure_forward_stream(envelope: dict):
    """Inbound encrypted-proxy endpoint (streaming variant).

    Same trust + verify pipeline as the one-shot endpoint, but the
    response body is NDJSON of envelope-wrapped Ollama chunks.
    Caller reads line-by-line, decrypts each, feeds to its existing
    NDJSON parser. The stream terminator + error markers are
    encoded as separate envelopes with `_stream` keys.
    """
    from . import p2p_secure_proxy as _sp
    from . import p2p_crypto as _pc

    # Verify upfront (synchronously) so a CryptoError surfaces as
    # HTTP 400 rather than mid-stream. The verify path is repeated
    # inside `serve_forward_stream` for the actual decrypt — cheap
    # double-check.
    try:
        _sp._verify_inbound(envelope)
    except _pc.CryptoError as e:
        raise HTTPException(400, str(e))

    async def _gen():
        try:
            async for chunk in _sp.serve_forward_stream(envelope):
                yield chunk
        except Exception as e:
            log.warning("secure_forward_stream error: %s", e)

    return StreamingResponse(
        _gen(), media_type="application/x-ndjson",
    )


@app.get("/api/p2p/rendezvous/status")
def api_p2p_rendezvous_status() -> dict:
    """Read live rendezvous loop state — used by the Settings UI to
    show "Connected to swarm" / "STUN failed" / "Configured but off"
    badges. Cheap, pure read of in-memory state.
    """
    from . import p2p_rendezvous as _rdv
    return _rdv.status()


class P2PRendezvousUrlBody(BaseModel):
    """Body for PATCH /api/p2p/rendezvous/url. Empty url clears the
    setting (loop will exit on the next tick)."""
    url: str = ""


class P2PFairnessConfigBody(BaseModel):
    """Body for PATCH /api/p2p/fairness/config. All fields optional."""
    donation_fraction: float | None = None
    max_concurrent_donations: int | None = None
    per_peer_rate_per_min: int | None = None


@app.get("/api/p2p/fairness/status")
def api_p2p_fairness_status() -> dict:
    """Real-time view of the fairness scheduler.

    Surfaced in Settings → Network so the user sees how their
    donation slice is being used right now: active jobs, per-peer
    slice (auto-balanced by `total_donations / active_consumers`),
    and the configured tunables.
    """
    from . import p2p_fairness as _fair
    return _fair.status()


@app.patch("/api/p2p/fairness/config")
def api_p2p_fairness_set_config(body: P2PFairnessConfigBody) -> dict:
    """Update donation tunables. Each takes effect immediately on
    the next admission decision (no process restart needed)."""
    from . import p2p_fairness as _fair
    return _fair.set_config(
        donation_fraction=body.donation_fraction,
        max_concurrent_donations=body.max_concurrent_donations,
        per_peer_rate_per_min=body.per_peer_rate_per_min,
    )


@app.patch("/api/p2p/rendezvous/url")
async def api_p2p_rendezvous_set_url(body: P2PRendezvousUrlBody) -> dict:
    """Set (or clear) the rendezvous URL.

    Stored in user_settings so the choice survives restarts. The
    rendezvous loop is re-bounced so the new URL takes effect
    within seconds (otherwise the next heartbeat would still
    point at the old URL until the loop re-checked).
    """
    from . import p2p_rendezvous as _rdv
    try:
        cleaned = _rdv.set_rendezvous_url(body.url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    # Bounce the loop so the new URL takes effect immediately
    # without waiting for the next 30 s heartbeat tick.
    try:
        await _rdv.stop()
        if cleaned:
            await _rdv.start()
    except Exception as e:
        log.warning("p2p_rendezvous bounce failed: %s", e)
    return {"url": cleaned}


# ----------------------------------------------------------------------
# Audit log endpoint — read-only cross-conversation list of every tool
# call the agent ran. Filterable by conversation / tool / since-timestamp
# so the user can scan "what did the agent do today" without trawling
# every chat manually. Backed by the audit_log table the dispatcher
# writes to on every tool call.
# ----------------------------------------------------------------------
@app.get("/api/audit-log")
def api_list_audit_log(
    limit: int = 200,
    conversation_id: str | None = None,
    tool_name: str | None = None,
    since_ts: float | None = None,
) -> dict:
    """List recent tool calls, newest-first.

    Read-only: there is no POST/DELETE endpoint by design. The audit log
    is append-only from the dispatcher and is not meant to be edited.
    """
    rows = db.list_audit_log(
        limit=limit,
        conversation_id=conversation_id,
        tool_name=tool_name,
        since_ts=since_ts,
    )
    return {"audit": rows}


# ----------------------------------------------------------------------
# Skill library endpoints — list / get / delete are exposed to the UI
# so the user can browse and prune the agent's saved playbooks.
# Creation is tool-only; the agent decides what's worth banking.
# ----------------------------------------------------------------------
@app.get("/api/skills")
def api_list_skills(limit: int = 200) -> dict:
    return {"skills": db.list_skills(limit=limit)}


@app.get("/api/skills/{name}")
def api_get_skill(name: str) -> dict:
    rec = db.get_skill(name)
    if not rec:
        raise HTTPException(404, "skill not found")
    return {"skill": rec}


@app.delete("/api/skills/{name}")
def api_delete_skill(name: str) -> dict:
    if not db.delete_skill(name):
        raise HTTPException(404, "skill not found")
    return {"deleted": True}


# ----------------------------------------------------------------------
# OpenAPI registry endpoints — list, fetch, delete. Loading specs is
# done via the agent's `openapi_load` tool so the model can register
# them on demand; this surface lets the user inspect / clean up.
# ----------------------------------------------------------------------
@app.get("/api/openapi")
def api_list_openapi_specs() -> dict:
    # Strip the heavy spec_json body from the listing — UI only needs
    # the metadata. The caller can fetch the full operations list via
    # the per-id endpoint below.
    return {
        "specs": [
            {k: v for k, v in s.items() if k != "operations"}
            for s in db.list_openapi_specs()
        ],
    }


@app.get("/api/openapi/{api_id}")
def api_get_openapi_spec(api_id: str) -> dict:
    rec = db.get_openapi_spec(api_id)
    if not rec:
        raise HTTPException(404, "API spec not found")
    return {"spec": rec}


@app.delete("/api/openapi/{api_id}")
def api_delete_openapi_spec(api_id: str) -> dict:
    if not db.delete_openapi_spec(api_id):
        raise HTTPException(404, "API spec not found")
    return {"deleted": True}


# ----------------------------------------------------------------------
# Webhook receiver endpoints.
#
# `POST /webhook/{token}` — public-ish receiver: anyone with the
# generated token can fire the corresponding agent turn. The token
# is 24 url-safe random bytes (~32 chars), drawn from `secrets`, so
# bruteforcing is computationally infeasible. Body is forwarded to
# the agent as the user message in `target_conversation_id`.
#
# Management endpoints under `/api/webhooks` — list / create / patch
# / delete. Creation generates a fresh token; the user copies it from
# the response and uses it in their external service.
# ----------------------------------------------------------------------
class WebhookCreateBody(BaseModel):
    """Body for POST /api/webhooks."""
    name: str
    target_conversation_id: str
    prompt_template: str | None = ""


class WebhookPatchBody(BaseModel):
    """Body for PATCH /api/webhooks/{id}."""
    name: str | None = None
    enabled: bool | None = None
    prompt_template: str | None = None


@app.get("/api/webhooks")
def api_list_webhooks() -> dict:
    return {"webhooks": db.list_webhooks()}


@app.post("/api/webhooks")
def api_create_webhook(body: WebhookCreateBody) -> dict:
    try:
        rec = db.create_webhook(
            name=body.name,
            target_conversation_id=body.target_conversation_id,
            prompt_template=body.prompt_template or "",
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"webhook": rec}


@app.patch("/api/webhooks/{wid}")
def api_update_webhook(wid: str, body: WebhookPatchBody) -> dict:
    rec = db.update_webhook(
        wid,
        enabled=body.enabled,
        prompt_template=body.prompt_template,
        name=body.name,
    )
    if not rec:
        raise HTTPException(404, "webhook not found")
    return {"webhook": rec}


@app.delete("/api/webhooks/{wid}")
def api_delete_webhook(wid: str) -> dict:
    if not db.delete_webhook(wid):
        raise HTTPException(404, "webhook not found")
    return {"deleted": True}


@app.post("/webhook/{token}")
async def public_webhook_receiver(
    token: str,
    request: Request,
) -> dict:
    """Public webhook fire endpoint. Spawns an agent turn with the
    request body inlined as the user message.

    This is the ONLY endpoint that does not require a logged-in
    session — any external service with the token can fire a turn.
    Defence: the token itself is the credential (24 random bytes), so
    leaked tokens give the same access-controls as a leaked API key.
    Rotate by deleting + recreating the webhook record.
    """
    rec = db.get_webhook_by_token(token)
    if not rec or not rec.get("enabled"):
        # Don't leak whether the webhook exists vs is just disabled —
        # a probe gets the same response either way.
        raise HTTPException(404, "not found")
    # Read the body as text. Limit size so a huge payload can't OOM us.
    raw_body = await request.body()
    if len(raw_body) > 256_000:
        raise HTTPException(413, "payload too large (max 256 KB)")
    body_text = raw_body.decode("utf-8", errors="replace")
    template = rec.get("prompt_template") or ""
    if template:
        # `{body}` placeholder is replaced with the request body so the
        # user can wrap it in instructions ("You received this webhook
        # payload:\n{body}\n\nSummarise it and decide if it warrants action.").
        prompt = template.replace("{body}", body_text)
    else:
        prompt = body_text or "(empty webhook body)"
    # Persist a user message + spawn a turn the same way the regular
    # send-message endpoint does. We don't await the agent loop —
    # webhook callers want a fast 202.
    try:
        # Schedule the turn on the existing agent runner. The send
        # function is async-generator-based; we drive it in the
        # background with a fire-and-forget task.
        asyncio.create_task(_drive_webhook_turn(rec, prompt))
        db.record_webhook_fire(rec["id"])
    except Exception as e:
        raise HTTPException(500, f"failed to spawn turn: {e}")
    return {"accepted": True, "webhook_id": rec["id"]}


async def _drive_webhook_turn(webhook: dict, prompt: str) -> None:
    """Drive the agent loop for a webhook-triggered turn until it ends.

    Errors are logged but never re-raised — the HTTP layer has already
    returned 202 Accepted by the time this runs in the background.
    """
    try:
        async for _ in agent.run_turn(
            webhook["target_conversation_id"],
            user_text=prompt,
        ):
            # Drop events on the floor — we don't need to forward them
            # over SSE for a non-interactive webhook fire. The events
            # ARE persisted into the conversation history so the user
            # sees them next time they open the chat.
            pass
    except Exception:
        # The agent loop's own error handlers persist a failure
        # message into the conversation; nothing more to do here.
        pass


# ----------------------------------------------------------------------
# File watcher endpoints — same shape as webhooks. The actual watching
# is done by the file_watcher_runtime module (registered at app
# startup) which polls the filesystem and fires agent turns.
# ----------------------------------------------------------------------
class FileWatcherCreateBody(BaseModel):
    """Body for POST /api/file-watchers."""
    name: str
    target_conversation_id: str
    path: str
    glob_pattern: str | None = "*"
    events: list[str] | None = None
    prompt_template: str | None = ""
    debounce_seconds: int | None = 5


class FileWatcherPatchBody(BaseModel):
    name: str | None = None
    enabled: bool | None = None
    prompt_template: str | None = None


@app.get("/api/file-watchers")
def api_list_file_watchers() -> dict:
    return {"watchers": db.list_file_watchers()}


@app.post("/api/file-watchers")
def api_create_file_watcher(body: FileWatcherCreateBody) -> dict:
    try:
        rec = db.create_file_watcher(
            name=body.name,
            target_conversation_id=body.target_conversation_id,
            path=body.path,
            glob_pattern=body.glob_pattern or "*",
            events=body.events,
            prompt_template=body.prompt_template or "",
            debounce_seconds=body.debounce_seconds or 5,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"watcher": rec}


@app.patch("/api/file-watchers/{wid}")
def api_update_file_watcher(wid: str, body: FileWatcherPatchBody) -> dict:
    rec = db.update_file_watcher(
        wid,
        enabled=body.enabled,
        prompt_template=body.prompt_template,
        name=body.name,
    )
    if not rec:
        raise HTTPException(404, "file watcher not found")
    return {"watcher": rec}


@app.delete("/api/file-watchers/{wid}")
def api_delete_file_watcher(wid: str) -> dict:
    if not db.delete_file_watcher(wid):
        raise HTTPException(404, "file watcher not found")
    return {"deleted": True}


@app.get("/sw.js")
def pwa_service_worker() -> FileResponse:
    return _serve_pwa_file("sw.js")


@app.get("/manifest.webmanifest")
def pwa_manifest() -> FileResponse:
    return _serve_pwa_file("manifest.webmanifest")


@app.get("/icon-192.png")
def pwa_icon_192() -> FileResponse:
    return _serve_pwa_file("icon-192.png")


@app.get("/icon-512.png")
def pwa_icon_512() -> FileResponse:
    return _serve_pwa_file("icon-512.png")


@app.get("/icon.svg")
def pwa_icon_svg() -> FileResponse:
    return _serve_pwa_file("icon.svg")
