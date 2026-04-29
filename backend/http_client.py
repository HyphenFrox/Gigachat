"""Shared httpx.AsyncClient — connection pooling for hot HTTP paths.

Prior to this module every embed call, every indexer iteration, every
short HTTP fetch built its own ``httpx.AsyncClient`` via
``async with httpx.AsyncClient(...)``. That tears down the client on
every call, which forces a fresh TCP+TLS handshake on the next call to
the same host. For Gigachat's hot paths — embedding chunks against the
local Ollama, hitting llama-server's chat completions, fanning out
embeds across pool workers — the handshake cost dominates the request
itself: ~10-30 ms TCP+TLS setup vs ~50-200 ms per embed.

Shared pooling cuts that overhead to ~1-2 ms after the first call (the
client keeps idle keepalive connections per host). For a 1000-chunk
codebase index that's ~30 s saved on a single index build.

Lifecycle:
  * Lazy singleton. The first ``await get_shared_client()`` from any
    coroutine creates the client; subsequent calls return the same
    instance. An ``asyncio.Lock`` serialises the creation so two
    coroutines racing the first call don't double-create.
  * Close on app shutdown via ``aclose_shared_client()``. FastAPI's
    lifespan hook in ``app.py`` calls this so uvicorn exits cleanly.
  * Tests can call ``aclose_shared_client()`` between tests if they
    care about isolation; the next call re-creates lazily.

Per-call timeouts: the shared client is configured WITHOUT a hard
timeout — callers pass ``timeout=`` on each ``client.get()`` /
``client.post()`` so a slow embed doesn't share its budget with a fast
chat completion. The default-wide pool ceiling (100 connections,
20 keepalive) is generous enough for the most aggressive concurrent
fan-out we do (16 in-flight embeds × ~3 backends) without leaking.
"""
from __future__ import annotations

import asyncio
import logging

import httpx

log = logging.getLogger(__name__)


# HTTP/2 support is optional — requires the `h2` package which we don't
# pin as a hard dep. Probe once at import time so the client init below
# can pick the right setting without a try/except per call.
try:
    import h2  # type: ignore  # noqa: F401
    _HAS_HTTP2 = True
except ImportError:
    _HAS_HTTP2 = False


# Module-level singleton. None until first use; reset to None after close
# so a subsequent caller can lazily re-create.
_SHARED_CLIENT: httpx.AsyncClient | None = None

# Single lock guards client creation. asyncio.Lock is event-loop-bound;
# we create it lazily so the module can import before any event loop
# exists (uvicorn spawns the loop after import).
_LOCK: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    """Lazily create the creation lock on first use.

    Pattern mirrors `asyncio.Lock()` semantics: we can't construct it at
    module import (no running loop yet), so we defer to first call. The
    `asyncio.Lock()` itself is bound to the current loop; if a fresh
    loop replaces ours (rare, only in tests) the old lock becomes a
    no-op no-contention object — still safe.
    """
    global _LOCK
    if _LOCK is None:
        _LOCK = asyncio.Lock()
    return _LOCK


# Pool sizing. 100 max connections / 20 keepalive scales to a hundred
# concurrent backends with steady-state reuse for the busiest 20. Set
# higher than necessary on purpose — the cost of an unused slot is one
# small struct, while OOM-style failures from saturating a too-small
# pool would be opaque to debug.
_POOL_LIMITS = httpx.Limits(
    max_keepalive_connections=20,
    max_connections=100,
    # 30 s keepalive matches the typical "burst of calls in a chat
    # turn" pattern. Idle past that, the connection gets reaped — fine
    # because the next call's setup will hit a warmer Ollama anyway.
    keepalive_expiry=30.0,
)


async def get_shared_client() -> httpx.AsyncClient:
    """Return the process-wide shared `httpx.AsyncClient`.

    Lazily created on first use; subsequent calls return the same
    instance. The client's pool keeps connections warm per (scheme,
    host, port) tuple, so back-to-back requests to the same Ollama /
    llama-server / worker amortise the TCP+TLS handshake to one cost
    paid up front.

    Callers MUST NOT close the returned client themselves —
    `aclose_shared_client` is the only correct close path, and it's
    called from the FastAPI shutdown hook.
    """
    global _SHARED_CLIENT
    if _SHARED_CLIENT is not None and not _SHARED_CLIENT.is_closed:
        return _SHARED_CLIENT
    async with _get_lock():
        # Re-check inside the lock — another coroutine may have created
        # the client between our outer check and acquiring the lock.
        if _SHARED_CLIENT is None or _SHARED_CLIENT.is_closed:
            _SHARED_CLIENT = httpx.AsyncClient(
                # Per-call timeouts — see module docstring.
                timeout=None,
                limits=_POOL_LIMITS,
                # Allow redirects on by default mirrors stock
                # `AsyncClient` behaviour; explicit so it doesn't
                # surprise on a future httpx default change.
                follow_redirects=True,
                # HTTP/2 enabled when the optional `h2` package is
                # available — multiplexed streams to the same host
                # (Ollama's /api/embeddings + /api/chat going to
                # localhost simultaneously) avoid head-of-line
                # blocking. Falls back to HTTP/1.1 silently when h2
                # isn't installed.
                http2=_HAS_HTTP2,
            )
            log.debug("http_client: created shared AsyncClient")
    return _SHARED_CLIENT


async def aclose_shared_client() -> None:
    """Close the shared client. Idempotent — safe to call from shutdown
    hooks, tests, or after a failure path that wants to reset state.

    Resets the global to None so a subsequent caller lazily recreates.
    """
    global _SHARED_CLIENT
    if _SHARED_CLIENT is None:
        return
    try:
        await _SHARED_CLIENT.aclose()
    except Exception as e:
        log.debug("http_client: aclose raised (ignored): %s", e)
    _SHARED_CLIENT = None
