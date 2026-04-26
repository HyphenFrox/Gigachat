"""Compute pool: capability probe + liveness sweep for registered workers.

The pool's data layer (CRUD on `compute_workers`) lives in `db.py`. This
module owns the operational side: ping each worker's Ollama, cache
what's installed there, and surface failures so the Settings UI and
the routing layer can grey out unreachable nodes.

Two entry points:
  * `probe_worker(wid)` — one-shot probe; used by the manual "Test
    connection" button in the UI and by the routing layer when it
    wants to confirm a worker is hot before sending real traffic.
  * `start_periodic_probe()` — background asyncio task scheduled on
    app startup. Sweeps every `_SWEEP_INTERVAL_SEC` (5 min by default)
    so capability and liveness data stay fresh even when no one is
    actively poking the Settings panel.

The probe is intentionally cheap: two parallel GETs (`/api/version`
and `/api/tags`) with a short timeout. If either fails we record the
error string on the worker row and reschedule for next sweep — no
exponential backoff yet because workers are typically on a stable LAN
or tailnet, not flaky public networks. Add backoff if real-world
flakiness appears.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from . import db

log = logging.getLogger(__name__)


# Sweep cadence. 5 min is a balance between "Settings UI shows
# something fresh when the user opens it" and "don't burn the worker's
# bandwidth on a stationary background poll." The sweep is cheap (two
# tiny GETs per worker) so this could go faster, but most workers'
# state changes — model pulls, hardware swaps — happen rarely.
_SWEEP_INTERVAL_SEC = 300

# Per-probe timeout. The worker is on the same LAN or tailnet; a
# multi-second response means it's overloaded or down, not slow.
_PROBE_TIMEOUT_SEC = 5.0

# Default TCP port `rpc-server` (llama.cpp's worker process) listens on.
# This is the upstream's default; users can override per-worker once the
# `rpc_port` schema column lands. Phase 2 commit 3 just probes this port
# to surface whether rpc-server is running on each worker.
_DEFAULT_RPC_PORT = 50052

# How long to wait for the TCP handshake on the rpc-server probe. Smaller
# than the HTTP probe timeout because we're literally just checking if
# the listener exists — a SYN-ACK round trip on LAN is sub-millisecond.
_RPC_PROBE_TIMEOUT_SEC = 2.0


def _worker_base_url(worker: dict) -> str:
    """Build the Ollama base URL for a worker row."""
    addr = (worker.get("address") or "").strip()
    port = int(worker.get("ollama_port") or 11434)
    # Strip any trailing slashes / scheme the user may have pasted in.
    if addr.startswith("http://"):
        addr = addr[len("http://"):]
    elif addr.startswith("https://"):
        addr = addr[len("https://"):]
    addr = addr.rstrip("/")
    return f"http://{addr}:{port}"


def _worker_host(worker: dict) -> str:
    """Bare hostname / IP for non-HTTP probes (rpc-server uses TCP, not HTTP).

    Same scheme stripping as `_worker_base_url` but returns just the host
    portion — `rpc-server` listens on its own port, not the Ollama port.
    """
    addr = (worker.get("address") or "").strip()
    if addr.startswith("http://"):
        addr = addr[len("http://"):]
    elif addr.startswith("https://"):
        addr = addr[len("https://"):]
    return addr.rstrip("/")


async def _probe_rpc_server(
    host: str, port: int, timeout: float = _RPC_PROBE_TIMEOUT_SEC,
) -> tuple[bool, str | None]:
    """TCP-connect probe for rpc-server. Returns (reachable, error_str).

    rpc-server speaks llama.cpp's binary RPC protocol — it doesn't have
    an HTTP health endpoint, and its protocol handshake is more involved
    than we want to replicate just for liveness. A successful TCP connect
    is enough to know the listener is up; a real RPC call from
    `llama-server --rpc <host>:<port>` will surface protocol mismatches
    later if the version is wrong.

    Why asyncio.open_connection instead of httpx: we don't need HTTP, and
    httpx has no clean "TCP connect only" mode. asyncio.open_connection +
    immediately closing the writer gives us exactly the SYN/SYN-ACK
    round-trip we want.
    """
    try:
        # asyncio.wait_for + open_connection is the canonical "is this
        # host:port accepting TCP connections within N seconds" pattern.
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return False, "rpc-server probe: timeout"
    except (OSError, ConnectionError) as e:
        # ECONNREFUSED most commonly — rpc-server isn't running.
        return False, f"rpc-server probe: {type(e).__name__}: {e}"
    except Exception as e:
        return False, f"rpc-server probe: {type(e).__name__}: {e}"
    # Connect succeeded — close immediately. We don't speak the binary
    # protocol; just the listener-existence check.
    try:
        writer.close()
        await writer.wait_closed()
    except Exception:
        pass
    return True, None


async def _probe_one(client: httpx.AsyncClient, base: str, token: str | None) -> dict:
    """Issue the two GETs in parallel and merge the results.

    Returns a dict with `version`, `models`, and any `error`. Either
    field may be missing if its specific GET failed; we don't fail the
    whole probe on a single endpoint hiccup so a worker running an
    Ollama version that hasn't shipped `/api/version` (rare, but
    possible on very old builds) still surfaces its model list.
    """
    headers: dict[str, str] = {}
    if token:
        # Bearer header is the convention Gigachat already uses for the
        # main loopback auth — re-use it so the worker side can validate
        # with the same code path. The worker is expected to enforce
        # this through its own AuthMiddleware-style gate.
        headers["Authorization"] = f"Bearer {token}"

    async def _ver() -> Any:
        r = await client.get(f"{base}/api/version", headers=headers)
        r.raise_for_status()
        return r.json()

    async def _tags() -> Any:
        r = await client.get(f"{base}/api/tags", headers=headers)
        r.raise_for_status()
        return r.json()

    out: dict[str, Any] = {}
    # gather() with return_exceptions so a partial failure on one
    # endpoint doesn't lose the other endpoint's payload.
    ver_res, tags_res = await asyncio.gather(
        _ver(), _tags(), return_exceptions=True,
    )
    if isinstance(ver_res, Exception):
        out["version_error"] = f"{type(ver_res).__name__}: {ver_res}"
    else:
        out["version"] = (ver_res or {}).get("version") or "unknown"
    if isinstance(tags_res, Exception):
        out["tags_error"] = f"{type(tags_res).__name__}: {tags_res}"
    else:
        models = []
        for m in (tags_res or {}).get("models", []) or []:
            details = (m.get("details") or {}) if isinstance(m, dict) else {}
            models.append({
                "name": m.get("name") if isinstance(m, dict) else None,
                "size": m.get("size") if isinstance(m, dict) else None,
                "family": details.get("family"),
                "parameter_size": details.get("parameter_size"),
                "quantization_level": details.get("quantization_level"),
            })
        # Drop entries with no name (defensive against malformed responses).
        out["models"] = [m for m in models if m.get("name")]
    return out


async def probe_worker(wid: str) -> dict:
    """Probe one worker now and persist the result.

    Returns a dict with `ok`, the merged probe payload (`version`,
    `models`, or error markers), and the `last_seen` timestamp. Safe
    to call from any context — failures are caught and recorded on
    the row, never propagated.
    """
    worker = db.get_compute_worker(wid)
    if not worker:
        return {"ok": False, "error": "worker not found"}
    if not worker.get("enabled"):
        return {"ok": False, "error": "worker disabled — enable it first"}

    base = _worker_base_url(worker)
    token = db.get_compute_worker_auth_token(wid)
    now = time.time()

    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SEC) as client:
            payload = await _probe_one(client, base, token)
    except Exception as e:
        # Network-level failure — connection refused, DNS miss, etc.
        # Record on the row so the UI can show "unreachable since X".
        err = f"{type(e).__name__}: {e}"
        try:
            db.update_compute_worker_capabilities(
                wid, last_seen=now, last_error=err,
            )
        except Exception:
            pass
        return {"ok": False, "error": err, "last_seen": now}

    # Two-endpoint outcome: success only if BOTH endpoints responded.
    # Partial success is still a usable signal — record what we got
    # but flag the error so the UI can warn.
    has_models = bool(payload.get("models"))
    has_version = "version" in payload
    error_parts = []
    if not has_models:
        error_parts.append(payload.get("tags_error") or "no models field")
    if not has_version:
        error_parts.append(payload.get("version_error") or "no version field")
    error_str = "; ".join(error_parts) if error_parts else ""

    # Phase 2 add-on: TCP-probe the worker's rpc-server port so the UI
    # can show whether layer-split inference is available on this
    # worker. Probe failure is NOT counted as `last_error` — rpc-server
    # is optional (Phase 1 routing works without it), so a worker with
    # Ollama up but rpc-server down is still "online" for chat /
    # embeddings / subagent routing. We just record the rpc state in
    # capabilities so commit 6's UI can render a separate badge.
    rpc_host = _worker_host(worker)
    rpc_port = _DEFAULT_RPC_PORT
    rpc_ok, rpc_err = await _probe_rpc_server(rpc_host, rpc_port)

    capabilities = {
        "version": payload.get("version"),
        "models": payload.get("models") or [],
        "rpc_server_reachable": rpc_ok,
        "rpc_port": rpc_port,
        "rpc_error": rpc_err,
    }
    try:
        db.update_compute_worker_capabilities(
            wid,
            capabilities=capabilities,
            last_seen=now,
            # Empty string clears any previous error; non-empty records.
            last_error=error_str if error_str else "",
        )
    except Exception:
        pass

    return {
        "ok": has_models and has_version,
        "capabilities": capabilities,
        "error": error_str or None,
        "last_seen": now,
    }


async def probe_all_enabled() -> list[dict]:
    """Probe every enabled worker concurrently. Returns a list of
    `{worker_id, label, ok, error}` summaries — the heavy capability
    payload is persisted on the row, this just tells the caller what
    happened in aggregate."""
    workers = db.list_compute_workers(enabled_only=True)
    if not workers:
        return []
    results = await asyncio.gather(
        *(probe_worker(w["id"]) for w in workers),
        return_exceptions=True,
    )
    summaries: list[dict] = []
    for w, r in zip(workers, results):
        if isinstance(r, Exception):
            summaries.append({
                "worker_id": w["id"],
                "label": w["label"],
                "ok": False,
                "error": f"{type(r).__name__}: {r}",
            })
        else:
            summaries.append({
                "worker_id": w["id"],
                "label": w["label"],
                "ok": bool(r.get("ok")),
                "error": r.get("error"),
            })
    return summaries


_PROBE_TASK: asyncio.Task | None = None


async def _periodic_loop() -> None:
    """Internal: sweep every `_SWEEP_INTERVAL_SEC`. Started/stopped via
    `start_periodic_probe` / `stop_periodic_probe` on app lifecycle."""
    while True:
        try:
            await probe_all_enabled()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("compute_pool periodic probe failed: %s", e)
        try:
            await asyncio.sleep(_SWEEP_INTERVAL_SEC)
        except asyncio.CancelledError:
            raise


def start_periodic_probe() -> None:
    """Schedule the background sweep. Idempotent — calling twice is
    a no-op. Called from app.py's startup hook."""
    global _PROBE_TASK
    if _PROBE_TASK and not _PROBE_TASK.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    _PROBE_TASK = loop.create_task(_periodic_loop())


def stop_periodic_probe() -> None:
    """Cancel the background sweep. Called from app shutdown so the
    test runner doesn't see a stranded task."""
    global _PROBE_TASK
    if _PROBE_TASK and not _PROBE_TASK.done():
        _PROBE_TASK.cancel()
    _PROBE_TASK = None


# ---------------------------------------------------------------------------
# Routing: pick a worker for a given workload.
#
# These helpers translate "I need to run an embed / chat / subagent call"
# into "send it to base URL X with bearer Y", or `None` to mean "no
# eligible worker — the host should handle it locally". Everything is
# read-only relative to the worker rows; the routing layer never mutates
# state.
#
# Eligibility for any workload:
#   * row is `enabled`
#   * the workload-specific flag is on (`use_for_embeddings`, etc.)
#   * the last probe succeeded — `last_seen` within `_FRESHNESS_SEC` AND
#     `last_error` is empty. A worker that hasn't been probed yet (last_seen
#     is None) is skipped: we don't want to trust a row the user just
#     added until the periodic sweep — or a manual "Test connection" —
#     confirms it can serve traffic.
#   * if `model` is supplied, the worker's cached capabilities list it as
#     installed (with or without a `:latest` tag suffix).
# ---------------------------------------------------------------------------

# How long after `last_seen` we still consider a worker "fresh enough" to
# route to. The periodic probe runs every 5 min; this 1-hour window covers
# 12 sweeps' worth of buffer for transient blips.
_FRESHNESS_SEC = 60 * 60


def _model_matches(installed_name: str, requested: str) -> bool:
    """Compare an installed model name to a requested name, tolerant to
    Ollama's `:latest` tag default.

    `nomic-embed-text` matches `nomic-embed-text:latest`, and an explicit
    tag (`gemma4:e4b`) matches itself. Mismatched explicit tags are NOT
    coerced — `gemma4:e4b` does not match `gemma4:e2b`.
    """
    if not installed_name or not requested:
        return False
    if installed_name == requested:
        return True
    # Strip `:latest` from either side and compare bare names.
    inst_bare = installed_name.split(":", 1)[0] if ":" in installed_name else installed_name
    req_bare = requested.split(":", 1)[0] if ":" in requested else requested
    inst_has_explicit_tag = ":" in installed_name and not installed_name.endswith(":latest")
    req_has_explicit_tag = ":" in requested and not requested.endswith(":latest")
    # If both sides have explicit tags, they had to match exactly above.
    if inst_has_explicit_tag and req_has_explicit_tag:
        return False
    return inst_bare == req_bare


def _worker_has_model(worker: dict, model: str) -> bool:
    caps = worker.get("capabilities") or {}
    for m in caps.get("models") or []:
        if _model_matches(m.get("name") or "", model):
            return True
    return False


def _is_fresh(worker: dict, now: float | None = None) -> bool:
    """Last successful probe within `_FRESHNESS_SEC`."""
    last_seen = worker.get("last_seen")
    if not last_seen:
        return False
    if worker.get("last_error"):
        return False
    if now is None:
        now = time.time()
    return (now - float(last_seen)) < _FRESHNESS_SEC


def _eligible_workers(flag: str, model: str | None = None) -> list[dict]:
    """Return enabled+fresh workers whose `flag` is on and (optionally) have
    `model` installed. Sorted newest-`last_seen` first so the picker grabs
    the most-recently-confirmed-healthy worker."""
    rows = db.list_compute_workers(enabled_only=True)
    now = time.time()
    out: list[dict] = []
    for w in rows:
        if not w.get(flag):
            continue
        if not _is_fresh(w, now=now):
            continue
        if model and not _worker_has_model(w, model):
            continue
        out.append(w)
    out.sort(key=lambda w: float(w.get("last_seen") or 0), reverse=True)
    return out


def pick_embed_target(model: str) -> tuple[str, str | None] | None:
    """Choose a worker to run an embed request against, or None for host.

    Returns `(base_url, auth_token_or_None)`. `auth_token` is fetched
    from the dedicated `get_compute_worker_auth_token` so the token never
    sits on a row dict. Caller composes the URL as `f"{base}/api/embeddings"`
    and adds `Authorization: Bearer …` when the token is set.
    """
    cands = _eligible_workers("use_for_embeddings", model=model)
    if not cands:
        return None
    w = cands[0]
    base = _worker_base_url(w)
    token = db.get_compute_worker_auth_token(w["id"])
    return (base, token)


def pick_chat_target(model: str) -> tuple[str, str | None] | None:
    """Choose a worker to run a single chat turn against, or None for host.

    Same eligibility rules as the embed picker, but gated on `use_for_chat`.
    Picks the freshest eligible worker so the route is deterministic for
    a given moment (helps tests; in practice with 1-2 workers a typical
    home setup has, fairness over multiple turns falls out of normal
    request timing rather than needing explicit round-robin).
    """
    cands = _eligible_workers("use_for_chat", model=model)
    if not cands:
        return None
    w = cands[0]
    return (_worker_base_url(w), db.get_compute_worker_auth_token(w["id"]))


def pick_split_chat_target(model_name: str) -> tuple[str, str] | None:
    """Resolve a conversation's model name to a running llama-server, or None.

    Convention: split-model conversations carry a model field of the form
    `split:<label>`. We strip the prefix, look up the matching split_models
    row, and — if its status is `running` — return `(base_url, label)`.
    Caller composes `<base_url>/v1/chat/completions` and uses `label` as
    the OpenAI-style `model` field in the POST body (llama-server doesn't
    care what you put there; one server, one loaded model).

    Returns None for:
      * Anything not prefixed with `split:` — that's a normal Ollama model.
      * A `split:<label>` whose row doesn't exist (label was renamed,
        deleted, etc.).
      * A row whose status is anything other than `running` — caller
        should NOT route chat to a stopped/loading server.
    """
    if not model_name or not model_name.startswith("split:"):
        return None
    label = model_name[len("split:"):].strip()
    if not label:
        return None
    # Linear scan over split_models — the table is tiny (typical user
    # has 0-5 rows) so an index would be over-engineering.
    for row in db.list_split_models(enabled_only=True):
        if row.get("label") == label and row.get("status") == "running":
            return (f"http://127.0.0.1:{row['llama_port']}", label)
    return None


def list_subagent_workers(model: str) -> list[tuple[str, str | None]]:
    """Return every eligible compute worker for parallel-subagent fan-out.

    The host itself is NOT in this list — the caller (`run_subagents_parallel`)
    composes `[host] + workers` and round-robins, so a 6-task fan-out across
    1 host + 2 workers schedules roughly 2 per machine. Returning workers
    only here keeps `compute_pool` host-agnostic; agent.py owns the host URL.

    Eligibility uses the same rules as embed routing: enabled, flag on,
    fresh probe, model installed. Ordered freshest-first so a tied
    distribution at least picks the healthier worker as the first
    non-host slot.
    """
    cands = _eligible_workers("use_for_subagents", model=model)
    return [
        (_worker_base_url(w), db.get_compute_worker_auth_token(w["id"]))
        for w in cands
    ]
