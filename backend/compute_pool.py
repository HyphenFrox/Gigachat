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
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

from . import db, sysdetect

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
    """Issue the probe GETs in parallel and merge the results.

    Returns a dict with `version`, `models`, hardware-capability hints,
    and any `*_error` markers. Any single endpoint may fail without
    breaking the others — we want the most signal we can get out of
    one round trip.

    Hardware-capability detection: workers run plain Ollama so they
    don't volunteer system specs. The closest signal Ollama exposes is
    `/api/ps` — currently-loaded models with their VRAM split. From
    that we infer:
      * `gpu_present`: True if any loaded model reports `size_vram > 0`.
        A worker with no loaded models doesn't tell us anything yet,
        which is why we treat absence as `False` rather than `None`.
      * `max_vram_seen_bytes`: the largest `size_vram` across loaded
        models, giving a rough lower bound on the worker's VRAM. The
        router uses this to prefer hardware-stronger workers when
        ranking ties.
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

    async def _ps() -> Any:
        r = await client.get(f"{base}/api/ps", headers=headers)
        r.raise_for_status()
        return r.json()

    out: dict[str, Any] = {}
    # gather() with return_exceptions so a partial failure on one
    # endpoint doesn't lose the other endpoints' payloads.
    ver_res, tags_res, ps_res = await asyncio.gather(
        _ver(), _tags(), _ps(), return_exceptions=True,
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
    # /api/ps is purely a heuristic source — failures shouldn't dim the
    # probe's overall verdict, just leave the hardware fields empty so
    # the router falls through to the no-info default.
    if isinstance(ps_res, Exception):
        out["gpu_present"] = False
        out["max_vram_seen_bytes"] = 0
        out["loaded_count"] = 0
    else:
        loaded = (ps_res or {}).get("models", []) or []
        max_vram = 0
        any_gpu = False
        for m in loaded:
            v = int(m.get("size_vram") or 0)
            if v > 0:
                any_gpu = True
            if v > max_vram:
                max_vram = v
        out["gpu_present"] = any_gpu
        out["max_vram_seen_bytes"] = max_vram
        out["loaded_count"] = len(loaded)
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
        # Hardware hints — best-effort, populated from /api/ps. Used by
        # the router to prefer hardware-stronger workers over weaker
        # ones when both are otherwise eligible.
        "gpu_present": bool(payload.get("gpu_present")),
        "max_vram_seen_bytes": int(payload.get("max_vram_seen_bytes") or 0),
        "loaded_count": int(payload.get("loaded_count") or 0),
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


def _capability_score(worker: dict) -> tuple:
    """Power-ranking key for sorting eligible workers.

    Returns a tuple suitable for `sorted(key=...)` — bigger is better,
    so callers use `reverse=True`. Components, in priority order:

      1. `gpu_present` (1 if the worker has a usable GPU). Inferred
         best-effort from `/api/ps`'s `size_vram > 0`. Heavily-weighted
         because GPU is dramatically faster than CPU for inference.
      2. `max_vram_seen_bytes`. Lower bound on the worker's VRAM —
         a worker that's loaded a 14 GB model has at least 14 GB. Among
         GPU workers, more VRAM wins.
      3. Freshness (last_seen). Tie-breaker.

    Workers with NO loaded-model history (`loaded_count==0`) have all-
    zero hardware fields. They sort to the bottom — not because they
    lack hardware, but because we can't see their hardware yet. The
    next probe after they load anything will fix the ranking.
    """
    caps = worker.get("capabilities") or {}
    return (
        1 if caps.get("gpu_present") else 0,
        int(caps.get("max_vram_seen_bytes") or 0),
        float(worker.get("last_seen") or 0),
    )


def _eligible_workers(flag: str, model: str | None = None) -> list[dict]:
    """Return enabled+fresh workers whose `flag` is on and (optionally) have
    `model` installed.

    Sort order (`_capability_score` desc): GPU-present workers first,
    then by max VRAM observed, then freshest probe. The user's
    intuition "use the more powerful machine" maps to (1) → GPU wins
    over CPU-only, (2) → more VRAM wins over less. Tie at hardware →
    fall back to freshest, like Phase 1 commit 6's original behavior.

    Auto-sync side effect: when a worker has `ssh_host` set, is
    enabled+fresh, and has the right `flag` on, but is missing the
    model, this function kicks off a background SCP from host so the
    model lands on the worker without the user touching a button. The
    current call still excludes that worker (it really doesn't have
    the model right now), but subsequent probes will see the new model
    and the worker becomes eligible. Failures are silent — auto-sync
    is best-effort.
    """
    rows = db.list_compute_workers(enabled_only=True)
    now = time.time()
    out: list[dict] = []
    for w in rows:
        if not w.get(flag):
            continue
        if not _is_fresh(w, now=now):
            continue
        if model:
            if _worker_has_model(w, model):
                out.append(w)
            elif w.get("ssh_host"):
                _maybe_kickoff_lan_sync(w, model)
            # else: no ssh_host, no path to install — skip silently.
        else:
            out.append(w)
    out.sort(key=_capability_score, reverse=True)
    return out


# Track in-flight auto-syncs so we don't fire the same SCP twice if
# `_eligible_workers` runs back-to-back (every chat turn). Keyed by
# (worker_id, model_name); value is a Task we never await but keep
# referenced so it doesn't get GC'd mid-flight.
_AUTO_SYNC_TASKS: dict[tuple[str, str], asyncio.Task] = {}


def _maybe_kickoff_lan_sync(worker: dict, model_name: str) -> None:
    """Launch a background SCP of `model_name` from host to `worker`.

    Called from the routing layer when a worker is *almost* eligible —
    same hardware/freshness/flag eligibility but missing the model. The
    auto-sync transparently fixes the model gap; subsequent routes get
    a more capable choice.

    Guards:
      * ssh_host must be set on the worker (the only way we can reach it).
      * Skip if the same (worker, model) is already mid-sync.
      * Wrapped in try/except — sync failures NEVER bubble to the caller.
    """
    wid = worker.get("id")
    if not wid or not model_name:
        return
    key = (wid, model_name)
    existing = _AUTO_SYNC_TASKS.get(key)
    if existing and not existing.done():
        return  # already in flight; let it finish

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Sync entry point — no event loop, skip. The next async caller
        # will retry.
        return

    async def _run():
        # Local import to dodge the import cycle (model_sync depends
        # on this module for `_model_matches`).
        try:
            from . import model_sync
            await model_sync.sync_model(model_name, wid)
            log.info("compute_pool: auto-synced %s → worker %s",
                     model_name, worker.get("label"))
        except Exception as e:
            log.info("compute_pool: auto-sync %s → %s deferred (%s)",
                     model_name, worker.get("label"), e)
        finally:
            _AUTO_SYNC_TASKS.pop(key, None)

    _AUTO_SYNC_TASKS[key] = loop.create_task(_run())


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
    """Legacy lookup retained for back-compat with tests / explicit
    `split:<label>` model names. The auto-router below
    (`route_chat_for`) is the live entry point now; it never produces
    `split:` prefixes — it just inspects the model and decides whether
    to engage the split path transparently.

    Returns `(base_url, label)` for an explicit `split:<label>` whose
    row is `running`; None otherwise.
    """
    if not model_name or not model_name.startswith("split:"):
        return None
    label = model_name[len("split:"):].strip()
    if not label:
        return None
    for row in db.list_split_models(enabled_only=True):
        if row.get("label") == label and row.get("status") == "running":
            return (f"http://127.0.0.1:{row['llama_port']}", label)
    return None


# ---------------------------------------------------------------------------
# Auto-routing: pick host-Ollama vs spawn-llama-server-with-RPC for a model
# ---------------------------------------------------------------------------
# This is the intelligent split-or-not decision the user actually picks
# their model against. They never see `split:<label>` — they pick e.g.
# `gemma3:27b` from the model picker, and the router decides:
#
#   1. Resolve the model's GGUF blob + size from Ollama's manifest store.
#   2. If size <= host_vram_budget → Ollama on host (fastest path; 0 LAN
#      overhead; workers stay free for parallel embeddings/subagents).
#   3. Else → ensure a llama-server is running for THIS exact model with
#      --rpc to every eligible compute worker, return its URL.
#   4. If the model can't fit even with the combined pool → raise so the
#      user gets a clear error rather than silent OOM.
#
# Per-conversation lifecycle: when the user switches between two big
# models, we stop the previously-running llama-server (one big model hot
# at a time — we have one finite VRAM budget). Switching back to a small
# model also stops any running llama-server so its VRAM is freed for
# Ollama.
# ---------------------------------------------------------------------------

# How much of the host's VRAM we're willing to let one model occupy
# without engaging the split path. 85% leaves headroom for the OS, the
# desktop compositor, any other Ollama models loaded simultaneously, and
# the model's KV cache. Below this fraction → Ollama. Above → split.
_HOST_VRAM_USE_FRACTION = 0.85

# Path to Ollama's on-disk model store. Default location on every
# platform Ollama supports; matches what `ollama_runtime` assumes.
_OLLAMA_MODELS_DIR = Path.home() / ".ollama" / "models"


def _resolve_ollama_manifest(model_name: str) -> dict | None:
    """Locate the manifest JSON for an Ollama model name.

    Ollama stores manifests at
        ~/.ollama/models/manifests/<registry>/<namespace>/<name>/<tag>
    The default registry is `registry.ollama.ai` and the default namespace
    is `library`. Custom registries / namespaces are rare for end users
    but we still walk the tree to find the file.

    Returns the parsed manifest dict, or None if no matching file exists
    (model not pulled, name typo, etc.).
    """
    name = (model_name or "").strip()
    if not name:
        return None
    # Tag defaults to `latest` when omitted.
    if ":" in name:
        bare, tag = name.split(":", 1)
    else:
        bare, tag = name, "latest"

    # Default location first — covers >99% of cases.
    candidate = _OLLAMA_MODELS_DIR / "manifests" / "registry.ollama.ai" / "library" / bare / tag
    if candidate.is_file():
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            return None

    # Fallback: walk the manifests tree. Slower but handles the user
    # who pulled a model from a non-default registry.
    manifests = _OLLAMA_MODELS_DIR / "manifests"
    if not manifests.is_dir():
        return None
    needle = (bare, tag)
    for root, _dirs, files in os.walk(manifests):
        for fname in files:
            if fname == tag and Path(root).name == bare:
                try:
                    return json.loads(Path(root, fname).read_text(encoding="utf-8"))
                except Exception:
                    continue
    return None


def resolve_ollama_model(model_name: str) -> dict | None:
    """Resolve a model name to its on-disk GGUF + size.

    Returns `{gguf_path, size_bytes, manifest}` or None if the manifest
    can't be found. The "model" layer of the manifest is the GGUF; we
    compute size from `layers[].size` for the layer whose mediaType is
    `application/vnd.ollama.image.model` (large) — license / params /
    template layers are tiny config blobs and don't count toward VRAM.

    `gguf_path` is the absolute path to the blob, suitable for passing
    as llama-server's `--model` flag.
    """
    manifest = _resolve_ollama_manifest(model_name)
    if not manifest:
        return None
    layers = manifest.get("layers") or []
    model_layer = None
    for layer in layers:
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            model_layer = layer
            break
    if not model_layer:
        return None
    digest = (model_layer.get("digest") or "").replace("sha256:", "")
    if not digest:
        return None
    blob_path = _OLLAMA_MODELS_DIR / "blobs" / f"sha256-{digest}"
    if not blob_path.is_file():
        return None
    return {
        "gguf_path": str(blob_path),
        "size_bytes": int(model_layer.get("size") or 0),
        "manifest": manifest,
    }


def _host_vram_budget_bytes() -> int:
    """Bytes of host VRAM we're willing to let a single Ollama-loaded
    model occupy. Below this → Ollama; above → engage split path."""
    try:
        spec = sysdetect.detect_system()
        vram_gb = float(spec.get("vram_gb") or 0.0)
    except Exception:
        vram_gb = 0.0
    return int(vram_gb * _HOST_VRAM_USE_FRACTION * 1024 * 1024 * 1024)


def _eligible_split_workers() -> list[dict]:
    """Workers that can contribute layers via rpc-server.

    Same `enabled` + `use_for_chat` gate as Phase 1's chat picker —
    workers the user toggled off for chat shouldn't suddenly start
    receiving inference traffic just because the model needs splitting.
    Plus the rpc-specific gate: probe must report rpc-server reachable.
    """
    rows = db.list_compute_workers(enabled_only=True)
    out = []
    for w in rows:
        if not w.get("use_for_chat"):
            continue
        caps = w.get("capabilities") or {}
        if not caps.get("rpc_server_reachable"):
            continue
        out.append(w)
    # Freshest probe first — if we have to pick a subset later, we'd
    # rather use the worker we just confirmed alive than one whose
    # last_seen is an hour stale.
    out.sort(key=lambda w: float(w.get("last_seen") or 0), reverse=True)
    return out


async def _ensure_split_running_for(
    model_name: str, gguf_path: str, worker_ids: list[str],
) -> str:
    """Idempotent: ensure a `split_models` row exists + is running for
    this exact (model_name, gguf_path) pair, then return its base_url.

    Auto-creates the row keyed by model_name as label. If a row with the
    same label already exists, we reuse it (updating worker_ids if the
    user added/removed workers between turns). If a DIFFERENT split row
    is currently running, we stop it first — only one big model hot at
    a time.
    """
    # Local import to dodge the circular dep — split_lifecycle imports
    # compute_pool indirectly via db / runtime.
    from . import split_lifecycle

    rows = db.list_split_models()
    target_row = next((r for r in rows if r.get("label") == model_name), None)

    # Stop any other running split row — finite VRAM means one big
    # model active at a time.
    for r in rows:
        if r["id"] != (target_row or {}).get("id") and r.get("status") in ("running", "loading"):
            try:
                await split_lifecycle.stop(r["id"])
            except Exception as e:
                log.warning("compute_pool: failed to stop %s: %s", r["id"], e)

    if target_row is None:
        sid = db.create_split_model(
            label=model_name,
            gguf_path=gguf_path,
            worker_ids=worker_ids,
        )
        target_row = db.get_split_model(sid)
    else:
        # Refresh gguf_path + worker_ids in case the user changed
        # things since the previous turn.
        if target_row.get("gguf_path") != gguf_path or target_row.get("worker_ids") != worker_ids:
            db.update_split_model(
                target_row["id"],
                gguf_path=gguf_path,
                worker_ids=worker_ids,
            )
            target_row = db.get_split_model(target_row["id"])

    sid = target_row["id"]
    if target_row.get("status") != "running":
        result = await split_lifecycle.start(sid)
        if not result.get("ok"):
            raise RuntimeError(
                f"failed to start llama-server for {model_name}: "
                f"{result.get('error') or 'unknown'}"
            )

    fresh = db.get_split_model(sid)
    return f"http://127.0.0.1:{fresh['llama_port']}"


async def stop_all_running_splits() -> None:
    """Free VRAM held by any running llama-server. Called when the
    router decides the upcoming chat turn fits Ollama on host alone —
    no point keeping a big-model llama-server warm if the active
    conversation no longer needs it."""
    from . import split_lifecycle

    for r in db.list_split_models():
        if r.get("status") in ("running", "loading"):
            try:
                await split_lifecycle.stop(r["id"])
            except Exception as e:
                log.warning("compute_pool: stop_all_running_splits %s: %s", r["id"], e)


class RouteChatError(RuntimeError):
    """Raised by route_chat_for when the model can't be served at all
    — e.g. the model file isn't present, or the combined pool is too
    small to hold it. Caller should surface this to the user."""


async def route_chat_for(model_name: str) -> dict:
    """Pick the right inference engine for this model + drive any
    needed lifecycle changes.

    Returns one of:
      {"engine": "ollama"}
        → use the existing Ollama path on host (or a Phase 1 worker
          via `pick_chat_target`). Caller continues with
          `_stream_ollama_chat`.

      {"engine": "llama_server", "base_url": "http://127.0.0.1:NNNN", "label": <model>}
        → llama-server with --rpc was spawned (or already running) for
          this model. Caller uses `_stream_llama_server_chat` instead.

    Raises `RouteChatError` for unrecoverable cases. Side effects:
    may stop a currently-running llama-server (model switch) and may
    auto-create a `split_models` row.
    """
    # Cheap reject: explicit `split:` prefix is back-compat — short-
    # circuit to the legacy picker so existing tests still pass.
    legacy = pick_split_chat_target(model_name)
    if legacy is not None:
        base, label = legacy
        return {"engine": "llama_server", "base_url": base, "label": label}

    info = resolve_ollama_model(model_name)
    if info is None:
        # Not an Ollama-managed model. Could be a custom name the user
        # set up another way. Stay on the Ollama path — Ollama will
        # surface its own error if the model truly doesn't exist.
        await stop_all_running_splits()
        return {"engine": "ollama"}

    size_bytes = info["size_bytes"]
    host_budget = _host_vram_budget_bytes()
    if host_budget > 0 and size_bytes <= host_budget:
        # Fits comfortably on host. Make sure no llama-server is
        # holding VRAM we'll need.
        await stop_all_running_splits()
        return {"engine": "ollama"}

    # Need the split path. Find eligible workers.
    workers = _eligible_split_workers()
    if not workers:
        # Without workers, the only thing we could do is run on host
        # with CPU offload — Ollama already does that automatically,
        # so falling through to the Ollama path is the right call. If
        # that fails the user gets Ollama's actual error message.
        await stop_all_running_splits()
        return {"engine": "ollama"}

    worker_ids = [w["id"] for w in workers]
    base_url = await _ensure_split_running_for(model_name, info["gguf_path"], worker_ids)
    return {"engine": "llama_server", "base_url": base_url, "label": model_name}


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
