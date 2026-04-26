"""Engagement orchestrator: tie config + workers + ngl + multi-tenant
registry into a single call any llama.cpp-based app can use.

Public entry points:

  * `engage(gguf_path, *, priority=100, in_split=True)` -
    returns `(rpc_endpoints, ngl, claim_id)`. The caller spawns its
    own `llama-server` with these values, then passes `claim_id`
    back to `disengage(claim_id)` on shutdown so the pool memory
    is freed for the next workload.

    Multi-tenant safe: under contention, pool memory is split
    proportionally by `priority`. Defaults to 100; pass higher for
    user-facing work, lower for background batch jobs.

  * `engage_async / disengage_async` - async variants for callers
    already in an event loop.

  * `get_workers()` - read-only snapshot of registered workers.

  * `set_workers_backend(workers, *, in_split)` - re-exported from
    the workers submodule for direct use.

Failure model
-------------
If pool can't fit even our priority-weighted share of one layer
(e.g., another big workload is using all the memory), `engage()`
returns empty rpc + ngl=0 so the caller falls through to host-only
inference rather than overcommitting and crashing.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import config
import ngl
import registry
import workers

log = logging.getLogger(__name__)

# Re-export for caller convenience.
set_workers_backend = workers.set_workers_backend


def get_workers(*, enabled_only: bool = True) -> list[dict[str, Any]]:
    """Return the configured workers list."""
    return config.list_workers(enabled_only=enabled_only)


def _persist_worker_backends(ws: list[dict[str, Any]]) -> None:
    """After `workers.set_workers_backend` mutates `current_rpc_backend`
    on each entry, mirror those values back into `~/.llamapool/config.json`
    so the *next* engage() can short-circuit when backend already
    matches."""
    try:
        cfg = config.load_config()
        labels = {w["label"]: w for w in ws if "current_rpc_backend" in w}
        for entry in cfg.get("workers", []):
            mirror = labels.get(entry.get("label"))
            if mirror is not None:
                entry["current_rpc_backend"] = mirror["current_rpc_backend"]
        config.save_config(cfg)
    except Exception as e:
        log.debug("could not persist worker backends: %s", e)


async def engage_async(
    gguf_path: str | None = None,
    *,
    in_split: bool = True,
    priority: int = 100,
) -> tuple[list[str], int | None, str | None]:
    """Async core of `engage`. See `engage` for semantics.

    Returns `(rpc_endpoints, ngl_or_None, claim_id_or_None)`:
      * `rpc_endpoints`: list of `host:port` strings for `--rpc`.
        Empty when no workers reachable OR pool can't fit our share.
      * `ngl`: int for `-ngl`, or None if `gguf_path` was not
        provided (caller should use llama.cpp's default).
      * `claim_id`: opaque token to pass to `disengage(claim_id)`.
        None if no claim was registered (e.g., empty pool).
    """
    ws = config.list_workers(enabled_only=True)
    if not ws:
        log.info("no workers configured; running on host alone")
        return [], None, None

    # Align worker backends iff a split is needed AND no other split
    # is already in flight (a peer workload may already have set the
    # right backend; don't bounce its rpc-server out from under it).
    if in_split:
        try:
            if not registry.any_split_active(exclude_pid=os.getpid()):
                await workers.set_workers_backend(ws, in_split=True)
                _persist_worker_backends(ws)
            else:
                log.info("peer claim already in split; skipping backend bounce")
        except Exception as e:
            log.warning("backend alignment failed: %s; continuing", e)

    # Build --rpc list from reachable workers only.
    rpc_endpoints = workers.resolve_rpc_endpoints(ws)
    if not rpc_endpoints:
        log.info("no rpc endpoints reachable; running on host alone")
        return [], None, None

    # Compute adaptive ngl, accounting for what other live workloads
    # have already reserved.
    ngl_value: int | None = None
    estimated_bytes = 0
    if gguf_path:
        my_pid = os.getpid()
        active = registry.get_active_claims()
        reserved = sum(
            int(c.get("estimated_bytes") or 0)
            for c in active
            if int(c.get("pid") or 0) != my_pid
        )
        other_priorities = [
            int(c.get("priority") or 100)
            for c in active
            if int(c.get("pid") or 0) != my_pid
        ]
        try:
            ngl_value, estimated_bytes = ngl.compute_optimal_ngl(
                gguf_path, ws,
                reserved_bytes=reserved,
                my_priority=priority,
                other_priorities=other_priorities,
            )
        except Exception as e:
            log.warning("adaptive ngl computation failed: %s", e)
            ngl_value, estimated_bytes = None, 0

        # Pool can't fit our share — get out of the way.
        if ngl_value == 0:
            log.info("pool can't fit our share (reserved=%.2fGB); "
                     "running host-only", reserved / (1024 ** 3))
            return [], 0, None

    # Register the claim so subsequent engage() calls account for it.
    claim_id = registry.register_claim(
        in_split=in_split,
        gguf_path=gguf_path,
        ngl=ngl_value,
        estimated_bytes=estimated_bytes,
        rpc_endpoints=rpc_endpoints,
        priority=priority,
    )
    return rpc_endpoints, ngl_value, claim_id


def engage(
    gguf_path: str | None = None,
    *,
    in_split: bool = True,
    priority: int = 100,
) -> tuple[list[str], int | None, str | None]:
    """Synchronous wrapper around `engage_async` for callers not
    already in an event loop. See `engage_async` for semantics.
    """
    return asyncio.run(
        engage_async(gguf_path, in_split=in_split, priority=priority),
    )


async def disengage_async(claim_id: str | None = None) -> None:
    """Release a claim and, if it was the last one, restore worker
    rpc-servers to idle-mode backend (Intel iGPU back to SYCL+CPU).

    Best-effort: errors are logged but not raised. Safe to call
    multiple times — unregistering a missing claim is a no-op.
    """
    if claim_id:
        registry.unregister_claim(claim_id)

    # Only restore idle backends if no other split claim remains.
    if registry.any_split_active(exclude_pid=os.getpid()):
        log.info("peer claim still in split; leaving worker backends as-is")
        return

    ws = config.list_workers(enabled_only=True)
    if not ws:
        return
    try:
        await workers.set_workers_backend(ws, in_split=False)
        _persist_worker_backends(ws)
    except Exception as e:
        log.warning("disengage failed: %s", e)


def disengage(claim_id: str | None = None) -> None:
    """Synchronous wrapper around `disengage_async`."""
    asyncio.run(disengage_async(claim_id))
