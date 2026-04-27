"""Periodic ngl re-evaluation for a running workload.

Background task that watches the pool's free-memory budget and asks
the wrapper to respawn the managed `llama-server` with a new `-ngl`
when the optimal value has drifted past a threshold AND a cooldown
has elapsed. The expensive part — subprocess kill + cold-load +
/health probe — is gated, so a noisy memory environment doesn't
trigger a respawn storm.

What "meaningful change" means
------------------------------
A respawn is triggered ONLY when ALL three are true:

  1. ``new_ngl`` differs from ``current_ngl`` by at least
     ``threshold_layers`` (default 3) — small drifts (e.g. 14 → 15)
     aren't worth a cold-load.
  2. At least ``cooldown_s`` seconds (default 300 = 5 min) have
     elapsed since the last respawn — protects against rapid
     oscillation when free memory is bouncing around a boundary.
  3. The resulting ``new_ngl`` is non-zero — we don't respawn into
     a "give up on the pool" state mid-flight; if the budget
     collapses completely, the workload keeps running with
     whatever layers it already placed.

Resource source
---------------
Reads worker free-memory from `~/.llamapool/config.json`, refreshed
by whatever populates that file (e.g. Gigachat's
`backend/llamapool_sync.py` runs on every CRUD operation + every
5-min capability probe). For purely-standalone use without Gigachat,
keep the JSON fresh via your own probe loop.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Awaitable, Callable

import config
import ngl
import registry

log = logging.getLogger(__name__)


class RebalanceWatcher:
    """Periodic ngl recomputation + threshold-gated respawn.

    Construction is cheap — the work happens in `start()` which
    schedules the watcher coroutine on the running event loop.
    Call `stop()` to cancel cleanly (e.g. on wrapper shutdown).
    """

    def __init__(
        self,
        *,
        gguf_path: str,
        claim_id: str,
        priority: int,
        initial_ngl: int,
        respawn: Callable[[int], Awaitable[None]],
        interval_s: float = 60.0,
        threshold_layers: int = 3,
        cooldown_s: float = 300.0,
        aggressive: bool = False,
    ) -> None:
        self.gguf_path = gguf_path
        self.claim_id = claim_id
        self.priority = priority
        self.respawn = respawn
        self.interval_s = interval_s
        self.threshold_layers = threshold_layers
        self.cooldown_s = cooldown_s
        # Carry the same OS-cooperation mode the initial engage used
        # — rebalance ticks must compute against the same memory
        # source so the math stays consistent across respawns.
        self.aggressive = aggressive
        # Seed last_ngl with what llama-server is actually running with;
        # first tick compares against this.
        self.last_ngl: int = int(initial_ngl)
        self.last_respawn_at = time.monotonic()
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        """Schedule the watcher loop on the current event loop."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="llamapool-rebalance")
        log.info(
            "rebalance watcher started: interval=%.0fs threshold=%d cooldown=%.0fs",
            self.interval_s, self.threshold_layers, self.cooldown_s,
        )

    async def stop(self) -> None:
        """Stop the watcher and wait for it to settle."""
        self._stop.set()
        if self._task is None:
            return
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._task.cancel()

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                # Wait for either the next tick or a stop signal.
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.interval_s,
                )
                return
            except asyncio.TimeoutError:
                pass  # interval elapsed -> tick
            try:
                await self._tick()
            except Exception as e:
                # Never let a tick error kill the loop; log and continue.
                log.debug("rebalance tick error: %s", e)

    async def _tick(self) -> None:
        """Recompute optimal ngl; trigger respawn iff thresholds pass."""
        ws = config.list_workers(enabled_only=True)
        active = registry.get_active_claims()
        my_pid = os.getpid()
        other_claims = [
            c for c in active if int(c.get("pid") or 0) != my_pid
        ]
        reserved = sum(
            int(c.get("estimated_bytes") or 0) for c in other_claims
        )
        other_priorities = [
            int(c.get("priority") or 100) for c in other_claims
        ]
        try:
            new_ngl, new_est = ngl.compute_optimal_ngl(
                self.gguf_path, ws,
                reserved_bytes=reserved,
                my_priority=self.priority,
                other_priorities=other_priorities,
                aggressive=self.aggressive,
            )
        except Exception as e:
            log.debug("ngl recompute failed: %s", e)
            return

        delta = abs(new_ngl - self.last_ngl)
        elapsed = time.monotonic() - self.last_respawn_at
        gated = []
        if delta < self.threshold_layers:
            gated.append(f"delta {delta}<{self.threshold_layers}")
        if elapsed < self.cooldown_s:
            gated.append(f"cooldown {elapsed:.0f}s<{self.cooldown_s:.0f}s")
        # Don't respawn into a host-only fallback (ngl=0); keep the
        # current allocation rather than tearing everything down.
        if new_ngl == 0:
            gated.append("new_ngl==0")
        if gated:
            log.debug(
                "rebalance: ngl %d->%d skipped (%s)",
                self.last_ngl, new_ngl, ", ".join(gated),
            )
            return

        log.info(
            "rebalance: ngl %d -> %d (delta=%d) — respawning",
            self.last_ngl, new_ngl, delta,
        )
        try:
            await self.respawn(new_ngl)
        except Exception as e:
            log.warning("rebalance respawn failed: %s; keeping current ngl", e)
            return

        # Update the claim's estimated_bytes so peers see our new
        # reservation footprint and adjust their own shares.
        try:
            self._update_claim_estimate(new_ngl, new_est)
        except Exception as e:
            log.debug("post-respawn registry update failed: %s", e)

        self.last_ngl = new_ngl
        self.last_respawn_at = time.monotonic()

    def _update_claim_estimate(self, new_ngl: int, new_est: int) -> None:
        """Mutate this PID's claim entry to reflect the new ngl."""
        from registry import _acquire_lock, _load_unlocked, _save_unlocked
        with _acquire_lock():
            state = _load_unlocked()
            for c in state.get("claims", []) or []:
                if c.get("id") == self.claim_id:
                    c["ngl"] = int(new_ngl)
                    c["estimated_bytes"] = int(new_est)
                    break
            _save_unlocked(state)
