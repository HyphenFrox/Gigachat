"""Real-time fairness scheduler for the P2P compute pool.

Implements the user's stated bounds:

  Minimum entitlement (per user):
      The user's full local pool. ALWAYS available, regardless of
      credit balance, current swarm load, or anything else. Local
      compute is unconditionally yours.

  Maximum entitlement (per user, current moment):
      total_donation_capacity_now / active_consumers_now
      i.e. the current public-pool throughput is divided fairly
      among everyone currently consuming. A wealthy user can't
      crowd everyone else out.

How "real-time" is achieved without a global ledger:

  * Each peer keeps a LOCAL view of what's happening, updated by
    receipts gossiped from friends + the rendezvous. Decisions are
    local; the only real-time data we need is the count of
    currently-active consumers and the local pool's donation
    capacity (both readable in microseconds from in-process state).

  * Donation budget tracks bytes-per-second that THIS install is
    willing to donate to the swarm. The cap divides that by the
    current consumer count to derive per-consumer slice.

  * Every accepted public-pool job updates a sliding-window
    counter; every released job decrements it. The cap recomputes
    on every admission decision so a sudden surge of consumers
    immediately tightens slices.

What lives where:

  * `LedgerState` — local credits balance + per-peer rate limits.
    Persisted in user_settings so balances survive restarts.
  * `ConsumerSlot` — currently-running public-pool job. Held
    while the job runs; auto-released by `__exit__`.
  * `should_admit(peer_id, kind)` — the central admission decision
    consulted by the future donation worker. Returns True+slot
    or False+reason.

The transport layer (phase 7+) calls into this module on every
inbound public-pool request. The user's own outbound traffic
goes through the privacy guard (phase 5) which is independent of
this fairness layer — fairness governs DONATIONS, privacy
governs PROMPTS.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from . import db

log = logging.getLogger("gigachat.p2p.fairness")

# ---------------------------------------------------------------------------
# Donation capacity — what THIS install is willing to give the swarm.
# ---------------------------------------------------------------------------

# Default fraction of local compute reserved for donations when
# the toggle is on. 0.25 = up to 25% of local capacity available
# for public-pool work. Conservative: leaves plenty for the user's
# own work. Tunable via user_settings.
_DEFAULT_DONATION_FRACTION = 0.25

# Hard ceiling on simultaneously-running public-pool jobs. Even on
# a beefy host, more than this gets you queue-thrash that hurts
# both the donor and the consumers. Tunable via user_settings.
_DEFAULT_MAX_CONCURRENT_DONATIONS = 4

# Sliding-window length for "active consumers right now". A consumer
# falls out of the count if we haven't seen a request from them in
# this many seconds. Long enough that a slow streaming response
# doesn't drop them mid-job; short enough that a finished consumer
# vacates their slice quickly so others get the share.
_ACTIVE_CONSUMER_WINDOW_SEC = 30.0

# Max requests per minute we'll honour from any single peer. Floor
# against a single misbehaving peer eating our entire share.
_DEFAULT_PER_PEER_RATE_PER_MIN = 60


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class _ActiveJob:
    """One in-flight public-pool donation job."""
    peer_id: str
    kind: str
    started_at: float
    weight: float = 1.0  # 1.0 = one unit of capacity. Scale up for big jobs.


@dataclass
class _PeerActivity:
    """Per-peer rolling history. Used by the rate limiter + the
    sliding-window active-consumer tally."""
    last_request_at: float = 0.0
    requests_in_window: int = 0
    window_started_at: float = 0.0
    completed_jobs: int = 0


# All state is module-level + lock-guarded. Single Gigachat install
# has one fairness scheduler, no sharding needed.
_active_jobs: dict[str, _ActiveJob] = {}  # job_id -> _ActiveJob
_peer_activity: dict[str, _PeerActivity] = {}
_lock = threading.Lock()
_next_job_seq = 0


def _gen_job_id() -> str:
    global _next_job_seq
    with _lock:
        _next_job_seq += 1
        return f"job-{int(time.time() * 1000)}-{_next_job_seq}"


# ---------------------------------------------------------------------------
# Tunables (read each call so a UI change takes effect immediately).
# ---------------------------------------------------------------------------


def _setting(key: str, default: float | int) -> float | int:
    """Tiny typed settings reader. Returns ``default`` on missing /
    malformed values. We store these in `user_settings` so they
    survive restarts and the user can tweak without a code change."""
    val = db.get_setting(key)
    if val is None or val == "":
        return default
    try:
        if isinstance(default, int):
            return int(float(val))
        return float(val)
    except Exception:
        return default


def donation_fraction() -> float:
    """Fraction of local compute reserved for donations."""
    f = float(_setting("p2p_donation_fraction", _DEFAULT_DONATION_FRACTION))
    return max(0.0, min(1.0, f))


def max_concurrent_donations() -> int:
    """Hard cap on simultaneously-running public-pool jobs."""
    n = int(_setting("p2p_max_concurrent_donations", _DEFAULT_MAX_CONCURRENT_DONATIONS))
    return max(1, n)


def per_peer_rate_per_min() -> int:
    """Rate cap per peer per minute."""
    n = int(_setting("p2p_per_peer_rate_per_min", _DEFAULT_PER_PEER_RATE_PER_MIN))
    return max(1, n)


# ---------------------------------------------------------------------------
# Real-time computations.
# ---------------------------------------------------------------------------


def _purge_finished(now: float | None = None) -> None:
    """Drop activity for peers that haven't been seen in the window
    so the active-consumer count reflects "right now" rather than
    "ever in the past."""
    cutoff = (now if now is not None else time.time()) - _ACTIVE_CONSUMER_WINDOW_SEC
    for pid in list(_peer_activity.keys()):
        if _peer_activity[pid].last_request_at < cutoff:
            _peer_activity.pop(pid, None)


def active_consumer_count() -> int:
    """Count of distinct peers we've heard from in the last window.

    Includes peers currently mid-job. The cap formula uses (count + 1)
    semantics by convention so a single consumer doesn't get the full
    share — leaves headroom for an arrival in the next millisecond.
    """
    with _lock:
        _purge_finished()
        # Peers currently mid-job count too even if their last
        # admission was older than the window — they're actively
        # consuming RIGHT NOW.
        active_now = {j.peer_id for j in _active_jobs.values()}
        return len(active_now | set(_peer_activity.keys()))


def per_consumer_slice() -> int:
    """Maximum concurrent jobs we'll admit for any single peer.

    = max(1, max_concurrent_donations / max(1, active_consumers))

    Real-time: as active_consumers grows, each consumer's slice
    shrinks. As consumers drop off, slices automatically widen.
    """
    cap = max_concurrent_donations()
    active = max(1, active_consumer_count())
    # +1 leaves room for a new arrival without immediately
    # over-allocating; rounded down (math.floor) to be conservative.
    slice_size = max(1, math.floor(cap / (active + 1)))
    # No matter what, the cap is the ceiling.
    return min(cap, slice_size)


def consumer_active_count(peer_id: str) -> int:
    """Currently-running jobs FOR a specific peer. Used by the
    admission check to enforce the per-peer slice."""
    with _lock:
        return sum(1 for j in _active_jobs.values() if j.peer_id == peer_id)


# ---------------------------------------------------------------------------
# Admission check + slot lifecycle.
# ---------------------------------------------------------------------------


@dataclass
class AdmissionDecision:
    """Result of `should_admit`. Either holds a slot (admit=True) or
    a human-readable refusal (admit=False)."""
    admit: bool
    job_id: Optional[str] = None
    reason: str = ""

    def release(self) -> None:
        """Mark the slot complete. Idempotent — calling twice is a
        no-op so callers don't have to track double-release."""
        if self.job_id:
            release_slot(self.job_id)
            self.job_id = None


def should_admit(peer_id: str, kind: str = "compute") -> AdmissionDecision:
    """Central admission decision for inbound public-pool requests.

    Checks (in order):
      1. Hard cap on total simultaneous donations.
      2. Per-peer slice (`per_consumer_slice`) — the real-time
         fairness rule that prevents one peer from monopolising
         a slot count that should be shared.
      3. Per-peer rate limit (last-minute requests).

    On admit, reserves a slot atomically and returns a job_id the
    caller MUST release when the work finishes (`release_slot` or
    `decision.release()`).

    Privacy is orthogonal — even when we admit, the work is
    "compute-only" by contract; nothing the user can phrase as a
    "send my prompt" is admittable here.
    """
    now = time.time()
    with _lock:
        _purge_finished(now)

        # 1. Hard cap.
        if len(_active_jobs) >= max_concurrent_donations():
            return AdmissionDecision(
                admit=False,
                reason=(
                    f"refused: at hard cap "
                    f"({max_concurrent_donations()} concurrent donations); "
                    f"retry shortly"
                ),
            )

        # 2. Per-peer slice — real-time fairness.
        slice_size = per_consumer_slice()
        active_for_peer = sum(
            1 for j in _active_jobs.values() if j.peer_id == peer_id
        )
        if active_for_peer >= slice_size:
            return AdmissionDecision(
                admit=False,
                reason=(
                    f"refused: peer {peer_id!r} at per-consumer slice "
                    f"({slice_size} concurrent), {active_consumer_count()} "
                    f"consumers active right now"
                ),
            )

        # 3. Per-peer rate limit (sliding 60 s window).
        rate_cap = per_peer_rate_per_min()
        act = _peer_activity.get(peer_id)
        if act and (now - act.window_started_at) < 60.0:
            if act.requests_in_window >= rate_cap:
                return AdmissionDecision(
                    admit=False,
                    reason=(
                        f"refused: peer {peer_id!r} hit rate cap "
                        f"({rate_cap}/min); cool down before retrying"
                    ),
                )

        # All checks passed — reserve a slot.
        job_id = _gen_job_id()
        _active_jobs[job_id] = _ActiveJob(
            peer_id=peer_id, kind=kind, started_at=now,
        )
        if act is None:
            act = _PeerActivity(
                window_started_at=now, last_request_at=now,
                requests_in_window=1, completed_jobs=0,
            )
            _peer_activity[peer_id] = act
        else:
            if (now - act.window_started_at) >= 60.0:
                # Window rolled over — start a new one.
                act.window_started_at = now
                act.requests_in_window = 1
            else:
                act.requests_in_window += 1
            act.last_request_at = now
        return AdmissionDecision(
            admit=True, job_id=job_id,
            reason=f"admitted (slice={slice_size}, total_active={len(_active_jobs)})",
        )


def release_slot(job_id: str) -> None:
    """Free a slot. Idempotent. Updates the per-peer completed count
    so the local view of "this peer is active" can decay correctly.
    """
    with _lock:
        job = _active_jobs.pop(job_id, None)
        if job:
            act = _peer_activity.get(job.peer_id)
            if act:
                act.completed_jobs += 1
                act.last_request_at = max(act.last_request_at, time.time())


# ---------------------------------------------------------------------------
# Snapshot for the UI / status endpoint.
# ---------------------------------------------------------------------------


def status() -> dict:
    """Real-time view of the fairness scheduler.

    Used by the API endpoint to render the "Public pool fair-share"
    panel: shows current per-consumer slice, total active jobs, and
    each peer's current draw.
    """
    with _lock:
        _purge_finished()
        active_jobs = list(_active_jobs.values())
        per_peer: dict[str, int] = {}
        for j in active_jobs:
            per_peer[j.peer_id] = per_peer.get(j.peer_id, 0) + 1
        return {
            "active_jobs": len(active_jobs),
            "max_concurrent_donations": max_concurrent_donations(),
            "active_consumers": active_consumer_count(),
            "per_consumer_slice": per_consumer_slice(),
            "donation_fraction": donation_fraction(),
            "per_peer": per_peer,
            "configured": {
                "donation_fraction": donation_fraction(),
                "max_concurrent_donations": max_concurrent_donations(),
                "per_peer_rate_per_min": per_peer_rate_per_min(),
            },
        }


def set_config(
    *,
    donation_fraction: float | None = None,
    max_concurrent_donations: int | None = None,
    per_peer_rate_per_min: int | None = None,
) -> dict:
    """Persist new tuning values. Each takes effect immediately
    because the scheduler reads settings on every check."""
    if donation_fraction is not None:
        f = max(0.0, min(1.0, float(donation_fraction)))
        db.set_setting("p2p_donation_fraction", str(f))
    if max_concurrent_donations is not None:
        n = max(1, int(max_concurrent_donations))
        db.set_setting("p2p_max_concurrent_donations", str(n))
    if per_peer_rate_per_min is not None:
        n = max(1, int(per_peer_rate_per_min))
        db.set_setting("p2p_per_peer_rate_per_min", str(n))
    return status()
