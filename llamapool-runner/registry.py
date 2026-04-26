"""Cross-process claim registry: shared state so multiple workloads
can engage the pool concurrently without tripping each other.

State lives at `~/.llamapool/active.json` and looks like:

    {
        "claims": [
            {
                "id": "uuid4-string",
                "pid": 1234,
                "started_at": 1714000000.0,
                "in_split": true,
                "gguf_path": "/path/to/model.gguf",
                "ngl": 28,
                "estimated_bytes": 16000000000,
                "rpc_endpoints": ["192.168.1.42:50052", ...]
            },
            ...
        ]
    }

Why we need this
----------------
Without a shared registry, two workloads engaging the pool at the
same time would each compute `-ngl` based on the *same* free pool
memory and overcommit. They'd also potentially restart each other's
rpc-server backends mid-session.

What this gives us
------------------
* `register_claim()` writes a claim atomically under a file lock.
* `purge_dead_claims()` drops entries whose PID is no longer alive
  (workload crashed without disengaging — best-effort cleanup).
* `total_reserved_bytes()` sums what other live claims have already
  budgeted, so a new caller computes ngl against `pool_free - reserved`.
* `any_split_active()` lets us avoid bouncing worker rpc-servers
  back to idle-mode backend while another claim is still in split.

The lock is a sentinel file (`active.json.lock`) acquired by
exclusive create. It's coarse-grained — held only for the duration
of a read-modify-write — so contention is negligible in practice
(workloads engage at startup, not in hot loops).
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from config import CONFIG_DIR

log = logging.getLogger(__name__)

ACTIVE_PATH = CONFIG_DIR / "active.json"
LOCK_PATH = CONFIG_DIR / "active.json.lock"

# Max time we wait for the lock before giving up (sec). Long enough
# to absorb a normal engage() call's I/O, short enough that a crashed
# lock-holder doesn't hang us forever — `_acquire_lock` falls back to
# stealing the lock after this much elapsed time.
_LOCK_TIMEOUT = 5.0
_LOCK_STALE_AFTER = 30.0  # if lock file is older than this, steal it


@contextmanager
def _acquire_lock() -> Iterator[None]:
    """Acquire a coarse exclusive lock for the registry file.

    Implementation: try `os.open(LOCK_PATH, O_CREAT|O_EXCL)` in a
    poll loop. Cross-platform (no fcntl dependency). Steals the lock
    if it's older than `_LOCK_STALE_AFTER` (so a crashed holder
    doesn't deadlock us forever).
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + _LOCK_TIMEOUT
    while True:
        try:
            fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            # Steal stale locks (process crashed mid-write).
            try:
                age = time.time() - LOCK_PATH.stat().st_mtime
                if age > _LOCK_STALE_AFTER:
                    LOCK_PATH.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() > deadline:
                # Last-resort: steal even if young, rather than hang.
                log.warning("registry lock contention; stealing %s", LOCK_PATH)
                LOCK_PATH.unlink(missing_ok=True)
                continue
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass


def _load_unlocked() -> dict[str, Any]:
    """Read active.json without locking. Caller must already hold the lock."""
    try:
        return json.loads(ACTIVE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"claims": []}
    except json.JSONDecodeError:
        # Corrupted — reset rather than crash callers.
        log.warning("active.json corrupted; resetting")
        return {"claims": []}


def _save_unlocked(state: dict[str, Any]) -> None:
    """Atomic write: tmp + rename. Caller must already hold the lock."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ACTIVE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, ACTIVE_PATH)


def _pid_alive(pid: int) -> bool:
    """Best-effort PID liveness check, cross-platform."""
    if pid <= 0:
        return False
    try:
        # POSIX: kill(0) returns success if the process exists and we
        # have permission; raises PermissionError if it exists but we
        # don't (still alive). Windows: os.kill(pid, 0) raises OSError
        # if pid doesn't exist.
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True  # exists, just not ours
    except (OSError, ProcessLookupError):
        return False


def _purge_dead(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop claims whose PID is no longer alive."""
    out = []
    for c in claims:
        pid = int(c.get("pid") or 0)
        if _pid_alive(pid):
            out.append(c)
        else:
            log.info("purging stale claim id=%s pid=%d", c.get("id"), pid)
    return out


def get_active_claims() -> list[dict[str, Any]]:
    """Read the registry, return live claims (dead ones purged from disk)."""
    with _acquire_lock():
        state = _load_unlocked()
        live = _purge_dead(state.get("claims") or [])
        if len(live) != len(state.get("claims") or []):
            state["claims"] = live
            _save_unlocked(state)
        return live


def register_claim(
    *,
    in_split: bool,
    gguf_path: str | None = None,
    ngl: int | None = None,
    estimated_bytes: int = 0,
    rpc_endpoints: list[str] | None = None,
    priority: int = 100,
) -> str:
    """Register a new claim for the current process; return its id.

    Caller passes the id back to `unregister_claim` on shutdown.
    Registering twice from the same PID adds a *second* claim — the
    common pattern is one-claim-per-engage-call, so the wrapper or
    library should match register/unregister 1:1.

    `priority` is the integer weight used when the pool is contested
    (higher = larger share). Default 100.
    """
    cid = uuid.uuid4().hex
    claim = {
        "id": cid,
        "pid": os.getpid(),
        "started_at": time.time(),
        "in_split": bool(in_split),
        "gguf_path": gguf_path,
        "ngl": ngl,
        "estimated_bytes": int(estimated_bytes),
        "rpc_endpoints": rpc_endpoints or [],
        "priority": int(priority),
    }
    with _acquire_lock():
        state = _load_unlocked()
        claims = _purge_dead(state.get("claims") or [])
        claims.append(claim)
        state["claims"] = claims
        _save_unlocked(state)
    return cid


def unregister_claim(claim_id: str) -> bool:
    """Remove the claim with the given id. Returns True if it was
    present. Idempotent: removing a missing claim is not an error.
    """
    with _acquire_lock():
        state = _load_unlocked()
        before = state.get("claims") or []
        after = [c for c in before if c.get("id") != claim_id]
        if len(after) == len(before):
            return False
        state["claims"] = _purge_dead(after)
        _save_unlocked(state)
        return True


def total_reserved_bytes(*, exclude_pid: int | None = None) -> int:
    """Sum of `estimated_bytes` across active claims, optionally
    excluding the current PID's own claims (you don't reserve against
    yourself when computing your own ngl).
    """
    total = 0
    for c in get_active_claims():
        if exclude_pid is not None and int(c.get("pid") or 0) == exclude_pid:
            continue
        total += int(c.get("estimated_bytes") or 0)
    return total


def any_split_active(*, exclude_pid: int | None = None) -> bool:
    """True if any active claim is currently in split mode. Used to
    avoid restoring worker backends to idle mode while another
    workload is still mid-split.
    """
    for c in get_active_claims():
        if exclude_pid is not None and int(c.get("pid") or 0) == exclude_pid:
            continue
        if c.get("in_split"):
            return True
    return False
