"""Disk retention policy for checkpoints and per-conversation memory.

Why this exists
---------------
Two paths accumulate files under `data/` forever with no upper bound:

  1. `data/checkpoints/<conv_id>/<stamp>/<hash>.bin` — created by every
     `write_file` / `edit_file` call that overwrites an existing file.
     A busy conversation can easily generate hundreds of stamps, and a
     backend that runs for months can accumulate gigabytes.

  2. `data/memory/<conv_id>.md` — one file per conversation that ever
     used `memory_put`. When a conversation is deleted from the DB, the
     memory file was previously left behind as a tombstone.

This module centralizes the cleanup so the rules are auditable in one
place.

Policy
------
Checkpoints:
  * Per conversation, keep the most recent `MAX_CHECKPOINTS_PER_CONV`
    stamps. Older ones are deleted. (JIT-enforced on every new checkpoint.)
  * Across all conversations, delete stamps older than
    `MAX_CHECKPOINT_AGE_DAYS`. (Startup + periodic sweep.)
  * Checkpoint directories whose `conv_id` is no longer in the DB are
    deleted wholesale. (Startup + periodic sweep.)

Memory files:
  * A `data/memory/<conv_id>.md` whose `conv_id` is no longer in the DB
    is deleted. (Startup + periodic sweep.)

Design notes
------------
  * Every path operation is wrapped in try/except. Retention is a
    best-effort janitor — a single unreadable file must not stop the
    rest of the sweep.
  * The sweep is idempotent and safe to run concurrently with writes;
    we only delete files that are strictly older than a fresh stamp, so
    a racing `_checkpoint_file` can't see a half-deleted directory.
  * Structured logs via `backend.telemetry` emit one
    `retention_sweep` event with counts, so you can tell from the logs
    how much disk was reclaimed.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Iterable

from . import db
from .telemetry import get_logger

# --- Tunables --------------------------------------------------------------

MAX_CHECKPOINTS_PER_CONV = 50
"""Keep this many most-recent checkpoint stamps per conversation. Older
stamps are deleted when a new one is written. 50 × a typical small file
(<1 MB) is a comfortable upper bound for even the longest coding sessions."""

MAX_CHECKPOINT_AGE_DAYS = 30
"""Delete checkpoint stamps whose mtime is older than this, irrespective
of per-conversation count. Protects against old conversations that never
hit the per-conv cap but still age out."""

SWEEP_INTERVAL_SECONDS = 6 * 3600
"""How often the periodic sweep runs when the background task is active.
Six hours is a reasonable balance: frequent enough that a runaway week-long
agent doesn't fill the disk, rare enough to avoid any visible load."""


# --- JIT per-conversation trim --------------------------------------------

def trim_conv_checkpoints(
    checkpoint_dir: Path,
    conv_id: str | None,
    keep: int = MAX_CHECKPOINTS_PER_CONV,
) -> int:
    """Delete the oldest checkpoint stamps for `conv_id` beyond `keep`.

    Called from `_checkpoint_file` immediately after a new stamp is
    written. Cheap: a `listdir` + sort + partial delete. O(N) where N
    is the number of stamps for this conversation, typically bounded
    by `keep` itself.

    Returns the number of stamps deleted so the caller can log it.
    """
    if not conv_id:
        return 0
    root = checkpoint_dir / conv_id
    if not root.is_dir():
        return 0
    try:
        # Stamp names are lexicographically ordered by construction
        # (`YYYYMMDDTHHMMSS_micro_uuid`), so sorted() == chronological.
        stamps = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return 0
    excess = len(stamps) - keep
    if excess <= 0:
        return 0
    deleted = 0
    for stamp_dir in stamps[:excess]:
        try:
            shutil.rmtree(stamp_dir, ignore_errors=True)
            deleted += 1
        except OSError:
            continue
    return deleted


# --- Startup / periodic sweep ---------------------------------------------

def _known_conv_ids() -> set[str]:
    """Return the set of conversation ids currently in the DB.

    Any conv_id found on disk that isn't in this set is an orphan —
    its owning conversation was deleted and the directory is safe to
    remove.
    """
    try:
        rows = db.list_conversations()
    except Exception:
        # If the DB isn't available (during early startup or a migration
        # window), skip orphan cleanup rather than delete everything.
        return set()
    ids: set[str] = set()
    for row in rows or []:
        cid = row.get("id") if isinstance(row, dict) else None
        if cid:
            ids.add(str(cid))
    return ids


def _delete_old_checkpoint_stamps(
    checkpoint_dir: Path,
    max_age_seconds: float,
) -> int:
    """Delete checkpoint stamps whose mtime is older than the cutoff.

    Walks CHECKPOINT_DIR/<conv_id>/<stamp>, checks each stamp's mtime
    against `now - max_age_seconds`, removes stale ones. Returns count.
    """
    if not checkpoint_dir.is_dir():
        return 0
    cutoff = time.time() - max_age_seconds
    deleted = 0
    try:
        conv_dirs = list(checkpoint_dir.iterdir())
    except OSError:
        return 0
    for conv_dir in conv_dirs:
        if not conv_dir.is_dir():
            continue
        try:
            stamps = list(conv_dir.iterdir())
        except OSError:
            continue
        for stamp_dir in stamps:
            if not stamp_dir.is_dir():
                continue
            try:
                if stamp_dir.stat().st_mtime < cutoff:
                    shutil.rmtree(stamp_dir, ignore_errors=True)
                    deleted += 1
            except OSError:
                continue
    return deleted


def _delete_orphan_checkpoint_dirs(
    checkpoint_dir: Path,
    known_ids: Iterable[str],
) -> int:
    """Remove `CHECKPOINT_DIR/<conv_id>/` trees whose conv_id is dead."""
    if not checkpoint_dir.is_dir():
        return 0
    known = set(known_ids)
    if not known:
        # Defensive: if the DB query failed we have an empty set, which
        # would nuke every checkpoint directory. Refuse.
        return 0
    deleted = 0
    try:
        entries = list(checkpoint_dir.iterdir())
    except OSError:
        return 0
    for entry in entries:
        if not entry.is_dir():
            continue
        if entry.name in known:
            continue
        try:
            shutil.rmtree(entry, ignore_errors=True)
            deleted += 1
        except OSError:
            continue
    return deleted


def _delete_orphan_memory_files(
    memory_dir: Path,
    known_ids: Iterable[str],
) -> int:
    """Remove `<memory_dir>/<conv_id>.md` for dead conversations."""
    if not memory_dir.is_dir():
        return 0
    known = set(known_ids)
    if not known:
        return 0
    deleted = 0
    try:
        entries = list(memory_dir.iterdir())
    except OSError:
        return 0
    for entry in entries:
        if not entry.is_file() or entry.suffix != ".md":
            continue
        conv_id = entry.stem
        if conv_id in known:
            continue
        try:
            entry.unlink()
            deleted += 1
        except OSError:
            continue
    return deleted


def sweep(
    checkpoint_dir: Path,
    memory_dir: Path,
) -> dict:
    """Run the full retention pass. Safe to call repeatedly.

    Returns a dict of counts so callers / tests can assert behavior:
      {old_stamps, orphan_conv_dirs, orphan_memory_files}

    Logs one `retention_sweep` event at INFO with the same counts.
    """
    known = _known_conv_ids()
    old_stamps = _delete_old_checkpoint_stamps(
        checkpoint_dir,
        max_age_seconds=MAX_CHECKPOINT_AGE_DAYS * 86400,
    )
    orphan_convs = _delete_orphan_checkpoint_dirs(checkpoint_dir, known)
    orphan_mems = _delete_orphan_memory_files(memory_dir, known)
    counts = {
        "old_stamps": old_stamps,
        "orphan_conv_dirs": orphan_convs,
        "orphan_memory_files": orphan_mems,
    }
    log = get_logger("retention")
    log.info(
        f"retention_sweep old_stamps={old_stamps} "
        f"orphan_conv_dirs={orphan_convs} orphan_memory_files={orphan_mems}",
        extra={"event": "retention_sweep", **counts},
    )
    return counts


async def sweep_daemon(
    checkpoint_dir: Path,
    memory_dir: Path,
    interval_seconds: float = SWEEP_INTERVAL_SECONDS,
) -> None:
    """Background task: sweep forever at `interval_seconds` cadence.

    Started from `app.py`'s startup hook. Any exception inside the loop
    is caught and logged — the task must not die silently or we lose the
    janitor for the whole process lifetime.
    """
    import asyncio

    log = get_logger("retention")
    # Run once immediately so startup reclaims whatever leaked across
    # the previous session's lifetime, then settle into the cadence.
    while True:
        try:
            sweep(checkpoint_dir, memory_dir)
        except Exception as exc:  # pragma: no cover — defensive
            log.exception(
                f"retention sweep failed: {type(exc).__name__}: {exc}",
                extra={"event": "retention_sweep_error"},
            )
        await asyncio.sleep(interval_seconds)
