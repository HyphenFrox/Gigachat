"""Disk retention policy for checkpoints, memory, screenshots, uploads,
and SQLite maintenance.

Why this exists
---------------
Multiple paths accumulate files under `data/` forever with no upper
bound; in addition, the SQLite database file itself bloats over time
because deletes only mark pages as free, leaving the file size constant
until a manual maintenance step.

  1. `data/checkpoints/<conv_id>/<stamp>/<hash>.bin` — created by every
     `write_file` / `edit_file` call that overwrites an existing file.
     A busy conversation can easily generate hundreds of stamps, and a
     backend that runs for months can accumulate gigabytes.

  2. `data/memory/<conv_id>.md` — one file per conversation that ever
     used `memory_put`. When a conversation is deleted from the DB, the
     memory file was previously left behind as a tombstone.

  3. `data/screenshots/<uuid>.png` — one PNG per screenshot/UIA tool
     call. Long computer-use sessions can accumulate hundreds of MB
     of images. Each surviving PNG must remain reachable while its
     referencing message exists; orphans can be reaped.

  4. `data/uploads/<rand>.<ext>` — one file per user paste/drag-drop.
     References are stored in `messages.images`. Deleted conversations
     leave their uploads behind.

  5. SQLite `app.db` — pages freed by deleting conversations / messages
     / embeddings stay allocated; the WAL file grows on every commit.
     Periodic `wal_checkpoint(TRUNCATE)` and `PRAGMA optimize` keep the
     file size and query plan stats sane.

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

Screenshots / uploads:
  * A file is "orphaned" if it's older than `ORPHAN_GRACE_SECONDS` AND
    its filename appears in no surviving message's tool_calls /
    images JSON. The grace window protects in-flight paths where the
    file exists on disk but the message persisting it hasn't landed
    yet (e.g. tool result emitted, message not yet added).
  * Orphaned files are deleted; surviving ones are kept regardless of
    age — they're still useful context for the agent on rewind.

SQLite maintenance:
  * `PRAGMA wal_checkpoint(TRUNCATE)` truncates WAL after a successful
    checkpoint. Reclaims write-amplification space.
  * `PRAGMA optimize` updates query-planner statistics on tables
    whose stats have drifted; near-free when there's nothing to do.

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

import json
import shutil
import sqlite3
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

ORPHAN_GRACE_SECONDS = 600
"""Don't delete an orphan upload / screenshot until it's been on disk
for at least this long. Protects the path where a tool wrote the file
but the assistant message persisting the reference hasn't been
committed yet. 10 minutes is comfortably longer than any single tool
call, while still bounding worst-case growth."""

DB_MAINTENANCE_INTERVAL_SECONDS = 24 * 3600
"""Run the heavier SQLite maintenance pragmas at most once a day. They're
cheap individually but `wal_checkpoint(TRUNCATE)` blocks writers briefly,
so we don't want to fire it every six hours alongside the disk sweep."""


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


# --- Orphan upload / screenshot cleanup -----------------------------------

def _referenced_upload_filenames() -> set[str]:
    """Filenames present in any `messages.images` JSON array.

    Read-only single-shot scan over the messages table. Performed via
    `db._conn` because the membership set is what we need — we don't
    need full message hydration. Returns an empty set if the read
    fails so the caller can choose to skip cleanup (better than a
    partial set that would falsely orphan everything).
    """
    out: set[str] = set()
    try:
        with db._conn() as c:
            rows = c.execute(
                "SELECT images FROM messages WHERE images IS NOT NULL"
            ).fetchall()
    except sqlite3.Error:
        return out
    for r in rows:
        raw = r["images"]
        if not raw:
            continue
        try:
            arr = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(arr, list):
            continue
        for name in arr:
            if isinstance(name, str) and name:
                out.add(name)
    return out


def _referenced_screenshot_filenames() -> set[str]:
    """Filenames cited as ``image_path`` inside any tool_calls JSON.

    Tool messages store an array of tool calls; screenshot / UIA
    tools attach the PNG name as ``tool_calls[i].image_path``. The
    set is the survivors of any orphan sweep — anything in
    ``data/screenshots/`` not in this set is unreferenced.
    """
    out: set[str] = set()
    try:
        with db._conn() as c:
            rows = c.execute(
                "SELECT tool_calls FROM messages WHERE tool_calls IS NOT NULL"
            ).fetchall()
    except sqlite3.Error:
        return out
    for r in rows:
        raw = r["tool_calls"]
        if not raw:
            continue
        try:
            arr = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(arr, list):
            continue
        for tc in arr:
            if not isinstance(tc, dict):
                continue
            ip = tc.get("image_path")
            if isinstance(ip, str) and ip:
                out.add(ip)
    return out


def _delete_orphan_files(
    directory: Path,
    referenced: set[str],
    *,
    grace_seconds: float = ORPHAN_GRACE_SECONDS,
) -> int:
    """Delete files in `directory` whose name is not in `referenced`.

    Skips files younger than `grace_seconds` so an in-flight write
    (file on disk, message not yet committed) isn't reaped before its
    referencing row lands. Returns count of deletions.
    """
    if not directory.is_dir():
        return 0
    cutoff = time.time() - grace_seconds
    deleted = 0
    try:
        entries = list(directory.iterdir())
    except OSError:
        return 0
    for entry in entries:
        if not entry.is_file():
            continue
        if entry.name in referenced:
            continue
        try:
            if entry.stat().st_mtime > cutoff:
                # Too young — likely just written, give it a window.
                continue
            entry.unlink()
            deleted += 1
        except OSError:
            continue
    return deleted


# --- SQLite maintenance ---------------------------------------------------

def db_maintenance() -> dict:
    """Run lightweight SQLite housekeeping on the primary DB.

    * ``wal_checkpoint(TRUNCATE)`` flushes the WAL back into the main
      DB and shrinks the WAL file. On a chat backend that's been up
      for weeks, the WAL alone can grow to hundreds of MB if it never
      checkpoints (most readers trigger a passive checkpoint, but a
      mostly-write workload doesn't, so we force one here).
    * ``optimize`` runs ANALYZE on tables whose stats have drifted —
      crucial as messages and embeddings tables grow by orders of
      magnitude over a long session.

    Returns a dict of counts so the sweep log shows what happened.
    Errors per pragma are caught and surfaced as a False flag — the
    daemon must keep running even if the DB is briefly locked.
    """
    counts: dict = {"checkpoint_pages": 0, "optimize_ok": False}
    try:
        with db._conn() as c:
            try:
                row = c.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE)"
                ).fetchone()
                if row is not None:
                    # Returns (busy, log_pages, ckpt_pages); the third
                    # column is the count we care about.
                    try:
                        counts["checkpoint_pages"] = int(row[2] or 0)
                    except (TypeError, IndexError, ValueError):
                        pass
            except sqlite3.Error:
                pass
            try:
                c.execute("PRAGMA optimize")
                counts["optimize_ok"] = True
            except sqlite3.Error:
                pass
    except sqlite3.Error:
        pass
    return counts


def sweep(
    checkpoint_dir: Path,
    memory_dir: Path,
    upload_dir: Path | None = None,
    screenshot_dir: Path | None = None,
) -> dict:
    """Run the full retention pass. Safe to call repeatedly.

    `upload_dir` / `screenshot_dir` are optional so existing callers
    (mainly tests) keep working unchanged. The production daemon
    passes both so orphan cleanup happens alongside checkpoint /
    memory cleanup.

    Returns a dict of counts so callers / tests can assert behavior:
      {old_stamps, orphan_conv_dirs, orphan_memory_files,
       orphan_uploads, orphan_screenshots}

    Logs one `retention_sweep` event at INFO with the same counts.
    """
    known = _known_conv_ids()
    old_stamps = _delete_old_checkpoint_stamps(
        checkpoint_dir,
        max_age_seconds=MAX_CHECKPOINT_AGE_DAYS * 86400,
    )
    orphan_convs = _delete_orphan_checkpoint_dirs(checkpoint_dir, known)
    orphan_mems = _delete_orphan_memory_files(memory_dir, known)

    orphan_uploads = 0
    if upload_dir is not None and upload_dir.is_dir():
        # Only run if we have a non-empty referenced set; the empty-set
        # case happens on a fresh DB where every existing file would
        # falsely look orphaned. The same defensive pattern is used by
        # `_delete_orphan_checkpoint_dirs`.
        ref_uploads = _referenced_upload_filenames()
        if ref_uploads or _has_any_messages():
            orphan_uploads = _delete_orphan_files(upload_dir, ref_uploads)

    orphan_screenshots = 0
    if screenshot_dir is not None and screenshot_dir.is_dir():
        ref_shots = _referenced_screenshot_filenames()
        if ref_shots or _has_any_messages():
            orphan_screenshots = _delete_orphan_files(
                screenshot_dir, ref_shots,
            )

    counts = {
        "old_stamps": old_stamps,
        "orphan_conv_dirs": orphan_convs,
        "orphan_memory_files": orphan_mems,
        "orphan_uploads": orphan_uploads,
        "orphan_screenshots": orphan_screenshots,
    }
    log = get_logger("retention")
    log.info(
        f"retention_sweep old_stamps={old_stamps} "
        f"orphan_conv_dirs={orphan_convs} "
        f"orphan_memory_files={orphan_mems} "
        f"orphan_uploads={orphan_uploads} "
        f"orphan_screenshots={orphan_screenshots}",
        extra={"event": "retention_sweep", **counts},
    )
    return counts


def _has_any_messages() -> bool:
    """Cheap check: at least one row in `messages`.

    Used by orphan-file cleanup to disambiguate "DB is fresh, every
    file is genuinely unreferenced" from "DB has rows but our
    SELECT failed and we'd false-orphan everything." Returns False
    on any read error — refuses to delete in the failure path.
    """
    try:
        with db._conn() as c:
            row = c.execute(
                "SELECT 1 FROM messages LIMIT 1"
            ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


async def sweep_daemon(
    checkpoint_dir: Path,
    memory_dir: Path,
    upload_dir: Path | None = None,
    screenshot_dir: Path | None = None,
    interval_seconds: float = SWEEP_INTERVAL_SECONDS,
    db_maintenance_interval: float = DB_MAINTENANCE_INTERVAL_SECONDS,
) -> None:
    """Background task: sweep forever at `interval_seconds` cadence.

    Started from `app.py`'s startup hook. Any exception inside the loop
    is caught and logged — the task must not die silently or we lose the
    janitor for the whole process lifetime.

    DB maintenance runs on a coarser cadence (`db_maintenance_interval`)
    because `wal_checkpoint(TRUNCATE)` is a brief writer-blocker we
    don't need every six hours.
    """
    import asyncio

    log = get_logger("retention")
    last_db_maint = 0.0
    # Run once immediately so startup reclaims whatever leaked across
    # the previous session's lifetime, then settle into the cadence.
    while True:
        try:
            sweep(checkpoint_dir, memory_dir, upload_dir, screenshot_dir)
        except Exception as exc:  # pragma: no cover — defensive
            log.exception(
                f"retention sweep failed: {type(exc).__name__}: {exc}",
                extra={"event": "retention_sweep_error"},
            )
        # DB maintenance lives next to the disk sweep; both are slow
        # janitors that benefit from running off-peak together.
        now = time.time()
        if (now - last_db_maint) >= db_maintenance_interval:
            try:
                stats = db_maintenance()
                log.info(
                    f"db_maintenance checkpoint_pages={stats['checkpoint_pages']} "
                    f"optimize_ok={stats['optimize_ok']}",
                    extra={"event": "db_maintenance", **stats},
                )
                last_db_maint = now
            except Exception as exc:  # pragma: no cover — defensive
                log.exception(
                    f"db maintenance failed: {type(exc).__name__}: {exc}",
                    extra={"event": "db_maintenance_error"},
                )
        await asyncio.sleep(interval_seconds)
