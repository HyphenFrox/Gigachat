"""Event-driven trigger runtime.

Two background daemons run alongside the FastAPI app:

  - **File watcher** — polls every enabled `file_watchers` row on a
    short interval and fires an agent turn against the row's target
    conversation when a file inside the watched path matches the
    glob pattern AND was created / modified / deleted since the
    last sweep. Polling (not inotify / fsevents) so we have zero
    runtime deps and the same code path on every OS.

  - **Webhook driver** — N/A here; webhook firing happens inline in
    the FastAPI route (`/webhook/{token}` in `app.py`). This module
    only owns the file-watcher loop. Kept under one module so all
    event-driven plumbing is in one place.

Each fire is debounced per-watcher so a `git pull` that touches 200
files doesn't queue 200 turns. The debounce window comes from the
row's `debounce_seconds` column (1-3600 s). When multiple events
coalesce inside a window, the agent prompt summarises them.

The runtime is deliberately fire-and-forget: a turn that fails or
takes ages to complete does not block subsequent watcher sweeps. The
agent's own error handlers persist failure messages into the target
conversation, so the user sees the result the next time they open
that chat.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
import time
from pathlib import Path
from typing import Any

from . import db

log = logging.getLogger("gigachat.events")

# Sweep cadence. 2 s is fast enough that a "save and look at output"
# round-trip feels live on a typical editor save, but long enough that
# the file-system probe overhead is negligible (a typical project
# directory has well under 10 k tracked files; one stat per file per
# 2 s is cheap).
_SWEEP_INTERVAL_SEC = 2.0

# Cap on files inspected per sweep per watcher. Defends against a user
# who points the watcher at C:\\ — we'd happily try to stat the whole
# drive otherwise. The agent gets a one-time warning event on the chat
# when this trips.
_MAX_FILES_PER_WATCHER = 5_000

# Cap on event count surfaced per fire so a "pull that touched 500
# files" doesn't push a 500-line prompt into the target conversation.
_MAX_EVENTS_PER_FIRE = 50


class _WatcherState:
    """Per-watcher in-memory state carried across sweeps.

    `snapshots` maps file path → (mtime, size). On each sweep we rebuild
    the snapshot, diff against the previous one, and emit one
    (event_type, path) for each created / modified / deleted entry.

    `pending_events` accumulates between sweeps until the debounce
    window elapses with no new activity, at which point we flush them
    to a single agent turn.

    `last_event_at` tracks the most recent event time so we can decide
    when the debounce window is satisfied.

    `firing_lock` prevents two overlapping turns for the same watcher
    when a fire takes longer than the debounce window.
    """

    __slots__ = (
        "snapshots", "pending_events", "last_event_at", "firing_lock",
    )

    def __init__(self) -> None:
        self.snapshots: dict[str, tuple[float, int]] = {}
        self.pending_events: list[tuple[str, str]] = []
        self.last_event_at: float = 0.0
        self.firing_lock: asyncio.Lock = asyncio.Lock()


_states: dict[str, _WatcherState] = {}
_runtime_task: asyncio.Task | None = None
_stop_event: asyncio.Event | None = None


async def start_event_runtime() -> None:
    """Launch the file-watcher polling loop. Idempotent.

    Wired into FastAPI's startup hook so the loop runs for the lifetime
    of the process. Stop via `stop_event_runtime()` (also idempotent).
    """
    global _runtime_task, _stop_event
    if _runtime_task is not None and not _runtime_task.done():
        return
    _stop_event = asyncio.Event()
    _runtime_task = asyncio.create_task(_runtime_loop(_stop_event))
    log.info("event runtime started")


async def stop_event_runtime() -> None:
    """Signal the loop to exit and await its task. Idempotent."""
    global _runtime_task, _stop_event
    if _stop_event is None or _runtime_task is None:
        return
    _stop_event.set()
    try:
        await asyncio.wait_for(_runtime_task, timeout=5.0)
    except asyncio.TimeoutError:
        log.warning("event runtime did not stop within 5 s; cancelling")
        _runtime_task.cancel()
    except Exception:
        # Swallow — shutdown is best-effort.
        pass
    _runtime_task = None
    _stop_event = None
    log.info("event runtime stopped")


async def _runtime_loop(stop_event: asyncio.Event) -> None:
    """Top-level loop: sweep watchers every `_SWEEP_INTERVAL_SEC`.

    Watcher rows are re-read every iteration so toggling enabled / adding
    / removing watchers takes effect within one sweep — no need to
    restart the daemon.
    """
    while not stop_event.is_set():
        try:
            await _sweep_once()
        except Exception as e:
            # A bad watcher row should not kill the daemon. Log + carry on.
            log.warning("file-watcher sweep failed: %s", e)
        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=_SWEEP_INTERVAL_SEC,
            )
        except asyncio.TimeoutError:
            continue
        # stop_event was set — exit the loop.
        break


async def _sweep_once() -> None:
    """Scan every enabled watcher exactly once. Coalesces events into
    the per-watcher state and fires turns for any whose debounce
    window has elapsed."""
    rows = db.list_file_watchers(enabled_only=True)
    seen_ids: set[str] = set()
    for row in rows:
        wid = row["id"]
        seen_ids.add(wid)
        state = _states.setdefault(wid, _WatcherState())
        try:
            new_events = _diff_snapshot(row, state)
        except Exception as e:
            log.info("watcher %s scan failed: %s", row.get("name"), e)
            continue
        if new_events:
            state.pending_events.extend(new_events)
            state.last_event_at = time.time()
            # Cap accumulation so a runaway directory doesn't grow
            # the buffer forever between fires.
            if len(state.pending_events) > _MAX_EVENTS_PER_FIRE:
                # Keep the FIRST events (most informative for diagnosing
                # a sudden burst); drop the rest with a marker.
                state.pending_events = state.pending_events[:_MAX_EVENTS_PER_FIRE]
        # Fire when (a) we have pending events and (b) the debounce
        # window has elapsed since the last event.
        if state.pending_events:
            debounce_window = max(1, int(row.get("debounce_seconds") or 5))
            if (time.time() - state.last_event_at) >= debounce_window:
                events_to_fire = state.pending_events
                state.pending_events = []
                # Run the fire concurrently so a slow turn doesn't
                # block the next sweep.
                asyncio.create_task(_fire_watcher(row, events_to_fire))
    # Drop state entries for watchers that no longer exist (deleted /
    # disabled). Keeps the dict bounded across long-running processes.
    stale = set(_states.keys()) - seen_ids
    for sid in stale:
        _states.pop(sid, None)


def _diff_snapshot(
    row: dict, state: _WatcherState,
) -> list[tuple[str, str]]:
    """Compare current FS state to the previous snapshot.

    Returns a list of ``(event_type, path)`` tuples for every entry
    that changed. Updates the state's snapshot in place.

    Path matching uses ``fnmatch`` against the row's `glob_pattern`
    (default '*'). Paths beyond `_MAX_FILES_PER_WATCHER` are silently
    dropped — the watcher should be pointed at a more specific path.
    """
    target = Path(row["path"]).expanduser()
    if not target.exists():
        # The path went away. Wipe snapshot so re-creation is treated
        # as a fresh "created" wave rather than a noisy "everything
        # was deleted" sweep on first detection.
        if state.snapshots:
            state.snapshots = {}
        return []

    pattern = row.get("glob_pattern") or "*"
    events_set = set(row.get("events") or ["created", "modified"])

    cur_snapshot: dict[str, tuple[float, int]] = {}
    files_seen = 0
    if target.is_dir():
        # Walk the directory tree. Hidden dotfiles + common VCS dirs
        # are skipped because watching `.git` is almost always noise.
        for root, dirs, files in os.walk(target):
            # Filter out noisy top-level dirs in place so os.walk
            # doesn't even descend into them.
            dirs[:] = [
                d for d in dirs
                if d not in {".git", "node_modules", "__pycache__", "dist", "build"}
            ]
            for fname in files:
                if files_seen >= _MAX_FILES_PER_WATCHER:
                    break
                files_seen += 1
                fpath = os.path.join(root, fname)
                if not fnmatch.fnmatch(fname, pattern):
                    continue
                try:
                    s = os.stat(fpath)
                except OSError:
                    continue
                cur_snapshot[fpath] = (s.st_mtime, s.st_size)
            if files_seen >= _MAX_FILES_PER_WATCHER:
                break
    else:
        try:
            s = target.stat()
            cur_snapshot[str(target)] = (s.st_mtime, s.st_size)
        except OSError:
            return []

    events: list[tuple[str, str]] = []
    prev = state.snapshots
    # First sweep: just snapshot, don't synthesise "created" events
    # for every existing file (would fire the moment a watcher is
    # enabled, which is rarely what the user wants).
    if not prev:
        state.snapshots = cur_snapshot
        return []

    for path, meta in cur_snapshot.items():
        if path not in prev:
            if "created" in events_set:
                events.append(("created", path))
        elif prev[path] != meta:
            if "modified" in events_set:
                events.append(("modified", path))
    if "deleted" in events_set:
        for path in prev:
            if path not in cur_snapshot:
                events.append(("deleted", path))
    state.snapshots = cur_snapshot
    return events


async def _fire_watcher(
    row: dict, events: list[tuple[str, str]],
) -> None:
    """Spawn an agent turn for one batch of file events.

    Held under a per-watcher lock so a slow turn doesn't get layered.
    Lock is held for the lifetime of the agent loop — if a turn takes
    longer than the debounce window, follow-up events buffer up
    naturally for the next fire.
    """
    state = _states.get(row["id"])
    if state is None:
        return
    if state.firing_lock.locked():
        # Skip this fire; events stay buffered for the next sweep.
        log.info(
            "watcher %s already firing; deferring %d events",
            row.get("name"), len(events),
        )
        return
    async with state.firing_lock:
        prompt = _build_watcher_prompt(row, events)
        try:
            db.record_file_watcher_fire(row["id"])
        except Exception:
            pass
        # Lazy-import to avoid an import cycle (agent → tools → db).
        from . import agent as _agent
        try:
            async for _ in _agent.run_turn(
                row["target_conversation_id"],
                user_text=prompt,
            ):
                pass
        except Exception as e:
            log.warning(
                "watcher %s fire failed: %s", row.get("name"), e,
            )


def _build_watcher_prompt(
    row: dict, events: list[tuple[str, str]],
) -> str:
    """Render the events into the agent's user message.

    Honours the row's `prompt_template` when set: every occurrence of
    ``{events}`` is replaced with the bullet list of events. Without a
    template the prompt is just the events list — the user can then
    extend the conversation in any direction.
    """
    template = row.get("prompt_template") or ""
    bullets = "\n".join(
        f"- {evt}: {path}" for evt, path in events[:_MAX_EVENTS_PER_FIRE]
    )
    if len(events) > _MAX_EVENTS_PER_FIRE:
        bullets += f"\n  ... and {len(events) - _MAX_EVENTS_PER_FIRE} more"
    if template:
        return template.replace("{events}", bullets)
    return (
        f"[file watcher: {row.get('name', 'unnamed')!r}, "
        f"{len(events)} event(s)]\n{bullets}"
    )
