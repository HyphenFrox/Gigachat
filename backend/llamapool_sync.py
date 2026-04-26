"""One-way sync: Gigachat's `compute_workers` table -> the standalone
llamapool package's JSON config at `~/.llamapool/config.json`.

Why
---
Gigachat's compute pool is the canonical source of worker registration
(SQLite-backed, exposed via the Settings UI). The decoupled `llamapool`
package needs to know about those same workers so other apps on the
host (e.g., Peerful's resume-extraction script, or any third-party
LLM workload using `llamapool-llama-server`) inherit the same pool
without the user having to register workers twice.

This module translates the schema and writes JSON. It does NOT import
the `llamapool` Python package — by writing JSON directly we avoid
making Gigachat's startup depend on a separate pip install.

Hook points (called from `backend/db.py`):
  * after `create_compute_worker`
  * after `update_compute_worker`
  * after `update_compute_worker_capabilities` (when GPU vendor /
    RAM info changes from a probe)
  * after `delete_compute_worker`

Failures are swallowed and logged — sync is best-effort so a corrupt
or read-only `~/.llamapool/` doesn't break Gigachat's worker CRUD.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Mirror of `llamapool.config.CONFIG_PATH`. Hardcoded to avoid an
# import dependency on the llamapool package.
_LLAMAPOOL_CONFIG_DIR = Path.home() / ".llamapool"
_LLAMAPOOL_CONFIG_PATH = _LLAMAPOOL_CONFIG_DIR / "config.json"

# Default RPC port for `rpc-server`. Matches `compute_pool._DEFAULT_RPC_PORT`.
_RPC_PORT = 50052


def _classify_gpu_vendor(caps: dict[str, Any]) -> str:
    """Pick the dominant GPU vendor from a capabilities probe payload.

    Mirrors `compute_pool._worker_gpu_vendor` so both code paths agree.
    Priority order: nvidia > amd > intel > none.
    """
    gpus = caps.get("gpus") or []
    names = [(g.get("name") or "").lower() for g in gpus]
    if any(
        "nvidia" in n or "geforce" in n or "rtx " in n or "gtx " in n
        for n in names
    ):
        return "nvidia"
    if any("amd" in n or "radeon" in n for n in names):
        return "amd"
    if any("intel" in n or "iris" in n or "uhd" in n for n in names):
        return "intel"
    return "none"


def _row_to_llamapool_entry(row: dict[str, Any]) -> dict[str, Any]:
    """Translate a Gigachat compute_workers row into a llamapool
    worker entry.

    Schema mapping:
      label                <- label
      address              <- address
      rpc_port             <- 50052 (Gigachat uses a single hardcoded port)
      ssh_host             <- ssh_host
      enabled              <- bool(enabled)
      gpu_vendor           <- caps.gpus -> classified
      ram_total_gb         <- caps.ram_total_gb (used by ngl computation)
      ram_free_gb          <- caps.ram_free_gb
      current_rpc_backend  <- caps.current_rpc_backend (CRITICAL: lets the
                              standalone llamapool see the backend Gigachat
                              already set, so it doesn't bounce rpc-server
                              out from under an in-flight Gigachat split.)
    """
    caps = row.get("capabilities") or {}
    entry: dict[str, Any] = {
        "label": row.get("label"),
        "address": row.get("address"),
        "rpc_port": _RPC_PORT,
        "ssh_host": row.get("ssh_host"),
        "enabled": bool(row.get("enabled", True)),
        "gpu_vendor": _classify_gpu_vendor(caps),
        "ram_total_gb": float(caps.get("ram_total_gb") or 0.0),
        "ram_free_gb": float(caps.get("ram_free_gb") or 0.0),
    }
    backend = caps.get("current_rpc_backend")
    if backend and backend != "unknown":
        entry["current_rpc_backend"] = backend
    return entry


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically — tmp + rename — to avoid a half-written
    config if the host crashes mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True),
                   encoding="utf-8")
    os.replace(tmp, path)


def sync_now() -> None:
    """Mirror the current `compute_workers` table to llamapool's JSON.

    Preserves any existing `current_rpc_backend` value llamapool has
    persisted on a worker entry, so we don't wipe runtime tracking
    state on every sync. Other fields are overwritten from Gigachat.

    Best-effort: any exception is logged and swallowed.
    """
    try:
        # Local import to avoid circular import at module load time.
        from . import db
    except ImportError:
        # Allow standalone testing of this module without Gigachat's db.
        log.debug("backend.db not importable; sync skipped")
        return

    try:
        rows = db.list_compute_workers(enabled_only=False)
    except Exception as e:
        log.warning("llamapool sync: list_compute_workers failed: %s", e)
        return

    new_entries = [_row_to_llamapool_entry(r) for r in rows]

    # Merge in any `current_rpc_backend` already persisted, keyed by label.
    try:
        existing = json.loads(_LLAMAPOOL_CONFIG_PATH.read_text(encoding="utf-8"))
        existing_by_label = {
            (w.get("label") or ""): w
            for w in (existing.get("workers") or [])
        }
        for entry in new_entries:
            old = existing_by_label.get(entry.get("label") or "")
            if old and "current_rpc_backend" in old:
                entry["current_rpc_backend"] = old["current_rpc_backend"]
    except FileNotFoundError:
        pass
    except (json.JSONDecodeError, OSError) as e:
        log.debug("llamapool sync: existing config unreadable (%s); "
                  "overwriting", e)

    payload = {"workers": new_entries}
    try:
        _atomic_write_json(_LLAMAPOOL_CONFIG_PATH, payload)
    except OSError as e:
        log.warning("llamapool sync: write failed: %s", e)
        return
    log.debug("llamapool sync: mirrored %d worker(s) -> %s",
              len(new_entries), _LLAMAPOOL_CONFIG_PATH)
