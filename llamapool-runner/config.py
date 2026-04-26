"""Worker config: JSON-on-disk, no DB.

A pool config has shape:

    {
        "workers": [
            {
                "label": "Naresh's Laptop",
                "address": "desktop-0692hok.local",
                "rpc_port": 50052,
                "ssh_host": "laptop",
                "enabled": true,
                "gpu_vendor": "intel",          // optional cached vendor
                "current_rpc_backend": "..."    // tracked by the runtime
            },
            ...
        ]
    }

Path: `~/.llamapool/config.json`. Created on first write; read-only
access is fine without it (returns empty config).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".llamapool"
CONFIG_PATH = CONFIG_DIR / "config.json"


def load_config() -> dict[str, Any]:
    """Read the config; return a default empty dict if missing."""
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"workers": []}
    except json.JSONDecodeError as e:
        raise RuntimeError(f"config malformed at {CONFIG_PATH}: {e}") from e


def save_config(cfg: dict[str, Any]) -> None:
    """Write atomically — write to .tmp, fsync, rename. Avoids a
    half-written config if the host crashes mid-save."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, CONFIG_PATH)


def add_worker(
    label: str,
    address: str,
    *,
    rpc_port: int = 50052,
    ssh_host: str | None = None,
    gpu_vendor: str | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Add a worker (or update if `label` already exists). Returns the
    persisted worker entry."""
    cfg = load_config()
    workers = cfg.setdefault("workers", [])
    existing = next((w for w in workers if w.get("label") == label), None)
    entry = {
        "label": label,
        "address": address,
        "rpc_port": rpc_port,
        "ssh_host": ssh_host,
        "gpu_vendor": gpu_vendor,
        "enabled": enabled,
    }
    if existing:
        existing.update(entry)
        entry = existing
    else:
        workers.append(entry)
    save_config(cfg)
    return entry


def remove_worker(label: str) -> bool:
    """Remove the worker with the given label. Returns True if found
    and removed, False otherwise."""
    cfg = load_config()
    workers = cfg.get("workers") or []
    n_before = len(workers)
    cfg["workers"] = [w for w in workers if w.get("label") != label]
    if len(cfg["workers"]) == n_before:
        return False
    save_config(cfg)
    return True


def list_workers(*, enabled_only: bool = False) -> list[dict[str, Any]]:
    """Return the workers list. Optionally filter to enabled-only."""
    workers = load_config().get("workers") or []
    if enabled_only:
        workers = [w for w in workers if w.get("enabled", True)]
    return workers


def update_worker(label: str, **fields: Any) -> dict[str, Any] | None:
    """Patch fields on an existing worker. Returns the updated entry
    or None if not found."""
    cfg = load_config()
    workers = cfg.setdefault("workers", [])
    for w in workers:
        if w.get("label") == label:
            w.update(fields)
            save_config(cfg)
            return w
    return None
