"""LAN-first model copy: push an Ollama model from host to worker via SCP.

Save internet bandwidth — when a model needs to land on another node and
this host already has it locally, copy via SSH/SCP rather than have the
worker pull from the internet. This is exactly the manual flow we used
for `nomic-embed-text` earlier in the project, generalized to any
registered worker.

Mechanics:
  * Each worker row optionally carries `ssh_host` (e.g. an alias from
    the user's `~/.ssh/config` like `Host laptop`). When set, this
    module shells out to `ssh` and `scp` to manipulate the worker's
    `~/.ollama/models/` directory.
  * On the host side we read Ollama's manifest store at
    `~/.ollama/models/manifests/registry.ollama.ai/library/<model>/<tag>`
    to enumerate the layer blobs we need to ship.
  * On the worker side we ensure the destination directories exist,
    copy each blob the worker doesn't already have, then drop the
    manifest in last (so a partial copy doesn't make Ollama think the
    model is fully present until every layer has landed).

Limitations (intentional v1 scope — extend if needed):
  * Source is always THIS host. We don't relay worker→worker because
    that would require either worker-to-worker SSH (rare setup) or
    routing the bytes through host's RAM (extra hop). Document the
    constraint; users add the model to host once, then push to each
    worker.
  * Auth: we rely entirely on the host's existing SSH config — the
    `ssh_host` field is just an alias / hostname, not credentials.
    Backend never stores SSH keys. If `ssh worker echo ok` works on
    the command line, this module works too.
  * Windows host assumed. The OpenSSH client + `scp` ship with
    Windows 10+; we shell out using `subprocess.run`. Linux/macOS
    hosts work too — same shell-out, same flags.

This module talks ONLY to local paths + an SSH-reachable worker; it
never goes near the internet, so it's a strict bandwidth saver.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import compute_pool, db

log = logging.getLogger(__name__)


# Where Ollama keeps its on-disk store. Default location on every
# platform Ollama supports; same path the auto-router resolves models
# from.
_HOST_OLLAMA_DIR = Path.home() / ".ollama" / "models"


@dataclass
class CopyPlan:
    """What we'd ship if we ran the copy. Used by the API to preview
    before doing the actual transfer (a 17 GB model is a real commitment
    even over LAN). Fields mirror what the user wants to confirm:

      * `manifest_path` — local path on host
      * `manifest_dest` — remote path on worker
      * `blob_digests` — list of `sha256-…` filenames to copy
      * `total_bytes` — sum of blob sizes (for progress UI)
      * `missing_on_worker` — subset of blob_digests not yet on worker
        (after a previous partial copy) — only this set gets shipped
    """
    manifest_path: Path
    manifest_dest: str
    blob_digests: list[str]
    total_bytes: int
    missing_on_worker: list[str]


class ModelSyncError(RuntimeError):
    """Surfaces user-readable failures (no SSH host configured, manifest
    missing on host, scp returned non-zero, etc.)."""


def _split_model_tag(model_name: str) -> tuple[str, str]:
    """Split `name:tag` into `(name, tag)`; default tag is `latest`."""
    if ":" in model_name:
        bare, tag = model_name.split(":", 1)
        return bare, tag
    return model_name, "latest"


def _host_manifest_path(model_name: str) -> Path:
    """Where on the host's disk Ollama stores this model's manifest.

    Default registry/namespace assumed. The router's `resolve_ollama_model`
    already handles the rare custom-registry case; for sync we keep it
    simple — the user's manually-pulled models all land under the
    default path.
    """
    bare, tag = _split_model_tag(model_name)
    return _HOST_OLLAMA_DIR / "manifests" / "registry.ollama.ai" / "library" / bare / tag


def _read_manifest(model_name: str) -> dict:
    p = _host_manifest_path(model_name)
    if not p.is_file():
        raise ModelSyncError(
            f"manifest for {model_name!r} not found on host at {p} — "
            "pull the model on this host first (e.g. `ollama pull <model>`) "
            "so it can be copied to workers from here."
        )
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise ModelSyncError(f"failed to parse manifest for {model_name!r}: {e}")


def _all_blob_digests(manifest: dict) -> list[tuple[str, int]]:
    """Return [(sha256-…, size_bytes), …] for every layer + the config.

    Both `layers[]` and `config` reference blobs by digest. Ollama needs
    all of them — config blob is small (~hundreds of bytes) and
    describes the model, layers carry weights/license/template.
    """
    out: list[tuple[str, int]] = []
    cfg = manifest.get("config") or {}
    if cfg.get("digest"):
        out.append((cfg["digest"].replace("sha256:", "sha256-"), int(cfg.get("size") or 0)))
    for layer in manifest.get("layers") or []:
        if layer.get("digest"):
            out.append(
                (layer["digest"].replace("sha256:", "sha256-"), int(layer.get("size") or 0))
            )
    return out


def _ssh_for(worker: dict) -> str:
    """Return the SSH alias / hostname configured for `worker`.

    Raises ModelSyncError if no `ssh_host` is set — the user can't push
    to a worker without giving us a way to reach it over SSH. Setup
    instructions live in the error message so the operator knows
    exactly what to add.
    """
    ssh = (worker.get("ssh_host") or "").strip()
    if not ssh:
        raise ModelSyncError(
            f"worker {worker.get('label')!r} has no ssh_host configured. "
            "To enable LAN-first model copy, add an entry to your "
            "~/.ssh/config (e.g. `Host laptop` pointing at the worker), "
            "then set ssh_host on this worker to that alias."
        )
    return ssh


async def _ssh_check_blobs(ssh_host: str, digests: list[str]) -> set[str]:
    """Return the subset of `digests` that already exist on the worker.

    Lets us skip already-present blobs on a re-push (e.g. a previous
    sync was interrupted). Implemented with a single `ssh` invocation
    that lists the worker's blobs dir and we filter locally.

    Returns an empty set on any error (treat as "nothing present, copy
    everything") — better to over-copy than miss a blob.
    """
    cmd = [
        "ssh", *compute_pool._ssh_persistent_args(),
        "-o", "BatchMode=yes", ssh_host,
        # Cross-shell-friendly: use python -c to read the blobs dir
        # rather than ls + glob (Windows cmd.exe doesn't glob, PowerShell
        # globs differently, etc.).
        "powershell", "-NoProfile", "-Command",
        '"Get-ChildItem $env:USERPROFILE\\.ollama\\models\\blobs -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name"',
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _stderr = await proc.communicate()
    except FileNotFoundError:
        raise ModelSyncError("`ssh` not found on PATH — install OpenSSH client")
    except Exception as e:
        log.warning("model_sync: blob-list ssh failed: %s", e)
        return set()
    if proc.returncode != 0:
        return set()
    present = set(stdout.decode("utf-8", errors="replace").split())
    return {d for d in digests if d in present}


def plan(model_name: str, worker_id: str) -> CopyPlan:
    """Synchronously inspect what would be shipped without doing the
    actual transfer. Returns CopyPlan; raises ModelSyncError on the
    obvious can't-proceed cases (manifest missing, no ssh_host)."""
    worker = db.get_compute_worker(worker_id)
    if not worker:
        raise ModelSyncError("worker not found")
    _ = _ssh_for(worker)  # validate now so the API surfaces a clear error
    manifest = _read_manifest(model_name)
    blob_pairs = _all_blob_digests(manifest)
    digests = [d for d, _ in blob_pairs]
    total = sum(s for _, s in blob_pairs)
    bare, tag = _split_model_tag(model_name)
    return CopyPlan(
        manifest_path=_host_manifest_path(model_name),
        manifest_dest=(
            f"~/.ollama/models/manifests/registry.ollama.ai/library/{bare}/{tag}"
        ),
        blob_digests=digests,
        total_bytes=total,
        missing_on_worker=digests,  # populated by sync_model
    )


async def _scp(local: Path, remote_target: str) -> None:
    """Run `scp` synchronously. Surfaces stderr on non-zero exit."""
    cmd = [
        "scp", *compute_pool._ssh_persistent_args(),
        "-o", "BatchMode=yes", str(local), remote_target,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await proc.communicate()
    except FileNotFoundError:
        raise ModelSyncError("`scp` not found on PATH — install OpenSSH client")
    if proc.returncode != 0:
        raise ModelSyncError(
            f"scp failed ({proc.returncode}): "
            f"{stderr.decode('utf-8', errors='replace').strip()[:300]}"
        )


async def _ssh_mkdir(ssh_host: str, remote_dir: str) -> None:
    """`mkdir -p` on the worker. Cross-shell friendly via powershell so
    the same call works whether the worker's default shell is cmd or
    pwsh or bash."""
    cmd = [
        "ssh", *compute_pool._ssh_persistent_args(),
        "-o", "BatchMode=yes", ssh_host,
        "powershell", "-NoProfile", "-Command",
        f'"New-Item -ItemType Directory -Force -Path \\"{remote_dir}\\" | Out-Null"',
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await proc.communicate()
    except FileNotFoundError:
        raise ModelSyncError("`ssh` not found on PATH — install OpenSSH client")
    if proc.returncode != 0:
        raise ModelSyncError(
            f"ssh mkdir failed: "
            f"{stderr.decode('utf-8', errors='replace').strip()[:200]}"
        )


async def sync_model(model_name: str, worker_id: str) -> dict:
    """Push `model_name` from this host to the named worker.

    Steps:
      1. Validate (manifest exists on host, worker has ssh_host).
      2. Probe worker for already-present blobs; ship only the missing
         ones (resumable on a partial previous transfer).
      3. SCP each blob to `~/.ollama/models/blobs/`.
      4. SCP the manifest into the worker's manifests tree LAST, so a
         partial copy doesn't fool the worker's Ollama into thinking
         the model is loadable.

    Returns a summary dict with paths + byte counts. Raises
    ModelSyncError on any unrecoverable error.
    """
    worker = db.get_compute_worker(worker_id)
    if not worker:
        raise ModelSyncError("worker not found")
    ssh_host = _ssh_for(worker)
    manifest = _read_manifest(model_name)
    bare, tag = _split_model_tag(model_name)
    blob_pairs = _all_blob_digests(manifest)
    if not blob_pairs:
        raise ModelSyncError("manifest references no blobs — refusing to copy")

    digests = [d for d, _ in blob_pairs]
    sizes = {d: s for d, s in blob_pairs}

    # Pre-create the target directories on the worker so scp doesn't
    # silently land manifests in the wrong place if the manifests tree
    # doesn't exist yet on a fresh Ollama install.
    blobs_dir = "~/.ollama/models/blobs"
    manifests_dir = (
        f"~/.ollama/models/manifests/registry.ollama.ai/library/{bare}"
    )
    await _ssh_mkdir(ssh_host, blobs_dir)
    await _ssh_mkdir(ssh_host, manifests_dir)

    # Skip blobs already present on the worker.
    already_there = await _ssh_check_blobs(ssh_host, digests)
    missing = [d for d in digests if d not in already_there]

    blobs_root = _HOST_OLLAMA_DIR / "blobs"
    bytes_shipped = 0
    for digest in missing:
        local = blobs_root / digest
        if not local.is_file():
            raise ModelSyncError(
                f"local blob missing: {local} — Ollama's store is corrupt? "
                "Try `ollama pull {model_name}` on host to restore it."
            )
        await _scp(local, f"{ssh_host}:{blobs_dir}/")
        bytes_shipped += sizes.get(digest, 0)

    # Manifest last — avoids a partial state where the worker thinks
    # the model is fully present but is missing a blob.
    manifest_local = _host_manifest_path(model_name)
    await _scp(manifest_local, f"{ssh_host}:{manifests_dir}/{tag}")

    return {
        "ok": True,
        "model": model_name,
        "worker_id": worker_id,
        "ssh_host": ssh_host,
        "blobs_total": len(digests),
        "blobs_already_present": len(already_there),
        "blobs_shipped": len(missing),
        "bytes_shipped": bytes_shipped,
    }


def find_lan_source_for(model_name: str, exclude_worker_id: str | None = None) -> dict | None:
    """Search the registered compute pool for a node that already has
    `model_name` available — preferring host first, then any worker
    whose latest probe lists the model in its `capabilities.models`.

    Returns the source descriptor:
        {"kind": "host"}                                          → host has it
        {"kind": "worker", "worker_id": ..., "label": ...}        → a peer has it
        None                                                       → nobody has it; caller falls back to internet pull

    `exclude_worker_id` is the destination worker — we don't want to
    "find" the model on the very node that's about to receive it.

    Note: peer→peer copy isn't implemented yet (would need worker-to-
    worker SSH, rare in practice). For now this function reports who
    has it; only the host case has a working push path. The router
    can use this to short-circuit "already on host" → push from host;
    "only on a peer" → surface that as a hint to the user ("model
    found on worker X — install on host first to enable LAN copy").
    """
    if _host_manifest_path(model_name).is_file():
        return {"kind": "host"}

    for w in db.list_compute_workers(enabled_only=False):
        if exclude_worker_id and w["id"] == exclude_worker_id:
            continue
        caps = w.get("capabilities") or {}
        for m in caps.get("models") or []:
            if compute_pool._model_matches(m.get("name") or "", model_name):
                return {"kind": "worker", "worker_id": w["id"], "label": w["label"]}
    return None
