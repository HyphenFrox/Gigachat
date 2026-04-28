"""Regression: LAN-first model copy (Phase 2 commit 10).

Pins:
  * Manifest parsing — `_all_blob_digests` extracts the right shas from
    Ollama's manifest JSON shape.
  * `_ssh_for` raises a clear error when ssh_host isn't set.
  * `find_lan_source_for` reports host first, then peer worker, then None.
  * `plan` produces a CopyPlan with the right blob count + total bytes.
  * `sync_model` ships only missing blobs and writes the manifest LAST.

ssh + scp are stubbed via a fake `asyncio.create_subprocess_exec` so
tests don't actually invoke either binary. Manifest + blob files live
under tmp_path so the developer's real ~/.ollama/ store is untouched.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from backend import model_sync, compute_pool

pytestmark = pytest.mark.smoke


# --- helpers --------------------------------------------------------------


def _redirect_ollama_dir(monkeypatch, tmp_path: Path) -> Path:
    """Point _HOST_OLLAMA_DIR at a temp directory so manifest/blob
    reads don't touch the developer's real Ollama store."""
    fake = tmp_path / "ollama" / "models"
    fake.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(model_sync, "_HOST_OLLAMA_DIR", fake)
    return fake


def _seed_manifest(ollama_dir: Path, model_name: str, layers: list[tuple[str, int]]) -> Path:
    """Drop a manifest JSON + matching blob files into the temp store."""
    bare, tag = (model_name.split(":", 1) + ["latest"])[:2] if ":" in model_name else (model_name, "latest")
    manifests_dir = ollama_dir / "manifests" / "registry.ollama.ai" / "library" / bare
    manifests_dir.mkdir(parents=True, exist_ok=True)
    blobs_dir = ollama_dir / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    # Pick a config digest that's unique per model_name but doesn't
    # contain a colon (only the leading `sha256:` prefix is a colon).
    cfg_inner = "cfg" + bare + tag
    manifest = {
        "schemaVersion": 2,
        "config": {
            "digest": f"sha256:{cfg_inner}",
            "size": 420,
        },
        "layers": [
            {"digest": f"sha256:{digest}", "size": size, "mediaType": "application/vnd.ollama.image.model"}
            for digest, size in layers
        ],
    }
    manifest_path = manifests_dir / tag
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Drop matching blob files (zero-byte placeholders are fine).
    (blobs_dir / f"sha256-{cfg_inner}").write_bytes(b"")
    for digest, _ in layers:
        (blobs_dir / f"sha256-{digest}").write_bytes(b"")

    return manifest_path


class _FakeProc:
    """Minimal stand-in for asyncio's subprocess.Process — enough that
    `await proc.communicate()` returns (stdout, stderr) and .returncode
    behaves."""

    def __init__(self, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self):
        return self._stdout, self._stderr


def _stub_subprocess(monkeypatch, *, blobs_already_present: list[str] | None = None):
    """Replace `asyncio.create_subprocess_exec` with a recording stub.

    Returns the calls list — each entry is the argv tuple the test code
    invoked. SSH list-blobs calls return `blobs_already_present` (defaults
    to empty); scp / mkdir calls return success.
    """
    calls: list[list[str]] = []
    blobs_already_present = blobs_already_present or []

    async def fake_exec(*args, **kwargs):
        calls.append(list(args))
        # SSH-via-powershell list-blobs heuristic — the only ssh call
        # whose stdout we care about.
        if "Get-ChildItem" in " ".join(args):
            return _FakeProc(stdout="\n".join(blobs_already_present).encode("utf-8"))
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return calls


def _seed_worker(isolated_db, *, ssh_host: str | None = "laptop"):
    return isolated_db.create_compute_worker(
        label="L", address="x.local", ssh_host=ssh_host,
    )


def _run(coro):
    return asyncio.run(coro)


# --- manifest parsing ----------------------------------------------------


def test_all_blob_digests_includes_config_and_layers():
    manifest = {
        "config": {"digest": "sha256:cfg1", "size": 100},
        "layers": [
            {"digest": "sha256:abc", "size": 1000},
            {"digest": "sha256:def", "size": 2000},
        ],
    }
    out = model_sync._all_blob_digests(manifest)
    digests = [d for d, _ in out]
    sizes = [s for _, s in out]
    assert "sha256-cfg1" in digests
    assert "sha256-abc" in digests
    assert "sha256-def" in digests
    assert sum(sizes) == 100 + 1000 + 2000


# --- _ssh_for + find_lan_source_for --------------------------------------


def test_ssh_for_raises_when_no_alias_set(isolated_db):
    wid = _seed_worker(isolated_db, ssh_host=None)
    with pytest.raises(model_sync.ModelSyncError, match="no ssh_host"):
        model_sync._ssh_for(isolated_db.get_compute_worker(wid))


def test_find_lan_source_returns_host_when_host_has_model(
    isolated_db, monkeypatch, tmp_path
):
    monkeypatch.setattr(model_sync, "db", isolated_db)
    fake = _redirect_ollama_dir(monkeypatch, tmp_path)
    _seed_manifest(fake, "x:tag", [("a", 100)])
    src = model_sync.find_lan_source_for("x:tag")
    assert src == {"kind": "host"}


def test_find_lan_source_returns_worker_when_host_lacks(isolated_db, monkeypatch, tmp_path):
    """Host doesn't have the model but a registered worker does
    (via its capabilities)."""
    monkeypatch.setattr(model_sync, "db", isolated_db)
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _redirect_ollama_dir(monkeypatch, tmp_path)
    wid = isolated_db.create_compute_worker(label="A", address="a.local")
    isolated_db.update_compute_worker_capabilities(
        wid,
        capabilities={
            "version": "0.5.4",
            "models": [{"name": "z:big", "details": {}}],
            "rpc_server_reachable": False, "rpc_port": 50052, "rpc_error": None,
            "gpu_present": False, "max_vram_seen_bytes": 0, "loaded_count": 0,
        },
        last_seen=1000.0, last_error="",
    )
    src = model_sync.find_lan_source_for("z:big")
    assert src is not None
    assert src["kind"] == "worker"
    assert src["worker_id"] == wid


def test_find_lan_source_returns_none_when_nobody_has_model(isolated_db, monkeypatch, tmp_path):
    monkeypatch.setattr(model_sync, "db", isolated_db)
    _redirect_ollama_dir(monkeypatch, tmp_path)
    src = model_sync.find_lan_source_for("nothing:here")
    assert src is None


# --- plan ----------------------------------------------------------------


def test_plan_summarizes_blobs(isolated_db, monkeypatch, tmp_path):
    monkeypatch.setattr(model_sync, "db", isolated_db)
    fake = _redirect_ollama_dir(monkeypatch, tmp_path)
    wid = _seed_worker(isolated_db, ssh_host="laptop")
    _seed_manifest(fake, "demo:latest", [("aa", 1000), ("bb", 2000)])
    p = model_sync.plan("demo:latest", wid)
    # config + 2 layers.
    assert len(p.blob_digests) == 3
    assert p.total_bytes == 1000 + 2000 + 420  # config size
    assert "demo" in p.manifest_dest


def test_plan_raises_when_manifest_missing(isolated_db, monkeypatch, tmp_path):
    monkeypatch.setattr(model_sync, "db", isolated_db)
    _redirect_ollama_dir(monkeypatch, tmp_path)
    wid = _seed_worker(isolated_db, ssh_host="laptop")
    with pytest.raises(model_sync.ModelSyncError, match="manifest"):
        model_sync.plan("not-pulled", wid)


# --- sync_model: actual orchestration ------------------------------------


def test_sync_model_ships_blobs_and_manifest(isolated_db, monkeypatch, tmp_path):
    monkeypatch.setattr(model_sync, "db", isolated_db)
    fake = _redirect_ollama_dir(monkeypatch, tmp_path)
    wid = _seed_worker(isolated_db, ssh_host="laptop")
    _seed_manifest(fake, "demo:latest", [("aa", 1000), ("bb", 2000)])

    calls = _stub_subprocess(monkeypatch)
    result = _run(model_sync.sync_model("demo:latest", wid))

    assert result["ok"] is True
    assert result["model"] == "demo:latest"
    # config + 2 layers = 3 blobs to ship (worker has nothing).
    assert result["blobs_shipped"] == 3
    assert result["blobs_already_present"] == 0

    # Confirm we ran scp for each blob + the manifest. Calls list
    # contains each subprocess invocation. Filter to scps:
    scp_calls = [c for c in calls if c and c[0] == "scp"]
    # 3 blobs + 1 manifest = 4 scp calls.
    assert len(scp_calls) == 4

    # Manifest must be the LAST scp — partial transfer protection.
    assert "demo" in scp_calls[-1][-1]


def test_sync_model_skips_already_present_blobs(isolated_db, monkeypatch, tmp_path):
    """If the worker already has some blobs (from a previous partial
    push), skip them — no point re-shipping bytes."""
    monkeypatch.setattr(model_sync, "db", isolated_db)
    fake = _redirect_ollama_dir(monkeypatch, tmp_path)
    wid = _seed_worker(isolated_db, ssh_host="laptop")
    _seed_manifest(fake, "demo:latest", [("aa", 1000), ("bb", 2000)])

    # Tell the SSH-list stub: 'aa' already on worker. config (sha256-cfgdemolatest) + 'bb' should ship.
    calls = _stub_subprocess(monkeypatch, blobs_already_present=["sha256-aa"])

    result = _run(model_sync.sync_model("demo:latest", wid))
    assert result["blobs_already_present"] == 1
    assert result["blobs_shipped"] == 2  # cfg + bb

    scp_calls = [c for c in calls if c and c[0] == "scp"]
    # Should NOT have scp'd 'sha256-aa'.
    assert not any("sha256-aa" in (c[-2] if len(c) > 1 else "") for c in scp_calls)


def test_sync_model_raises_when_no_ssh_host(isolated_db, monkeypatch, tmp_path):
    monkeypatch.setattr(model_sync, "db", isolated_db)
    fake = _redirect_ollama_dir(monkeypatch, tmp_path)
    wid = _seed_worker(isolated_db, ssh_host=None)
    _seed_manifest(fake, "demo:latest", [("aa", 1000)])
    with pytest.raises(model_sync.ModelSyncError, match="no ssh_host"):
        _run(model_sync.sync_model("demo:latest", wid))


def test_sync_model_raises_when_local_blob_missing(isolated_db, monkeypatch, tmp_path):
    """Manifest references a blob that doesn't exist on disk → corrupt
    Ollama store. Surface the failure clearly rather than silently
    shipping a 0-byte file."""
    monkeypatch.setattr(model_sync, "db", isolated_db)
    fake = _redirect_ollama_dir(monkeypatch, tmp_path)
    wid = _seed_worker(isolated_db, ssh_host="laptop")
    _seed_manifest(fake, "demo:latest", [("aa", 1000)])
    # Remove the blob file after seeding.
    (fake / "blobs" / "sha256-aa").unlink()

    _stub_subprocess(monkeypatch)
    with pytest.raises(model_sync.ModelSyncError, match="local blob missing"):
        _run(model_sync.sync_model("demo:latest", wid))
