"""Regression: split_lifecycle — start / stop / status invariants.

Phase 2 commit 4. Tests pin the parts that don't need a real
llama-server binary:

  * `_build_command` produces the exact argv we promise (including
    --rpc presence/absence, -ngl default, --jinja, etc.)
  * `_resolve_rpc_endpoints` filters correctly: unknown/disabled
    workers and workers whose probe says rpc-server is down get
    dropped (they'd just cause llama-server to error mid-load).
  * `_preflight` raises clear messages on missing binary / missing
    GGUF / unknown row.
  * `start` happy path drives the right subprocess + DB transitions
    (loading → running, last_error cleared) — `subprocess.Popen` and
    `_wait_for_health` are stubbed.
  * `start` health-timeout path kills the child + marks status=error.
  * `stop` is idempotent and resets DB status from a `running` to
    `stopped`.
  * `status` reconciles registry (process alive?) with DB row,
    surfacing `crashed` when DB thinks `running` but the OS process
    is gone.

The actual subprocess spawn against a real binary is left to manual
end-to-end testing — too brittle / slow / install-dependent for CI.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from backend import split_lifecycle, split_runtime

pytestmark = pytest.mark.smoke


# --- _build_command ------------------------------------------------------


def test_build_command_minimal_no_rpc():
    cmd = split_lifecycle._build_command(
        llama_server=Path("C:/llama-server.exe"),
        gguf_path="C:/m.gguf",
        port=11500,
        rpc_endpoints=[],
    )
    assert cmd[0] == "C:\\llama-server.exe" or cmd[0] == "C:/llama-server.exe"
    # Required flag pairs.
    assert "--model" in cmd and "C:/m.gguf" in cmd
    assert "--port" in cmd and "11500" in cmd
    assert "--host" in cmd and "127.0.0.1" in cmd
    # No rpc workers → no --rpc flag.
    assert "--rpc" not in cmd
    # Default ngl is the "all layers to GPU" sentinel.
    assert "-ngl" in cmd
    assert str(split_lifecycle._DEFAULT_NGL) in cmd
    # Jinja chat templates on (most modern GGUFs ship one).
    assert "--jinja" in cmd


def test_build_command_with_rpc_endpoints_in_order():
    cmd = split_lifecycle._build_command(
        llama_server=Path("C:/llama-server.exe"),
        gguf_path="/m.gguf",
        port=11500,
        rpc_endpoints=["a.local:50052", "b.local:50052"],
    )
    # llama-server takes --rpc as a comma-separated list.
    assert "--rpc" in cmd
    rpc_idx = cmd.index("--rpc")
    assert cmd[rpc_idx + 1] == "a.local:50052,b.local:50052"


def test_build_command_custom_ngl():
    cmd = split_lifecycle._build_command(
        llama_server=Path("C:/llama-server.exe"),
        gguf_path="/m.gguf",
        port=11500,
        rpc_endpoints=[],
        ngl=20,
    )
    ngl_idx = cmd.index("-ngl")
    assert cmd[ngl_idx + 1] == "20"


# --- _resolve_rpc_endpoints ----------------------------------------------


def _seed_worker_with_rpc(isolated_db, *, label, address, rpc_reachable: bool, enabled: bool = True) -> str:
    """Helper: create a worker AND record a probe outcome with rpc state."""
    import time
    wid = isolated_db.create_compute_worker(
        label=label, address=address, transport="lan", enabled=enabled,
    )
    isolated_db.update_compute_worker_capabilities(
        wid,
        capabilities={
            "version": "0.5.4",
            "models": [],
            "rpc_server_reachable": rpc_reachable,
            "rpc_port": 50052,
            "rpc_error": None if rpc_reachable else "stubbed: not reachable",
        },
        last_seen=time.time(),
        last_error="",
    )
    return wid


def test_resolve_endpoints_includes_only_reachable_workers(isolated_db, monkeypatch):
    monkeypatch.setattr(split_lifecycle, "db", isolated_db)
    a = _seed_worker_with_rpc(isolated_db, label="A", address="a.local", rpc_reachable=True)
    b = _seed_worker_with_rpc(isolated_db, label="B", address="b.local", rpc_reachable=False)
    c = _seed_worker_with_rpc(isolated_db, label="C", address="c.local", rpc_reachable=True)
    eps = split_lifecycle._resolve_rpc_endpoints([a, b, c])
    # B dropped (rpc unreachable); A and C kept in order.
    assert eps == ["a.local:50052", "c.local:50052"]


def test_resolve_endpoints_skips_disabled(isolated_db, monkeypatch):
    monkeypatch.setattr(split_lifecycle, "db", isolated_db)
    a = _seed_worker_with_rpc(
        isolated_db, label="A", address="a.local", rpc_reachable=True, enabled=False,
    )
    eps = split_lifecycle._resolve_rpc_endpoints([a])
    assert eps == []


def test_resolve_endpoints_skips_unknown_id(isolated_db, monkeypatch):
    monkeypatch.setattr(split_lifecycle, "db", isolated_db)
    eps = split_lifecycle._resolve_rpc_endpoints(["never-existed"])
    assert eps == []


def test_resolve_endpoints_strips_http_prefix(isolated_db, monkeypatch):
    """A user might paste an Ollama-style URL into the worker form;
    the rpc endpoint must be the bare host:port, not http://host:port."""
    monkeypatch.setattr(split_lifecycle, "db", isolated_db)
    a = _seed_worker_with_rpc(
        isolated_db, label="A", address="http://a.local/", rpc_reachable=True,
    )
    eps = split_lifecycle._resolve_rpc_endpoints([a])
    assert eps == ["a.local:50052"]


# --- _preflight ----------------------------------------------------------


def test_preflight_raises_when_llama_server_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(split_runtime, "LLAMA_CPP_INSTALL_DIR", tmp_path / "nope")
    monkeypatch.setenv("PATH", "")  # nothing on PATH either
    with pytest.raises(split_lifecycle.SplitLifecycleError, match="not installed"):
        split_lifecycle._preflight(
            {"gguf_path": "/m.gguf", "worker_ids": []}
        )


def test_preflight_raises_when_gguf_missing(monkeypatch, tmp_path):
    """Even with llama-server installed, a non-existent GGUF must
    surface a clear error before we burn time on a doomed start."""
    install = tmp_path / "llama-cpp"
    install.mkdir()
    # Platform-aware fake binary: Windows expects `.exe`, POSIX
    # expects no extension. Without this the test passes on Windows
    # (where the binary name happens to match) but fails on Linux CI
    # (`find_llama_server` looks for `llama-server`, not `.exe`).
    import sys
    exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    (install / exe_name).write_bytes(b"")
    monkeypatch.setattr(split_runtime, "LLAMA_CPP_INSTALL_DIR", install)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(split_lifecycle.SplitLifecycleError, match="does not exist"):
        split_lifecycle._preflight(
            {"gguf_path": "/does/not/exist.gguf", "worker_ids": []}
        )


def test_preflight_happy_path(monkeypatch, tmp_path):
    install = tmp_path / "llama-cpp"
    install.mkdir()
    # Platform-aware fake binary: Windows expects `.exe`, POSIX
    # expects no extension. Without this the test passes on Windows
    # (where the binary name happens to match) but fails on Linux CI
    # (`find_llama_server` looks for `llama-server`, not `.exe`).
    import sys
    exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    (install / exe_name).write_bytes(b"")
    monkeypatch.setattr(split_runtime, "LLAMA_CPP_INSTALL_DIR", install)
    monkeypatch.setenv("PATH", "")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"")
    server, eps = split_lifecycle._preflight(
        {"gguf_path": str(gguf), "worker_ids": []}
    )
    assert server == install / exe_name
    assert eps == []


# --- start (subprocess stubbed) ------------------------------------------


class _FakePopen:
    """Drop-in for subprocess.Popen that records cmd + stays "alive"
    until .terminate() is called, mirroring the real interface as much
    as the lifecycle module touches."""

    def __init__(self, cmd, *args, **kwargs):
        self.cmd = cmd
        self.args = args
        self.kwargs = kwargs
        self.pid = 12345
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self._alive = False

    def kill(self):
        self._alive = False

    def wait(self, timeout=None):
        self._alive = False
        return 0


def _arrange_start(isolated_db, monkeypatch, tmp_path):
    """Common scaffolding for start/stop tests — install fake binary,
    create a real GGUF on disk, register a split_model row."""
    install = tmp_path / "llama-cpp"
    install.mkdir()
    # Platform-aware fake binary: Windows expects `.exe`, POSIX
    # expects no extension. Without this the test passes on Windows
    # (where the binary name happens to match) but fails on Linux CI
    # (`find_llama_server` looks for `llama-server`, not `.exe`).
    import sys
    exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    (install / exe_name).write_bytes(b"")
    monkeypatch.setattr(split_runtime, "LLAMA_CPP_INSTALL_DIR", install)
    monkeypatch.setenv("PATH", "")

    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"")

    monkeypatch.setattr(split_lifecycle, "db", isolated_db)

    sid = isolated_db.create_split_model(
        label="big-model", gguf_path=str(gguf), llama_port=11500,
    )

    # Stub Popen.
    fakes: list[_FakePopen] = []

    def _fake_popen_factory(cmd, *args, **kwargs):
        fp = _FakePopen(cmd, *args, **kwargs)
        fakes.append(fp)
        return fp

    monkeypatch.setattr(split_lifecycle.subprocess, "Popen", _fake_popen_factory)

    # Stub the health wait — succeed immediately.
    # Accepts the same kwargs as the real function so call sites that
    # pass `proc=` (added to bail early on child death) don't trip a
    # TypeError on this stub.
    async def _stub_health(port, timeout=90.0, proc=None):
        return None

    monkeypatch.setattr(split_lifecycle, "_wait_for_health", _stub_health)
    return sid, fakes


def test_start_happy_path(isolated_db, monkeypatch, tmp_path):
    sid, fakes = _arrange_start(isolated_db, monkeypatch, tmp_path)

    result = asyncio.run(split_lifecycle.start(sid))
    assert result["ok"] is True
    assert result["status"] == "running"
    assert result["port"] == 11500

    # DB transitions: loading → running, error cleared.
    row = isolated_db.get_split_model(sid)
    assert row["status"] == "running"
    assert row["last_error"] is None

    # Exactly one Popen invocation; argv contains --port 11500 and
    # the GGUF path.
    assert len(fakes) == 1
    cmd = fakes[0].cmd
    assert "11500" in cmd

    # Cleanup so the registry is clean for the next test.
    asyncio.run(split_lifecycle.stop(sid))


def test_start_idempotent_when_already_running(isolated_db, monkeypatch, tmp_path):
    sid, fakes = _arrange_start(isolated_db, monkeypatch, tmp_path)
    asyncio.run(split_lifecycle.start(sid))
    again = asyncio.run(split_lifecycle.start(sid))
    assert again["ok"] is True
    assert again["status"] == "running"
    assert again.get("note") == "already running"
    # Only one process spawned.
    assert len(fakes) == 1
    asyncio.run(split_lifecycle.stop(sid))


def test_start_health_timeout_marks_error(isolated_db, monkeypatch, tmp_path):
    sid, fakes = _arrange_start(isolated_db, monkeypatch, tmp_path)

    async def _failing_health(port, timeout=90.0, proc=None):
        raise split_lifecycle.SplitLifecycleError(
            "llama-server on port X did not become healthy"
        )

    monkeypatch.setattr(split_lifecycle, "_wait_for_health", _failing_health)

    result = asyncio.run(split_lifecycle.start(sid))
    assert result["ok"] is False
    assert result["status"] == "error"
    assert "did not become healthy" in result["error"]
    row = isolated_db.get_split_model(sid)
    assert row["status"] == "error"
    assert row["last_error"]
    # Child was killed during cleanup.
    assert fakes[0].poll() == 0


# --- stop ----------------------------------------------------------------


def test_stop_idempotent_when_not_running(isolated_db, monkeypatch, tmp_path):
    sid, _ = _arrange_start(isolated_db, monkeypatch, tmp_path)
    result = asyncio.run(split_lifecycle.stop(sid))
    assert result["ok"] is True
    assert result["status"] == "stopped"


def test_stop_resets_stale_db_status(isolated_db, monkeypatch, tmp_path):
    """If the DB says `running` but the in-memory registry has
    nothing (app restarted, process gone), stopping must reset the
    DB back to `stopped` so the UI doesn't show a phantom green pill."""
    sid, _ = _arrange_start(isolated_db, monkeypatch, tmp_path)
    isolated_db.update_split_model_status(sid, status="running")
    result = asyncio.run(split_lifecycle.stop(sid))
    assert result["ok"] is True
    assert isolated_db.get_split_model(sid)["status"] == "stopped"


# --- status --------------------------------------------------------------


def test_status_reflects_running_when_alive(isolated_db, monkeypatch, tmp_path):
    sid, _ = _arrange_start(isolated_db, monkeypatch, tmp_path)
    asyncio.run(split_lifecycle.start(sid))
    s = split_lifecycle.status(sid)
    assert s["ok"] is True
    assert s["alive"] is True
    assert s["effective_status"] == "running"
    asyncio.run(split_lifecycle.stop(sid))


def test_status_surfaces_crashed_when_process_died(isolated_db, monkeypatch, tmp_path):
    """DB says `running`, but the process died (or never was). Reported
    as `crashed` so the UI can show a distinct state vs a clean
    `stopped` row the user explicitly halted."""
    sid, fakes = _arrange_start(isolated_db, monkeypatch, tmp_path)
    asyncio.run(split_lifecycle.start(sid))
    # Simulate process death without the registry knowing.
    fakes[0].terminate()
    s = split_lifecycle.status(sid)
    assert s["effective_status"] == "crashed"
    asyncio.run(split_lifecycle.stop(sid))


def test_status_unknown_id(isolated_db, monkeypatch):
    monkeypatch.setattr(split_lifecycle, "db", isolated_db)
    s = split_lifecycle.status("nonexistent")
    assert s["ok"] is False
    assert "not found" in s["error"]


# --- reconcile_on_boot ---------------------------------------------------


def test_reconcile_resets_running_rows_to_stopped(isolated_db, monkeypatch):
    """A row that the DB thinks is `running` after an app crash must
    be reset, with a clear last_error explaining why."""
    monkeypatch.setattr(split_lifecycle, "db", isolated_db)
    sid_running = isolated_db.create_split_model(label="A", gguf_path="/a.gguf")
    isolated_db.update_split_model_status(sid_running, status="running")
    sid_loading = isolated_db.create_split_model(label="B", gguf_path="/b.gguf")
    isolated_db.update_split_model_status(sid_loading, status="loading")
    sid_stopped = isolated_db.create_split_model(label="C", gguf_path="/c.gguf")
    # Already stopped — should NOT be touched.

    n = split_lifecycle.reconcile_on_boot()
    assert n == 2

    # A and B reset; their last_error explains.
    a = isolated_db.get_split_model(sid_running)
    b = isolated_db.get_split_model(sid_loading)
    c = isolated_db.get_split_model(sid_stopped)
    assert a["status"] == "stopped"
    assert "boot" in (a["last_error"] or "")
    assert b["status"] == "stopped"
    # C untouched (already stopped, no error history to overwrite).
    assert c["status"] == "stopped"
    assert c["last_error"] is None


def test_reconcile_no_op_when_all_clean(isolated_db, monkeypatch):
    monkeypatch.setattr(split_lifecycle, "db", isolated_db)
    isolated_db.create_split_model(label="A", gguf_path="/a.gguf")
    n = split_lifecycle.reconcile_on_boot()
    assert n == 0


# --- read_log_tail -------------------------------------------------------


def test_read_log_tail_empty_when_no_log(isolated_db, monkeypatch, tmp_path):
    monkeypatch.setattr(split_runtime, "LLAMA_CPP_INSTALL_DIR", tmp_path / "llama-cpp")
    monkeypatch.setattr(split_lifecycle, "db", isolated_db)
    assert split_lifecycle.read_log_tail("any-id") == ""


def test_read_log_tail_returns_last_n_lines(isolated_db, monkeypatch, tmp_path):
    monkeypatch.setattr(split_runtime, "LLAMA_CPP_INSTALL_DIR", tmp_path / "llama-cpp")
    monkeypatch.setattr(split_lifecycle, "db", isolated_db)
    log_path = split_lifecycle._log_path_for("test-id")
    log_path.write_text(
        "\n".join(f"line {i}" for i in range(200)) + "\n",
        encoding="utf-8",
    )
    out = split_lifecycle.read_log_tail("test-id", lines=20)
    out_lines = out.splitlines()
    # Last 20 should land — line 180 .. line 199.
    assert len(out_lines) == 20
    assert out_lines[-1] == "line 199"
    assert out_lines[0] == "line 180"
