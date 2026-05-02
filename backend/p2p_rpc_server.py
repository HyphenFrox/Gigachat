"""Local rpc-server lifecycle, spawned by a paired peer over the
encrypted P2P channel rather than SSH.

Why this exists
===============
``compute_pool._attempt_rpc_server_restart`` already knows how to bring
up llama.cpp's ``rpc-server`` on a worker — but it does the work over
SSH. That means a paired LAN peer with no ssh_host configured can
NEVER become eligible for split-model inference, because:

  * ``compute_pool._eligible_split_workers`` filters on
    ``capabilities.rpc_server_reachable``
  * ``rpc_server_reachable`` is only set after a successful SSH-driven
    restart attempt
  * paired peers don't have ``ssh_host`` by default — that field
    requires the user to manually set up SSH between machines

This module gives the orchestrator a second path: POST to the peer's
``/api/p2p/rpc-server/start`` endpoint via the existing encrypted
secure-proxy. The peer then spawns rpc-server.exe locally, in the
same user session as its Gigachat backend, with appropriate detach
flags so the child outlives the API handler.

Authentication: the endpoint is whitelisted in
``p2p_secure_proxy._FORWARDABLE_PATHS`` and routed to local Gigachat
(not Ollama) via ``_GIGACHAT_INTERNAL_PATHS``. So the only callers
are (a) loopback, (b) a paired peer whose envelope passes X25519+
ChaCha20+Ed25519 verification.

Scope
=====
Spawn / status / stop only. The ngl decision, layer placement, and
adaptive rebalance live in ``split_lifecycle``. This module just
makes sure the rpc-server process is running on each peer.
"""
from __future__ import annotations

import logging
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)


# Where the binary should be on every install. Matches the path
# `_attempt_rpc_server_restart` looks for when going via SSH so the
# two paths converge on the same artifact.
_RPC_SERVER_BIN_DIR = Path.home() / ".gigachat" / "llama-cpp"
_RPC_SERVER_EXE = _RPC_SERVER_BIN_DIR / (
    "rpc-server.exe" if platform.system() == "Windows" else "rpc-server"
)

# Default port + backend selection. Same as the SSH path.
_DEFAULT_PORT = 50052
_DEFAULT_BACKEND = "SYCL0,CPU"

# Backend the currently-running rpc-server was launched with. We
# track this on the peer so the orchestrator can ask "are you
# running CPU-only or SYCL+CPU?" and decide whether to request a
# restart. Reset to None when nothing's running, set in
# `start_local_rpc_server` on a successful spawn.
_active_backend: str | None = None

# Stability env vars for SYCL on Intel Xe2 / Meteor Lake.
# - GGML_SYCL_DISABLE_OPT=1 — workaround for #21893 (still OPEN
#   upstream); mandatory on Xe2 / Battlemage to avoid silent
#   weight corruption.
# - GGML_SYCL_DISABLE_GRAPH=1 — defensive net for the warmup-crash
#   regression that #21474 (closed) tracked.
# - SYCL_CACHE_PERSISTENT=1 — keeps SYCL JIT cache across runs.
# Set in the child's environment via subprocess rather than the
# WMI-style user-scope dance because we control the spawn directly
# (not through SSH).
_RPC_SPAWN_ENV: dict[str, str] = {
    "GGML_SYCL_DISABLE_OPT": "1",
    "GGML_SYCL_DISABLE_GRAPH": "1",
    "SYCL_CACHE_PERSISTENT": "1",
}

# How long to wait for the freshly-spawned process to start listening
# on the rpc port. 12 s — the SSH path used 4 s, but real-world SYCL
# initialization on Intel iGPUs takes 5-10 s on the FIRST start of a
# session (kernel JIT + driver warmup). After the first start the
# binary is fast (sub-second) thanks to SYCL_CACHE_PERSISTENT=1, but
# we'd rather pay 12 s once on cold start than have the orchestrator
# falsely conclude "spawn failed" and never engage split.
_LISTEN_WAIT_SEC = 12.0


def _is_listening_on(port: int) -> bool:
    """Return True iff TCP port `port` has at least one listener.

    Uses a simple connect-to-loopback probe — same signal the SSH
    PowerShell payload checks via ``Get-NetTCPConnection``, but
    portable. False on any error so a flaky probe never lies about
    a process being up.
    """
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def _kill_running_rpc_servers() -> int:
    """Best-effort kill of every existing rpc-server process owned by
    this user. Returns the count killed.

    Reuses psutil (already a hard dep of sysdetect) instead of
    shelling out to taskkill / pkill so the same code works on
    Windows + Linux + macOS without per-OS branches.
    """
    killed = 0
    try:
        import psutil
    except ImportError:
        log.debug("p2p_rpc_server: psutil missing, can't enumerate processes")
        return 0
    me = os.getlogin() if hasattr(os, "getlogin") else ""
    for p in psutil.process_iter(["pid", "name", "username"]):
        try:
            name = (p.info.get("name") or "").lower()
            if not name.startswith("rpc-server"):
                continue
            # Only touch processes owned by us — never reach into
            # another user's rpc-server (they may be running it for
            # an unrelated reason).
            if me and (p.info.get("username") or "").lower().split("\\")[-1] != me.lower():
                continue
            p.terminate()
            killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if killed:
        # Give terminate a moment to settle before any subsequent
        # spawn binds the same port.
        time.sleep(0.8)
    return killed


def _spawn_detached(cmd: list[str], cwd: str, env: dict[str, str]) -> int:
    """Spawn `cmd` so it survives this process exiting.

    Windows: DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP +
    CREATE_BREAKAWAY_FROM_JOB so the child isn't tied to the
    backend's job object.
    POSIX: start_new_session=True (setsid).

    stdout + stderr are redirected into ``rpc-server.log`` next to
    the binary. This is critical for diagnosing RPC crashes — when
    the orchestrator's llama-server reports
    ``ggml-rpc.cpp:534: Remote RPC server crashed``, the actual
    panic / segfault / "couldn't allocate" message lives in the
    rpc-server's stderr; sending it to DEVNULL would lose it. The
    log is opened append-only so successive spawns add to (rather
    than truncate) the history.

    Returns child PID.
    """
    log_path = _RPC_SERVER_BIN_DIR / "rpc-server.log"
    try:
        _RPC_SERVER_BIN_DIR.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "ab")
    except OSError:
        log_f = subprocess.DEVNULL  # type: ignore[assignment]
    if platform.system() == "Windows":
        # Constants from win32con; using literal ints keeps us free
        # of the pywin32 hard dep.
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_BREAKAWAY_FROM_JOB = 0x01000000
        flags = (
            DETACHED_PROCESS
            | CREATE_NEW_PROCESS_GROUP
            | CREATE_BREAKAWAY_FROM_JOB
        )
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            close_fds=True,
        )
        return proc.pid
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    return proc.pid


def _try_lower_priority(pid: int) -> None:
    """Best-effort drop the new rpc-server's priority to BelowNormal.

    Mirrors what the SSH path does. Mismatched priority means the
    user feels the worker scrambling when they're actively using it
    for other work; lower priority lets the OS scheduler cooperate.
    """
    try:
        import psutil
        p = psutil.Process(pid)
        if platform.system() == "Windows":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(10)
    except Exception:
        # Priority adjustment is a nice-to-have — don't fail the
        # whole spawn if it doesn't take.
        pass


# DLLs llama.cpp's rpc-server NEEDS to actually serve layer pushes.
# Detected as "missing backend" failures: rpc-server -d SYCL0,CPU
# without ggml-sycl.dll exits with "Failed to find RPC backend"
# regardless of what the user passes for `-d`.
_REQUIRED_RPC_DLLS = (
    "ggml-rpc.dll",     # the RPC backend itself
)
# DLLs needed for SYCL acceleration on Intel iGPU laptops. Optional
# — we'll spawn without them if missing, just won't get GPU-accelerated
# RPC. CPU still works.
_OPTIONAL_RPC_DLLS = (
    "ggml-sycl.dll",
)


def _missing_rpc_dlls() -> list[str]:
    """Return the list of REQUIRED DLLs that aren't yet installed
    next to rpc-server.exe. SYCL is optional — we don't gate the
    spawn on it (CPU-only still works) but the orchestrator will
    auto-install it too if it's pullable from a LAN peer.
    """
    out = []
    for name in _REQUIRED_RPC_DLLS:
        if not (_RPC_SERVER_BIN_DIR / name).is_file():
            out.append(name)
    return out


def _missing_optional_dlls() -> list[str]:
    """Optional DLLs (e.g. SYCL acceleration) we'd LIKE to have.
    Used to drive opportunistic LAN-first install — we don't fail
    the spawn when these are missing."""
    out = []
    for name in _OPTIONAL_RPC_DLLS:
        if not (_RPC_SERVER_BIN_DIR / name).is_file():
            out.append(name)
    return out


def _try_lan_fetch_dll(filename: str) -> bool:
    """Walk paired peers; ask each for ``/api/p2p/binary/list``;
    if a peer has ``filename`` with non-zero size, fetch it via
    direct LAN HTTP and drop into our llama-cpp dir.

    Returns True iff the file landed locally. Best-effort: returns
    False on any failure path. Synchronous because it's called
    from the spawn flow which is itself synchronous (blocks the
    rpc-start handler for a few seconds while we pull a 30 MB DLL).

    The fetch is plain HTTP — see the comment in
    ``p2p_secure_proxy._GIGACHAT_INTERNAL_PATHS`` block: the binary
    is a standard llama.cpp release artifact, not user data, so
    confidentiality isn't a concern. The destination is the
    paired peer's Gigachat backend on its LAN address, which is
    already trusted at the IP-allowlist level.
    """
    try:
        import httpx
        from . import db
    except ImportError:
        return False
    # Walk paired peers in last-seen order (freshest first — most
    # likely to be reachable). Skip ourselves (paired_devices doesn't
    # contain us anyway, but defensive).
    try:
        peers = db.list_paired_devices()
    except Exception:
        return False
    if not peers:
        return False
    peers.sort(key=lambda p: float(p.get("last_seen_at") or 0), reverse=True)
    target = _RPC_SERVER_BIN_DIR / filename
    # Atomic write — pull to .partial then rename, so a partial
    # download can't be mistaken for the real DLL on a retry.
    partial = _RPC_SERVER_BIN_DIR / f"{filename}.partial"
    for peer in peers:
        ip = peer.get("ip")
        port = int(peer.get("port") or 8000)
        if not ip:
            continue
        list_url = f"http://{ip}:{port}/api/p2p/binary/list"
        try:
            with httpx.Client(timeout=8.0) as client:
                r = client.get(list_url)
                if r.status_code != 200:
                    continue
                manifest = r.json()
                files = {f["name"]: f for f in (manifest.get("files") or [])}
                if filename not in files or not files[filename].get("size"):
                    continue
                expected_size = int(files[filename]["size"])
                # Fetch the body. Long timeout — multi-hundred-MB
                # DLLs over a 1 Gbps LAN take 4-8 seconds; over
                # Wi-Fi can take 30+ seconds.
                get_url = f"http://{ip}:{port}/api/p2p/binary/get/{filename}"
                with client.stream("GET", get_url, timeout=120.0) as resp:
                    if resp.status_code != 200:
                        continue
                    _RPC_SERVER_BIN_DIR.mkdir(parents=True, exist_ok=True)
                    with partial.open("wb") as out_f:
                        for chunk in resp.iter_bytes(chunk_size=1024 * 256):
                            out_f.write(chunk)
                if partial.stat().st_size != expected_size:
                    log.info(
                        "p2p_rpc_server: LAN fetch of %s from %s short-read "
                        "(%d != %d), discarding",
                        filename, ip,
                        partial.stat().st_size, expected_size,
                    )
                    try:
                        partial.unlink()
                    except OSError:
                        pass
                    continue
                # Success — atomic rename into final location.
                partial.replace(target)
                log.info(
                    "p2p_rpc_server: LAN-installed %s from %s (%.1f MB)",
                    filename, ip, expected_size / (1024 * 1024),
                )
                return True
        except Exception as e:
            log.debug(
                "p2p_rpc_server: LAN fetch of %s from %s failed: %s",
                filename, ip, e,
            )
            try:
                if partial.is_file():
                    partial.unlink()
            except OSError:
                pass
            continue
    return False


def auto_install_missing_dlls() -> dict:
    """Try to LAN-install every missing rpc-server dependency.
    Returns a summary so the spawn handler can include it in the
    response payload.

    Required DLLs (e.g. ``ggml-rpc.dll``) gate the spawn — if any
    can't be sourced, we abort. Optional DLLs (e.g. ``ggml-sycl.dll``)
    are best-effort: pulled if available, ignored if not.
    """
    summary: dict = {"installed": [], "still_missing": []}
    for name in _missing_rpc_dlls():
        if _try_lan_fetch_dll(name):
            summary["installed"].append(name)
        else:
            summary["still_missing"].append(name)
    for name in _missing_optional_dlls():
        if _try_lan_fetch_dll(name):
            summary["installed"].append(name)
        # Optional — silent on still-missing.
    return summary


def start_local_rpc_server(
    *, backend: str = _DEFAULT_BACKEND, port: int = _DEFAULT_PORT,
) -> dict:
    """Bring up rpc-server.exe locally (idempotent w.r.t. listener state).

    If port is already listening, we return the existing PID's status
    without restarting — saves the orchestrator from triggering a
    pointless tear-down + cold start every probe.

    On a clean spawn we kill any stale rpc-server processes first
    (the "two rpc-servers fighting for port 50052" failure mode is
    a guaranteed source of intermittent split start failures), then
    spawn fresh.

    Auto-install: before spawning, we check that the DLLs llama.cpp's
    rpc-server actually needs are present. If any are missing, we
    try to pull them from a paired LAN peer that has them — turning
    a hard failure ("Failed to find RPC backend" → split never
    engages) into a transparent self-heal that adds 5-10 s on the
    first call and 0 s thereafter.
    """
    # Declared at the top of the function — Python's `global` rule
    # requires it BEFORE any read of the same name. We reset on
    # restart paths and stamp on successful spawn below.
    global _active_backend
    out: dict = {
        "binary_path": str(_RPC_SERVER_EXE),
        "backend": backend,
        "port": port,
    }
    # Already up with the SAME backend? Don't rebuild — saves the
    # orchestrator from churn. If the desired backend differs from
    # what's currently running, fall through to the kill+respawn
    # path below so the worker actually serves what the caller
    # asked for.
    if _is_listening_on(port) and _active_backend == backend:
        out["status"] = "already_running"
        out["listening"] = True
        out["active_backend"] = _active_backend
        return out

    if not _RPC_SERVER_EXE.is_file():
        out["status"] = "no_binary"
        out["listening"] = False
        return out

    # LAN-first auto-install of missing DLLs. Cheap when nothing
    # needs installing (a few stat() calls); pulls files in tens
    # of seconds when something IS missing. Either way we don't
    # block the spawn if optional DLLs can't be sourced.
    install_summary = auto_install_missing_dlls()
    if install_summary["installed"] or install_summary["still_missing"]:
        out["lan_install"] = install_summary
    if install_summary["still_missing"]:
        # Required DLLs we couldn't source on LAN. Spawning will
        # fail; surface the gap clearly so the orchestrator can
        # tell the user to install llama.cpp manually on this peer.
        out["status"] = "missing_dlls"
        out["listening"] = False
        return out

    killed = _kill_running_rpc_servers()
    out["killed_stale"] = killed

    cmd = [
        str(_RPC_SERVER_EXE),
        "-H", "0.0.0.0",
        "-p", str(port),
        "-d", backend,
    ]
    env = dict(os.environ)
    env.update(_RPC_SPAWN_ENV)
    try:
        pid = _spawn_detached(cmd, cwd=str(_RPC_SERVER_BIN_DIR), env=env)
    except Exception as e:
        out["status"] = "spawn_failed"
        out["error"] = f"{type(e).__name__}: {e}"
        out["listening"] = False
        return out

    out["pid"] = pid

    # Wait for the listener — same 4 s budget as the SSH path. Poll
    # at 250 ms cadence so a fast-start is reflected immediately and
    # we don't sleep the full budget on success.
    deadline = time.time() + _LISTEN_WAIT_SEC
    listening = False
    while time.time() < deadline:
        if _is_listening_on(port):
            listening = True
            break
        time.sleep(0.25)

    out["listening"] = listening
    if listening:
        _try_lower_priority(pid)
        out["status"] = "started"
        # Stamp the backend so subsequent calls can short-circuit
        # the restart when the request matches what's running.
        # `global` was declared at the top of the function.
        _active_backend = backend
        out["active_backend"] = backend
    else:
        out["status"] = "spawned_but_not_listening"
    return out


def stop_local_rpc_server() -> dict:
    """Kill every rpc-server process this user owns. Returns count.

    Used by `stop_all_running_splits` cleanup paths and by tests
    that want a known-clean state.
    """
    killed = _kill_running_rpc_servers()
    global _active_backend
    _active_backend = None
    return {"killed": killed, "listening": _is_listening_on(_DEFAULT_PORT)}


def get_local_rpc_server_status(port: int = _DEFAULT_PORT) -> dict:
    """Snapshot of the local rpc-server's state for the orchestrator
    to query before deciding whether to spawn.

    Includes ``active_backend`` so the orchestrator can compare it
    to what it WANTS the worker on (e.g. CPU-only for split safety
    vs SYCL+CPU for non-split paths) and decide whether to request
    a restart with a different backend.
    """
    return {
        "binary_present": _RPC_SERVER_EXE.is_file(),
        "binary_path": str(_RPC_SERVER_EXE),
        "listening": _is_listening_on(port),
        "port": port,
        "active_backend": _active_backend,
    }
