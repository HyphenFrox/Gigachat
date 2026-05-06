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

# Per-port active backend map. Replaces the old singleton so the
# orchestrator can ask the worker to run MULTIPLE rpc-servers
# concurrently — typically one bound to `-d SYCL0` (iGPU only) on
# 50052 and one bound to `-d CPU` (CPU+RAM only) on 50053. Each
# rpc-server has a single backend internally so the hybrid-allocator
# layout-mismatch crash (ggml-rpc.cpp's RPC_STATUS_ASSERT) is bypassed,
# AND the worker contributes BOTH its iGPU memory AND its system RAM
# as separate compute targets to the orchestrator's --tensor-split.
_active_backends: dict[int, str] = {}

# Backwards-compatible singleton. Mirrors `_active_backends.get(
# _DEFAULT_PORT)`. Code paths that only know about the default port
# (legacy callers, single-port status endpoint) keep working.
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


def _kill_running_rpc_servers(only_listening_on_port: int | None = None) -> int:
    """Best-effort kill of every existing rpc-server process owned by
    this user. Returns the count killed.

    `only_listening_on_port`: when set, only kill rpc-server processes
    whose listening sockets include that port. Used by the multi-port
    spawn path so kill of port 50052 doesn't take down an unrelated
    rpc-server bound to 50053. Default behaviour (None) kills every
    rpc-server we own — matches the legacy single-port assumption.

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
            # Per-port scoping: skip processes whose listener doesn't
            # include the requested port. Each rpc-server binds one
            # TCP port, so `psutil.Process(pid).net_connections()`
            # tells us which port it's holding.
            if only_listening_on_port is not None:
                try:
                    proc = psutil.Process(p.info["pid"])
                    ports = {
                        c.laddr.port for c in proc.net_connections(kind="inet")
                        if c.status == psutil.CONN_LISTEN
                    }
                    if only_listening_on_port not in ports:
                        continue
                except (psutil.NoSuchProcess, psutil.AccessDenied):
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
    the spawn when these are missing.

    Skip-marker support: a file ``<dll>.skip-install`` next to the
    install dir marks that backend DLL as deliberately disabled —
    we should NOT try to fetch it again. This is the escape hatch
    used by the host-orchestrator path to keep host's llama-server
    from loading a SYCL backend that triggers the SYCL_Split crash
    (ggml-backend.cpp:898) when host has an Intel iGPU AND there's
    at least one RPC peer in the split. Without this guard, every
    time the host's compute_pool calls ``start_local_rpc_server``
    (e.g. for the peer-orchestrated split path) the auto-fetcher
    silently re-downloads ggml-sycl.dll from a LAN peer and the
    next split spawn crashes again.
    """
    out = []
    for name in _OPTIONAL_RPC_DLLS:
        if (_RPC_SERVER_BIN_DIR / name).is_file():
            continue
        if (_RPC_SERVER_BIN_DIR / f"{name}.skip-install").is_file():
            continue
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


def _local_has_intel_igpu() -> bool:
    """Cheap self-check: does THIS machine have an Intel iGPU?

    Used to decide whether shipping ``ggml-sycl.dll`` is safe. On a
    no-iGPU machine (NVIDIA-only desktop, AMD laptop, Apple silicon),
    the SYCL backend tries to enumerate Intel level_zero devices,
    finds zero, and SILENTLY ABORTS the rpc-server process before it
    binds the port — leaving us with a "spawn succeeded but never
    listened" failure that the orchestrator surfaces as
    ``rpc-server probe: timeout`` with no log trail. The user-visible
    symptom: FBS-style dGPU peers can never join the split pool.

    The fix is to disable the SYCL DLL on these peers so the
    rpc-server doesn't try to load it. Detection has to be
    process-local — we can't ask the orchestrator because it's the
    one calling US to bring up the rpc-server in the first place.
    """
    try:
        from . import sysdetect
        spec = sysdetect.detect_system()
        return (spec.get("gpu_kind") or "").lower() == "intel"
    except Exception:
        return False


def _quarantine_sycl_dll_if_no_igpu() -> bool:
    """Rename ``ggml-sycl.dll`` to ``ggml-sycl.dll.skip-install`` when
    this machine has no Intel iGPU.

    Idempotent + symmetric: also UN-quarantines (renames the marker
    back) when the machine DOES have an iGPU but the DLL is currently
    sidelined — handles the case where a user adds an Intel GPU later
    or moves the install to a different machine.

    Returns True if any rename happened (caller can log it).
    """
    sycl_dll = _RPC_SERVER_BIN_DIR / "ggml-sycl.dll"
    skip_marker = _RPC_SERVER_BIN_DIR / "ggml-sycl.dll.skip-install"
    no_igpu_marker = _RPC_SERVER_BIN_DIR / "ggml-sycl.dll.removed-no-igpu"
    has_igpu = _local_has_intel_igpu()
    changed = False
    if not has_igpu:
        # Move active DLL OUT of the way so the rpc-server doesn't
        # try to dlopen it during backend enumeration.
        if sycl_dll.is_file():
            try:
                if no_igpu_marker.is_file():
                    no_igpu_marker.unlink()
                sycl_dll.rename(no_igpu_marker)
                log.info(
                    "p2p_rpc_server: no Intel iGPU detected — quarantined "
                    "ggml-sycl.dll → %s so rpc-server can spawn without "
                    "the silent-abort that hits no-iGPU peers",
                    no_igpu_marker.name,
                )
                changed = True
            except OSError as e:
                log.warning(
                    "p2p_rpc_server: failed to quarantine ggml-sycl.dll: %s",
                    e,
                )
        # ALSO drop the skip-install marker so the LAN auto-fetcher
        # doesn't silently re-pull the DLL on the next spawn.
        if not skip_marker.is_file():
            try:
                skip_marker.write_bytes(b"no Intel iGPU on this machine\n")
            except OSError:
                pass
    else:
        # Has iGPU now — restore from the no-iGPU marker if we'd
        # previously sidelined it. Don't touch a user-placed
        # ``.skip-install`` (that's the host-orchestrator's
        # explicit "stay disabled" knob).
        if no_igpu_marker.is_file() and not sycl_dll.is_file():
            try:
                no_igpu_marker.rename(sycl_dll)
                log.info(
                    "p2p_rpc_server: Intel iGPU now present — restored "
                    "ggml-sycl.dll from %s", no_igpu_marker.name,
                )
                changed = True
            except OSError as e:
                log.warning(
                    "p2p_rpc_server: failed to restore ggml-sycl.dll: %s",
                    e,
                )
    return changed


def auto_install_missing_dlls() -> dict:
    """Try to LAN-install every missing rpc-server dependency.
    Returns a summary so the spawn handler can include it in the
    response payload.

    Required DLLs (e.g. ``ggml-rpc.dll``) gate the spawn — if any
    can't be sourced, we abort. Optional DLLs (e.g. ``ggml-sycl.dll``)
    are best-effort: pulled if available, ignored if not.

    Pre-step: quarantine ``ggml-sycl.dll`` on no-iGPU peers so the
    rpc-server doesn't silently abort during SYCL backend
    enumeration. This is the difference between FBS-style NVIDIA
    boxes joining the split pool versus reporting
    ``rpc-server probe: timeout`` forever.
    """
    _quarantine_sycl_dll_if_no_igpu()
    summary: dict = {"installed": [], "still_missing": []}
    for name in _missing_rpc_dlls():
        if _try_lan_fetch_dll(name):
            summary["installed"].append(name)
        else:
            summary["still_missing"].append(name)
    for name in _missing_optional_dlls():
        # Don't auto-fetch SYCL DLL on a no-iGPU machine — would
        # re-create exactly the silent-abort condition we just
        # quarantined.
        if name == "ggml-sycl.dll" and not _local_has_intel_igpu():
            continue
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
    # Already up with the SAME backend on this port? Don't rebuild —
    # saves the orchestrator from churn. If the desired backend
    # differs from what's currently running ON THIS PORT, fall
    # through to the kill+respawn path below so the worker actually
    # serves what the caller asked for. Other ports (e.g. a sibling
    # rpc-server on 50053 with a different backend) are untouched.
    current_on_port = _active_backends.get(port)
    if _is_listening_on(port) and current_on_port == backend:
        out["status"] = "already_running"
        out["listening"] = True
        out["active_backend"] = current_on_port
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

    # Kill ONLY the rpc-server bound to THIS port, leaving sibling
    # rpc-servers on other ports running. Critical for the multi-
    # backend deployment: restarting the SYCL rpc-server on 50052
    # must not take down the CPU rpc-server on 50053.
    killed = _kill_running_rpc_servers(only_listening_on_port=port)
    out["killed_stale"] = killed
    # Drop the stale entry from the active map (the new spawn below
    # will re-stamp on success).
    _active_backends.pop(port, None)

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
        # Stamp BOTH the per-port map (multi-rpc world) and the
        # legacy singleton (port == _DEFAULT_PORT only) so old
        # readers keep working. `global` was declared at the top.
        _active_backends[port] = backend
        if port == _DEFAULT_PORT:
            _active_backend = backend
        out["active_backend"] = backend
    else:
        out["status"] = "spawned_but_not_listening"
    return out


def ensure_local_rpc_servers(specs: list[dict]) -> dict:
    """Ensure a SET of rpc-servers are running, one per (port, backend).

    `specs` is a list like::

        [{"port": 50052, "backend": "SYCL0"},
         {"port": 50053, "backend": "CPU"}]

    Spawns / restarts each as needed using `start_local_rpc_server`,
    which is idempotent per port. Returns a dict mapping port to the
    per-port spawn result so the caller can see which succeeded.

    This is the entry point the orchestrator calls when it wants the
    worker to expose BOTH iGPU and CPU as separate compute targets.
    Each rpc-server has a single backend internally — the prior
    SYCL+CPU hybrid layout-mismatch bug doesn't apply to either —
    and the orchestrator's `--rpc` lists both endpoints so layers
    can be placed on either device with correct `--tensor-split`
    weight per ENDPOINT.
    """
    results: dict[int, dict] = {}
    for spec in specs or []:
        try:
            port = int(spec.get("port") or _DEFAULT_PORT)
            backend = (spec.get("backend") or _DEFAULT_BACKEND).strip()
        except (TypeError, ValueError):
            continue
        if not backend:
            continue
        results[port] = start_local_rpc_server(backend=backend, port=port)
    return {
        "specs": specs,
        "results": results,
        "active_backends": dict(_active_backends),
    }


def stop_local_rpc_server(port: int | None = None) -> dict:
    """Kill rpc-server process(es) this user owns. Returns count.

    `port`: when set, only kill the rpc-server bound to that port
    (sibling rpc-servers on other ports keep running). When None,
    kills every rpc-server we own — matches the legacy single-port
    assumption used by full-shutdown / test-reset paths.
    """
    global _active_backend
    if port is not None:
        killed = _kill_running_rpc_servers(only_listening_on_port=port)
        _active_backends.pop(port, None)
        if port == _DEFAULT_PORT:
            _active_backend = None
        return {"killed": killed, "port": port,
                "listening": _is_listening_on(port)}
    killed = _kill_running_rpc_servers()
    _active_backends.clear()
    _active_backend = None
    return {"killed": killed, "listening": _is_listening_on(_DEFAULT_PORT)}


def get_local_rpc_server_status(port: int = _DEFAULT_PORT) -> dict:
    """Snapshot of the local rpc-server's state for the orchestrator
    to query before deciding whether to spawn.

    Includes:
      * ``active_backend`` for the requested port (legacy single-port
        callers keep their existing API contract)
      * ``active_backends`` — the FULL per-port map so multi-rpc
        callers can ask "what's running on every port?" in one call.
    """
    return {
        "binary_present": _RPC_SERVER_EXE.is_file(),
        "binary_path": str(_RPC_SERVER_EXE),
        "listening": _is_listening_on(port),
        "port": port,
        "active_backend": _active_backends.get(port),
        "active_backends": dict(_active_backends),
    }


# ---------------------------------------------------------------------------
# Supervisor — keep rpc-servers up FOREVER once Gigachat is running.
#
# Without this, rpc-servers only spawn at split-time (host calls
# `_set_workers_backend` → `ensure_rpc_servers_via_proxy_multi`). Between
# splits — or if a peer reboots while no split is active — the rpc-server
# is simply not running, and the next split-time spawn pays the cold-start
# tax (5–30 s of SYCL JIT / CUDA init / weight upload) AND, on the
# no-iGPU peer bug we just fixed, may fail outright.
#
# The supervisor:
#   1. Picks an OPTIMAL backend set for THIS machine based on local
#      hardware (Intel iGPU → SYCL0+CPU, NVIDIA → CUDA0+CPU,
#      AMD → Vulkan0+CPU, generic CPU → CPU only).
#   2. At startup, calls `ensure_local_rpc_servers(specs)` so the
#      rpc-servers are listening immediately.
#   3. Polls every 30 s; if any port no longer listens, respawns it.
#   4. Re-runs the SYCL DLL quarantine on every tick so a hardware
#      change (user adds an Intel GPU later, or moves the install to
#      a different box) self-heals without a Gigachat restart.
#
# This implements the user's "once we run the app it should stay up
# forever" expectation. ~minimal CPU cost (one TCP self-connect per
# port per 30 s).
# ---------------------------------------------------------------------------

_SUPERVISOR_TICK_SEC = 30.0
_supervisor_task: "object | None" = None


def _decide_local_rpc_specs() -> list[dict]:
    """Pick the (port, backend) tuples this machine should expose
    based on its hardware. One iGPU/dGPU endpoint + one CPU endpoint
    on machines with a GPU; CPU-only on machines without one.

    Port assignments match what the orchestrator expects via
    ``capabilities.rpc_endpoints``: 50052 for the GPU backend,
    50053 for the CPU backend. Single-backend machines just expose
    50052 with the appropriate `-d` so legacy single-port routing
    keeps working.

    GPU detection: tries `sysdetect.detect_system()` first, but
    that helper is `@lru_cache`'d at the module level — if its
    Intel WMI probe was called during early boot (before WMI was
    fully responsive) and returned ``gpu_kind=""``, the cached
    miss sticks for the whole process lifetime. So we additionally
    consult the resource_tracker BG sampler, which uses a
    re-attempted PowerShell query and is much more reliable on
    Intel iGPU laptops post-boot. Fall back to CPU only when both
    say "no GPU".
    """
    gpu_kind = ""
    try:
        from . import sysdetect
        spec = sysdetect.detect_system()
        gpu_kind = (spec.get("gpu_kind") or "").lower()
    except Exception:
        gpu_kind = ""
    # Cross-check against the BG sampler's live GPU probe. This
    # catches the post-boot case where Intel WMI returned 0 in
    # `sysdetect.detect_system()` during the first call (cached).
    if not gpu_kind:
        try:
            from . import resource_tracker as _rt
            gpu_snap = _rt._BgSampler.get().gpu_snap or {}
            gpu_kind = (gpu_snap.get("gpu_kind") or "").lower()
        except Exception:
            pass
    if gpu_kind == "intel":
        return [
            {"port": 50052, "backend": "SYCL0"},
            {"port": 50053, "backend": "CPU"},
        ]
    if gpu_kind == "nvidia":
        return [
            {"port": 50052, "backend": "CUDA0"},
            {"port": 50053, "backend": "CPU"},
        ]
    if gpu_kind == "amd":
        # ROCm/Vulkan path — most consumer AMDs land on Vulkan via
        # llama.cpp's auto-detection; we expose Vulkan0 and CPU.
        return [
            {"port": 50052, "backend": "Vulkan0"},
            {"port": 50053, "backend": "CPU"},
        ]
    if gpu_kind == "apple":
        # Apple silicon — Metal handled by ggml-metal; the rpc-server
        # binary on Apple uses Metal as its single backend.
        return [{"port": 50052, "backend": "Metal"}]
    # CPU-only machine — one CPU endpoint is plenty.
    return [{"port": 50052, "backend": "CPU"}]


def _supervisor_tick() -> None:
    """One pass of the supervisor: ensure every desired rpc-server is
    listening on its assigned port. Idempotent — no-op when nothing
    needs to change.
    """
    if not _RPC_SERVER_EXE.is_file():
        # rpc-server not installed yet — let the auto-installer pick
        # it up on the next user-driven action. We don't try to
        # bootstrap the binary itself here; that's the install.bat
        # path, not the supervisor's job.
        return
    # Run the no-iGPU quarantine first — cheap stat-only check that's
    # essential on hardware-change boundaries.
    try:
        _quarantine_sycl_dll_if_no_igpu()
    except Exception as e:
        log.debug("p2p_rpc_server supervisor: quarantine check failed: %s", e)
    specs = _decide_local_rpc_specs()
    for spec in specs:
        port = int(spec.get("port") or _DEFAULT_PORT)
        backend = (spec.get("backend") or _DEFAULT_BACKEND).strip()
        already_listening = _is_listening_on(port)
        recorded_backend = _active_backends.get(port)
        # Already up with the right backend — fast skip.
        if already_listening and recorded_backend == backend:
            continue
        # Port is listening but we don't know the backend (typical when
        # the supervisor's first tick runs after a Gigachat backend
        # restart — the rpc-server child survived the parent restart
        # because it was detached, but `_active_backends` is a fresh
        # in-process map). Adopt the listener as ours rather than
        # kill+respawn, which on Windows triggers a 5-12 s SYCL JIT
        # cycle for every supervisor tick and floods asyncio with
        # ConnectionResetError as in-flight LAN connections from peers
        # get torn down. The orchestrator's
        # `ensure_rpc_servers_via_proxy_multi` will trigger a real
        # restart whenever it actually needs a different backend.
        if already_listening and recorded_backend is None:
            _active_backends[port] = backend
            global _active_backend
            if port == _DEFAULT_PORT:
                _active_backend = backend
            log.info(
                "p2p_rpc_server supervisor: adopting listener on :%d as %s "
                "(spawned by previous instance, no kill+respawn churn)",
                port, backend,
            )
            continue
        try:
            result = start_local_rpc_server(backend=backend, port=port)
            if result.get("listening"):
                log.info(
                    "p2p_rpc_server supervisor: (re)started %s on :%d",
                    backend, port,
                )
            else:
                log.warning(
                    "p2p_rpc_server supervisor: failed to bring up %s on "
                    ":%d — %s", backend, port, result.get("status"),
                )
        except Exception as e:
            log.warning(
                "p2p_rpc_server supervisor: spawn raised on %s:%d: %s",
                backend, port, e,
            )


async def start_supervisor() -> None:
    """Schedule the supervisor as a background asyncio task.

    Idempotent — calling twice is safe (the second call is a no-op).
    First tick fires immediately so rpc-servers come up at peer boot
    BEFORE the first chat / split request arrives. Subsequent ticks
    every ``_SUPERVISOR_TICK_SEC`` seconds.
    """
    import asyncio
    global _supervisor_task
    if _supervisor_task is not None:
        return

    async def _loop() -> None:
        # First tick: synchronous (no sleep) so rpc-servers are up
        # before we yield back to the lifespan startup chain.
        try:
            await asyncio.to_thread(_supervisor_tick)
        except Exception as e:
            log.warning("p2p_rpc_server supervisor: first tick failed: %s", e)
        while True:
            try:
                await asyncio.sleep(_SUPERVISOR_TICK_SEC)
            except asyncio.CancelledError:
                return
            try:
                await asyncio.to_thread(_supervisor_tick)
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.warning(
                    "p2p_rpc_server supervisor: tick failed: %s", e,
                )

    _supervisor_task = asyncio.create_task(
        _loop(), name="p2p_rpc_server_supervisor",
    )
    log.info(
        "p2p_rpc_server: supervisor started (tick=%ds) — keeps rpc-servers "
        "up forever once the app is running",
        int(_SUPERVISOR_TICK_SEC),
    )


async def stop_supervisor() -> None:
    """Cancel the supervisor task. Called from app shutdown."""
    import asyncio
    global _supervisor_task
    if _supervisor_task is None:
        return
    _supervisor_task.cancel()
    try:
        await _supervisor_task
    except (asyncio.CancelledError, Exception):
        pass
    _supervisor_task = None
