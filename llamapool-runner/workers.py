"""Worker control: probe rpc-server, restart via SSH+WMI, pick
backend per GPU vendor.

Every helper here takes a worker dict from config.list_workers()
and acts on it. No database; no cross-worker state — workers are
addressed by their `label` + `address` + `rpc_port` from the JSON
config.

The SSH+WMI restart is Windows-target-specific (the pool's typical
deployment pattern: a Windows host driving Windows laptops via SSH).
For a Linux/macOS worker, the rpc-server lifecycle is up to the
operator — this module's `restart_rpc_server` is best-effort and
fails cleanly when ssh_host isn't set or the worker is non-Windows.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import socket
import subprocess
from typing import Any

log = logging.getLogger(__name__)

# Default backend to pick per vendor. Intel iGPU + RPC has an
# upstream llama.cpp bug (#21420 / #20259 / #21474) that crashes
# the rpc-server when tensors are pushed over RPC to a SYCL
# backend. We work around it by NOT exposing SYCL during split:
# Intel workers run `-d CPU` only (CPU device for system RAM as
# layer storage). NVIDIA / AMD workers don't have this bug.
_BACKEND_FOR_VENDOR_IN_SPLIT = {
    "nvidia": "CUDA0,CPU",
    "amd": "Vulkan0,CPU",
    "intel": "CPU",       # workaround for SYCL+RPC bug
    "none": "CPU",
}
_BACKEND_FOR_VENDOR_IDLE = {
    "nvidia": "CUDA0,CPU",
    "amd": "Vulkan0,CPU",
    "intel": "SYCL0,CPU",  # full Intel iGPU acceleration when not split
    "none": "CPU",
}


def detect_gpu_vendor(worker: dict[str, Any]) -> str:
    """Pick the dominant GPU vendor for this worker.

    Reads `worker.gpu_vendor` if explicitly configured (skips probing).
    Otherwise returns "unknown" — for auto-detection, consider running
    `llamapool probe-worker LABEL` which SSHes in to inspect
    Win32_VideoController and persists the result.
    """
    return (worker.get("gpu_vendor") or "unknown").lower()


def select_backend(worker: dict[str, Any], *, in_split: bool) -> str:
    """Pick the right `-d` flag for this worker given its hardware
    AND the current routing mode (split-engaged vs idle)."""
    vendor = detect_gpu_vendor(worker)
    table = _BACKEND_FOR_VENDOR_IN_SPLIT if in_split else _BACKEND_FOR_VENDOR_IDLE
    return table.get(vendor, "CPU")


def is_rpc_reachable(worker: dict[str, Any], timeout: float = 3.0) -> bool:
    """TCP-probe the rpc-server port. Returns True iff reachable."""
    host = worker.get("address") or ""
    port = int(worker.get("rpc_port") or 50052)
    if not host:
        return False
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def _build_restart_powershell(backend: str) -> str:
    """Build the PowerShell payload that kills any stale rpc-server,
    spawns a fresh one with `-d <backend>`, lowers its priority for
    cooperative resource use, and confirms the port is listening."""
    return (
        "$ErrorActionPreference = 'Continue';"
        "Get-Process -Name 'rpc-server' -ErrorAction SilentlyContinue | "
        "  ForEach-Object { Stop-Process -Id $_.Id -Force };"
        "Start-Sleep -Milliseconds 800;"
        # Intel SYCL stability env vars — harmless on non-Intel hosts.
        "[Environment]::SetEnvironmentVariable('GGML_SYCL_DISABLE_OPT', '1', 'User');"
        "[Environment]::SetEnvironmentVariable('GGML_SYCL_DISABLE_GRAPH', '1', 'User');"
        "[Environment]::SetEnvironmentVariable('SYCL_CACHE_PERSISTENT', '1', 'User');"
        "$exe = \"$env:USERPROFILE\\.llamapool\\llama-cpp\\rpc-server.exe\";"
        "if (-not (Test-Path $exe)) {"
        "  $exe = \"$env:USERPROFILE\\.gigachat\\llama-cpp\\rpc-server.exe\";"
        "}"
        "if (-not (Test-Path $exe)) { Write-Output 'NO_BINARY'; exit 2 };"
        f"$cmdline = '\"' + $exe + '\" -H 0.0.0.0 -p 50052 -d {backend}';"
        "$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
        "  -Arguments @{ CommandLine = $cmdline; "
        "                CurrentDirectory = (Split-Path $exe) };"
        "if ($result.ReturnValue -ne 0) { Write-Output 'WMI_FAIL'; exit 3 };"
        "Start-Sleep -Seconds 4;"
        "$rpc = Get-Process -Name 'rpc-server' -ErrorAction SilentlyContinue;"
        "if ($rpc) {"
        "  try { $rpc.PriorityClass = "
        "        [System.Diagnostics.ProcessPriorityClass]::BelowNormal } catch {};"
        "}"
        "$port = Get-NetTCPConnection -LocalPort 50052 -State Listen "
        "  -ErrorAction SilentlyContinue;"
        "if ($rpc -and $port) { Write-Output 'OK' } else { Write-Output 'NO_LISTEN' }"
    )


async def restart_rpc_server(
    worker: dict[str, Any], *, backend: str = "SYCL0,CPU",
) -> bool:
    """SSH into the worker and re-spawn its rpc-server with the given
    `-d` backend flag.

    Returns True if rpc-server is alive + listening after the spawn.
    Failure modes (all return False):
      * `ssh_host` not configured for this worker
      * SSH connect timeout / auth failure
      * rpc-server.exe not found on the worker
      * Process spawned but port not listening within 4 s
    """
    ssh_host = (worker.get("ssh_host") or "").strip()
    if not ssh_host:
        return False
    ps = _build_restart_powershell(backend)
    encoded = base64.b64encode(ps.encode("utf-16-le")).decode("ascii")
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
           ssh_host, "powershell", "-NoProfile", "-EncodedCommand", encoded]

    def _run() -> tuple[int, bytes, bytes]:
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=30.0)
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return -1, b"", b"timeout"
        except Exception as e:
            return -2, b"", repr(e).encode()

    try:
        rc, stdout, stderr = await asyncio.to_thread(_run)
    except Exception as e:
        log.info("rpc-server restart ssh failed for %s: %s", ssh_host, e)
        return False
    out = stdout.decode("utf-8", errors="replace").strip()
    if rc != 0 or "OK" not in out:
        log.info(
            "rpc-server restart on %s did not come up "
            "(rc=%d, output=%r)", ssh_host, rc, out[-200:],
        )
        return False
    return True


async def set_workers_backend(
    workers: list[dict[str, Any]], *, in_split: bool,
) -> int:
    """Ensure every worker is running its rpc-server with the right
    backend for the current mode. Restarts mismatched workers.

    Tracks current backend in `worker.current_rpc_backend` so we don't
    bounce rpc-server unnecessarily on every spawn. The caller is
    responsible for persisting the updated worker dict (we mutate
    in place; saving back to JSON config is the caller's choice).
    """
    aligned = 0
    for w in workers:
        if not (w.get("ssh_host") or "").strip():
            continue
        backend = select_backend(w, in_split=in_split)
        if w.get("current_rpc_backend") == backend:
            aligned += 1
            continue
        ok = await restart_rpc_server(w, backend=backend)
        if ok:
            w["current_rpc_backend"] = backend
            aligned += 1
            log.info(
                "worker %s switched rpc-server backend -> %s",
                w.get("label"), backend,
            )
    return aligned


def resolve_rpc_endpoints(workers: list[dict[str, Any]]) -> list[str]:
    """Build the `<host>:<port>` list to feed to llama-server's
    `--rpc` flag. Skips workers that are disabled or unreachable."""
    out: list[str] = []
    for w in workers:
        if not w.get("enabled", True):
            continue
        if not is_rpc_reachable(w):
            continue
        host = (w.get("address") or "").strip()
        port = int(w.get("rpc_port") or 50052)
        if host:
            out.append(f"{host}:{port}")
    return out
