"""Compute pool: capability probe + liveness sweep for registered workers.

The pool's data layer (CRUD on `compute_workers`) lives in `db.py`. This
module owns the operational side: ping each worker's Ollama, cache
what's installed there, and surface failures so the Settings UI and
the routing layer can grey out unreachable nodes.

Two entry points:
  * `probe_worker(wid)` — one-shot probe; used by the manual "Test
    connection" button in the UI and by the routing layer when it
    wants to confirm a worker is hot before sending real traffic.
  * `start_periodic_probe()` — background asyncio task scheduled on
    app startup. Sweeps every `_SWEEP_INTERVAL_SEC` (5 min by default)
    so capability and liveness data stay fresh even when no one is
    actively poking the Settings panel.

The probe is intentionally cheap: two parallel GETs (`/api/version`
and `/api/tags`) with a short timeout. If either fails we record the
error string on the worker row and reschedule for next sweep — no
exponential backoff yet because workers are typically on a stable LAN
or tailnet, not flaky public networks. Add backoff if real-world
flakiness appears.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
from pathlib import Path
from typing import Any

import httpx

from . import db, sysdetect

log = logging.getLogger(__name__)


# Sweep cadence. 5 min is a balance between "Settings UI shows
# something fresh when the user opens it" and "don't burn the worker's
# bandwidth on a stationary background poll." The sweep is cheap (two
# tiny GETs per worker) so this could go faster, but most workers'
# state changes — model pulls, hardware swaps — happen rarely.
_SWEEP_INTERVAL_SEC = 300

# Per-probe timeout. The worker is on the same LAN or tailnet; a
# multi-second response means it's overloaded or down, not slow.
_PROBE_TIMEOUT_SEC = 5.0

# Default TCP port `rpc-server` (llama.cpp's worker process) listens on.
# This is the upstream's default; users can override per-worker once the
# `rpc_port` schema column lands. Phase 2 commit 3 just probes this port
# to surface whether rpc-server is running on each worker.
_DEFAULT_RPC_PORT = 50052

# How long to wait for the TCP handshake on the rpc-server probe.
# Slightly larger than you'd expect for a SYN-ACK because mDNS
# hostnames often resolve to multiple addresses (IPv4 + IPv6 link-
# local) and asyncio's `open_connection` tries them in turn — the
# IPv6 link-local hop can take several seconds to fail before the
# IPv4 fallback connects. 5 seconds covers that.
_RPC_PROBE_TIMEOUT_SEC = 5.0

# How long a measured `tokens_per_second` is good for before we
# re-benchmark. 1 hour is the right balance: long enough that the
# bench cost amortizes (loading + 30 tokens of generation) but
# short enough that hardware changes (different model loaded on
# host changing GPU contention, worker reboot, etc.) get reflected
# within a session.
_THROUGHPUT_CACHE_TTL_SEC = 3600.0

# Number of tokens to generate during the throughput benchmark.
# 20 is enough to amortize first-token latency without dragging
# probe time too long. On a typical 7B-Q4 model on host VRAM
# this completes in <1 s; on a laptop CPU it might take 5-10 s.
_THROUGHPUT_BENCH_TOKENS = 20

# Cached host throughput, keyed by model name. Measured lazily on
# first route decision and refreshed every TTL.
_HOST_THROUGHPUT_CACHE: dict[str, tuple[float, float]] = {}  # model -> (tps, measured_at)

# Subagent fan-out performance gate. A worker is included in the
# parallel-subagent target list only if its measured TPS is at least
# this fraction of the host's measured TPS for the same model.
# Rationale: round-robin fan-out gets bottlenecked by the slowest
# task; a worker dramatically slower than host slows down the whole
# fan-out instead of accelerating it. 0.25 = "must be at least
# one-quarter as fast as host." Tunable; chosen so a weak laptop
# doesn't degrade `delegate_parallel` on a fast host.
_SUBAGENT_MIN_PERF_RATIO = 0.25


def _worker_base_url(worker: dict) -> str:
    """Build the Ollama base URL for a worker row."""
    addr = (worker.get("address") or "").strip()
    port = int(worker.get("ollama_port") or 11434)
    # Strip any trailing slashes / scheme the user may have pasted in.
    if addr.startswith("http://"):
        addr = addr[len("http://"):]
    elif addr.startswith("https://"):
        addr = addr[len("https://"):]
    addr = addr.rstrip("/")
    return f"http://{addr}:{port}"


def _worker_host(worker: dict) -> str:
    """Bare hostname / IP for non-HTTP probes (rpc-server uses TCP, not HTTP).

    Same scheme stripping as `_worker_base_url` but returns just the host
    portion — `rpc-server` listens on its own port, not the Ollama port.
    """
    addr = (worker.get("address") or "").strip()
    if addr.startswith("http://"):
        addr = addr[len("http://"):]
    elif addr.startswith("https://"):
        addr = addr[len("https://"):]
    return addr.rstrip("/")


async def _measure_throughput(
    base_url: str,
    auth_token: str | None,
    model_name: str,
    n_tokens: int = _THROUGHPUT_BENCH_TOKENS,
) -> tuple[float, float]:
    """Run a small Ollama generation, return (tokens_per_second, total_seconds).

    Folds CPU, RAM bandwidth, GPU compute, and memory size into one
    bottom-line number — the real speed signal for routing. We measure
    rate AFTER the first token to skip cold-load time (model file
    paging from disk into VRAM/RAM dominates the first hundred ms);
    what callers actually want is "how fast does this machine generate
    tokens once it's hot."

    Returns (0, 0) on any failure — caller can fall back to the static
    heuristic ranking when the bench couldn't run (model missing,
    Ollama down, etc.).
    """
    headers: dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    payload = {
        "model": model_name,
        "prompt": "Hello",
        "stream": True,
        "options": {"num_predict": int(n_tokens), "temperature": 0.0},
    }
    timeout = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=10.0)
    first_token_at: float | None = None
    tok = 0
    t_total_start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            async with c.stream("POST", f"{base_url}/api/generate", json=payload, headers=headers) as r:
                if r.status_code >= 400:
                    return 0.0, 0.0
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("response"):
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        tok += 1
                    if obj.get("done"):
                        break
    except Exception as e:
        log.info("throughput bench failed for %s on %s: %s", model_name, base_url, e)
        return 0.0, 0.0

    total = time.perf_counter() - t_total_start
    if tok == 0 or first_token_at is None:
        return 0.0, total
    gen_seconds = time.perf_counter() - first_token_at
    if gen_seconds <= 0:
        return 0.0, total
    return tok / gen_seconds, total


async def _probe_worker_specs_via_ssh(worker: dict) -> dict:
    """Query CPU + RAM + GPU details over SSH using the worker's
    `ssh_host` alias. Returns a dict suitable for merging into
    capabilities. Empty dict on any failure.

    Workers run plain Ollama which doesn't expose system specs. The
    /api/ps proxy gives us a binary "GPU present" + a VRAM lower
    bound, but not CPU model, RAM total, or GPU class. SSH'ing to the
    worker is the most reliable way to fill in the gaps without
    requiring extra worker-side software. Free when ssh_host is
    already set (the same alias used for LAN model copy).
    """
    ssh_host = (worker.get("ssh_host") or "").strip()
    if not ssh_host:
        return {}

    # PowerShell one-liner that emits a JSON blob the host can parse.
    # We pipe it through ssh -o BatchMode=yes so the call fails fast
    # on auth issues instead of hanging on a password prompt.
    ps = (
        "$cs = Get-CimInstance Win32_ComputerSystem;"
        "$os = Get-CimInstance Win32_OperatingSystem;"
        "$cpu = Get-CimInstance Win32_Processor;"
        "$gpus = @();"
        "Get-CimInstance Win32_VideoController | ForEach-Object {"
        "  $gpus += [pscustomobject]@{"
        "    name = $_.Name;"
        "    adapter_ram_gb = [math]::Round([uint64]$_.AdapterRAM/1GB, 2);"
        "    driver_version = $_.DriverVersion"
        "  }"
        "};"
        "$out = [pscustomobject]@{"
        "  cpu_name = $cpu.Name;"
        "  cpu_cores = $cpu.NumberOfCores;"
        "  cpu_threads = $cpu.NumberOfLogicalProcessors;"
        "  ram_total_gb = [math]::Round($cs.TotalPhysicalMemory/1GB, 1);"
        "  ram_free_gb = [math]::Round($os.FreePhysicalMemory/1MB, 1);"
        "  gpus = $gpus"
        "};"
        "$out | ConvertTo-Json -Compress -Depth 4"
    )
    # PowerShell -EncodedCommand bypasses every layer of arg-list /
    # cmdline / shell quoting nightmare. The script is base64-encoded
    # UTF-16LE bytes — PowerShell decodes and runs it verbatim with
    # zero whitespace mangling.
    import base64
    encoded = base64.b64encode(ps.encode("utf-16-le")).decode("ascii")
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
           ssh_host, "powershell", "-NoProfile", "-EncodedCommand", encoded]
    import subprocess as _sp
    def _run() -> tuple[int, bytes, bytes]:
        try:
            r = _sp.run(
                cmd, capture_output=True, timeout=15.0,
            )
            return r.returncode, r.stdout, r.stderr
        except _sp.TimeoutExpired:
            return -1, b"", b"timeout"
        except Exception as e:
            return -2, b"", repr(e).encode()
    try:
        rc, stdout, stderr = await asyncio.to_thread(_run)
    except Exception as e:
        log.info("ssh specs probe failed for %s: %s", ssh_host, e)
        return {}
    if rc != 0:
        return {}

    # PowerShell's ConvertTo-Json may emit the object as either a JSON
    # object (single record) or a JSON array (when implicit
    # enumeration kicks in). Handle both.
    text = stdout.decode("utf-8", errors="replace").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, list):
        if not data:
            return {}
        data = data[0]
    if not isinstance(data, dict):
        return {}

    # Stamp the measurement so we know how stale the data is.
    data["specs_measured_at"] = time.time()
    return data


async def _attempt_rpc_server_restart(worker: dict) -> bool:
    """SSH into the worker and re-spawn its rpc-server.

    Used by `probe_worker` when the rpc-server TCP probe fails. If the
    worker has an `ssh_host` set, we kill any stale rpc-server.exe and
    relaunch it via WMI Win32_Process.Create so the new process
    survives the SSH session's exit. Returns True if the post-restart
    process appears alive (process found AND port 50052 listening).

    Locks to SYCL backend (`-d SYCL0`) so dual-loaded backends
    (SYCL + Vulkan both targeting the same Intel iGPU) don't get
    enumerated as separate RPC devices and double-allocate the iGPU
    during model load — that double-counting is what causes "Remote
    RPC server crashed" mid-push when the second virtual device runs
    out of physical memory.

    Failure modes (all return False, no exceptions):
      * No ssh_host configured (caller falls back to "unreachable").
      * ssh connect timeout / auth failure.
      * rpc-server.exe missing on the worker.
      * Process spawned but port not listening within ~4 s.
    """
    ssh_host = (worker.get("ssh_host") or "").strip()
    if not ssh_host:
        return False

    # Single PowerShell payload: kill stale, spawn fresh, verify alive.
    # Same WMI pattern used in the manual recovery commands so the
    # spawned process outlives the SSH session.
    ps = (
        "$ErrorActionPreference = 'Continue';"
        "Get-Process -Name 'rpc-server' -ErrorAction SilentlyContinue | "
        "  ForEach-Object { Stop-Process -Id $_.Id -Force };"
        "Start-Sleep -Milliseconds 800;"
        "$exe = \"$env:USERPROFILE\\.gigachat\\llama-cpp\\rpc-server.exe\";"
        "if (-not (Test-Path $exe)) { Write-Output 'NO_BINARY'; exit 2 };"
        "$cmdline = '\"' + $exe + '\" -H 0.0.0.0 -p 50052 -d SYCL0';"
        "$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
        "  -Arguments @{ CommandLine = $cmdline; "
        "                CurrentDirectory = \"$env:USERPROFILE\\.gigachat\\llama-cpp\" };"
        "if ($result.ReturnValue -ne 0) { Write-Output 'WMI_FAIL'; exit 3 };"
        "Start-Sleep -Seconds 4;"
        "$rpc = Get-Process -Name 'rpc-server' -ErrorAction SilentlyContinue;"
        "$port = Get-NetTCPConnection -LocalPort 50052 -State Listen -ErrorAction SilentlyContinue;"
        "if ($rpc -and $port) { Write-Output 'OK' } else { Write-Output 'NO_LISTEN' }"
    )
    import base64
    encoded = base64.b64encode(ps.encode("utf-16-le")).decode("ascii")
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
           ssh_host, "powershell", "-NoProfile", "-EncodedCommand", encoded]

    import subprocess as _sp
    def _run() -> tuple[int, bytes, bytes]:
        try:
            r = _sp.run(cmd, capture_output=True, timeout=30.0)
            return r.returncode, r.stdout, r.stderr
        except _sp.TimeoutExpired:
            return -1, b"", b"timeout"
        except Exception as e:
            return -2, b"", repr(e).encode()

    try:
        rc, stdout, stderr = await asyncio.to_thread(_run)
    except Exception as e:
        log.info("compute_pool: rpc-server restart ssh failed for %s: %s", ssh_host, e)
        return False

    out = stdout.decode("utf-8", errors="replace").strip()
    if rc != 0 or "OK" not in out:
        log.info(
            "compute_pool: rpc-server restart on %s did not come up "
            "(rc=%d, output=%r)", ssh_host, rc, out[-200:],
        )
        return False
    return True


async def _probe_rpc_server(
    host: str, port: int, timeout: float = _RPC_PROBE_TIMEOUT_SEC,
) -> tuple[bool, str | None]:
    """TCP-connect probe for rpc-server. Returns (reachable, error_str).

    rpc-server speaks llama.cpp's binary RPC protocol — it doesn't have
    an HTTP health endpoint, and its protocol handshake is more involved
    than we want to replicate just for liveness. A successful TCP connect
    is enough to know the listener is up.

    Address-family handling: mDNS hostnames like `worker.local` resolve
    to multiple addresses — typically IPv6 link-local first, then IPv6
    global, then IPv4. asyncio's `open_connection` tries them in
    sequence; the first IPv6 link-local fails after a multi-second OS-
    level timeout that defeats our deadline before the IPv4 fallback
    even gets a chance. To make the probe deterministic on a normal
    LAN, we resolve the host ourselves and try IPv4 first (or any
    IPv6 entry that isn't link-local), in parallel via gather. First
    successful connect wins.
    """
    try:
        # getaddrinfo returns ((family, type, proto, canonname, sockaddr), …).
        infos = await asyncio.get_event_loop().getaddrinfo(
            host, port, type=socket.SOCK_STREAM,
        )
    except Exception as e:
        return False, f"rpc-server probe: getaddrinfo failed: {e}"

    # Order: IPv4 first (LAN happy path), then IPv6 globals, then
    # IPv6 link-local last. We don't drop link-local entirely — on
    # some setups (no IPv4 stack at all) it's the only path — but it
    # shouldn't gate the probe.
    def _rank(info):
        family, _, _, _, sockaddr = info
        if family == socket.AF_INET:
            return 0
        ip = sockaddr[0] if sockaddr else ""
        if ip.startswith("fe80"):
            return 2  # link-local — last resort
        return 1
    infos = sorted(infos, key=_rank)

    async def _try(info):
        family, type_, proto, _, sockaddr = info
        try:
            sock = socket.socket(family, type_, proto)
            sock.setblocking(False)
            await asyncio.get_event_loop().sock_connect(sock, sockaddr)
            sock.close()
            return True, None
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    last_err: str | None = None
    deadline = asyncio.get_event_loop().time() + timeout
    for info in infos:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            break
        try:
            ok, err = await asyncio.wait_for(_try(info), timeout=remaining)
        except asyncio.TimeoutError:
            last_err = "timeout"
            continue
        if ok:
            return True, None
        last_err = err
    return False, f"rpc-server probe: {last_err or 'no addresses tried'}"


async def _probe_one(client: httpx.AsyncClient, base: str, token: str | None) -> dict:
    """Issue the probe GETs in parallel and merge the results.

    Returns a dict with `version`, `models`, hardware-capability hints,
    and any `*_error` markers. Any single endpoint may fail without
    breaking the others — we want the most signal we can get out of
    one round trip.

    Hardware-capability detection: workers run plain Ollama so they
    don't volunteer system specs. The closest signal Ollama exposes is
    `/api/ps` — currently-loaded models with their VRAM split. From
    that we infer:
      * `gpu_present`: True if any loaded model reports `size_vram > 0`.
        A worker with no loaded models doesn't tell us anything yet,
        which is why we treat absence as `False` rather than `None`.
      * `max_vram_seen_bytes`: the largest `size_vram` across loaded
        models, giving a rough lower bound on the worker's VRAM. The
        router uses this to prefer hardware-stronger workers when
        ranking ties.
    """
    headers: dict[str, str] = {}
    if token:
        # Bearer header is the convention Gigachat already uses for the
        # main loopback auth — re-use it so the worker side can validate
        # with the same code path. The worker is expected to enforce
        # this through its own AuthMiddleware-style gate.
        headers["Authorization"] = f"Bearer {token}"

    async def _ver() -> Any:
        r = await client.get(f"{base}/api/version", headers=headers)
        r.raise_for_status()
        return r.json()

    async def _tags() -> Any:
        r = await client.get(f"{base}/api/tags", headers=headers)
        r.raise_for_status()
        return r.json()

    async def _ps() -> Any:
        r = await client.get(f"{base}/api/ps", headers=headers)
        r.raise_for_status()
        return r.json()

    out: dict[str, Any] = {}
    # gather() with return_exceptions so a partial failure on one
    # endpoint doesn't lose the other endpoints' payloads.
    # Time the gather as a coarse LAN-latency proxy. With three small
    # GETs in parallel the wall time is dominated by the slowest
    # roundtrip, which gives us a reasonable "how snappy is this LAN
    # link" signal for the routing decision.
    _t0 = time.perf_counter()
    ver_res, tags_res, ps_res = await asyncio.gather(
        _ver(), _tags(), _ps(), return_exceptions=True,
    )
    out["probe_latency_ms"] = int((time.perf_counter() - _t0) * 1000)
    if isinstance(ver_res, Exception):
        out["version_error"] = f"{type(ver_res).__name__}: {ver_res}"
    else:
        out["version"] = (ver_res or {}).get("version") or "unknown"
    if isinstance(tags_res, Exception):
        out["tags_error"] = f"{type(tags_res).__name__}: {tags_res}"
    else:
        models = []
        for m in (tags_res or {}).get("models", []) or []:
            details = (m.get("details") or {}) if isinstance(m, dict) else {}
            models.append({
                "name": m.get("name") if isinstance(m, dict) else None,
                "size": m.get("size") if isinstance(m, dict) else None,
                "family": details.get("family"),
                "parameter_size": details.get("parameter_size"),
                "quantization_level": details.get("quantization_level"),
            })
        # Drop entries with no name (defensive against malformed responses).
        out["models"] = [m for m in models if m.get("name")]
    # /api/ps is purely a heuristic source — failures shouldn't dim the
    # probe's overall verdict, just leave the hardware fields empty so
    # the router falls through to the no-info default.
    if isinstance(ps_res, Exception):
        out["gpu_present"] = False
        out["max_vram_seen_bytes"] = 0
        out["loaded_count"] = 0
    else:
        loaded = (ps_res or {}).get("models", []) or []
        max_vram = 0
        any_gpu = False
        for m in loaded:
            v = int(m.get("size_vram") or 0)
            if v > 0:
                any_gpu = True
            if v > max_vram:
                max_vram = v
        out["gpu_present"] = any_gpu
        out["max_vram_seen_bytes"] = max_vram
        out["loaded_count"] = len(loaded)
    return out


async def probe_worker(wid: str) -> dict:
    """Probe one worker now and persist the result.

    Returns a dict with `ok`, the merged probe payload (`version`,
    `models`, or error markers), and the `last_seen` timestamp. Safe
    to call from any context — failures are caught and recorded on
    the row, never propagated.
    """
    worker = db.get_compute_worker(wid)
    if not worker:
        return {"ok": False, "error": "worker not found"}
    if not worker.get("enabled"):
        return {"ok": False, "error": "worker disabled — enable it first"}

    base = _worker_base_url(worker)
    token = db.get_compute_worker_auth_token(wid)
    now = time.time()

    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SEC) as client:
            payload = await _probe_one(client, base, token)
    except Exception as e:
        # Network-level failure — connection refused, DNS miss, etc.
        # Record on the row so the UI can show "unreachable since X".
        err = f"{type(e).__name__}: {e}"
        try:
            db.update_compute_worker_capabilities(
                wid, last_seen=now, last_error=err,
            )
        except Exception:
            pass
        return {"ok": False, "error": err, "last_seen": now}

    # Two-endpoint outcome: success only if BOTH endpoints responded.
    # Partial success is still a usable signal — record what we got
    # but flag the error so the UI can warn.
    has_models = bool(payload.get("models"))
    has_version = "version" in payload
    error_parts = []
    if not has_models:
        error_parts.append(payload.get("tags_error") or "no models field")
    if not has_version:
        error_parts.append(payload.get("version_error") or "no version field")
    error_str = "; ".join(error_parts) if error_parts else ""

    # Phase 2 add-on: TCP-probe the worker's rpc-server port so the UI
    # can show whether layer-split inference is available on this
    # worker. Probe failure is NOT counted as `last_error` — rpc-server
    # is optional (Phase 1 routing works without it), so a worker with
    # Ollama up but rpc-server down is still "online" for chat /
    # embeddings / subagent routing. We just record the rpc state in
    # capabilities so commit 6's UI can render a separate badge.
    rpc_host = _worker_host(worker)
    rpc_port = _DEFAULT_RPC_PORT
    rpc_ok, rpc_err = await _probe_rpc_server(rpc_host, rpc_port)

    # Self-heal: if rpc-server is down AND the worker has ssh_host
    # configured, attempt to restart it remotely. This covers the
    # common case where the rpc-server process died (crash, OS
    # restart, transient driver issue) since the previous probe.
    # We try restart at most once per probe cycle to avoid tight
    # loops if the binary is genuinely broken — a successful restart
    # is verified by an immediate re-probe of the port.
    if not rpc_ok and (worker.get("ssh_host") or "").strip():
        log.info(
            "compute_pool: rpc-server unreachable on %s; attempting "
            "remote restart via ssh_host=%s",
            rpc_host, worker["ssh_host"],
        )
        restarted = await _attempt_rpc_server_restart(worker)
        if restarted:
            # Re-probe to confirm the new process is actually serving.
            rpc_ok, rpc_err = await _probe_rpc_server(rpc_host, rpc_port)
            if rpc_ok:
                log.info("compute_pool: rpc-server self-heal succeeded on %s", rpc_host)

    capabilities = {
        "version": payload.get("version"),
        "models": payload.get("models") or [],
        "rpc_server_reachable": rpc_ok,
        "rpc_port": rpc_port,
        "rpc_error": rpc_err,
        # Hardware hints — best-effort, populated from /api/ps. Used by
        # the router to prefer hardware-stronger workers over weaker
        # ones when both are otherwise eligible.
        "gpu_present": bool(payload.get("gpu_present")),
        "max_vram_seen_bytes": int(payload.get("max_vram_seen_bytes") or 0),
        "loaded_count": int(payload.get("loaded_count") or 0),
        # LAN-latency hint from the parallel probe-GET round trip.
        # Used by the routing layer as a proxy for "how expensive is
        # a per-token RPC roundtrip to this worker": low latency →
        # split path is viable; high latency → biased toward host
        # CPU offload over split.
        "probe_latency_ms": int(payload.get("probe_latency_ms") or 0),
    }

    # Carry over throughput + spec data from the previous probe IF
    # they're still fresh (1 hour TTL). Re-measure only when stale —
    # both are expensive enough that we don't want to run them on
    # every 5-min sweep.
    prev_caps = worker.get("capabilities") or {}
    now_t = time.time()
    if prev_caps.get("tokens_per_second") and (
        now_t - (prev_caps.get("tps_measured_at") or 0)
        < _THROUGHPUT_CACHE_TTL_SEC
    ):
        capabilities["tokens_per_second"] = prev_caps["tokens_per_second"]
        capabilities["tps_measured_at"] = prev_caps["tps_measured_at"]
        capabilities["tps_model"] = prev_caps.get("tps_model")
    if prev_caps.get("specs_measured_at") and (
        now_t - prev_caps["specs_measured_at"] < _THROUGHPUT_CACHE_TTL_SEC
    ):
        for k in ("cpu_name", "cpu_cores", "cpu_threads",
                  "ram_total_gb", "ram_free_gb", "gpus", "specs_measured_at"):
            if k in prev_caps:
                capabilities[k] = prev_caps[k]

    # Throughput bench — pick the smallest chat model on the worker
    # so the bench loads fast. Embed-only models give us no chat
    # speed signal; skip them as bench targets.
    if "tokens_per_second" not in capabilities:
        chat_models = [
            m for m in (capabilities.get("models") or [])
            if (m.get("name") or "") and "embed" not in (m.get("name") or "").lower()
        ]
        if chat_models:
            chat_models.sort(key=lambda m: int(m.get("size") or 1 << 60))
            bench_model = chat_models[0]["name"]
            tps, _ = await _measure_throughput(base, token, bench_model)
            if tps > 0:
                capabilities["tokens_per_second"] = tps
                capabilities["tps_measured_at"] = now_t
                capabilities["tps_model"] = bench_model

    # SSH-based hardware spec probe — fills in CPU/RAM/GPU details
    # the API can't surface. Free when the user has set ssh_host.
    if "specs_measured_at" not in capabilities:
        specs = await _probe_worker_specs_via_ssh(worker)
        capabilities.update(specs)
    try:
        db.update_compute_worker_capabilities(
            wid,
            capabilities=capabilities,
            last_seen=now,
            # Empty string clears any previous error; non-empty records.
            last_error=error_str if error_str else "",
        )
    except Exception:
        pass

    return {
        "ok": has_models and has_version,
        "capabilities": capabilities,
        "error": error_str or None,
        "last_seen": now,
    }


async def probe_all_enabled() -> list[dict]:
    """Probe every enabled worker concurrently. Returns a list of
    `{worker_id, label, ok, error}` summaries — the heavy capability
    payload is persisted on the row, this just tells the caller what
    happened in aggregate."""
    workers = db.list_compute_workers(enabled_only=True)
    if not workers:
        return []
    results = await asyncio.gather(
        *(probe_worker(w["id"]) for w in workers),
        return_exceptions=True,
    )
    summaries: list[dict] = []
    for w, r in zip(workers, results):
        if isinstance(r, Exception):
            summaries.append({
                "worker_id": w["id"],
                "label": w["label"],
                "ok": False,
                "error": f"{type(r).__name__}: {r}",
            })
        else:
            summaries.append({
                "worker_id": w["id"],
                "label": w["label"],
                "ok": bool(r.get("ok")),
                "error": r.get("error"),
            })
    return summaries


_PROBE_TASK: asyncio.Task | None = None


async def _periodic_loop() -> None:
    """Internal: sweep every `_SWEEP_INTERVAL_SEC`. Started/stopped via
    `start_periodic_probe` / `stop_periodic_probe` on app lifecycle.

    After each liveness sweep we drain any queued auto-syncs that
    routing calls deferred. Doing it here (rather than from the
    routing call) keeps SCP off the chat hot path — bench-confirmed
    win in commit 18.
    """
    while True:
        try:
            await probe_all_enabled()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("compute_pool periodic probe failed: %s", e)
        # Drain the auto-sync queue after the sweep — workers'
        # capabilities reflect any newly-installed models from the
        # previous drain, so the queue from this turn is fresh.
        try:
            n = await drain_deferred_syncs()
            if n:
                log.info("compute_pool: started %d deferred LAN-copy task(s)", n)
        except Exception as e:
            log.warning("compute_pool: drain_deferred_syncs failed: %s", e)
        try:
            await asyncio.sleep(_SWEEP_INTERVAL_SEC)
        except asyncio.CancelledError:
            raise


def start_periodic_probe() -> None:
    """Schedule the background sweep. Idempotent — calling twice is
    a no-op. Called from app.py's startup hook."""
    global _PROBE_TASK
    if _PROBE_TASK and not _PROBE_TASK.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    _PROBE_TASK = loop.create_task(_periodic_loop())


def stop_periodic_probe() -> None:
    """Cancel the background sweep. Called from app shutdown so the
    test runner doesn't see a stranded task."""
    global _PROBE_TASK
    if _PROBE_TASK and not _PROBE_TASK.done():
        _PROBE_TASK.cancel()
    _PROBE_TASK = None


# ---------------------------------------------------------------------------
# Routing: pick a worker for a given workload.
#
# These helpers translate "I need to run an embed / chat / subagent call"
# into "send it to base URL X with bearer Y", or `None` to mean "no
# eligible worker — the host should handle it locally". Everything is
# read-only relative to the worker rows; the routing layer never mutates
# state.
#
# Eligibility for any workload:
#   * row is `enabled`
#   * the workload-specific flag is on (`use_for_embeddings`, etc.)
#   * the last probe succeeded — `last_seen` within `_FRESHNESS_SEC` AND
#     `last_error` is empty. A worker that hasn't been probed yet (last_seen
#     is None) is skipped: we don't want to trust a row the user just
#     added until the periodic sweep — or a manual "Test connection" —
#     confirms it can serve traffic.
#   * if `model` is supplied, the worker's cached capabilities list it as
#     installed (with or without a `:latest` tag suffix).
# ---------------------------------------------------------------------------

# How long after `last_seen` we still consider a worker "fresh enough" to
# route to. The periodic probe runs every 5 min; this 1-hour window covers
# 12 sweeps' worth of buffer for transient blips.
_FRESHNESS_SEC = 60 * 60


def _model_matches(installed_name: str, requested: str) -> bool:
    """Compare an installed model name to a requested name, tolerant to
    Ollama's `:latest` tag default.

    `nomic-embed-text` matches `nomic-embed-text:latest`, and an explicit
    tag (`gemma4:e4b`) matches itself. Mismatched explicit tags are NOT
    coerced — `gemma4:e4b` does not match `gemma4:e2b`.
    """
    if not installed_name or not requested:
        return False
    if installed_name == requested:
        return True
    # Strip `:latest` from either side and compare bare names.
    inst_bare = installed_name.split(":", 1)[0] if ":" in installed_name else installed_name
    req_bare = requested.split(":", 1)[0] if ":" in requested else requested
    inst_has_explicit_tag = ":" in installed_name and not installed_name.endswith(":latest")
    req_has_explicit_tag = ":" in requested and not requested.endswith(":latest")
    # If both sides have explicit tags, they had to match exactly above.
    if inst_has_explicit_tag and req_has_explicit_tag:
        return False
    return inst_bare == req_bare


def _worker_has_model(worker: dict, model: str) -> bool:
    caps = worker.get("capabilities") or {}
    for m in caps.get("models") or []:
        if _model_matches(m.get("name") or "", model):
            return True
    return False


def _is_fresh(worker: dict, now: float | None = None) -> bool:
    """Last successful probe within `_FRESHNESS_SEC`."""
    last_seen = worker.get("last_seen")
    if not last_seen:
        return False
    if worker.get("last_error"):
        return False
    if now is None:
        now = time.time()
    return (now - float(last_seen)) < _FRESHNESS_SEC


def _capability_score(worker: dict) -> tuple:
    """Power-ranking key for sorting eligible workers.

    Returns a tuple suitable for `sorted(key=...)` — bigger is better,
    so callers use `reverse=True`. Components, in priority order:

      1. `tokens_per_second` — real measured throughput from a tiny
         /api/generate bench. Folds CPU + RAM bandwidth + GPU compute
         + memory size into one bottom-line "how fast can this machine
         run a model" number. Workers we haven't benchmarked yet
         contribute 0 here and fall back to the heuristic axes below.
      2. `gpu_present` — binary GPU detection from /api/ps.
      3. `max_vram_seen_bytes` — lower bound on the worker's VRAM.
      4. `ram_total_gb` — worker RAM total (from SSH probe). Held
         out when ssh_host isn't set.
      5. `cpu_threads` — CPU parallelism (also from SSH probe).
      6. `last_seen` — final tie-breaker.

    All four axes are intentionally factored: throughput is the
    bottom-line speed signal but only available where benchmarked;
    the heuristic axes (gpu / vram / ram / cpu) keep ranking sane
    for workers without measurement data yet.
    """
    caps = worker.get("capabilities") or {}
    return (
        float(caps.get("tokens_per_second") or 0.0),
        1 if caps.get("gpu_present") else 0,
        int(caps.get("max_vram_seen_bytes") or 0),
        float(caps.get("ram_total_gb") or 0.0),
        int(caps.get("cpu_threads") or 0),
        float(worker.get("last_seen") or 0),
    )


async def _measure_host_throughput(model_name: str) -> float:
    """Cached host throughput in tokens/sec for `model_name`. Runs a
    tiny /api/generate against `localhost:11434`; result valid for
    `_THROUGHPUT_CACHE_TTL_SEC`. Returns 0 on failure (caller falls
    back to heuristic ranking).
    """
    cached = _HOST_THROUGHPUT_CACHE.get(model_name)
    now = time.time()
    if cached and (now - cached[1]) < _THROUGHPUT_CACHE_TTL_SEC:
        return cached[0]
    tps, _ = await _measure_throughput("http://localhost:11434", None, model_name)
    if tps > 0:
        _HOST_THROUGHPUT_CACHE[model_name] = (tps, now)
    return tps


def _host_capability_score(model_name: str | None = None) -> tuple:
    """Host's capability tuple, same shape as `_capability_score`.

    When `model_name` is provided AND we've benchmarked the host on
    that model recently, slot the measured TPS in as the primary
    axis. Otherwise we leave it at 0 and rely on the static heuristic
    axes (vram bytes, ram_gb, cpu_threads, last_seen=inf for
    "always online"). Last component is +inf so a host that ties on
    all measurable axes still wins over a worker (no LAN hop).
    """
    try:
        spec = sysdetect.detect_system()
        vram_gb = float(spec.get("vram_gb") or 0.0)
        ram_gb = float(spec.get("ram_gb") or 0.0)
    except Exception:
        vram_gb = 0.0
        ram_gb = 0.0

    host_tps = 0.0
    if model_name:
        cached = _HOST_THROUGHPUT_CACHE.get(model_name)
        if cached and (time.time() - cached[1]) < _THROUGHPUT_CACHE_TTL_SEC:
            host_tps = cached[0]

    try:
        cpu_threads = int(os.cpu_count() or 0)
    except Exception:
        cpu_threads = 0

    return (
        host_tps,
        1 if vram_gb > 0 else 0,
        int(vram_gb * 1024 ** 3),
        ram_gb,
        cpu_threads,
        float("inf"),
    )


def _eligible_workers(flag: str, model: str | None = None) -> list[dict]:
    """Return enabled+fresh workers whose `flag` is on and (optionally) have
    `model` installed.

    Sort order (`_capability_score` desc): GPU-present workers first,
    then by max VRAM observed, then freshest probe. The user's
    intuition "use the more powerful machine" maps to (1) → GPU wins
    over CPU-only, (2) → more VRAM wins over less. Tie at hardware →
    fall back to freshest, like Phase 1 commit 6's original behavior.

    Auto-sync side effect: when a worker has `ssh_host` set, is
    enabled+fresh, and has the right `flag` on, but is missing the
    model, this function kicks off a background SCP from host so the
    model lands on the worker without the user touching a button. The
    current call still excludes that worker (it really doesn't have
    the model right now), but subsequent probes will see the new model
    and the worker becomes eligible. Failures are silent — auto-sync
    is best-effort.
    """
    rows = db.list_compute_workers(enabled_only=True)
    now = time.time()
    out: list[dict] = []
    for w in rows:
        if not w.get(flag):
            continue
        if not _is_fresh(w, now=now):
            continue
        if model:
            if _worker_has_model(w, model):
                out.append(w)
            elif w.get("ssh_host"):
                _maybe_kickoff_lan_sync(w, model)
            # else: no ssh_host, no path to install — skip silently.
        else:
            out.append(w)
    out.sort(key=_capability_score, reverse=True)
    return out


# Auto-sync architecture (Phase 2 commit 18 redesign):
#
# Auto-syncing was previously fired DIRECTLY from the routing call
# (`_eligible_workers`) — every chat turn that found a worker missing
# a model would kick off a background SCP. Real-world bench showed
# that SCP eats host disk + CPU bandwidth, contending with the chat
# turn that's actively streaming. Net result: chat got 4% slower with
# pool enabled, embed savings didn't compound.
#
# New design: routing calls only ENQUEUE the (worker, model) sync
# requests; they do NOT spawn SCPs. The periodic probe loop
# (`_periodic_loop`, every 5 min) drains the queue and runs the syncs
# then. So:
#   * Chat turns never compete with SCP for host bandwidth.
#   * Background sync still happens — just batched on the probe
#     cadence instead of per-turn.
#   * The bench now measures hot inference cleanly, no SCP noise.
_DEFERRED_SYNC_QUEUE: set[tuple[str, str]] = set()
_AUTO_SYNC_TASKS: dict[tuple[str, str], asyncio.Task] = {}


def _maybe_kickoff_lan_sync(worker: dict, model_name: str) -> None:
    """Defer (not fire) a LAN-copy of `model_name` to `worker`.

    Called from `_eligible_workers` when a worker is otherwise eligible
    but missing the model. Just adds to the deferred queue; the
    periodic probe loop drains it. Means routing decisions are pure
    DB reads — no SCP work in the chat hot path.

    Guards:
      * ssh_host must be set on the worker.
      * Skip if already in queue or already mid-sync.
    """
    wid = worker.get("id")
    if not wid or not model_name:
        return
    if not (worker.get("ssh_host") or "").strip():
        return
    key = (wid, model_name)
    if key in _DEFERRED_SYNC_QUEUE:
        return
    existing = _AUTO_SYNC_TASKS.get(key)
    if existing and not existing.done():
        return  # already mid-flight from a prior drain
    _DEFERRED_SYNC_QUEUE.add(key)


async def drain_deferred_syncs() -> int:
    """Run any queued auto-syncs in the background.

    Called from the periodic probe loop AFTER the liveness sweep
    finishes — so syncs never compete with chat traffic the way the
    old per-routing trigger did. Returns the number of syncs
    started so the caller can log it.
    """
    started = 0
    pending = list(_DEFERRED_SYNC_QUEUE)
    _DEFERRED_SYNC_QUEUE.clear()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Re-enqueue and bail; next sweep will catch it.
        _DEFERRED_SYNC_QUEUE.update(pending)
        return 0

    for wid, model_name in pending:
        existing = _AUTO_SYNC_TASKS.get((wid, model_name))
        if existing and not existing.done():
            continue
        async def _run(wid=wid, model_name=model_name):
            try:
                from . import model_sync
                await model_sync.sync_model(model_name, wid)
                log.info("compute_pool: auto-synced %s -> worker %s",
                         model_name, wid)
            except Exception as e:
                log.info("compute_pool: auto-sync %s -> %s failed: %s",
                         model_name, wid, e)
            finally:
                _AUTO_SYNC_TASKS.pop((wid, model_name), None)
        _AUTO_SYNC_TASKS[(wid, model_name)] = loop.create_task(_run())
        started += 1
    return started


def pick_embed_target(model: str) -> tuple[str, str | None] | None:
    """Choose a worker to run an embed request against, or None for host.

    Returns `(base_url, auth_token_or_None)`. `auth_token` is fetched
    from the dedicated `get_compute_worker_auth_token` so the token never
    sits on a row dict. Caller composes the URL as `f"{base}/api/embeddings"`
    and adds `Authorization: Bearer …` when the token is set.
    """
    cands = _eligible_workers("use_for_embeddings", model=model)
    if not cands:
        return None
    w = cands[0]
    base = _worker_base_url(w)
    token = db.get_compute_worker_auth_token(w["id"])
    return (base, token)


def pick_chat_target(model: str) -> tuple[str, str | None] | None:
    """Choose where the chat turn runs — return (base_url, token) for a
    worker, or None to keep it on host.

    This is where the "use the more powerful machine" rule lives.
    Among eligible chat workers we already sort by `_capability_score`
    (GPU > CPU, more VRAM > less). Now we ALSO compare the strongest
    eligible worker against the host's own capability — if the worker
    isn't strictly more powerful than host, we keep the turn local
    (no LAN round-trip, KV cache stays warm). The worker only wins
    when its proven hardware beats the host's.

    Concretely: a worker beats host when EITHER
      * host has no GPU and the worker does, OR
      * both have a GPU but the worker has loaded a model larger than
        the host's total VRAM (max_vram_seen_bytes > host_vram_bytes),
        which is a hard lower-bound proof of more capacity.
    """
    cands = _eligible_workers("use_for_chat", model=model)
    if not cands:
        return None
    w = cands[0]
    # Schedule a background bench of the host on this exact model so
    # the cache is populated for the next routing call. We can't await
    # here (this function is sync), so the FIRST call falls through
    # to heuristic axes only — that's safer than letting the worker's
    # measured-TPS field beat the host's missing-measurement field.
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_measure_host_throughput(model))
    except RuntimeError:
        pass

    worker_score = _capability_score(w)
    host_score = _host_capability_score(model)
    # If either side hasn't been benchmarked on this exact model
    # (TPS=0 means cache miss or measurement failure), fall through
    # to the heuristic axes so the comparison is apples-to-apples.
    # Otherwise the worker's "measured 4 tok/s on a different model"
    # would beat the host's "no measurement yet" → false positive.
    if worker_score[0] <= 0 or host_score[0] <= 0:
        worker_score = (0.0,) + worker_score[1:]
        host_score = (0.0,) + host_score[1:]

    if worker_score <= host_score:
        # Host wins (using measured throughput when available, else
        # falling back to GPU/VRAM/RAM/CPU heuristics). Stay local.
        return None
    return (_worker_base_url(w), db.get_compute_worker_auth_token(w["id"]))


def pick_split_chat_target(model_name: str) -> tuple[str, str] | None:
    """Legacy lookup retained for back-compat with tests / explicit
    `split:<label>` model names. The auto-router below
    (`route_chat_for`) is the live entry point now; it never produces
    `split:` prefixes — it just inspects the model and decides whether
    to engage the split path transparently.

    Returns `(base_url, label)` for an explicit `split:<label>` whose
    row is `running`; None otherwise.
    """
    if not model_name or not model_name.startswith("split:"):
        return None
    label = model_name[len("split:"):].strip()
    if not label:
        return None
    for row in db.list_split_models(enabled_only=True):
        if row.get("label") == label and row.get("status") == "running":
            return (f"http://127.0.0.1:{row['llama_port']}", label)
    return None


# ---------------------------------------------------------------------------
# Auto-routing: pick host-Ollama vs spawn-llama-server-with-RPC for a model
# ---------------------------------------------------------------------------
# This is the intelligent split-or-not decision the user actually picks
# their model against. They never see `split:<label>` — they pick e.g.
# `gemma3:27b` from the model picker, and the router decides:
#
#   1. Resolve the model's GGUF blob + size from Ollama's manifest store.
#   2. If size <= host_vram_budget → Ollama on host (fastest path; 0 LAN
#      overhead; workers stay free for parallel embeddings/subagents).
#   3. Else → ensure a llama-server is running for THIS exact model with
#      --rpc to every eligible compute worker, return its URL.
#   4. If the model can't fit even with the combined pool → raise so the
#      user gets a clear error rather than silent OOM.
#
# Per-conversation lifecycle: when the user switches between two big
# models, we stop the previously-running llama-server (one big model hot
# at a time — we have one finite VRAM budget). Switching back to a small
# model also stops any running llama-server so its VRAM is freed for
# Ollama.
# ---------------------------------------------------------------------------

# How much of the host's VRAM we're willing to let one model occupy
# without engaging the split path. 85% leaves headroom for the OS, the
# desktop compositor, any other Ollama models loaded simultaneously, and
# the model's KV cache. Below this fraction → Ollama. Above → split.
_HOST_VRAM_USE_FRACTION = 0.85

# How much of the host's TOTAL memory (VRAM + RAM) we'll trust Ollama
# to run a single model from. Ollama auto-offloads layers that don't
# fit VRAM into system RAM and runs them on CPU — slow per layer, but
# strictly faster than streaming layer activations across a LAN every
# token. 70% of (vram + ram) leaves room for the OS, browser, the
# Gigachat backend itself, and KV cache. Below this fraction → host
# (Ollama). Above → engage split path with workers.
_HOST_TOTAL_USE_FRACTION = 0.70

# Path to Ollama's on-disk model store. Default location on every
# platform Ollama supports; matches what `ollama_runtime` assumes.
_OLLAMA_MODELS_DIR = Path.home() / ".ollama" / "models"

# Where users drop replacement GGUFs that override Ollama's blob for a given
# model name. Necessary because some upstream GGUFs ship with a tensor layout
# that stock llama.cpp can't load even though the architecture is supported
# (e.g. Ollama's `gemma4:26b` packs the MoE expert weights as fused
# `ffn_gate_up_exps` per layer — 658 tensors total — while llama.cpp's
# `gemma4` loader spec expects them unfused as `ffn_gate_exps` +
# `ffn_up_exps` separately, 1014 tensors). The override path lets the user
# install Unsloth's GGUF (which uses the unfused layout) and have Phase 2
# split routing pick it up automatically without us mutating Ollama's
# blob store. Filename is the Ollama model name with `:` replaced by `-`,
# so `gemma4:26b` becomes `gemma4-26b.gguf`.
_OVERRIDE_GGUF_DIR = Path.home() / ".gigachat" / "llama-cpp" / "models"


def _override_gguf_path_for(model_name: str) -> Path:
    """Map an Ollama model name to its override GGUF path.

    Returns the path regardless of whether the file exists — callers
    check `.is_file()` to decide whether to use it.
    """
    safe = (model_name or "").strip().replace(":", "-").replace("/", "-")
    return _OVERRIDE_GGUF_DIR / f"{safe}.gguf"


def _override_mmproj_path_for(model_name: str) -> Path:
    """Map an Ollama model name to its multimodal projector path.

    Same naming convention as the main override but with a `.mmproj`
    infix: `gemma4:26b` → `gemma4-26b.mmproj.gguf`. When this file
    exists alongside the main override, llama-server is launched
    with `--mmproj <path>` enabling vision input. The mmproj is
    distributed/produced separately from the main LLM GGUF; for
    Ollama-bundled multimodal models the user can either run our
    extraction script once or download a pre-built mmproj from the
    upstream repo (e.g. Unsloth's `mmproj-BF16.gguf`).
    """
    safe = (model_name or "").strip().replace(":", "-").replace("/", "-")
    return _OVERRIDE_GGUF_DIR / f"{safe}.mmproj.gguf"


def _resolve_ollama_manifest(model_name: str) -> dict | None:
    """Locate the manifest JSON for an Ollama model name.

    Ollama stores manifests at
        ~/.ollama/models/manifests/<registry>/<namespace>/<name>/<tag>
    The default registry is `registry.ollama.ai` and the default namespace
    is `library`. Custom registries / namespaces are rare for end users
    but we still walk the tree to find the file.

    Returns the parsed manifest dict, or None if no matching file exists
    (model not pulled, name typo, etc.).
    """
    name = (model_name or "").strip()
    if not name:
        return None
    # Tag defaults to `latest` when omitted.
    if ":" in name:
        bare, tag = name.split(":", 1)
    else:
        bare, tag = name, "latest"

    # Default location first — covers >99% of cases.
    candidate = _OLLAMA_MODELS_DIR / "manifests" / "registry.ollama.ai" / "library" / bare / tag
    if candidate.is_file():
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            return None

    # Fallback: walk the manifests tree. Slower but handles the user
    # who pulled a model from a non-default registry.
    manifests = _OLLAMA_MODELS_DIR / "manifests"
    if not manifests.is_dir():
        return None
    needle = (bare, tag)
    for root, _dirs, files in os.walk(manifests):
        for fname in files:
            if fname == tag and Path(root).name == bare:
                try:
                    return json.loads(Path(root, fname).read_text(encoding="utf-8"))
                except Exception:
                    continue
    return None


def resolve_ollama_model(model_name: str) -> dict | None:
    """Resolve a model name to its on-disk GGUF + size.

    Returns `{gguf_path, size_bytes, manifest}` or None if the manifest
    can't be found. The "model" layer of the manifest is the GGUF; we
    compute size from `layers[].size` for the layer whose mediaType is
    `application/vnd.ollama.image.model` (large) — license / params /
    template layers are tiny config blobs and don't count toward VRAM.

    `gguf_path` is the absolute path to the blob, suitable for passing
    as llama-server's `--model` flag.

    Override path: if a file exists at `_override_gguf_path_for(name)`,
    we return that instead of the Ollama blob. Used to swap in GGUFs
    with tensor layouts that stock llama.cpp can load (see the
    `_OVERRIDE_GGUF_DIR` docstring) without disturbing Ollama's own
    blob store. The Ollama manifest is still consulted for the size
    metadata so routing / VRAM-budget logic stays consistent.
    """
    manifest = _resolve_ollama_manifest(model_name)
    if not manifest:
        return None
    layers = manifest.get("layers") or []
    model_layer = None
    for layer in layers:
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            model_layer = layer
            break
    if not model_layer:
        return None
    digest = (model_layer.get("digest") or "").replace("sha256:", "")
    if not digest:
        return None
    blob_path = _OLLAMA_MODELS_DIR / "blobs" / f"sha256-{digest}"
    if not blob_path.is_file():
        return None

    # Override hook: prefer a user-installed replacement GGUF if one
    # exists. Size is taken from THAT file (not the manifest), since
    # the override may be a different quantization than the Ollama
    # blob. Manifest-derived size is kept as a hint for callers that
    # want the original.
    override = _override_gguf_path_for(model_name)
    if override.is_file():
        try:
            override_size = override.stat().st_size
        except OSError:
            override_size = 0
        # Multimodal projector — surface its path if installed beside
        # the main override. Phase 2 split (`split_lifecycle._build_command`)
        # passes it as `--mmproj` so llama-server can serve image input.
        mmproj = _override_mmproj_path_for(model_name)
        mmproj_str = str(mmproj) if mmproj.is_file() else None
        return {
            "gguf_path": str(override),
            "size_bytes": override_size or int(model_layer.get("size") or 0),
            "manifest": manifest,
            "ollama_blob_path": str(blob_path),
            "override": True,
            "mmproj_path": mmproj_str,
        }

    return {
        "gguf_path": str(blob_path),
        "size_bytes": int(model_layer.get("size") or 0),
        "manifest": manifest,
    }


def _host_vram_budget_bytes() -> int:
    """Bytes of host VRAM we're willing to let a single Ollama-loaded
    model occupy in VRAM only. Used for ranking host vs workers in
    `_host_capability_score` (which compares pure GPU memory). NOT the
    threshold for engaging the split path — that's
    `_host_total_capacity_bytes` below."""
    try:
        spec = sysdetect.detect_system()
        vram_gb = float(spec.get("vram_gb") or 0.0)
    except Exception:
        vram_gb = 0.0
    return int(vram_gb * _HOST_VRAM_USE_FRACTION * 1024 * 1024 * 1024)


def _host_total_capacity_bytes() -> int:
    """Bytes the host can hold a single model in (VRAM + RAM). Ollama
    natively uses CPU offload — layers that don't fit VRAM live in
    system RAM and run on CPU. That stays single-node (no per-token
    LAN overhead), so we should NOT engage split unless the model
    truly exceeds the host's total memory budget.

    For this host (8 GB VRAM + 15.8 GB RAM, 70% safety margin): ≈16.7 GB.
    A 9 GB model fits trivially, even though it doesn't fit VRAM alone.
    """
    try:
        spec = sysdetect.detect_system()
        vram_gb = float(spec.get("vram_gb") or 0.0)
        ram_gb = float(spec.get("ram_gb") or 0.0)
    except Exception:
        vram_gb = 0.0
        ram_gb = 0.0
    total_gb = vram_gb + ram_gb
    return int(total_gb * _HOST_TOTAL_USE_FRACTION * 1024 * 1024 * 1024)


def _eligible_split_workers() -> list[dict]:
    """Workers that can contribute layers via rpc-server.

    Same `enabled` + `use_for_chat` gate as Phase 1's chat picker —
    workers the user toggled off for chat shouldn't suddenly start
    receiving inference traffic just because the model needs splitting.
    Plus the rpc-specific gate: probe must report rpc-server reachable.
    """
    rows = db.list_compute_workers(enabled_only=True)
    out = []
    for w in rows:
        if not w.get("use_for_chat"):
            continue
        caps = w.get("capabilities") or {}
        if not caps.get("rpc_server_reachable"):
            continue
        out.append(w)
    # Freshest probe first — if we have to pick a subset later, we'd
    # rather use the worker we just confirmed alive than one whose
    # last_seen is an hour stale.
    out.sort(key=lambda w: float(w.get("last_seen") or 0), reverse=True)
    return out


async def _ensure_split_running_for(
    model_name: str,
    gguf_path: str,
    worker_ids: list[str],
    mmproj_path: str | None = None,
) -> str:
    """Idempotent: ensure a `split_models` row exists + is running for
    this exact (model_name, gguf_path[, mmproj_path]) tuple, then
    return its base_url.

    Auto-creates the row keyed by model_name as label. If a row with the
    same label already exists, we reuse it (updating worker_ids /
    mmproj_path if the user added/removed workers or installed a new
    multimodal projector since the previous turn). If a DIFFERENT split
    row is currently running, we stop it first — only one big model
    hot at a time.

    `mmproj_path`, when non-None, is forwarded to llama-server's
    `--mmproj` flag so vision-capable models (e.g. gemma4:26b after
    its vision tower has been extracted into a separate GGUF) can
    accept image input via Phase 2 split.
    """
    # Local import to dodge the circular dep — split_lifecycle imports
    # compute_pool indirectly via db / runtime.
    from . import split_lifecycle

    rows = db.list_split_models()
    target_row = next((r for r in rows if r.get("label") == model_name), None)

    # Stop any other running split row — finite VRAM means one big
    # model active at a time.
    for r in rows:
        if r["id"] != (target_row or {}).get("id") and r.get("status") in ("running", "loading"):
            try:
                await split_lifecycle.stop(r["id"])
            except Exception as e:
                log.warning("compute_pool: failed to stop %s: %s", r["id"], e)

    if target_row is None:
        sid = db.create_split_model(
            label=model_name,
            gguf_path=gguf_path,
            mmproj_path=mmproj_path,
            worker_ids=worker_ids,
        )
        target_row = db.get_split_model(sid)
    else:
        # Refresh gguf_path / mmproj_path / worker_ids in case the
        # user changed things since the previous turn.
        if (
            target_row.get("gguf_path") != gguf_path
            or target_row.get("mmproj_path") != mmproj_path
            or target_row.get("worker_ids") != worker_ids
        ):
            db.update_split_model(
                target_row["id"],
                gguf_path=gguf_path,
                mmproj_path=mmproj_path,
                worker_ids=worker_ids,
            )
            target_row = db.get_split_model(target_row["id"])

    sid = target_row["id"]
    if target_row.get("status") != "running":
        result = await split_lifecycle.start(sid)
        if not result.get("ok"):
            raise RuntimeError(
                f"failed to start llama-server for {model_name}: "
                f"{result.get('error') or 'unknown'}"
            )

    fresh = db.get_split_model(sid)
    return f"http://127.0.0.1:{fresh['llama_port']}"


async def stop_all_running_splits() -> None:
    """Free VRAM held by any running llama-server. Called when the
    router decides the upcoming chat turn fits Ollama on host alone —
    no point keeping a big-model llama-server warm if the active
    conversation no longer needs it."""
    from . import split_lifecycle

    for r in db.list_split_models():
        if r.get("status") in ("running", "loading"):
            try:
                await split_lifecycle.stop(r["id"])
            except Exception as e:
                log.warning("compute_pool: stop_all_running_splits %s: %s", r["id"], e)


class RouteChatError(RuntimeError):
    """Raised by route_chat_for when the model can't be served at all
    — e.g. the model file isn't present, or the combined pool is too
    small to hold it. Caller should surface this to the user."""


async def route_chat_for(model_name: str) -> dict:
    """Pick the right inference engine for this model + drive any
    needed lifecycle changes.

    Returns one of:
      {"engine": "ollama"}
        → use the existing Ollama path. Downstream `pick_chat_target`
          decides whether the actual node is host or a more-powerful
          worker. Caller continues with `_stream_ollama_chat`.

      {"engine": "llama_server", "base_url": "http://127.0.0.1:NNNN", "label": <model>}
        → llama-server with --rpc was spawned (or already running) for
          this model. Caller uses `_stream_llama_server_chat` instead.

    Decision logic (intelligent — the strongest single node wins):
      * Compute the strongest single-node VRAM budget across host +
        every eligible worker. A worker's contribution is its
        `max_vram_seen_bytes` — only counts as "real" VRAM once the
        worker has actually loaded a model that big.
      * If the model fits the strongest single node → Ollama path.
        (`pick_chat_target` then routes to whichever node won.)
      * Else → split path: spawn llama-server with --rpc to every
        eligible worker so layers fan across host + every node's
        compute.

    Raises `RouteChatError` for unrecoverable cases. Side effects:
    may stop a currently-running llama-server (model switch) and may
    auto-create a `split_models` row.
    """
    # Cheap reject: explicit `split:` prefix is back-compat — short-
    # circuit to the legacy picker so existing tests still pass.
    legacy = pick_split_chat_target(model_name)
    if legacy is not None:
        base, label = legacy
        return {"engine": "llama_server", "base_url": base, "label": label}

    info = resolve_ollama_model(model_name)
    if info is None:
        # Not an Ollama-managed model. Could be a custom name the user
        # set up another way. Stay on the Ollama path — Ollama will
        # surface its own error if the model truly doesn't exist.
        await stop_all_running_splits()
        return {"engine": "ollama"}

    size_bytes = info["size_bytes"]

    # Three-tier decision, fastest path first:
    #   1. **Single-node VRAM fit** — strongest GPU (host or worker)
    #      can hold the whole model in pure VRAM. No CPU offload, no
    #      LAN. Always wins on per-token rate.
    #   2. **Pool VRAM fit (split)** — combined VRAM across host + every
    #      eligible worker covers the model. Engages llama-server
    #      with --rpc; layers ride GPU memory across machines via
    #      llama.cpp's native pipeline. Beats host CPU offload when the
    #      LAN is fast (Ethernet) and workers have real GPUs (dGPU).
    #      Loses on Wi-Fi + iGPU because per-token RPC overhead exceeds
    #      the win from avoiding CPU offload on host — but that's a
    #      real-world calibration the user can correct by setting
    #      `use_for_chat=False` on a slow worker.
    #   3. **Host VRAM + host RAM (CPU offload)** — single-node Ollama
    #      with layer-spill into RAM. Slower per layer than VRAM, but
    #      no LAN per-token cost. Last resort when split path can't
    #      cover the model either.
    host_vram = _host_vram_budget_bytes()
    host_total = _host_total_capacity_bytes()
    chat_workers = _eligible_workers("use_for_chat", model=model_name)
    best_worker_capacity = 0
    pool_vram_total = host_vram
    if chat_workers:
        worker_vrams = [
            (w.get("capabilities") or {}).get("max_vram_seen_bytes") or 0
            for w in chat_workers
        ]
        best_worker_capacity = max(worker_vrams) if worker_vrams else 0
        # Sum every eligible worker's proven VRAM with host's. Honest
        # under-count: max_vram_seen is a lower bound (it's whatever
        # was loaded so far, not the worker's true VRAM ceiling).
        pool_vram_total = host_vram + sum(worker_vrams)

    # Tier 1: fits the strongest single GPU (host or worker)?
    strongest_single_vram = max(host_vram, best_worker_capacity)
    if strongest_single_vram > 0 and size_bytes <= strongest_single_vram:
        await stop_all_running_splits()
        return {"engine": "ollama"}

    # Tier 2: engage Phase 2 split. Split-rpc-eligible workers
    # (rpc-server reachable) are what matters here — Phase 2 doesn't
    # need the model installed on the worker; layer weights stream
    # from host's local GGUF.
    #
    # When to engage split:
    #   * If model is so big the host CAN'T run it alone
    #     (size > host_total) → engage split regardless of estimated
    #     pool VRAM. Workers' `max_vram_seen_bytes` is a hard lower
    #     bound (often 0 for never-benchmarked workers), so the
    #     estimated combined VRAM is always an under-count. llama.cpp
    #     itself does the actual per-layer placement based on each
    #     rpc-server's real available memory; if it truly can't fit,
    #     llama-server reports a clean error — strictly better than
    #     "Ollama on host says no tokens" with no diagnostic.
    #   * If host CAN run alone but estimated pool VRAM covers the
    #     model AND the slowest worker LAN latency is < 150 ms,
    #     engage split anyway — combined GPU VRAM beats single-host
    #     CPU offload on a fast LAN. Above 150 ms latency we keep it
    #     local (LAN cost dominates the GPU win).
    rpc_workers = _eligible_split_workers()
    if rpc_workers:
        rpc_pool_vram = host_vram + sum(
            (w.get("capabilities") or {}).get("max_vram_seen_bytes") or 0
            for w in rpc_workers
        )
        host_can_run_alone = host_total > 0 and size_bytes <= host_total
        worst_lan_latency_ms = max(
            (w.get("capabilities") or {}).get("probe_latency_ms") or 0
            for w in rpc_workers
        )

        # Decision: engage split if host can't run alone (mandatory
        # — only path that might work) OR if pool VRAM covers it AND
        # LAN is fast enough.
        engage_split = (
            not host_can_run_alone
            or (rpc_pool_vram > 0 and size_bytes <= rpc_pool_vram and worst_lan_latency_ms <= 150)
        )

        if engage_split:
            worker_ids = [w["id"] for w in rpc_workers]
            try:
                base_url = await _ensure_split_running_for(
                    model_name,
                    info["gguf_path"],
                    worker_ids,
                    mmproj_path=info.get("mmproj_path"),
                )
                return {"engine": "llama_server", "base_url": base_url, "label": model_name}
            except Exception as e:
                log.warning("compute_pool: split start failed (%s); falling back to host", e)
                # fall through to host CPU offload

    # Tier 3: host CPU offload (still single-node, no LAN). If the
    # model fits host's total memory, Ollama will run it slowly but
    # correctly. Above the budget we fall through to the legacy split
    # attempt one more time — even though we already established the
    # split pool is too small, llama-server's per-layer placement may
    # squeeze it in (workers have unreported RAM headroom).
    if host_total > 0 and size_bytes <= host_total:
        await stop_all_running_splits()
        return {"engine": "ollama"}

    # Need the split path. Find rpc-eligible workers (not the same as
    # chat-eligible — split needs rpc-server, not whole-model-loading).
    workers = _eligible_split_workers()
    if not workers:
        # No rpc workers. Fall back to Ollama on host — its CPU offload
        # at least lets big models run slowly on host alone, which is
        # better than refusing.
        await stop_all_running_splits()
        return {"engine": "ollama"}

    worker_ids = [w["id"] for w in workers]
    base_url = await _ensure_split_running_for(
        model_name,
        info["gguf_path"],
        worker_ids,
        mmproj_path=info.get("mmproj_path"),
    )
    return {"engine": "llama_server", "base_url": base_url, "label": model_name}


def list_subagent_workers(model: str) -> list[tuple[str, str | None]]:
    """Return every eligible compute worker for parallel-subagent fan-out.

    The host itself is NOT in this list — the caller (`run_subagents_parallel`)
    composes `[host] + workers` and round-robins, so a 6-task fan-out across
    1 host + 2 workers schedules roughly 2 per machine.

    Performance gate (Phase 2 commit 18): a worker is included ONLY if
    its measured throughput is at least `_SUBAGENT_MIN_PERF_RATIO` of
    the host's measured throughput on this model. Without this gate,
    a worker that's 25× slower than host (e.g. a Core 3 100U laptop
    iGPU vs an RTX 3060 Ti) would still get round-robin-assigned tasks
    in `run_subagents_parallel` — and the slowest task bottlenecks the
    whole fan-out. Net result: parallel calls get SLOWER with the
    weak worker than without it. The gate keeps weak workers out of
    parallel jobs while still letting them serve embed routing and
    contribute layer storage to Phase 2 splits.

    When neither host nor worker has a measured TPS yet (cold cache),
    we fall back to including the worker — there's no signal to make
    a smarter call, and the next probe will fix it.
    """
    cands = _eligible_workers("use_for_subagents", model=model)
    if not cands:
        return []

    host_score = _host_capability_score(model)
    host_tps = host_score[0]

    if host_tps > 0:
        min_tps = host_tps * _SUBAGENT_MIN_PERF_RATIO
        gated = []
        for w in cands:
            w_tps = float((w.get("capabilities") or {}).get("tokens_per_second") or 0.0)
            if w_tps <= 0:
                # No measurement yet — include and let the probe sort
                # this out next sweep. Excluding unknown workers would
                # never let a fresh worker bootstrap into the pool.
                gated.append(w)
            elif w_tps >= min_tps:
                gated.append(w)
            else:
                log.info(
                    "subagent fan-out: skipping %r (tps=%.1f) — below %.0f%% of host's %.1f tps",
                    w.get("label"), w_tps,
                    _SUBAGENT_MIN_PERF_RATIO * 100, host_tps,
                )
        cands = gated

    return [
        (_worker_base_url(w), db.get_compute_worker_auth_token(w["id"]))
        for w in cands
    ]
