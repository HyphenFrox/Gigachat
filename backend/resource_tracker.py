"""Per-device + pool-wide resource tracker.

Captures live CPU / RAM / GPU / VRAM / disk / network usage for the
local install, **plus** the slice of those resources currently held by
the Gigachat process tree (python.exe + llama-server.exe + rpc-server.exe
+ ollama.exe + ollama runners). The host can fan out the same probe to
every paired peer over the encrypted P2P channel and aggregate the
results into a single snapshot.

This module is the source of truth for the user's compute-pool directive:
> The app must intelligently and automatically use ALL resources from
> ALL devices on the compute pool to run as fast as possible AND adapt
> to changes in available resources in realtime, while staying within
> min and max usage limits. Visible 95 % memory usage on every
> participating device under load is the success criterion.

Design notes:
- All disk I/O / nvidia-smi calls are short-timeout & best-effort. A
  missing tool returns zero so the snapshot still goes out — the
  orchestrator must make decisions even when one device is partially
  blind.
- A background `_BgSampler` thread keeps `cpu_pct` and net I/O rate
  always-fresh so the per-call latency stays near zero. Without it
  `psutil.cpu_percent(interval=0.0)` always returns 0.0 for cold calls
  and net I/O has no rate, only cumulative counters.
- Aggregation deliberately runs the per-peer fan-out in parallel via
  `concurrent.futures.ThreadPoolExecutor` so a slow peer can't stall
  the dashboard.
- The Gigachat process detector is conservative on purpose: anything
  ambiguous (e.g. a stray python.exe outside the venv) is excluded so
  the "app portion" never *over*-reports.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil is a hard dep
    psutil = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background sampler - keeps cpu_pct & net I/O rate fresh without making the
# caller pay an interval-blocked psutil.cpu_percent() on every request.
# ---------------------------------------------------------------------------
class _BgSampler:
    """Singleton thread that snapshots CPU% + net counters + GPU stats
    every `tick` seconds. Holds the latest values for cheap reads from
    the request path.

    Why a background thread: the GPU probes (nvidia-smi / rocm-smi /
    PowerShell Get-CimInstance for Intel) are subprocess shell-outs
    that can take 100 ms - 4 s each. The first iteration of /api/p2p/
    system-stats discovered that running them inline jammed the AnyIO
    worker pool: every queued request waited on the slow subprocess.
    Now they run on this dedicated daemon thread; the endpoint just
    reads the latest cached snapshot - request latency is ~1 ms.
    """

    _instance: "_BgSampler | None" = None
    _lock = threading.Lock()

    def __init__(self, tick: float = 3.0, gpu_tick: float = 8.0) -> None:
        self._tick = max(1.0, float(tick))
        self._gpu_tick = max(2.0, float(gpu_tick))
        self._cpu_pct: float = 0.0
        self._net_send_bps: float = 0.0  # bytes/sec
        self._net_recv_bps: float = 0.0
        self._disk_read_bps: float = 0.0
        self._disk_write_bps: float = 0.0
        self._last_net_counters: tuple[int, int] | None = None
        self._last_disk_counters: tuple[int, int] | None = None
        self._last_sample_ts: float = 0.0
        # GPU snapshot, polled less frequently. None means "not yet probed";
        # endpoint falls back to a zeroed shape until first probe lands.
        self._gpu_snap: dict[str, Any] | None = None
        self._last_gpu_ts: float = 0.0
        # Per-pid GPU usage (cached on the same cadence as the device-
        # level probe — both shell out to the same vendor tools).
        self._per_pid_gpu_snap: dict[int, dict[str, Any]] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="ResourceTrackerBgSampler", daemon=True
        )

    @classmethod
    def get(cls) -> "_BgSampler":
        with cls._lock:
            if cls._instance is None:
                inst = cls()
                inst._thread.start()
                cls._instance = inst
            return cls._instance

    def _run(self) -> None:
        if psutil is None:
            return
        # Prime cpu_percent so the first reading is meaningful.
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass
        # Prime GPU snapshot in the bg so the first request has data.
        try:
            self._gpu_snap = _probe_gpu()
            self._per_pid_gpu_snap = _probe_per_pid_gpu()
            self._last_gpu_ts = time.time()
        except Exception as e:
            log.debug("BgSampler initial gpu probe failed: %s", e)
        while not self._stop.wait(self._tick):
            try:
                # cpu_percent without interval reports the % since the
                # previous call - matches our tick window exactly.
                self._cpu_pct = float(psutil.cpu_percent(interval=None))
                # net I/O delta -> bytes/sec
                now = time.time()
                io = psutil.net_io_counters()
                cur = (int(io.bytes_sent), int(io.bytes_recv))
                if self._last_net_counters is not None and self._last_sample_ts > 0:
                    dt = max(0.001, now - self._last_sample_ts)
                    self._net_send_bps = max(
                        0.0, (cur[0] - self._last_net_counters[0]) / dt
                    )
                    self._net_recv_bps = max(
                        0.0, (cur[1] - self._last_net_counters[1]) / dt
                    )
                self._last_net_counters = cur
                # disk I/O delta -> bytes/sec (system-wide)
                try:
                    dio = psutil.disk_io_counters()
                    if dio is not None:
                        dcur = (int(dio.read_bytes), int(dio.write_bytes))
                        if (
                            self._last_disk_counters is not None
                            and self._last_sample_ts > 0
                        ):
                            dt = max(0.001, now - self._last_sample_ts)
                            self._disk_read_bps = max(
                                0.0, (dcur[0] - self._last_disk_counters[0]) / dt
                            )
                            self._disk_write_bps = max(
                                0.0, (dcur[1] - self._last_disk_counters[1]) / dt
                            )
                        self._last_disk_counters = dcur
                except Exception:
                    pass
                self._last_sample_ts = now
                # GPU snapshot — slower cadence, separate call window.
                if (now - self._last_gpu_ts) >= self._gpu_tick:
                    try:
                        self._gpu_snap = _probe_gpu()
                    except Exception as e:
                        log.debug("BgSampler gpu probe failed: %s", e)
                    try:
                        self._per_pid_gpu_snap = _probe_per_pid_gpu()
                    except Exception as e:
                        log.debug("BgSampler per-pid gpu probe failed: %s", e)
                    self._last_gpu_ts = now
            except Exception as e:
                log.debug("BgSampler tick failed: %s", e)

    @property
    def per_pid_gpu_snap(self) -> dict[int, dict[str, Any]]:
        return dict(self._per_pid_gpu_snap)

    @property
    def gpu_snap(self) -> dict[str, Any]:
        if self._gpu_snap is not None:
            return dict(self._gpu_snap)
        return {
            "gpu_kind": "",
            "gpu_name": "",
            "vram_total_gb": 0.0,
            "vram_used_gb": 0.0,
            "vram_used_pct": 0.0,
            "gpu_util_pct": 0.0,
        }

    @property
    def cpu_pct(self) -> float:
        return self._cpu_pct

    @property
    def net_send_kbps(self) -> float:
        return round(self._net_send_bps / 1024.0, 1)

    @property
    def net_recv_kbps(self) -> float:
        return round(self._net_recv_bps / 1024.0, 1)

    @property
    def disk_read_kbps(self) -> float:
        return round(self._disk_read_bps / 1024.0, 1)

    @property
    def disk_write_kbps(self) -> float:
        return round(self._disk_write_bps / 1024.0, 1)


# ---------------------------------------------------------------------------
# GPU probes - each one is best-effort. Returns dict so callers can
# merge without checking which backend was used.
# ---------------------------------------------------------------------------
def _probe_nvidia() -> dict[str, Any] | None:
    """Live VRAM used + GPU util via `nvidia-smi`. Returns None if no NVIDIA."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,utilization.gpu,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    line = (result.stdout or "").strip().splitlines()
    if not line:
        return None
    parts = [p.strip() for p in line[0].split(",")]
    if len(parts) < 3 or not parts[0].isdigit():
        return None
    vram_total_mb = int(parts[0])
    vram_used_mb = int(parts[1]) if parts[1].isdigit() else 0
    gpu_util = float(parts[2]) if parts[2].replace(".", "", 1).isdigit() else 0.0
    name = parts[3] if len(parts) > 3 else "NVIDIA GPU"
    return {
        "gpu_kind": "nvidia",
        "gpu_name": name,
        "vram_total_gb": round(vram_total_mb / 1024.0, 2),
        "vram_used_gb": round(vram_used_mb / 1024.0, 2),
        "vram_used_pct": round(100.0 * vram_used_mb / max(1, vram_total_mb), 1),
        "gpu_util_pct": round(gpu_util, 1),
    }


def _probe_amd() -> dict[str, Any] | None:
    """Live VRAM used via rocm-smi. Returns None if no AMD GPU."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--showuse", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    out = result.stdout or ""
    vram_total = 0
    vram_used = 0
    util = 0.0
    name = ""
    for line in out.splitlines():
        low = line.lower()
        if "vram total memory" in low:
            tail = line.rsplit(":", 1)[-1].strip()
            if tail.isdigit():
                vram_total = max(vram_total, int(tail))
        elif "vram total used memory" in low:
            tail = line.rsplit(":", 1)[-1].strip()
            if tail.isdigit():
                vram_used = max(vram_used, int(tail))
        elif "gpu use" in low:
            tail = line.rsplit(":", 1)[-1].strip().rstrip("%").strip()
            try:
                util = max(util, float(tail))
            except ValueError:
                pass
        elif "card series" in low or "card model" in low:
            if not name:
                name = line.rsplit(":", 1)[-1].strip()
    if vram_total <= 0:
        return None
    return {
        "gpu_kind": "amd",
        "gpu_name": name or "AMD GPU",
        "vram_total_gb": round(vram_total / (1024 ** 3), 2),
        "vram_used_gb": round(vram_used / (1024 ** 3), 2),
        "vram_used_pct": round(100.0 * vram_used / max(1, vram_total), 1),
        "gpu_util_pct": round(util, 1),
    }


# Cache the most expensive Intel probes. The total-VRAM probe stays
# valid for the lifetime of the process (hardware doesn't change), the
# util probe expires after 30 s. Either way the BG sampler refreshes
# at its own cadence, so cached values may be slightly stale during
# rapid GPU-utilisation spikes — acceptable, since the orchestrator
# uses these as a coarse load signal not a fine-grained measurement.
_INTEL_TOTAL_CACHE: dict[str, Any] = {"vram_total_bytes": None, "name": None}
_INTEL_UTIL_CACHE: dict[str, Any] = {"ts": 0.0, "value": 0.0}


def _powershell_run(cmd: str, timeout: float) -> str:
    """Run a single PowerShell command line, returning stdout (stripped)
    or empty string on any failure / timeout. Uses ``-NonInteractive``
    + ``-NoProfile`` so it can't hang on user-profile prompts.
    """
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return (result.stdout or "").strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return ""


def _probe_intel() -> dict[str, Any] | None:
    """Live Intel iGPU stats on Windows. Total VRAM cached for the
    process lifetime (hardware doesn't change); util cached 30 s.

    Both probes are timeout-bounded subprocess calls. Designed to be
    called only from the BG sampler thread, NEVER from a request
    handler — even with timeouts a hung PowerShell can still wait
    multiple seconds for the kernel to deliver SIGTERM.
    """
    if sys.platform != "win32":
        return None
    # ---- VRAM total — probed once per process ------------------------
    if _INTEL_TOTAL_CACHE["vram_total_bytes"] is None:
        line = _powershell_run(
            "Get-CimInstance Win32_VideoController "
            "| Where-Object { $_.Name -match 'Intel' } "
            "| Select-Object -First 1 "
            "| ForEach-Object { \"$($_.AdapterRAM)|$($_.Name)\" }",
            timeout=3.0,
        )
        vram_total_bytes = 0
        name = "Intel GPU"
        if line:
            parts = line.splitlines()[0].split("|", 1)
            if parts and parts[0].strip().isdigit():
                vram_total_bytes = int(parts[0].strip())
            if len(parts) > 1 and parts[1].strip():
                name = parts[1].strip()
        _INTEL_TOTAL_CACHE["vram_total_bytes"] = vram_total_bytes
        _INTEL_TOTAL_CACHE["name"] = name
    vram_total_bytes = _INTEL_TOTAL_CACHE["vram_total_bytes"] or 0
    name = _INTEL_TOTAL_CACHE["name"] or "Intel GPU"
    if vram_total_bytes <= 0:
        return None
    # ---- GPU util — refresh at most every 30 s -----------------------
    now = time.time()
    if (now - _INTEL_UTIL_CACHE["ts"]) >= 30.0:
        s = _powershell_run(
            "(Get-Counter '\\GPU Engine(*engtype_3D)\\Utilization Percentage' "
            "-ErrorAction SilentlyContinue).CounterSamples "
            "| Measure-Object -Property CookedValue -Sum "
            "| ForEach-Object { $_.Sum }",
            timeout=3.0,
        )
        try:
            _INTEL_UTIL_CACHE["value"] = float(s) if s else 0.0
        except ValueError:
            _INTEL_UTIL_CACHE["value"] = 0.0
        _INTEL_UTIL_CACHE["ts"] = now
    gpu_util = _INTEL_UTIL_CACHE["value"]
    return {
        "gpu_kind": "intel",
        "gpu_name": name,
        "vram_total_gb": round(vram_total_bytes / (1024 ** 3), 2),
        # Intel iGPU shared with system RAM: we don't have a free,
        # zero-dep way to read "VRAM used" - leave as 0 rather than
        # invent a number. The orchestrator already treats Intel
        # devices as RAM-bounded.
        "vram_used_gb": 0.0,
        "vram_used_pct": 0.0,
        "gpu_util_pct": round(min(100.0, max(0.0, gpu_util)), 1),
    }


def _probe_gpu() -> dict[str, Any]:
    """Try NVIDIA -> AMD -> Intel -> none. First non-None wins."""
    for probe in (_probe_nvidia, _probe_amd, _probe_intel):
        try:
            r = probe()
        except Exception as e:
            log.debug("GPU probe %s failed: %s", probe.__name__, e)
            r = None
        if r is not None:
            return r
    return {
        "gpu_kind": "",
        "gpu_name": "",
        "vram_total_gb": 0.0,
        "vram_used_gb": 0.0,
        "vram_used_pct": 0.0,
        "gpu_util_pct": 0.0,
    }


# ---------------------------------------------------------------------------
# Gigachat process detector - sums RAM + CPU% across the install's tree.
# ---------------------------------------------------------------------------
# Process-name patterns that always belong to Gigachat. Lower-cased for
# match. `python.exe` is excluded from the name-only set because plenty
# of unrelated python processes can run on a dev box - we additionally
# require the cmdline to point at our backend module.
_GIGACHAT_BIN_NAMES = frozenset({
    "llama-server.exe", "llama-server",
    "rpc-server.exe", "rpc-server",
    "ollama.exe", "ollama",
    "ollama_llama_server.exe", "ollama_llama_server",
    "ollama runner.exe",
})

# Substrings we look for in a python.exe cmdline to claim it as ours.
_GIGACHAT_PY_MARKERS = (
    "backend.app",
    "backend\\app.py",
    "backend/app.py",
    "gigachat",
    "uvicorn backend",
)


def _is_gigachat_process(p: "psutil.Process") -> bool:
    try:
        name = (p.info.get("name") or "").lower()
    except Exception:
        return False
    if name in _GIGACHAT_BIN_NAMES:
        return True
    if name in ("python.exe", "python", "python3", "python3.exe"):
        try:
            cmd = " ".join(p.info.get("cmdline") or []).lower()
        except Exception:
            return False
        return any(marker in cmd for marker in _GIGACHAT_PY_MARKERS)
    return False


_PYTHON_BIN_NAMES = frozenset({"python.exe", "python", "python3", "python3.exe"})


def _safe_cmdline(p: "psutil.Process", timeout: float = 0.4) -> list[str] | None:
    """Fetch ``p.cmdline()`` with a timeout. Returns None on
    NoSuchProcess / AccessDenied / timeout / unreadable PEB.

    Why the timeout: on Windows, hung / unkillable / ``deletePending``
    processes can make ``psutil.Process.cmdline()`` block in the kernel
    PEB read for an unbounded time, which then jams the AnyIO worker
    thread that's running the FastAPI sync endpoint - ALL subsequent
    HTTP requests stall behind it. This made the resource-tracker
    snapshot freeze the whole backend the first time we shipped it.
    """
    result: list[list[str] | None] = [None]

    def _runner() -> None:
        try:
            result[0] = p.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            result[0] = None
        except Exception:
            result[0] = None

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        # Hung in the kernel - leave the daemon thread dangling; it will
        # die when the process exits. Returning None lets us skip this pid.
        return None
    return result[0]


def _app_process_snapshot() -> dict[str, Any]:
    """Sum CPU + RAM across processes that belong to the Gigachat install.

    Implementation notes:
      - We iterate ``psutil.pids()`` and read each pid's ``name()`` cheaply.
      - ``cmdline()`` (the slow, hang-prone call on Windows) is ONLY
        invoked for ``python.exe`` candidates, and behind a timeout
        (see ``_safe_cmdline``). All other process types are claimed
        / dismissed by name alone.
      - First call to ``cpu_percent(interval=None)`` for a fresh pid
        returns 0 (psutil contract); subsequent calls report a real
        delta, so the dashboard's first tick under-reports CPU% and
        every tick afterwards is accurate.
    """
    if psutil is None:
        return {"processes": 0, "cpu_pct": 0.0, "ram_used_gb": 0.0, "names": []}

    matched: list[psutil.Process] = []
    try:
        pids = psutil.pids()
    except Exception as e:
        log.debug("psutil.pids() failed: %s", e)
        return {"processes": 0, "cpu_pct": 0.0, "ram_used_gb": 0.0, "names": []}

    for pid in pids:
        try:
            p = psutil.Process(pid)
            name = (p.name() or "").lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except Exception:
            continue
        # Cheap path: known-Gigachat binary name.
        if name in _GIGACHAT_BIN_NAMES:
            matched.append(p)
            continue
        # Slow path: python.exe — claim only if cmdline mentions us.
        if name in _PYTHON_BIN_NAMES:
            cmd_list = _safe_cmdline(p)
            if cmd_list is None:
                continue
            cmd = " ".join(cmd_list).lower()
            if any(marker in cmd for marker in _GIGACHAT_PY_MARKERS):
                matched.append(p)

    total_cpu = 0.0
    total_rss = 0
    names_seen: dict[str, int] = {}
    for p in matched:
        try:
            total_cpu += float(p.cpu_percent(interval=None))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except Exception:
            pass
        try:
            total_rss += int(p.memory_info().rss)
        except Exception:
            pass
        try:
            n = (p.name() or "").lower()
            names_seen[n] = names_seen.get(n, 0) + 1
        except Exception:
            pass

    cpu_count = max(1, psutil.cpu_count(logical=True) or 1)

    # Per-app GPU usage. Cross-vendor: nvidia-smi gives us per-pid
    # VRAM + util on NVIDIA; Windows perf-counter `GPU Process Memory`
    # gives per-pid VRAM on Intel iGPU; AMD has no portable per-pid
    # query so we leave it 0. Only sum entries whose pid is in our
    # `matched` set so other apps' GPU usage doesn't get attributed
    # to Gigachat.
    matched_pids = {int(p.pid) for p in matched}
    app_vram_used_gb = 0.0
    app_gpu_util_pct = 0.0
    try:
        gpu_per_pid = _per_pid_gpu_snapshot()
        for pid, info in gpu_per_pid.items():
            if pid in matched_pids:
                app_vram_used_gb += float(info.get("vram_used_gb") or 0)
                app_gpu_util_pct += float(info.get("gpu_util_pct") or 0)
    except Exception as e:
        log.debug("per-pid GPU snapshot failed: %s", e)

    # Disk I/O per process via psutil (Windows: requires admin for
    # OTHER processes, but processes WE spawned report fine because
    # they share our token).
    app_disk_read_bytes = 0
    app_disk_write_bytes = 0
    for p in matched:
        try:
            io = p.io_counters()
            app_disk_read_bytes += int(io.read_bytes)
            app_disk_write_bytes += int(io.write_bytes)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except Exception:
            pass

    return {
        "processes": len(matched),
        "cpu_pct": round(total_cpu / cpu_count, 1),
        "ram_used_gb": round(total_rss / (1024 ** 3), 2),
        "vram_used_gb": round(app_vram_used_gb, 2),
        "gpu_util_pct": round(min(100.0, app_gpu_util_pct), 1),
        "disk_read_total_gb": round(app_disk_read_bytes / (1024 ** 3), 2),
        "disk_write_total_gb": round(app_disk_write_bytes / (1024 ** 3), 2),
        "names": [f"{k}x{v}" if v > 1 else k for k, v in sorted(names_seen.items())],
    }


def _per_pid_gpu_snapshot() -> dict[int, dict[str, Any]]:
    """Map of {pid: {vram_used_gb, gpu_util_pct}} for processes with
    GPU activity. Aggregates across NVIDIA + Intel sources; AMD has
    no portable per-pid path so its processes are absent (treated as
    zero by the caller). All probes are cached briefly inside the
    BG sampler so the request handler doesn't pay the subprocess
    cost on every snapshot.
    """
    sampler = _BgSampler.get()
    return sampler.per_pid_gpu_snap


def _probe_per_pid_gpu_nvidia() -> dict[int, dict[str, Any]]:
    """nvidia-smi --query-compute-apps=pid,used_memory — pid + VRAM
    in MB. Util is per-device only on NVIDIA, not per-pid; we
    distribute the device's util pro-rata to each pid by VRAM
    share so a Gigachat process that's holding 6 of 8 GB of
    weights gets 75 % of the device util.
    """
    out: dict[int, dict[str, Any]] = {}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=3.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return out
    if result.returncode != 0:
        return out
    apps: list[tuple[int, float]] = []
    for line in (result.stdout or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        apps.append((int(parts[0]), int(parts[1]) / 1024.0))
    if not apps:
        return out
    # Device util — for distribution.
    device_util = 0.0
    try:
        r2 = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0,
        )
        if r2.returncode == 0:
            line = (r2.stdout or "").strip().splitlines()
            if line and line[0].strip().replace(".", "", 1).isdigit():
                device_util = float(line[0].strip())
    except Exception:
        pass
    total_vram = sum(v for _, v in apps) or 1.0
    for pid, vram_gb in apps:
        share = vram_gb / total_vram
        out[pid] = {
            "vram_used_gb": vram_gb,
            "gpu_util_pct": device_util * share,
        }
    return out


def _probe_per_pid_gpu_intel() -> dict[int, dict[str, Any]]:
    """Windows-only. Reads `\\GPU Process Memory(pid_*)\\Local Usage`
    perf counters which Task Manager itself uses. The `pid_<N>` part
    of the instance name carries the pid, so we map back to our
    matched processes. Intel iGPU has no per-pid util signal — we
    leave gpu_util_pct as 0 here and rely on the device-level total.
    """
    out: dict[int, dict[str, Any]] = {}
    if sys.platform != "win32":
        return out
    cmd = (
        "(Get-Counter '\\GPU Process Memory(*)\\Local Usage' "
        "-ErrorAction SilentlyContinue).CounterSamples "
        "| ForEach-Object { \"$($_.InstanceName)|$($_.CookedValue)\" }"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
            capture_output=True, text=True, timeout=4.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return out
    if result.returncode != 0:
        return out
    # Aggregate across all GPU engines per pid.
    per_pid_bytes: dict[int, float] = {}
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 1)
        if len(parts) != 2:
            continue
        instance = parts[0]
        # "pid_12345_luid_..._phys_0_eng_0_engtype_3D"
        if "pid_" not in instance:
            continue
        try:
            pid_str = instance.split("pid_", 1)[1].split("_", 1)[0]
            pid = int(pid_str)
            val = float(parts[1])
        except (ValueError, IndexError):
            continue
        per_pid_bytes[pid] = per_pid_bytes.get(pid, 0.0) + val
    for pid, b in per_pid_bytes.items():
        if b <= 0:
            continue
        out[pid] = {
            "vram_used_gb": b / (1024 ** 3),
            "gpu_util_pct": 0.0,
        }
    return out


def _probe_per_pid_gpu() -> dict[int, dict[str, Any]]:
    """Merge per-pid GPU snapshots from every available vendor probe."""
    merged: dict[int, dict[str, Any]] = {}
    for probe in (_probe_per_pid_gpu_nvidia, _probe_per_pid_gpu_intel):
        try:
            for pid, info in probe().items():
                if pid not in merged:
                    merged[pid] = dict(info)
                else:
                    merged[pid]["vram_used_gb"] = (
                        merged[pid].get("vram_used_gb", 0.0)
                        + info.get("vram_used_gb", 0.0)
                    )
                    merged[pid]["gpu_util_pct"] = max(
                        merged[pid].get("gpu_util_pct", 0.0),
                        info.get("gpu_util_pct", 0.0),
                    )
        except Exception:
            continue
    return merged


# ---------------------------------------------------------------------------
# Top-level local snapshot
# ---------------------------------------------------------------------------
def local_snapshot() -> dict[str, Any]:
    """Full per-device resource snapshot. Backward-compatible: every
    field that the *old* /api/p2p/system-stats endpoint returned is still
    present at the same key, so old orchestrators keep working.
    """
    sampler = _BgSampler.get()
    snap: dict[str, Any] = {
        "ts": time.time(),
        "schema": 2,  # bump if we ever change the field shapes
    }
    # ---- RAM (matches old endpoint) -------------------------------------
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            snap["ram_total_gb"] = round(vm.total / (1024 ** 3), 2)
            snap["ram_free_gb"] = round(vm.available / (1024 ** 3), 2)
            snap["ram_used_gb"] = round((vm.total - vm.available) / (1024 ** 3), 2)
            snap["ram_used_pct"] = round(vm.percent, 1)
        except Exception as e:
            snap["ram_error"] = f"{type(e).__name__}: {e}"
    # ---- CPU ------------------------------------------------------------
    snap["cpu_pct"] = round(sampler.cpu_pct, 1)
    if psutil is not None:
        try:
            snap["cpu_count_logical"] = int(psutil.cpu_count(logical=True) or 0)
            snap["cpu_count_physical"] = int(psutil.cpu_count(logical=False) or 0)
        except Exception:
            pass
    # ---- GPU / VRAM (from BG sampler cache) -----------------------------
    snap.update(sampler.gpu_snap)
    # ---- Disk (the volume holding the user profile / models) -----------
    try:
        home_drive = os.path.expanduser("~")
        usage = psutil.disk_usage(home_drive) if psutil is not None else None
        if usage is not None:
            snap["disk_total_gb"] = round(usage.total / (1024 ** 3), 1)
            snap["disk_free_gb"] = round(usage.free / (1024 ** 3), 1)
            snap["disk_used_gb"] = round(usage.used / (1024 ** 3), 1)
            snap["disk_used_pct"] = round(usage.percent, 1)
    except Exception:
        pass
    # ---- Disk I/O rate (system-wide, from background sampler) ----------
    snap["disk_read_kbps"] = sampler.disk_read_kbps
    snap["disk_write_kbps"] = sampler.disk_write_kbps
    # ---- Network rate (system-wide, from background sampler) -----------
    snap["net_send_kbps"] = sampler.net_send_kbps
    snap["net_recv_kbps"] = sampler.net_recv_kbps
    # ---- Gigachat-app share --------------------------------------------
    snap["app"] = _app_process_snapshot()
    return snap


# ---------------------------------------------------------------------------
# Pool aggregator - runs on the orchestrator host. Calls local_snapshot()
# for "self" and fans out /api/p2p/system-stats to every paired peer.
# ---------------------------------------------------------------------------
def aggregate_snapshot(timeout_per_peer: float = 8.0) -> dict[str, Any]:
    """Aggregate this device + every enabled compute worker (= every
    paired peer that's wired up to contribute) into a single snapshot.

    Parallel fan-out so a slow peer doesn't stall the dashboard. A peer
    that fails the probe still appears in the result with ``error``
    set so the user can see WHY a device isn't contributing — that's
    the whole point of the tracker.
    """
    import asyncio

    from . import db as _db
    from . import identity as _ident
    from . import p2p_secure_client as _sec

    devices: list[dict[str, Any]] = []
    # ---- Local device first --------------------------------------------
    try:
        me = _ident.get_identity()
        local_label = getattr(me, "label", None) or "this device"
        local_did = getattr(me, "device_id", "") or ""
    except Exception:
        local_label, local_did = "this device", ""
    try:
        local_snap = local_snapshot()
        local_snap.update({
            "kind": "self",
            "device_id": local_did,
            "label": local_label,
            "address": "127.0.0.1",
        })
        devices.append(local_snap)
    except Exception as e:
        devices.append({
            "kind": "self",
            "device_id": local_did,
            "label": local_label,
            "error": f"{type(e).__name__}: {e}",
        })
    # ---- Compute workers in parallel -----------------------------------
    try:
        workers = _db.list_compute_workers(enabled_only=False) or []
    except Exception as e:
        log.debug("list_compute_workers failed: %s", e)
        workers = []

    def _probe_worker(worker: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": "peer",
            "device_id": worker.get("gigachat_device_id"),
            "label": worker.get("label") or worker.get("gigachat_device_id", ""),
            "address": worker.get("address"),
            "enabled": bool(worker.get("enabled")),
        }
        if not worker.get("enabled"):
            out["error"] = "worker disabled"
            return out
        # `forward` is async — run it in a fresh loop on this worker thread.
        try:
            status, body_text = asyncio.run(_sec.forward(
                worker,
                method="GET",
                path="/api/p2p/system-stats",
                body=None,
                timeout=timeout_per_peer,
            ))
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
            return out
        if int(status) >= 400:
            out["error"] = f"HTTP {status}: {body_text[:200]}"
            return out
        try:
            data = json.loads(body_text)
        except Exception:
            out["error"] = f"bad JSON from peer: {body_text[:200]}"
            return out
        if isinstance(data, dict):
            out.update(data)
        else:
            out["error"] = f"non-dict response: {data!r}"
        return out

    if workers:
        ex = ThreadPoolExecutor(max_workers=min(8, len(workers)))
        try:
            futures = {ex.submit(_probe_worker, w): w for w in workers}
            deadline = time.time() + timeout_per_peer + 4.0
            for fut, w in futures.items():
                remaining = max(0.5, deadline - time.time())
                try:
                    devices.append(fut.result(timeout=remaining))
                except Exception as e:
                    devices.append({
                        "kind": "peer",
                        "device_id": w.get("gigachat_device_id"),
                        "label": w.get("label"),
                        "error": f"{type(e).__name__}: {e}",
                    })
        finally:
            # Don't wait for workers — a hung peer probe shouldn't
            # block the response. Daemon threads die at process exit.
            ex.shutdown(wait=False, cancel_futures=True)

    # ---- Roll-up totals -------------------------------------------------
    rollup = {
        "device_count": len(devices),
        "reachable_count": sum(1 for d in devices if "error" not in d),
        "ram_total_gb": round(sum(d.get("ram_total_gb", 0) or 0 for d in devices), 2),
        "ram_used_gb": round(sum(d.get("ram_used_gb", 0) or 0 for d in devices), 2),
        "vram_total_gb": round(sum(d.get("vram_total_gb", 0) or 0 for d in devices), 2),
        "vram_used_gb": round(sum(d.get("vram_used_gb", 0) or 0 for d in devices), 2),
        "app_ram_used_gb": round(
            sum((d.get("app") or {}).get("ram_used_gb", 0) or 0 for d in devices), 2
        ),
        "app_processes": sum(
            (d.get("app") or {}).get("processes", 0) or 0 for d in devices
        ),
    }
    if rollup["ram_total_gb"] > 0:
        rollup["ram_used_pct_pool"] = round(
            100.0 * rollup["ram_used_gb"] / rollup["ram_total_gb"], 1
        )
    return {
        "ts": time.time(),
        "rollup": rollup,
        "devices": devices,
    }


def warm_up() -> None:
    """Start the background sampler so the first /system-stats call has
    real data. Safe to call multiple times - the singleton guards init.
    """
    _BgSampler.get()
