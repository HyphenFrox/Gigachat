"""Split-model lifecycle: start / stop the local `llama-server` process.

For each registered `split_models` row (commit 1 schema), this module
brings up a single `llama-server.exe` instance on the host with:

  llama-server
    --model     <gguf_path>
    --host      127.0.0.1                 (loopback only — the routing
                                           layer runs in-process here;
                                           no need to expose to LAN)
    --port      <split_models.llama_port>
    --rpc       <worker1_addr>:<port>,<worker2_addr>:<port>,...
    -ngl        <gpu_layers>               (default tuned per row;
                                            commit 4 hard-codes 99 to
                                            "as many as fit" — schema
                                            knob lands later)
    --jinja                                (use the GGUF's embedded
                                            chat template if present)

Architecture notes:

* **Resource pooling is native to llama.cpp.** With `-ngl N`, N layers
  go to GPUs in the pool (host VRAM via CUDA + each worker's GPU via
  Vulkan via rpc-server). Layers that don't fit GPUs cascade to the
  orchestrator's CPU+RAM and to each rpc-server's CPU+RAM. All four
  resource tiers (host VRAM, host RAM, worker GPU, worker RAM) get
  used; no resource is wasted. This module just supplies the flags;
  llama-server figures out per-layer placement.

* **One llama-server per split_model row.** Two rows can run
  concurrently on different `llama_port`s (default 11500, 11501, …).
  The chat router (commit 5) picks which one based on the conversation's
  active model.

* **Process supervision is intentionally minimal.** If llama-server
  exits, we record `status=error` with stderr tail and stop. We do
  NOT restart automatically — most failures are misconfiguration
  (wrong GGUF path, port already bound, GGUF newer than this
  llama.cpp build) and a silent retry-loop would just spam logs.
  Caller restarts manually after fixing the config.

* **In-process registry.** We track running processes in a
  module-level dict keyed by split_model_id. The DB status reflects
  the same state but the dict holds the Popen handle for stop().
  The dict survives the app process; DB status survives across
  restarts. On boot, anything DB-marked `running` is cross-checked
  against the OS — if no process exists, status is reset to `stopped`
  (commit 7 polish).

* **Log files** land in `~/.gigachat/llama-cpp/logs/<split_id>.log`
  for stdout+stderr merged, so the Settings UI can tail them.

Test surface kept tight: this module's pure-Python helpers
(command-building, log-rotation paths) are tested directly; the
actual `subprocess.Popen` spawn is exercised through stubbing in
test_split_lifecycle.py — we don't try to integration-test against
a real llama-server binary in CI.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from . import db, split_runtime

log = logging.getLogger(__name__)


# Default GPU-offload setting: 99 means "offload as many layers as
# possible to GPU memory" — llama.cpp clamps to the model's actual
# layer count and falls through to CPU+RAM for layers that don't fit.
# The user can override per row once the schema gains an `ngl` field
# (deferred — most users won't tune this manually).
_DEFAULT_NGL = 99

# How long we wait for `llama-server` to come up before declaring the
# start a failure. Loading large models is slow:
#   * Reading GGUF from disk: ~10 s for a 17 GB model on NVMe.
#   * Pushing layer weights to host VRAM via CUDA: ~5–10 s.
#   * Streaming layer weights to each worker's rpc-server over LAN:
#     dominated by LAN bandwidth — at 100 Mb/s Wi-Fi, 10 GB takes
#     ~14 minutes; at 1 Gb/s Ethernet, ~80 s. We size the timeout for
#     the median home-LAN case (5 GHz Wi-Fi, ~30–50 MB/s effective):
#     a 10 GB push lands in ~3–5 minutes.
# 600 s (10 min) is conservative enough that any reasonable hardware
# combination gets a fair shot, and short enough that a permanently-
# wedged llama-server (port-bound, OOM, GGUF mismatch) doesn't make
# the user wait forever.
_BOOT_TIMEOUT_SEC = 600.0

# Polling cadence for the readiness check. llama-server's `/health`
# becomes available the moment the HTTP server binds; we poll often
# enough to react fast but not so often that we spin during the
# (potentially) tens of seconds of model loading.
_HEALTH_POLL_SEC = 0.5


# ---------------------------------------------------------------------------
# Process registry — running llama-server instances keyed by split_model id
# ---------------------------------------------------------------------------

@dataclass
class _RunningProcess:
    """Bookkeeping for one live llama-server child."""
    proc: subprocess.Popen
    port: int
    started_at: float
    log_path: Path
    cmd: list[str] = field(default_factory=list)


_running: dict[str, _RunningProcess] = {}


# ---------------------------------------------------------------------------
# Command building (pure; tested directly)
# ---------------------------------------------------------------------------

def _resolve_rpc_endpoints(worker_ids: list[str]) -> list[str]:
    """Look up each worker_id and produce a `<host>:<port>` string.

    Skips:
      * worker rows that have been deleted since the split_model row
        was created (DB referential integrity is loose by design — the
        split row holds string IDs, not foreign keys, so a worker can
        be removed without breaking the split row).
      * disabled workers (the user toggled them off explicitly).
      * workers whose probe last reported `rpc_server_reachable=False`
        — passing those to llama-server would just cause connection
        refusals during the actual inference.

    Returns the endpoints in the same order as `worker_ids`. Order
    matters: llama.cpp uses it to assign layer ranges (worker[0] gets
    the first GPU-layer chunk, worker[1] the next, etc.).
    """
    out: list[str] = []
    for wid in worker_ids:
        w = db.get_compute_worker(wid)
        if not w:
            log.info("split_lifecycle: skipping unknown worker_id %r", wid)
            continue
        if not w.get("enabled"):
            log.info(
                "split_lifecycle: skipping disabled worker %r (%s)",
                w["label"], wid,
            )
            continue
        caps = w.get("capabilities") or {}
        if not caps.get("rpc_server_reachable"):
            log.info(
                "split_lifecycle: skipping worker %r — rpc-server not reachable",
                w["label"],
            )
            continue
        host = (w.get("address") or "").strip()
        # Strip http(s):// like _worker_base_url does in compute_pool.
        for prefix in ("http://", "https://"):
            if host.startswith(prefix):
                host = host[len(prefix):]
        host = host.rstrip("/")
        port = caps.get("rpc_port") or 50052
        out.append(f"{host}:{port}")
    return out


def _build_command(
    *,
    llama_server: Path,
    gguf_path: str,
    port: int,
    rpc_endpoints: list[str],
    ngl: int = _DEFAULT_NGL,
) -> list[str]:
    """Assemble the argv for `llama-server`.

    Pure: takes resolved endpoints, returns a list. Caller is in charge
    of validation (port range etc.) and supplying the binary path.
    Keeping this pure means the test suite can pin the exact flag
    layout without spawning a process.
    """
    cmd: list[str] = [
        str(llama_server),
        "--model", gguf_path,
        # Loopback bind: only the host's own routing layer talks to
        # llama-server. Exposing it on LAN would re-create Ollama's
        # auth-vs-no-auth dilemma without any benefit; the workers
        # access the model via the rpc-server protocol, not HTTP.
        "--host", "127.0.0.1",
        "--port", str(port),
        # Flash-attention: newer llama.cpp builds require an explicit
        # value (on/off/auto). 'auto' lets llama-server enable it when
        # the model + backend support it; older code passed bare `-fa`
        # which the new parser rejects (consumes the next arg as a
        # value, e.g. swallows `--jinja`).
        "-fa", "auto",
        # Use the GGUF's embedded chat template — most modern GGUFs
        # ship a Jinja template under metadata; without --jinja the
        # server falls back to a generic role-marker format that some
        # models reject.
        "--jinja",
        # Number of layers to offload to GPUs in the pool. llama.cpp
        # clamps to the model's actual layer count, so 99 effectively
        # means "as many as fit". Layers that don't fit GPUs cascade
        # to CPU+RAM.
        "-ngl", str(ngl),
        # Context size: cap at 4096 tokens. llama-server otherwise
        # allocates the model's full native context (e.g. llama3.1
        # defaults to 131k tokens / 54k effective with 4 seqs), which
        # blows up KV-cache memory on small workers — the actual
        # crash mode in early bench runs was OOM during KV buffer
        # allocation on a laptop, AFTER the model layers had already
        # loaded successfully across the pool. 4096 is plenty for
        # chat turns and keeps KV memory bounded.
        "-c", "4096",
        # Skip the empty-run warmup. By default llama-server does one
        # forward pass on an empty input to JIT-compile kernels and
        # prime caches. Across an RPC pool with 32 layers × 3 nodes
        # over LAN, that warmup takes 10+ minutes — long enough that
        # /health never reports OK within our boot timeout. The first
        # real request pays the JIT cost instead (one-shot ~5–10 s
        # extra TTFT), which is dramatically less than waiting for
        # warmup before any traffic at all.
        "--no-warmup",
    ]
    if rpc_endpoints:
        # llama-server takes --rpc as a comma-separated list of
        # `<host>:<port>` endpoints. Order controls layer assignment.
        cmd.extend(["--rpc", ",".join(rpc_endpoints)])
    return cmd


def _log_path_for(split_id: str) -> Path:
    """Per-split log file under our private install dir.

    Living next to the binaries keeps everything Phase 2 in one
    directory; `~/.gigachat/llama-cpp/logs/<id>.log`. Created lazily.
    """
    log_dir = split_runtime.LLAMA_CPP_INSTALL_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{split_id}.log"


# ---------------------------------------------------------------------------
# Lifecycle: start / stop / status
# ---------------------------------------------------------------------------

class SplitLifecycleError(RuntimeError):
    """Raised on any pre-flight failure (binary missing, GGUF missing,
    port already in use, etc.). Caller surfaces the message to the UI."""


def _preflight(row: dict) -> tuple[Path, list[str]]:
    """Resolve binary + endpoints + sanity-check the GGUF before spawn.

    Returns (llama_server_path, rpc_endpoints). Raises
    SplitLifecycleError with a user-readable message on any miss.
    """
    server = split_runtime.find_llama_server()
    if not server:
        raise SplitLifecycleError(
            "llama.cpp not installed — install via Settings → Compute → "
            "Install llama.cpp, then start this split model."
        )

    gguf = (row.get("gguf_path") or "").strip()
    if not gguf:
        raise SplitLifecycleError("split_model has no gguf_path")
    if not Path(gguf).is_file():
        raise SplitLifecycleError(
            f"gguf_path does not exist on host: {gguf}"
        )

    rpc = _resolve_rpc_endpoints(row.get("worker_ids") or [])
    return server, rpc


async def _wait_for_health(port: int, timeout: float | None = None) -> None:
    """Poll `http://127.0.0.1:<port>/health` until it returns 200, or
    raise SplitLifecycleError on timeout.

    llama-server's `/health` returns:
      200 + {"status":"ok"}     once the model is loaded and serving
      503                       while the model is still loading
      (no response)             before the HTTP server has bound

    We treat all non-200s and connection-refused as "not ready yet"
    and keep polling until the timeout fires.
    """
    # Read the constant at call time, not at function-def time, so
    # tests / runtime tweaks of `_BOOT_TIMEOUT_SEC` are respected.
    if timeout is None:
        timeout = _BOOT_TIMEOUT_SEC
    deadline = time.monotonic() + timeout
    last_err: str | None = None
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(f"http://127.0.0.1:{port}/health")
                if r.status_code == 200:
                    return
                last_err = f"HTTP {r.status_code}"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(_HEALTH_POLL_SEC)
    raise SplitLifecycleError(
        f"llama-server on port {port} did not become healthy within "
        f"{timeout:.0f}s (last error: {last_err})"
    )


async def start(split_id: str) -> dict:
    """Bring up the llama-server for one split_model row.

    Idempotent: if it's already running according to our registry, we
    return the existing status without spawning a duplicate.
    """
    row = db.get_split_model(split_id)
    if not row:
        raise SplitLifecycleError("split_model not found")

    # Already running? Caller probably hit the API twice; just report.
    if split_id in _running and _running[split_id].proc.poll() is None:
        return {
            "ok": True,
            "status": "running",
            "port": _running[split_id].port,
            "note": "already running",
        }

    # Mark loading BEFORE spawning so the UI can render "starting…"
    # immediately on the next list call. If preflight raises, we'll
    # clear it back to error/stopped below.
    db.update_split_model_status(split_id, status="loading", last_error="")

    try:
        server, rpc_endpoints = _preflight(row)
    except SplitLifecycleError as e:
        db.update_split_model_status(split_id, status="error", last_error=str(e))
        return {"ok": False, "status": "error", "error": str(e)}

    cmd = _build_command(
        llama_server=server,
        gguf_path=row["gguf_path"],
        port=row["llama_port"],
        rpc_endpoints=rpc_endpoints,
    )
    log_path = _log_path_for(split_id)

    # Spawn detached enough that the child outlives our Python
    # process if it's restarted, but kept under the same console
    # group so we can SIGTERM it cleanly. On Windows that's
    # CREATE_NEW_PROCESS_GROUP; on POSIX, no special flags.
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    log_file = log_path.open("ab")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,   # merge into one log file
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except Exception as e:
        log_file.close()
        msg = f"failed to spawn llama-server: {type(e).__name__}: {e}"
        db.update_split_model_status(split_id, status="error", last_error=msg)
        return {"ok": False, "status": "error", "error": msg}

    _running[split_id] = _RunningProcess(
        proc=proc,
        port=row["llama_port"],
        started_at=time.time(),
        log_path=log_path,
        cmd=cmd,
    )

    # Wait for the HTTP server to come up. If health never reports OK,
    # we kill the child and surface the timeout.
    try:
        await _wait_for_health(row["llama_port"])
    except SplitLifecycleError as e:
        # Best-effort cleanup so we don't leave a half-loaded process
        # hogging VRAM.
        await stop(split_id, _from_failed_start=True)
        db.update_split_model_status(split_id, status="error", last_error=str(e))
        return {"ok": False, "status": "error", "error": str(e)}

    db.update_split_model_status(split_id, status="running", last_error="")
    return {
        "ok": True,
        "status": "running",
        "port": row["llama_port"],
        "log_path": str(log_path),
    }


async def stop(split_id: str, *, _from_failed_start: bool = False) -> dict:
    """Terminate the llama-server child and update DB status.

    Idempotent: stopping an already-stopped row is a no-op success.
    `_from_failed_start` is internal — when start() bailed because
    /health never came up, we still need to kill the child but we
    DON'T want to overwrite the error status that start() already
    recorded.
    """
    rp = _running.pop(split_id, None)
    if rp is None:
        # Nothing in our registry. Reset DB if it claims `running` —
        # the row's status is stale (process died, app restarted, etc.).
        row = db.get_split_model(split_id)
        if row and row["status"] in ("loading", "running"):
            db.update_split_model_status(split_id, status="stopped")
        return {"ok": True, "status": "stopped", "note": "was not running"}

    if rp.proc.poll() is None:
        try:
            rp.proc.terminate()
            try:
                rp.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                rp.proc.kill()
                rp.proc.wait(timeout=2.0)
        except Exception as e:
            log.warning("split_lifecycle: error stopping %s: %s", split_id, e)

    if not _from_failed_start:
        db.update_split_model_status(split_id, status="stopped")
    return {"ok": True, "status": "stopped", "port": rp.port}


def status(split_id: str) -> dict:
    """Read-only status snapshot — combines DB row + process registry.

    The DB row is authoritative for "what the user sees" (status,
    last_error). The registry tells us if the OS-level process is
    still alive. Mismatches (DB says `running` but process is gone)
    are common after a crash; we report them as `crashed`.
    """
    row = db.get_split_model(split_id)
    if not row:
        return {"ok": False, "error": "not found"}
    rp = _running.get(split_id)
    pid = None
    alive = False
    if rp:
        pid = rp.proc.pid
        alive = rp.proc.poll() is None
    out = {
        "ok": True,
        "id": split_id,
        "label": row["label"],
        "db_status": row["status"],
        "last_error": row["last_error"],
        "pid": pid,
        "alive": alive,
        "port": rp.port if rp else row["llama_port"],
    }
    # Effective status reconciles DB vs registry.
    if row["status"] == "running" and not alive:
        out["effective_status"] = "crashed"
    else:
        out["effective_status"] = row["status"]
    return out


async def stop_all() -> None:
    """Shutdown helper — terminate every running llama-server child.

    Called from the FastAPI shutdown hook so uvicorn exits cleanly
    (otherwise the children leak GPU memory until OS reclaims them).
    """
    ids = list(_running.keys())
    for sid in ids:
        try:
            await stop(sid)
        except Exception as e:
            log.warning("split_lifecycle: stop_all error for %s: %s", sid, e)


def reconcile_on_boot() -> int:
    """Reset DB rows that claim `running` / `loading` but have no process.

    Scenario: the app crashed (or was Ctrl-C'd hard) while a split
    model was active. The DB row still says `running`, but the
    llama-server child is gone — Windows reclaimed it when uvicorn
    died. Without this reset, the UI would show a phantom green pill
    and the routing layer would happily try to send chat to a port
    nothing is listening on.

    Called once at app startup. Returns the number of rows that were
    reset so the startup hook can log it.
    """
    reset_count = 0
    for row in db.list_split_models():
        if row.get("status") in ("running", "loading"):
            db.update_split_model_status(
                row["id"],
                status="stopped",
                last_error="reset on app boot — process did not survive restart",
            )
            reset_count += 1
    return reset_count


def read_log_tail(split_id: str, lines: int = 100) -> str:
    """Return the last `lines` lines of the per-split log file.

    Used by the Settings UI to show what llama-server is actually
    saying — most start failures (port already bound, GGUF too new
    for this build, OOM during layer offload) surface here BEFORE
    the bare 'failed to start' status reaches the API. Returns empty
    string if the log doesn't exist yet.

    No streaming/follow — keep it simple. The UI fetches once on
    open and re-fetches on user click.
    """
    log_path = _log_path_for(split_id)
    if not log_path.is_file():
        return ""
    try:
        with log_path.open("rb") as f:
            # Cheap tail: read up to ~256 KB and split. For typical
            # llama-server logs (~few KB total in the first few minutes)
            # this is plenty; a runaway log gets truncated to the most
            # recent slice, which is what the operator wants anyway.
            f.seek(0, 2)  # end
            size = f.tell()
            read_bytes = min(size, 256 * 1024)
            f.seek(size - read_bytes)
            data = f.read()
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace")
    out_lines = text.splitlines()
    if len(out_lines) > lines:
        out_lines = out_lines[-lines:]
    return "\n".join(out_lines)
