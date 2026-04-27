r"""Drop-in `llama-server` wrapper that engages the compute pool.

Use this anywhere a workload would normally invoke `llama-server`:
swap the binary path to this script and the workload gets pool
routing + adaptive ngl + per-vendor backend selection + periodic
rebalancing for free.

Usage shapes
------------

    # Direct invocation; same args you'd pass llama-server, plus
    # an optional `--llama-server=<path>` to point at the binary.
    python run_llama_server.py -m model.gguf -c 8192 --port 11434

    # When a workload runner accepts a binary path:
    node my-script.js --llama-server="python /path/to/run_llama_server.py" \\
        -m model.gguf -c 8192 --np 4

What it does
------------
1. Reads the same CLI args you'd pass to `llama-server` (-m, -c, --np,
   --batch, --port, etc.) - passed through verbatim.
2. Engages the pool: picks per-vendor backend for each enabled +
   reachable worker, restarts mismatched rpc-servers via SSH+WMI,
   computes adaptive `-ngl` from GGUF metadata + pool free memory.
3. Spawns the real `llama-server` with `--rpc <hosts>` and `-ngl <n>`
   appended; forwards stdin/stdout/stderr unchanged.
4. Runs a background rebalance watcher (`rebalance.RebalanceWatcher`)
   that re-evaluates `-ngl` on a configurable interval and respawns
   `llama-server` only when the optimal value has drifted past a
   threshold AND a cooldown has elapsed (no respawn storm on noisy
   memory). Disable via `--pool-rebalance-interval=0`.
5. Forwards SIGTERM/Ctrl-C to the child for clean shutdown.

Failure modes degrade transparently:
  * No workers configured / none reachable -> spawn plain llama-server
  * GGUF metadata unreadable -> use llama.cpp's default `-ngl 99`
  * Backend switching fails -> warn and continue with current backends

Pass `--no-pool` to bypass engagement entirely (useful when the host
is the only machine available, or for debugging).
"""
from __future__ import annotations

import argparse
import asyncio
import atexit
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
from rebalance import RebalanceWatcher  # noqa: E402

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Binary lookup, arg parsing, model-path extraction
# ---------------------------------------------------------------------------


def _find_llama_server_binary(explicit: str | None) -> str:
    """Locate the underlying `llama-server` binary. Tries:
      1. Explicit `--llama-server=<path>`.
      2. PATH lookup for the platform-appropriate name.
      3. ~/.llamapool/llama-cpp/, ~/.gigachat/llama-cpp/ (well-known).
    """
    if explicit and Path(explicit).is_file():
        return explicit

    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"

    for d in os.environ.get("PATH", "").split(os.pathsep):
        if not d:
            continue
        candidate = Path(d) / name
        if candidate.is_file():
            return str(candidate)

    home = Path.home()
    for prefix in (".llamapool", ".gigachat"):
        candidate = home / prefix / "llama-cpp" / name
        if candidate.is_file():
            return str(candidate)

    raise SystemExit(
        f"ERROR: {name} binary not found. Pass --llama-server=<path>, "
        f"place it on PATH, or install it under ~/.llamapool/llama-cpp/.",
    )


def _extract_model_path(passthrough_args: list[str]) -> str | None:
    """Find `-m <path>` or `--model <path>` so we can compute adaptive ngl."""
    i = 0
    while i < len(passthrough_args):
        a = passthrough_args[i]
        if a in ("-m", "--model") and i + 1 < len(passthrough_args):
            return passthrough_args[i + 1]
        if a.startswith("--model="):
            return a.split("=", 1)[1]
        i += 1
    return None


def _extract_port(passthrough_args: list[str], default: int = 8080) -> int:
    """Find `--port <n>` so the rebalance watcher knows where to /health-probe."""
    i = 0
    while i < len(passthrough_args):
        a = passthrough_args[i]
        if a == "--port" and i + 1 < len(passthrough_args):
            try:
                return int(passthrough_args[i + 1])
            except ValueError:
                return default
        if a.startswith("--port="):
            try:
                return int(a.split("=", 1)[1])
            except ValueError:
                return default
        i += 1
    return default


def _strip_arg(args: list[str], name: str) -> list[str]:
    """Drop `<name>` (and its value if separate). Handles `--rpc value`
    and `--rpc=value` forms."""
    out: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == name and i + 1 < len(args):
            i += 2
            continue
        if a.startswith(f"{name}="):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


# ---------------------------------------------------------------------------
# Spawn + health probe + respawn primitives
# ---------------------------------------------------------------------------


class _ProcHolder:
    """Mutable container for the active llama-server child process so
    the rebalance watcher can swap it transparently. The supervise
    coroutine watches `respawn_seq` to distinguish a respawn-driven
    child exit from an unexpected one."""

    def __init__(self, proc: subprocess.Popen[bytes]) -> None:
        self.proc = proc
        self.respawn_seq = 0


async def _wait_for_health(
    port: int, *, deadline_s: float = 600.0,
    proc: subprocess.Popen[bytes] | None = None,
) -> None:
    """Poll http://127.0.0.1:<port>/health until 200 OK, deadline, or
    child exits. 600 s default tolerates large cold loads (30+ GB
    over RPC can take 8–10 min)."""
    end = time.monotonic() + deadline_s
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < end:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited with code {proc.returncode} "
                f"during health probe",
            )
        try:
            r = urllib.request.urlopen(url, timeout=2)
            if 200 <= r.status < 300:
                return
        except Exception:
            pass
        await asyncio.sleep(2.0)
    raise RuntimeError(
        f"llama-server did not become healthy within {deadline_s:.0f}s",
    )


async def _spawn_child(cmd: list[str], port: int, *, verbose: bool) -> subprocess.Popen[bytes]:
    """Launch a new llama-server child and wait for it to come up."""
    if verbose:
        print(f"[pool] spawning: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.Popen(cmd)
    try:
        await _wait_for_health(port, proc=proc)
    except Exception:
        try:
            proc.terminate()
            await asyncio.to_thread(proc.wait, 5.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        raise
    return proc


async def _terminate_child(proc: subprocess.Popen[bytes]) -> None:
    """Stop a child gracefully; escalate to SIGKILL if it doesn't yield."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=30.0)
    except (asyncio.TimeoutError, subprocess.TimeoutExpired):
        try:
            proc.kill()
        except Exception:
            pass
        try:
            await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=5.0)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _amain(known: argparse.Namespace, passthrough: list[str]) -> int:
    binary = _find_llama_server_binary(known.llama_server)
    port = _extract_port(passthrough, default=8080)
    gguf_path = _extract_model_path(passthrough)

    rpc_endpoints: list[str] = []
    ngl_value: int | None = None
    claim_id: str | None = None
    if not known.no_pool:
        try:
            rpc_endpoints, ngl_value, claim_id = await core.engage_async(
                gguf_path,
                in_split=True,
                priority=known.pool_priority,
            )
        except Exception as e:
            print(
                f"[pool] engagement raised: {e}; falling through to plain spawn",
                file=sys.stderr,
            )

    # Cleanup hook fires on every exit path.
    def _release_claim_sync() -> None:
        if claim_id:
            try:
                asyncio.run(core.disengage_async(claim_id))
            except Exception:
                import registry  # local import to avoid circular at startup
                try:
                    registry.unregister_claim(claim_id)
                except Exception:
                    pass
    atexit.register(_release_claim_sync)

    # Build the pool-injected argv (caller's flags + our --rpc / -ngl).
    final_args = list(passthrough)
    if rpc_endpoints:
        final_args = _strip_arg(final_args, "--rpc")
        final_args.extend(["--rpc", ",".join(rpc_endpoints)])
        print(f"[pool] engaging split: --rpc {','.join(rpc_endpoints)}",
              file=sys.stderr)
    if ngl_value is not None:
        final_args = _strip_arg(final_args, "-ngl")
        final_args = _strip_arg(final_args, "--n-gpu-layers")
        final_args.extend(["-ngl", str(ngl_value)])
        print(f"[pool] adaptive ngl: -ngl {ngl_value}", file=sys.stderr)

    cmd = [binary] + final_args
    proc = await _spawn_child(cmd, port, verbose=known.pool_verbose)
    holder = _ProcHolder(proc)

    # Coordination primitives:
    #   `shutdown` is set on SIGTERM/SIGINT — supervise then exits without
    #     looping into respawn detection.
    #   `respawn_idle` is set whenever NO respawn is in flight. The
    #     supervise loop, if it wakes from a child exit, waits for this
    #     event before deciding "real crash" — closing the race where
    #     terminate finishes before spawn begins.
    shutdown = asyncio.Event()
    respawn_idle = asyncio.Event()
    respawn_idle.set()

    def _forward_signal(signum: int, _frame: object) -> None:
        shutdown.set()
        try:
            holder.proc.send_signal(signum)
        except Exception:
            pass
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _forward_signal)
        except (ValueError, OSError):
            pass

    # Build the respawn callback the rebalance watcher will invoke.
    # `final_args` stays immutable; each respawn builds new_args off
    # it so successive respawns always start from the original caller
    # flags. `respawn_idle` brackets the kill-then-spawn so supervise
    # waits before declaring a child exit a "real crash".
    async def _respawn(new_ngl: int) -> None:
        respawn_idle.clear()
        try:
            new_args = _strip_arg(final_args, "-ngl")
            new_args = _strip_arg(new_args, "--n-gpu-layers")
            new_args.extend(["-ngl", str(new_ngl)])
            new_cmd = [binary] + new_args
            print(
                f"[pool] rebalance: respawning llama-server with -ngl {new_ngl}",
                file=sys.stderr,
            )
            await _terminate_child(holder.proc)
            new_proc = await _spawn_child(
                new_cmd, port, verbose=known.pool_verbose,
            )
            holder.proc = new_proc
            holder.respawn_seq += 1
        finally:
            respawn_idle.set()

    # Start the rebalance watcher iff enabled + we actually have a
    # pool-computed ngl to compare against.
    watcher: RebalanceWatcher | None = None
    if (
        known.pool_rebalance_interval > 0
        and ngl_value is not None
        and ngl_value > 0
        and gguf_path is not None
        and claim_id is not None
        and rpc_endpoints
    ):
        watcher = RebalanceWatcher(
            gguf_path=gguf_path,
            claim_id=claim_id,
            priority=known.pool_priority,
            initial_ngl=ngl_value,
            respawn=_respawn,
            interval_s=known.pool_rebalance_interval,
            threshold_layers=known.pool_rebalance_threshold,
            cooldown_s=known.pool_rebalance_cooldown,
        )
        await watcher.start()

    # Supervise the child. When the watcher respawns it, holder.proc
    # is swapped underneath us — the old wait() returns for the dead
    # process; we wait for `respawn_idle` (covering the kill-then-spawn
    # gap), then check `respawn_seq` to confirm the swap happened. Only
    # an unexpected (non-respawn) exit propagates as our return code.
    try:
        last_seq = holder.respawn_seq
        while True:
            current = holder.proc
            rc = await asyncio.to_thread(current.wait)
            if shutdown.is_set():
                return rc
            # If a respawn is in-flight, wait up to 15 min for it to
            # complete (covers the worst-case cold-load over RPC for a
            # 30 GB GGUF). Timeout falls through to "treat as crash".
            if not respawn_idle.is_set():
                try:
                    await asyncio.wait_for(
                        respawn_idle.wait(), timeout=900.0,
                    )
                except asyncio.TimeoutError:
                    return rc
            if holder.respawn_seq != last_seq:
                last_seq = holder.respawn_seq
                continue  # respawn happened; wait on the new child
            # Real exit (crash or normal). Don't loop.
            return rc
    finally:
        if watcher is not None:
            await watcher.stop()
        await _terminate_child(holder.proc)
        # _release_claim_sync also runs via atexit; calling here makes
        # the claim release happen before the function returns so any
        # caller chaining further work sees an up-to-date registry.
        _release_claim_sync()


def main(argv: list[str] | None = None) -> int:
    """Entry point for the wrapper."""
    parser = argparse.ArgumentParser(
        description="Pool-aware llama-server wrapper",
        allow_abbrev=False,
        add_help=False,
    )
    parser.add_argument(
        "--llama-server",
        help="Explicit path to the llama-server binary (else PATH / "
             "well-known install dirs).",
    )
    parser.add_argument(
        "--no-pool", action="store_true",
        help="Skip pool engagement; act as a transparent llama-server wrapper.",
    )
    parser.add_argument(
        "--pool-verbose", action="store_true",
        help="Log pool engagement decisions to stderr.",
    )
    parser.add_argument(
        "--pool-priority", type=int, default=100,
        help="Integer weight (default 100) for this workload's share of "
             "pool memory under contention. Higher = bigger share.",
    )
    parser.add_argument(
        "--pool-rebalance-interval", type=float, default=60.0,
        help="Seconds between rebalance-watcher ticks (default 60). "
             "Set to 0 to disable periodic rebalancing entirely.",
    )
    parser.add_argument(
        "--pool-rebalance-threshold", type=int, default=3,
        help="Minimum layer-count delta that triggers a respawn "
             "(default 3). Smaller drifts are ignored as not worth a "
             "cold-load.",
    )
    parser.add_argument(
        "--pool-rebalance-cooldown", type=float, default=300.0,
        help="Minimum seconds between respawns (default 300 = 5 min). "
             "Protects against oscillation when free memory is bouncing "
             "around a boundary.",
    )
    known, passthrough = parser.parse_known_args(argv)

    if known.pool_verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="[pool] %(message)s",
            stream=sys.stderr,
        )

    try:
        return asyncio.run(_amain(known, passthrough))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
