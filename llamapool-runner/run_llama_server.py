r"""Drop-in `llama-server` wrapper that engages the compute pool.

Use this anywhere a workload would normally invoke `llama-server`:
swap the binary path to `llamapool-llama-server` (installed as a
console script by this package) and the workload gets pool routing
+ adaptive ngl + per-vendor backend selection for free.

Usage shapes
------------

    # Direct invocation; same args you'd pass llama-server, plus
    # an optional `--llama-server=<path>` to point at the binary.
    llamapool-llama-server -m model.gguf -c 8192 --port 11434

    # When a workload runner accepts a binary path:
    node my-script.js --llama-server="$(which llamapool-llama-server)" \\
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
4. Forwards SIGTERM/Ctrl-C to the child for clean shutdown.

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
from pathlib import Path

# Make our sibling modules importable when this script is invoked
# directly (`python /path/to/llamapool-runner/run_llama_server.py ...`).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import core  # noqa: E402

log = logging.getLogger(__name__)


def _find_llama_server_binary(explicit: str | None) -> str:
    """Locate the underlying `llama-server` binary. Tries:
      1. Explicit `--llama-server=<path>`.
      2. PATH lookup for the platform-appropriate name.
      3. The Gigachat install dir (~/.gigachat/llama-cpp), if present.
      4. The llamapool install dir (~/.llamapool/llama-cpp), if present.
    """
    if explicit and Path(explicit).is_file():
        return explicit

    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"

    # PATH first (most explicit).
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if not d:
            continue
        candidate = Path(d) / name
        if candidate.is_file():
            return str(candidate)

    # Common install locations as a final fallback.
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


def _strip_arg(args: list[str], name: str) -> list[str]:
    """Drop `<name>` (and its value if separate) so we can replace it.
    Handles both `--rpc value` and `--rpc=value` forms."""
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


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `llamapool-llama-server` console script."""
    # Carve out our wrapper-specific flags from the rest, which we
    # forward to the child verbatim.
    parser = argparse.ArgumentParser(
        description="Pool-aware llama-server wrapper",
        allow_abbrev=False,
        add_help=False,  # forward --help to underlying llama-server
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
        help="Integer weight (default 100) deciding this workload's "
             "share of pool memory when other workloads are also active. "
             "Higher = bigger share. Use lower (e.g. 50) for background "
             "batch jobs, higher (e.g. 200) for user-facing work.",
    )
    known, passthrough = parser.parse_known_args(argv)

    if known.pool_verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="[pool] %(message)s",
            stream=sys.stderr,
        )

    binary = _find_llama_server_binary(known.llama_server)

    rpc_endpoints: list[str] = []
    ngl_value: int | None = None
    claim_id: str | None = None
    if not known.no_pool:
        gguf_path = _extract_model_path(passthrough)
        try:
            rpc_endpoints, ngl_value, claim_id = asyncio.run(
                core.engage_async(
                    gguf_path,
                    in_split=True,
                    priority=known.pool_priority,
                ),
            )
        except Exception as e:
            print(f"[pool] engagement raised: {e}; falling through to "
                  "plain spawn", file=sys.stderr)

    # Release the pool claim on any exit path: normal shutdown, signal,
    # crash, or KeyboardInterrupt. atexit + signal handler covers the
    # common cases; the `finally` block below catches the rest.
    def _release_claim() -> None:
        if claim_id:
            try:
                asyncio.run(core.disengage_async(claim_id))
            except Exception:
                # Last-ditch: directly drop the claim from the registry
                # so the next workload sees the right reservation.
                import registry
                try:
                    registry.unregister_claim(claim_id)
                except Exception:
                    pass
    atexit.register(_release_claim)

    # Build final argv. Strip caller-provided -ngl/--rpc if present
    # (we override with pool-computed values when engaging).
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
    if known.pool_verbose:
        print(f"[pool] spawning: {binary} {' '.join(final_args)}",
              file=sys.stderr)

    # Spawn child with pass-through stdio. Forward signals so the
    # parent's auto-restart loop's Ctrl-C reaches llama-server.
    proc = subprocess.Popen(cmd)

    def _forward_signal(signum: int, frame: object) -> None:
        try:
            proc.send_signal(signum)
        except Exception:
            pass

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _forward_signal)
        except (ValueError, OSError):
            # Some signals are unavailable on Windows / when not in main thread.
            pass

    try:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                return proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                return proc.wait()
    finally:
        # Belt-and-suspenders cleanup: atexit will also fire, but
        # release here so subsequent code runs against an up-to-date
        # registry (e.g., test harnesses re-spawning the wrapper).
        _release_claim()


if __name__ == "__main__":
    sys.exit(main())
