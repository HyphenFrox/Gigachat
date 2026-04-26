"""Command-line entry points for llamapool.

Subcommands:
  * `add-worker LABEL ADDRESS [--rpc-port N] [--ssh-host HOST]
                              [--gpu-vendor {nvidia,amd,intel,none}]`
        Register a worker in the JSON config.

  * `remove-worker LABEL`
        Drop a worker from the config.

  * `list-workers [--reachable-only]`
        Print the registered workers + a quick reachability probe.

  * `probe-worker LABEL`
        SSH into the worker, inspect Win32_VideoController to learn
        the GPU vendor, and persist the result. Skips the user
        through having to specify `--gpu-vendor` manually for Windows
        workers.

The drop-in `llama-server` wrapper lives in `run_llama_server.py`
and is exposed as a separate entry point (`llamapool-llama-server`),
not as a subcommand here, so existing apps can swap their
`llama-server.exe` path without needing to know about a subcommand.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Make our sibling modules importable when this script is invoked
# directly (`python /path/to/llamapool-runner/cli.py ...`).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import workers  # noqa: E402


def _cmd_add_worker(args: argparse.Namespace) -> int:
    """Add or update a worker."""
    entry = config.add_worker(
        label=args.label,
        address=args.address,
        rpc_port=args.rpc_port,
        ssh_host=args.ssh_host,
        gpu_vendor=args.gpu_vendor,
        enabled=not args.disabled,
    )
    print(json.dumps(entry, indent=2))
    return 0


def _cmd_remove_worker(args: argparse.Namespace) -> int:
    """Remove a worker by label."""
    ok = config.remove_worker(args.label)
    if not ok:
        print(f"no worker with label {args.label!r}", file=sys.stderr)
        return 1
    print(f"removed worker {args.label!r}")
    return 0


def _cmd_list_workers(args: argparse.Namespace) -> int:
    """Print workers + reachability."""
    ws = config.list_workers(enabled_only=False)
    if not ws:
        print("(no workers configured — use `llamapool add-worker ...`)",
              file=sys.stderr)
        return 0
    rows = []
    for w in ws:
        reachable = workers.is_rpc_reachable(w)
        if args.reachable_only and not reachable:
            continue
        rows.append({
            "label": w.get("label"),
            "address": w.get("address"),
            "rpc_port": w.get("rpc_port", 50052),
            "enabled": w.get("enabled", True),
            "gpu_vendor": w.get("gpu_vendor"),
            "ssh_host": w.get("ssh_host"),
            "reachable": reachable,
            "current_rpc_backend": w.get("current_rpc_backend"),
        })
    print(json.dumps(rows, indent=2))
    return 0


_PROBE_PS = (
    "$gpus = Get-CimInstance Win32_VideoController | "
    "  Select-Object Name, AdapterCompatibility;"
    "$gpus | ConvertTo-Json -Compress"
)


def _ps_classify_vendor(out: str) -> str:
    """Pick a single vendor label from the probe payload."""
    text = out.lower()
    # Order matters: NVIDIA dominates if present (it's a discrete GPU);
    # then AMD; finally Intel iGPU (frequently coexists with NVIDIA dGPU).
    if "nvidia" in text:
        return "nvidia"
    if "amd" in text or "advanced micro devices" in text or "radeon" in text:
        return "amd"
    if "intel" in text:
        return "intel"
    return "none"


def _cmd_probe_worker(args: argparse.Namespace) -> int:
    """SSH into the worker, learn GPU vendor, persist."""
    import base64
    import subprocess
    w = next((x for x in config.list_workers() if x.get("label") == args.label), None)
    if not w:
        print(f"no worker with label {args.label!r}", file=sys.stderr)
        return 1
    ssh_host = (w.get("ssh_host") or "").strip()
    if not ssh_host:
        print(f"worker {args.label!r} has no ssh_host configured", file=sys.stderr)
        return 2
    encoded = base64.b64encode(_PROBE_PS.encode("utf-16-le")).decode("ascii")
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
           ssh_host, "powershell", "-NoProfile", "-EncodedCommand", encoded]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30.0)
    except subprocess.TimeoutExpired:
        print(f"ssh probe timed out for {ssh_host}", file=sys.stderr)
        return 3
    if r.returncode != 0:
        print(f"ssh probe failed (rc={r.returncode}): {r.stderr.strip()}",
              file=sys.stderr)
        return 4
    vendor = _ps_classify_vendor(r.stdout)
    config.update_worker(args.label, gpu_vendor=vendor)
    print(f"worker {args.label!r}: gpu_vendor = {vendor}")
    print(f"  raw probe: {r.stdout.strip()[:200]}")
    return 0


def _cmd_engage(args: argparse.Namespace) -> int:
    """Print the rpc-endpoints + ngl + claim_id computed for a given GGUF.
    Useful for shell scripts that need to inject these flags into
    their own llama-server command without using the wrapper.

    Note: this DOES register a claim in the registry (so other
    workloads see your reservation). The claim is released when this
    process exits — for long-running shell scripts, prefer the
    wrapper which keeps the claim alive for llama-server's lifetime.
    """
    import core
    rpc, ngl, claim = asyncio.run(core.engage_async(
        args.gguf, in_split=True, priority=args.priority,
    ))
    out: dict[str, Any] = {"rpc": rpc, "claim_id": claim}
    if ngl is not None:
        out["ngl"] = ngl
    print(json.dumps(out, indent=2))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    """Print active pool claims (which workloads are using the pool now)."""
    import registry
    claims = registry.get_active_claims()
    if not claims:
        print("(no active claims — pool is idle)", file=sys.stderr)
        return 0
    print(json.dumps(claims, indent=2, default=str))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with subcommands."""
    p = argparse.ArgumentParser(
        prog="llamapool",
        description="Compute-pool runner for llama.cpp workloads.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    add_p = sub.add_parser("add-worker", help="Register a worker in the config.")
    add_p.add_argument("label", help="Friendly worker label.")
    add_p.add_argument(
        "address",
        help="Network address to reach this worker's rpc-server "
        "(LAN hostname or IP).",
    )
    add_p.add_argument("--rpc-port", type=int, default=50052,
                       help="rpc-server port (default 50052).")
    add_p.add_argument("--ssh-host",
                       help="SSH alias for remote rpc-server lifecycle.")
    add_p.add_argument("--gpu-vendor",
                       choices=["nvidia", "amd", "intel", "none"],
                       help="GPU vendor (skips probing if set).")
    add_p.add_argument("--disabled", action="store_true",
                       help="Add but mark disabled (excluded from engage).")
    add_p.set_defaults(func=_cmd_add_worker)

    rm_p = sub.add_parser("remove-worker", help="Remove a worker from the config.")
    rm_p.add_argument("label")
    rm_p.set_defaults(func=_cmd_remove_worker)

    ls_p = sub.add_parser("list-workers",
                          help="Show workers and their reachability.")
    ls_p.add_argument("--reachable-only", action="store_true",
                      help="Filter to workers whose rpc-server responds.")
    ls_p.set_defaults(func=_cmd_list_workers)

    probe_p = sub.add_parser("probe-worker",
                             help="SSH probe to detect GPU vendor; persist.")
    probe_p.add_argument("label")
    probe_p.set_defaults(func=_cmd_probe_worker)

    eng_p = sub.add_parser(
        "engage",
        help="Print --rpc and -ngl values for a given GGUF (no spawn).",
    )
    eng_p.add_argument("gguf", help="Path to the GGUF model file.")
    eng_p.add_argument("--priority", type=int, default=100,
                       help="Pool-share weight (default 100).")
    eng_p.set_defaults(func=_cmd_engage)

    st_p = sub.add_parser(
        "status",
        help="Show active pool claims (workloads currently using the pool).",
    )
    st_p.set_defaults(func=_cmd_status)

    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `llamapool` CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
