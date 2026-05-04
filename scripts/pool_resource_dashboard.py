#!/usr/bin/env python3
"""Live per-device resource dashboard for the Gigachat compute pool.

Polls ``/api/compute-pool/resources`` on the local Gigachat backend and
prints a refreshable table showing CPU / RAM / GPU / VRAM / disk /
network usage on EVERY device in the pool, plus the slice the
Gigachat process tree is currently using on each.

This script is the user-facing answer to the compute-pool directive:
> Visible 95% memory usage on every participating device under load
> + per-device resource tracker (total vs app portion) + adaptive
> routing proof are the user's success criteria.

Usage:
    python scripts/pool_resource_dashboard.py            # one snapshot
    python scripts/pool_resource_dashboard.py --watch    # live refresh, 3s
    python scripts/pool_resource_dashboard.py --watch --interval 5
    python scripts/pool_resource_dashboard.py --json     # raw JSON dump

The script makes only the local 127.0.0.1:8000 call - the host backend
fans out to every paired peer via the encrypted P2P channel and stitches
the result. No SSH, no extra deps beyond `requests`.
"""

import argparse
import json
import os
import sys
import time
from typing import Any

import requests


HOST = os.environ.get("GIGACHAT_HOST", "127.0.0.1")
PORT = int(os.environ.get("GIGACHAT_PORT", "8000"))
URL = f"http://{HOST}:{PORT}/api/compute-pool/resources"

# ANSI: cyan for headers, green/yellow/red for utilisation thresholds,
# reset at end. Auto-disabled when stdout isn't a tty so log captures
# stay clean.
_USE_COLOR = sys.stdout.isatty() and os.name != "nt" or "WT_SESSION" in os.environ
def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def _pct_color(pct: float) -> str:
    """Green<60, yellow<85, red>=85. Red flips to bright at 95% — the
    user's target for 'pool fully engaged'."""
    if pct >= 95:
        return "1;31"  # bright red
    if pct >= 85:
        return "31"
    if pct >= 60:
        return "33"
    return "32"


def _fmt_bytes(gb: float | int | None) -> str:
    if gb is None:
        return "-"
    g = float(gb)
    if g == 0:
        return "0"
    if g < 1.0:
        return f"{g*1024:.0f}M"
    if g < 100:
        return f"{g:.1f}G"
    return f"{g:.0f}G"


def _fmt_kbps(kbps: float | None) -> str:
    if kbps is None:
        return "-"
    k = float(kbps)
    if k < 1:
        return "0"
    if k < 1024:
        return f"{k:.0f}K/s"
    if k < 1024 * 1024:
        return f"{k/1024:.1f}M/s"
    return f"{k/(1024*1024):.1f}G/s"


def fetch() -> dict[str, Any] | None:
    try:
        r = requests.get(URL, timeout=15)
    except requests.RequestException as e:
        print(_c("31", f"backend unreachable: {e}"))
        return None
    if r.status_code != 200:
        print(_c("31", f"HTTP {r.status_code}: {r.text[:200]}"))
        return None
    try:
        return r.json()
    except ValueError:
        print(_c("31", f"non-JSON response: {r.text[:200]}"))
        return None


def render(snap: dict[str, Any]) -> str:
    rollup = snap.get("rollup") or {}
    devices = snap.get("devices") or []

    lines: list[str] = []
    ts = time.strftime("%H:%M:%S", time.localtime(snap.get("ts") or time.time()))
    lines.append(_c("36;1",
        f"=== Gigachat Compute Pool — {ts} "
        f"({rollup.get('reachable_count', 0)}/{rollup.get('device_count', 0)} reachable) ==="
    ))
    # Pool roll-up line.
    pool_ram = float(rollup.get("ram_used_gb") or 0)
    pool_total = float(rollup.get("ram_total_gb") or 0)
    pool_pct = float(rollup.get("ram_used_pct_pool") or 0)
    pool_app_ram = float(rollup.get("app_ram_used_gb") or 0)
    pool_vram = float(rollup.get("vram_used_gb") or 0)
    pool_vram_total = float(rollup.get("vram_total_gb") or 0)
    lines.append(
        f"Pool RAM: {_c(_pct_color(pool_pct), f'{pool_ram:.1f}/{pool_total:.1f} GB ({pool_pct:.1f}%)')}"
        f"  App-RAM: {pool_app_ram:.1f} GB ({rollup.get('app_processes', 0)} procs)"
        f"  Pool VRAM: {pool_vram:.1f}/{pool_vram_total:.1f} GB"
    )
    lines.append("")

    # Per-device table.
    header = (
        f"{'DEVICE':<22} {'KIND':<5} {'CPU%':>6} {'RAM (used/total)':>22} "
        f"{'GPU':<8} {'VRAM (used/total)':>20} {'GPU%':>6} "
        f"{'NET ↓/↑':>16} {'DISK W':>8} {'APP-RAM':>8} {'PROCS':>5}"
    )
    lines.append(_c("36", header))
    lines.append(_c("36", "-" * len(header)))

    for d in devices:
        label = (d.get("label") or d.get("device_id") or "?")[:22]
        kind = d.get("kind") or "?"
        if "error" in d:
            err = d["error"][:80]
            lines.append(f"{label:<22} {kind:<5} {_c('31', err)}")
            continue
        cpu = float(d.get("cpu_pct") or 0)
        ram_used = float(d.get("ram_used_gb") or 0)
        ram_total = float(d.get("ram_total_gb") or 0)
        ram_pct = float(d.get("ram_used_pct") or 0)
        gpu = (d.get("gpu_kind") or "-")[:8]
        vram_used = float(d.get("vram_used_gb") or 0)
        vram_total = float(d.get("vram_total_gb") or 0)
        gpu_util = float(d.get("gpu_util_pct") or 0)
        net_recv = float(d.get("net_recv_kbps") or 0)
        net_send = float(d.get("net_send_kbps") or 0)
        disk_w = float(d.get("disk_write_kbps") or 0)
        app = d.get("app") or {}
        app_ram = float(app.get("ram_used_gb") or 0)
        app_procs = int(app.get("processes") or 0)

        ram_cell = f"{ram_used:>5.1f}/{ram_total:<5.1f}G ({ram_pct:>4.1f}%)"
        vram_cell = f"{vram_used:>5.1f}/{vram_total:<5.1f}G"
        net_cell = f"{_fmt_kbps(net_recv)}/{_fmt_kbps(net_send)}"
        cpu_str = _c(_pct_color(cpu), f"{cpu:>5.1f}%")
        gpu_str = _c(_pct_color(gpu_util), f"{gpu_util:>5.1f}%")
        ram_str = _c(_pct_color(ram_pct), ram_cell)
        lines.append(
            f"{label:<22} {kind:<5} {cpu_str:>14} {ram_str:>30} "
            f"{gpu:<8} {vram_cell:>20} {gpu_str:>14} "
            f"{net_cell:>16} {_fmt_kbps(disk_w):>8} "
            f"{app_ram:>6.1f}G {app_procs:>5}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--watch", action="store_true", help="refresh until Ctrl+C")
    ap.add_argument("--interval", type=float, default=3.0, help="watch interval seconds")
    ap.add_argument("--json", action="store_true", help="dump raw JSON, exit")
    args = ap.parse_args()

    if args.json:
        snap = fetch()
        if snap is None:
            return 1
        print(json.dumps(snap, indent=2, default=str))
        return 0

    if not args.watch:
        snap = fetch()
        if snap is None:
            return 1
        print(render(snap))
        return 0

    # Live mode — clear screen each tick.
    try:
        while True:
            snap = fetch()
            # ANSI clear + home
            sys.stdout.write("\033[2J\033[H")
            if snap is None:
                sys.stdout.write("(no data)\n")
            else:
                sys.stdout.write(render(snap))
                sys.stdout.write(f"\n\n(refresh every {args.interval}s — Ctrl+C to exit)\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
