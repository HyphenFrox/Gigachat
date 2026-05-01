"""Active TCP scan of every local /24 to find Gigachat peers.

Why this exists
===============
mDNS multicast is the primary discovery mechanism (see ``p2p_discovery.py``)
and works flawlessly on most home networks. But on a non-trivial fraction
of consumer Wi-Fi setups it just doesn't work in one or both directions:

  * Some Wi-Fi access points enable "AP isolation" / "PSPMS" by default,
    which blocks Wi-Fi-to-Wi-Fi multicast even on the same SSID.
  * Some routers do IGMP snooping with aggressive timeouts and silently
    drop the multicast group membership for clients they think are
    inactive.
  * Switching between Wi-Fi and Ethernet on the same /24 sometimes
    leaves multicast working in one direction (Ethernet → Wi-Fi)
    while the other direction is silently dropped.
  * VPN clients (NordVPN, ExpressVPN) sometimes intercept outbound
    multicast even with their kill-switch nominally off.

Active TCP scanning sidesteps all of that. We probe every IPv4 host
on each of our routable LAN /24 subnets with a short HTTP GET to
``/api/p2p/identity``. Any host running Gigachat answers in <100 ms
on a wired LAN; non-Gigachat hosts either time out or close the
connection. The scan completes in 1–2 seconds wall-clock for a /24
with reasonable concurrency.

Discovered peers are merged into the same dict ``p2p_discovery``
populates via mDNS so the rest of the app (UI, pairing, routing)
sees a single unified list.

Privacy & politeness
====================
We HTTP GET ``/api/p2p/identity`` with a 250 ms timeout. That endpoint
is deliberately unauthenticated and returns only public info
(device_id + label + Ed25519 pubkey). A non-Gigachat HTTP server on
the LAN will see one short request to a path it doesn't recognise
and return a 404 — the behaviour any reverse-DNS / local-services
probe would. We don't probe arbitrary ports; only the canonical
Gigachat port (configurable, defaulting to 8000) plus a small fallback
list common to dev setups.

Scan cadence
============
Slow by default (60 s) because the primary discovery path is mDNS,
and active scanning is a fallback. Faster on first boot / when the
discovered list is empty, so a fresh install on a flaky-multicast
network sees peers within seconds.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import time
from typing import Any

import httpx

log = logging.getLogger("gigachat.p2p.lan_scan")

# Per-host probe timeout. Tight — we don't want to wait 1 s per host
# on a /24 (would be 4 minutes serial). 200 ms is enough for any
# Gigachat install on a wired or Wi-Fi LAN to respond.
_PROBE_TIMEOUT_SEC = 0.2

# Maximum concurrent probes. Higher = faster scan but more open
# sockets. 64 is a comfortable middle ground for a typical Windows
# Python build.
_MAX_CONCURRENT_PROBES = 64

# How often to re-scan when we already have at least one discovered
# peer. Slow because mDNS does the heavy lifting; we're just a
# fallback. Bumped on first boot.
_SCAN_INTERVAL_NORMAL_SEC = 60.0

# How often to re-scan when the discovered list is empty. Faster so a
# fresh install on a multicast-broken network sees peers within
# seconds.
_SCAN_INTERVAL_EMPTY_SEC = 10.0

# Ports to probe. The canonical Gigachat backend listens on 8000;
# users running a second install on the same box (dev + prod) might
# bump it to 8001 / 8080. Keep the list tiny — every extra port
# multiplies the scan cost.
_PROBE_PORTS = (8000,)

# Hard cap on total addresses scanned per cycle. Defends against a
# misconfigured host with a /16 or /8 netmask producing 65 K hosts
# to probe. /24 is 254 hosts; /23 is 510; we cap at ~1000.
_MAX_HOSTS_PER_CYCLE = 1000


_loop_task: asyncio.Task | None = None
_loop_stop_event: asyncio.Event | None = None
_last_scan_at: float = 0.0


def _enumerate_subnet_hosts() -> list[str]:
    """Build the list of IPv4 addresses to probe.

    Walks every routable LAN-private interface from
    `p2p_discovery._local_lan_ips()`, treats each as a /24, and
    enumerates every host in the subnet (excluding our own address
    and the network/broadcast addresses).

    Caps the total at ``_MAX_HOSTS_PER_CYCLE`` to prevent a /16
    netmask from spawning a 5-minute scan.
    """
    from . import p2p_discovery as _disc
    out: list[str] = []
    seen: set[str] = set()
    my_ips = set(_disc._local_lan_ips())
    for my_ip in my_ips:
        try:
            iface = ipaddress.ip_interface(f"{my_ip}/24")
        except ValueError:
            continue
        net = iface.network
        for host in net.hosts():
            host_str = str(host)
            if host_str == my_ip or host_str in seen:
                continue
            seen.add(host_str)
            out.append(host_str)
            if len(out) >= _MAX_HOSTS_PER_CYCLE:
                return out
    return out


async def _probe_one(
    client: httpx.AsyncClient, ip: str, port: int,
) -> dict | None:
    """Probe one (ip, port). Return a discovered-peer dict on a hit,
    None otherwise.

    Returns None on every failure path: connection refused, timeout,
    HTTP error status, malformed JSON, missing fields. The caller
    just collects the truthy results — every other return is a
    non-Gigachat host that we silently move past.
    """
    url = f"http://{ip}:{port}/api/p2p/identity"
    try:
        r = await client.get(url, timeout=_PROBE_TIMEOUT_SEC)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    device_id = (data.get("device_id") or "").strip()
    pubkey = (data.get("public_key_b64") or "").strip()
    if not device_id or not pubkey:
        return None
    label = (data.get("label") or device_id).strip()
    return {
        "device_id": device_id,
        "label": label,
        "ip": ip,
        "port": port,
        "version": "1",
        "public_key_b64": pubkey,
        "last_seen_at": time.time(),
        "_source": "lan_scan",
    }


async def scan_once() -> list[dict]:
    """Run one full scan and return every Gigachat peer found.

    Public so the UI / startup paths can trigger a synchronous scan
    when the user opens the Compute pool tab without waiting for the
    next periodic tick.
    """
    global _last_scan_at
    hosts = _enumerate_subnet_hosts()
    if not hosts:
        return []
    sem = asyncio.Semaphore(_MAX_CONCURRENT_PROBES)
    me_device_id = ""
    try:
        from . import identity as _ident
        me_device_id = _ident.get_identity().device_id
    except Exception:
        pass
    found: list[dict] = []
    started = time.time()

    async def _bounded(client: httpx.AsyncClient, ip: str, port: int) -> None:
        async with sem:
            peer = await _probe_one(client, ip, port)
            if peer and peer["device_id"] != me_device_id:
                found.append(peer)

    # One AsyncClient for the whole scan so we get HTTP/2 connection
    # pooling + reuse where possible. Probes are short-lived, so
    # most connections will be re-established each call anyway.
    async with httpx.AsyncClient() as client:
        tasks = [
            _bounded(client, ip, port)
            for ip in hosts
            for port in _PROBE_PORTS
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - started
    log.info(
        "p2p_lan_scan: probed %d hosts in %.1fs, found %d Gigachat peer(s)",
        len(hosts), elapsed, len(found),
    )

    # Merge findings into the canonical discovered dict so the UI
    # sees them alongside mDNS results. The merger preserves the
    # trust anchor (public_key_b64) on first-write so a malicious
    # LAN-scan response can't displace a real mDNS-learned key.
    if found:
        from . import p2p_discovery as _disc
        _disc.merge_discovered_peers(found)

    _last_scan_at = time.time()
    return found


async def _background_loop(stop_event: asyncio.Event) -> None:
    """Periodic scan loop. Runs until stop_event is set."""
    # First scan immediately so a fresh boot finds peers before any
    # user action — important when mDNS is broken on this network.
    try:
        await scan_once()
    except Exception as e:
        log.info("p2p_lan_scan: first scan failed: %s", e)

    while not stop_event.is_set():
        # Adaptive cadence: faster when we have nothing to show.
        try:
            from . import p2p_discovery as _disc
            current = _disc.list_discovered()
        except Exception:
            current = []
        interval = (
            _SCAN_INTERVAL_EMPTY_SEC if not current
            else _SCAN_INTERVAL_NORMAL_SEC
        )
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass
        try:
            await scan_once()
        except Exception as e:
            log.info("p2p_lan_scan: scan failed: %s", e)


async def start() -> None:
    """Kick off the background LAN scan loop. No-op if already running."""
    global _loop_task, _loop_stop_event
    if _loop_task is not None and not _loop_task.done():
        return
    _loop_stop_event = asyncio.Event()
    _loop_task = asyncio.create_task(
        _background_loop(_loop_stop_event),
        name="p2p_lan_scan_loop",
    )
    log.info("p2p_lan_scan: background scan loop started")


async def stop() -> None:
    """Stop the loop cleanly. Idempotent."""
    global _loop_task, _loop_stop_event
    if _loop_stop_event is not None:
        _loop_stop_event.set()
    if _loop_task is not None:
        try:
            await asyncio.wait_for(_loop_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            _loop_task.cancel()
        except Exception:
            pass
    _loop_task = None
    _loop_stop_event = None


def status() -> dict:
    """Snapshot of the loop's state for diagnostics."""
    return {
        "running": _loop_task is not None and not _loop_task.done(),
        "last_scan_at": _last_scan_at,
        "subnets": [],  # filled in by _enumerate_subnet_hosts caller if needed
    }
