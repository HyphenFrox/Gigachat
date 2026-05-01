"""LAN service discovery via multicast DNS (Bonjour / Avahi-compatible).

Each Gigachat install publishes itself as ``_gigachat._tcp.local.`` so
peers on the same LAN can find each other without configuration —
"Bluetooth-style" pairing UX. The advertisement carries enough TXT
metadata to identify the device + show a friendly label in the
discovery UI; the actual pairing handshake happens over the existing
HTTP API once the user picks a device.

Two responsibilities:

  1. **Publish** the local instance's identity (device_id + label +
     port). Pure data — no compute exposed yet, that's the pairing /
     transport step.

  2. **Browse** for other Gigachat instances. Maintain a
     ``discovered`` map the API endpoint reads on poll. We also
     trigger an auto-reconnect for any paired device whose mDNS
     record reappears with a different IP — the trust anchor is the
     ``device_id`` (== Ed25519 pubkey hash), not the address.

Lifecycle is driven from FastAPI's lifespan hook (start at boot,
stop on shutdown). Idempotent: calling start() twice is a no-op.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
import time
from typing import Any

from . import db, identity

log = logging.getLogger("gigachat.p2p.discovery")

# RFC 6763 service-type identifier. The shape is
# `_<servicename>._<transport>.local.`. We pick `gigachat` as a
# reasonably-unique app key — collisions are functionally impossible
# on a typical LAN, and a peer that sees an advertisement for some
# other app's `_gigachat._tcp` service will fail the trust handshake
# and quietly disappear from the UI list.
_SERVICE_TYPE = "_gigachat._tcp.local."

# How long we keep a discovered peer in the map without seeing it.
# The mDNS browser fires `update_service` on any change; absence for
# longer than this means the peer has gone away (laptop closed,
# Wi-Fi dropped, app exited).
#
# Set generously (5 min) so a single dropped multicast packet doesn't
# kick a peer out of the picker. mDNS multicast is unreliable in
# practice on home networks — Wi-Fi PSPMS, VPN split-tunnel software
# (NordVPN, etc.), and some consumer routers all drop multicast
# silently. The 5 min window absorbs those drops; a peer that's
# genuinely gone (laptop closed) is still pruned, just on a slower
# cadence than network jitter.
_DISCOVERY_TTL_SEC = 300.0

# Fallback advertised port. We default to the FastAPI port the user
# is running on; if `start()` isn't passed an explicit value we read
# it out of the env (uvicorn sets PORT) or fall back to 8000.
_DEFAULT_ADVERTISE_PORT = 8000


class _DiscoveryState:
    """Module-level singleton holding the live mDNS objects.

    Wrapping in a class keeps the module's top-level small — `start()`
    constructs an instance lazily and `stop()` tears it down. The
    `discovered` dict is the read-side state every endpoint pulls from.
    """

    __slots__ = (
        "zeroconf", "service_info", "browser", "started_at",
        "discovered", "lock",
    )

    def __init__(self) -> None:
        self.zeroconf = None
        self.service_info = None
        self.browser = None
        self.started_at = 0.0
        # Keyed by device_id. Each entry:
        #   {"device_id", "label", "ip", "port", "version", "last_seen_at"}
        # Updated by the Zeroconf listener callbacks (which run on the
        # zeroconf engine's own thread); read by any FastAPI handler
        # via the API endpoints. Lock guards both sides.
        self.discovered: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()


_state: _DiscoveryState | None = None


def _local_ip() -> str:
    """Best-effort guess at the host's primary LAN address.

    Trick: open a UDP socket to a public DNS server and read the
    socket's local address — the OS routing table picks the right
    interface for us. No packet is actually sent (UDP, connect()
    only). Falls back to 127.0.0.1 when offline / no route exists.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 8.8.8.8 is unreachable from many networks but routing
        # works regardless — connect() is local-state-only on UDP.
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


class _GigachatListener:
    """Zeroconf service listener — receives add/update/remove events.

    Each event lands here on the zeroconf engine's own thread. We do
    minimal work (parse TXT, update the discovered map, kick the
    auto-reconnect for paired peers). DB writes happen inline; they're
    SQLite + tiny, so the latency cost is negligible.
    """

    def add_service(self, zc, type_, name) -> None:  # type: ignore[no-untyped-def]
        self._on_change(zc, name)

    def update_service(self, zc, type_, name) -> None:  # type: ignore[no-untyped-def]
        self._on_change(zc, name)

    def remove_service(self, zc, type_, name) -> None:  # type: ignore[no-untyped-def]
        # Pull the device_id out of the cached entry (we may not be
        # able to query the service after it's gone). Best-effort.
        if _state is None:
            return
        with _state.lock:
            stale = [
                did for did, rec in _state.discovered.items()
                if rec.get("_service_name") == name
            ]
            for did in stale:
                _state.discovered.pop(did, None)

    def _on_change(self, zc, name: str) -> None:
        if _state is None:
            return
        try:
            info = zc.get_service_info(_SERVICE_TYPE, name, timeout=2000)
        except Exception:
            return
        if info is None:
            return
        # Skip our own advertisement — we don't need to "discover"
        # ourselves on the wire.
        try:
            self_id = identity.get_identity().device_id
        except Exception:
            self_id = ""
        # Parse TXT records. Zeroconf returns them as bytes.
        txt = {
            (k.decode("ascii", errors="replace") if isinstance(k, (bytes, bytearray)) else str(k)):
            (v.decode("utf-8", errors="replace") if isinstance(v, (bytes, bytearray)) else "")
            for k, v in (info.properties or {}).items()
        }
        device_id = txt.get("device_id", "")
        if not device_id or device_id == self_id:
            return

        # Pick the first IPv4 address that's actually routable. The
        # `addresses_by_version` API is the v0.146+ shape; we
        # fall back to the legacy `addresses` attribute on older
        # zeroconf versions.
        ip = ""
        try:
            addrs = info.parsed_addresses() or []
            for a in addrs:
                if "." in a:  # naive but correct for v4 vs v6
                    ip = a
                    break
            if not ip and addrs:
                ip = addrs[0]
        except Exception:
            try:
                if info.addresses:
                    ip = ".".join(str(b) for b in info.addresses[0])
            except Exception:
                ip = ""

        port = int(getattr(info, "port", 0) or 0)
        label = txt.get("label") or device_id
        version = txt.get("version") or ""
        # Pubkey is in the TXT record under "public_key" — propagate it
        # under the canonical name "public_key_b64" so the frontend can
        # use it for the cross-device pair-claim signature without an
        # extra round-trip to /api/p2p/identity. Falls back to empty
        # string when an older peer's mDNS record omits it.
        public_key_b64 = txt.get("public_key") or ""

        record = {
            "device_id": device_id,
            "label": label,
            "ip": ip,
            "port": port,
            "version": version,
            "public_key_b64": public_key_b64,
            "last_seen_at": time.time(),
            "_service_name": name,
        }
        with _state.lock:
            _state.discovered[device_id] = record

        # Auto-reconnect: if this device is already paired, refresh
        # the IP/port + label. Trust anchor (public_key) is unchanged —
        # the pairing handshake captured it once and we never overwrite.
        try:
            paired = db.get_paired_device(device_id)
            if paired and ip:
                # Only log + propagate when the address actually
                # changed — a stable peer's mDNS keep-alive should
                # not spam the log.
                address_changed = paired.get("ip") != ip or paired.get("port") != port
                db.update_paired_device_last_seen(
                    device_id, ip=ip, port=port, label=label,
                )
                if address_changed:
                    log.info(
                        "p2p: paired peer %s moved to %s:%d (label=%r)",
                        device_id, ip, port, label,
                    )
                    # Phase 2: keep the compute_workers row in sync.
                    # Same identity, new address. Routing scoring +
                    # probe data are preserved by the targeted
                    # `update_compute_worker_address` helper.
                    try:
                        worker = db.get_compute_worker_by_device_id(device_id)
                        if worker:
                            db.update_compute_worker_address(
                                worker["id"],
                                address=ip,
                                # mDNS-advertised port is Gigachat's
                                # FastAPI port, NOT Ollama's. Leave
                                # ollama_port alone (the column has
                                # its own value the user set).
                                label=label,
                            )
                    except Exception as e:
                        log.debug(
                            "p2p: compute_worker address update failed: %s", e,
                        )
        except Exception as e:
            log.debug("p2p: paired-device refresh failed: %s", e)


async def start(advertise_port: int | None = None) -> None:
    """Start advertising + browsing. Idempotent."""
    global _state
    if _state is not None:
        return
    # Lazy import — keeps the rest of the codebase free of zeroconf
    # at module-load time. Anyone who runs the backend without the
    # dep installed sees a clean error here, not on every import.
    try:
        from zeroconf import IPVersion, ServiceInfo, Zeroconf
        from zeroconf import ServiceBrowser
    except ImportError as e:
        log.warning(
            "p2p_discovery disabled — zeroconf not installed (%s). "
            "`pip install zeroconf` to enable LAN device discovery.",
            e,
        )
        return

    state = _DiscoveryState()
    state.zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
    state.started_at = time.time()

    me = identity.get_identity()
    port = int(advertise_port or _DEFAULT_ADVERTISE_PORT)
    ip = _local_ip()
    # mDNS service name needs to be unique on the LAN. Suffixing with
    # the device_id guarantees uniqueness even when two installs use
    # the same hostname (the more common case than you'd think — two
    # laptops out of the box are both `DESKTOP-XXXX`).
    instance_name = f"gigachat-{me.device_id}.{_SERVICE_TYPE}"
    txt: dict[str, str] = {
        "device_id": me.device_id,
        "label": me.label,
        "version": "1",
        "public_key": me.public_key_b64,
    }
    state.service_info = ServiceInfo(
        _SERVICE_TYPE,
        instance_name,
        addresses=[socket.inet_aton(ip)],
        port=port,
        properties=txt,
        server=f"gigachat-{me.device_id.lower()}.local.",
    )
    try:
        await asyncio.to_thread(
            state.zeroconf.register_service, state.service_info,
        )
    except Exception as e:
        log.warning("p2p_discovery: register_service failed: %s", e)

    state.browser = ServiceBrowser(
        state.zeroconf, _SERVICE_TYPE, _GigachatListener(),
    )
    _state = state
    log.info(
        "p2p: mDNS published as %s on %s:%d (device_id=%s, label=%r)",
        _SERVICE_TYPE, ip, port, me.device_id, me.label,
    )


async def stop() -> None:
    """Tear down advertisement + browser. Idempotent."""
    global _state
    if _state is None:
        return
    state = _state
    _state = None
    try:
        if state.service_info is not None and state.zeroconf is not None:
            await asyncio.to_thread(
                state.zeroconf.unregister_service, state.service_info,
            )
    except Exception as e:
        log.debug("p2p_discovery: unregister_service failed: %s", e)
    try:
        if state.zeroconf is not None:
            await asyncio.to_thread(state.zeroconf.close)
    except Exception as e:
        log.debug("p2p_discovery: zeroconf.close failed: %s", e)
    log.info("p2p: mDNS stopped")


def list_discovered() -> list[dict[str, Any]]:
    """Snapshot of currently-discovered LAN peers (excluding self).

    Stale entries (no advertisement update for `_DISCOVERY_TTL_SEC`)
    are pruned on read so the API always returns a live view.
    """
    if _state is None:
        return []
    cutoff = time.time() - _DISCOVERY_TTL_SEC
    with _state.lock:
        live = [
            {k: v for k, v in rec.items() if not k.startswith("_")}
            for rec in _state.discovered.values()
            if (rec.get("last_seen_at") or 0) >= cutoff
        ]
        # Drop expired entries from the cache so memory doesn't grow
        # over a long-running install with peers transiently joining
        # and leaving.
        for did in list(_state.discovered.keys()):
            if (_state.discovered[did].get("last_seen_at") or 0) < cutoff:
                _state.discovered.pop(did, None)
    return live


def is_running() -> bool:
    """True when the discovery service is up. Useful for the status
    endpoint so the UI can show "mDNS unavailable" when the user's
    OS blocked the multicast socket."""
    return _state is not None
