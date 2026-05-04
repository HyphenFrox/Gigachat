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
        "discovered", "lock", "port",
    )

    def __init__(self) -> None:
        self.zeroconf = None
        self.service_info = None
        self.browser = None
        self.started_at = 0.0
        # The advertised FastAPI port. Captured at start() so other
        # modules (e.g. p2p_lan_client.push_pair_notify) can tell
        # peers where to reach us without re-deriving from env.
        self.port = 0
        # Keyed by device_id. Each entry:
        #   {"device_id", "label", "ip", "port", "version", "last_seen_at"}
        # Updated by the Zeroconf listener callbacks (which run on the
        # zeroconf engine's own thread); read by any FastAPI handler
        # via the API endpoints. Lock guards both sides.
        self.discovered: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()


_state: _DiscoveryState | None = None


def get_advertised_port() -> int:
    """Return the FastAPI port we advertise to peers, or 8000 if the
    discovery loop hasn't started yet.

    Used by sibling modules (e.g. ``p2p_lan_client.push_pair_notify``)
    to tell freshly-paired peers exactly where to reach us so they
    don't have to guess + fall through to the 8000 default. Falling
    through is fine when we're on the canonical port, but a dev
    instance on 8001 / 8080 would otherwise be stored as
    ``port: null`` on the peer side and become unroutable until mDNS
    or an active scan fills the gap.
    """
    if _state is not None and _state.port:
        return _state.port
    return _DEFAULT_ADVERTISE_PORT


def merge_discovered_peers(peers: list[dict]) -> int:
    """Merge externally-discovered peers (e.g. from the active LAN
    scan in `p2p_lan_scan.py`) into the canonical `_state.discovered`
    dict.

    Each ``peer`` dict must carry at least ``device_id``, ``ip``,
    ``port``, ``public_key_b64``. ``last_seen_at`` is set to now if
    absent so the TTL tracking works.

    Returns the count of peers merged.

    Idempotent — calling with the same peer multiple times just
    refreshes the ``last_seen_at`` timestamp, keeping it fresh in
    the discovered list. Doesn't overwrite the trust anchor (the
    public key) once set, so a malicious LAN-scan response with a
    spoofed pubkey can't displace a real mDNS-learned one.
    """
    if not _state or not peers:
        return 0
    now = time.time()
    merged = 0
    with _state.lock:
        for peer in peers:
            did = (peer.get("device_id") or "").strip()
            if not did:
                continue
            existing = _state.discovered.get(did)
            if existing:
                # Refresh address + last_seen, preserve the original
                # public_key_b64 (trust anchor — first-write-wins).
                existing["ip"] = peer.get("ip") or existing.get("ip")
                existing["port"] = int(peer.get("port") or existing.get("port") or 0)
                existing["label"] = peer.get("label") or existing.get("label")
                existing["last_seen_at"] = peer.get("last_seen_at") or now
                if not existing.get("public_key_b64"):
                    existing["public_key_b64"] = peer.get("public_key_b64") or ""
            else:
                rec = dict(peer)
                rec.setdefault("last_seen_at", now)
                _state.discovered[did] = rec
            merged += 1
    return merged


def _local_ip() -> str:
    """Best-effort guess at the host's primary LAN address.

    Trick: open a UDP socket to a public DNS server and read the
    socket's local address — the OS routing table picks the right
    interface for us. No packet is actually sent (UDP, connect()
    only). Falls back to 127.0.0.1 when offline / no route exists.

    Note: this picks the interface that routes to the PUBLIC internet,
    which on multi-NIC hosts (corporate Ethernet + home Wi-Fi)
    is typically NOT the LAN-private interface that peers with other
    Gigachat installs. Use `_local_lan_ips()` instead when advertising
    over mDNS — it returns every RFC1918 / IPv6-ULA address so peers
    on any local subnet can reach us.
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


# Interface-name substrings that indicate a VPN tunnel / virtual
# adapter we should NEVER advertise as a LAN interface. Matched
# case-insensitively against psutil's `net_if_addrs()` keys.
# Order doesn't matter; any match → skip the whole interface.
_NON_LAN_IFACE_HINTS = (
    "tailscale",
    "nordlynx",
    "openvpn",
    "wireguard",
    "tap-",
    "tun-",
    "vethernet",     # Hyper-V virtual switch (host side, often loopback-only)
    "loopback",
    "isatap",
    "teredo",
    "bluetooth",
)


def _local_lan_ips() -> list[str]:
    """Enumerate every routable RFC1918 IPv4 address on the host.

    Multi-NIC reality: a typical Windows laptop has Wi-Fi (home LAN),
    Ethernet (corporate / wired), one or more VPN tunnels (NordVPN,
    Tailscale), Hyper-V virtual switches, and several APIPA
    `169.254.x.x` interfaces. The kernel's "default route" picks ONE
    for outbound internet traffic; that one is rarely the LAN-private
    interface you'd want to advertise to other Gigachat installs.

    Filtering rules:
      * RFC1918 (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`) — kept.
      * Link-local `169.254.0.0/16` (APIPA, no DHCP) — DROPPED. mDNS
        on these works on a wire-direct link but never reaches another
        regular LAN; advertising them just confuses receivers.
      * Tailscale CGNAT `100.64.0.0/10` — DROPPED (not LAN; see
        SECURITY.md).
      * Any address on an interface whose name contains a VPN /
        virtual-adapter hint (`tailscale`, `nordlynx`, `openvpn`,
        `wireguard`, `tap-`, `tun-`, `vethernet`, etc.) — DROPPED.
        These tunnels carry the address through a different routing
        domain than the physical LAN, and a peer trying to reach them
        from the LAN will time out.

    mDNS broadcasts go out on every interface zeroconf is bound to,
    but a `ServiceInfo` only advertises the addresses we tell it to.
    Returning every routable LAN IPv4 here means peers on any local
    subnet pick the one that routes to them.

    Falls back to `[<routed_ip>]` if psutil isn't importable.
    """
    import ipaddress
    out: list[str] = []
    seen: set[str] = set()
    try:
        import psutil  # type: ignore
        for iface, addrs in psutil.net_if_addrs().items():
            iface_lower = (iface or "").lower()
            if any(hint in iface_lower for hint in _NON_LAN_IFACE_HINTS):
                continue
            for snic in addrs:
                if getattr(snic, "family", None) != socket.AF_INET:
                    continue
                addr = (snic.address or "").strip()
                if not addr or addr in seen:
                    continue
                if addr.startswith(("127.", "0.", "169.254.")):
                    continue
                try:
                    ip_obj = ipaddress.ip_address(addr)
                except ValueError:
                    continue
                if not ip_obj.is_private:
                    continue
                # Tailscale CGNAT — explicitly excluded.
                if addr.startswith("100.") and 64 <= int(addr.split(".")[1]) <= 127:
                    continue
                seen.add(addr)
                out.append(addr)
    except Exception:
        pass
    if not out:
        fallback = _local_ip()
        if fallback and fallback != "127.0.0.1":
            out.append(fallback)
    return out


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

        # Pick the routable IPv4 address. mDNS records can carry
        # MULTIPLE addresses (we advertise every LAN-private NIC we
        # have on this side too). Prefer the one that's on the SAME
        # subnet as one of OUR own LAN interfaces — that's the one
        # we can actually reach. Falls back to the first IPv4 if no
        # subnet match (e.g. peer is on a different subnet we still
        # somehow route to).
        ip = ""
        try:
            v4_addrs = [a for a in (info.parsed_addresses() or []) if "." in a]
        except Exception:
            v4_addrs = []
            try:
                if info.addresses:
                    v4_addrs = [
                        ".".join(str(b) for b in raw) for raw in info.addresses
                    ]
            except Exception:
                v4_addrs = []
        if v4_addrs:
            try:
                import ipaddress as _ipaddr
                my_v4_networks = []
                for my_ip in _local_lan_ips():
                    try:
                        my_v4_networks.append(
                            _ipaddr.ip_interface(f"{my_ip}/24").network,
                        )
                    except ValueError:
                        continue
                for cand in v4_addrs:
                    try:
                        cand_obj = _ipaddr.ip_address(cand)
                    except ValueError:
                        continue
                    for net in my_v4_networks:
                        if cand_obj in net:
                            ip = cand
                            break
                    if ip:
                        break
            except Exception:
                pass
            if not ip:
                ip = v4_addrs[0]

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
    # Explicitly bind zeroconf to every routable LAN IPv4 instead of
    # the default `InterfaceChoice.All`. The default has known issues
    # on multi-NIC Windows where it claims to bind every interface but
    # silently picks one for outbound multicast, breaking discovery in
    # one direction (peers receive our broadcasts on some subnets
    # but not others). Passing the explicit list forces zeroconf to
    # join the multicast group on each named interface, so our
    # outbound queries + announcements actually reach every LAN we're
    # plugged into.
    bind_ips = _local_lan_ips()
    if bind_ips:
        state.zeroconf = Zeroconf(
            interfaces=bind_ips,
            ip_version=IPVersion.V4Only,
        )
    else:
        # No LAN-private addresses (offline laptop, every interface a
        # VPN tunnel). Fall back to default behaviour so the rest of
        # the app still boots; discovery just won't find anyone.
        state.zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
    state.started_at = time.time()

    me = identity.get_identity()
    port = int(advertise_port or _DEFAULT_ADVERTISE_PORT)
    state.port = port
    # Advertise on EVERY RFC1918 / link-local IPv4 we have. On a
    # multi-NIC laptop (Wi-Fi + Ethernet + virtual switches), peers
    # on any of those subnets pick the address that routes to them.
    # See `_local_lan_ips()` for the rationale.
    ips = _local_lan_ips() or [_local_ip()]
    addresses = []
    for ip in ips:
        try:
            addresses.append(socket.inet_aton(ip))
        except OSError:
            # Malformed IP for some reason — skip silently.
            pass
    if not addresses:
        # Last-ditch: advertise loopback so the test path on a
        # disconnected box still works.
        addresses = [socket.inet_aton("127.0.0.1")]
        ips = ["127.0.0.1"]
    log.info(
        "p2p_discovery: advertising on %d LAN address(es): %s",
        len(addresses), ", ".join(ips),
    )
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
        addresses=addresses,
        port=port,
        properties=txt,
        server=f"gigachat-{me.device_id.lower()}.local.",
    )
    # zeroconf's register_service can hang for many minutes on Windows
    # after a quick restart (multicast slot stuck in OS TIME_WAIT) or
    # when one of the bound interfaces is a VPN tunnel that drops
    # multicast packets silently. We do TWO things to keep startup
    # responsive:
    #   1. cap the inline await at 8 s — if it exceeds, log + move on,
    #      the rest of the lifespan can finish + uvicorn binds.
    #   2. on timeout, re-try the register on a daemon thread so the
    #      announcement EVENTUALLY happens once the network unsticks.
    try:
        await asyncio.wait_for(
            asyncio.to_thread(state.zeroconf.register_service, state.service_info),
            timeout=8.0,
        )
        log.info(
            "p2p: mDNS published as %s on %s:%d (device_id=%s, label=%r)",
            _SERVICE_TYPE, ip, port, me.device_id, me.label,
        )
    except asyncio.TimeoutError:
        log.warning(
            "p2p_discovery: register_service > 8 s — deferring to bg thread"
        )

        def _bg_register() -> None:
            try:
                state.zeroconf.register_service(state.service_info)
                log.info("p2p: mDNS published (bg-deferred)")
            except Exception as e:
                log.warning("p2p_discovery: bg register failed: %s", e)
        threading.Thread(
            target=_bg_register, name="p2p_discovery_register_bg", daemon=True
        ).start()
    except Exception as e:
        log.warning("p2p_discovery: register_service failed: %s", e)
    state.browser = ServiceBrowser(
        state.zeroconf, _SERVICE_TYPE, _GigachatListener(),
    )
    _state = state


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
