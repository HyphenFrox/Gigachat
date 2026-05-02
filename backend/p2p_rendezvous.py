"""Rendezvous client — keeps THIS install registered with the GCP
Cloud Run rendezvous so other peers can find us for P2P NAT-traversal.

Lifecycle:
  1. On startup (when Public Pool toggle is ON), discover our public
     IP/port via STUN and POST it to /register.
  2. Every ~30s, POST /heartbeat to extend the TTL.
  3. Every ~5min, re-discover via STUN in case the NAT mapping
     drifted.
  4. On Public Pool toggle OFF — or on app shutdown — silently
     stop. The peer entry expires from the rendezvous within 60s.

Privacy contract:
  * The rendezvous receives our identity (public_key) and
    STUN-discovered candidates ONLY. It never sees prompts,
    models, conversation content, or even what kinds of work we'd
    accept. The privacy boundary is enforced by what we send,
    not by trust in the server.
  * If the user toggles Public Pool OFF, this module unregisters
    immediately. The Public Pool toggle is the master kill-switch.

Failure mode is "silent + retry": rendezvous unreachable, STUN
unreachable, signature rejected — every error logs at INFO and
the loop continues. The user shouldn't have to babysit the
discovery layer.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import socket
import struct
import time
from dataclasses import dataclass

import httpx

from . import db, identity

log = logging.getLogger("gigachat.p2p.rendezvous")

# Default public rendezvous URL — the Cloud Run instance the project
# ships with. Means a fresh install on a new laptop joins the swarm
# the moment Public Pool toggles on, without the user pasting a URL
# anywhere. Override per-install via the Settings UI (stored in
# user_settings.p2p_rendezvous_url) or env var GIGACHAT_RENDEZVOUS_URL
# — both win over this fallback. Power users self-hosting their own
# rendezvous (rendezvous/ folder + Cloud Run / Fly / VPS) just paste
# their URL into Settings and it takes precedence.
_DEFAULT_RENDEZVOUS_URL = "https://gigachat-rendezvous-1073047996056.us-central1.run.app"

# Rendezvous URL resolution order:
#   1. user_settings.p2p_rendezvous_url (set via the Settings UI;
#      survives restarts; per-install).
#   2. GIGACHAT_RENDEZVOUS_URL env var (operator override; useful for
#      headless / CI deployments).
#   3. _DEFAULT_RENDEZVOUS_URL (the project's public Cloud Run instance)
#      — always present, so a fresh install just works.
def _current_rendezvous_url() -> str:
    """Resolve the rendezvous URL each time the loop checks it.

    Reading on every loop iteration means a UI update via
    `set_rendezvous_url(...)` takes effect within one heartbeat tick
    without a process restart.
    """
    try:
        stored = db.get_setting("p2p_rendezvous_url")
        if stored:
            return str(stored).strip().rstrip("/")
    except Exception:
        pass
    env_override = (os.environ.get("GIGACHAT_RENDEZVOUS_URL") or "").strip().rstrip("/")
    if env_override:
        return env_override
    return _DEFAULT_RENDEZVOUS_URL


def set_rendezvous_url(url: str | None) -> str:
    """Persist a rendezvous URL choice. Empty string clears it.

    Caller is responsible for restarting the loop (or letting the
    next tick pick it up). The status endpoint reflects the new URL
    on the very next read.
    """
    cleaned = (url or "").strip().rstrip("/")
    if cleaned:
        # Light sanity: must be http(s):// — protects against the
        # user accidentally typing a hostname without the scheme,
        # which would silently fail every register call.
        if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
            raise ValueError(
                "rendezvous URL must start with http:// or https://"
            )
    db.set_setting("p2p_rendezvous_url", cleaned)
    return cleaned

# Heartbeat cadence. Server TTL is 60s; we heartbeat every 30 to stay
# resident with safety margin against a missed network round.
_HEARTBEAT_INTERVAL_SEC = 30.0

# Re-discover STUN candidates this often. NAT mappings drift on home
# routers; refreshing every ~5 min catches IP changes (mobile hotspot,
# Wi-Fi switch) without spamming the public STUN servers.
_STUN_REFRESH_INTERVAL_SEC = 300.0

# Public STUN servers. Free, no auth, RFC 5389. We try them in order
# and keep the first that answers — picks up a fresh public IP within
# one round-trip on the first call.
_STUN_SERVERS = [
    ("stun.l.google.com", 19302),
    ("stun.cloudflare.com", 3478),
    ("stun.nextcloud.com", 443),
]

# Per-call timeout for the STUN UDP probe. STUN servers respond in
# tens of milliseconds; if a server takes >2s we give up and try
# the next one. Tight to keep startup latency small on flaky links.
_STUN_TIMEOUT_SEC = 2.0


@dataclass
class _StunCandidate:
    """One reachable endpoint discovered via STUN.

    `source` reports how the candidate was learned — purely
    informational; the rendezvous treats every candidate equally.
    """
    ip: str
    port: int
    source: str = "stun"


class _RendezvousState:
    """Module-level singleton holding the heartbeat task + last STUN
    result. Wrapped in a class so start/stop ownership is explicit."""

    __slots__ = (
        "task", "stop_event", "client",
        "candidates", "last_stun_at", "last_register_at",
        "last_error",
    )

    def __init__(self) -> None:
        self.task: asyncio.Task | None = None
        self.stop_event: asyncio.Event | None = None
        self.client: httpx.AsyncClient | None = None
        self.candidates: list[_StunCandidate] = []
        self.last_stun_at: float = 0.0
        self.last_register_at: float = 0.0
        self.last_error: str = ""


_state: _RendezvousState | None = None


# ---------------------------------------------------------------------------
# STUN: discover our public IP/port via a single UDP round-trip.
# ---------------------------------------------------------------------------

# RFC 5389 STUN message constants. We implement the minimum needed
# to send a Binding Request and parse the XOR-MAPPED-ADDRESS attribute
# from the response — no third-party dep.
_STUN_BINDING_REQUEST = 0x0001
_STUN_MAGIC_COOKIE = 0x2112A442
_STUN_ATTR_XOR_MAPPED_ADDRESS = 0x0020


def _stun_discover_blocking(server: str, port: int) -> _StunCandidate | None:
    """Send a STUN Binding Request and parse the response.

    Synchronous because the entire round-trip is one UDP datagram
    each way; running this on the asyncio thread would require a
    UDP-on-asyncio dance for ~30 ms of network time. Caller invokes
    via `asyncio.to_thread` instead.

    Returns None on any error — DNS failure, server unreachable,
    malformed response, NAT type that obscures the public IP.
    """
    # 12-byte STUN header: type (2) + length (2) + magic cookie (4) +
    # transaction id (12). Length=0 because we send no attributes.
    txn_id = os.urandom(12)
    header = struct.pack(">HHI", _STUN_BINDING_REQUEST, 0, _STUN_MAGIC_COOKIE) + txn_id

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(_STUN_TIMEOUT_SEC)
    try:
        try:
            ip = socket.gethostbyname(server)
        except socket.gaierror:
            return None
        try:
            sock.sendto(header, (ip, port))
            data, _ = sock.recvfrom(2048)
        except socket.timeout:
            return None
        except OSError:
            return None
        # Parse response header: type (2) + length (2) + magic (4) +
        # txn id (12). We expect the type to indicate Binding Success
        # Response (0x0101) and the txn id to match what we sent.
        if len(data) < 20:
            return None
        msg_type, msg_len, magic = struct.unpack(">HHI", data[:8])
        if magic != _STUN_MAGIC_COOKIE:
            return None
        if msg_type != 0x0101:  # Binding Success Response
            return None
        if data[8:20] != txn_id:
            return None
        # Walk attributes looking for XOR-MAPPED-ADDRESS.
        offset = 20
        end = 20 + msg_len
        while offset + 4 <= end and offset + 4 <= len(data):
            attr_type, attr_len = struct.unpack(">HH", data[offset:offset + 4])
            attr_val_start = offset + 4
            attr_val_end = attr_val_start + attr_len
            if attr_val_end > len(data):
                break
            if attr_type == _STUN_ATTR_XOR_MAPPED_ADDRESS:
                # Family (2) + xor-port (2) + xor-address (4 for IPv4).
                if attr_len < 8:
                    return None
                family = data[attr_val_start + 1]  # high byte is reserved (0)
                if family != 0x01:  # IPv4
                    return None
                xor_port = struct.unpack(
                    ">H", data[attr_val_start + 2: attr_val_start + 4],
                )[0]
                # XOR-MAPPED port = port XOR (high 16 bits of magic).
                public_port = xor_port ^ (_STUN_MAGIC_COOKIE >> 16)
                xor_addr_bytes = data[attr_val_start + 4: attr_val_start + 8]
                magic_bytes = struct.pack(">I", _STUN_MAGIC_COOKIE)
                public_addr_bytes = bytes(
                    a ^ b for a, b in zip(xor_addr_bytes, magic_bytes)
                )
                public_ip = ".".join(str(b) for b in public_addr_bytes)
                return _StunCandidate(
                    ip=public_ip, port=public_port, source="stun",
                )
            # Attributes are 4-byte aligned.
            offset = attr_val_end + ((4 - attr_len % 4) % 4)
        return None
    finally:
        sock.close()


async def _gather_candidates() -> list[_StunCandidate]:
    """Try each public STUN server in turn; return the first success.

    PRIVACY: we DELIBERATELY do NOT publish our RFC1918 LAN IP to
    the rendezvous. Earlier versions tagged the host's local LAN
    address as a "lan" candidate so peers on the SAME network could
    skip NAT traversal — but that exposed our internal subnet
    (192.168.x.y / 10.x.y.z / 172.16-31.x.y) to every public-pool
    peer AND to any attacker observing the rendezvous. Same-LAN
    peers don't need it: mDNS (`p2p_discovery`) and the active
    LAN-scan fallback (`p2p_lan_scan`) discover each other on the
    LAN without going near the rendezvous. The only candidate we
    publish is the STUN-discovered public IP, which is what public-
    pool peers actually need to reach us through NAT.
    """
    cands: list[_StunCandidate] = []
    # STUN-only: the public NAT mapping that internet peers actually
    # need. Stops at first success — no need to multiply candidates
    # from different servers (they'd all report the same public IP
    # for the same NAT mapping anyway).
    for server, sport in _STUN_SERVERS:
        try:
            cand = await asyncio.to_thread(
                _stun_discover_blocking, server, sport,
            )
            if cand:
                cands.append(cand)
                break
        except Exception as e:
            log.debug("p2p: STUN %s:%d failed: %s", server, sport, e)
    return cands


# ---------------------------------------------------------------------------
# Rendezvous registration / heartbeat protocol — mirrors the server
# message format exactly. Diverging here would silently fail signature
# verification on the server side.
# ---------------------------------------------------------------------------

def _build_register_message(
    device_id: str, public_key_b64: str,
    x25519_public_b64: str,
    candidates: list[_StunCandidate],
    timestamp: float,
) -> bytes:
    """Match `rendezvous/main.py::_build_register_message` v2 byte-for-byte.

    The empty-models slot is preserved in the signed payload so already-
    deployed rendezvous servers (which still expect v2 with a models
    field in the signature) keep verifying our signatures. We just
    always send empty — the rendezvous's role is now NAT-traversal /
    identity only; model inventory lives entirely on the peers
    themselves and is fetched directly via the encrypted proxy. This
    keeps the swarm truly P2P: the rendezvous is dumb infrastructure,
    not a centralised "who has what" index.
    """
    cand_repr = ",".join(f"{c.ip}:{c.port}:{c.source}" for c in candidates)
    parts = [
        b"gigachat-rdv-register-v2",
        device_id.encode("ascii"),
        public_key_b64.encode("ascii"),
        (x25519_public_b64 or "").encode("ascii"),
        cand_repr.encode("ascii"),
        b"",  # models field — always empty by design (see docstring).
        f"{timestamp:.6f}".encode("ascii"),
    ]
    return b"|".join(parts)


def _build_heartbeat_message(device_id: str, timestamp: float) -> bytes:
    parts = [
        b"gigachat-rdv-heartbeat-v1",
        device_id.encode("ascii"),
        f"{timestamp:.6f}".encode("ascii"),
    ]
    return b"|".join(parts)


async def _post_register(state: _RendezvousState) -> None:
    me = identity.get_public_identity()
    ts = time.time()
    msg = _build_register_message(
        me.device_id, me.public_key_b64, me.x25519_public_b64,
        state.candidates, ts,
    )
    sig = base64.b64encode(me.sign(msg)).decode("ascii")
    body = {
        "device_id": me.device_id,
        "public_key_b64": me.public_key_b64,
        "x25519_public_b64": me.x25519_public_b64,
        "candidates": [
            {"ip": c.ip, "port": c.port, "source": c.source}
            for c in state.candidates
        ],
        # Empty models list — see _build_register_message docstring.
        # Rendezvous accepts the field but new peers never populate it.
        "models": [],
        "timestamp": ts,
        "signature_b64": sig,
    }
    if state.client is None:
        return
    r = await state.client.post(
        f"{_current_rendezvous_url().rstrip('/')}/register",
        json=body, timeout=10.0,
    )
    if r.status_code >= 400:
        state.last_error = (
            f"register HTTP {r.status_code}: {r.text[:200]}"
        )
        log.info("p2p: rendezvous register failed: %s", state.last_error)
        return
    state.last_register_at = time.time()
    state.last_error = ""


async def _post_heartbeat(state: _RendezvousState) -> None:
    me = identity.get_public_identity()
    ts = time.time()
    msg = _build_heartbeat_message(me.device_id, ts)
    sig = base64.b64encode(me.sign(msg)).decode("ascii")
    body = {
        "device_id": me.device_id,
        "timestamp": ts,
        "signature_b64": sig,
    }
    if state.client is None:
        return
    r = await state.client.post(
        f"{_current_rendezvous_url().rstrip('/')}/heartbeat",
        json=body, timeout=10.0,
    )
    # 404 = our entry expired (rendezvous restarted, missed window).
    # Re-register immediately rather than waiting for the next
    # STUN-refresh tick.
    if r.status_code == 404:
        log.info("p2p: rendezvous lost our entry; re-registering")
        await _post_register(state)
        return
    if r.status_code >= 400:
        state.last_error = (
            f"heartbeat HTTP {r.status_code}: {r.text[:200]}"
        )
        log.info("p2p: rendezvous heartbeat failed: %s", state.last_error)


# ---------------------------------------------------------------------------
# Lifecycle.
# ---------------------------------------------------------------------------

def _public_pool_enabled() -> bool:
    """Read the user's Public Pool opt-in. Default ON."""
    val = db.get_setting("p2p_public_pool_enabled")
    if val is None:
        return True
    return str(val).lower() in ("1", "true", "yes", "on")


async def _loop(state: _RendezvousState) -> None:
    """Top-level rendezvous loop. Runs for the lifetime of the process
    (when the toggle is ON) or until stop() is called.

    Two timers running in lockstep:
      * _STUN_REFRESH_INTERVAL_SEC — re-discover STUN candidates and
        re-register. Catches IP / NAT mapping changes.
      * _HEARTBEAT_INTERVAL_SEC — keep the existing entry alive.

    We attempt initial registration immediately so a freshly-toggled-on
    install is discoverable within seconds.
    """
    me = identity.get_public_identity()
    log.info(
        "p2p: rendezvous loop starting (device_id=%s, server=%s)",
        me.device_id, _current_rendezvous_url(),
    )

    # Initial discovery + registration. Errors logged at INFO; the loop
    # continues regardless.
    try:
        state.candidates = await _gather_candidates()
        state.last_stun_at = time.time()
        if state.candidates:
            await _post_register(state)
    except Exception as e:
        state.last_error = f"initial register: {type(e).__name__}: {e}"
        log.info("p2p: %s", state.last_error)

    next_stun = state.last_stun_at + _STUN_REFRESH_INTERVAL_SEC
    next_heartbeat = state.last_register_at + _HEARTBEAT_INTERVAL_SEC

    while not (state.stop_event and state.stop_event.is_set()):
        # Recheck the toggle each iteration so a flip-to-OFF stops
        # the loop within one heartbeat interval.
        if not _public_pool_enabled():
            log.info("p2p: Public Pool toggled off; stopping rendezvous loop")
            break

        now = time.time()
        sleep_for = min(
            max(1.0, next_stun - now),
            max(1.0, next_heartbeat - now),
        )
        try:
            await asyncio.wait_for(
                state.stop_event.wait(),
                timeout=min(sleep_for, _HEARTBEAT_INTERVAL_SEC),
            )
            break  # stop_event was set
        except asyncio.TimeoutError:
            pass

        now = time.time()
        try:
            if now >= next_stun:
                state.candidates = await _gather_candidates()
                state.last_stun_at = now
                next_stun = now + _STUN_REFRESH_INTERVAL_SEC
                if state.candidates:
                    await _post_register(state)
                    next_heartbeat = now + _HEARTBEAT_INTERVAL_SEC
            elif now >= next_heartbeat:
                await _post_heartbeat(state)
                next_heartbeat = now + _HEARTBEAT_INTERVAL_SEC
        except Exception as e:
            state.last_error = f"{type(e).__name__}: {e}"
            log.info("p2p: rendezvous tick failed: %s", state.last_error)


async def start() -> None:
    """Launch the rendezvous loop iff Public Pool is on AND a
    rendezvous URL is configured. Idempotent."""
    global _state
    if _state is not None and _state.task is not None and not _state.task.done():
        return
    if not _current_rendezvous_url():
        log.info(
            "p2p: rendezvous disabled — set the URL via Settings → "
            "Network or the GIGACHAT_RENDEZVOUS_URL env var to enable "
            "cross-internet peer discovery"
        )
        return
    if not _public_pool_enabled():
        log.info(
            "p2p: rendezvous not started — Public Pool is toggled off"
        )
        return
    state = _RendezvousState()
    state.stop_event = asyncio.Event()
    state.client = httpx.AsyncClient(timeout=httpx.Timeout(15.0))
    state.task = asyncio.create_task(_loop(state))
    _state = state


async def stop() -> None:
    """Signal the loop to exit + close the HTTP client. Idempotent.

    Note: we do NOT call /unregister on the rendezvous (no such
    endpoint by design — keeps the API surface tiny). The peer entry
    expires within 60 s of the last heartbeat. A fast rejoin works
    because /register is idempotent.
    """
    global _state
    if _state is None:
        return
    state = _state
    _state = None
    if state.stop_event is not None:
        state.stop_event.set()
    if state.task is not None:
        try:
            await asyncio.wait_for(state.task, timeout=5.0)
        except asyncio.TimeoutError:
            state.task.cancel()
        except Exception:
            pass
    if state.client is not None:
        try:
            await state.client.aclose()
        except Exception:
            pass


def status() -> dict:
    """Snapshot of rendezvous loop state for the API endpoint /
    Settings UI. Cheap — pure read of module-level state."""
    me = identity.get_public_identity()
    url = _current_rendezvous_url()
    return {
        "configured": bool(url),
        "url": url,
        "running": (
            _state is not None
            and _state.task is not None
            and not _state.task.done()
        ),
        "device_id": me.device_id,
        "candidates": [
            {"ip": c.ip, "port": c.port, "source": c.source}
            for c in (_state.candidates if _state else [])
        ],
        "last_stun_at": _state.last_stun_at if _state else 0.0,
        "last_register_at": _state.last_register_at if _state else 0.0,
        "last_error": _state.last_error if _state else "",
    }


async def lookup_peers() -> list[dict]:
    """Bootstrap discovery — fetch the list of currently-online peers
    from the rendezvous. IDENTITY ONLY — no model / capability info.

    Returns ``[{device_id, public_key_b64, x25519_public_b64,
    candidates: [{ip, port, source}, ...], last_seen_at}, ...]``.

    The local pool-inventory cache (`p2p_pool_inventory.py`) is the
    sole caller of this function. It periodically refreshes the peer
    list from here, then queries each peer's own `/api/tags` directly
    via the encrypted proxy to learn what models they have. The
    rendezvous never sees the model graph — that stays peer-to-peer.

    Each returned peer has `device_id` cross-checked against
    `public_key_b64` so a malicious rendezvous can't substitute a
    peer's identity. Verification is cheap: base32(pubkey)[:16] must
    equal the device_id, same rule the rendezvous applies at
    registration.

    Returns empty list on any error (rendezvous unreachable, no URL
    configured). Treated as "no public peers right now" by callers —
    they continue working with host + LAN.
    """
    url = _current_rendezvous_url()
    if not url:
        return []
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
            r = await client.get(f"{url.rstrip('/')}/peers")
            if r.status_code == 404:
                # Old rendezvous deployment without /peers — degrade
                # gracefully so the swarm still works for installs
                # talking to a stale server.
                log.debug("p2p: rendezvous lacks /peers endpoint")
                return []
            r.raise_for_status()
            data = r.json() or {}
            raw = data.get("peers") or []
            if not isinstance(raw, list):
                return []
            verified: list[dict] = []
            for entry in raw:
                if not isinstance(entry, dict):
                    continue
                device_id = entry.get("device_id") or ""
                pub = entry.get("public_key_b64") or ""
                if not device_id or not pub:
                    continue
                # Cross-check device_id ↔ pubkey binding so the
                # rendezvous can't lie about identity. Mirrors
                # `identity._device_id_from_pubkey`.
                try:
                    pub_bytes = base64.b64decode(pub)
                    expected = (
                        base64.b32encode(pub_bytes)
                        .decode("ascii").rstrip("=")[:16].upper()
                    )
                except Exception:
                    continue
                if expected != device_id:
                    log.warning(
                        "p2p: rejecting peer with mismatched "
                        "device_id/pubkey: %s",
                        device_id,
                    )
                    continue
                verified.append(entry)
            return verified
    except Exception as e:
        log.debug("p2p: rendezvous /peers lookup failed: %s", e)
        return []


async def lookup_peer(device_id: str) -> dict | None:
    """Look up another peer's candidate endpoints via the rendezvous.

    Used by the QUIC transport (future commit) when initiating a
    session with a friend by their device_id. Returns None on any
    error — the caller decides whether to retry or fail the request.
    """
    url = _current_rendezvous_url()
    if not url or not device_id:
        return None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            r = await client.get(
                f"{url.rstrip('/')}/lookup/{device_id}",
            )
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
    except Exception as e:
        log.debug("p2p: rendezvous lookup of %s failed: %s", device_id, e)
        return None
