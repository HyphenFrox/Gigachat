"""Per-request access control.

Gigachat binds to ``0.0.0.0`` so other devices on the same physical
network can reach the **P2P endpoints** (encrypted compute proxy +
PIN pair handshake). The ``AuthMiddleware`` in ``app.py`` then
filters requests by source IP + path:

  * **Loopback** — full access. The local browser using the install
    on the same machine sees everything.
  * **LAN (RFC1918 / IPv6 ULA / link-local)** — only the P2P inbound
    endpoints are reachable. Their X25519 + Ed25519 envelope crypto
    is the actual auth — no password layer needed on top. Other paths
    (chat UI, settings, etc.) are refused with a clear "loopback only"
    message because each device is expected to run its own Gigachat
    install for chat. Cross-device chat from another device's browser
    isn't a supported use case.
  * **Public IPs / Tailscale CGNAT** — flat 403. The app stays on the
    user's own LAN.

There is **no password feature**. Earlier versions had an opt-in
LAN-web-UI mode with a PBKDF2 password gate; that was removed because
the only reason to access the chat UI from another device on your LAN
was historical, and the cleaner answer is "install Gigachat on that
device too and pair them via Compute pool."

This module exposes three things:

  * ``is_loopback(host)`` — does this client IP look like 127.* / ::1?
  * ``is_lan_client(host)`` — RFC1918 IPv4 / IPv6 ULA / link-local?
  * ``resolve_bind_host(host)`` — translate the optional ``host``
    config into a uvicorn bind value. Default is ``0.0.0.0``; explicit
    ``127.0.0.1`` / ``localhost`` / ``::1`` opts into hard isolation
    (nothing on the LAN can reach anything, not even the P2P endpoints).
"""

from __future__ import annotations

import ipaddress
import os
import sys


# Anything that starts with one of these is considered loopback. We
# check IPv4 dotted form, IPv6 ``::1``, the dual-stack IPv4-mapped form
# that some proxies emit (``::ffff:127.0.0.1``), and the unresolved
# ``localhost`` literal.
_LOOPBACK_PREFIXES = ("127.", "::ffff:127.")
_LOOPBACK_LITERALS = {"::1", "localhost"}

# Private (RFC1918) IPv4 ranges plus the IPv6 unique-local block. A client
# whose source IP falls inside one of these is "on the LAN" — typical home
# / small-office networks live in 192.168.0.0/16 or 10.0.0.0/8, and a few
# routers default to 172.16.0.0/12. fc00::/7 is the IPv6 unique-local
# space (the moral equivalent of RFC1918 for IPv6).
#
# Note: 100.64.0.0/10 is the CGNAT block Tailscale uses, and it's also
# used by some real ISPs. We DO NOT include it here. LAN access is for
# traffic on the user's own home/office network only — Tailscale traffic
# is encrypted but transits relay servers and burns metered internet
# bandwidth.
_LAN_CIDRS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),  # link-local (e.g. mDNS)
    ipaddress.ip_network("fc00::/7"),         # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),        # IPv6 link-local
)

_ALLOWED_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def is_loopback(host: str | None) -> bool:
    """Is this ``request.client.host`` the loopback interface?

    Covers IPv4 ``127.*``, IPv6 ``::1``, the IPv4-mapped form, and the
    unresolved ``localhost`` literal (seen from some proxies).
    """
    if not host:
        return False
    if host in _LOOPBACK_LITERALS:
        return True
    return any(host.startswith(p) for p in _LOOPBACK_PREFIXES)


def is_lan_client(host: str | None) -> bool:
    """Is ``host`` a private LAN address (RFC1918 / IPv6 ULA / link-local)?

    Used by the access-control middleware to admit the user's own LAN
    devices to the P2P endpoints and reject everything else. Tailscale
    CGNAT clients (100.64.0.0/10) and public IPs both fail this check.

    Loopback is handled separately by ``is_loopback``; this function
    returns False for loopback addresses so the middleware can keep the
    two checks distinct.
    """
    if not host:
        return False
    # Strip an IPv4-mapped IPv6 prefix like ``::ffff:192.168.1.10`` so
    # the CIDR check sees the plain dotted form.
    if host.startswith("::ffff:"):
        host = host[len("::ffff:"):]
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(addr in net for net in _LAN_CIDRS)


def resolve_bind_host(host: str | None = None) -> str:
    """Translate an optional ``host`` config into a uvicorn bind value.

    Two modes:

      - **default (empty / unset)** → binds ``0.0.0.0``. The P2P
        endpoints are reachable from any RFC1918 LAN IP (envelope
        crypto handles auth); the chat UI is loopback-only (filter
        in ``AuthMiddleware``).
      - **explicit loopback** (``127.0.0.1`` / ``localhost`` / ``::1``)
        → binds ``127.0.0.1``. Hard isolation — nothing on the LAN
        can reach anything, not even the P2P endpoints. Use when this
        device shouldn't participate in any cross-device compute pool
        (e.g. on an untrusted public Wi-Fi).

    ``host`` is read from the optional ``GIGACHAT_HOST`` env var; the
    parameter is here for tests + the legacy startup banner.

    Anything else (raw LAN IPs, ``0.0.0.0``, ``tailscale``, ``public``,
    etc.) is rejected outright. There's exactly ONE way to opt into
    hard loopback isolation, and it's spelled out as ``127.0.0.1`` in
    the env var.
    """
    if host is None:
        host = (os.environ.get("GIGACHAT_HOST") or "").strip()
    else:
        host = (host or "").strip()
    if host == "":
        return "0.0.0.0"
    if host in _ALLOWED_LOOPBACK_HOSTS:
        return "127.0.0.1"
    raise ValueError(
        f"unsupported GIGACHAT_HOST={host!r}. Leave unset for the "
        "default (LAN-reachable P2P + loopback-only chat UI), or set "
        "'127.0.0.1' for hard isolation. Raw IPs, '0.0.0.0', "
        "'tailscale' and 'public' are not supported."
    )
