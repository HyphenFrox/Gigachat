"""Entry point that resolves the bind host before launching uvicorn.

Run this instead of calling ``uvicorn backend.app:app`` directly — it reads
``backend.auth.get_config()`` so the host configured via env var or
``data/auth.json`` takes effect.

Two hosts are supported:

  - loopback (default): no auth, binds to 127.0.0.1. Only same-machine
    processes can connect.
  - ``lan``: binds to 0.0.0.0 with a source-IP allowlist (loopback + RFC1918
    LAN ranges). Other devices on the same physical network can reach the
    app once they enter the password. Public IPs and Tailscale CGNAT
    clients are refused — the app is intentionally not exposed over the
    internet or over the Tailscale overlay.

A misconfigured host aborts startup with a clear error rather than
silently binding to an unexpected interface.
"""

from __future__ import annotations

import os
import socket
import sys

import uvicorn

from . import auth


def _detect_lan_ipv4() -> str | None:
    """Best-effort discovery of the host's primary LAN IPv4.

    Used purely for the startup banner so the user knows which URL to
    paste into another device on the same network. The middleware does
    the real access-control, so this helper isn't security-relevant —
    if it returns the wrong NIC the app still works.

    Strategy: open a UDP socket "to" a public address (no packet is
    actually sent) and read the kernel's chosen source IP. This picks
    whichever interface the OS would route an outbound packet through,
    which on a typical home setup is the active Wi-Fi or Ethernet NIC.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 203.0.113.1 is a TEST-NET-3 address that no network actually
        # routes to — picking it avoids ever leaking a real packet to a
        # third-party host while still letting the kernel resolve the
        # routing table to a source address.
        s.connect(("203.0.113.1", 1))
        ip = s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()
    if not ip or ip.startswith("127."):
        return None
    return ip


def main() -> None:
    cfg = auth.get_config()
    configured_host = (cfg.get("host") or "").strip()
    try:
        host = auth.resolve_bind_host(configured_host)
    except ValueError as e:
        print(f"\n  [!] config error: {e}\n", file=sys.stderr)
        sys.exit(1)
    port_str = os.environ.get("GIGACHAT_PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        port = 8000

    auth_required = auth.requires_password(cfg)
    print()

    if host == "0.0.0.0":
        # LAN mode: listening on every interface but the middleware only
        # admits loopback + RFC1918 LAN clients. Show both URLs so the
        # user knows how to reach the app from this machine and from
        # another device on the same Wi-Fi/Ethernet.
        print(f"  Gigachat listening on http://localhost:{port}  (this machine)")
        lan_ip = _detect_lan_ipv4()
        if lan_ip:
            print(f"  Gigachat listening on http://{lan_ip}:{port}  (LAN)")
        else:
            # Couldn't auto-detect — give the user a hint without printing
            # anything that might be wrong.
            print("  (use this machine's LAN IPv4 from another device on the same network)")
    else:
        print(f"  Gigachat listening on http://{host}:{port}")

    if auth_required and not (cfg.get("password") or "").strip():
        print(
            "  [!] host is 'lan' but no password is configured.\n"
            "      All non-loopback requests will be rejected with 401 until you set\n"
            "      GIGACHAT_PASSWORD=... or write a password into data/auth.json.",
            file=sys.stderr,
        )
    elif auth_required:
        print("  password required for LAN clients — localhost is free")
    else:
        print("  loopback-only (no password required)")
    print()

    uvicorn.run("backend.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
