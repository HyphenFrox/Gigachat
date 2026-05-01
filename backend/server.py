"""Entry point that resolves the bind host before launching uvicorn.

Run this instead of calling ``uvicorn backend.app:app`` directly — it
reads ``GIGACHAT_HOST`` so the bind mode picked by the user takes effect.

Two bind modes:

  - **default (unset)** → ``0.0.0.0``. P2P endpoints are reachable from
    any RFC1918 LAN IP (envelope crypto handles auth); the chat UI is
    loopback-only (the AuthMiddleware in ``app.py`` rejects non-loopback
    requests for non-P2P paths). Compute-pool sharing between two
    devices on the same Wi-Fi works without any per-install
    configuration.
  - **explicit loopback** (``GIGACHAT_HOST=127.0.0.1``) → ``127.0.0.1``.
    Hard isolation — nothing on the LAN reaches anything, not even the
    P2P endpoints. Use when this device shouldn't participate in any
    cross-device compute pool.

Anything else aborts startup with a clear error.

Optional: ``GIGACHAT_TLS_PORT=<port>`` spins up a SECOND uvicorn on a
TLS-enabled port for forward-secret streaming P2P traffic. Default
deployments don't need this.
"""

from __future__ import annotations

import asyncio
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
    actually sent) and read the kernel's chosen source IP.
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
    try:
        host = auth.resolve_bind_host()
    except ValueError as e:
        print(f"\n  [!] config error: {e}\n", file=sys.stderr)
        sys.exit(1)
    port_str = os.environ.get("GIGACHAT_PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        port = 8000

    print()
    if host == "0.0.0.0":
        # Default mode: bind everywhere; AuthMiddleware filters per-path
        # so the chat UI stays loopback-only while P2P endpoints are
        # reachable from RFC1918 LAN clients.
        print(f"  Gigachat chat UI:  http://localhost:{port}  (this machine)")
        lan_ip = _detect_lan_ipv4()
        if lan_ip:
            print(
                f"  P2P endpoints:     reachable on the LAN at "
                f"{lan_ip}:{port}  (compute pool pairing + encrypted proxy)"
            )
        else:
            print(
                "  P2P endpoints:     reachable on every interface "
                "(LAN IP not auto-detected)"
            )
    else:
        print(f"  Gigachat listening on http://{host}:{port}  (loopback-only)")

    # Optional TLS streaming port — opt-in via env var. When set, a
    # second uvicorn binds the same FastAPI app on the TLS port using
    # our self-signed identity cert (peers pin the cert pubkey, so no
    # CA required). Useful for full forward secrecy on streaming chat
    # paths via TLS 1.3 ECDHE.
    tls_port_env = (os.environ.get("GIGACHAT_TLS_PORT") or "").strip()
    tls_port: int | None = None
    if tls_port_env:
        try:
            tls_port = int(tls_port_env)
        except ValueError:
            print(
                f"  [!] GIGACHAT_TLS_PORT={tls_port_env!r} is not a valid "
                "integer — TLS port disabled.",
                file=sys.stderr,
            )
        else:
            print(f"  P2P TLS streaming port: https://{host}:{tls_port}")
    print()

    if tls_port is None:
        # Single uvicorn — the common default deployment.
        uvicorn.run("backend.app:app", host=host, port=port, log_level="info")
        return

    # Dual-server: HTTP (browser + non-stream P2P) + HTTPS (streaming
    # P2P). Both serve the same FastAPI app under one asyncio loop so
    # they share startup/shutdown lifecycle and in-process state.
    from . import p2p_tls
    cert_path, key_path = p2p_tls.ensure_identity_cert()

    http_config = uvicorn.Config(
        "backend.app:app", host=host, port=port,
        log_level="info", lifespan="on",
    )
    tls_config = uvicorn.Config(
        "backend.app:app", host=host, port=tls_port,
        log_level="info", lifespan="off",  # share lifespan with HTTP server
        ssl_certfile=str(cert_path),
        ssl_keyfile=str(key_path),
    )
    http_server = uvicorn.Server(http_config)
    tls_server = uvicorn.Server(tls_config)

    async def _run_both() -> None:
        await asyncio.gather(http_server.serve(), tls_server.serve())

    try:
        asyncio.run(_run_both())
    except KeyboardInterrupt:
        http_server.should_exit = True
        tls_server.should_exit = True


if __name__ == "__main__":
    main()
