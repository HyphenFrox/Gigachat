"""Entry point that resolves the bind host before launching uvicorn.

Run this instead of calling ``uvicorn backend.app:app`` directly — it reads
``backend.auth.get_config()`` so the host configured via env var or
``data/auth.json`` takes effect, including the magic ``tailscale`` value
which auto-detects the Tailscale IPv4 via ``tailscale ip -4``.

Three hosts are supported:

  - loopback (default): no auth, binds to 127.0.0.1.
  - ``tailscale``: binds to 0.0.0.0 with IP allowlist for loopback + tailnet
    CGNAT. Auth required for non-loopback clients.
  - ``public``: binds to 127.0.0.1 and expects a TLS-terminating reverse
    proxy (cloudflared / Caddy / nginx) on the same host to forward public
    traffic in. Auth required for every request — loopback is NOT trusted
    because the proxy delivers everything as loopback.

A misconfigured host aborts startup with a clear error rather than
silently binding to an unexpected interface.

If ``tailscale`` is configured but the Tailscale daemon isn't running,
``resolve_bind_host`` falls back to loopback so the app still runs
locally — we print a warning and the user can restart once Tailscale
is up.
"""

from __future__ import annotations

import os
import sys

import uvicorn

from . import auth


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
    public_mode = configured_host == "public"
    print()

    if host == "0.0.0.0":
        # Tailscale mode: we're listening on every interface but the
        # middleware only admits loopback + Tailscale CGNAT clients.
        # Show both URLs so the user knows how to reach the app from the
        # host machine and from remote tailnet peers.
        print(f"  Gigachat listening on http://localhost:{port}  (this machine)")
        ts_ip = auth.get_tailscale_ip()
        if ts_ip:
            print(f"  Gigachat listening on http://{ts_ip}:{port}  (tailnet)")
    elif public_mode:
        # Public mode binds to loopback — the reverse proxy (cloudflared,
        # Caddy, nginx) forwards internet traffic onto it. Tell the user so
        # there's no surprise when :8000 isn't directly reachable.
        print(f"  Gigachat listening on http://{host}:{port}  (behind reverse proxy)")
    else:
        print(f"  Gigachat listening on http://{host}:{port}")

    if configured_host == "tailscale" and host == "127.0.0.1":
        # Tailscale was requested but unreachable — we silently downgraded
        # to loopback. Tell the user so they can start Tailscale and retry.
        print(
            "  [!] host 'tailscale' requested but tailscale is not running.\n"
            "      Falling back to loopback — remote access is disabled until\n"
            "      you start Tailscale and restart this server.",
            file=sys.stderr,
        )
    elif auth_required and not (cfg.get("password") or "").strip():
        mode_name = "public" if public_mode else "tailscale"
        print(
            f"  [!] host is '{mode_name}' but no password is configured.\n"
            "      All requests will be rejected with 401 until you set\n"
            "      GIGACHAT_PASSWORD=... or write a password into data/auth.json.",
            file=sys.stderr,
        )
    elif public_mode:
        print("  public mode — password required for every request (loopback is NOT trusted)")
        print("  point your reverse proxy at http://127.0.0.1:" + str(port) + " to expose this over HTTPS")
    elif auth_required:
        print("  password required for remote (tailnet) requests — localhost is free")
    else:
        print("  loopback-only (no password required)")
    print()

    uvicorn.run("backend.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
