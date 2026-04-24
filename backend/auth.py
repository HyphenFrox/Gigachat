"""Password authentication for non-loopback requests.

Gigachat binds to 127.0.0.1 by default so only processes on the same machine
can reach it. When the user wants to expose the app over LAN, Tailscale, or
the public internet, they change the bind host — and this module guarantees
the change is paired with a real password check rather than silently throwing
the door open.

Config sources (later overrides earlier):
  1. ``data/auth.json``       — JSON file, shipped untracked, chmod 0600.
                                 Expected shape: {"host": "...", "password": "..."}.
                                 "password" may be a plaintext string or the
                                 PBKDF2 hash format produced by ``hash_password()``.
  2. ``GIGACHAT_HOST``         env var — wins over the file.
  3. ``GIGACHAT_PASSWORD``     env var — wins over the file.

"host" accepts two special literals:

  - ``tailscale`` — auto-detects the first Tailscale IPv4 via ``tailscale ip
    -4`` and binds to ``0.0.0.0`` with an IP allowlist restricting clients
    to the tailnet + loopback.
  - ``public`` — binds to ``127.0.0.1`` only, but expects a reverse proxy
    (Cloudflare Tunnel, Caddy, nginx) running on the same host to forward
    public traffic. Password auth is mandatory, loopback is NOT auto-trusted
    (the reverse proxy delivers public traffic as 127.0.0.1), and session
    cookies are set with the ``Secure`` flag so they only travel over HTTPS.

Raw LAN IPs and ``0.0.0.0`` are not supported — use one of the above modes.

Session tokens are HMAC-SHA256 signed against ``data/auth_secret.key`` (created
on first access with 0600 permissions). They're plain ``<issued_at>.<hmac>``
strings — no external JWT dependency, no replay protection beyond the 30-day
TTL (the attack surface of a single-user localhost app doesn't justify more).
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import subprocess
import time
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_AUTH_JSON = _DATA_DIR / "auth.json"
_SECRET_KEY_FILE = _DATA_DIR / "auth_secret.key"

# Cookie name and TTL. 30 days is long enough that a user on their own
# Tailscale net doesn't have to re-enter the password every week; it's also
# short enough that a stolen cookie eventually rots.
SESSION_COOKIE = "gigachat_session"
SESSION_TTL_SECONDS = 30 * 24 * 3600

# Anything that starts with one of these is considered loopback and bypasses
# the auth check. We check IPv4 dotted form, IPv6 ``::1``, and the
# dual-stack IPv4-mapped form that some proxies emit (``::ffff:127.0.0.1``).
_LOOPBACK_PREFIXES = ("127.", "::ffff:127.")
_LOOPBACK_LITERALS = {"::1", "localhost"}

# Tailscale assigns addresses from the CGNAT block 100.64.0.0/10 (IPv4) and
# the fd7a:115c:a1e0::/48 ULA prefix (IPv6). We treat any source IP inside
# these ranges as "on the tailnet" for the purposes of the access-control
# middleware — auth cookies are still required, this just means the packet
# wasn't delivered by some other interface (e.g. the LAN NIC).
_TAILSCALE_CIDRS = (
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("fd7a:115c:a1e0::/48"),
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def get_config() -> dict:
    """Load merged config from ``data/auth.json`` + env overrides.

    Missing file / unparseable JSON degrades to all-default (localhost,
    no password). This keeps the first-run experience smooth.
    """
    cfg: dict = {"host": "127.0.0.1", "password": ""}
    if _AUTH_JSON.exists():
        try:
            loaded = json.loads(_AUTH_JSON.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for k in ("host", "password"):
                    v = loaded.get(k)
                    if isinstance(v, str):
                        cfg[k] = v
        except (json.JSONDecodeError, OSError):
            pass
    env_host = os.environ.get("GIGACHAT_HOST", "").strip()
    env_pwd = os.environ.get("GIGACHAT_PASSWORD", "").strip()
    if env_host:
        cfg["host"] = env_host
    if env_pwd:
        cfg["password"] = env_pwd
    return cfg


def requires_password(cfg: dict | None = None) -> bool:
    """Auth is required iff the configured host is ``tailscale`` or ``public``.

    Three bind modes are supported: loopback (default, no auth), ``tailscale``
    (tailnet-only, password required), and ``public`` (reverse-proxy mode,
    password required on every request including loopback). See
    ``resolve_bind_host`` for the full list of accepted values — anything
    else is rejected at startup.
    """
    cfg = cfg or get_config()
    host = (cfg.get("host") or "127.0.0.1").strip()
    return host in {"tailscale", "public"}


def is_public_mode(cfg: dict | None = None) -> bool:
    """True when the bind host is ``public`` — reverse-proxy deployments.

    In public mode the operator runs a TLS-terminating proxy (Cloudflare
    Tunnel, Caddy, nginx) on the same host that forwards traffic to the
    backend over loopback. Two behaviours differ from tailscale mode:

      1. Loopback is NOT auto-trusted. The proxy delivers public requests as
         ``127.0.0.1``, so skipping auth there would hand anonymous sessions
         to anyone on the internet.
      2. The Tailscale CGNAT IP allowlist is skipped — the client IP the
         backend sees is always loopback anyway (the proxy is on the same
         host), so there's nothing to filter on.
    """
    cfg = cfg or get_config()
    return (cfg.get("host") or "").strip() == "public"


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """Return PBKDF2-SHA256 hash in ``<salt_hex>:<hash_hex>`` form.

    200 000 iterations is a reasonable 2026 floor on commodity hardware —
    fast enough that a legitimate login is imperceptible, slow enough that
    a stolen ``auth.json`` costs real CPU time to brute-force.
    """
    salt = secrets.token_bytes(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"{salt.hex()}:{h.hex()}"


def _verify_hashed(password: str, stored: str) -> bool:
    try:
        salt_hex, h_hex = stored.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(h_hex)
    except (ValueError, TypeError):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return hmac.compare_digest(expected, actual)


def check_password(password: str) -> bool:
    """Does ``password`` match the configured value?

    Accepts either a PBKDF2 hash (canonical form, recommended — write it via
    ``hash_password()`` into ``auth.json``) or a plain-text password (dev
    convenience — the whole point of Gigachat is that you run it yourself,
    so storing a plaintext secret in a chmod-0600 file is a pragmatic
    tradeoff). Always runs the PBKDF2 path when the stored value looks like
    a hash, falls back to constant-time string compare otherwise.
    """
    if not password:
        return False
    cfg = get_config()
    stored = (cfg.get("password") or "").strip()
    if not stored:
        return False
    # Heuristic: a PBKDF2 hash is hex-hex with a colon and comfortably longer
    # than any sane plaintext password. Anything else is treated as plaintext.
    if ":" in stored and len(stored) >= 65:  # 32-char salt + ":" + 64-char hash
        return _verify_hashed(password, stored)
    return hmac.compare_digest(password, stored)


# ---------------------------------------------------------------------------
# Session tokens
# ---------------------------------------------------------------------------
def _secret_key() -> bytes:
    """Read or lazily create the HMAC signing key (0600 permissions).

    One global key per install — rotating it invalidates every existing
    session, which is a reasonable "log everyone out" lever if you ever
    suspect the file leaked.
    """
    if _SECRET_KEY_FILE.exists():
        try:
            data = _SECRET_KEY_FILE.read_bytes()
            if len(data) >= 16:
                return data
        except OSError:
            pass
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    _SECRET_KEY_FILE.write_bytes(key)
    try:
        os.chmod(_SECRET_KEY_FILE, 0o600)
    except OSError:
        # Non-POSIX filesystem (e.g. FAT on a USB stick) — nothing to do.
        pass
    return key


def make_token() -> str:
    """Mint a fresh session token: ``<issued_at>.<hmac>``.

    Encodes only the issuance timestamp so the server is stateless. TTL is
    enforced at verify time; there's no mid-session rotation because the
    app has exactly one user role (the owner) and no priveledge separation
    that would justify one.
    """
    issued = str(int(time.time()))
    mac = hmac.new(_secret_key(), issued.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{issued}.{mac}"


def verify_token(token: str | None) -> bool:
    """Return True iff ``token`` is a well-formed, un-tampered, un-expired mint."""
    if not token or "." not in token:
        return False
    try:
        issued_str, mac = token.rsplit(".", 1)
        issued = int(issued_str)
    except ValueError:
        return False
    if issued <= 0 or time.time() - issued > SESSION_TTL_SECONDS:
        return False
    expected = hmac.new(
        _secret_key(), issued_str.encode("ascii"), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, mac)


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------
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


def is_tailscale_client(host: str | None) -> bool:
    """Is ``host`` a Tailscale tailnet address?

    Tailscale IPv4 addresses live in the CGNAT block ``100.64.0.0/10`` and
    its IPv6 addresses live in ``fd7a:115c:a1e0::/48``. Packets from any
    other source — LAN IPs like ``192.168.1.50``, the Wi-Fi NIC's own
    address, or public internet — return False and should be rejected by
    the access-control middleware when the server is bound to all
    interfaces for Tailscale access.
    """
    if not host:
        return False
    # Strip an IPv4-mapped IPv6 prefix like ``::ffff:100.64.0.1`` so the
    # CGNAT check sees the plain dotted form.
    if host.startswith("::ffff:"):
        host = host[len("::ffff:"):]
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(addr in net for net in _TAILSCALE_CIDRS)


def get_tailscale_ip() -> str | None:
    """Return the first IPv4 reported by ``tailscale ip -4``, or None.

    Used purely for the startup banner — ``resolve_bind_host`` binds to
    ``0.0.0.0`` in tailscale mode and the middleware does the real
    access-control, so this helper doesn't affect security.
    """
    try:
        out = subprocess.check_output(
            ["tailscale", "ip", "-4"],
            timeout=5,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    for line in out.splitlines():
        ip = line.strip()
        if ip:
            return ip
    return None


_ALLOWED_LOOPBACK_HOSTS = {"", "127.0.0.1", "localhost", "::1"}


def resolve_bind_host(host: str | None) -> str:
    """Translate a config value into something uvicorn can bind to.

    Three modes are supported:
      - loopback (``127.0.0.1`` / ``localhost`` / ``::1`` / empty)
        → binds to ``127.0.0.1``. No auth required.
      - ``tailscale`` → binds to ``0.0.0.0`` so that both the loopback
        interface *and* the Tailscale interface accept traffic on the
        host machine. The real access-control is done by the middleware
        in ``app.py``, which rejects any client that isn't loopback or
        inside the Tailscale CGNAT range (100.64.0.0/10 or
        fd7a:115c:a1e0::/48). If the Tailscale daemon isn't running we
        degrade to pure loopback so the app still starts locally and
        ``server.py`` prints a warning.
      - ``public`` → binds to ``127.0.0.1`` ONLY. Intended for deployments
        where a TLS-terminating reverse proxy (Cloudflare Tunnel, Caddy,
        nginx, …) runs on the same host and forwards traffic to the
        backend. Listening on loopback only means the raw HTTP port cannot
        be reached from the LAN or public internet even if the firewall is
        misconfigured — the proxy is the only gateway. Password auth is
        mandatory; see ``is_public_mode`` and the AuthMiddleware.

    Anything else (raw LAN IPs, ``192.168.x.x``, bare ``0.0.0.0``, etc.)
    is rejected outright — exposing Gigachat on a LAN NIC or unrestricted
    all-interfaces is intentionally not supported.
    """
    host = (host or "").strip()
    if host in _ALLOWED_LOOPBACK_HOSTS:
        return "127.0.0.1"
    if host == "tailscale":
        # Only bind to all interfaces when Tailscale is actually up. If
        # the daemon isn't running there's nothing to listen for, so stay
        # on pure loopback — the banner in server.py tells the user what
        # happened and they can restart after starting Tailscale.
        if get_tailscale_ip() is not None:
            return "0.0.0.0"
        return "127.0.0.1"
    if host == "public":
        # Reverse-proxy mode — the operator runs cloudflared / caddy /
        # nginx in front and we only accept traffic from that local proxy.
        # Refusing to listen on 0.0.0.0 is defence in depth: a typo in the
        # proxy config or an off-by-one firewall rule can't accidentally
        # expose the raw HTTP port to the world.
        return "127.0.0.1"
    raise ValueError(
        f"unsupported host {host!r}. Set host to 'tailscale' for tailnet-only "
        "remote access, 'public' to run behind a reverse proxy (Cloudflare "
        "Tunnel / Caddy / nginx on the same host), or leave it empty / "
        "'127.0.0.1' for local-only. Raw LAN IPs and '0.0.0.0' are not "
        "supported — pick one of the named modes."
    )
