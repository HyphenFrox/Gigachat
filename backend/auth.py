"""Password authentication for non-loopback requests.

Gigachat binds to 127.0.0.1 by default so only processes on the same machine
can reach it. The user can opt into wider exposure by setting the bind host
to ``lan`` — the server then accepts connections from other devices on the
same physical network (Wi-Fi / Ethernet) but never from the public internet
or from a Tailscale overlay.

Config sources (later overrides earlier):
  1. ``data/auth.json``       — JSON file, shipped untracked, chmod 0600.
                                 Expected shape: {"host": "...", "password": "..."}.
                                 "password" may be a plaintext string or the
                                 PBKDF2 hash format produced by ``hash_password()``.
  2. ``GIGACHAT_HOST``         env var — wins over the file.
  3. ``GIGACHAT_PASSWORD``     env var — wins over the file.

"host" accepts two values:

  - loopback (the default — also written as ``127.0.0.1``, ``localhost``,
    ``::1``, or empty): binds to ``127.0.0.1``, no password required.
    Only processes on the same machine can connect.
  - ``lan``: binds to ``0.0.0.0`` so the OS will accept TCP connections on
    every interface. The middleware then rejects any client whose source
    IP isn't loopback or a private RFC1918 LAN address. Public IPs and
    Tailscale CGNAT clients are always refused — by design, this app is
    not exposed to the internet, and Tailscale traffic is kept off the
    pipe to save metered bandwidth. Password auth is mandatory for any
    non-loopback client.

Anything else (raw LAN IPs, ``0.0.0.0``, ``tailscale``, ``public``, etc.)
is rejected at startup with a clear error rather than silently binding to
an unexpected interface.

Session tokens are HMAC-SHA256 signed against ``data/auth_secret.key`` (created
on first access with 0600 permissions). They're plain ``<issued_at>.<hmac>``
strings — no external JWT dependency, no replay protection beyond the 30-day
TTL (the attack surface of a single-user LAN app doesn't justify more).
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import time
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_AUTH_JSON = _DATA_DIR / "auth.json"
_SECRET_KEY_FILE = _DATA_DIR / "auth_secret.key"

# Cookie name and TTL. 30 days is long enough that a user on their own
# LAN doesn't have to re-enter the password every week; it's also short
# enough that a stolen cookie eventually rots.
SESSION_COOKIE = "gigachat_session"
SESSION_TTL_SECONDS = 30 * 24 * 3600

# Anything that starts with one of these is considered loopback and bypasses
# the auth check. We check IPv4 dotted form, IPv6 ``::1``, and the
# dual-stack IPv4-mapped form that some proxies emit (``::ffff:127.0.0.1``).
_LOOPBACK_PREFIXES = ("127.", "::ffff:127.")
_LOOPBACK_LITERALS = {"::1", "localhost"}

# Private (RFC1918) IPv4 ranges plus the IPv6 unique-local block. A client
# whose source IP falls inside one of these is "on the LAN" — typical home
# / small-office networks live in 192.168.0.0/16 or 10.0.0.0/8, and a few
# routers default to 172.16.0.0/12. fc00::/7 is the IPv6 unique-local space
# (the moral equivalent of RFC1918 for IPv6).
#
# Note: 100.64.0.0/10 is the CGNAT block Tailscale uses, and it's also used
# by some real ISPs. We DO NOT include it here. LAN mode is for traffic on
# the user's own home/office network only — Tailscale traffic is encrypted
# but transits relay servers and burns metered internet bandwidth, which
# the user has explicitly opted out of for ongoing app access.
_LAN_CIDRS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),  # link-local (e.g. mDNS)
    ipaddress.ip_network("fc00::/7"),         # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),        # IPv6 link-local
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
    """Auth is required iff the configured host is ``lan``.

    Two bind modes are supported: loopback (default, no auth) and ``lan``
    (other devices on the same LAN can connect, password required for
    every non-loopback request). See ``resolve_bind_host`` for the full
    list of accepted values — anything else is rejected at startup.
    """
    cfg = cfg or get_config()
    host = (cfg.get("host") or "127.0.0.1").strip()
    return host == "lan"


def is_lan_mode(cfg: dict | None = None) -> bool:
    """True when the bind host is ``lan``.

    Convenience alias kept separate from ``requires_password`` so that
    if a future mode is added that also requires auth (without being
    LAN), the call sites stay legible.
    """
    cfg = cfg or get_config()
    return (cfg.get("host") or "").strip() == "lan"


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
    app has exactly one user role (the owner) and no privilege separation
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


def is_lan_client(host: str | None) -> bool:
    """Is ``host`` a private LAN address (RFC1918 / IPv6 ULA / link-local)?

    Used by the access-control middleware in ``lan`` mode to admit the
    user's own LAN devices and reject everything else — Tailscale CGNAT
    clients (100.64.0.0/10) and public IPs both fail this check.

    Loopback is handled separately by ``is_loopback``; this function
    returns False for loopback addresses so the middleware can keep the
    two checks distinct (loopback bypasses auth, LAN clients still need
    a session cookie).
    """
    if not host:
        return False
    # Strip an IPv4-mapped IPv6 prefix like ``::ffff:192.168.1.10`` so the
    # CIDR check sees the plain dotted form.
    if host.startswith("::ffff:"):
        host = host[len("::ffff:"):]
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(addr in net for net in _LAN_CIDRS)


_ALLOWED_LOOPBACK_HOSTS = {"", "127.0.0.1", "localhost", "::1"}


def resolve_bind_host(host: str | None) -> str:
    """Translate a config value into something uvicorn can bind to.

    Two modes are supported:
      - loopback (``127.0.0.1`` / ``localhost`` / ``::1`` / empty)
        → binds to ``127.0.0.1``. No auth required. Only same-machine
        processes can connect.
      - ``lan`` → binds to ``0.0.0.0`` so the host accepts traffic on
        every NIC. The real access-control happens in the middleware
        in ``app.py``, which rejects any client that isn't loopback or
        on a private LAN range (RFC1918 IPv4 + IPv6 ULA + link-local).
        Public IPs and Tailscale CGNAT addresses are refused — this app
        is not designed for internet exposure or overlay-network access.

    Anything else (raw LAN IPs, ``0.0.0.0``, ``tailscale``, ``public``,
    etc.) is rejected outright. Forcing the user through the named
    ``lan`` literal keeps the audit trail honest: there's exactly one
    way to opt into wider listening, and it's spelled out.
    """
    host = (host or "").strip()
    if host in _ALLOWED_LOOPBACK_HOSTS:
        return "127.0.0.1"
    if host == "lan":
        # Bind on every interface; the middleware filters by source IP.
        # We can't pre-pick a single LAN NIC because home networks often
        # have several (Wi-Fi + Ethernet + a Hyper-V virtual switch) and
        # the user shouldn't have to enumerate them.
        return "0.0.0.0"
    raise ValueError(
        f"unsupported host {host!r}. Set host to 'lan' to allow other "
        "devices on the same physical network to connect, or leave it "
        "empty / '127.0.0.1' for local-only. Raw IPs, '0.0.0.0', "
        "'tailscale' and 'public' are not supported — pick one of the "
        "named modes."
    )
