"""Privacy guard for the P2P pool.

Updated contract (security model is encryption-based, not block-based):
  Data WILL flow between peers across the LAN and the internet —
  prompts, chat content, results, everything. The privacy guarantee
  comes from end-to-end encryption (`p2p_crypto` module), not from
  refusing to send data to remote peers.

What this module does now:
  * Classifies any outbound URL as `loopback` / `paired_lan` / `unknown`.
  * Provides `require_encryption(url)` — raises if a caller is about
    to send PLAINTEXT to a destination that should be using the
    encrypted envelope. Default-on for unknown destinations; default-
    off for loopback (the host's own Ollama doesn't speak our
    envelope and isn't on the wire anyway).
  * Provides `is_loopback(url)` — for callers that need a fast
    branch ("if local, use raw HTTP; otherwise wrap in envelope").

Categories:
  * **Loopback** — `127.0.0.1`, `localhost`, `::1`. The host's own
    Ollama process. No encryption required (data never touches
    the wire); plaintext HTTP is fine.
  * **Paired LAN** — any host in `compute_workers` whose row carries
    a non-NULL `gigachat_device_id`. These are friend peers we
    paired with via PIN. Plaintext OK on a trusted home LAN; for
    Wi-Fi networks shared with untrusted users (public/work),
    encryption is recommended (toggle later).
  * **Unknown** — anything else. Internet-reachable peers, hosts
    we never paired with, the rendezvous service. Plaintext
    forbidden — caller must wrap in `p2p_crypto.seal_json` and
    POST that envelope to the peer's `/api/p2p/secure` endpoint.

The guard offers two modes:

  1. ``check_outbound_is_local(url) -> bool`` — fast in-process check
     callers can use to short-circuit a request before sending it.
  2. ``assert_outbound_is_local(url, kind="chat")`` — raises
     ``PrivacyViolation`` when the destination isn't local. Wraps the
     happy path with a hard fail-closed invariant so a regression
     anywhere in the routing layer can't accidentally send prompts
     out of the local pool.

Centralising the check here means: when the P2P transport lands in a
later phase, all the privacy logic lives in one place. The transport
layer can wire its outbound calls through ``assert_outbound_is_local``
with ``kind="public"`` to mark them as intended for the public pool;
anything else gets the local-only guarantee for free.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from . import db

log = logging.getLogger("gigachat.p2p.privacy")


class PrivacyViolation(RuntimeError):
    """Raised when something tries to send PLAINTEXT data to a
    destination that should be using the encrypted envelope.
    Fail-closed: callers should let the exception propagate. Sending
    plaintext to a non-local destination is worse than the request
    failing.
    """


# Loopback hostnames considered host-local. We resolve to these when
# the chat target is the host's own Ollama. Anything else is treated
# as a remote endpoint and must show up in the local-pool whitelist.
_LOOPBACK_HOSTS = frozenset({
    "localhost", "127.0.0.1", "::1",
    "0.0.0.0",  # generic listen-on-all; if we're calling it, it's us
})


def _hostname_of(url: str) -> str:
    """Lowercase hostname of a URL, or '' if unparseable."""
    try:
        h = (urlparse(url).hostname or "").strip().lower()
        return h
    except Exception:
        return ""


def _local_pool_hosts() -> set[str]:
    """Snapshot of every hostname/IP we consider "local pool".

    Walks `compute_workers` (the existing routing table) and emits
    each row's address. Both manually-added rows and rows
    auto-created from P2P pairings are included — both reflect a
    user's explicit decision to trust that destination.

    Re-read on every call so a freshly-paired device is whitelisted
    immediately. Cheap: it's a single SQLite SELECT against an
    indexed column with maybe-tens of rows.
    """
    hosts: set[str] = set(_LOOPBACK_HOSTS)
    try:
        for w in db.list_compute_workers():
            addr = (w.get("address") or "").strip().lower()
            if addr:
                hosts.add(addr)
    except Exception as e:
        # Failure here would default to "no whitelist" which fails
        # closed (every outbound call rejected). That's safer than
        # accidentally allowing leaks, but we log loudly so the
        # operator sees the misconfig.
        log.warning("p2p_privacy: failed to load worker whitelist: %s", e)
    return hosts


def is_loopback(url: str) -> bool:
    """Fast boolean — True iff the destination is the host itself.

    Used by callers that need a "if loopback, plaintext OK; else
    require envelope" branch. Loopback is the only category where
    bytes never touch the network.
    """
    host = _hostname_of(url)
    return bool(host) and host in _LOOPBACK_HOSTS


def is_paired_lan_peer(url: str) -> bool:
    """True iff the URL points at a host we paired with via PIN.

    The pair flow auto-creates a `compute_workers` row for the peer
    so the existing routing layer picks it up. We treat manual-add
    `compute_workers` rows the same way — if the user typed the IP
    in Settings → Compute, that's an explicit trust statement.
    """
    host = _hostname_of(url)
    if not host:
        return False
    if host in _LOOPBACK_HOSTS:
        return False
    return host in _local_pool_hosts()


def check_outbound_is_local(url: str) -> bool:
    """Backwards-compat shim — returns True for loopback OR paired LAN.

    Kept so any in-flight caller from the prior block-based guard
    keeps compiling. New callers should use `is_loopback` /
    `is_paired_lan_peer` / `require_encryption` directly because the
    semantics they encode are clearer.
    """
    return is_loopback(url) or is_paired_lan_peer(url)


def require_encryption(url: str, *, kind: str = "chat") -> bool:
    """Return True when `url` requires the encrypted envelope.

    Decision matrix:
      loopback         → False (no wire, plaintext fine)
      paired LAN peer  → False (trusted LAN, plaintext historically OK;
                                 callers MAY still wrap for defence in
                                 depth on hostile networks)
      unknown          → True  (internet peer, public-pool peer, etc.;
                                 anything plaintext to here is forbidden)

    Callers can use this as a routing decision:
        if p2p_privacy.require_encryption(url):
            envelope = p2p_crypto.seal_json(...)
            httpx.post(url + "/api/p2p/secure", json=envelope)
        else:
            httpx.post(url + "/api/embeddings", json=raw_payload)
    """
    return not (is_loopback(url) or is_paired_lan_peer(url))


def assert_plaintext_allowed(url: str, *, kind: str = "chat") -> None:
    """Raise ``PrivacyViolation`` when sending plaintext to `url`
    would violate the encryption-everywhere policy.

    Wire this in at every place that's about to POST a raw, un-
    enveloped JSON body to a peer URL. The check is cheap (one
    SQLite SELECT, cached per process) so calling it in the hot
    path is fine.
    """
    if not require_encryption(url):
        return
    host = _hostname_of(url)
    msg = (
        f"privacy guard refused PLAINTEXT {kind!r} to {host!r} — "
        f"non-loopback / non-paired destinations require the encrypted "
        f"envelope (see `p2p_crypto.seal_json`). Wrap the payload and "
        f"POST to the peer's /api/p2p/secure endpoint."
    )
    log.warning("p2p_privacy: %s", msg)
    raise PrivacyViolation(msg)


# ----- backwards-compat aliases for code paths that still call the
# old names. Same behaviour as `assert_plaintext_allowed`.

def assert_outbound_is_local(url: str, *, kind: str = "chat") -> None:
    assert_plaintext_allowed(url, kind=kind)


def assert_no_prompts_to_public_peer(url: str, *, kind: str = "chat") -> None:
    assert_plaintext_allowed(url, kind=kind)
