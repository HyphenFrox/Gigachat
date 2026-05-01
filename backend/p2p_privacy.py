"""Privacy guard for the P2P pool.

Hard contract: the user's prompts NEVER leave their local pool.

Definitions (per the user's policy):
  * **Local pool peer** — host's own loopback Ollama OR any device
    the user has explicitly added on their LAN. In practice this
    means every row in the `compute_workers` table — both the rows
    the user typed into Settings → Compute manually AND the rows
    auto-created when the user pairs a device on the same LAN via
    Settings → Network (the pair flow creates a worker row keyed
    by the device id). Either way the trust anchor is "the user
    explicitly included this destination on their LAN".

  * **Non-local peer** — any destination NOT in `compute_workers`.
    Public-pool peers found via the internet rendezvous, friend
    peers added by public-key exchange across the internet,
    anything reachable only via QUIC NAT-traversal. These are
    donate-only targets: the user's spare compute can run public
    workloads for these peers when Public Pool is on, but the
    user's prompts never go out.

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
    """Raised when something tries to send prompt-bearing traffic to a
    non-local-pool destination. Fail-closed: callers should let the
    exception propagate. The user's prompts being leaked is a worse
    outcome than the request failing.
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


def check_outbound_is_local(url: str) -> bool:
    """Fast boolean check. Returns True iff the destination is in
    the local-pool whitelist.

    Use this when you want to BRANCH on local vs. remote (e.g.
    "send the prompt only if local; otherwise fall back to host-only
    inference"). Use ``assert_outbound_is_local`` when the call is
    SUPPOSED to be local and a non-local destination is a bug.
    """
    host = _hostname_of(url)
    if not host:
        return False
    if host in _LOOPBACK_HOSTS:
        return True
    return host in _local_pool_hosts()


def assert_outbound_is_local(
    url: str, *, kind: str = "chat",
) -> None:
    """Raise ``PrivacyViolation`` if the destination isn't local pool.

    `kind` is a free-text descriptor that ends up in the error
    message and the audit log. Conventional values:
      * "chat"      — main-loop inference (carries user prompts)
      * "embed"     — semantic recall / search (sees query text)
      * "subagent"  — delegated subagent (carries user prompt)
      * "tool"      — generic tool-call dispatch
    Any of these is prompt-bearing in our threat model and must
    stay local.
    """
    host = _hostname_of(url)
    if check_outbound_is_local(url):
        return
    msg = (
        f"privacy guard refused {kind!r} to {host!r} — destination is "
        f"not in the local pool whitelist. The local pool consists of "
        f"loopback addresses + every host in `compute_workers`. To "
        f"include a peer, pair it via Settings → Network or add it "
        f"manually under Settings → Compute."
    )
    log.warning("p2p_privacy: %s", msg)
    raise PrivacyViolation(msg)


def assert_no_prompts_to_public_peer(
    url: str, *, kind: str = "chat",
) -> None:
    """Alias used by the future P2P transport at the call site that
    sends donation-only workloads to public-pool peers. The transport
    is expected to mark its outbound traffic with a "public_workload"
    flag (e.g. an HTTP header); anything that ISN'T marked must pass
    this check.

    Today this is identical to `assert_outbound_is_local`. Kept as a
    separate symbol so the future transport call sites read cleanly:
        # Donation-only outbound to a public peer:
        if not is_marked_public_workload(req):
            p2p_privacy.assert_no_prompts_to_public_peer(url, kind="chat")
    """
    assert_outbound_is_local(url, kind=kind)
