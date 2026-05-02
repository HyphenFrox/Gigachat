"""Public-pool routing — turn discovered peers into routable compute
workers when the user picks a model the local install doesn't have.

The flow when the user picks a model that the host + paired LAN
peers don't have installed:

  1. Look up the model in the local pool inventory cache
     (`p2p_pool_inventory.find_peers_with_model`) — this cache is
     populated by direct peer-to-peer queries of each peer's
     `/api/tags`, NOT by the rendezvous. Rendezvous knows nothing
     about which peers have which models.
  2. Pick a peer using the same fairness/capability heuristics the
     rest of the routing layer uses.
  3. Promote that peer in `paired_devices` from role='discovered'
     (which only allows `/api/tags`) to role='public' (full chat /
     embed allowed).
  4. Auto-create a `compute_workers` row pointing at the peer's
     STUN-discovered candidate so the existing routing picks it up.
     `use_encrypted_proxy=True` so the chat traffic flows over the
     X25519+ChaCha20 envelope — same wire-format as LAN-paired peers.

NAT traversal limitation
========================
Today the public-pool path only succeeds when the peer's Gigachat
is publicly reachable at one of its STUN-discovered candidates.
Symmetric-NAT peers (most home routers without UPnP) won't be
reachable directly — they'd need a TURN relay or the not-yet-built
QUIC hole-punching client. This is documented; users see a clear
"peer unreachable" error rather than a hang.

Auto-pull from official source
==============================
Per the project's privacy policy, model bytes NEVER flow peer-to-peer
over the public swarm (would bottleneck on slow home internet). When
the executing machine doesn't have the model, the executing machine
pulls it from the OFFICIAL Ollama registry directly
(`ollama pull <name>`). For the host that's a local subprocess; for
paired peers it's a request via the encrypted proxy that triggers
their local Ollama to pull. Either way no model bytes cross the swarm.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from . import db, p2p_pool_inventory

log = logging.getLogger("gigachat.p2p.pool_routing")


def _public_pool_enabled() -> bool:
    val = db.get_setting("p2p_public_pool_enabled")
    if val is None:
        return True
    return str(val).lower() in ("1", "true", "yes", "on")


async def ensure_public_peer_worker(model_name: str) -> dict | None:
    """Find a public-pool peer offering ``model_name`` and ensure it
    appears in ``compute_workers`` so the existing routing picks it up.

    Returns the worker dict on success, None when:
      * Public pool is toggled off.
      * Rendezvous is unreachable / not configured.
      * No peer in the swarm currently offers this model.
      * The candidate peers all lack a STUN candidate we can reach.

    Idempotent — calling twice with the same model picks the same
    peer (or refreshes their address) without duplicate rows.
    """
    if not _public_pool_enabled():
        return None
    # Force a fresh inventory if the cache is stale — the user just
    # picked a model and we don't want them waiting on the next 5-min
    # refresh tick.
    try:
        await p2p_pool_inventory.ensure_fresh(max_age_sec=120.0)
    except Exception as e:
        log.debug("public-pool: ensure_fresh failed: %s", e)
    peers = p2p_pool_inventory.find_peers_with_model(model_name)
    if not peers:
        return None
    # Prefer a peer with both STUN-discovered candidates AND an
    # X25519 pubkey on file — both are required for the encrypted
    # proxy to function. Others stay in the candidate list as a
    # fallback so a partially-registered peer is still tried last.
    def _score(p: dict) -> tuple[int, int]:
        has_x25519 = 1 if p.get("x25519_public_b64") else 0
        cand_count = len(p.get("candidates") or [])
        return (has_x25519, cand_count)
    peers.sort(key=_score, reverse=True)

    for peer in peers:
        device_id = peer.get("device_id")
        x25519 = peer.get("x25519_public_b64")
        public_key = peer.get("public_key_b64")
        if not all((device_id, x25519, public_key)):
            continue
        # Pick the best candidate. Prefer "stun" (NAT-public) over
        # "lan" (only useful when peer is on same network) over the
        # rest. STUN candidates may not actually be reachable for
        # symmetric-NAT peers — caller will get a connect error
        # downstream and the routing layer falls through to next.
        candidates = peer.get("candidates") or []
        best_cand = next(
            (c for c in candidates if c.get("source") == "stun"),
            candidates[0] if candidates else None,
        )
        if not best_cand:
            continue
        ip = best_cand.get("ip")
        port = int(best_cand.get("port") or 0)
        if not ip or not port:
            continue

        # Persist trust anchors as a paired device with role='public'.
        # The encrypted-proxy server checks paired_devices for inbound
        # envelopes; tagging role='public' lets the UI tell the user
        # this peer was rendezvous-vouched (not PIN-paired).
        try:
            db.upsert_paired_device(
                device_id=device_id,
                public_key_b64=public_key,
                label=device_id,  # rendezvous doesn't carry friendly labels
                ip=ip,
                port=port,
                role="public",
                x25519_public_b64=x25519,
            )
        except Exception as e:
            log.warning(
                "public-pool: upsert_paired_device failed for %s: %s",
                device_id, e,
            )
            continue

        # Materialise as a compute_worker so the existing routing
        # picks it up. Same defaults the LAN pair flow uses except
        # role-tagged via gigachat_device_id (the routing layer
        # doesn't care about the role; the UI does).
        try:
            existing = db.get_compute_worker_by_device_id(device_id)
            if existing:
                # Refresh address only — capability data + scoring
                # state preserved.
                db.update_compute_worker_address(
                    existing["id"],
                    address=ip,
                    ollama_port=port,
                )
                worker = db.get_compute_worker(existing["id"])
                wid = existing["id"]
            else:
                wid = db.create_compute_worker(
                    label=f"public:{device_id}",
                    address=ip,
                    ollama_port=port,
                    enabled=True,
                    use_for_chat=True,
                    use_for_embeddings=True,
                    use_for_subagents=True,
                    gigachat_device_id=device_id,
                    use_encrypted_proxy=True,
                )
                worker = db.get_compute_worker(wid)
            # Seed capabilities.models from what we already know about
            # this peer (via the inventory cache). Without this, the
            # very next routing call goes through `_eligible_workers`
            # which skips workers whose `_worker_has_model` returns
            # False — which it would until the periodic probe loop
            # fires. Seeding makes the new worker immediately routable
            # for the model we just registered them for. Background
            # probe later overwrites with full capabilities.
            try:
                caps_seed = (worker or {}).get("capabilities") or {}
                # Preserve any existing capabilities (refresh path);
                # only overlay models so we don't blow away VRAM /
                # GPU info on re-registration.
                seeded_models = list(caps_seed.get("models") or [])
                seen_names = {m.get("name") for m in seeded_models}
                for m in peer.get("models") or []:
                    if m.get("name") and m["name"] not in seen_names:
                        seeded_models.append(m)
                caps_seed["models"] = seeded_models
                db.update_compute_worker_capabilities(
                    wid, capabilities=caps_seed,
                    last_seen=time.time(),
                    last_error="",
                )
                worker = db.get_compute_worker(wid)
            except Exception as e:
                log.debug(
                    "public-pool: seed capabilities failed for %s: %s",
                    device_id, e,
                )

            # Seed live-stats + bandwidth + latency on registration so
            # the next routing decision has REAL numbers, not zeros.
            # Without this, a public peer's `probe_latency_ms`
            # defaults to 0 and `worst_lan_latency_ms <= 150` passes
            # trivially — split engagement engages a 200+ ms public
            # peer as if it were a LAN peer. The probe is async-safe
            # because we're already in an async function.
            try:
                from . import compute_pool as _pool
                worker_for_probe = db.get_compute_worker(wid)
                if worker_for_probe:
                    import time as _t
                    t0 = _t.perf_counter()
                    stats = await _pool.probe_worker_live_stats(
                        worker_for_probe, timeout=8.0,
                    )
                    rtt_ms = int((_t.perf_counter() - t0) * 1000)
                    bw = await _pool.probe_worker_bandwidth(worker_for_probe)
                    if stats or rtt_ms > 0:
                        caps = dict(worker_for_probe.get("capabilities") or {})
                        if stats:
                            caps["ram_free_gb"] = float(stats.get("ram_free_gb") or 0)
                            caps["ram_total_gb"] = float(stats.get("ram_total_gb") or 0)
                            caps["vram_total_gb"] = float(stats.get("vram_total_gb") or 0)
                            caps["gpu_kind"] = stats.get("gpu_kind") or ""
                        if rtt_ms > 0:
                            caps["probe_latency_ms"] = rtt_ms
                        if bw > 0:
                            caps["bandwidth_mbps"] = bw
                            caps["bandwidth_probed_at"] = time.time()
                        db.update_compute_worker_capabilities(
                            wid, capabilities=caps,
                        )
                        worker = db.get_compute_worker(wid)
                        log.info(
                            "public-pool: probed %s rtt=%dms bw=%.2f MB/s",
                            device_id, rtt_ms, bw,
                        )
            except Exception as e:
                log.debug(
                    "public-pool: live probe on registration failed for "
                    "%s: %s", device_id, e,
                )
            log.info(
                "public-pool: registered peer %s as compute worker for %r",
                device_id, model_name,
            )
            return worker
        except Exception as e:
            log.warning(
                "public-pool: compute_worker creation failed for %s: %s",
                device_id, e,
            )
            continue
    return None


# ---------------------------------------------------------------------------
# Auto-pull from the OFFICIAL Ollama registry (NOT peer-to-peer transfer)
# ---------------------------------------------------------------------------

async def auto_pull_on_host(
    model_name: str,
    *,
    timeout_sec: float = 600.0,
    on_progress: Any = None,
) -> bool:
    """Trigger ``ollama pull`` against the local Ollama instance.

    Used when the user picks a model and the host doesn't have it.
    Uses Ollama's HTTP API (the same /api/pull endpoint the CLI
    drives) so we can stream progress to the UI.

    ``on_progress`` is an optional async callable that receives a
    progress dict every time Ollama emits a status line. Shape:

      {
        "status": "downloading|verifying|writing|success|...",
        "digest": "sha256:...",            # only for layer-level events
        "completed": <int bytes>,          # only when downloading
        "total": <int bytes>,              # only when downloading
        "percent": <float 0-100>,          # derived helper
      }

    The agent provides one that bridges to the SSE turn so the user
    sees a "Downloading model… 23% (1.2/5.4 GB)" indicator instead of
    a silent multi-minute hang. Errors in ``on_progress`` are
    swallowed — surfacing progress is best-effort.

    Returns True on success, False otherwise. Failure is logged at
    INFO; the caller surfaces a user-friendly error.

    Per the project's privacy policy: model bytes come from the
    OFFICIAL registry (registry.ollama.ai), NOT from another peer.
    Peers' upload bandwidth is not consumed.
    """
    if not model_name:
        return False
    url = "http://127.0.0.1:11434/api/pull"
    body = {"name": model_name, "stream": True}
    import json as _json
    try:
        timeout = httpx.Timeout(connect=10.0, read=timeout_sec, write=30.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=body) as r:
                if r.status_code >= 400:
                    body_bytes = await r.aread()
                    log.info(
                        "auto-pull: HTTP %d for %r: %s",
                        r.status_code, model_name,
                        body_bytes[:200].decode("utf-8", errors="replace"),
                    )
                    return False
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    # Parse the NDJSON status line so we can surface
                    # progress to the caller. Tolerate malformed
                    # lines silently — Ollama occasionally emits
                    # blank or partial chunks during reconnect.
                    if on_progress is None:
                        continue
                    try:
                        evt = _json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(evt, dict):
                        continue
                    completed = evt.get("completed")
                    total = evt.get("total")
                    pct = None
                    if (
                        isinstance(completed, (int, float))
                        and isinstance(total, (int, float))
                        and total > 0
                    ):
                        pct = max(0.0, min(100.0, (completed / total) * 100.0))
                    payload = {
                        "model": model_name,
                        "status": evt.get("status") or "",
                        "digest": evt.get("digest") or "",
                        "completed": completed,
                        "total": total,
                        "percent": pct,
                    }
                    try:
                        await on_progress(payload)
                    except Exception:
                        # Best-effort callback — never block the
                        # actual download on a UI-side failure.
                        pass
        log.info("auto-pull: completed pull of %r on host", model_name)
        return True
    except Exception as e:
        log.info("auto-pull: %r failed: %s", model_name, e)
        return False


async def auto_pull_on_worker(worker: dict, model_name: str) -> bool:
    """Trigger ``ollama pull`` on a paired peer via the encrypted proxy.

    Same as `auto_pull_on_host` but routes the pull request through
    the peer's `/api/p2p/secure/forward` so the request is wrapped
    in an envelope. The peer's Gigachat decrypts and triggers their
    local Ollama to pull from the OFFICIAL registry — model bytes
    NEVER flow peer-to-peer through Gigachat.

    Returns True on success, False on failure.
    """
    if not worker.get("use_encrypted_proxy"):
        return False
    if not worker.get("gigachat_device_id"):
        return False
    try:
        from . import p2p_secure_client as _sec
        # /api/pull is whitelisted in p2p_secure_proxy._FORWARDABLE_PATHS
        # for paired peers (LAN + public). Discovered peers (those we
        # only know about via /peers but haven't accepted into our
        # pool) cannot trigger pulls — see _DISCOVERED_PEER_PATHS.
        status, body = await _sec.forward(
            worker, method="POST", path="/api/pull",
            body={"name": model_name, "stream": False},
        )
        if status >= 400:
            log.info(
                "auto-pull: peer %s returned %d for pull %r: %s",
                worker.get("label"), status, model_name, body[:200],
            )
            return False
        log.info(
            "auto-pull: peer %s completed pull of %r",
            worker.get("label"), model_name,
        )
        return True
    except Exception as e:
        log.info(
            "auto-pull: peer %s pull %r failed: %s",
            worker.get("label"), model_name, e,
        )
        return False
