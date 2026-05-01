"""Regression: TURN-style relay client (request/response correlation).

Pins the contracts callers depend on:

  * `forward_via_relay` parks a Future keyed by `_relay_req_id` and
    returns the matched response when the inbox loop sets it.
  * Stale / unmatched response envelopes are dropped (no exception,
    no leaked Future).
  * Timeout cleans up the pending Future even on the failure path
    so a slow round-trip can't grow `_pending_responses` without bound.
"""
from __future__ import annotations

import asyncio
import base64

import pytest

from backend import p2p_crypto, p2p_relay, identity


pytestmark = pytest.mark.smoke


def test_forward_via_relay_matches_response(monkeypatch):
    """Happy path: the inbox dispatcher resolves the pending Future
    when a response envelope carries the same `_relay_req_id` we
    sent. forward_via_relay returns (status, body) extracted from
    the response payload.

    We exercise the correlation through `_pending_responses` directly
    rather than via `_dispatch_inbound_envelope` so this test stays
    focused on forward_via_relay's contract — the dispatcher's
    verify path is covered by `test_dispatch_unknown_rid_is_dropped_silently`
    and the integration test against the real rendezvous.
    """
    me = identity.get_identity()
    captured: list[dict] = []

    async def _stub_relay_send(recipient_device_id, envelope):
        captured.append(envelope)
        return True
    monkeypatch.setattr(p2p_relay, "relay_send", _stub_relay_send)

    async def _scenario():
        forward_task = asyncio.create_task(
            p2p_relay.forward_via_relay(
                recipient_device_id=me.device_id,
                recipient_x25519_pub_b64=me.x25519_public_b64,
                recipient_ed25519_pub_b64=me.public_key_b64,
                method="GET",
                path="/api/tags",
                timeout_sec=5.0,
            ),
        )
        # Tick so the Future is registered in _pending_responses.
        await asyncio.sleep(0.05)
        assert len(captured) == 1

        # Decrypt the outbound envelope to read the request id.
        req_payload, _ = p2p_crypto.open_envelope_json(
            captured[0],
            expected_sender_ed25519_pub_b64=me.public_key_b64,
        )
        rid = req_payload.get("_relay_req_id")
        assert isinstance(rid, str) and rid
        assert rid in p2p_relay._pending_responses

        # Resolve the Future as the dispatcher would after verify.
        p2p_relay._pending_responses[rid].set_result({
            "status": 200,
            "content_type": "application/json",
            "body": '{"models":[]}',
            "_relay_req_id": rid,
        })

        status, body = await forward_task
        assert status == 200
        assert body == '{"models":[]}'

    asyncio.run(_scenario())


def test_forward_via_relay_times_out_cleanly(monkeypatch):
    """No matching response arrives — forward_via_relay raises and
    the pending Future is removed so memory doesn't leak."""
    me = identity.get_identity()

    async def _stub_relay_send(recipient_device_id, envelope):
        return True
    monkeypatch.setattr(p2p_relay, "relay_send", _stub_relay_send)

    async def _scenario():
        before_count = len(p2p_relay._pending_responses)
        with pytest.raises((RuntimeError, asyncio.TimeoutError)):
            await p2p_relay.forward_via_relay(
                recipient_device_id=me.device_id,
                recipient_x25519_pub_b64=me.x25519_public_b64,
                recipient_ed25519_pub_b64=me.public_key_b64,
                method="GET",
                path="/api/tags",
                timeout_sec=0.5,
            )
        # Cleanup ran in the finally block.
        assert len(p2p_relay._pending_responses) == before_count

    asyncio.run(_scenario())


def test_dispatch_unknown_rid_is_dropped_silently():
    """A response envelope arriving for a req_id we don't know
    about (timed-out future, peer retry, malicious replay) must
    not crash the inbox loop OR leak state."""
    me = identity.get_identity()
    bogus_env = p2p_crypto.seal_json(
        recipient_x25519_pub_b64=me.x25519_public_b64,
        recipient_device_id=me.device_id,
        payload={"status": 200, "body": "ignored", "_relay_req_id": "ghost"},
    )

    async def _scenario():
        # Should NOT raise.
        await p2p_relay._dispatch_inbound_envelope(bogus_env)

    asyncio.run(_scenario())
