"""Web Push (VAPID) utilities.

This module owns three concerns:

  1. **VAPID key lifecycle.** On first use we generate a P-256 keypair with
     `cryptography` and persist it under `data/vapid.json`. Subsequent runs
     reuse the same keys so a returning browser's stored subscription stays
     valid (the endpoint URL is keyed on the VAPID public key).

  2. **Sending a push.** `send_to_all(payload)` iterates every saved
     subscription and fires an encrypted payload via `pywebpush`. Endpoints
     that the push service says are permanently gone (HTTP 404 / 410) are
     deleted from the DB so a dead browser doesn't cause a retry storm.

  3. **Public-key export.** `vapid_public_key_b64url()` returns the browser-
     friendly base64url-encoded form the service worker feeds to
     `pushManager.subscribe({applicationServerKey: …})`.

Security notes
--------------
  - The VAPID private key never leaves the server. The JSON file is created
    with permission 0600 on POSIX; on Windows the file is also hidden-only.
  - Push payloads are end-to-end encrypted by `pywebpush` with the
    subscription's own p256dh/auth keys — the push service cannot read them.
  - We cap payload size at 3 KB so a pathological caller can't send a
    megabyte-long notification body (most browsers reject >4KB anyway).
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

# pywebpush is imported lazily inside `send_to_all` because it
# transitively pulls in aiohttp (~250 ms of cold-start time on
# Windows). Push notifications are an opt-in feature — most users
# never send one — so paying the import cost up-front penalises
# every backend boot for the minority that do. The two functions
# from pywebpush we use (`webpush`, `WebPushException`) are only
# referenced inside `send_to_all`, so deferral is safe.

from . import db

# Keys live alongside app.db under data/. Separate file (not a DB column) so
# rotating the keys is as simple as deleting this file; the next send_to_all
# call regenerates on demand.
_VAPID_PATH = Path(__file__).resolve().parent.parent / "data" / "vapid.json"

# "Contact email" exposed in the VAPID JWT — required by some push services.
# We default to a mailto: with the local user; override via GIGACHAT_VAPID_SUB.
_VAPID_SUB = os.environ.get("GIGACHAT_VAPID_SUB", "mailto:admin@gigachat.local")

# Hard cap on a single outbound payload (before encryption). 3 KB is well
# below the 4 KB Chrome / FCM limit and keeps telemetry honest.
MAX_PUSH_PAYLOAD_BYTES = 3072


# ---------------------------------------------------------------------------
# VAPID key storage
# ---------------------------------------------------------------------------
def _b64url(data: bytes) -> str:
    """URL-safe base64 without padding — the format Web Push expects."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _generate_vapid_keys() -> dict[str, str]:
    """Generate a fresh P-256 keypair and return base64url-encoded strings.

    - `private_key` is the 32-byte secret scalar (NOT a full PEM — pywebpush
      accepts the raw b64url form).
    - `public_key` is the uncompressed 65-byte form (leading 0x04 + X + Y),
      which is exactly what `applicationServerKey` in the browser needs.
    """
    priv = ec.generate_private_key(ec.SECP256R1(), default_backend())
    priv_number = priv.private_numbers().private_value
    priv_bytes = priv_number.to_bytes(32, "big")

    pub = priv.public_key()
    pub_bytes = pub.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )

    return {
        "private_key": _b64url(priv_bytes),
        "public_key": _b64url(pub_bytes),
    }


def _write_keys_securely(path: Path, data: dict[str, str]) -> None:
    """Write the VAPID JSON with restrictive permissions.

    On POSIX we chmod 0600 so only the owning user can read the secret. On
    Windows we rely on the data/ directory's ACL — if the app is running as
    the user, no other non-admin user can read the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_CREAT | O_WRONLY | O_TRUNC is what json.dump would do via open()
    # but we want 0600 from the start rather than racing a chmod after.
    flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
    mode = 0o600 if sys.platform != "win32" else 0o666
    fd = os.open(str(path), flags, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def load_or_create_vapid_keys() -> dict[str, str]:
    """Return the persisted VAPID keypair, generating it on first run."""
    if _VAPID_PATH.is_file():
        try:
            data = json.loads(_VAPID_PATH.read_text(encoding="utf-8"))
            if "private_key" in data and "public_key" in data:
                return data
        except (OSError, json.JSONDecodeError):
            # File is corrupt — regenerate so the feature recovers instead of
            # erroring on every call.
            pass
    keys = _generate_vapid_keys()
    _write_keys_securely(_VAPID_PATH, keys)
    return keys


def vapid_public_key_b64url() -> str:
    """Public key in the form the browser expects for applicationServerKey."""
    return load_or_create_vapid_keys()["public_key"]


# ---------------------------------------------------------------------------
# Outbound push
# ---------------------------------------------------------------------------
def _subscription_to_dict(row: dict) -> dict:
    """Shape a DB row to what pywebpush.webpush() wants as `subscription_info`."""
    return {
        "endpoint": row["endpoint"],
        "keys": {"p256dh": row["p256dh"], "auth": row["auth"]},
    }


def send_to_all(payload: dict[str, Any]) -> dict[str, int]:
    """Fan out a payload to every registered browser.

    Returns a small summary ({sent, pruned, failed}) so callers can log how
    the send went without needing to iterate the DB themselves.

    Any 404/410 from a push service means the subscription is permanently
    gone (browser uninstall, push service rotation) — we prune those so the
    table doesn't grow unbounded and we don't keep trying to contact dead
    endpoints.
    """
    subs = db.list_push_subscriptions()
    if not subs:
        return {"sent": 0, "pruned": 0, "failed": 0}

    keys = load_or_create_vapid_keys()
    # pywebpush expects the private key as raw b64url.
    vapid_private = keys["private_key"]

    body = json.dumps(payload, ensure_ascii=False)
    if len(body.encode("utf-8")) > MAX_PUSH_PAYLOAD_BYTES:
        raise ValueError(
            f"push payload too large "
            f"({len(body)} bytes, max {MAX_PUSH_PAYLOAD_BYTES})"
        )

    # Lazy pywebpush import — see module-level comment for the
    # cold-start rationale. Module is loaded on first `send_to_all`
    # call and cached in sys.modules thereafter.
    from pywebpush import WebPushException, webpush

    sent = 0
    pruned = 0
    failed = 0
    for row in subs:
        try:
            webpush(
                subscription_info=_subscription_to_dict(row),
                data=body,
                vapid_private_key=vapid_private,
                vapid_claims={"sub": _VAPID_SUB},
                ttl=60 * 60 * 24,  # up to a day of offline buffering
            )
            sent += 1
        except WebPushException as e:
            # Inspect the response code where available. 404/410 = Gone.
            status = getattr(e.response, "status_code", None) if e.response else None
            if status in (404, 410):
                db.delete_push_subscription(row["endpoint"])
                pruned += 1
            else:
                failed += 1
                print(
                    f"[push] {status or '?'} sending to "
                    f"{row['endpoint'][:60]}…: {e}",
                    file=sys.stderr,
                )
        except Exception as e:
            failed += 1
            print(f"[push] unexpected error: {e!r}", file=sys.stderr)

    return {"sent": sent, "pruned": pruned, "failed": failed}
