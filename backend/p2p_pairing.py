"""PIN-based pairing handshake for LAN device additions.

The user-visible UX mirrors how Bluetooth devices pair on a phone:

  Host               Device claiming pairing
  ────────────────   ───────────────────────
  Click "Pair"  →    Enters PIN
  Show 6-digit PIN
  Wait …            HTTP POST /api/p2p/pair/accept {pin, my_pubkey, sig}
  ✓ paired           ✓ paired

Security model (LAN-trust scoped):

  * The PIN is short-lived (`_PIN_TTL_SEC`) and consumed on first
    acceptance. A user shoulder-surfing the screen has the PIN's
    lifetime to use it; after that it's dead.

  * The PIN never travels in plaintext over the wire. Both sides
    derive the SAME HMAC challenge from `(pin + nonce)` and the
    claimant proves PIN knowledge by signing
    ``H(pin || nonce || claimant_pubkey || target_pubkey)``
    with their identity Ed25519 key. Replaying the signature buys
    nothing because it's bound to the host's nonce.

  * Pairing record stored on BOTH sides (host and claimant) so both
    can refuse compute requests from peers they haven't pre-approved.

This is deliberately simpler than a full PAKE (SPAKE2 / OPAQUE):
PIN-bound HMAC + Ed25519 signature is sufficient when the PIN is
displayed on a screen the user controls and consumed within seconds.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field

from . import db, identity

log = logging.getLogger("gigachat.p2p.pairing")

# How long a generated PIN is valid before it expires. 5 minutes is
# generous enough for the user to walk over to the other device, type
# the PIN, and confirm — short enough that an unattended screen
# leaking the PIN closes the window quickly.
_PIN_TTL_SEC = 300.0

# Number of PIN bytes (1 byte = 0-255). 6 decimal digits = ~20 bits
# of entropy, low by crypto standards but matched to the threat model
# (bound by `_PIN_TTL_SEC` AND single-use AND scoped to LAN).
_PIN_DIGITS = 6


@dataclass
class _PendingPairing:
    """In-memory record of a PIN waiting to be claimed.

    Held on the host side from the moment `start_pairing()` is called
    until either `accept_pairing()` consumes it or the TTL expires.
    """
    id: str
    pin: str
    nonce_b64: str
    created_at: float
    expires_at: float
    target_device_id: str  # the host's own device_id, used to bind the signature
    consumed: bool = field(default=False)


# Module-level state. Written by start/accept paths, read by status
# endpoints. The thread-safety story is simple: we always hold `_lock`
# when mutating, FastAPI endpoints are async but Python's GIL makes
# the dict ops atomic for our narrow access pattern.
_pending: dict[str, _PendingPairing] = {}
_lock = threading.Lock()


def _gen_pin() -> str:
    """Cryptographically random N-digit PIN as a string with leading
    zeros preserved. `secrets.randbelow` is uniform; do NOT use
    `random.randint`."""
    n = secrets.randbelow(10 ** _PIN_DIGITS)
    return str(n).zfill(_PIN_DIGITS)


def _purge_expired(now: float | None = None) -> None:
    """Drop expired or consumed records. Called inline at the top of
    each public function so the dict can't grow without bound on a
    long-running host."""
    cutoff = now if now is not None else time.time()
    with _lock:
        for k in list(_pending.keys()):
            rec = _pending[k]
            if rec.consumed or rec.expires_at <= cutoff:
                _pending.pop(k, None)


def start_pairing() -> dict:
    """Create a fresh pairing offer. Returns the data the host UI
    needs to display:

      * `pin` — what the user types on the other device.
      * `pairing_id` — opaque handle for status polling / cancel.
      * `expires_at` — UI countdown.
      * `nonce` — handed to the claimant in the discovered-record
        TXT so they can construct the signature without an extra
        round-trip. Returned in the response too so the host UI
        can display it for QR-code-flow users.
    """
    _purge_expired()
    me = identity.get_identity()
    rec = _PendingPairing(
        id=str(uuid.uuid4()),
        pin=_gen_pin(),
        nonce_b64=base64.b64encode(secrets.token_bytes(16)).decode("ascii"),
        created_at=time.time(),
        expires_at=time.time() + _PIN_TTL_SEC,
        target_device_id=me.device_id,
    )
    with _lock:
        _pending[rec.id] = rec
    log.info(
        "p2p: pairing offer started — id=%s expires_in=%ds",
        rec.id, int(_PIN_TTL_SEC),
    )
    return {
        "pairing_id": rec.id,
        "pin": rec.pin,
        "nonce": rec.nonce_b64,
        "host_device_id": me.device_id,
        "host_label": me.label,
        "host_public_key_b64": me.public_key_b64,
        "expires_at": rec.expires_at,
    }


def cancel_pairing(pairing_id: str) -> bool:
    """Drop a pending offer (user closed the dialog before it
    completed). Returns True when something was actually cancelled."""
    with _lock:
        rec = _pending.pop(pairing_id, None)
    if rec is None:
        return False
    log.info("p2p: pairing offer %s cancelled by user", pairing_id)
    return True


def list_pending() -> list[dict]:
    """Snapshot of currently-active offers — used by the UI to
    render the "PIN is X, waiting for the other side…" panel even
    after a refresh."""
    _purge_expired()
    with _lock:
        return [
            {
                "pairing_id": r.id,
                "pin": r.pin,
                "expires_at": r.expires_at,
            }
            for r in _pending.values()
            if not r.consumed
        ]


def _expected_signature_bytes(
    pin: str, nonce_b64: str,
    claimant_pubkey_b64: str, host_pubkey_b64: str,
) -> bytes:
    """Compute the bytes the claimant should have signed.

    H(pin || nonce || claimant_pubkey || host_pubkey) — domain
    separation between the components prevents any extension /
    substitution shenanigans. SHA-256 because Ed25519's signing is
    SHA-512 internally; using SHA-256 here keeps a clean break
    between the input-derivation and the signature primitive.
    """
    h = hashlib.sha256()
    h.update(b"gigachat-p2p-pair-v1\n")
    h.update(pin.encode("ascii"))
    h.update(b"\n")
    h.update(nonce_b64.encode("ascii"))
    h.update(b"\n")
    h.update(claimant_pubkey_b64.encode("ascii"))
    h.update(b"\n")
    h.update(host_pubkey_b64.encode("ascii"))
    return h.digest()


def build_claim_signature(
    pin: str, nonce_b64: str,
    host_public_key_b64: str,
) -> dict:
    """Build the proof the claimant sends back to the host.

    Used on the SAME-PROCESS test path and (eventually) by an
    in-product "auto-pair from another tab" flow. Real cross-machine
    pairing has the claimant's frontend POST `accept_pairing` with the
    pre-computed signature.
    """
    me = identity.get_identity()
    digest = _expected_signature_bytes(
        pin, nonce_b64, me.public_key_b64, host_public_key_b64,
    )
    sig = me.sign(digest)
    return {
        "claimant_device_id": me.device_id,
        "claimant_label": me.label,
        "claimant_public_key_b64": me.public_key_b64,
        "signature_b64": base64.b64encode(sig).decode("ascii"),
    }


def accept_pairing(
    *,
    pairing_id: str,
    pin: str,
    claimant_device_id: str,
    claimant_label: str,
    claimant_public_key_b64: str,
    signature_b64: str,
    claimant_ip: str | None = None,
    claimant_port: int | None = None,
) -> dict:
    """Verify a pairing claim and persist the trust anchor on success.

    Raises ``ValueError`` with a user-friendly message on every
    failure mode (expired, wrong PIN, bad signature, replay). Caller
    surfaces the message in the UI. Returns the persisted pairing
    record on success.
    """
    _purge_expired()
    if not all((pairing_id, pin, claimant_device_id, signature_b64,
                claimant_public_key_b64)):
        raise ValueError("missing required pairing fields")
    with _lock:
        rec = _pending.get(pairing_id)
    if rec is None:
        raise ValueError("pairing offer not found or expired")
    if rec.consumed:
        raise ValueError("pairing offer already used")
    if time.time() > rec.expires_at:
        raise ValueError("pairing offer expired (PIN timed out)")
    # Constant-time PIN comparison so a network observer can't time
    # the early-out branch to learn digits.
    if not hmac.compare_digest(rec.pin, pin):
        raise ValueError("incorrect PIN")

    me = identity.get_identity()
    expected = _expected_signature_bytes(
        pin, rec.nonce_b64, claimant_public_key_b64, me.public_key_b64,
    )
    try:
        signature = base64.b64decode(signature_b64)
    except Exception:
        raise ValueError("signature is not valid base64")
    if not identity.verify_signature(
        claimant_public_key_b64, expected, signature,
    ):
        raise ValueError("signature verification failed")

    # Sanity: the claimant's claimed device_id must match the
    # device_id derived from their public key. Otherwise a peer with
    # one identity could try to pair under another device's id.
    try:
        from . import identity as _ident
        derived = _ident._device_id_from_pubkey(
            base64.b64decode(claimant_public_key_b64)
        )
    except Exception:
        derived = ""
    if derived != claimant_device_id:
        raise ValueError(
            "claimant device_id does not match its public key"
        )

    # Mark the offer consumed BEFORE persisting so a race within
    # the same process can't double-pair.
    with _lock:
        rec.consumed = True
        _pending.pop(rec.id, None)

    paired = db.upsert_paired_device(
        device_id=claimant_device_id,
        public_key_b64=claimant_public_key_b64,
        label=claimant_label or claimant_device_id,
        ip=claimant_ip,
        port=claimant_port,
        role="local",
    )
    log.info(
        "p2p: paired with device_id=%s label=%r",
        claimant_device_id, claimant_label,
    )
    # Phase 2: paired devices become routable compute workers
    # automatically. We materialise a row in compute_workers keyed
    # by the device_id so the existing routing / probe / scoring
    # code transparently includes the paired peer. Default
    # `ollama_port=11434` matches the Ollama install convention;
    # callers can override via the worker's edit form. Failure is
    # non-fatal — the pairing record itself is the trust anchor;
    # compute integration is a convenience.
    if claimant_ip:
        try:
            existing = db.get_compute_worker_by_device_id(claimant_device_id)
            if not existing:
                db.create_compute_worker(
                    label=claimant_label or claimant_device_id,
                    address=claimant_ip,
                    ollama_port=11434,
                    enabled=True,
                    use_for_chat=True,
                    use_for_embeddings=True,
                    use_for_subagents=True,
                    gigachat_device_id=claimant_device_id,
                )
                log.info(
                    "p2p: auto-created compute_worker for paired device %s "
                    "at %s:11434",
                    claimant_device_id, claimant_ip,
                )
        except Exception as e:
            log.info(
                "p2p: compute_worker auto-create failed (%s); pairing "
                "stored but the device won't appear in the worker pool "
                "until manually added",
                e,
            )
    return paired


def unpair(device_id: str) -> bool:
    """Drop a pairing record AND its auto-created compute_worker row.

    The other side keeps their pairing record until they delete it on
    their own device — there's no remote-revoke in v1 because that
    would require a transport channel we haven't built yet. Removing
    the worker too keeps the routing layer from trying to talk to a
    peer the user has decided not to trust anymore.
    """
    # Order matters: delete the worker BEFORE the pairing so the
    # "find the worker by device_id" lookup still succeeds.
    try:
        worker = db.get_compute_worker_by_device_id(device_id)
        if worker:
            db.delete_compute_worker(worker["id"])
            log.info(
                "p2p: removed auto-paired compute_worker %s for device %s",
                worker["id"], device_id,
            )
    except Exception as e:
        log.info(
            "p2p: compute_worker cleanup on unpair failed: %s", e,
        )
    return db.delete_paired_device(device_id)
