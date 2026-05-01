# P2P architecture — pairing, encryption, rendezvous, relay

Settings → **Compute pool** lets you pair other Gigachat installs the way phones pair Bluetooth devices: click, show a 6-digit PIN, type it on the other side, done. Once paired, all compute traffic between the two devices is **end-to-end encrypted** — anyone observing the network sees only ciphertext.

This document covers the cryptographic + networking internals. For end-user setup see the README's [Compute pool & P2P](../README.md#compute-pool--p2p) section.

---

## Identity & encryption keys

Every install generates two keypairs on first launch and stores them in `data/identity.json` (mode 0600 on POSIX):

- **Ed25519** — for signing. The public key, base32-truncated to 16 chars, is the device's `device_id` (e.g. `PBJ4-GCBV-5NH3-PGXH`). The trust anchor across IP changes, re-pairings, and reinstalls.
- **X25519** — for ECDH key agreement. Used to derive a per-pair AEAD key (HKDF-SHA256 over the ECDH output) for ChaCha20-Poly1305 envelope encryption.

Same primitives Signal and WhatsApp use; no homemade algorithms.

---

## Discovery & pairing

### LAN discovery

- **mDNS** — each install advertises `_gigachat._tcp.local.` (zero-conf / Bonjour / Avahi). The TXT record carries `device_id`, `label`, `version`, and the Ed25519 public key. Cross-platform: Windows mDNS service, macOS Bonjour, Linux Avahi all interoperate.

### PIN-based pair handshake

- Host generates a 6-digit PIN with a 5-minute TTL and a 16-byte nonce.
- The claimant signs `H("gigachat-p2p-pair-v1" || pin || nonce || claimant_pubkey || host_pubkey)` with their Ed25519 key, AND ships their X25519 pubkey alongside.
- Host verifies signature + cross-checks `device_id` derives from the public key, then stores the trust anchors.
- PIN is single-use (atomic consume) and replay-proof (signature binds to the host's per-offer nonce).

### Symmetric friendship

After the host accepts, an Ed25519-signed pair-notify is pushed to the claimant so both sides have a record. Either side's unpair triggers a signed unpair-notify so the friendship is removed symmetrically. The notify itself is wrapped in the encrypted envelope when both sides have each other's X25519 keys. The unilateral unpair endpoint (`DELETE /api/p2p/paired/{device_id}`) also drops the matching `compute_workers` row + clears the per-pair crypto cache so the routing layer has a single source of truth.

### Auto-reconnect

The mDNS browser stays running. When a paired device's advertisement reappears with the same `device_id` (different IP), the backend updates the address + the corresponding `compute_workers` row in place. The user does nothing.

---

## End-to-end encryption — every byte on the wire

After pairing, all peer-to-peer traffic flows through the encrypted compute proxy (`/api/p2p/secure/forward` and `/forward-stream`). Each payload is wrapped in a `p2p_crypto` v2 envelope:

```
{ "v":2, "sender":<id>, "sender_eph_x25519_pub":<b64>, "recipient":<id>,
  "nonce_b64":<12 random bytes>, "timestamp":<float>,
  "ciphertext_b64":<ChaCha20-Poly1305 output>,
  "signature_b64":<Ed25519 over (sender|recipient|nonce|timestamp|sender_eph_pub|sha256(plaintext))> }
```

What's protected:

| Data path | Mechanism |
|---|---|
| Embed request/response (carries query text + vector) | per-request envelope |
| Chat completion (carries full prompt history + token-by-token output) | per-NDJSON-chunk envelope |
| Worker probe (`/api/version`, `/api/tags`, `/api/ps`) | per-request envelope |
| Subagent fan-out to paired peers | per-NDJSON-chunk envelope |
| Pair-notify and unpair-notify metadata | wrapped envelope |
| Auto-pull on a peer (`/api/pull`) | per-request envelope |

### Defence in depth

- **Sender authenticity** pinned to the receiver's stored `paired_devices.public_key_b64` — substituting a different pubkey in the envelope fails the signature check.
- **AEAD tag** detects any byte tamper before plaintext is extracted.
- **Replay window** ±120 s — captured envelopes can't be replayed later.
- **Per-peer rate limit** 60 req/min on inbound — defends against compromised friend keypair flooding.
- **Envelope size cap** 256 KB inbound — memory-DoS defence.
- **Upstream response cap** 4 MB — one-way amplification defence.
- **Path whitelist** — even authenticated peers can only reach a strict set of Ollama endpoints (no admin / model-delete); discovered-but-not-paired peers get a tighter read-only subset.
- **Refuses peers without X25519 on file** — would mean we can't seal the response back; refuse cleanly.
- **Cache wipe on unpair** — revoking trust drops cached legacy key material immediately.

### Forward secrecy (sender-ephemeral)

Every v2 envelope generates a fresh X25519 ephemeral keypair on the sender side. The AEAD key is derived from `ECDH(eph_priv, recipient_long_term_pub)`; once `seal()` returns, the ephemeral private key is dropped. **Captured envelopes can NOT be decrypted later from sender-side compromise of long-term keys.**

Recipient-side compromise still reveals past traffic addressed to that recipient — full FS in both directions arrives with the **TLS-with-pinning** path on the streaming endpoints (TLS 1.3's ECDHE handshake gives full FS by exchanging ephemerals on both ends).

v1 (long-term ECDH) envelopes still verify on the receive path during the rolling upgrade.

### No key cache for v2

Caching the derived AEAD key per (eph_pub, recipient_pub) would defeat FS by leaving key material in memory after the ephemeral private key is destroyed; v2 derivation runs every envelope. Legacy v1 inbound still uses a per-pair cache with FIFO eviction (will disappear once v1 is removed).

---

## TLS-with-pinning (streaming paths)

- **Self-signed identity cert** — `backend/p2p_tls.ensure_identity_cert()` generates an X.509 cert whose Subject Pubkey is your Ed25519 identity pubkey (the same value that's already in `paired_devices.public_key_b64`). Stored alongside `identity.json` at `~/.gigachat/identity-cert.pem` (mode 0600 on POSIX). Auto-rotated when expiry is within 30 days OR identity has changed.
- **Pin-based trust, no CA** — peer certs are self-signed. `verify_peer_cert(cert_bytes, expected_ed25519_pub_b64)` does a constant-time compare against the value we already have on file (paired_devices / inventory cache / rendezvous /peers). No certificate authority is involved; the trust anchor IS the identity pubkey we'd otherwise verify Ed25519 sigs against.
- **Server SSL context** (`make_server_ssl_context`) — TLS 1.3-only, compression off, ready to plug into uvicorn's `ssl_certfile=` / `ssl_keyfile=`.
- **Client SSL context** (`make_pinned_client_ssl_context`) — accepts ANY cert chain at handshake; the caller invokes `verify_peer_cert(socket.getpeercert(binary_form=True), expected_pub)` before sending the request body. This is the pattern recommended by the cryptography lib for non-CA pinning.
- **Streaming port enablement** — opt-in via `GIGACHAT_TLS_PORT=<port>` env var. When set, `backend/server.py` spins up a second uvicorn instance bound to that port, sharing the same FastAPI app + identity cert. The HTTP port keeps serving the browser UI + non-streaming endpoints unchanged. Default deployments are byte-identical to before — TLS adoption is gradual.

---

## At-rest encryption (sensitive SQLite columns)

- **What's encrypted** — `messages.content` (every chat message body), `global_memories.content` (your saved memos), `project_memories.content` (per-cwd memos). All wrapped with ChaCha20-Poly1305 AEAD before INSERT/UPDATE; decrypted transparently in the row hydrators.
- **Master key** — derived via HKDF-SHA256 from your X25519 identity private key (`identity.json`). Held in memory only; re-derived on each app start. No user passphrase prompt — chat app stays "just works".
- **Threat model**: defends against backups / disk thefts that leak only `app.db` without `identity.json`. Does NOT defend against an attacker with both files (they can run our key derivation). For full-disk protection, a future opt-in passphrase wraps the master key.
- **Migration** — wrapping is opportunistic on writes. Existing legacy rows (plaintext) remain readable through the same `decrypt()` path which is a pass-through for unwrapped values.
- **Search fallback** — substring search across encrypted bodies decrypts in Python and filters; SQL `LIKE` only matches plaintext columns (titles, tags, etc.).

---

## Public pool & internet rendezvous

- **Rendezvous service — bootstrap discovery ONLY** — a tiny stateless FastAPI app deployable to GCP Cloud Run (`rendezvous/`). Holds only `(device_id, public_key, x25519_pubkey, [STUN candidates], last_seen_at)`; never sees prompts, model weights, chat data, OR what models each peer offers. Peers register every 30 s with an Ed25519-signed registration; lookup is by device_id. Same role Signal and BitTorrent give their trackers — pure peer index.
- **Default URL** — Gigachat ships pointing at the project's public Cloud Run instance (`backend/p2p_rendezvous._DEFAULT_RENDEZVOUS_URL`). Override per-install via Settings → Compute pool → rendezvous URL, or `GIGACHAT_RENDEZVOUS_URL` env var. Self-host your own rendezvous from the `rendezvous/` folder if you want full control of the discovery server.
- **Local pool inventory** (`backend/p2p_pool_inventory.py`) — once we know who's online (from the rendezvous `/peers` list), each install queries every other peer DIRECTLY via the encrypted proxy (`/api/p2p/secure/forward` with `path=/api/tags`) to build a local cache of "who has what model". The model graph stays peer-to-peer; the rendezvous never sees it. Cross-checks `device_id == base32(pubkey)[:16]` on every peer record so a malicious rendezvous can't substitute identities.
- **Discovered-peer trust gate** — peers we know about via the rendezvous bootstrap but haven't accepted into our pool can ONLY call read-only metadata endpoints (`/api/tags`, `/api/show`, `/api/ps`) via the secure proxy. They cannot drive `/api/chat`, `/api/embed`, or `/api/pull` until the user explicitly picks one of their models, which promotes them to `role='public'` in `paired_devices`.
- **STUN endpoint discovery** — pure-stdlib STUN client (`backend/p2p_rendezvous.py`) discovers our public IP/port via Google / Cloudflare / Nextcloud STUN servers. Re-discovers every 5 minutes to catch NAT mapping drift.
- **Public pool toggle** (Settings → Compute pool, default ON):
  - **ON** — registers with the rendezvous so other peers can find us. Local inventory loop polls peers' `/api/tags` over the encrypted proxy. Donates idle compute to the wider Gigachat swarm. **Uses** other peers' GPUs when a model isn't local.
  - **OFF** — fully isolated to local pool. No rendezvous registration, no inventory polling, discovered-peer cache wiped (revokes other peers' tighter-whitelist access to our `/api/tags`).
- **Auto-pull from official source** — when the user picks a model that no peer in the swarm offers, the executing machine pulls it from the OFFICIAL Ollama registry (`registry.ollama.ai`), NOT peer-to-peer. Model bytes never traverse another user's home internet.

---

## TURN-style relay (symmetric-NAT fallback)

- **Why** — direct STUN-discovered candidates fail when both peers are behind symmetric-NAT routers (most home networks without UPnP). The relay shuttles encrypted envelopes through the rendezvous so symmetric-NAT pairs stay reachable.
- **Server** (`rendezvous/main.py`) — `POST /relay/send` drops an envelope into the recipient's queue; `GET /relay/inbox/{device_id}` long-polls (up to 25 s) for queued envelopes. Per-IP rate limit (60 sends/min), per-recipient queue cap (200 envelopes), payload cap (256 KB), 60 s TTL.
- **Privacy** — the relay sees ONLY ciphertext. Confidentiality is end-to-end via the existing `p2p_crypto` envelope, so adding the relay doesn't widen the rendezvous's trust profile.
- **Client** (`backend/p2p_relay.py`) — `forward_via_relay(...)` is a drop-in replacement for `p2p_secure_client.forward()`, generating a `_relay_req_id` for request/response correlation and parking an asyncio Future until the matching response envelope arrives via the inbox poll loop. `p2p_secure_client.forward()` falls back to the relay automatically when the direct HTTP POST fails.
- **Latency** — ~100-300 ms relay roundtrip on Cloud Run, vs. sub-millisecond direct LAN. Acceptable for one-shot calls (chat completions, embeddings); too slow for streaming chat where each NDJSON chunk would pay the relay tax. Streaming over relay needs WebSocket transport — deferred.

---

## Real-time fairness scheduler

When public pool is on and inbound requests arrive, `backend/p2p_fairness.py` enforces:

- **Minimum entitlement**: your full local pool, always available (no quota).
- **Maximum entitlement** (per-consumer slice, real-time): `total_donations / active_consumers`. Recomputed on EVERY admission decision — a surge of new consumers immediately tightens existing slices; departures widen them.
- **Per-peer rate cap** + **hard concurrency cap** as belt-and-braces.

---

## API surface

| Endpoint | Use |
|---|---|
| `GET /api/p2p/identity` | This install's identity (device_id, label, Ed25519 + X25519 pubkeys) |
| `PATCH /api/p2p/identity` | Rename the local device |
| `GET /api/p2p/discover` | Snapshot of LAN peers from mDNS browser |
| `POST /api/p2p/pair/start` | Generate a pairing PIN |
| `POST /api/p2p/pair/build-claim` | Build a signed pairing claim from this device's identity |
| `POST /api/p2p/pair/accept` | Accept a pairing claim (host side) |
| `POST /api/p2p/pair/notify` | Receive symmetric pair record from peer (encrypted envelope) |
| `POST /api/p2p/pair/unpair-notify` | Receive symmetric unpair from peer (encrypted envelope) |
| `DELETE /api/p2p/pair/{id}` | Cancel a pending pairing offer |
| `GET /api/p2p/pair/pending` | List active pairing offers |
| `GET /api/p2p/paired` | List paired devices |
| `DELETE /api/p2p/paired/{device_id}` | Unpair (also drops the matching compute_worker + clears crypto cache) |
| `POST /api/p2p/secure/forward` | Inbound encrypted compute proxy (one-shot) |
| `POST /api/p2p/secure/forward-stream` | Inbound encrypted compute proxy (streaming) |
| `GET /api/p2p/public-pool` | Read public-pool toggle |
| `PATCH /api/p2p/public-pool` | Toggle public-pool on/off |
| `GET /api/p2p/rendezvous/status` | STUN candidates + last register/heartbeat |
| `PATCH /api/p2p/rendezvous/url` | Set the rendezvous server URL (overrides the default) |
| `GET /api/p2p/fairness/status` | Real-time fairness scheduler view |
| `PATCH /api/p2p/fairness/config` | Tune donation fraction + concurrency caps |

## Rendezvous endpoints

| Endpoint | Use |
|---|---|
| `POST /register` | Peer registers identity + STUN candidates (Ed25519-signed) |
| `POST /heartbeat` | Extend the registration TTL |
| `GET /lookup/{device_id}` | Look up one specific peer |
| `GET /peers` | List every registered peer (identity + endpoints, NO model info) |
| `POST /relay/send` | Drop an encrypted envelope into a recipient's relay inbox |
| `GET /relay/inbox/{device_id}` | Long-poll the relay inbox for queued envelopes |
| `GET /health` | Cloud Run probe |
