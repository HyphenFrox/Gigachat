# Gigachat Rendezvous Service

Tiny stateless service that lets Gigachat peers find each other for
P2P NAT-traversal. **No prompts, no model traffic, no user data ever
passes through this service** — peers register their identity +
STUN-discovered endpoints, then connect directly via the encrypted
compute proxy. The only payloads the rendezvous touches are opaque
ciphertext envelopes shuttled through the optional TURN-style relay
for symmetric-NAT pairs (see `/relay/*` below).

## What it stores

```
{
  device_id:           Ed25519 public key hash (the trust anchor)
  public_key_b64:      the full Ed25519 pubkey (signing)
  x25519_public_b64:   the X25519 pubkey (key agreement for envelopes)
  candidates:          [{ip, port, source}]   ← STUN-discovered endpoints
  last_seen_at:        Unix epoch (60-second TTL)
}
```

That's it. ~200 bytes per peer. 1000 peers ≈ 200 KB of memory.

The rendezvous deliberately does NOT store model lists or
capability info — what each peer offers stays peer-to-peer. Clients
query each known peer's `/api/tags` directly via the encrypted
proxy and cache the result locally.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /register` | Peer registers identity + STUN candidates (Ed25519-signed). |
| `POST /heartbeat` | Extend the registration TTL. |
| `GET /lookup/{device_id}` | Look up one specific peer. |
| `GET /peers` | List every registered peer (identity + endpoints, NO model info). |
| `POST /relay/send` | Drop an encrypted envelope into a recipient's relay inbox. |
| `GET /relay/inbox/{device_id}` | Long-poll the relay inbox for queued envelopes. |
| `GET /health` | Cloud Run probe. |

## Default deployment

The project ships pointing at a public Cloud Run instance hard-coded
into `backend/p2p_rendezvous._DEFAULT_RENDEZVOUS_URL`, so a fresh
Gigachat install joins the swarm the moment Public Pool toggles on
— no manual paste required. Override per-install via Settings →
Compute pool → rendezvous URL editor, or the
`GIGACHAT_RENDEZVOUS_URL` env var.

## Deploying your own (optional — power users)

```bash
# Authenticate once
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Deploy from this directory
gcloud run deploy gigachat-rendezvous \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --max-instances 5 \
  --memory 256Mi \
  --cpu 1 \
  --concurrency 100 \
  --timeout 60s

# Cloud Run prints the public URL; paste it into Settings → Compute
# pool → rendezvous URL on each Gigachat client (or set
# GIGACHAT_RENDEZVOUS_URL=<url> in the backend's env).
```

## Cost & scaling

Cloud Run free tier covers most users:

- Free tier: 2 million requests / month + 360,000 vCPU-seconds.
- Each peer hits `/heartbeat` every 30 s = ~86,000 reqs/month/peer.
- ~20 peers fit comfortably in the free tier.

For larger swarms, set `--max-instances` higher and Cloud Run scales
horizontally. Memory state isn't shared across instances — each
instance keeps its own peer table and its own relay queue — so a
peer registered on instance A won't show up in lookups served by
instance B until it heartbeats again. The 60-second TTL bounds this
anomaly window.

## Security

- **Signed registrations** — every `/register` and `/heartbeat`
  carries an Ed25519 signature over a canonical message. The
  rendezvous verifies the signature against the claimed public
  key before accepting. An attacker can't impersonate a peer's
  device_id without their private key.
- **Replay protection** — registrations carry a timestamp; the
  service rejects anything older than 120 seconds.
- **Per-IP rate limit** — 30 registrations/min and 60 relay
  sends/min per source IP. Sybil-farming becomes quadratically
  expensive across IPs.
- **Stateless across restarts** — Cloud Run scales to zero between
  bursts; peers re-register on the next heartbeat. No persistent
  storage means no data-at-rest concerns.
- **Relay sees only ciphertext** — `/relay/send` payloads are
  the same end-to-end-encrypted `p2p_crypto` envelopes used for
  direct P2P. The rendezvous never holds a key that can decrypt
  them, so adding the relay doesn't widen the rendezvous's trust
  profile.
- **Per-recipient queue cap** — 200 envelopes; oldest dropped on
  overflow so a freshly-arrived envelope can't be locked out by a
  malicious flood.
- **Per-payload size cap** — 256 KB matches the secure-proxy
  inbound cap so legitimate requests can ride either path
  without surgery.
- **No client data** — prompts, models, and chat content stay
  end-to-end between peers. The rendezvous is a phone book + a
  best-effort cipher-text mailbox.

## Local testing

```bash
pip install -r requirements.txt
python main.py
# Service runs on http://127.0.0.1:8080

curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/
curl http://127.0.0.1:8080/peers
```
