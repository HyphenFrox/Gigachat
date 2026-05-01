# Gigachat Rendezvous Service

Tiny stateless service that lets Gigachat peers find each other for
P2P NAT-traversal. **No prompts, no model traffic, no user data ever
passes through this service** — peers register their identity +
STUN-discovered endpoints, then connect directly via QUIC + UDP
hole-punching.

## What it stores

```
{
  device_id:        Ed25519 public key hash (the trust anchor)
  public_key_b64:   the full pubkey
  candidates:       [{ip, port, source}]   ← STUN-discovered endpoints
  last_seen_at:     Unix epoch (60-second TTL)
}
```

That's it. ~100 bytes per peer. 1000 peers = 100 KB of memory.

## Deployment to GCP Cloud Run

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
  --cpu 0.5 \
  --concurrency 100 \
  --timeout 60s

# Cloud Run prints the public URL; copy it into your Gigachat client
# config. e.g. https://gigachat-rendezvous-xxxxxxx-uc.a.run.app
```

After the first deploy, point each Gigachat client at the URL via the
`GIGACHAT_RENDEZVOUS_URL` environment variable (or the equivalent
setting in `Settings → Network → Public pool`).

## Cost & scaling

Cloud Run free tier covers most users:

- Free tier: 2 million requests / month + 360,000 vCPU-seconds.
- Each peer hits `/heartbeat` every 30 s = ~86,000 reqs/month/peer.
- ~20 peers fit comfortably in the free tier.

For larger swarms, set `--max-instances` higher and Cloud Run scales
horizontally. Memory state isn't shared across instances — each
instance keeps its own peer table — so a peer registered on instance A
won't show up in lookups served by instance B until it heartbeats
again. The 60-second TTL bounds this anomaly window.

## Security

- **Signed registrations** — every `/register` and `/heartbeat`
  carries an Ed25519 signature over a canonical message. The
  rendezvous verifies the signature against the claimed public
  key before accepting. An attacker can't impersonate a peer's
  device_id without their private key.
- **Replay protection** — registrations carry a timestamp; the
  service rejects anything older than 120 seconds.
- **Per-IP rate limit** — 30 registrations/min per source IP.
  Sybil-farming becomes quadratically expensive across IPs.
- **Stateless across restarts** — Cloud Run scales to zero between
  bursts; peers re-register on the next heartbeat. No persistent
  storage means no data-at-rest concerns.
- **No client data** — prompts, models, and chat content stay
  end-to-end between peers. The rendezvous is a phone book.

## Local testing

```bash
pip install -r requirements.txt
python main.py
# Service runs on http://127.0.0.1:8080

curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/
```
