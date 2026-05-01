import React, { useCallback, useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import {
  Loader2,
  Network,
  RefreshCw,
  Smartphone,
  Laptop,
  Monitor,
  Trash2,
  Wifi,
  WifiOff,
  ShieldCheck,
  Globe,
  Pencil,
  Check,
  X,
  KeyRound,
  Lock,
  Unlock,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { api } from '@/lib/api'
import { cn, formatMessageTime, formatFullTimestamp } from '@/lib/utils'

/**
 * P2PSection — Settings → Network panel.
 *
 * Three areas:
 *   1. **My device** — identity card (device_id + label). Click pencil to
 *      rename the local device (keypair is unchanged).
 *   2. **Available on this network** — devices currently advertising via
 *      mDNS. Click "Pair" on any of them to start the PIN flow.
 *   3. **Paired devices** — trusted peers, with online/offline indicator
 *      driven by mDNS last-seen and an unpair button.
 *
 * Plus the **Public pool toggle** at the top — opt-in to the global
 * compute swarm. Default on. Off → fully isolated to local pool.
 *
 * The "PIN displayed, waiting…" panel renders inline at the top of the
 * Available list when a pair offer is active. Polling every ~2 s keeps
 * the discovery + paired lists fresh; cheap because the underlying
 * endpoint is just an in-memory snapshot.
 */
export default function P2PSection() {
  const [identity, setIdentity] = useState(null)
  const [labelDraft, setLabelDraft] = useState('')
  const [editingLabel, setEditingLabel] = useState(false)
  const [savingLabel, setSavingLabel] = useState(false)

  const [discovered, setDiscovered] = useState([])
  const [discoveryRunning, setDiscoveryRunning] = useState(false)
  const [paired, setPaired] = useState([])

  const [publicPool, setPublicPool] = useState(true)
  const [rendezvous, setRendezvous] = useState(null)

  // Pending pair offer (host side). When set, the panel renders the
  // "Show this PIN to the other device" card. PIN expires after 5 min;
  // the countdown ticks via a setInterval.
  const [pairOffer, setPairOffer] = useState(null)
  const [pairStarting, setPairStarting] = useState(false)

  // Claim-side state — used when the user is on THIS device and types
  // a PIN displayed on another paired host. The two halves are split
  // into separate UI rows because the user is acting in different
  // roles (host vs. claimant) at different times.
  const [claimDeviceId, setClaimDeviceId] = useState('')
  const [claimPin, setClaimPin] = useState('')
  const [claimSubmitting, setClaimSubmitting] = useState(false)

  // Refresh tick — polls discovery + paired lists every 2 s. Cheap because
  // the underlying endpoint is in-memory only.
  const [tick, setTick] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setTick((n) => n + 1), 2000)
    return () => clearInterval(t)
  }, [])

  // Initial bulk fetch + every tick.
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const [id, disc, pair, pp, rdv] = await Promise.all([
          api.p2pIdentity(),
          api.p2pDiscover(),
          api.p2pListPaired(),
          api.p2pPublicPoolStatus(),
          api.p2pRendezvousStatus().catch(() => null),
        ])
        if (cancelled) return
        setIdentity(id)
        setLabelDraft(id?.label || '')
        setDiscovered(disc?.devices || [])
        setDiscoveryRunning(!!disc?.running)
        setPaired(pair?.paired || [])
        setPublicPool(!!pp?.enabled)
        setRendezvous(rdv)
      } catch (e) {
        if (!cancelled) {
          toast.error('Could not load P2P state', {
            description: e?.message || String(e),
          })
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [tick])

  // Poll the pending-offers endpoint independently so a refresh of
  // the panel restores the active PIN dialog (the offer lives on the
  // server even if the user closed the modal accidentally).
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const { pending } = await api.p2pPairPending()
        if (cancelled) return
        if (Array.isArray(pending) && pending.length) {
          setPairOffer((cur) => cur || pending[0])
        } else {
          setPairOffer(null)
        }
      } catch {
        // Silent — offer endpoint failure shouldn't spam the user.
      }
    })()
    return () => {
      cancelled = true
    }
  }, [tick])

  const startPair = useCallback(async () => {
    setPairStarting(true)
    try {
      const offer = await api.p2pPairStart()
      setPairOffer(offer)
      // Auto-fill the host_device_id field for the test/same-machine
      // path so a single-machine demo is one click.
      setClaimDeviceId(offer?.host_device_id || '')
    } catch (e) {
      toast.error('Could not start pairing', {
        description: e?.message || String(e),
      })
    } finally {
      setPairStarting(false)
    }
  }, [])

  const cancelPair = useCallback(async () => {
    if (!pairOffer?.pairing_id) return
    try {
      await api.p2pPairCancel(pairOffer.pairing_id)
    } catch {
      // Silently fall through — purpose is to clear the UI.
    }
    setPairOffer(null)
  }, [pairOffer])

  // Submit a claim FROM this device against a host's PIN. Single-machine
  // pairing flow: "Pair" on host → host displays PIN → user enters PIN
  // here → ButtonClick → backend builds the signed claim from THIS
  // device's identity → POSTs to host's /accept endpoint.
  //
  // Cross-machine: the claimant device runs THIS exact flow from its
  // own browser; `host_device_id` + `host_public_key_b64` come from
  // the host's `/api/p2p/identity` (which the discovery list exposes).
  const submitClaim = useCallback(async () => {
    if (!pairOffer || !claimPin) return
    setClaimSubmitting(true)
    try {
      const claim = await api.p2pPairBuildClaim(
        claimPin, pairOffer.nonce, pairOffer.host_public_key_b64,
      )
      const result = await api.p2pPairAccept({
        pairing_id: pairOffer.pairing_id,
        pin: claimPin,
        claimant_device_id: claim.claimant_device_id,
        claimant_label: claim.claimant_label,
        claimant_public_key_b64: claim.claimant_public_key_b64,
        signature_b64: claim.signature_b64,
      })
      toast.success('Device paired', {
        description: result?.paired?.label || claim.claimant_device_id,
      })
      setPairOffer(null)
      setClaimPin('')
      setClaimDeviceId('')
    } catch (e) {
      toast.error('Pairing failed', {
        description: e?.message || String(e),
      })
    } finally {
      setClaimSubmitting(false)
    }
  }, [pairOffer, claimPin])

  const unpair = useCallback(async (deviceId, label) => {
    if (!confirm(`Unpair ${label || deviceId}?`)) return
    try {
      await api.p2pUnpair(deviceId)
      toast.success(`Unpaired ${label || deviceId}`)
      setTick((n) => n + 1)
    } catch (e) {
      toast.error('Unpair failed', {
        description: e?.message || String(e),
      })
    }
  }, [])

  const togglePublicPool = useCallback(async (next) => {
    try {
      const { enabled } = await api.p2pPublicPoolSet(next)
      setPublicPool(!!enabled)
      toast.success(
        enabled ? 'Joined public pool' : 'Left public pool',
        {
          description: enabled
            ? 'Donating spare compute to peers. Your prompts still run only on your local pool.'
            : 'Fully isolated to your local pool.',
        },
      )
    } catch (e) {
      toast.error('Could not change public pool state', {
        description: e?.message || String(e),
      })
    }
  }, [])

  const saveLabel = useCallback(async () => {
    const next = labelDraft.trim()
    if (!next || next === identity?.label) {
      setEditingLabel(false)
      return
    }
    setSavingLabel(true)
    try {
      const updated = await api.p2pSetLabel(next)
      setIdentity(updated)
      setLabelDraft(updated.label)
      setEditingLabel(false)
      toast.success('Device label updated')
    } catch (e) {
      toast.error('Could not update label', {
        description: e?.message || String(e),
      })
    } finally {
      setSavingLabel(false)
    }
  }, [labelDraft, identity])

  // Drop discovered entries that are already paired so the "Available"
  // list doesn't double-up with the "Paired" list. Trust anchor stays
  // the device_id either way.
  const pairedIds = new Set(paired.map((p) => p.device_id))
  const fresh = discovered.filter((d) => !pairedIds.has(d.device_id))

  return (
    <div className="space-y-6">
      <header className="flex items-center gap-3">
        <Network className="size-5 text-primary" />
        <div>
          <h2 className="text-base font-semibold">Network &amp; pairing</h2>
          <p className="text-xs text-muted-foreground">
            Pair your devices on this Wi-Fi like Bluetooth. Decide whether to
            share spare compute with the wider Gigachat swarm.
          </p>
        </div>
      </header>

      {/* Public-pool toggle — visually distinct because it's the most
          consequential decision on this panel and the user comes here
          expecting to see it. */}
      <PublicPoolCard enabled={publicPool} onChange={togglePublicPool} />

      {/* Rendezvous status — only render when public pool is on (no
          status to show otherwise). Tells the user whether their
          install is actually discoverable across the internet. */}
      {publicPool && rendezvous ? <RendezvousStatusCard status={rendezvous} /> : null}

      {/* My identity */}
      <section className="rounded-lg border border-border bg-card p-4">
        <h3 className="mb-3 flex items-center gap-2 text-sm font-medium">
          <KeyRound className="size-4 text-muted-foreground" />
          This device
        </h3>
        {identity ? (
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground w-16">Label</span>
              {editingLabel ? (
                <>
                  <Input
                    value={labelDraft}
                    onChange={(e) => setLabelDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') saveLabel()
                      if (e.key === 'Escape') {
                        setLabelDraft(identity.label)
                        setEditingLabel(false)
                      }
                    }}
                    className="h-7 max-w-[260px]"
                    disabled={savingLabel}
                    autoFocus
                  />
                  <Button
                    size="icon"
                    variant="ghost"
                    onClick={saveLabel}
                    disabled={savingLabel}
                    className="size-7"
                    title="Save"
                  >
                    {savingLabel ? <Loader2 className="size-3.5 animate-spin" /> : <Check className="size-3.5" />}
                  </Button>
                  <Button
                    size="icon"
                    variant="ghost"
                    onClick={() => {
                      setLabelDraft(identity.label)
                      setEditingLabel(false)
                    }}
                    disabled={savingLabel}
                    className="size-7"
                    title="Cancel"
                  >
                    <X className="size-3.5" />
                  </Button>
                </>
              ) : (
                <>
                  <span className="font-medium">{identity.label}</span>
                  <Button
                    size="icon"
                    variant="ghost"
                    onClick={() => setEditingLabel(true)}
                    className="size-7 text-muted-foreground hover:text-foreground"
                    title="Rename device"
                  >
                    <Pencil className="size-3.5" />
                  </Button>
                </>
              )}
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground w-16">ID</span>
              <code className="rounded bg-muted px-2 py-0.5 text-xs font-mono">
                {identity.device_id_pretty || identity.device_id}
              </code>
            </div>
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">Loading identity…</p>
        )}
      </section>

      {/* Active pairing offer (host side). Renders only when a PIN was
          generated; auto-clears when the offer is cancelled / accepted. */}
      {pairOffer && (
        <PairOfferCard
          offer={pairOffer}
          claimPin={claimPin}
          setClaimPin={setClaimPin}
          submitting={claimSubmitting}
          onSubmitClaim={submitClaim}
          onCancel={cancelPair}
        />
      )}

      {/* Available LAN devices */}
      <section className="rounded-lg border border-border bg-card p-4">
        <h3 className="mb-3 flex items-center gap-2 text-sm font-medium">
          <Wifi className="size-4 text-muted-foreground" />
          Devices on this network
          {!discoveryRunning && (
            <span className="ml-2 rounded bg-amber-500/10 px-2 py-0.5 text-[10px] text-amber-500">
              mDNS unavailable — discovery disabled
            </span>
          )}
        </h3>
        {fresh.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            No other Gigachat devices found on the network. Open Gigachat on
            another laptop / desktop on the same Wi-Fi to see it here, then
            click "Pair new device" below to start.
          </p>
        ) : (
          <ul className="space-y-1.5">
            {fresh.map((d) => (
              <DeviceRow key={d.device_id} device={d} role="discovered" />
            ))}
          </ul>
        )}
        <div className="mt-4">
          <Button
            onClick={startPair}
            disabled={pairStarting || !!pairOffer}
            size="sm"
          >
            {pairStarting ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              'Pair new device'
            )}
          </Button>
        </div>
      </section>

      {/* Paired devices */}
      <section className="rounded-lg border border-border bg-card p-4">
        <h3 className="mb-3 flex items-center gap-2 text-sm font-medium">
          <ShieldCheck className="size-4 text-emerald-500" />
          Paired devices
        </h3>
        {paired.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            No paired devices yet. After pairing, devices reconnect
            automatically when their IP changes — same as Bluetooth.
            All compute traffic to paired devices is end-to-end
            encrypted (X25519 + ChaCha20-Poly1305).
          </p>
        ) : (
          <ul className="space-y-1.5">
            {paired.map((d) => {
              const lastSeenAt = d.last_seen_at || 0
              const online = lastSeenAt > 0 && (Date.now() / 1000 - lastSeenAt) < 120
              const e2e = !!d.x25519_public_b64
              return (
                <li
                  key={d.device_id}
                  className="flex items-center gap-3 rounded border border-border/50 bg-background/50 px-3 py-2"
                >
                  {iconForLabel(d.label, online)}
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2 text-sm">
                      <span className="truncate font-medium">{d.label}</span>
                      <span
                        className={cn(
                          'inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px]',
                          online
                            ? 'bg-emerald-500/15 text-emerald-500'
                            : 'bg-muted text-muted-foreground',
                        )}
                      >
                        {online ? <Wifi className="size-2.5" /> : <WifiOff className="size-2.5" />}
                        {online ? 'Online' : 'Offline'}
                      </span>
                      {/* Encryption status badge — green lock when
                          we have the peer's X25519 key on file
                          (auto-exchanged at pair time); amber unlock
                          for legacy peers paired before E2E shipped
                          who need to re-pair to enable encryption. */}
                      <span
                        className={cn(
                          'inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px]',
                          e2e
                            ? 'bg-emerald-500/15 text-emerald-500'
                            : 'bg-amber-500/15 text-amber-500',
                        )}
                        title={
                          e2e
                            ? 'End-to-end encrypted: all compute traffic to this peer is wrapped in X25519+ChaCha20-Poly1305 envelopes; nothing on the wire is plaintext.'
                            : 'No encryption key on file — peer was paired before E2E shipped. Re-pair to enable end-to-end encryption.'
                        }
                      >
                        {e2e ? <Lock className="size-2.5" /> : <Unlock className="size-2.5" />}
                        {e2e ? 'E2E' : 'plaintext'}
                      </span>
                    </div>
                    <div className="text-[11px] text-muted-foreground">
                      <code>{d.device_id}</code>
                      {d.ip ? <> · {d.ip}{d.port ? `:${d.port}` : ''}</> : null}
                      {lastSeenAt ? (
                        <>
                          {' · '}
                          <span title={formatFullTimestamp(lastSeenAt)}>
                            last seen {formatMessageTime(lastSeenAt)}
                          </span>
                        </>
                      ) : null}
                    </div>
                  </div>
                  <Button
                    size="icon"
                    variant="ghost"
                    onClick={() => unpair(d.device_id, d.label)}
                    className="size-7 text-muted-foreground hover:text-destructive"
                    title="Unpair"
                  >
                    <Trash2 className="size-4" />
                  </Button>
                </li>
              )
            })}
          </ul>
        )}
      </section>

      {/* Security model summary — explicit so users can see what's
          protected and what isn't, without trawling through code. */}
      <section className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-4">
        <h3 className="mb-2 flex items-center gap-2 text-sm font-medium">
          <Lock className="size-4 text-emerald-500" />
          End-to-end encryption
        </h3>
        <ul className="space-y-1 text-xs leading-snug text-muted-foreground">
          <li>
            <strong className="text-foreground">Identity:</strong>{' '}
            Each device has an Ed25519 (signing) + X25519 (key
            agreement) keypair generated on first launch and stored
            in <code>data/identity.json</code> with mode 0600.
          </li>
          <li>
            <strong className="text-foreground">Compute traffic:</strong>{' '}
            Every chat, embedding, and probe call to a paired peer is
            wrapped in an X25519+ChaCha20-Poly1305 envelope, signed
            with Ed25519. Anyone observing the network sees only
            ciphertext — prompts, model output, even peer metadata.
          </li>
          <li>
            <strong className="text-foreground">Replay protection:</strong>{' '}
            ±120 s timestamp window enforced on every envelope. Captured
            traffic can't be replayed later.
          </li>
          <li>
            <strong className="text-foreground">Sender authenticity:</strong>{' '}
            Receiver verifies the signature against the SENDER's pinned
            public key from the pair record. Identity substitution by a
            MITM is rejected before decryption is attempted.
          </li>
          <li>
            <strong className="text-foreground">Path whitelist:</strong>{' '}
            Even authenticated peers can only reach Ollama compute
            endpoints through the secure proxy — no admin/model-delete
            paths exposed.
          </li>
        </ul>
      </section>
    </div>
  )
}

/* ----------------------- Sub-components ----------------------- */

function RendezvousStatusCard({ status }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(status?.url || '')
  const [saving, setSaving] = useState(false)

  // When the underlying status updates from the parent (poll tick),
  // refresh the draft only when we're NOT actively editing — otherwise
  // typing would get clobbered by every poll.
  useEffect(() => {
    if (!editing) setDraft(status?.url || '')
  }, [status?.url, editing])

  const save = useCallback(async () => {
    setSaving(true)
    try {
      await api.p2pRendezvousSetUrl(draft.trim())
      toast.success(
        draft.trim() ? 'Rendezvous URL updated' : 'Rendezvous URL cleared',
      )
      setEditing(false)
    } catch (e) {
      toast.error('Could not update rendezvous URL', {
        description: e?.message || String(e),
      })
    } finally {
      setSaving(false)
    }
  }, [draft])

  const running = !!status?.running
  const lastReg = status?.last_register_at || 0
  const lastRegPretty = lastReg ? formatMessageTime(lastReg) : 'never'
  const cands = Array.isArray(status?.candidates) ? status.candidates : []
  const configured = !!status?.configured

  return (
    <section
      className={cn(
        'rounded-lg border p-4',
        configured && running && lastReg
          ? 'border-emerald-500/30 bg-emerald-500/5'
          : configured
            ? 'border-amber-500/30 bg-amber-500/5'
            : 'border-border bg-card',
      )}
    >
      <h3 className="mb-2 flex items-center gap-2 text-sm font-medium">
        <Globe
          className={cn(
            'size-4',
            configured && running && lastReg
              ? 'text-emerald-500'
              : configured
                ? 'text-amber-500'
                : 'text-muted-foreground',
          )}
        />
        Internet rendezvous
        <span
          className={cn(
            'ml-auto rounded px-1.5 py-0.5 text-[10px]',
            configured && running && lastReg
              ? 'bg-emerald-500/15 text-emerald-500'
              : configured
                ? 'bg-amber-500/15 text-amber-500'
                : 'bg-muted text-muted-foreground',
          )}
        >
          {!configured
            ? 'Not configured'
            : running && lastReg
              ? 'Connected'
              : running
                ? 'Connecting…'
                : 'Disconnected'}
        </span>
      </h3>

      {/* URL editor — works whether or not a URL is currently set. */}
      <div className="mb-2 flex items-center gap-2">
        <span className="w-16 shrink-0 text-xs text-muted-foreground">URL</span>
        {editing ? (
          <>
            <Input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') save()
                if (e.key === 'Escape') {
                  setDraft(status?.url || '')
                  setEditing(false)
                }
              }}
              placeholder="https://gigachat-rendezvous-…run.app"
              className="h-7 flex-1 font-mono text-xs"
              disabled={saving}
              autoFocus
            />
            <Button
              size="icon"
              variant="ghost"
              onClick={save}
              disabled={saving}
              className="size-7"
              title="Save"
            >
              {saving ? <Loader2 className="size-3.5 animate-spin" /> : <Check className="size-3.5" />}
            </Button>
            <Button
              size="icon"
              variant="ghost"
              onClick={() => {
                setDraft(status?.url || '')
                setEditing(false)
              }}
              disabled={saving}
              className="size-7"
              title="Cancel"
            >
              <X className="size-3.5" />
            </Button>
          </>
        ) : (
          <>
            <code className="flex-1 truncate text-xs text-muted-foreground">
              {status?.url || '(not set)'}
            </code>
            <Button
              size="icon"
              variant="ghost"
              onClick={() => setEditing(true)}
              className="size-7 text-muted-foreground hover:text-foreground"
              title="Edit"
            >
              <Pencil className="size-3.5" />
            </Button>
          </>
        )}
      </div>

      {!configured ? (
        <p className="text-xs leading-snug text-muted-foreground">
          Public Pool is on, but no rendezvous URL is set. Deploy the
          Cloud Run service (<code>rendezvous/README.md</code>) and paste
          the URL above so peers across the internet can find this
          device. LAN pairing keeps working without it.
        </p>
      ) : (
        <div className="space-y-1 text-xs text-muted-foreground">
          {cands.length > 0 ? (
            <div>
              <span className="font-medium text-foreground">Candidates:</span>{' '}
              {cands.map((c, i) => (
                <span key={i} className="mr-2">
                  <code>{c.ip}:{c.port}</code>
                  <span className="ml-1 opacity-70">({c.source})</span>
                </span>
              ))}
            </div>
          ) : (
            <div>
              <span className="font-medium text-foreground">Candidates:</span>{' '}
              none yet
            </div>
          )}
          <div>
            <span className="font-medium text-foreground">Last registered:</span>{' '}
            <span title={status.last_register_at ? formatFullTimestamp(status.last_register_at) : ''}>
              {lastRegPretty}
            </span>
          </div>
          {status.last_error ? (
            <div className="text-red-400">
              <span className="font-medium">Last error:</span> {status.last_error}
            </div>
          ) : null}
        </div>
      )}
    </section>
  )
}

function PublicPoolCard({ enabled, onChange }) {
  return (
    <section
      className={cn(
        'rounded-lg border-2 p-4 transition-colors',
        enabled
          ? 'border-emerald-500/40 bg-emerald-500/5'
          : 'border-border bg-card',
      )}
    >
      <div className="flex items-start gap-3">
        <Globe className={cn('mt-0.5 size-5', enabled ? 'text-emerald-500' : 'text-muted-foreground')} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-3">
            <h3 className="text-sm font-semibold">Public pool</h3>
            <Switch checked={enabled} onCheckedChange={onChange} />
          </div>
          <p className="mt-1 text-xs leading-snug text-muted-foreground">
            {enabled ? (
              <>
                <strong className="text-foreground">On — donating idle compute.</strong>{' '}
                Your spare GPU/CPU cycles run other peers' public workloads
                when you aren't using them, and you benefit from cooperative
                model-weight distribution.{' '}
                <strong className="text-foreground">
                  Your prompts still run only on your local pool — they never leave this network.
                </strong>
              </>
            ) : (
              <>
                <strong className="text-foreground">Off — local pool only.</strong>{' '}
                You're disconnected from the global swarm. Inference and any
                background workloads run exclusively on your paired LAN devices.
              </>
            )}
          </p>
        </div>
      </div>
    </section>
  )
}

function PairOfferCard({ offer, claimPin, setClaimPin, submitting, onSubmitClaim, onCancel }) {
  const expiresAt = offer?.expires_at || 0
  const [now, setNow] = useState(Date.now() / 1000)
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now() / 1000), 1000)
    return () => clearInterval(t)
  }, [])
  const remaining = Math.max(0, Math.floor(expiresAt - now))
  const mm = String(Math.floor(remaining / 60)).padStart(2, '0')
  const ss = String(remaining % 60).padStart(2, '0')

  return (
    <section className="rounded-lg border-2 border-primary/40 bg-primary/5 p-4">
      <div className="flex items-start gap-3">
        <KeyRound className="mt-0.5 size-5 text-primary" />
        <div className="min-w-0 flex-1 space-y-3">
          <div>
            <h3 className="text-sm font-semibold">Pair a device</h3>
            <p className="mt-0.5 text-xs leading-snug text-muted-foreground">
              Open Gigachat on the other device, choose <strong>Pair to existing host</strong>,
              and enter the PIN below. The PIN expires in{' '}
              <span className="font-mono font-semibold text-foreground">
                {mm}:{ss}
              </span>
              .
            </p>
          </div>

          <div className="flex items-center justify-center rounded-md border bg-background py-3">
            <span className="font-mono text-3xl font-semibold tracking-[0.4em] text-primary">
              {offer.pin}
            </span>
          </div>

          <div className="border-t border-border/50 pt-3">
            <p className="mb-2 text-xs text-muted-foreground">
              Pairing FROM this device? Type the PIN that's displayed on the
              <em> other </em>screen here:
            </p>
            <div className="flex items-center gap-2">
              <Input
                value={claimPin}
                onChange={(e) => setClaimPin(e.target.value.replace(/\D/g, '').slice(0, 6))}
                placeholder="6-digit PIN"
                maxLength={6}
                className="h-8 max-w-[160px] font-mono tracking-widest"
                disabled={submitting}
              />
              <Button
                size="sm"
                onClick={onSubmitClaim}
                disabled={submitting || claimPin.length !== 6}
              >
                {submitting ? <Loader2 className="size-4 animate-spin" /> : 'Confirm'}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={onCancel}
                disabled={submitting}
              >
                Cancel
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

function DeviceRow({ device, role }) {
  const lastSeenAt = device.last_seen_at || 0
  return (
    <li className="flex items-center gap-3 rounded border border-border/50 bg-background/50 px-3 py-2">
      {iconForLabel(device.label, true)}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 text-sm">
          <span className="truncate font-medium">{device.label}</span>
          <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
            v{device.version || '1'}
          </span>
        </div>
        <div className="text-[11px] text-muted-foreground">
          <code>{device.device_id}</code>
          {device.ip ? <> · {device.ip}{device.port ? `:${device.port}` : ''}</> : null}
          {lastSeenAt ? (
            <>
              {' · '}
              <span title={formatFullTimestamp(lastSeenAt)}>
                seen {formatMessageTime(lastSeenAt)}
              </span>
            </>
          ) : null}
        </div>
      </div>
      <p className="text-[11px] text-muted-foreground">
        Open Gigachat on this device to pair from there.
      </p>
    </li>
  )
}

function iconForLabel(label, online) {
  const lower = (label || '').toLowerCase()
  let Icon = Monitor
  if (/(phone|mobile|android|ios|iphone)/.test(lower)) Icon = Smartphone
  else if (/(laptop|book|surface|notebook)/.test(lower)) Icon = Laptop
  return (
    <Icon
      className={cn(
        'size-5 shrink-0',
        online ? 'text-foreground' : 'text-muted-foreground',
      )}
    />
  )
}
