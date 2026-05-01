import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'
import {
  Loader2,
  RefreshCw,
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
  Server,
  Smartphone,
  Laptop,
  Monitor,
  Cpu,
  CircleCheck,
  CircleX,
  CircleHelp,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { cn, formatMessageTime, formatFullTimestamp } from '@/lib/utils'
import { api } from '@/lib/api'
import AutoSplitInstallSection from './SplitModelsPanel'

/**
 * ComputePoolSection — single Settings tab that owns the entire
 * "other devices doing work for me" UX.
 *
 * Replaces the pre-merge split between Settings → Compute (manual IP /
 * SSH workers) and Settings → Network (PIN-paired Gigachat peers).
 * Both surfaces backed the same routing layer, so the user saw a
 * paired device twice — once in each tab — and could edit it from
 * either side, which was confusing.
 *
 * Layout (top → bottom):
 *   1. Public-pool toggle — most consequential decision, gets the
 *      visual weight.
 *   2. Internet rendezvous status — only when public pool is on.
 *   3. This device — identity card with rename.
 *   4. Active pair offer — renders when a pairing is in flight.
 *   5. Devices on this network — mDNS-discovered, ready to pair.
 *   6. Paired devices — the actual compute pool, joined view of
 *      `paired_devices` + `compute_workers` (same physical machine,
 *      one row each in the underlying schema). Shows live status,
 *      capabilities (Ollama version, GPU, model count), workload
 *      routing toggles, test-connection + unpair affordances.
 *   7. Legacy workers — manually-added rows from the pre-merge UI.
 *      Delete-only; new manual entries aren't accepted because
 *      paired devices cover every supported case.
 *   8. End-to-end encryption summary — pinned info card.
 *   9. AutoSplit install banner — one-time llama.cpp install for
 *      the big-model split path.
 *
 * Polling: discovery + paired + worker lists tick every 2 s. Cheap
 * because all three endpoints read in-memory snapshots / cached DB
 * rows (no probe fan-out per tick).
 */
export default function ComputePoolSection() {
  // --- identity / labels ------------------------------------------------
  const [identity, setIdentity] = useState(null)
  const [labelDraft, setLabelDraft] = useState('')
  const [editingLabel, setEditingLabel] = useState(false)
  const [savingLabel, setSavingLabel] = useState(false)

  // --- public pool / rendezvous ----------------------------------------
  const [publicPool, setPublicPool] = useState(true)
  const [rendezvous, setRendezvous] = useState(null)

  // --- discovery + pairing flow ----------------------------------------
  const [discovered, setDiscovered] = useState([])
  const [discoveryRunning, setDiscoveryRunning] = useState(false)
  // Local "I'm offering a PIN" state. Other devices type the PIN we
  // display here into THEIR Pair dialog.
  const [pairOffer, setPairOffer] = useState(null)
  const [pairStarting, setPairStarting] = useState(false)
  // Cross-device pair dialog: when set, we render a modal asking for
  // the PIN currently displayed on `pairingTarget`'s screen. The
  // claim is then POSTed cross-device to that target's /pair/accept.
  const [pairingTarget, setPairingTarget] = useState(null)
  const [pairingPin, setPairingPin] = useState('')
  const [pairingSubmitting, setPairingSubmitting] = useState(false)

  // --- pool: paired peers + matched workers ----------------------------
  const [paired, setPaired] = useState([])
  const [workers, setWorkers] = useState([])
  const [probing, setProbing] = useState(() => new Set())
  const [refreshingAll, setRefreshingAll] = useState(false)
  const [pendingDelete, setPendingDelete] = useState(null)

  // 2s polling tick. Discovery + paired + workers all share it so the
  // panel stays internally consistent (you don't see a paired device
  // appear before its worker row catches up, etc.).
  const [tick, setTick] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setTick((n) => n + 1), 2000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const [id, disc, pair, pp, rdv, ws] = await Promise.all([
          api.p2pIdentity(),
          api.p2pDiscover(),
          api.p2pListPaired(),
          api.p2pPublicPoolStatus(),
          api.p2pRendezvousStatus().catch(() => null),
          api.listComputeWorkers(),
        ])
        if (cancelled) return
        setIdentity(id)
        setLabelDraft(id?.label || '')
        setDiscovered(disc?.devices || [])
        setDiscoveryRunning(!!disc?.running)
        setPaired(pair?.paired || [])
        setPublicPool(!!pp?.enabled)
        setRendezvous(rdv)
        setWorkers(ws?.workers || [])
      } catch (e) {
        if (!cancelled) {
          toast.error('Could not load compute pool', {
            description: e?.message || String(e),
          })
        }
      }
    })()
    return () => { cancelled = true }
  }, [tick])

  // Pending pair offer poller — survives accidental modal-close.
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
        // Silent — a single failed poll shouldn't spam the user.
      }
    })()
    return () => { cancelled = true }
  }, [tick])

  // --- joined view -----------------------------------------------------
  // Each physical paired device has TWO underlying DB rows:
  //   * paired_devices  — identity + crypto (label, pubkeys, role)
  //   * compute_workers — routing target + capabilities (auto-created
  //                        by the pair flow, keyed by gigachat_device_id)
  // We join them client-side so the UI shows ONE row per device with
  // all the relevant info (online status, Ollama version, model count,
  // workload toggles).
  const pairedView = useMemo(() => {
    return (paired || []).map((p) => {
      const worker = (workers || []).find(
        (w) => w.gigachat_device_id === p.device_id,
      ) || null
      return { paired: p, worker }
    })
  }, [paired, workers])

  // Workers that have no matching paired_device — these are the
  // legacy manually-added rows from the pre-merge UI. New manual
  // entries aren't accepted; existing ones can be deleted.
  const legacyWorkers = useMemo(
    () => (workers || []).filter((w) => !w.gigachat_device_id),
    [workers],
  )

  // Drop already-paired devices from the discovered list so they
  // don't show up twice ("ready to pair" + "paired").
  const pairedIds = new Set(paired.map((p) => p.device_id))
  const freshDiscovered = discovered.filter((d) => !pairedIds.has(d.device_id))

  // --- summary line at top --------------------------------------------
  const summary = useMemo(() => {
    const total = pairedView.length + legacyWorkers.length
    if (total === 0) return 'No devices in your pool yet'
    const online = pairedView.filter((row) => isPairedOnline(row)).length
      + legacyWorkers.filter((w) => isWorkerOnline(w)).length
    return `${total} device${total === 1 ? '' : 's'} · ${online} online`
  }, [pairedView, legacyWorkers])

  // --- actions ---------------------------------------------------------
  const startPair = useCallback(async () => {
    setPairStarting(true)
    try {
      const offer = await api.p2pPairStart()
      setPairOffer(offer)
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
      // Silent — purpose is to clear the UI.
    }
    setPairOffer(null)
  }, [pairOffer])

  // Cross-device pair: send a single local POST to /api/p2p/pair/initiate
  // and let OUR backend do the cross-device HTTP exchange with the
  // host. Doing it server-side avoids the browser-CORS problem
  // (cross-origin fetch from this page to http://<host_lan_ip>:8000
  // fails with "Failed to fetch" because the host doesn't send
  // Access-Control-Allow-Origin headers — and adding them would
  // expose the API to any web page on the internet that knows the
  // host's LAN IP). Backend-to-backend is the cleaner trust boundary.
  const submitClaimToPeer = useCallback(async () => {
    if (!pairingTarget || !pairingPin) return
    setPairingSubmitting(true)
    try {
      const result = await api.p2pPairInitiate({
        device_id: pairingTarget.device_id,
        pin: pairingPin,
      })
      const label = result?.paired?.label || pairingTarget.label || pairingTarget.device_id
      toast.success('Device paired', { description: label })
      setPairingTarget(null)
      setPairingPin('')
      setTick((n) => n + 1)
    } catch (e) {
      toast.error('Pairing failed', {
        description: e?.message || String(e),
      })
    } finally {
      setPairingSubmitting(false)
    }
  }, [pairingTarget, pairingPin])

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

  const probeOne = useCallback(async (worker) => {
    if (!worker?.id) return
    setProbing((s) => new Set(s).add(worker.id))
    try {
      const res = await api.probeComputeWorker(worker.id)
      if (res.ok) {
        const ver = res.capabilities?.version
        const n = res.capabilities?.models?.length || 0
        toast.success(`${worker.label} is online`, {
          description: `Ollama ${ver || '?'} · ${n} model${n === 1 ? '' : 's'}`,
        })
      } else {
        toast.error(`${worker.label} probe failed`, {
          description: res.error || 'unknown error',
        })
      }
    } catch (e) {
      toast.error('Probe failed', { description: e.message })
    } finally {
      setProbing((s) => {
        const copy = new Set(s)
        copy.delete(worker.id)
        return copy
      })
      setTick((n) => n + 1)
    }
  }, [])

  const probeAll = useCallback(async () => {
    setRefreshingAll(true)
    try {
      const res = await api.probeAllComputeWorkers()
      const results = res.results || []
      const okCount = results.filter((r) => r.ok).length
      const failCount = results.length - okCount
      if (failCount === 0) {
        toast.success(`All ${okCount} worker${okCount === 1 ? '' : 's'} online`)
      } else {
        toast.warning(`${okCount} online · ${failCount} unreachable`, {
          description: 'See each row for details.',
        })
      }
    } catch (e) {
      toast.error('Refresh failed', { description: e.message })
    } finally {
      setRefreshingAll(false)
      setTick((n) => n + 1)
    }
  }, [])

  const toggleWorkerFlag = useCallback(async (worker, key, value) => {
    if (!worker?.id) return
    try {
      await api.updateComputeWorker(worker.id, { [key]: value })
      setTick((n) => n + 1)
    } catch (e) {
      toast.error('Update failed', { description: e.message })
    }
  }, [])

  const unpair = useCallback(async (deviceId, label) => {
    if (!confirm(`Unpair ${label || deviceId}?`)) return
    try {
      // Backend's DELETE /api/p2p/paired/{id} now also drops the
      // matching compute_worker row, so this single call cleans up
      // both halves of the join.
      await api.p2pUnpair(deviceId)
      toast.success(`Unpaired ${label || deviceId}`)
      setTick((n) => n + 1)
    } catch (e) {
      toast.error('Unpair failed', {
        description: e?.message || String(e),
      })
    }
  }, [])

  const confirmDeleteLegacy = useCallback(async () => {
    if (!pendingDelete) return
    try {
      await api.deleteComputeWorker(pendingDelete.id)
      toast.success('Worker removed')
      setPendingDelete(null)
      setTick((n) => n + 1)
    } catch (e) {
      toast.error('Delete failed', { description: e.message })
    }
  }, [pendingDelete])

  // --- render ----------------------------------------------------------
  return (
    <>
      <div className="space-y-6">
        <header className="flex items-start justify-between gap-3">
          <div>
            <h2 className="flex items-center gap-2 text-base font-semibold">
              <Server className="size-5 text-primary" />
              Compute pool
            </h2>
            <p className="text-xs text-muted-foreground">
              {summary}. Pair other Gigachat devices on this network so they
              share the workload — chat, embeddings, parallel subagents.
            </p>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={probeAll}
            disabled={refreshingAll || (!pairedView.length && !legacyWorkers.length)}
            className="gap-1.5 text-xs"
            title="Re-probe every device now"
          >
            <RefreshCw
              className={cn('h-3.5 w-3.5', refreshingAll && 'animate-spin')}
            />
            Refresh
          </Button>
        </header>

        <PublicPoolCard enabled={publicPool} onChange={togglePublicPool} />

        {publicPool && rendezvous ? (
          <RendezvousStatusCard status={rendezvous} />
        ) : null}

        {/* My identity card */}
        <section className="rounded-lg border border-border bg-card p-4">
          <h3 className="mb-3 flex items-center gap-2 text-sm font-medium">
            <KeyRound className="size-4 text-muted-foreground" />
            This device
          </h3>
          {identity ? (
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <span className="w-16 text-xs text-muted-foreground">Label</span>
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
                      {savingLabel ? (
                        <Loader2 className="size-3.5 animate-spin" />
                      ) : (
                        <Check className="size-3.5" />
                      )}
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
                <span className="w-16 text-xs text-muted-foreground">ID</span>
                <code className="rounded bg-muted px-2 py-0.5 font-mono text-xs">
                  {identity.device_id_pretty || identity.device_id}
                </code>
              </div>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">Loading identity…</p>
          )}
        </section>

        {pairOffer && (
          <PairOfferCard
            offer={pairOffer}
            onCancel={cancelPair}
          />
        )}

        {/* Available LAN devices ready to pair */}
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
          {freshDiscovered.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              No other Gigachat devices found on the network. Open Gigachat on
              another laptop / desktop on the same Wi-Fi to see it here, then
              click <em>Pair</em> next to it.
            </p>
          ) : (
            <ul className="space-y-1.5">
              {freshDiscovered.map((d) => (
                <DiscoveredRow
                  key={d.device_id}
                  device={d}
                  onPair={() => {
                    setPairingTarget(d)
                    setPairingPin('')
                  }}
                />
              ))}
            </ul>
          )}
          <div className="mt-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-[11px] leading-snug text-muted-foreground">
              To pair, click <em>Pair</em> on the device above and type the PIN it shows.
              <br />
              Or generate a PIN here for the OTHER device to pair with us:
            </p>
            <Button
              onClick={startPair}
              disabled={pairStarting || !!pairOffer}
              size="sm"
            >
              {pairStarting ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                'Show our PIN'
              )}
            </Button>
          </div>
        </section>

        {/* Paired devices — the actual compute pool */}
        <section className="rounded-lg border border-border bg-card p-4">
          <h3 className="mb-3 flex items-center gap-2 text-sm font-medium">
            <ShieldCheck className="size-4 text-emerald-500" />
            Paired devices
          </h3>
          {pairedView.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              No paired devices yet. Pair one above and it will appear here
              with live status, capabilities, and per-workload routing
              toggles. Reconnects after IP changes are automatic — same as
              Bluetooth. All compute traffic is end-to-end encrypted
              (X25519 + ChaCha20-Poly1305).
            </p>
          ) : (
            <ul className="space-y-2">
              {pairedView.map((row) => (
                <PairedDeviceRow
                  key={row.paired.device_id}
                  row={row}
                  probing={row.worker ? probing.has(row.worker.id) : false}
                  onProbe={() => row.worker && probeOne(row.worker)}
                  onToggle={(key, value) =>
                    row.worker && toggleWorkerFlag(row.worker, key, value)
                  }
                  onUnpair={() => unpair(row.paired.device_id, row.paired.label)}
                />
              ))}
            </ul>
          )}
        </section>

        {/* Legacy manually-added workers — delete-only */}
        {legacyWorkers.length > 0 && (
          <section className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-4">
            <h3 className="mb-2 flex items-center gap-2 text-sm font-medium">
              <Cpu className="size-4 text-amber-500" />
              Legacy workers ({legacyWorkers.length})
            </h3>
            <p className="mb-3 text-xs text-muted-foreground">
              These workers were added by typed IP / SSH alias before the
              Compute and Network panels were merged. Manual entry is no
              longer supported — use <em>Pair new device</em> above for
              new workers. Existing rows can be removed below; routing
              still uses them until you do.
            </p>
            <ul className="space-y-2">
              {legacyWorkers.map((w) => (
                <LegacyWorkerRow
                  key={w.id}
                  worker={w}
                  probing={probing.has(w.id)}
                  onProbe={() => probeOne(w)}
                  onToggle={(key, value) => toggleWorkerFlag(w, key, value)}
                  onDelete={() => setPendingDelete(w)}
                />
              ))}
            </ul>
          </section>
        )}

        {/* End-to-end encryption summary */}
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
              <strong className="text-foreground">Forward secrecy:</strong>{' '}
              Each envelope generates a fresh ephemeral X25519 keypair on
              the sender side. Captured envelopes can't be decrypted
              later from sender-side compromise of long-term keys.
            </li>
            <li>
              <strong className="text-foreground">Replay protection:</strong>{' '}
              ±120 s timestamp window on every envelope.
            </li>
            <li>
              <strong className="text-foreground">Path whitelist:</strong>{' '}
              Even authenticated peers can only reach a strict set of
              Ollama compute endpoints — no admin / model-delete paths.
              Discovered-but-not-paired peers get an even tighter
              read-only subset.
            </li>
          </ul>
        </section>

        {/* AutoSplit (llama.cpp) install banner — only relevant for the
            big-model split-across-pool path. Leave it at the bottom so
            the primary device-management workflows above stay focused. */}
        <AutoSplitInstallSection />
      </div>

      {/* Legacy delete confirm */}
      <Dialog
        open={!!pendingDelete}
        onOpenChange={(o) => !o && setPendingDelete(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Remove this legacy worker?</DialogTitle>
            <DialogDescription>
              <strong>{pendingDelete?.label}</strong> will stop receiving any
              routed traffic. Capabilities and history are deleted; the worker
              process itself isn't touched.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPendingDelete(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={confirmDeleteLegacy}>
              <Trash2 className="mr-1 h-4 w-4" /> Remove
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Cross-device pair dialog (rendered when user clicks "Pair" on
          a discovered peer). Asks for the PIN displayed on that peer
          and dispatches the cross-device claim flow via
          submitClaimToPeer. */}
      <PairWithPeerDialog
        peer={pairingTarget}
        pin={pairingPin}
        setPin={setPairingPin}
        submitting={pairingSubmitting}
        onConfirm={submitClaimToPeer}
        onCancel={() => {
          if (pairingSubmitting) return
          setPairingTarget(null)
          setPairingPin('')
        }}
      />
    </>
  )
}

/* ----------------------- Sub-components ----------------------- */

/** Liveness rule: paired-row online iff the matched worker probed
 *  recently OR the mDNS last_seen is fresh. We accept either signal
 *  because the worker probe runs every 5 min while mDNS broadcasts
 *  every ~10 s — fresh mDNS without a recent probe still means "the
 *  device is on the network and we'll hear back from it shortly." */
function isPairedOnline({ paired, worker }) {
  if (worker && isWorkerOnline(worker)) return true
  const mdnsAt = paired?.last_seen_at || 0
  if (mdnsAt > 0 && Date.now() / 1000 - mdnsAt < 120) return true
  return false
}

function isWorkerOnline(w) {
  if (!w?.enabled) return false
  if (!w.last_seen) return false
  if (w.last_error) return false
  const ageSec = Date.now() / 1000 - Number(w.last_seen)
  return ageSec >= 0 && ageSec < 60 * 60
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
        <Globe
          className={cn(
            'mt-0.5 size-5',
            enabled ? 'text-emerald-500' : 'text-muted-foreground',
          )}
        />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-3">
            <h3 className="text-sm font-semibold">Public pool</h3>
            <Switch checked={enabled} onCheckedChange={onChange} />
          </div>
          {/* Two-line accuracy contract: tells the user the swarm is
              bidirectional (donate AND consume) and that consumption
              crosses the internet (encrypted). The previous copy
              claimed prompts "never leave this network", which became
              wrong once the public-pool consumer path shipped — chat
              CAN dispatch to a peer's GPU when no local device has
              the model. End-to-end encryption is the protection;
              local-network-only is not. */}
          <p className="mt-1 text-xs leading-snug text-muted-foreground">
            {enabled ? (
              <>
                <strong className="text-foreground">On.</strong>{' '}
                Donates idle compute; uses peers' GPUs when a model isn't
                local. Traffic is end-to-end encrypted; local pool wins routing.
              </>
            ) : (
              <>
                <strong className="text-foreground">Off.</strong>{' '}
                Local devices only. New models auto-pull from the OFFICIAL
                Ollama registry — never from peers.
              </>
            )}
          </p>
        </div>
      </div>
    </section>
  )
}

function RendezvousStatusCard({ status }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(status?.url || '')
  const [saving, setSaving] = useState(false)

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
              {saving ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <Check className="size-3.5" />
              )}
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

/** Local-side "we generated a PIN; the OTHER device should type it" card.
 *
 *  No inline PIN-typing field anymore. The cross-device flow now lives
 *  on the discovered-peer rows: each row has a "Pair" button that
 *  opens a dialog asking for the PIN displayed on THAT peer. So this
 *  card just shows OUR PIN to the user (so they can read it off
 *  the screen and type it on the other device's Pair dialog) +
 *  a Cancel button.
 */
function PairOfferCard({ offer, onCancel }) {
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
            <h3 className="text-sm font-semibold">Show this PIN on the other device</h3>
            <p className="mt-0.5 text-xs leading-snug text-muted-foreground">
              Open Gigachat on the other device, find this device in its
              <strong> Devices on this network</strong> list, click <strong>Pair</strong>,
              and type the PIN below. Expires in{' '}
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

          <div className="flex justify-end">
            <Button size="sm" variant="ghost" onClick={onCancel}>
              Cancel
            </Button>
          </div>
        </div>
      </div>
    </section>
  )
}

function DiscoveredRow({ device, onPair }) {
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
      <Button
        size="sm"
        onClick={onPair}
        disabled={!device.ip || !device.public_key_b64}
        title={
          !device.ip
            ? 'Waiting for mDNS to surface this peer\'s LAN address.'
            : !device.public_key_b64
            ? 'Peer\'s mDNS record is missing its public key (legacy install). Restart Gigachat on that device.'
            : 'Type the PIN displayed on this peer to pair with it.'
        }
      >
        Pair
      </Button>
    </li>
  )
}

/** Cross-device pair dialog. The user picked a discovered peer; this
 *  asks for the PIN currently displayed on that peer's screen. On
 *  submit, ComputePoolSection.submitClaimToPeer() does the cross-device
 *  HTTP exchange (fetch host's pending offers → build claim → POST
 *  claim to host's /pair/accept). */
function PairWithPeerDialog({ peer, pin, setPin, submitting, onConfirm, onCancel }) {
  return (
    <Dialog open={!!peer} onOpenChange={(o) => !o && onCancel?.()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <KeyRound className="size-4 text-primary" />
            Pair with {peer?.label || peer?.device_id}
          </DialogTitle>
          <DialogDescription>
            On the OTHER device ({peer?.label || 'that device'}), open
            Settings → Compute pool → click <strong>Show our PIN</strong>.
            Type the 6-digit PIN that appears below.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-3">
          <Input
            autoFocus
            value={pin}
            onChange={(e) => setPin(e.target.value.replace(/\D/g, '').slice(0, 6))}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && pin.length === 6 && !submitting) {
                onConfirm()
              }
            }}
            placeholder="6-digit PIN"
            maxLength={6}
            className="h-10 text-center font-mono text-lg tracking-[0.4em]"
            disabled={submitting}
          />
          <p className="text-[11px] leading-snug text-muted-foreground">
            <code>{peer?.ip}:{peer?.port}</code>
            {' · '}
            device <code>{peer?.device_id}</code>
          </p>
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={onCancel} disabled={submitting}>
            Cancel
          </Button>
          <Button
            onClick={onConfirm}
            disabled={submitting || pin.length !== 6}
          >
            {submitting ? <Loader2 className="size-4 animate-spin" /> : 'Pair'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

/** Joined paired_device + compute_worker row. The merge gives us a single
 *  visual line per physical device with online status, capabilities, and
 *  per-workload routing toggles. Inline actions: probe, unpair. */
function PairedDeviceRow({ row, probing, onProbe, onToggle, onUnpair }) {
  const { paired, worker } = row
  const online = isPairedOnline(row)
  const e2e = !!paired.x25519_public_b64
  const caps = (worker && worker.capabilities) || {}
  const modelCount = Array.isArray(caps.models) ? caps.models.length : 0
  const ver = caps.version
  const lastSeenAt = paired.last_seen_at || 0

  return (
    <li className="rounded border border-border/50 bg-background/50 p-3">
      <div className="flex items-start gap-3">
        {iconForLabel(paired.label, online)}
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="truncate font-medium">{paired.label}</span>
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
            <span
              className={cn(
                'inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px]',
                e2e
                  ? 'bg-emerald-500/15 text-emerald-500'
                  : 'bg-amber-500/15 text-amber-500',
              )}
              title={
                e2e
                  ? 'End-to-end encrypted via X25519+ChaCha20-Poly1305'
                  : 'Re-pair to enable end-to-end encryption'
              }
            >
              {e2e ? <Lock className="size-2.5" /> : <Unlock className="size-2.5" />}
              {e2e ? 'E2E' : 'plaintext'}
            </span>
          </div>

          <div className="mt-0.5 text-[11px] text-muted-foreground">
            <code>{paired.device_id}</code>
            {paired.ip ? <> · {paired.ip}{paired.port ? `:${paired.port}` : ''}</> : null}
            {lastSeenAt ? (
              <>
                {' · '}
                <span title={formatFullTimestamp(lastSeenAt)}>
                  last seen {formatMessageTime(lastSeenAt)}
                </span>
              </>
            ) : null}
          </div>

          {worker ? (
            <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] text-muted-foreground">
              {ver && (
                <span className="inline-flex items-center gap-1">
                  <Cpu className="size-3" /> Ollama {ver}
                </span>
              )}
              {modelCount > 0 && (
                <span title={(caps.models || []).map((m) => m.name).join(', ')}>
                  {modelCount} model{modelCount === 1 ? '' : 's'}
                </span>
              )}
              {caps.gpu_present && <span>GPU</span>}
              {caps.max_vram_seen_bytes ? (
                <span>
                  {(caps.max_vram_seen_bytes / 1e9).toFixed(1)} GB VRAM seen
                </span>
              ) : null}
            </div>
          ) : (
            <div className="mt-1 text-[11px] text-amber-500">
              No worker row yet — capability probe pending.
            </div>
          )}

          {worker?.last_error && (
            <div className="mt-1 truncate text-[11px] text-destructive">
              {worker.last_error}
            </div>
          )}

          {/* Inline workload-routing toggles. Showing them per-row on the
              paired list means the user can shape the pool without
              opening a separate edit dialog. */}
          {worker && (
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
              <InlineToggle
                label="chat"
                checked={!!worker.use_for_chat}
                onChange={(v) => onToggle('use_for_chat', v)}
              />
              <InlineToggle
                label="embed"
                checked={!!worker.use_for_embeddings}
                onChange={(v) => onToggle('use_for_embeddings', v)}
              />
              <InlineToggle
                label="subagents"
                checked={!!worker.use_for_subagents}
                onChange={(v) => onToggle('use_for_subagents', v)}
              />
              <InlineToggle
                label="enabled"
                checked={!!worker.enabled}
                onChange={(v) => onToggle('enabled', v)}
              />
            </div>
          )}
        </div>

        <div className="flex shrink-0 gap-1">
          {worker && (
            <Button
              variant="ghost"
              size="icon"
              onClick={onProbe}
              disabled={probing || !worker.enabled}
              className="h-7 w-7 text-muted-foreground hover:text-foreground"
              title="Test connection now"
            >
              <RefreshCw className={cn('h-4 w-4', probing && 'animate-spin')} />
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={onUnpair}
            className="h-7 w-7 text-muted-foreground hover:text-destructive"
            title="Unpair (also removes the worker row)"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </li>
  )
}

/** Legacy worker row (manually-added, no paired peer). Slimmer than the
 *  paired-row because there's no E2E badge and no identity to show.
 *  Inline workload toggles + delete-only action. */
function LegacyWorkerRow({ worker, probing, onProbe, onToggle, onDelete }) {
  const online = isWorkerOnline(worker)
  const caps = worker.capabilities || {}
  const modelCount = Array.isArray(caps.models) ? caps.models.length : 0

  let StatusIcon = CircleHelp
  let statusLabel = 'Never seen'
  let statusClass = 'text-muted-foreground'
  if (!worker.enabled) {
    StatusIcon = CircleX
    statusLabel = 'Disabled'
  } else if (online) {
    StatusIcon = CircleCheck
    statusLabel = 'Online'
    statusClass = 'text-emerald-500'
  } else if (worker.last_error) {
    StatusIcon = CircleX
    statusLabel = 'Unreachable'
    statusClass = 'text-destructive'
  }

  return (
    <li className="rounded border border-border/50 bg-background/50 p-3">
      <div className="flex items-start gap-3">
        <Server className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-2 text-sm">
            <span className="font-semibold">{worker.label}</span>
            <span
              className={cn(
                'inline-flex items-center gap-1 text-[11px]',
                statusClass,
              )}
              title={worker.last_error || statusLabel}
            >
              <StatusIcon className="h-3 w-3" /> {statusLabel}
            </span>
          </div>
          <div className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
            {worker.address}:{worker.ollama_port}
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] text-muted-foreground">
            {caps.version && (
              <span className="inline-flex items-center gap-1">
                <Cpu className="h-3 w-3" /> Ollama {caps.version}
              </span>
            )}
            {modelCount > 0 && (
              <span title={(caps.models || []).map((m) => m.name).join(', ')}>
                {modelCount} model{modelCount === 1 ? '' : 's'}
              </span>
            )}
          </div>
          {worker.last_error && (
            <div className="mt-1 truncate text-[11px] text-destructive">
              {worker.last_error}
            </div>
          )}
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
            <InlineToggle
              label="chat"
              checked={!!worker.use_for_chat}
              onChange={(v) => onToggle('use_for_chat', v)}
            />
            <InlineToggle
              label="embed"
              checked={!!worker.use_for_embeddings}
              onChange={(v) => onToggle('use_for_embeddings', v)}
            />
            <InlineToggle
              label="subagents"
              checked={!!worker.use_for_subagents}
              onChange={(v) => onToggle('use_for_subagents', v)}
            />
            <InlineToggle
              label="enabled"
              checked={!!worker.enabled}
              onChange={(v) => onToggle('enabled', v)}
            />
          </div>
        </div>
        <div className="flex shrink-0 gap-1">
          <Button
            variant="ghost"
            size="icon"
            onClick={onProbe}
            disabled={probing || !worker.enabled}
            className="h-7 w-7 text-muted-foreground hover:text-foreground"
            title="Test connection now"
          >
            <RefreshCw className={cn('h-4 w-4', probing && 'animate-spin')} />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={onDelete}
            className="h-7 w-7 text-destructive hover:text-destructive"
            title="Remove worker"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </li>
  )
}

function InlineToggle({ label, checked, onChange }) {
  return (
    <label className="inline-flex cursor-pointer items-center gap-1.5 text-muted-foreground">
      <Switch
        checked={!!checked}
        onCheckedChange={onChange}
        className="scale-75"
      />
      <span>{label}</span>
    </label>
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
