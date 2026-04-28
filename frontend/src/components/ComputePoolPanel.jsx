import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'
import {
  Plus,
  Trash2,
  Pencil,
  RefreshCw,
  Server,
  Wifi,
  CircleCheck,
  CircleX,
  CircleHelp,
  Cpu,
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
import { cn } from '@/lib/utils'
import { api } from '@/lib/api'
import AutoSplitInstallSection from './SplitModelsPanel'

/**
 * ComputePoolSection — body for the "Compute Pool" tab inside SettingsPanel.
 *
 * The compute pool lets the user register other PCs (laptops, spare desktops)
 * as Ollama workers. The host then routes a slice of its workload to those
 * machines so big models / parallel subagents finish faster, and cross-LAN
 * fanout doesn't fight a single GPU.
 *
 * All ongoing traffic — chat, embeddings, subagent calls — flows over the
 * LAN address. An optional Tailscale identifier is stored alongside it
 * purely as a self-repair handle: when the worker rejoins the LAN and DHCP
 * gives it a different IP, the backend reaches it over Tailscale just long
 * enough to rediscover the new LAN address. Tailscale is never used for
 * regular traffic, so the user's metered internet bandwidth is preserved.
 *
 * Each row shows the live probe status (online / unreachable / never seen),
 * the worker's Ollama version, and how many models it has installed. The
 * "Test connection" action triggers an out-of-band probe so the user
 * doesn't have to wait for the 5-min sweep to confirm a freshly-edited row.
 */
export default function ComputePoolSection() {
  const [workers, setWorkers] = useState([])
  const [loading, setLoading] = useState(false)
  // Form state for the add/edit dialog. `null` = dialog closed.
  const [editing, setEditing] = useState(null)
  const [pendingDelete, setPendingDelete] = useState(null)
  // Worker IDs currently being probed — used to spin the per-row icon.
  const [probing, setProbing] = useState(() => new Set())
  const [refreshingAll, setRefreshingAll] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.listComputeWorkers()
      setWorkers(res.workers || [])
    } catch (e) {
      toast.error('Failed to load compute workers', { description: e.message })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  /** Default form state for "Add device". */
  function blankForm() {
    return {
      label: '',
      address: '',
      ollama_port: 11434,
      auth_token: '',
      ssh_host: '',
      tailscale_host: '',
      enabled: true,
      use_for_chat: true,
      use_for_embeddings: true,
      use_for_subagents: true,
    }
  }

  function startAdd() {
    setEditing({ ...blankForm(), _mode: 'create' })
  }

  function startEdit(w) {
    setEditing({
      _mode: 'edit',
      id: w.id,
      label: w.label || '',
      address: w.address || '',
      ollama_port: w.ollama_port ?? 11434,
      // Empty string means "leave token alone" on save (the placeholder
      // "••••••" is rendered when the row already has one).
      auth_token: '',
      auth_token_was_set: !!w.auth_token_set,
      ssh_host: w.ssh_host || '',
      tailscale_host: w.tailscale_host || '',
      enabled: !!w.enabled,
      use_for_chat: !!w.use_for_chat,
      use_for_embeddings: !!w.use_for_embeddings,
      use_for_subagents: !!w.use_for_subagents,
    })
  }

  async function saveEditing() {
    if (!editing) return
    const label = (editing.label || '').trim()
    const address = (editing.address || '').trim()
    if (!label) return toast.error('Label is required')
    if (!address) return toast.error('Address is required')

    const port = Number(editing.ollama_port)
    if (!Number.isFinite(port) || port < 1 || port > 65535) {
      return toast.error('Port must be 1–65535')
    }

    try {
      if (editing._mode === 'edit') {
        const patch = {
          label,
          address,
          ollama_port: port,
          ssh_host: editing.ssh_host || '',
          tailscale_host: editing.tailscale_host || '',
          enabled: editing.enabled,
          use_for_chat: editing.use_for_chat,
          use_for_embeddings: editing.use_for_embeddings,
          use_for_subagents: editing.use_for_subagents,
        }
        // Only patch the token if the user typed something; an empty
        // textbox means "no change". Backend treats "" as a clear, but we
        // never actually send "" from this UI — there's a separate "Clear
        // token" affordance to cover that intent explicitly.
        if (editing.auth_token) patch.auth_token = editing.auth_token
        await api.updateComputeWorker(editing.id, patch)
        toast.success('Worker updated')
      } else {
        await api.createComputeWorker({
          label,
          address,
          ollama_port: port,
          auth_token: editing.auth_token || null,
          ssh_host: editing.ssh_host || null,
          tailscale_host: editing.tailscale_host || null,
          enabled: editing.enabled,
          use_for_chat: editing.use_for_chat,
          use_for_embeddings: editing.use_for_embeddings,
          use_for_subagents: editing.use_for_subagents,
        })
        toast.success('Worker added', {
          description: `${label} registered. A capability probe will fire shortly — or use "Test connection" now.`,
        })
      }
      setEditing(null)
      refresh()
    } catch (e) {
      toast.error('Save failed', { description: e.message })
    }
  }

  async function clearToken() {
    if (!editing?.id) return
    try {
      await api.updateComputeWorker(editing.id, { auth_token: '' })
      toast.success('Auth token cleared')
      setEditing({ ...editing, auth_token: '', auth_token_was_set: false })
      refresh()
    } catch (e) {
      toast.error('Clear failed', { description: e.message })
    }
  }

  async function confirmDelete() {
    if (!pendingDelete) return
    try {
      await api.deleteComputeWorker(pendingDelete.id)
      toast.success('Worker removed')
      setPendingDelete(null)
      refresh()
    } catch (e) {
      toast.error('Delete failed', { description: e.message })
    }
  }

  async function probeOne(w) {
    setProbing((s) => new Set(s).add(w.id))
    try {
      const res = await api.probeComputeWorker(w.id)
      if (res.ok) {
        const ver = res.capabilities?.version
        const n = res.capabilities?.models?.length || 0
        toast.success(`${w.label} is online`, {
          description: `Ollama ${ver || '?'} · ${n} model${n === 1 ? '' : 's'} installed`,
        })
      } else {
        toast.error(`${w.label} probe failed`, {
          description: res.error || 'unknown error',
        })
      }
    } catch (e) {
      toast.error('Probe failed', { description: e.message })
    } finally {
      setProbing((s) => {
        const copy = new Set(s)
        copy.delete(w.id)
        return copy
      })
      // Re-fetch the list so the row's last_seen / last_error reflects the
      // probe outcome we just persisted server-side.
      refresh()
    }
  }

  async function probeAll() {
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
      refresh()
    }
  }

  const summary = useMemo(() => {
    if (loading) return 'Loading…'
    if (!workers.length) return 'No workers yet'
    const online = workers.filter((w) => isOnline(w)).length
    return `${workers.length} worker${workers.length === 1 ? '' : 's'} · ${online} online`
  }, [workers, loading])

  return (
    <>
      <div className="flex max-h-[60vh] flex-col overflow-hidden">
        <div className="flex items-center justify-between pb-2">
          <div className="text-xs text-muted-foreground">{summary}</div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={probeAll}
              disabled={refreshingAll || !workers.length}
              className="gap-1.5 text-xs"
              title="Re-probe every enabled worker now"
            >
              <RefreshCw
                className={cn('h-3.5 w-3.5', refreshingAll && 'animate-spin')}
              />
              Refresh all
            </Button>
            <Button size="sm" onClick={startAdd} className="gap-2">
              <Plus className="h-4 w-4" /> Add device
            </Button>
          </div>
        </div>

        <p className="pb-2 text-xs text-muted-foreground">
          Register other PCs running Ollama as compute workers. The host will
          send a slice of chat / embedding / subagent traffic to them so
          parallel work finishes faster.
          <br />
          All ongoing traffic stays on the <strong className="text-foreground">LAN</strong>.
          A Tailscale identifier is optional — used only to rediscover the
          worker's LAN IP after it changes (e.g. a fresh DHCP lease).
        </p>

        <div className="flex-1 space-y-2 overflow-y-auto pr-1">
          {workers.length === 0 && !loading && (
            <p className="py-8 text-center text-sm text-muted-foreground">
              No compute workers yet. Click <em>Add device</em> to register a
              laptop or spare desktop. The host probes it for Ollama version
              and installed models, then starts routing work to it.
            </p>
          )}
          {workers.map((w) => (
            <WorkerRow
              key={w.id}
              worker={w}
              probing={probing.has(w.id)}
              onProbe={() => probeOne(w)}
              onEdit={() => startEdit(w)}
              onDelete={() => setPendingDelete(w)}
            />
          ))}
        </div>

        {/* Phase 2 install banner only — actual split-model routing is
            fully automatic. When the user picks a model in chat, the
            backend's `compute_pool.route_chat_for` decides whether the
            model fits the host's VRAM (→ Ollama) or needs to fan across
            workers via llama.cpp RPC (→ llama-server auto-spawned).
            The user never sees a "split models" registry — just this
            small banner that asks for the one-time llama.cpp install
            when they want to enable the big-model path. */}
        <AutoSplitInstallSection />
      </div>

      {/* Add / edit drawer */}
      <Dialog
        open={!!editing}
        onOpenChange={(o) => {
          if (!o) setEditing(null)
        }}
      >
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>
              {editing?._mode === 'edit' ? 'Edit worker' : 'Add compute worker'}
            </DialogTitle>
            <DialogDescription>
              {editing?._mode === 'edit'
                ? 'Update the connection details. Capabilities re-probe automatically; click Save and then "Test connection" on the row.'
                : 'Point to another PC running Ollama on the same LAN. Add a Tailscale identifier if you want the LAN IP to self-heal after a DHCP rebind.'}
            </DialogDescription>
          </DialogHeader>
          {editing && (
            <div className="space-y-3">
              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">
                  Label
                </label>
                <Input
                  value={editing.label}
                  onChange={(e) =>
                    setEditing({ ...editing, label: e.target.value })
                  }
                  placeholder="e.g. office laptop, spare desktop"
                  autoFocus={editing._mode !== 'edit'}
                />
              </div>

              <div className="grid grid-cols-3 gap-2">
                <div className="col-span-2">
                  <label className="mb-1 block text-xs font-medium text-muted-foreground">
                    LAN address
                  </label>
                  <Input
                    value={editing.address}
                    onChange={(e) =>
                      setEditing({ ...editing, address: e.target.value })
                    }
                    placeholder="worker.local or 192.168.x.x"
                  />
                  <p className="mt-1 text-[11px] text-muted-foreground">
                    Hostname or private IPv4 on the same Wi-Fi/Ethernet.
                    Public IPs and Tailscale CGNAT addresses are not
                    accepted here — all ongoing traffic stays on the LAN.
                  </p>
                </div>
                <div>
                  <label className="mb-1 block text-xs font-medium text-muted-foreground">
                    Port
                  </label>
                  <Input
                    type="number"
                    inputMode="numeric"
                    min={1}
                    max={65535}
                    value={editing.ollama_port}
                    onChange={(e) =>
                      setEditing({
                        ...editing,
                        ollama_port: e.target.value,
                      })
                    }
                  />
                </div>
              </div>

              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">
                  Auth token (optional)
                </label>
                <Input
                  type="password"
                  value={editing.auth_token}
                  onChange={(e) =>
                    setEditing({ ...editing, auth_token: e.target.value })
                  }
                  placeholder={
                    editing.auth_token_was_set
                      ? '••••••• (leave blank to keep current)'
                      : 'Bearer token the worker validates'
                  }
                />
                <div className="mt-1 flex items-center justify-between">
                  <p className="text-[11px] text-muted-foreground">
                    Sent as <code>Authorization: Bearer …</code> on every
                    request to this worker. Required if its Ollama is exposed
                    beyond loopback.
                  </p>
                  {editing._mode === 'edit' && editing.auth_token_was_set && (
                    <button
                      type="button"
                      onClick={clearToken}
                      className="ml-2 shrink-0 text-[11px] text-destructive hover:underline"
                    >
                      Clear token
                    </button>
                  )}
                </div>
              </div>

              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">
                  SSH host (optional, for LAN model copy)
                </label>
                <Input
                  value={editing.ssh_host}
                  onChange={(e) =>
                    setEditing({ ...editing, ssh_host: e.target.value })
                  }
                  placeholder="e.g. laptop (an alias from your ~/.ssh/config)"
                />
                <p className="mt-1 text-[11px] text-muted-foreground">
                  When set, this host can scp Ollama model blobs to the
                  worker over LAN instead of having the worker pull from
                  the internet. Saves bandwidth on multi-GB models. Add
                  the alias to <code className="rounded bg-muted px-1">~/.ssh/config</code>{' '}
                  on this host first; the backend just reuses your existing SSH setup.
                </p>
              </div>

              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">
                  Tailscale host (optional, for auto-repair only)
                </label>
                <Input
                  value={editing.tailscale_host}
                  onChange={(e) =>
                    setEditing({ ...editing, tailscale_host: e.target.value })
                  }
                  placeholder="e.g. worker.your-tailnet.ts.net or 100.x.x.x"
                />
                <p className="mt-1 text-[11px] text-muted-foreground">
                  Stable Tailscale identifier (MagicDNS name or CGNAT IPv4).
                  When the LAN address goes stale — typically because the
                  worker rejoined the network and DHCP gave it a new lease —
                  the host SSHes over Tailscale just long enough to rediscover
                  the new LAN IP. Never used for ongoing chat / embedding
                  traffic, so your metered internet bandwidth is preserved.
                </p>
              </div>

              <div className="rounded-md border border-border bg-muted/30 p-3">
                <div className="mb-2 text-xs font-medium text-foreground">
                  Workload routing
                </div>
                <ToggleRow
                  label="Use for chat"
                  hint="Stream user-facing chat through this worker."
                  value={editing.use_for_chat}
                  onChange={(v) =>
                    setEditing({ ...editing, use_for_chat: v })
                  }
                />
                <ToggleRow
                  label="Use for embeddings"
                  hint="Run vector embedding requests here."
                  value={editing.use_for_embeddings}
                  onChange={(v) =>
                    setEditing({ ...editing, use_for_embeddings: v })
                  }
                />
                <ToggleRow
                  label="Use for subagents"
                  hint="Distribute parallel delegate_parallel calls."
                  value={editing.use_for_subagents}
                  onChange={(v) =>
                    setEditing({ ...editing, use_for_subagents: v })
                  }
                />
                <div className="mt-2 border-t border-border/60 pt-2">
                  <ToggleRow
                    label="Enabled"
                    hint="Toggle off to skip this worker without deleting the row."
                    value={editing.enabled}
                    onChange={(v) =>
                      setEditing({ ...editing, enabled: v })
                    }
                  />
                </div>
              </div>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditing(null)}>
              Cancel
            </Button>
            <Button onClick={saveEditing}>
              {editing?._mode === 'edit' ? 'Save changes' : 'Add worker'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete confirm */}
      <Dialog
        open={!!pendingDelete}
        onOpenChange={(o) => !o && setPendingDelete(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Remove this worker?</DialogTitle>
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
            <Button variant="destructive" onClick={confirmDelete}>
              <Trash2 className="mr-1 h-4 w-4" /> Remove
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

/** Liveness rule: a worker counts as online when its last probe succeeded
 * (no `last_error`) and we have a `last_seen` timestamp from within the
 * last hour. The 1-hour window covers the 5-min sweep cadence with plenty
 * of buffer for intermittent network blips. */
function isOnline(w) {
  if (!w?.enabled) return false
  if (!w.last_seen) return false
  if (w.last_error) return false
  const ageSec = Date.now() / 1000 - Number(w.last_seen)
  return ageSec >= 0 && ageSec < 60 * 60
}

/** One row inside the "Workload routing" form section. */
function ToggleRow({ label, hint, value, onChange }) {
  return (
    <div className="flex items-center justify-between gap-3 py-1">
      <div className="min-w-0">
        <div className="text-xs font-medium text-foreground">{label}</div>
        <div className="text-[11px] text-muted-foreground">{hint}</div>
      </div>
      <Switch checked={!!value} onCheckedChange={onChange} />
    </div>
  )
}

/** One worker row — status pill, label/address, model count, action buttons. */
function WorkerRow({ worker, probing, onProbe, onEdit, onDelete }) {
  const online = isOnline(worker)
  const neverSeen = !worker.last_seen
  const caps = worker.capabilities || {}
  const modelCount = Array.isArray(caps.models) ? caps.models.length : 0

  // Status badge — three states cover the meaningful outcomes:
  //   * green — last probe succeeded recently
  //   * red   — probe failed or row is disabled
  //   * gray  — never probed yet (just added)
  let StatusIcon = CircleHelp
  let statusLabel = 'Never seen'
  let statusClass = 'text-muted-foreground'
  if (!worker.enabled) {
    StatusIcon = CircleX
    statusLabel = 'Disabled'
    statusClass = 'text-muted-foreground'
  } else if (online) {
    StatusIcon = CircleCheck
    statusLabel = 'Online'
    statusClass = 'text-emerald-500'
  } else if (worker.last_error) {
    StatusIcon = CircleX
    statusLabel = 'Unreachable'
    statusClass = 'text-destructive'
  } else if (neverSeen) {
    StatusIcon = CircleHelp
    statusLabel = 'Never probed'
    statusClass = 'text-muted-foreground'
  }

  // Build a compact "uses" pill string — chat / embed / subagents — so the
  // user can see at a glance which workloads route here without opening the
  // edit dialog. Order matches the form.
  const uses = [
    worker.use_for_chat && 'chat',
    worker.use_for_embeddings && 'embed',
    worker.use_for_subagents && 'subagents',
  ].filter(Boolean)

  return (
    <div className="flex items-start gap-3 rounded-md border border-border bg-card/40 p-3">
      <Server className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-2">
          <span className="text-sm font-semibold text-foreground">
            {worker.label}
          </span>
          <span
            className={cn('inline-flex items-center gap-1 text-[11px]', statusClass)}
            title={worker.last_error || statusLabel}
          >
            <StatusIcon className="h-3 w-3" /> {statusLabel}
          </span>
          <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
            <Wifi className="h-3 w-3" /> LAN
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
          {uses.length > 0 && (
            <span>uses: {uses.join(', ')}</span>
          )}
          {worker.auth_token_set && <span>auth: set</span>}
          {worker.tailscale_host && <span title={`auto-repair via ${worker.tailscale_host}`}>auto-repair: on</span>}
        </div>

        {worker.last_error && (
          <div className="mt-1 truncate text-[11px] text-destructive">
            {worker.last_error}
          </div>
        )}
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
          onClick={onEdit}
          className="h-7 w-7 text-muted-foreground hover:text-foreground"
          title="Edit worker"
        >
          <Pencil className="h-4 w-4" />
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
  )
}
