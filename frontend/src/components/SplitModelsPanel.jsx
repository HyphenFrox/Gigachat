import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import {
  Plus,
  Trash2,
  Pencil,
  Play,
  Square,
  RefreshCw,
  Layers,
  CircleCheck,
  CircleX,
  CircleHelp,
  Download,
  AlertTriangle,
  Loader2,
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

/**
 * SplitModelsSection — Phase 2 of the compute pool.
 *
 * Lets the user register "split models": one big GGUF whose layers fan
 * across host VRAM/RAM + every selected worker's CPU/GPU/RAM via
 * llama.cpp's --rpc mechanism. The UI is a sibling to the Workers
 * section above (Phase 1) inside the Compute tab.
 *
 * Flow:
 *   1. User installs llama.cpp on the host (one-time, ~150 MB download).
 *   2. User adds a split-model row: gguf_path + which workers to split
 *      across.
 *   3. User clicks Start — backend spawns llama-server; UI polls status
 *      until it goes to `running`.
 *   4. Once `running`, the model appears in the main chat picker as
 *      `split:<label>` and chat turns route there.
 */
export default function SplitModelsSection({ workers = [] }) {
  const [rows, setRows] = useState([])
  const [llamaCpp, setLlamaCpp] = useState(null)
  const [loading, setLoading] = useState(false)
  const [installing, setInstalling] = useState(false)
  const [editing, setEditing] = useState(null)
  const [pendingDelete, setPendingDelete] = useState(null)
  const [busyIds, setBusyIds] = useState(() => new Set()) // start/stop in flight
  // Polling timer — kicks in while any row is `loading` so the UI
  // reflects the transition to running/error without the user having
  // to click Refresh.
  const pollRef = useRef(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.listSplitModels()
      setRows(res.split_models || [])
      setLlamaCpp(res.llama_cpp || null)
    } catch (e) {
      toast.error('Failed to load split models', { description: e.message })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  // Status-polling: while any row is in a transient state (`loading`),
  // re-fetch every 2 s so the UI shows the running/error transition
  // without manual refresh.
  useEffect(() => {
    const anyLoading = rows.some((r) => r.status === 'loading')
    if (anyLoading) {
      if (!pollRef.current) {
        pollRef.current = setInterval(refresh, 2000)
      }
    } else if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [rows, refresh])

  function blankForm() {
    return {
      label: '',
      gguf_path: '',
      worker_ids: [],
      llama_port: 11500,
      enabled: true,
    }
  }

  function startAdd() {
    setEditing({ ...blankForm(), _mode: 'create' })
  }

  function startEdit(r) {
    setEditing({
      _mode: 'edit',
      id: r.id,
      label: r.label || '',
      gguf_path: r.gguf_path || '',
      worker_ids: Array.isArray(r.worker_ids) ? [...r.worker_ids] : [],
      llama_port: r.llama_port || 11500,
      enabled: !!r.enabled,
    })
  }

  function toggleWorkerInForm(wid) {
    setEditing((e) => {
      if (!e) return e
      const ids = e.worker_ids.includes(wid)
        ? e.worker_ids.filter((x) => x !== wid)
        : [...e.worker_ids, wid]
      return { ...e, worker_ids: ids }
    })
  }

  async function saveEditing() {
    if (!editing) return
    const label = (editing.label || '').trim()
    const path = (editing.gguf_path || '').trim()
    if (!label) return toast.error('Label is required')
    if (!path) return toast.error('GGUF path is required')
    const port = Number(editing.llama_port)
    if (!Number.isFinite(port) || port < 1 || port > 65535) {
      return toast.error('Port must be 1–65535')
    }

    try {
      if (editing._mode === 'edit') {
        await api.updateSplitModel(editing.id, {
          label,
          gguf_path: path,
          worker_ids: editing.worker_ids,
          llama_port: port,
          enabled: editing.enabled,
        })
        toast.success('Split model updated')
      } else {
        await api.createSplitModel({
          label,
          gguf_path: path,
          worker_ids: editing.worker_ids,
          llama_port: port,
          enabled: editing.enabled,
        })
        toast.success('Split model added', {
          description: 'Click Start to spawn llama-server with these workers.',
        })
      }
      setEditing(null)
      refresh()
    } catch (e) {
      toast.error('Save failed', { description: e.message })
    }
  }

  async function confirmDelete() {
    if (!pendingDelete) return
    try {
      await api.deleteSplitModel(pendingDelete.id)
      toast.success('Split model removed')
      setPendingDelete(null)
      refresh()
    } catch (e) {
      toast.error('Delete failed', { description: e.message })
    }
  }

  async function startOne(r) {
    setBusyIds((s) => new Set(s).add(r.id))
    // Optimistic: mark row loading immediately so the UI reflects
    // the click before the (possibly multi-second) API call returns.
    setRows((rs) =>
      rs.map((x) => (x.id === r.id ? { ...x, status: 'loading' } : x)),
    )
    try {
      const res = await api.startSplitModel(r.id)
      if (res.ok) {
        toast.success(`${r.label} running on port ${res.port}`)
      } else {
        toast.error(`${r.label} failed to start`, {
          description: res.error || 'unknown error',
        })
      }
    } catch (e) {
      toast.error('Start failed', { description: e.message })
    } finally {
      setBusyIds((s) => {
        const c = new Set(s)
        c.delete(r.id)
        return c
      })
      refresh()
    }
  }

  async function stopOne(r) {
    setBusyIds((s) => new Set(s).add(r.id))
    try {
      await api.stopSplitModel(r.id)
      toast.success(`${r.label} stopped`)
    } catch (e) {
      toast.error('Stop failed', { description: e.message })
    } finally {
      setBusyIds((s) => {
        const c = new Set(s)
        c.delete(r.id)
        return c
      })
      refresh()
    }
  }

  async function doInstall() {
    setInstalling(true)
    toast.info('Downloading llama.cpp…', {
      description: 'Multi-hundred-MB download — this can take a minute.',
    })
    try {
      await api.installLlamaCpp('host')
      toast.success('llama.cpp installed')
    } catch (e) {
      toast.error('Install failed', { description: e.message })
    } finally {
      setInstalling(false)
      refresh()
    }
  }

  // Workers eligible to be selected as RPC contributors. Only
  // enabled workers whose latest probe says rpc-server is reachable
  // make sense — otherwise llama-server would just connection-refuse
  // when it tried to dial them.
  const eligibleWorkers = useMemo(
    () =>
      (workers || []).filter(
        (w) => w.enabled && w.capabilities?.rpc_server_reachable,
      ),
    [workers],
  )
  // Workers the user added but rpc isn't running on yet — surface as
  // a hint inside the form so they know why their laptop isn't on the
  // checkbox list.
  const workersNeedingRpc = useMemo(
    () =>
      (workers || []).filter(
        (w) => w.enabled && !w.capabilities?.rpc_server_reachable,
      ),
    [workers],
  )

  return (
    <>
      <div className="border-t border-border pt-4">
        <div className="flex items-center justify-between pb-2">
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-semibold">Split models</span>
            <span className="text-xs text-muted-foreground">
              ({loading ? 'loading…' : `${rows.length}`})
            </span>
          </div>
          <Button
            size="sm"
            onClick={startAdd}
            className="gap-2"
            disabled={!llamaCpp?.installed}
            title={
              llamaCpp?.installed
                ? 'Register a new split model'
                : 'Install llama.cpp first'
            }
          >
            <Plus className="h-4 w-4" /> Add split model
          </Button>
        </div>

        <p className="pb-2 text-xs text-muted-foreground">
          Run one big model whose layers fan across the host (CPU + GPU + RAM
          + VRAM) and one or more workers (CPU + iGPU + RAM). Useful for models
          too big to fit a single machine. Slower per-token than running fully
          on one machine, but it{"'"}s the only way to run a model that exceeds
          any single node{"'"}s memory.
        </p>

        {/* llama.cpp install banner */}
        {llamaCpp && !llamaCpp.installed && (
          <div className="mb-3 flex items-start gap-3 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-xs">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />
            <div className="flex-1">
              <p className="font-medium text-foreground">
                llama.cpp not installed on this host
              </p>
              <p className="mt-0.5 text-muted-foreground">
                Phase 2 needs llama.cpp{"'"}s prebuilt Windows binaries
                (~150 MB). Install once; afterward you can register split
                models and start them from this panel.
              </p>
              {!llamaCpp.platform_supported && (
                <p className="mt-1 text-destructive">
                  Auto-install unsupported here: {llamaCpp.platform_reason}
                </p>
              )}
            </div>
            <Button
              size="sm"
              onClick={doInstall}
              disabled={installing || !llamaCpp.platform_supported}
              className="gap-1.5"
            >
              {installing ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Download className="h-3.5 w-3.5" />
              )}
              {installing ? 'Installing…' : 'Install llama.cpp'}
            </Button>
          </div>
        )}

        {llamaCpp?.installed && (
          <p className="mb-2 text-[11px] text-muted-foreground">
            llama.cpp <span className="font-medium text-foreground">{llamaCpp.version}</span>{' '}
            at <code className="rounded bg-muted px-1">{llamaCpp.install_dir}</code>
          </p>
        )}

        <div className="space-y-2">
          {rows.length === 0 && !loading && (
            <p className="py-6 text-center text-sm text-muted-foreground">
              No split models yet.{' '}
              {llamaCpp?.installed
                ? 'Click Add split model to register one.'
                : 'Install llama.cpp first, then add a model.'}
            </p>
          )}
          {rows.map((r) => (
            <SplitRow
              key={r.id}
              row={r}
              busy={busyIds.has(r.id)}
              workers={workers}
              onStart={() => startOne(r)}
              onStop={() => stopOne(r)}
              onEdit={() => startEdit(r)}
              onDelete={() => setPendingDelete(r)}
            />
          ))}
        </div>
      </div>

      {/* Add / edit dialog */}
      <Dialog
        open={!!editing}
        onOpenChange={(o) => {
          if (!o) setEditing(null)
        }}
      >
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>
              {editing?._mode === 'edit' ? 'Edit split model' : 'Add split model'}
            </DialogTitle>
            <DialogDescription>
              Point at a GGUF on this host and choose which workers should
              host its layers. The host{"'"}s GPU + CPU always participate —
              workers add more memory + compute.
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
                  placeholder="e.g. qwen-27b-q3, gemma-22b-q4"
                  autoFocus={editing._mode !== 'edit'}
                />
                <p className="mt-1 text-[11px] text-muted-foreground">
                  Shows up in the main chat model picker as{' '}
                  <code className="rounded bg-muted px-1">
                    split:{editing.label || '<label>'}
                  </code>
                  .
                </p>
              </div>

              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">
                  GGUF path
                </label>
                <Input
                  value={editing.gguf_path}
                  onChange={(e) =>
                    setEditing({ ...editing, gguf_path: e.target.value })
                  }
                  placeholder="C:\\Users\\you\\.ollama\\models\\blobs\\sha256-..."
                />
                <p className="mt-1 text-[11px] text-muted-foreground">
                  Absolute path on this host. Ollama-managed blobs work
                  directly — find the right hash via{' '}
                  <code className="rounded bg-muted px-1">ollama show MODEL --modelfile</code>.
                </p>
              </div>

              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">
                  Local port
                </label>
                <Input
                  type="number"
                  inputMode="numeric"
                  min={1024}
                  max={65535}
                  value={editing.llama_port}
                  onChange={(e) =>
                    setEditing({
                      ...editing,
                      llama_port: Number(e.target.value),
                    })
                  }
                />
                <p className="mt-1 text-[11px] text-muted-foreground">
                  Where llama-server listens locally (default 11500). Only
                  the host{"'"}s router talks to it; not exposed on LAN.
                </p>
              </div>

              <div>
                <label className="mb-2 block text-xs font-medium text-muted-foreground">
                  Workers contributing layers ({editing.worker_ids.length} selected)
                </label>
                {eligibleWorkers.length === 0 ? (
                  <p className="rounded-md border border-border bg-muted/30 p-3 text-xs text-muted-foreground">
                    No workers have rpc-server reachable. Run rpc-server on
                    each worker (Vulkan build for iGPU support), then
                    Refresh on the worker row above.
                  </p>
                ) : (
                  <div className="space-y-1.5 rounded-md border border-border bg-muted/30 p-2">
                    {eligibleWorkers.map((w) => {
                      const checked = editing.worker_ids.includes(w.id)
                      return (
                        <label
                          key={w.id}
                          className="flex items-center gap-2 rounded p-1.5 hover:bg-accent"
                        >
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={() => toggleWorkerInForm(w.id)}
                            className="h-3.5 w-3.5"
                          />
                          <span className="text-sm font-medium text-foreground">
                            {w.label}
                          </span>
                          <span className="text-[11px] text-muted-foreground">
                            {w.address}
                          </span>
                        </label>
                      )
                    })}
                  </div>
                )}
                {workersNeedingRpc.length > 0 && (
                  <p className="mt-1 text-[11px] text-amber-500">
                    {workersNeedingRpc.length} worker{workersNeedingRpc.length === 1 ? '' : 's'}{' '}
                    enabled but rpc-server not reachable; not selectable.
                  </p>
                )}
              </div>

              <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-muted/30 p-3">
                <div>
                  <div className="text-xs font-medium text-foreground">Enabled</div>
                  <div className="text-[11px] text-muted-foreground">
                    Routing skips disabled rows even if llama-server happens
                    to still be running.
                  </div>
                </div>
                <Switch
                  checked={!!editing.enabled}
                  onCheckedChange={(v) =>
                    setEditing({ ...editing, enabled: v })
                  }
                />
              </div>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditing(null)}>
              Cancel
            </Button>
            <Button onClick={saveEditing}>
              {editing?._mode === 'edit' ? 'Save changes' : 'Add'}
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
            <DialogTitle>Remove this split model?</DialogTitle>
            <DialogDescription>
              <strong>{pendingDelete?.label}</strong> will be stopped (if
              running) and the row deleted. Conversations using this model
              will need to be re-pointed.
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

/** One split-model row. Status pill + start/stop/edit/delete actions. */
function SplitRow({ row, busy, workers, onStart, onStop, onEdit, onDelete }) {
  const isRunning = row.status === 'running'
  const isLoading = row.status === 'loading'
  const isError = row.status === 'error'

  let StatusIcon = CircleHelp
  let statusLabel = row.status
  let statusClass = 'text-muted-foreground'
  if (isRunning) {
    StatusIcon = CircleCheck
    statusLabel = 'Running'
    statusClass = 'text-emerald-500'
  } else if (isLoading) {
    StatusIcon = Loader2
    statusLabel = 'Loading…'
    statusClass = 'text-amber-500'
  } else if (isError) {
    StatusIcon = CircleX
    statusLabel = 'Error'
    statusClass = 'text-destructive'
  } else if (row.status === 'stopped') {
    StatusIcon = Square
    statusLabel = 'Stopped'
    statusClass = 'text-muted-foreground'
  }

  // Resolve worker labels for display.
  const workerLabels = (row.worker_ids || []).map((wid) => {
    const w = (workers || []).find((x) => x.id === wid)
    return w ? w.label : wid.slice(0, 8)
  })

  return (
    <div className="flex items-start gap-3 rounded-md border border-border bg-card/40 p-3">
      <Layers className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-2">
          <span className="text-sm font-semibold text-foreground">
            {row.label}
          </span>
          <span
            className={cn(
              'inline-flex items-center gap-1 text-[11px]',
              statusClass,
            )}
            title={row.last_error || statusLabel}
          >
            <StatusIcon
              className={cn('h-3 w-3', isLoading && 'animate-spin')}
            />
            {statusLabel}
          </span>
          <span className="text-[11px] text-muted-foreground">
            port {row.llama_port}
          </span>
          {!row.enabled && (
            <span className="text-[11px] text-muted-foreground">disabled</span>
          )}
        </div>
        <div
          className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground"
          title={row.gguf_path}
        >
          {row.gguf_path}
        </div>
        <div className="mt-1 text-[11px] text-muted-foreground">
          Workers: {workerLabels.length === 0 ? 'host only' : workerLabels.join(', ')}
        </div>
        {row.last_error && (
          <div className="mt-1 truncate text-[11px] text-destructive">
            {row.last_error}
          </div>
        )}
      </div>
      <div className="flex shrink-0 gap-1">
        {isRunning ? (
          <Button
            variant="ghost"
            size="icon"
            onClick={onStop}
            disabled={busy}
            className="h-7 w-7 text-amber-500 hover:text-amber-400"
            title="Stop llama-server"
          >
            <Square className="h-4 w-4" />
          </Button>
        ) : (
          <Button
            variant="ghost"
            size="icon"
            onClick={onStart}
            disabled={busy || !row.enabled || isLoading}
            className="h-7 w-7 text-emerald-500 hover:text-emerald-400"
            title="Spawn llama-server with these workers"
          >
            {isLoading || busy ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Play className="h-4 w-4" />
            )}
          </Button>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={onEdit}
          disabled={isRunning}
          className="h-7 w-7 text-muted-foreground hover:text-foreground"
          title={isRunning ? 'Stop the model first to edit' : 'Edit split model'}
        >
          <Pencil className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={onDelete}
          className="h-7 w-7 text-destructive hover:text-destructive"
          title="Remove split model"
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}
