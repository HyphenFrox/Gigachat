import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'
import {
  Plus,
  RefreshCw,
  Trash2,
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  XCircle,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Switch } from '@/components/ui/switch'
import { api } from '@/lib/api'

/**
 * MCPSection — embedded body for the "MCP" tab inside SettingsPanel.
 *
 * Lets the user CRUD external Model Context Protocol servers. Each row is
 * an executable + args + env the backend spawns over stdio; tools the
 * server advertises merge into the agent's tool palette automatically
 * (no prompt changes, no restart).
 *
 * Form-heavy by design:
 *   - MCP launch config maps 1:1 to argv/env so the user needs direct
 *     access to command, args, and env without inventing new abstractions.
 *   - Secrets usually live in env — rendering that as rows (not a free-text
 *     blob) lets us hide values by default and still offer copy/delete.
 *
 * The parent owns the outer Dialog; this section fills a tab body.
 */
export default function MCPSection() {
  const [servers, setServers] = useState([])
  const [loading, setLoading] = useState(false)
  const [expanded, setExpanded] = useState({}) // id -> bool
  const [adding, setAdding] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const { servers } = await api.listMcpServers()
      setServers(servers || [])
    } catch (e) {
      toast.error('Failed to load MCP servers', { description: e.message })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  async function forceRefresh() {
    try {
      await api.refreshMcpServers()
      toast.success('MCP servers refreshed')
      await refresh()
    } catch (e) {
      toast.error('Refresh failed', { description: e.message })
    }
  }

  async function onCreate(draft) {
    try {
      const { server, report } = await api.createMcpServer(draft)
      const r = report?.[server.name]
      if (r?.error) {
        toast.warning(`Added "${server.name}", but startup failed`, {
          description: r.error,
        })
      } else {
        toast.success(
          `Added "${server.name}" — ${r?.tools?.length || 0} tool(s) available`,
        )
      }
      setAdding(false)
      await refresh()
    } catch (e) {
      toast.error('Add failed', { description: e.message })
    }
  }

  async function onToggleEnabled(srv) {
    try {
      await api.updateMcpServer(srv.id, { enabled: !srv.enabled })
      await refresh()
    } catch (e) {
      toast.error('Update failed', { description: e.message })
    }
  }

  async function onDelete(srv) {
    if (!window.confirm(`Remove MCP server "${srv.name}"?`)) return
    try {
      await api.deleteMcpServer(srv.id)
      toast.success(`Removed "${srv.name}"`)
      await refresh()
    } catch (e) {
      toast.error('Delete failed', { description: e.message })
    }
  }

  async function onEditSave(srv, patch) {
    try {
      const { report } = await api.updateMcpServer(srv.id, patch)
      const r = report?.[patch.name || srv.name]
      if (r?.error) {
        toast.warning('Saved, but server did not start', {
          description: r.error,
        })
      } else {
        toast.success('Saved')
      }
      await refresh()
    } catch (e) {
      toast.error('Save failed', { description: e.message })
    }
  }

  return (
    <div className="flex flex-col">
      <div className="flex items-center justify-between gap-2 pb-2">
        <Button
          size="sm"
          variant="outline"
          onClick={forceRefresh}
          disabled={loading}
        >
          <RefreshCw className={`size-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
        <Button size="sm" onClick={() => setAdding((v) => !v)}>
          <Plus className="size-4" />
          {adding ? 'Cancel' : 'Add server'}
        </Button>
      </div>

      {adding && <AddServerForm onCancel={() => setAdding(false)} onSave={onCreate} />}

      <div className="max-h-[50vh] space-y-2 overflow-y-auto pr-1">
        {servers.length === 0 && !adding ? (
          <div className="rounded-md border border-dashed border-border bg-muted/30 p-4 text-sm text-muted-foreground">
            <p className="mb-1 font-medium text-foreground">
              No servers configured
            </p>
            <p>
              MCP servers run as local subprocesses (stdio transport). Try
              e.g. <code className="rounded bg-muted px-1">npx</code> with args{' '}
              <code className="rounded bg-muted px-1">-y @modelcontextprotocol/server-filesystem /path</code>.
            </p>
          </div>
        ) : (
          servers.map((srv) => (
            <ServerRow
              key={srv.id}
              server={srv}
              expanded={!!expanded[srv.id]}
              onToggle={() =>
                setExpanded((prev) => ({ ...prev, [srv.id]: !prev[srv.id] }))
              }
              onToggleEnabled={() => onToggleEnabled(srv)}
              onDelete={() => onDelete(srv)}
              onSave={(patch) => onEditSave(srv, patch)}
            />
          ))
        )}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------- */
/* One server row — collapsible edit form beneath a status summary.    */
/* ------------------------------------------------------------------- */
function ServerRow({ server, expanded, onToggle, onToggleEnabled, onDelete, onSave }) {
  const status = server.status || { running: false, tools: [] }
  const running = !!status.running
  const toolCount = status.tools?.length || 0
  const statusIcon = running ? (
    <CheckCircle2 className="size-4 text-emerald-500" />
  ) : (
    <XCircle className="size-4 text-muted-foreground" />
  )

  return (
    <div className="rounded-md border border-border bg-card">
      <div className="flex items-center gap-2 p-3">
        <button
          onClick={onToggle}
          className="text-muted-foreground hover:text-foreground"
          aria-label={expanded ? 'Collapse' : 'Expand'}
        >
          {expanded ? (
            <ChevronDown className="size-4" />
          ) : (
            <ChevronRight className="size-4" />
          )}
        </button>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="font-medium">{server.name}</span>
            {statusIcon}
            <span className="text-xs text-muted-foreground">
              {running
                ? `${toolCount} tool${toolCount === 1 ? '' : 's'}`
                : server.enabled
                  ? 'not running'
                  : 'disabled'}
            </span>
          </div>
          <div className="truncate text-xs text-muted-foreground">
            {server.command}{' '}
            {(server.args || []).join(' ')}
          </div>
        </div>
        <Switch checked={server.enabled} onCheckedChange={onToggleEnabled} />
        <Button variant="ghost" size="icon" onClick={onDelete} title="Delete">
          <Trash2 className="size-4" />
        </Button>
      </div>

      {expanded && (
        <div className="border-t border-border p-3">
          <ServerForm initial={server} onSave={onSave} />
          {status.tools?.length > 0 && (
            <div className="mt-3">
              <div className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">
                Advertised tools
              </div>
              <div className="flex flex-wrap gap-1">
                {status.tools.map((t) => (
                  <span
                    key={t}
                    className="rounded-md bg-muted px-2 py-0.5 text-xs"
                  >
                    {t}
                  </span>
                ))}
              </div>
            </div>
          )}
          {status.error && (
            <div className="mt-3 rounded-md border border-destructive/40 bg-destructive/10 p-2 text-xs text-destructive">
              {status.error}
            </div>
          )}
          {status.stderr_tail && (
            <details className="mt-3">
              <summary className="cursor-pointer text-xs text-muted-foreground">
                Server stderr (last 2 KB)
              </summary>
              <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded-md bg-muted p-2 text-xs">
                {status.stderr_tail}
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------- */
/* Add-server inline form.                                             */
/* ------------------------------------------------------------------- */
function AddServerForm({ onCancel, onSave }) {
  return (
    <div className="mb-2 rounded-md border border-border bg-card p-3">
      <div className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">
        New server
      </div>
      <ServerForm initial={null} onSave={onSave} submitLabel="Add & start" onCancel={onCancel} />
    </div>
  )
}

/* ------------------------------------------------------------------- */
/* Shared edit form, used for both add and edit flows.                 */
/* Controlled locally — the parent only sees the final payload on save.*/
/* ------------------------------------------------------------------- */
function ServerForm({ initial, onSave, submitLabel = 'Save', onCancel }) {
  const [name, setName] = useState(initial?.name || '')
  const [command, setCommand] = useState(initial?.command || '')
  const [argsText, setArgsText] = useState((initial?.args || []).join('\n'))
  const [envPairs, setEnvPairs] = useState(() =>
    Object.entries(initial?.env || {}).map(([k, v]) => ({ k, v })),
  )
  const [busy, setBusy] = useState(false)

  const parsedArgs = useMemo(
    () => argsText.split('\n').map((s) => s.trim()).filter(Boolean),
    [argsText],
  )
  const parsedEnv = useMemo(() => {
    const out = {}
    for (const p of envPairs) {
      const k = (p.k || '').trim()
      if (k) out[k] = p.v || ''
    }
    return out
  }, [envPairs])

  async function onSubmit() {
    if (!name.trim() || !command.trim()) {
      toast.error('Name and command are required')
      return
    }
    setBusy(true)
    try {
      await onSave({
        name: name.trim(),
        command: command.trim(),
        args: parsedArgs,
        env: parsedEnv,
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-3 text-sm">
      <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
        <label className="space-y-1">
          <div className="text-xs text-muted-foreground">Name (namespace)</div>
          <Input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="filesystem"
          />
        </label>
        <label className="space-y-1">
          <div className="text-xs text-muted-foreground">Command</div>
          <Input
            value={command}
            onChange={(e) => setCommand(e.target.value)}
            placeholder="npx"
          />
        </label>
      </div>
      <label className="space-y-1 block">
        <div className="text-xs text-muted-foreground">
          Arguments (one per line)
        </div>
        <Textarea
          value={argsText}
          onChange={(e) => setArgsText(e.target.value)}
          placeholder={`-y\n@modelcontextprotocol/server-filesystem\n/Users/you/Documents`}
          rows={4}
          className="font-mono text-xs"
        />
      </label>
      <div>
        <div className="mb-1 flex items-center justify-between">
          <div className="text-xs text-muted-foreground">
            Environment variables
          </div>
          <Button
            size="sm"
            variant="ghost"
            onClick={() =>
              setEnvPairs((prev) => [...prev, { k: '', v: '' }])
            }
          >
            <Plus className="size-3" />
            Add
          </Button>
        </div>
        {envPairs.length === 0 ? (
          <div className="rounded-md border border-dashed border-border p-2 text-xs text-muted-foreground">
            None — click Add to attach an API key or config flag.
          </div>
        ) : (
          <div className="space-y-1">
            {envPairs.map((p, i) => (
              <div key={i} className="flex items-center gap-2">
                <Input
                  value={p.k}
                  onChange={(e) =>
                    setEnvPairs((prev) => {
                      const next = [...prev]
                      next[i] = { ...next[i], k: e.target.value }
                      return next
                    })
                  }
                  placeholder="KEY"
                  className="max-w-[40%] font-mono text-xs"
                />
                <Input
                  type="password"
                  value={p.v}
                  onChange={(e) =>
                    setEnvPairs((prev) => {
                      const next = [...prev]
                      next[i] = { ...next[i], v: e.target.value }
                      return next
                    })
                  }
                  placeholder="value"
                  className="flex-1 font-mono text-xs"
                />
                <Button
                  size="icon"
                  variant="ghost"
                  onClick={() =>
                    setEnvPairs((prev) => prev.filter((_, j) => j !== i))
                  }
                  title="Remove"
                >
                  <Trash2 className="size-4" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="flex items-center justify-end gap-2 pt-1">
        {onCancel && (
          <Button variant="outline" size="sm" onClick={onCancel}>
            Cancel
          </Button>
        )}
        <Button size="sm" onClick={onSubmit} disabled={busy}>
          {submitLabel}
        </Button>
      </div>
    </div>
  )
}
