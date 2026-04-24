import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'
import {
  Plus,
  Trash2,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  ShieldAlert,
  Sparkles,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Switch } from '@/components/ui/switch'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { api } from '@/lib/api'

/**
 * UserToolsSection — embedded body for the "Tools" tab inside SettingsPanel.
 *
 * User-defined tools are Python snippets registered in the `user_tools`
 * SQLite table that become first-class tool-palette entries for every future
 * conversation. Only the user can create, edit, or delete them from this
 * panel — the LLM has no self-extension tool-call route, which is a
 * deliberate safety boundary against a model minting its own privileges.
 *
 * Security posture (mirroring HooksPanel):
 *   - Big amber warning banner at the top — user tools run as the backend
 *     user inside an isolated venv, but still with FS / network access.
 *   - GIGACHAT_DISABLE_USER_TOOLS renders a red kill-switch banner and the
 *     "New tool" button is disabled.
 */
export default function UserToolsSection() {
  const [tools, setTools] = useState([])
  const [disabled, setDisabled] = useState(false)
  const [loading, setLoading] = useState(false)
  const [expanded, setExpanded] = useState({}) // id -> bool
  const [adding, setAdding] = useState(false)
  const [pendingDelete, setPendingDelete] = useState(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const { tools: rows, disabled: off } = await api.listUserTools()
      setTools(rows || [])
      setDisabled(!!off)
    } catch (e) {
      toast.error('Failed to load tools', { description: e.message })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  async function onCreate(draft) {
    try {
      const { tool, install_log: log } = await api.createUserTool(draft)
      toast.success(`Added tool "${tool.name}"`, {
        description: log
          ? 'Dependencies installed — ready to use in new turns.'
          : 'Ready to use in new turns.',
      })
      setAdding(false)
      await refresh()
    } catch (e) {
      toast.error('Could not add tool', { description: e.message })
    }
  }

  async function toggleEnabled(tool) {
    try {
      await api.updateUserTool(tool.id, { enabled: !tool.enabled })
      refresh()
    } catch (e) {
      toast.error('Toggle failed', { description: e.message })
    }
  }

  async function confirmDelete() {
    if (!pendingDelete) return
    try {
      await api.deleteUserTool(pendingDelete.id)
      toast.success(`Deleted "${pendingDelete.name}"`)
      setPendingDelete(null)
      refresh()
    } catch (e) {
      toast.error('Delete failed', { description: e.message })
    }
  }

  // Sort: enabled first, newest within each group.
  const sorted = useMemo(() => {
    const copy = [...tools]
    copy.sort((a, b) => {
      if (a.enabled !== b.enabled) return a.enabled ? -1 : 1
      return (b.created_at || 0) - (a.created_at || 0)
    })
    return copy
  }, [tools])

  return (
    <>
      <div className="flex max-h-[60vh] flex-col overflow-hidden">
        {disabled ? (
          <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            <div className="flex items-start gap-2">
              <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
              <div>
                User tools are disabled on this instance
                (<code>GIGACHAT_DISABLE_USER_TOOLS</code>). Existing tools stay
                visible but no new tools can be created and none will be
                callable from the agent loop.
              </div>
            </div>
          </div>
        ) : (
          <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
            <div className="flex items-start gap-2">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <div>
                User tools run as Python scripts in an isolated venv with
                their own pip dependencies. They still have filesystem and
                network access as the backend user — only add tools you wrote
                or reviewed yourself.
              </div>
            </div>
          </div>
        )}

        <div className="mt-2 flex items-center justify-between pb-2">
          <div className="text-xs text-muted-foreground">
            {loading
              ? 'Loading…'
              : `${sorted.length} tool${sorted.length === 1 ? '' : 's'}`}
          </div>
          <Button
            size="sm"
            onClick={() => setAdding(true)}
            className="gap-2"
            disabled={disabled}
          >
            <Plus className="h-4 w-4" /> New tool
          </Button>
        </div>

        <div className="flex-1 space-y-2 overflow-y-auto pr-1">
          {sorted.length === 0 && !loading && (
            <p className="py-8 text-center text-sm text-muted-foreground">
              No custom tools yet. The agent will mint them automatically when
              a task needs one, or you can add one manually with &ldquo;New
              tool&rdquo;.
            </p>
          )}
          {sorted.map((t) => (
            <ToolRow
              key={t.id}
              tool={t}
              expanded={!!expanded[t.id]}
              onToggleExpand={() =>
                setExpanded((prev) => ({ ...prev, [t.id]: !prev[t.id] }))
              }
              onToggleEnabled={() => toggleEnabled(t)}
              onDelete={() => setPendingDelete(t)}
            />
          ))}
        </div>
      </div>

      {/* Add dialog */}
      <AddToolDialog
        open={adding}
        onOpenChange={setAdding}
        onSubmit={onCreate}
      />

      {/* Delete confirm */}
      <Dialog
        open={!!pendingDelete}
        onOpenChange={(o) => !o && setPendingDelete(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete tool?</DialogTitle>
            <DialogDescription>
              This will permanently remove <code>{pendingDelete?.name}</code>.
              The venv and any installed pip packages are left untouched in
              case another tool needs them. Use the toggle to pause it instead
              if you want to keep the code around.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPendingDelete(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={confirmDelete}>
              <Trash2 className="mr-1 h-4 w-4" /> Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

/* ------------------------------------------------------------------ */
/* One tool row — collapsible body shows code + schema + deps.        */
/* ------------------------------------------------------------------ */
function ToolRow({ tool, expanded, onToggleExpand, onToggleEnabled, onDelete }) {
  const deps = Array.isArray(tool.deps) ? tool.deps : []
  return (
    <div
      className={[
        'rounded-md border border-border bg-card/40',
        tool.enabled ? '' : 'opacity-60',
      ].join(' ')}
    >
      <div className="flex items-start gap-2 p-3">
        <button
          onClick={onToggleExpand}
          className="mt-0.5 text-muted-foreground hover:text-foreground"
          aria-label={expanded ? 'Collapse' : 'Expand'}
        >
          {expanded ? (
            <ChevronDown className="size-4" />
          ) : (
            <ChevronRight className="size-4" />
          )}
        </button>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="rounded bg-accent px-2 py-0.5 font-mono text-[11px]">
              {tool.name}
            </span>
            <span
              className={[
                'rounded px-2 py-0.5 text-[10px] uppercase tracking-wide',
                tool.category === 'write'
                  ? 'bg-amber-500/20 text-amber-700 dark:text-amber-300'
                  : 'bg-muted text-muted-foreground',
              ].join(' ')}
            >
              {tool.category}
            </span>
            <span className="text-muted-foreground">
              timeout {tool.timeout_seconds}s
            </span>
            {deps.length > 0 && (
              <span className="text-muted-foreground">
                {deps.length} dep{deps.length === 1 ? '' : 's'}
              </span>
            )}
          </div>
          <div className="mt-1 line-clamp-2 text-xs text-muted-foreground">
            {tool.description}
          </div>
        </div>
        <Switch
          checked={!!tool.enabled}
          onCheckedChange={onToggleEnabled}
          className="mt-1"
        />
        <Button
          variant="ghost"
          size="icon"
          onClick={onDelete}
          className="h-7 w-7 text-destructive hover:text-destructive"
          title="Delete tool"
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>

      {expanded && (
        <div className="space-y-3 border-t border-border p-3">
          {deps.length > 0 && (
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                Dependencies
              </div>
              <div className="flex flex-wrap gap-1">
                {deps.map((d) => (
                  <code
                    key={d}
                    className="rounded bg-muted px-1.5 py-0.5 text-xs"
                  >
                    {d}
                  </code>
                ))}
              </div>
            </div>
          )}
          <div>
            <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">
              Code
            </div>
            <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-all rounded-md bg-muted p-2 font-mono text-[11px] leading-snug">
              {tool.code}
            </pre>
          </div>
          {tool.schema && Object.keys(tool.schema || {}).length > 0 && (
            <details>
              <summary className="cursor-pointer text-[10px] uppercase tracking-wide text-muted-foreground">
                Input schema
              </summary>
              <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-all rounded-md bg-muted p-2 font-mono text-[11px] leading-snug">
                {JSON.stringify(tool.schema, null, 2)}
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Add-tool dialog — rich form with AST-friendly starter code.        */
/* ------------------------------------------------------------------ */
const STARTER_CODE = `def run(args):
    """Entry point — args is the dict parsed from the tool-call JSON."""
    name = args.get("name") or "world"
    return {"greeting": f"Hello, {name}!"}
`

function AddToolDialog({ open, onOpenChange, onSubmit }) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [code, setCode] = useState(STARTER_CODE)
  const [schemaText, setSchemaText] = useState(
    JSON.stringify(
      {
        type: 'object',
        properties: {
          name: { type: 'string', description: 'Who to greet' },
        },
        required: [],
      },
      null,
      2,
    ),
  )
  const [depsText, setDepsText] = useState('')
  const [category, setCategory] = useState('write')
  const [timeoutSec, setTimeoutSec] = useState(60)
  const [busy, setBusy] = useState(false)

  // Reset every time the dialog opens so a previous half-typed draft doesn't
  // resurface unexpectedly. Users who want drafts can toggle enabled=false.
  useEffect(() => {
    if (open) {
      setName('')
      setDescription('')
      setCode(STARTER_CODE)
      setSchemaText(
        JSON.stringify(
          {
            type: 'object',
            properties: {
              name: { type: 'string', description: 'Who to greet' },
            },
            required: [],
          },
          null,
          2,
        ),
      )
      setDepsText('')
      setCategory('write')
      setTimeoutSec(60)
      setBusy(false)
    }
  }, [open])

  async function submit() {
    const trimmedName = name.trim().toLowerCase()
    if (!/^[a-z][a-z0-9_]{0,47}$/.test(trimmedName)) {
      toast.error('Invalid name', {
        description:
          'Lowercase letters, digits, underscores. Must start with a letter. Max 48 chars.',
      })
      return
    }
    if (!description.trim()) {
      toast.error('Description is required')
      return
    }
    if (!code.trim()) {
      toast.error('Code is required')
      return
    }
    // Best-effort client-side JSON parse so an obvious typo is caught before
    // round-tripping through pip. Server still has its own validation.
    let schemaObj = {}
    if (schemaText.trim()) {
      try {
        schemaObj = JSON.parse(schemaText)
      } catch (e) {
        toast.error('Schema is not valid JSON', { description: e.message })
        return
      }
    }
    const deps = depsText
      .split(/[\n,]/)
      .map((d) => d.trim())
      .filter(Boolean)

    setBusy(true)
    try {
      await onSubmit({
        name: trimmedName,
        description: description.trim(),
        code,
        schema: schemaObj,
        deps,
        category,
        timeout_seconds: Math.max(1, Math.min(600, Number(timeoutSec) || 60)),
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onOpenChange(false)}>
      <DialogContent className="flex max-h-[92vh] flex-col overflow-hidden sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Sparkles className="size-4 text-primary" />
            New user tool
          </DialogTitle>
          <DialogDescription>
            The code must define a <code>def run(args)</code> function. Its
            return value is serialised to JSON and shown to the agent.
          </DialogDescription>
        </DialogHeader>
        <div className="flex-1 space-y-3 overflow-y-auto pr-1">
          <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
            <label className="space-y-1">
              <div className="text-xs text-muted-foreground">
                Name (namespace)
              </div>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="greet_user"
                className="font-mono"
              />
            </label>
            <label className="space-y-1">
              <div className="text-xs text-muted-foreground">Category</div>
              <select
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
              >
                <option value="read">
                  read — safe, no approval in approve_edits mode
                </option>
                <option value="write">
                  write — requires approval in approve_edits mode
                </option>
              </select>
            </label>
          </div>
          <label className="block space-y-1">
            <div className="text-xs text-muted-foreground">
              Description (what it does, when to use it)
            </div>
            <Textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Returns a friendly greeting for a given name."
              rows={2}
            />
          </label>
          <label className="block space-y-1">
            <div className="text-xs text-muted-foreground">Python code</div>
            <Textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              placeholder={STARTER_CODE}
              rows={10}
              className="font-mono text-xs"
            />
          </label>
          <label className="block space-y-1">
            <div className="text-xs text-muted-foreground">
              Input schema (JSON, JSON-Schema subset)
            </div>
            <Textarea
              value={schemaText}
              onChange={(e) => setSchemaText(e.target.value)}
              rows={6}
              className="font-mono text-xs"
            />
          </label>
          <label className="block space-y-1">
            <div className="text-xs text-muted-foreground">
              Pip dependencies (comma or newline separated, e.g.{' '}
              <code>requests&gt;=2.31,beautifulsoup4</code>)
            </div>
            <Textarea
              value={depsText}
              onChange={(e) => setDepsText(e.target.value)}
              rows={2}
              className="font-mono text-xs"
              placeholder="requests>=2.31"
            />
          </label>
          <label className="flex items-center gap-2 text-xs">
            <span className="text-muted-foreground">Timeout (s)</span>
            <Input
              type="number"
              min={1}
              max={600}
              value={timeoutSec}
              onChange={(e) => setTimeoutSec(e.target.value)}
              className="h-8 w-24"
            />
          </label>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={submit} disabled={busy}>
            {busy ? 'Installing…' : 'Add tool'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
