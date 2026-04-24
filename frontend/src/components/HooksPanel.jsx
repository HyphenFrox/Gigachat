import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'
import { Plus, Trash2, AlertTriangle } from 'lucide-react'
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
 * HooksSection — embedded body for the "Hooks" tab inside SettingsPanel.
 *
 * Shows every hook (enabled + disabled), lets the user add a new row (with
 * a clear warning that the command runs with their full shell privileges),
 * toggle enabled, or delete. Changes hit the backend immediately — no
 * explicit save button on the list.
 *
 * The parent owns the outer Dialog. Nested add/edit and delete-confirm
 * dialogs still use shadcn Dialog so they float above the settings drawer.
 */
export default function HooksSection() {
  const [hooks, setHooks] = useState([])
  const [events, setEvents] = useState([])
  const [loading, setLoading] = useState(false)
  const [adding, setAdding] = useState(false)
  const [draft, setDraft] = useState(null) // partial hook while the user types
  const [pendingDelete, setPendingDelete] = useState(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const { hooks: rows, events: evs } = await api.listHooks()
      setHooks(rows || [])
      setEvents(evs || [])
    } catch (e) {
      toast.error('Failed to load hooks', { description: e.message })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  function startAdd() {
    setDraft({
      event: events[0] || 'post_tool',
      matcher: '',
      command: '',
      timeout_seconds: 10,
      enabled: true,
    })
    setAdding(true)
  }

  async function saveDraft() {
    if (!draft) return
    const trimmed = (draft.command || '').trim()
    if (!trimmed) {
      toast.error('Command cannot be empty')
      return
    }
    try {
      await api.createHook({
        event: draft.event,
        command: trimmed,
        matcher: (draft.matcher || '').trim() || null,
        timeout_seconds: Number(draft.timeout_seconds) || 10,
        enabled: !!draft.enabled,
      })
      toast.success('Hook added')
      setAdding(false)
      setDraft(null)
      refresh()
    } catch (e) {
      toast.error('Could not add hook', { description: e.message })
    }
  }

  async function toggleEnabled(hook) {
    try {
      await api.updateHook(hook.id, { enabled: !hook.enabled })
      refresh()
    } catch (e) {
      toast.error('Toggle failed', { description: e.message })
    }
  }

  async function confirmDelete() {
    if (!pendingDelete) return
    try {
      await api.deleteHook(pendingDelete.id)
      toast.success('Hook deleted')
      setPendingDelete(null)
      refresh()
    } catch (e) {
      toast.error('Delete failed', { description: e.message })
    }
  }

  // Sort: enabled first, then by event, then newest-first within each group.
  const sorted = useMemo(() => {
    const copy = [...hooks]
    copy.sort((a, b) => {
      if (a.enabled !== b.enabled) return a.enabled ? -1 : 1
      if (a.event !== b.event) return a.event.localeCompare(b.event)
      return (b.created_at || 0) - (a.created_at || 0)
    })
    return copy
  }, [hooks])

  return (
    <>
      <div className="flex max-h-[60vh] flex-col overflow-hidden">
        <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
          <div className="flex items-start gap-2">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <div>
              Hooks run with your full shell privileges. Only add commands you
              wrote yourself. A hook's stdout is injected back into the
              conversation so the agent can see it.
            </div>
          </div>
        </div>

        <div className="mt-2 flex items-center justify-between pb-2">
          <div className="text-xs text-muted-foreground">
            {loading
              ? 'Loading…'
              : `${sorted.length} hook${sorted.length === 1 ? '' : 's'}`}
          </div>
          <Button size="sm" onClick={startAdd} className="gap-2">
            <Plus className="h-4 w-4" /> New hook
          </Button>
        </div>

        <div className="flex-1 space-y-2 overflow-y-auto pr-1">
          {sorted.length === 0 && !loading && (
            <p className="py-8 text-center text-sm text-muted-foreground">
              No hooks yet. Click &ldquo;New hook&rdquo; to add one.
            </p>
          )}
          {sorted.map((h) => (
            <HookRow
              key={h.id}
              hook={h}
              onToggle={() => toggleEnabled(h)}
              onDelete={() => setPendingDelete(h)}
            />
          ))}
        </div>
      </div>

      {/* Add / edit drawer */}
      <Dialog
        open={adding}
        onOpenChange={(o) => {
          if (!o) {
            setAdding(false)
            setDraft(null)
          }
        }}
      >
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>New hook</DialogTitle>
            <DialogDescription>
              The command runs via <code>sh -c</code> with a JSON payload
              (tool name, args, result, etc.) on stdin.
            </DialogDescription>
          </DialogHeader>
          {draft && (
            <div className="space-y-3">
              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">
                  Event
                </label>
                <select
                  value={draft.event}
                  onChange={(e) => setDraft({ ...draft, event: e.target.value })}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                >
                  {events.map((ev) => (
                    <option key={ev} value={ev}>
                      {ev}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">
                  Tool matcher (optional — pre_tool / post_tool only)
                </label>
                <Input
                  value={draft.matcher}
                  onChange={(e) => setDraft({ ...draft, matcher: e.target.value })}
                  placeholder="e.g. write_file — substring match on tool name"
                />
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">
                  Command
                </label>
                <Textarea
                  value={draft.command}
                  onChange={(e) => setDraft({ ...draft, command: e.target.value })}
                  placeholder='e.g. jq -r .tool_name  # echo which tool was called'
                  className="font-mono text-xs"
                  rows={4}
                />
              </div>
              <div className="flex items-center justify-between">
                <label className="flex items-center gap-2 text-xs">
                  <span className="text-muted-foreground">Timeout (s)</span>
                  <Input
                    type="number"
                    min={1}
                    max={120}
                    value={draft.timeout_seconds}
                    onChange={(e) =>
                      setDraft({ ...draft, timeout_seconds: e.target.value })
                    }
                    className="h-8 w-20"
                  />
                </label>
                <label className="flex items-center gap-2 text-xs">
                  <span className="text-muted-foreground">Enabled</span>
                  <Switch
                    checked={!!draft.enabled}
                    onCheckedChange={(v) =>
                      setDraft({ ...draft, enabled: !!v })
                    }
                  />
                </label>
              </div>
            </div>
          )}
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setAdding(false)
                setDraft(null)
              }}
            >
              Cancel
            </Button>
            <Button onClick={saveDraft}>Add hook</Button>
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
            <DialogTitle>Delete hook?</DialogTitle>
            <DialogDescription>
              This will permanently remove the <code>{pendingDelete?.event}</code>{' '}
              hook. Use the toggle to pause it instead if you want to keep the
              command around.
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

/** One row inside the hook list — label, command snippet, toggle, delete. */
function HookRow({ hook, onToggle, onDelete }) {
  return (
    <div
      className={[
        'flex items-start gap-3 rounded-md border border-border bg-card/40 p-3',
        hook.enabled ? '' : 'opacity-60',
      ].join(' ')}
    >
      <Switch checked={!!hook.enabled} onCheckedChange={onToggle} className="mt-1" />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="rounded bg-accent px-2 py-0.5 font-mono">
            {hook.event}
          </span>
          {hook.matcher && (
            <span className="rounded bg-muted px-2 py-0.5 text-muted-foreground">
              matches: {hook.matcher}
            </span>
          )}
          <span className="text-muted-foreground">
            timeout {hook.timeout_seconds}s
          </span>
        </div>
        <div className="mt-1 overflow-hidden text-ellipsis whitespace-pre-wrap break-all font-mono text-xs">
          {hook.command}
        </div>
      </div>
      <Button
        variant="ghost"
        size="icon"
        onClick={onDelete}
        className="h-7 w-7 text-destructive hover:text-destructive"
        title="Delete hook"
      >
        <Trash2 className="h-4 w-4" />
      </Button>
    </div>
  )
}
