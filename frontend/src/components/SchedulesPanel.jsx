import React, { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Plus, Trash2, CalendarClock, Repeat, RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
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
 * SchedulesSection — embedded body for the "Schedules" tab inside SettingsPanel.
 *
 * Scheduled tasks fire a fresh agent turn at a specific time (one-shot) or on
 * a recurring interval. Each run lands in its own conversation prefixed with
 * "Scheduled:" so the user can open it later and see what happened.
 *
 * This panel is a CRUD view over the same `scheduled_tasks` table the agent's
 * `schedule_task` tool writes into — both producers share storage, so a job
 * the agent scheduled shows up here too.
 */
export default function SchedulesSection() {
  const [tasks, setTasks] = useState([])
  const [loading, setLoading] = useState(false)
  const [draft, setDraft] = useState(null) // null = form closed; object = editing
  const [pendingDelete, setPendingDelete] = useState(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const { tasks: rows } = await api.listScheduledTasks()
      setTasks(Array.isArray(rows) ? rows : [])
    } catch (e) {
      toast.error('Failed to load schedules', { description: e.message })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  function startAdd() {
    // Default the picker to "five minutes from now" — far enough out that the
    // user has time to tweak the form, close enough that "Save" feels like it
    // does something immediately.
    const soon = new Date(Date.now() + 5 * 60 * 1000)
    setDraft({
      name: '',
      prompt: '',
      run_at: toDatetimeLocal(soon),
      interval_seconds: '',
      cwd: '',
    })
  }

  async function save() {
    if (!draft) return
    const name = (draft.name || '').trim()
    const prompt = (draft.prompt || '').trim()
    if (!name || !prompt) {
      toast.error('Name and prompt are required')
      return
    }
    const interval = draft.interval_seconds
      ? Number(draft.interval_seconds)
      : null
    if (interval !== null && (!Number.isFinite(interval) || interval < 60)) {
      toast.error('Interval must be at least 60 seconds')
      return
    }
    try {
      await api.createScheduledTask({
        name,
        prompt,
        run_at: draft.run_at, // datetime-local string; backend parses it
        interval_seconds: interval,
        cwd: (draft.cwd || '').trim() || null,
      })
      toast.success('Scheduled', {
        description: `${name} will fire at ${draft.run_at.replace('T', ' ')}.`,
      })
      setDraft(null)
      refresh()
    } catch (e) {
      toast.error('Could not schedule', { description: e.message })
    }
  }

  async function confirmDelete() {
    if (!pendingDelete) return
    try {
      await api.deleteScheduledTask(pendingDelete.id)
      toast.success('Schedule cancelled')
      setPendingDelete(null)
      refresh()
    } catch (e) {
      toast.error('Delete failed', { description: e.message })
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          Fire an agent prompt at a scheduled time, one-shot or recurring.
          Each run opens its own conversation with{' '}
          <span className="font-medium text-foreground">Auto-approve</span> on.
        </p>
        <div className="flex gap-1">
          <Button variant="ghost" size="icon" onClick={refresh} title="Refresh">
            <RefreshCw className={cn('size-4', loading && 'animate-spin')} />
          </Button>
          <Button size="sm" onClick={startAdd} className="gap-1">
            <Plus className="size-4" />
            New
          </Button>
        </div>
      </div>

      {tasks.length === 0 && !loading && (
        <div className="rounded-md border border-dashed border-border bg-muted/20 p-6 text-center text-xs text-muted-foreground">
          No scheduled tasks yet. Click <span className="font-medium">New</span> to add one, or ask the agent to
          schedule a prompt for you.
        </div>
      )}

      <ul className="space-y-2">
        {tasks.map((t) => (
          <li
            key={t.id}
            className="rounded-md border border-border bg-card p-3 text-xs"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium text-foreground">
                  {t.name}
                </div>
                <div className="mt-0.5 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <CalendarClock className="size-3" />
                    {formatWhen(t.next_run_at)}
                  </span>
                  {t.interval_seconds ? (
                    <span className="flex items-center gap-1">
                      <Repeat className="size-3" />
                      every {formatInterval(t.interval_seconds)}
                    </span>
                  ) : (
                    <span className="text-muted-foreground/70">one-shot</span>
                  )}
                  <span className="truncate font-mono text-[10px]" title={t.cwd}>
                    {t.cwd}
                  </span>
                </div>
                <div className="mt-1 line-clamp-2 text-muted-foreground">
                  {t.prompt}
                </div>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setPendingDelete(t)}
                title="Cancel scheduled task"
                aria-label="Cancel scheduled task"
              >
                <Trash2 className="size-4 text-destructive" />
              </Button>
            </div>
          </li>
        ))}
      </ul>

      {/* Create dialog. Edit isn't supported by the backend (no PATCH route);
          users delete and recreate, which keeps the daemon logic dead-simple. */}
      <Dialog open={!!draft} onOpenChange={(o) => !o && setDraft(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>New scheduled task</DialogTitle>
            <DialogDescription>
              The prompt will run as a fresh conversation at the chosen time.
            </DialogDescription>
          </DialogHeader>
          {draft && (
            <div className="space-y-3">
              <div>
                <label className="mb-1 block text-xs font-medium">Name</label>
                <Input
                  autoFocus
                  value={draft.name}
                  onChange={(e) =>
                    setDraft((d) => ({ ...d, name: e.target.value }))
                  }
                  placeholder="Daily standup summary"
                />
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium">Prompt</label>
                <Textarea
                  value={draft.prompt}
                  onChange={(e) =>
                    setDraft((d) => ({ ...d, prompt: e.target.value }))
                  }
                  placeholder="Summarise yesterday's git log and post the result to standup-notes.md"
                  rows={4}
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="mb-1 block text-xs font-medium">Run at</label>
                  <Input
                    type="datetime-local"
                    value={draft.run_at}
                    onChange={(e) =>
                      setDraft((d) => ({ ...d, run_at: e.target.value }))
                    }
                  />
                </div>
                <div>
                  <label className="mb-1 block text-xs font-medium">
                    Repeat every (seconds, optional)
                  </label>
                  <Input
                    type="number"
                    min={60}
                    value={draft.interval_seconds}
                    onChange={(e) =>
                      setDraft((d) => ({
                        ...d,
                        interval_seconds: e.target.value,
                      }))
                    }
                    placeholder="3600 = hourly"
                  />
                </div>
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium">
                  Working directory (optional)
                </label>
                <Input
                  value={draft.cwd}
                  onChange={(e) =>
                    setDraft((d) => ({ ...d, cwd: e.target.value }))
                  }
                  placeholder="Leave blank to use the app root"
                  className="font-mono text-xs"
                />
              </div>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setDraft(null)}>
              Cancel
            </Button>
            <Button onClick={save}>Schedule</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={!!pendingDelete}
        onOpenChange={(o) => !o && setPendingDelete(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Cancel scheduled task?</DialogTitle>
            <DialogDescription>
              “{pendingDelete?.name}” will not run. This doesn't affect any
              runs that have already fired.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPendingDelete(null)}>
              Keep
            </Button>
            <Button variant="destructive" onClick={confirmDelete}>
              Cancel task
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

// ---------- helpers ---------------------------------------------------------

/** Turn a Date into the "YYYY-MM-DDTHH:mm" string <input type="datetime-local"> expects. */
function toDatetimeLocal(d) {
  const pad = (n) => String(n).padStart(2, '0')
  return (
    d.getFullYear() +
    '-' +
    pad(d.getMonth() + 1) +
    '-' +
    pad(d.getDate()) +
    'T' +
    pad(d.getHours()) +
    ':' +
    pad(d.getMinutes())
  )
}

/** "in 2h" / "tomorrow 14:30" / "12 Apr 09:00" — low-fidelity but readable. */
function formatWhen(unixSec) {
  if (!unixSec) return 'unscheduled'
  const ms = unixSec * 1000
  const d = new Date(ms)
  const now = Date.now()
  const diff = ms - now
  const abs = Math.abs(diff)
  if (abs < 60_000) return diff > 0 ? 'in <1 min' : 'overdue'
  if (abs < 60 * 60_000)
    return diff > 0
      ? `in ${Math.round(abs / 60_000)} min`
      : `${Math.round(abs / 60_000)} min ago`
  if (abs < 24 * 60 * 60_000)
    return diff > 0
      ? `in ${Math.round(abs / 3_600_000)} h`
      : `${Math.round(abs / 3_600_000)} h ago`
  return d.toLocaleString()
}

/** "1h" / "30 min" / "2d" — trims trailing zero units for readability. */
function formatInterval(sec) {
  if (sec < 60) return `${sec}s`
  if (sec < 3600) return `${Math.round(sec / 60)} min`
  if (sec < 86400) return `${(sec / 3600).toFixed(sec % 3600 === 0 ? 0 : 1)} h`
  return `${(sec / 86400).toFixed(sec % 86400 === 0 ? 0 : 1)} d`
}
