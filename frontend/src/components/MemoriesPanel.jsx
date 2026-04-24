import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'
import { Plus, Trash2, Pencil } from 'lucide-react'
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
import { api } from '@/lib/api'

/**
 * MemoriesSection — embedded body for the "Memories" tab inside SettingsPanel.
 *
 * Lists every "global" memory (notes injected into the system prompt of every
 * conversation, including subagents) and lets the user add / edit / delete
 * them inline. The component renders as a plain section — the parent owns
 * the surrounding Dialog. Nested add/edit and delete-confirm dialogs still
 * use shadcn Dialog so they float above the settings drawer.
 *
 * Mutations hit the API immediately — no explicit save button on the list.
 * The agent can also write here via `remember(scope="global")`.
 */
export default function MemoriesSection() {
  const [memories, setMemories] = useState([])
  const [loading, setLoading] = useState(false)
  const [editing, setEditing] = useState(null) // {id?, content, topic} or null
  const [pendingDelete, setPendingDelete] = useState(null)

  // Re-fetch on mount so the list stays in sync with anything the agent (or
  // another tab) added in the background. Mount happens every time the user
  // switches to the Memories tab, which is the UX we want.
  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const { memories: rows } = await api.listMemories()
      setMemories(rows || [])
    } catch (e) {
      toast.error('Failed to load memories', { description: e.message })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  function startAdd() {
    setEditing({ content: '', topic: '' })
  }

  function startEdit(m) {
    setEditing({ id: m.id, content: m.content, topic: m.topic || '' })
  }

  async function saveEditing() {
    if (!editing) return
    const content = (editing.content || '').trim()
    if (!content) {
      toast.error('Memory cannot be empty')
      return
    }
    // Topic is optional — empty string → null so the backend treats it as
    // "no grouping label" rather than a literal empty heading.
    const topic = (editing.topic || '').trim() || null
    try {
      if (editing.id) {
        await api.updateMemory(editing.id, { content, topic })
        toast.success('Memory updated')
      } else {
        await api.createMemory({ content, topic })
        toast.success('Memory added')
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
      await api.deleteMemory(pendingDelete.id)
      toast.success('Memory deleted')
      setPendingDelete(null)
      refresh()
    } catch (e) {
      toast.error('Delete failed', { description: e.message })
    }
  }

  // Group memories by topic so the UI mirrors the way they're rendered in
  // the system prompt — easier for the user to spot duplicates and clusters.
  const grouped = useMemo(() => {
    const out = new Map()
    for (const m of memories) {
      const key = m.topic || 'General'
      if (!out.has(key)) out.set(key, [])
      out.get(key).push(m)
    }
    return Array.from(out.entries())
  }, [memories])

  return (
    <>
      <div className="flex max-h-[60vh] flex-col overflow-hidden">
        <div className="flex items-center justify-between pb-2">
          <div className="text-xs text-muted-foreground">
            {loading
              ? 'Loading…'
              : `${memories.length} ${memories.length === 1 ? 'memory' : 'memories'}`}
          </div>
          <Button size="sm" onClick={startAdd} className="gap-2">
            <Plus className="h-4 w-4" /> New memory
          </Button>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto pr-1">
          {memories.length === 0 && !loading && (
            <p className="py-8 text-center text-sm text-muted-foreground">
              No global memories yet. Click &ldquo;New memory&rdquo; to add
              one — for example, your name, your role, or a tool preference
              every chat should know.
            </p>
          )}
          {grouped.map(([topic, items]) => (
            <section key={topic}>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                {topic}
              </h3>
              <div className="space-y-2">
                {items.map((m) => (
                  <MemoryRow
                    key={m.id}
                    memory={m}
                    onEdit={() => startEdit(m)}
                    onDelete={() => setPendingDelete(m)}
                  />
                ))}
              </div>
            </section>
          ))}
        </div>
      </div>

      {/* Add / edit drawer — same form, distinguished by editing.id. */}
      <Dialog
        open={!!editing}
        onOpenChange={(o) => {
          if (!o) setEditing(null)
        }}
      >
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>
              {editing?.id ? 'Edit memory' : 'New memory'}
            </DialogTitle>
            <DialogDescription>
              Keep it short and factual. Topic is optional — entries with the
              same topic are grouped under one heading.
            </DialogDescription>
          </DialogHeader>
          {editing && (
            <div className="space-y-3">
              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">
                  Topic (optional)
                </label>
                <Input
                  value={editing.topic}
                  onChange={(e) =>
                    setEditing({ ...editing, topic: e.target.value })
                  }
                  placeholder="e.g. user preferences, environment, identity"
                />
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">
                  Content
                </label>
                <Textarea
                  value={editing.content}
                  onChange={(e) =>
                    setEditing({ ...editing, content: e.target.value })
                  }
                  placeholder='e.g. "Prefers SCSS over plain CSS" or "Lives in UTC+5:30"'
                  rows={4}
                  autoFocus
                />
              </div>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditing(null)}>
              Cancel
            </Button>
            <Button onClick={saveEditing}>
              {editing?.id ? 'Save changes' : 'Add memory'}
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
            <DialogTitle>Delete memory?</DialogTitle>
            <DialogDescription>
              This entry will be removed from every future conversation's
              system prompt. This cannot be undone.
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

/** One memory row — content + edit + delete. */
function MemoryRow({ memory, onEdit, onDelete }) {
  return (
    <div className="flex items-start gap-3 rounded-md border border-border bg-card/40 p-3">
      <div className="min-w-0 flex-1">
        <div className="whitespace-pre-wrap break-words text-sm">
          {memory.content}
        </div>
      </div>
      <div className="flex shrink-0 gap-1">
        <Button
          variant="ghost"
          size="icon"
          onClick={onEdit}
          className="h-7 w-7 text-muted-foreground hover:text-foreground"
          title="Edit memory"
        >
          <Pencil className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={onDelete}
          className="h-7 w-7 text-destructive hover:text-destructive"
          title="Delete memory"
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}
