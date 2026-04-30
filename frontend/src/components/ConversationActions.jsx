import React, { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'
import {
  MoreHorizontal,
  Sparkles,
  Gauge,
  Pin,
  BookText,
  Trash2,
  ExternalLink,
  Loader2,
} from 'lucide-react'
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
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { cn, formatMessageTime, formatFullTimestamp } from '@/lib/utils'
import { api } from '@/lib/api'

/**
 * ConversationActions — "…" menu in the chat header with four panels for
 * per-conversation preferences and house-keeping:
 *
 *   1. Persona           — free-text system-prompt extension (adds tone /
 *                          behaviour to this conversation only).
 *   2. Budget             — optional soft caps on assistant turns / tokens.
 *   3. Pinned messages    — view, jump-to, unpin, delete sticky rows.
 *   4. Conversation memory — view / edit / clear the per-conv markdown file
 *                          the `remember` tool writes to.
 *
 * Each panel is its own Dialog so state is scoped and a failure in one can't
 * clobber the others. The dropdown is the only new element in the header; all
 * heavy UI lives behind clicks so the header bar doesn't get crowded.
 *
 * Props:
 *   - conv: the current conversation object (id, persona, budget_*, …)
 *   - onUpdate: (conversation) => void; called after any PATCH so the parent
 *               can update its cached conv.
 *   - onScrollToMessage: optional (messageId) => void; if provided, the
 *               pinned-messages dialog renders a "jump to" affordance.
 */
export default function ConversationActions({ conv, onUpdate, onScrollToMessage }) {
  const [open, setOpen] = useState(null) // 'persona' | 'budget' | 'pinned' | 'memory' | null

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="size-8"
            aria-label="More conversation actions"
            title="More actions"
          >
            <MoreHorizontal className="size-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-52">
          <DropdownMenuLabel className="text-xs">
            This conversation
          </DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuItem onClick={() => setOpen('persona')}>
            <Sparkles className="mr-2 size-4" />
            Persona
            {conv.persona ? (
              <span className="ml-auto text-[10px] text-primary">set</span>
            ) : null}
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => setOpen('budget')}>
            <Gauge className="mr-2 size-4" />
            Budget
            {conv.budget_turns || conv.budget_tokens ? (
              <span className="ml-auto text-[10px] text-primary">set</span>
            ) : null}
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => setOpen('pinned')}>
            <Pin className="mr-2 size-4" />
            Pinned messages
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => setOpen('memory')}>
            <BookText className="mr-2 size-4" />
            Conversation memory
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <PersonaDialog
        open={open === 'persona'}
        onClose={() => setOpen(null)}
        conv={conv}
        onUpdate={onUpdate}
      />
      <BudgetDialog
        open={open === 'budget'}
        onClose={() => setOpen(null)}
        conv={conv}
        onUpdate={onUpdate}
      />
      <PinnedMessagesDialog
        open={open === 'pinned'}
        onClose={() => setOpen(null)}
        conv={conv}
        onScrollToMessage={onScrollToMessage}
      />
      <MemoryDialog
        open={open === 'memory'}
        onClose={() => setOpen(null)}
        conv={conv}
      />
    </>
  )
}

/* ----------------------------- Persona ------------------------------------ */
const PERSONA_MAX = 4000

/**
 * PersonaDialog — edit the free-text persona. Empty / whitespace-only saves
 * clear the override. Hard-cap matches the backend's 4 000-char truncation.
 */
function PersonaDialog({ open, onClose, conv, onUpdate }) {
  const [draft, setDraft] = useState(conv.persona || '')
  const [saving, setSaving] = useState(false)

  // Re-seed the draft every time the dialog reopens so a stale edit from a
  // cancel doesn't bleed into the next open.
  useEffect(() => {
    if (open) setDraft(conv.persona || '')
  }, [open, conv.persona])

  async function save() {
    setSaving(true)
    try {
      // Backend treats whitespace-only as "clear the override" so we don't
      // need a separate "Clear" button when the textarea is empty — just save.
      const { conversation } = await api.updateConversation(conv.id, {
        persona: draft,
      })
      onUpdate?.(conversation)
      toast.success(
        draft.trim() ? 'Persona saved' : 'Persona cleared',
        {
          description: draft.trim()
            ? 'Takes effect on the next turn.'
            : 'This conversation will use the default persona.',
        },
      )
      onClose?.()
    } catch (e) {
      toast.error('Save failed', { description: e.message })
    } finally {
      setSaving(false)
    }
  }

  const overLimit = draft.length > PERSONA_MAX
  const dirty = (draft || '') !== (conv.persona || '')

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose?.()}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Sparkles className="size-4" />
            Persona for this conversation
          </DialogTitle>
          <DialogDescription>
            Free-form system-prompt extension. Use it to nudge tone, style, or
            scope — e.g. "Answer like a senior Rust reviewer" or "Always reply
            in French". Safety rules still apply.
          </DialogDescription>
        </DialogHeader>
        <Textarea
          autoFocus
          rows={8}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="e.g. You are a blunt code reviewer. Skip pleasantries; point out design smells directly."
          className="resize-y text-sm"
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>Leave empty to clear.</span>
          <span className={cn(overLimit && 'text-destructive')}>
            {draft.length.toLocaleString()} / {PERSONA_MAX.toLocaleString()}
          </span>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={save} disabled={!dirty || saving || overLimit}>
            {saving ? 'Saving…' : 'Save'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

/* ----------------------------- Budget ------------------------------------- */
/**
 * BudgetDialog — set soft per-conversation caps on assistant turns and/or
 * tokens. 0 / blank clears the cap. Live usage is fetched from
 * /api/conversations/:id/usage so the user sees the current consumption.
 */
function BudgetDialog({ open, onClose, conv, onUpdate }) {
  const [turnsDraft, setTurnsDraft] = useState(conv.budget_turns ?? '')
  const [tokensDraft, setTokensDraft] = useState(conv.budget_tokens ?? '')
  const [usage, setUsage] = useState(null)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const u = await api.getConversationUsage(conv.id)
      setUsage(u)
    } catch (e) {
      toast.error('Could not load usage', { description: e.message })
    } finally {
      setLoading(false)
    }
  }, [conv.id])

  useEffect(() => {
    if (!open) return
    setTurnsDraft(conv.budget_turns ?? '')
    setTokensDraft(conv.budget_tokens ?? '')
    refresh()
  }, [open, conv.budget_turns, conv.budget_tokens, refresh])

  async function save() {
    const parse = (raw) => {
      if (raw === '' || raw === null || raw === undefined) return 0
      const n = parseInt(raw, 10)
      return Number.isFinite(n) && n > 0 ? n : 0
    }
    setSaving(true)
    try {
      const { conversation } = await api.updateConversation(conv.id, {
        budget_turns: parse(turnsDraft),
        budget_tokens: parse(tokensDraft),
      })
      onUpdate?.(conversation)
      toast.success('Budget saved')
      onClose?.()
    } catch (e) {
      toast.error('Save failed', { description: e.message })
    } finally {
      setSaving(false)
    }
  }

  const dirty =
    String(turnsDraft ?? '') !== String(conv.budget_turns ?? '') ||
    String(tokensDraft ?? '') !== String(conv.budget_tokens ?? '')

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose?.()}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Gauge className="size-4" />
            Budget for this conversation
          </DialogTitle>
          <DialogDescription>
            Soft caps — the agent refuses to start a new turn once either limit
            is hit. Useful for scheduled tasks that shouldn't loop forever.
            Leave both blank (or 0) for unlimited.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-3">
          <BudgetField
            label="Max assistant turns"
            hint="Counts completed assistant replies. Tool calls inside one reply don't multiply it."
            value={turnsDraft}
            onChange={setTurnsDraft}
            usageLabel={
              usage
                ? `${usage.assistant_turns.toLocaleString()} used`
                : loading
                ? 'loading…'
                : ''
            }
          />
          <BudgetField
            label="Max tokens (estimate)"
            hint="Rough char-count proxy — matches the header gauge. One token ≈ 4 chars."
            value={tokensDraft}
            onChange={setTokensDraft}
            usageLabel={
              usage
                ? `~${usage.tokens_estimate.toLocaleString()} used`
                : loading
                ? 'loading…'
                : ''
            }
          />
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={save} disabled={!dirty || saving}>
            {saving ? 'Saving…' : 'Save'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function BudgetField({ label, hint, value, onChange, usageLabel }) {
  return (
    <div className="space-y-1">
      <label className="text-sm font-medium text-foreground">{label}</label>
      <div className="flex items-center gap-2">
        <Input
          type="number"
          min="0"
          inputMode="numeric"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="unlimited"
          className="font-mono text-sm"
        />
        {usageLabel ? (
          <span className="whitespace-nowrap text-xs text-muted-foreground">
            {usageLabel}
          </span>
        ) : null}
      </div>
      <p className="text-xs text-muted-foreground">{hint}</p>
    </div>
  )
}

/* ------------------------- Pinned messages -------------------------------- */
/**
 * PinnedMessagesDialog — list every pinned message in the current
 * conversation, with inline unpin + delete actions and an optional
 * "jump to" hop that scrolls the transcript to the row.
 *
 * We re-fetch from /pinned on open rather than filtering the parent's cached
 * message list so the dialog reflects server state even after an auto-compact
 * dropped some rows we thought were pinned but actually weren't.
 */
function PinnedMessagesDialog({ open, onClose, conv, onScrollToMessage }) {
  const [rows, setRows] = useState([])
  const [loading, setLoading] = useState(false)
  const [pendingDelete, setPendingDelete] = useState(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.listPinnedMessages(conv.id)
      setRows(Array.isArray(res.messages) ? res.messages : [])
    } catch (e) {
      toast.error('Could not load pinned messages', { description: e.message })
    } finally {
      setLoading(false)
    }
  }, [conv.id])

  useEffect(() => {
    if (open) refresh()
  }, [open, refresh])

  async function unpin(mid) {
    try {
      await api.pinMessage(conv.id, mid, false)
      setRows((prev) => prev.filter((r) => r.id !== mid))
      toast.success('Unpinned')
    } catch (e) {
      toast.error('Unpin failed', { description: e.message })
    }
  }

  async function remove(mid) {
    setPendingDelete(null)
    try {
      await api.deleteMessage(conv.id, mid)
      setRows((prev) => prev.filter((r) => r.id !== mid))
      toast.success('Message deleted')
    } catch (e) {
      toast.error('Delete failed', { description: e.message })
    }
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose?.()}>
      <DialogContent className="flex max-h-[80vh] flex-col sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Pin className="size-4" />
            Pinned messages
          </DialogTitle>
          <DialogDescription>
            Pinned rows survive auto-compaction — the model still sees them 50
            turns later. Jump to the row in context, unpin to let it age out
            normally, or delete it entirely.
          </DialogDescription>
        </DialogHeader>

        <div className="min-h-0 flex-1 overflow-y-auto">
          {loading ? (
            <div className="flex items-center justify-center py-8 text-muted-foreground">
              <Loader2 className="mr-2 size-4 animate-spin" />
              Loading…
            </div>
          ) : rows.length === 0 ? (
            <p className="py-8 text-center text-sm text-muted-foreground">
              Nothing pinned yet. Hover any message in the transcript and click
              the pin icon to add it.
            </p>
          ) : (
            <ul className="space-y-2">
              {rows.map((m) => (
                <li
                  key={m.id}
                  className="rounded-md border border-border bg-card/40 p-2 text-xs"
                >
                  <div className="mb-1 flex items-center gap-2 text-[10px] uppercase tracking-wide text-muted-foreground">
                    <span className="font-medium text-foreground">{m.role}</span>
                    <span>·</span>
                    <span title={formatFullTimestamp(m.created_at)}>
                      {formatMessageTime(m.created_at)}
                    </span>
                    <div className="ml-auto flex items-center gap-1">
                      {onScrollToMessage ? (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="size-6"
                          title="Jump to message"
                          onClick={() => {
                            onClose?.()
                            onScrollToMessage(m.id)
                          }}
                        >
                          <ExternalLink className="size-3.5" />
                        </Button>
                      ) : null}
                      <Button
                        variant="ghost"
                        size="icon"
                        className="size-6"
                        title="Unpin"
                        onClick={() => unpin(m.id)}
                      >
                        <Pin className="size-3.5" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="size-6 text-destructive hover:text-destructive"
                        title="Delete message"
                        onClick={() => setPendingDelete(m)}
                      >
                        <Trash2 className="size-3.5" />
                      </Button>
                    </div>
                  </div>
                  <p className="line-clamp-4 whitespace-pre-wrap break-words text-foreground/90">
                    {m.content || <span className="italic text-muted-foreground">(empty)</span>}
                  </p>
                </li>
              ))}
            </ul>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>

      {/* Confirm delete — nested dialog so the pinned list stays behind it. */}
      <Dialog
        open={!!pendingDelete}
        onOpenChange={(o) => !o && setPendingDelete(null)}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Delete this message?</DialogTitle>
            <DialogDescription>
              Permanently removes the row from this conversation's history.
              The model will no longer see it on future turns. This can't be
              undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPendingDelete(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => pendingDelete && remove(pendingDelete.id)}
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Dialog>
  )
}

/* ------------------------- Conversation memory ---------------------------- */
/**
 * MemoryDialog — view / edit / clear the per-conversation memory markdown
 * file (data/memory/<conv_id>.md). The `remember` tool writes here; the user
 * can manually refine, add, or delete entries.
 *
 * The server-side cap (tools.MEMORY_MAX_CHARS) is surfaced by the GET
 * response so we can highlight an over-limit save before the server rejects
 * it. A missing file is a valid empty state, not an error.
 */
function MemoryDialog({ open, onClose, conv }) {
  const [draft, setDraft] = useState('')
  const [original, setOriginal] = useState('')
  const [maxChars, setMaxChars] = useState(16000)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.getConversationMemory(conv.id)
      setDraft(res.content || '')
      setOriginal(res.content || '')
      if (res.max_chars) setMaxChars(res.max_chars)
    } catch (e) {
      toast.error('Could not load memory', { description: e.message })
    } finally {
      setLoading(false)
    }
  }, [conv.id])

  useEffect(() => {
    if (open) refresh()
  }, [open, refresh])

  async function save() {
    setSaving(true)
    try {
      await api.putConversationMemory(conv.id, draft)
      setOriginal(draft)
      toast.success('Memory saved')
    } catch (e) {
      toast.error('Save failed', { description: e.message })
    } finally {
      setSaving(false)
    }
  }

  async function clearAll() {
    setSaving(true)
    try {
      await api.deleteConversationMemory(conv.id)
      setDraft('')
      setOriginal('')
      toast.success('Memory cleared')
    } catch (e) {
      toast.error('Clear failed', { description: e.message })
    } finally {
      setSaving(false)
    }
  }

  const dirty = draft !== original
  const overLimit = draft.length > maxChars

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose?.()}>
      <DialogContent className="flex max-h-[80vh] flex-col sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <BookText className="size-4" />
            Conversation memory
          </DialogTitle>
          <DialogDescription>
            Markdown file injected into the system prompt on every turn. The
            `remember` tool writes here; you can edit by hand or clear it.
          </DialogDescription>
        </DialogHeader>
        {loading ? (
          <div className="flex items-center justify-center py-8 text-muted-foreground">
            <Loader2 className="mr-2 size-4 animate-spin" />
            Loading…
          </div>
        ) : (
          <Textarea
            rows={14}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="(empty — the `remember` tool will fill this in as the conversation progresses)"
            className="flex-1 resize-y font-mono text-xs"
          />
        )}
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>Saved to <code className="rounded bg-muted px-1">data/memory/{conv.id}.md</code></span>
          <span className={cn(overLimit && 'text-destructive')}>
            {draft.length.toLocaleString()} / {maxChars.toLocaleString()}
          </span>
        </div>
        <DialogFooter className="gap-2">
          <Button
            variant="outline"
            className="text-destructive hover:text-destructive"
            onClick={clearAll}
            disabled={saving || (!original && !draft)}
          >
            <Trash2 className="mr-1 size-3.5" />
            Clear
          </Button>
          <div className="flex-1" />
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
          <Button onClick={save} disabled={!dirty || saving || overLimit}>
            {saving ? 'Saving…' : 'Save'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
