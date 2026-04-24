import React, { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Plus, Trash2, Pencil, Eye, EyeOff, KeyRound, Copy } from 'lucide-react'
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
 * SecretsSection — embedded body for the "Secrets" tab inside SettingsPanel.
 *
 * Lets the user register named credentials (API tokens, Bearer strings, etc.)
 * that the `http_request` tool can then substitute via `{{secret:NAME}}`
 * placeholders. Raw values are never displayed in the list — the UI fetches
 * the plaintext on-demand when the user clicks the eye icon.
 *
 * Mutations hit the API immediately; the agent itself never writes here
 * (there is no tool-facing secrets API — credentials are a strictly user-
 * owned concern).
 */
export default function SecretsSection() {
  const [secrets, setSecrets] = useState([])
  const [loading, setLoading] = useState(false)
  const [editing, setEditing] = useState(null) // {id?, name, value, description}
  const [pendingDelete, setPendingDelete] = useState(null)
  // Map<secretId, plaintextValue> — populated lazily by the reveal action and
  // discarded when the panel unmounts / the user hides it again.
  const [revealed, setRevealed] = useState({})

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const { secrets: rows } = await api.listSecrets()
      setSecrets(rows || [])
    } catch (e) {
      toast.error('Failed to load secrets', { description: e.message })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  function startAdd() {
    setEditing({ name: '', value: '', description: '' })
  }

  function startEdit(s) {
    // Start with a blank value — the user must re-enter (or fetch) the
    // plaintext to change it. This keeps the Edit dialog from loading a
    // secret into the DOM just because someone opened it to tweak the
    // description.
    setEditing({
      id: s.id,
      name: s.name,
      value: '',
      description: s.description || '',
    })
  }

  async function saveEditing() {
    if (!editing) return
    const name = (editing.name || '').trim()
    if (!name) {
      toast.error('Name is required')
      return
    }
    const description = (editing.description || '').trim() || null
    const value = editing.value || ''
    try {
      if (editing.id) {
        // On edit, only send a value if the user actually typed one; empty
        // means "leave it alone".
        const patch = { name, description }
        if (value) patch.value = value
        await api.updateSecret(editing.id, patch)
        toast.success('Secret updated')
      } else {
        if (!value) {
          toast.error('Value is required')
          return
        }
        await api.createSecret({ name, value, description })
        toast.success('Secret stored')
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
      await api.deleteSecret(pendingDelete.id)
      toast.success('Secret deleted')
      // Drop any revealed plaintext for this id.
      setRevealed((r) => {
        const copy = { ...r }
        delete copy[pendingDelete.id]
        return copy
      })
      setPendingDelete(null)
      refresh()
    } catch (e) {
      toast.error('Delete failed', { description: e.message })
    }
  }

  async function toggleReveal(s) {
    if (revealed[s.id] !== undefined) {
      // Hide
      setRevealed((r) => {
        const copy = { ...r }
        delete copy[s.id]
        return copy
      })
      return
    }
    try {
      const { secret } = await api.revealSecret(s.id)
      setRevealed((r) => ({ ...r, [s.id]: secret.value }))
    } catch (e) {
      toast.error('Reveal failed', { description: e.message })
    }
  }

  async function copyValue(s) {
    try {
      const plaintext =
        revealed[s.id] !== undefined
          ? revealed[s.id]
          : (await api.revealSecret(s.id)).secret.value
      await navigator.clipboard.writeText(plaintext)
      toast.success('Copied to clipboard')
    } catch (e) {
      toast.error('Copy failed', { description: e.message })
    }
  }

  return (
    <>
      <div className="flex max-h-[60vh] flex-col overflow-hidden">
        <div className="flex items-center justify-between pb-2">
          <div className="text-xs text-muted-foreground">
            {loading
              ? 'Loading…'
              : `${secrets.length} ${secrets.length === 1 ? 'secret' : 'secrets'}`}
          </div>
          <Button size="sm" onClick={startAdd} className="gap-2">
            <Plus className="h-4 w-4" /> New secret
          </Button>
        </div>

        <p className="pb-2 text-xs text-muted-foreground">
          Reference these in HTTP tool calls with{' '}
          <code className="rounded bg-muted px-1">{'{{secret:NAME}}'}</code>.
          The agent never sees the raw value — only the placeholder.
        </p>

        <div className="flex-1 space-y-2 overflow-y-auto pr-1">
          {secrets.length === 0 && !loading && (
            <p className="py-8 text-center text-sm text-muted-foreground">
              No secrets yet. Store API tokens, Bearer strings, or any other
              credential the agent might need — then reference them as{' '}
              <code className="rounded bg-muted px-1">{'{{secret:NAME}}'}</code>.
            </p>
          )}
          {secrets.map((s) => (
            <SecretRow
              key={s.id}
              secret={s}
              revealedValue={revealed[s.id]}
              onToggleReveal={() => toggleReveal(s)}
              onCopy={() => copyValue(s)}
              onEdit={() => startEdit(s)}
              onDelete={() => setPendingDelete(s)}
            />
          ))}
        </div>
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
              {editing?.id ? 'Edit secret' : 'New secret'}
            </DialogTitle>
            <DialogDescription>
              {editing?.id
                ? 'Leave value blank to keep the existing one.'
                : 'Stored locally in the SQLite database. Referenced from HTTP tool calls via {{secret:NAME}}.'}
            </DialogDescription>
          </DialogHeader>
          {editing && (
            <div className="space-y-3">
              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">
                  Name
                </label>
                <Input
                  value={editing.name}
                  onChange={(e) =>
                    setEditing({ ...editing, name: e.target.value })
                  }
                  placeholder="e.g. OPENAI_API_KEY, GITHUB_TOKEN"
                  autoFocus={!editing.id}
                />
                <p className="mt-1 text-[11px] text-muted-foreground">
                  Letters, digits, underscores only. Max 64 chars.
                </p>
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">
                  {editing.id ? 'New value (leave blank to keep current)' : 'Value'}
                </label>
                <Textarea
                  value={editing.value}
                  onChange={(e) =>
                    setEditing({ ...editing, value: e.target.value })
                  }
                  placeholder={editing.id ? '••••••••' : 'Paste the token / credential here'}
                  rows={3}
                  autoFocus={!!editing.id}
                />
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">
                  Description (optional)
                </label>
                <Input
                  value={editing.description}
                  onChange={(e) =>
                    setEditing({ ...editing, description: e.target.value })
                  }
                  placeholder="e.g. personal use, expires 2027-04"
                />
              </div>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditing(null)}>
              Cancel
            </Button>
            <Button onClick={saveEditing}>
              {editing?.id ? 'Save changes' : 'Store secret'}
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
            <DialogTitle>Delete secret?</DialogTitle>
            <DialogDescription>
              Any tool call referencing{' '}
              <code className="rounded bg-muted px-1">
                {'{{secret:' + (pendingDelete?.name || '') + '}}'}
              </code>{' '}
              will fail until you add it back. This cannot be undone.
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

/** One secret row — name, description, reveal/copy/edit/delete buttons. */
function SecretRow({
  secret,
  revealedValue,
  onToggleReveal,
  onCopy,
  onEdit,
  onDelete,
}) {
  const isRevealed = revealedValue !== undefined
  return (
    <div className="flex items-start gap-3 rounded-md border border-border bg-card/40 p-3">
      <KeyRound className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          <code className="text-sm font-semibold">{secret.name}</code>
          {secret.description && (
            <span className="truncate text-xs text-muted-foreground">
              {secret.description}
            </span>
          )}
        </div>
        <div className="mt-1 font-mono text-xs text-muted-foreground">
          {isRevealed ? (
            <span className="break-all text-foreground">{revealedValue}</span>
          ) : (
            <span>••••••••••••••••</span>
          )}
        </div>
      </div>
      <div className="flex shrink-0 gap-1">
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleReveal}
          className="h-7 w-7 text-muted-foreground hover:text-foreground"
          title={isRevealed ? 'Hide value' : 'Reveal value'}
        >
          {isRevealed ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={onCopy}
          className="h-7 w-7 text-muted-foreground hover:text-foreground"
          title="Copy value to clipboard"
        >
          <Copy className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={onEdit}
          className="h-7 w-7 text-muted-foreground hover:text-foreground"
          title="Edit secret"
        >
          <Pencil className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={onDelete}
          className="h-7 w-7 text-destructive hover:text-destructive"
          title="Delete secret"
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}
