import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import {
  Plus,
  Trash2,
  RefreshCw,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Globe,
  ExternalLink,
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
import { api } from '@/lib/api'

/**
 * DocsSection — embedded body for the "Docs" tab inside SettingsPanel.
 *
 * Manages the list of public documentation URLs the agent can search via
 * the `docs_search` tool. Each row shows crawl status (pending/crawling/
 * ready/error), counts (pages + chunks), and offers reindex / delete.
 *
 * Crawls are fire-and-forget on the backend — the UI polls every 2 s
 * while any row is in a non-terminal state so users see the counts tick
 * up without a manual refresh.
 */
const NON_TERMINAL = new Set(['pending', 'crawling'])

export default function DocsSection() {
  const [urls, setUrls] = useState([])
  const [loading, setLoading] = useState(false)
  const [adding, setAdding] = useState(false)
  const [draft, setDraft] = useState(null) // partial seed while the user types
  const [pendingDelete, setPendingDelete] = useState(null)
  const pollRef = useRef(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const { urls: rows } = await api.listDocUrls()
      setUrls(rows || [])
    } catch (e) {
      toast.error('Failed to load docs', { description: e.message })
    } finally {
      setLoading(false)
    }
  }, [])

  // One-shot refresh on mount + poll while something is in flight so the
  // page counts keep climbing without a manual refresh click.
  useEffect(() => {
    refresh()
  }, [refresh])

  useEffect(() => {
    const anyActive = urls.some((u) => NON_TERMINAL.has(u.status))
    if (!anyActive) {
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
      return
    }
    if (pollRef.current) return
    pollRef.current = setInterval(() => {
      refresh()
    }, 2000)
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [urls, refresh])

  function startAdd() {
    setDraft({
      url: '',
      max_pages: 20,
      same_origin_only: true,
    })
    setAdding(true)
  }

  async function saveDraft() {
    if (!draft) return
    const u = (draft.url || '').trim()
    if (!u) {
      toast.error('URL cannot be empty')
      return
    }
    if (!/^https?:\/\//i.test(u)) {
      toast.error('URL must start with http:// or https://')
      return
    }
    try {
      await api.createDocUrl({
        url: u,
        max_pages: Number(draft.max_pages) || 20,
        same_origin_only: !!draft.same_origin_only,
      })
      toast.success('Indexing started')
      setAdding(false)
      setDraft(null)
      refresh()
    } catch (e) {
      toast.error('Could not add URL', { description: e.message })
    }
  }

  async function reindex(row) {
    try {
      await api.reindexDocUrl(row.id)
      toast.success('Re-crawl queued')
      refresh()
    } catch (e) {
      toast.error('Re-crawl failed', { description: e.message })
    }
  }

  async function confirmDelete() {
    if (!pendingDelete) return
    try {
      await api.deleteDocUrl(pendingDelete.id)
      toast.success('URL removed')
      setPendingDelete(null)
      refresh()
    } catch (e) {
      toast.error('Delete failed', { description: e.message })
    }
  }

  // Sort: active crawls first, then ready, then error, newest within each.
  const sorted = useMemo(() => {
    const order = { crawling: 0, pending: 1, ready: 2, error: 3 }
    const copy = [...urls]
    copy.sort((a, b) => {
      const ra = order[a.status] ?? 4
      const rb = order[b.status] ?? 4
      if (ra !== rb) return ra - rb
      return (b.updated_at || 0) - (a.updated_at || 0)
    })
    return copy
  }, [urls])

  return (
    <>
      <div className="flex max-h-[60vh] flex-col overflow-hidden">
        <div className="rounded-md border border-border bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
          Point Gigachat at a docs site and it will crawl, chunk, and embed
          every same-origin page so the agent can answer questions against
          your reference material instead of hallucinating. Use the{' '}
          <code>docs_search</code> tool inside a conversation to query the
          index.
        </div>

        <div className="mt-2 flex items-center justify-between pb-2">
          <div className="text-xs text-muted-foreground">
            {loading
              ? 'Loading…'
              : `${sorted.length} indexed URL${sorted.length === 1 ? '' : 's'}`}
          </div>
          <Button size="sm" onClick={startAdd} className="gap-2">
            <Plus className="h-4 w-4" /> Add URL
          </Button>
        </div>

        <div className="flex-1 space-y-2 overflow-y-auto pr-1">
          {sorted.length === 0 && !loading && (
            <p className="py-8 text-center text-sm text-muted-foreground">
              No documentation URLs indexed yet. Click &ldquo;Add URL&rdquo;
              to crawl one.
            </p>
          )}
          {sorted.map((u) => (
            <DocUrlRow
              key={u.id}
              row={u}
              onReindex={() => reindex(u)}
              onDelete={() => setPendingDelete(u)}
            />
          ))}
        </div>
      </div>

      {/* Add drawer */}
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
            <DialogTitle>Add docs URL</DialogTitle>
            <DialogDescription>
              The crawler fetches the seed page, extracts readable prose,
              and follows same-origin links breadth-first up to the page
              cap. Private / loopback URLs are rejected.
            </DialogDescription>
          </DialogHeader>
          {draft && (
            <div className="space-y-3">
              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">
                  Seed URL
                </label>
                <Input
                  value={draft.url}
                  onChange={(e) => setDraft({ ...draft, url: e.target.value })}
                  placeholder="https://docs.python.org/3/"
                  autoFocus
                />
              </div>
              <div className="flex items-center justify-between">
                <label className="flex items-center gap-2 text-xs">
                  <span className="text-muted-foreground">Max pages</span>
                  <Input
                    type="number"
                    min={1}
                    max={100}
                    value={draft.max_pages}
                    onChange={(e) =>
                      setDraft({ ...draft, max_pages: e.target.value })
                    }
                    className="h-8 w-20"
                  />
                </label>
                <label className="flex items-center gap-2 text-xs">
                  <span className="text-muted-foreground">Same origin only</span>
                  <Switch
                    checked={!!draft.same_origin_only}
                    onCheckedChange={(v) =>
                      setDraft({ ...draft, same_origin_only: !!v })
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
            <Button onClick={saveDraft}>Start crawl</Button>
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
            <DialogTitle>Remove this URL?</DialogTitle>
            <DialogDescription>
              Drops the seed and every page chunk crawled beneath it. The
              agent will no longer be able to search its content.
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

/** One row inside the URL list — status badge, counts, reindex, delete. */
function DocUrlRow({ row, onReindex, onDelete }) {
  const { Icon, tint, label } = statusVisual(row.status)
  // Prefer the crawled title, fall back to hostname + path for a compact
  // display name when the crawl is still pending.
  const title = row.title || prettyUrl(row.url)
  return (
    <div className="flex items-start gap-3 rounded-md border border-border bg-card/40 p-3">
      <div className={`mt-0.5 shrink-0 ${tint}`}>
        <Icon className="h-4 w-4" />
      </div>
      <div className="flex-1 overflow-hidden">
        <div className="flex items-center gap-1.5 text-sm font-medium">
          <span className="truncate">{title}</span>
          <a
            href={row.url}
            target="_blank"
            rel="noreferrer"
            className="text-muted-foreground hover:text-foreground"
            title="Open source"
          >
            <ExternalLink className="h-3 w-3" />
          </a>
        </div>
        <div className="truncate text-xs text-muted-foreground">{row.url}</div>
        <div className="mt-1 flex items-center gap-3 text-[11px] text-muted-foreground">
          <span>{label}</span>
          <span>
            {row.pages_crawled || 0} page{row.pages_crawled === 1 ? '' : 's'}
          </span>
          <span>
            {row.chunk_count || 0} chunk{row.chunk_count === 1 ? '' : 's'}
          </span>
          {row.max_pages ? <span>cap {row.max_pages}</span> : null}
        </div>
        {row.error ? (
          <div className="mt-1 truncate text-[11px] text-red-500" title={row.error}>
            {row.error}
          </div>
        ) : null}
      </div>
      <div className="flex shrink-0 items-center gap-1">
        <Button
          variant="ghost"
          size="icon"
          title="Re-crawl"
          onClick={onReindex}
          disabled={NON_TERMINAL.has(row.status)}
          className="h-8 w-8"
        >
          <RefreshCw className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          title="Remove"
          onClick={onDelete}
          className="h-8 w-8 text-red-500 hover:text-red-600"
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}

/** Map a crawl status onto an icon + colour + label for the row header. */
function statusVisual(status) {
  switch (status) {
    case 'crawling':
    case 'pending':
      return { Icon: Loader2, tint: 'text-blue-500 animate-spin', label: 'Crawling…' }
    case 'ready':
      return { Icon: CheckCircle2, tint: 'text-green-500', label: 'Ready' }
    case 'error':
      return { Icon: AlertCircle, tint: 'text-red-500', label: 'Error' }
    default:
      return { Icon: Globe, tint: 'text-muted-foreground', label: status || '—' }
  }
}

/** Compact hostname + first path segment for the row title before the
 *  real page title arrives from the crawler. */
function prettyUrl(u) {
  try {
    const p = new URL(u)
    const seg = p.pathname.split('/').filter(Boolean)[0] || ''
    return seg ? `${p.hostname}/${seg}` : p.hostname
  } catch {
    return u
  }
}
