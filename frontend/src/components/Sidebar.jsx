import React, { useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import {
  Plus,
  MoreHorizontal,
  Trash2,
  Pencil,
  X,
  Search,
  Sparkles,
  Pin,
  PinOff,
  Tag,
  Bell,
  RefreshCw,
  FolderOpen,
  Settings as SettingsIcon,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Separator } from '@/components/ui/separator'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
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
import NotificationsPanel from './NotificationsPanel'
import SettingsPanel from './SettingsPanel'
import BrandLogo from './BrandLogo'

/**
 * Sidebar — lists conversations, creates new ones, handles rename/delete,
 * pin/unpin, tag editing, and a global search box.
 *
 * Props:
 *   - conversations: array of {id, title, updated_at, pinned, tags, ...}
 *   - activeId: currently selected conversation id
 *   - models: array of installed model name strings
 *   - onSelect: (id) => void
 *   - onReload: () => void  — refetch conversations list after mutation
 *   - onClose: optional, collapses sidebar on mobile when a conversation is picked
 *
 * The visible list is the unfiltered `conversations` prop UNLESS the user has
 * typed something in the search box, in which case we hit the backend search
 * endpoint (debounced) and render those hits instead. Search runs against
 * title, tags, AND message content so the user can find a conversation by
 * something they discussed in it, not just what they named it.
 */
export default function Sidebar({
  conversations,
  activeId,
  models,
  onSelect,
  onJumpToMessage,
  onReload,
  onClose,
}) {
  const [pendingDelete, setPendingDelete] = useState(null) // conv or null
  const [renaming, setRenaming] = useState(null) // {id, value}
  const [notificationsOpen, setNotificationsOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  // Tracks the user's saved default chat model so createNewChat can prefer
  // it over the built-in fallback without a round-trip on every click.
  const [defaultModel, setDefaultModel] = useState('')

  // Pull the user's default-model setting once on mount; refresh when the
  // settings panel closes so a new choice takes effect for the next "New
  // chat" click without requiring a page reload.
  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const res = await api.getSettings()
        if (cancelled) return
        setDefaultModel(
          res?.settings?.default_chat_model ||
            res?.effective_chat_model ||
            '',
        )
      } catch {
        // Non-fatal: createNewChat falls back to the installed-models heuristic.
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [settingsOpen])
  const [taggingConv, setTaggingConv] = useState(null) // conv being tag-edited
  const [tagDraft, setTagDraft] = useState('')
  // Project editor — identical UX to the tag editor: a modal with a single
  // text input, pre-populated with the current project name. Empty/whitespace
  // input clears the project (stored as NULL), grouping the conv under
  // "No project" in the sidebar.
  const [projectingConv, setProjectingConv] = useState(null)
  const [projectDraft, setProjectDraft] = useState('')

  // ----- search state ------------------------------------------------------
  // The sidebar supports two search modes:
  //   - 'keyword': SQL LIKE against title/tags/content — returns conversations.
  //   - 'semantic': cosine similarity over message embeddings — returns
  //                 individual message hits across all conversations.
  // The user toggles between them with a small button next to the search box.
  const [query, setQuery] = useState('')
  const [searchMode, setSearchMode] = useState('keyword') // 'keyword' | 'semantic'
  const [searchHits, setSearchHits] = useState(null) // keyword: conversations[] | null
  const [semanticHits, setSemanticHits] = useState(null) // semantic: hits[] | null
  const [semanticStats, setSemanticStats] = useState(null) // {indexed, total, error?}
  const [semanticLoading, setSemanticLoading] = useState(false)
  const [reindexing, setReindexing] = useState(false)

  // Debounce search so we don't hammer the API on every keystroke. Semantic
  // search is especially worth debouncing because it runs a local embedding
  // inference — cheap per call, but not free.
  const debounceRef = useRef(null)
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    const trimmed = query.trim()
    if (!trimmed) {
      setSearchHits(null)
      setSemanticHits(null)
      setSemanticStats(null)
      return
    }
    const delay = searchMode === 'semantic' ? 300 : 150
    debounceRef.current = setTimeout(async () => {
      try {
        if (searchMode === 'semantic') {
          setSemanticLoading(true)
          const res = await api.semanticSearchConversations(trimmed)
          setSemanticHits(res.hits || [])
          setSemanticStats({
            indexed: res.indexed,
            total: res.total,
            error: res.error,
          })
          if (res.error === 'embedding_unavailable') {
            toast.warning('Semantic search unavailable', {
              description:
                'Run `ollama pull nomic-embed-text` to enable meaning-based search.',
            })
          }
        } else {
          const { conversations: hits } = await api.searchConversations(trimmed)
          setSearchHits(hits || [])
        }
      } catch (e) {
        toast.error('Search failed', { description: e.message })
      } finally {
        setSemanticLoading(false)
      }
    }, delay)
    return () => debounceRef.current && clearTimeout(debounceRef.current)
  }, [query, searchMode])

  // The conversation list we render in keyword mode. Semantic mode uses a
  // completely different layout (see below) so this only applies to keyword.
  const visibleConvos = useMemo(
    () => (searchHits === null ? conversations : searchHits),
    [searchHits, conversations],
  )

  async function runReindex() {
    setReindexing(true)
    try {
      const res = await api.reindexEmbeddings()
      toast.success(
        `Indexed ${res.indexed_now} messages (${res.indexed}/${res.total} total)`,
        res.pending > 0
          ? { description: `${res.pending} still pending — click again to continue.` }
          : undefined,
      )
      setSemanticStats({ indexed: res.indexed, total: res.total })
    } catch (e) {
      toast.error('Reindex failed', { description: e.message })
    } finally {
      setReindexing(false)
    }
  }

  // New-chat flow: opens a modal that forces the user to pick a working
  // directory up front. cwd is fixed for the life of a conversation (the
  // backend refuses PATCHes that change it) so there's no "change later"
  // escape hatch — getting this right on creation matters.
  const [newChatOpen, setNewChatOpen] = useState(false)
  const [newChatCwd, setNewChatCwd] = useState('')
  const [creatingChat, setCreatingChat] = useState(false)
  // Cached "backend install directory" — the zero-config default for a new
  // chat's cwd. Fetched once on mount so the dialog opens instantly.
  const [defaultCwd, setDefaultCwd] = useState('')

  useEffect(() => {
    let cancelled = false
    api
      .getDefaultCwd()
      .then(({ cwd }) => {
        if (!cancelled) setDefaultCwd(cwd || '')
      })
      .catch(() => {
        // Non-fatal: dialog falls back to the remembered cwd / empty input.
      })
    return () => {
      cancelled = true
    }
  }, [])

  function openNewChat() {
    // Zero-config default: the folder Gigachat itself is running from.
    // That's what the user's terminal would already be sitting in if they
    // launched Gigachat from it, so it's the most intuitive starting
    // point. User can still overwrite or Browse before creating.
    setNewChatCwd(defaultCwd)
    setNewChatOpen(true)
  }

  async function browseForNewChatCwd() {
    try {
      const { path } = await api.pickDirectory()
      if (path) setNewChatCwd(path)
    } catch (e) {
      toast.error('Folder picker unavailable', { description: e.message })
    }
  }

  // Model preference order for the new chat:
  //   1. User's saved default (Settings → Default chat model).
  //   2. First installed Gemma 4 variant (auto-detected "best" tier).
  //   3. Any installed model — better than failing to create the chat.
  // We pass an empty string when we have nothing usable; the backend will
  // substitute its auto-tune recommendation.
  async function createNewChat() {
    const cwd = newChatCwd.trim()
    if (!cwd) {
      toast.error('Pick a working directory first')
      return
    }
    const userDefault = defaultModel && models.includes(defaultModel)
      ? defaultModel
      : ''
    const preferred =
      userDefault ||
      models.find((m) => m.startsWith('gemma4:e4b')) ||
      models.find((m) => m.startsWith('gemma4')) ||
      models[0] ||
      ''
    setCreatingChat(true)
    try {
      const { conversation } = await api.createConversation({
        title: 'New chat',
        model: preferred,
        cwd,
      })
      setNewChatOpen(false)
      onReload()
      onSelect(conversation.id)
    } catch (e) {
      toast.error('Could not create chat', { description: e.message })
    } finally {
      setCreatingChat(false)
    }
  }

  async function handleDelete() {
    if (!pendingDelete) return
    try {
      await api.deleteConversation(pendingDelete.id)
      toast.success('Conversation deleted')
      setPendingDelete(null)
      onReload()
      if (pendingDelete.id === activeId) onSelect(null)
    } catch (e) {
      toast.error('Delete failed', { description: e.message })
    }
  }

  async function commitRename() {
    if (!renaming || !renaming.value.trim()) {
      setRenaming(null)
      return
    }
    try {
      await api.updateConversation(renaming.id, { title: renaming.value.trim() })
      onReload()
    } catch (e) {
      toast.error('Rename failed', { description: e.message })
    } finally {
      setRenaming(null)
    }
  }

  // Optimistically flip the pinned flag, then call the API. On failure we
  // refresh the list to roll back to the server's view.
  async function togglePin(conv) {
    try {
      await api.updateConversation(conv.id, { pinned: !conv.pinned })
      onReload()
    } catch (e) {
      toast.error('Pin failed', { description: e.message })
    }
  }

  // ----- tag editing -------------------------------------------------------
  function openTagEditor(conv) {
    setTagDraft((conv.tags || []).join(', '))
    setTaggingConv(conv)
  }

  async function commitTags() {
    if (!taggingConv) return
    // Split on commas, trim, drop blanks. Keep order so the user can re-order
    // by editing the string. No upper bound — typical use is 1–5 tags.
    const next = tagDraft
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean)
    try {
      await api.updateConversation(taggingConv.id, { tags: next })
      onReload()
      toast.success('Tags updated')
    } catch (e) {
      toast.error('Tag update failed', { description: e.message })
    } finally {
      setTaggingConv(null)
      setTagDraft('')
    }
  }

  // ----- project editing ---------------------------------------------------
  function openProjectEditor(conv) {
    setProjectDraft(conv.project || '')
    setProjectingConv(conv)
  }

  async function commitProject() {
    if (!projectingConv) return
    // Empty/whitespace input clears the project (backend normalizes to NULL
    // so the row shows up under the "No project" section).
    const next = projectDraft.trim()
    try {
      await api.updateConversation(projectingConv.id, { project: next })
      onReload()
      toast.success(next ? `Moved to "${next}"` : 'Cleared project')
    } catch (e) {
      toast.error('Project update failed', { description: e.message })
    } finally {
      setProjectingConv(null)
      setProjectDraft('')
    }
  }

  // Unique, sorted list of existing project names — drives the suggestion
  // chips in the project-editor dialog so users can jump into an existing
  // group without having to retype the name exactly.
  const existingProjects = useMemo(() => {
    const set = new Set()
    for (const c of conversations) if (c.project) set.add(c.project)
    return Array.from(set).sort((a, b) =>
      a.localeCompare(b, undefined, { sensitivity: 'base' }),
    )
  }, [conversations])

  // Group the rendered conversations by project. Preserves the parent's pinned-
  // first / most-recent ordering: groups appear in the order their first conv
  // appears, so a project containing a pinned chat floats to the top; "No
  // project" sinks to wherever its first ungrouped conv sits. Keeps the UX
  // predictable — users don't see projects jump around after a new message.
  const groupedConvos = useMemo(() => {
    const groups = [] // [{name, convos}]
    const byName = new Map()
    for (const c of visibleConvos) {
      const key = c.project || ''
      let g = byName.get(key)
      if (!g) {
        g = { name: key, convos: [] }
        byName.set(key, g)
        groups.push(g)
      }
      g.convos.push(c)
    }
    return groups
  }, [visibleConvos])

  return (
    <aside className="flex h-full w-full flex-col bg-card/60 backdrop-blur">
      <div className="flex items-center justify-between p-3">
        <div className="flex items-center gap-2">
          <BrandLogo size="size-7" />
          <span className="font-semibold tracking-tight">Gigachat</span>
        </div>
        {onClose && (
          <Button variant="ghost" size="icon" onClick={onClose} className="md:hidden">
            <X />
          </Button>
        )}
      </div>

      <div className="px-3 pb-2">
        <Button onClick={openNewChat} className="w-full gap-2" size="sm">
          <Plus /> New chat
        </Button>
      </div>

      {/* Search box with keyword/semantic toggle. */}
      <div className="space-y-1 px-3 pb-2">
        <div className="relative">
          {searchMode === 'semantic' ? (
            <Sparkles className="pointer-events-none absolute left-2 top-1/2 size-3.5 -translate-y-1/2 text-primary" />
          ) : (
            <Search className="pointer-events-none absolute left-2 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
          )}
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={
              searchMode === 'semantic'
                ? 'Meaning-based search…'
                : 'Search conversations…'
            }
            className="h-8 pl-7 pr-7 text-xs"
            aria-label={
              searchMode === 'semantic'
                ? 'Semantic search across message history'
                : 'Keyword search conversations'
            }
          />
          {query && (
            <button
              type="button"
              onClick={() => setQuery('')}
              className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-0.5 text-muted-foreground hover:bg-accent"
              aria-label="Clear search"
              title="Clear search"
            >
              <X className="size-3" />
            </button>
          )}
        </div>
        <div className="flex items-center justify-between gap-2">
          <div className="flex overflow-hidden rounded-md border border-border text-[10px]">
            <button
              type="button"
              onClick={() => setSearchMode('keyword')}
              className={cn(
                'px-2 py-0.5 transition-colors',
                searchMode === 'keyword'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-transparent text-muted-foreground hover:bg-accent/60',
              )}
              aria-pressed={searchMode === 'keyword'}
            >
              Keyword
            </button>
            <button
              type="button"
              onClick={() => setSearchMode('semantic')}
              className={cn(
                'px-2 py-0.5 transition-colors',
                searchMode === 'semantic'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-transparent text-muted-foreground hover:bg-accent/60',
              )}
              aria-pressed={searchMode === 'semantic'}
              title="Search by meaning (uses local embeddings)"
            >
              Semantic
            </button>
          </div>
          {searchMode === 'semantic' && semanticStats && (
            <button
              type="button"
              onClick={runReindex}
              disabled={reindexing}
              className="flex items-center gap-1 text-[10px] text-muted-foreground hover:text-foreground disabled:opacity-50"
              title={
                semanticStats.total > semanticStats.indexed
                  ? `${semanticStats.total - semanticStats.indexed} messages not yet indexed. Click to backfill.`
                  : 'All eligible messages are indexed.'
              }
            >
              <RefreshCw
                className={cn('size-3', reindexing && 'animate-spin')}
              />
              {semanticStats.indexed}/{semanticStats.total}
            </button>
          )}
        </div>
      </div>

      <Separator />

      <div className="flex-1 overflow-y-auto p-2">
        {/* Semantic mode: list of message hits with snippet + conv title. */}
        {searchMode === 'semantic' && query.trim() && (
          <SemanticResults
            hits={semanticHits}
            loading={semanticLoading}
            onJump={(cid, mid) => {
              if (onJumpToMessage) onJumpToMessage(cid, mid)
              else onSelect(cid)
              onClose?.()
            }}
          />
        )}

        {/* Keyword mode (or semantic mode with empty query): conversation list. */}
        {(searchMode === 'keyword' || !query.trim()) && (
          <>
            {visibleConvos.length === 0 && (
              <p className="mt-8 px-3 text-center text-xs text-muted-foreground">
                {searchHits === null
                  ? 'No conversations yet. Start one above.'
                  : 'No matches.'}
              </p>
            )}
            {/* Conversations grouped by project. Each group renders its
                header only when we actually have multiple groups OR a
                non-empty project — a single ungrouped list still looks
                identical to the pre-projects sidebar (no empty header). */}
            {groupedConvos.map((group) => {
              const showHeader =
                groupedConvos.length > 1 || Boolean(group.name)
              return (
                <div key={group.name || '__none__'} className="mb-2">
                  {showHeader && (
                    <div className="flex items-center gap-1.5 px-3 pb-1 pt-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                      <FolderOpen className="size-3" />
                      <span className="truncate">
                        {group.name || 'No project'}
                      </span>
                    </div>
                  )}
                  <ul className="space-y-1">
                    {group.convos.map((c) => {
                      const isActive = c.id === activeId
                      const isEditing = renaming?.id === c.id
                      return (
                        <li key={c.id}>
                          <div
                            className={cn(
                              'group relative flex flex-col rounded-md transition-colors',
                              isActive ? 'bg-accent' : 'hover:bg-accent/60',
                            )}
                          >
                            <div className="flex items-center">
                              {isEditing ? (
                                <form
                                  onSubmit={(e) => {
                                    e.preventDefault()
                                    commitRename()
                                  }}
                                  className="flex-1 p-1"
                                >
                                  <Input
                                    autoFocus
                                    value={renaming.value}
                                    onChange={(e) =>
                                      setRenaming({ ...renaming, value: e.target.value })
                                    }
                                    onBlur={commitRename}
                                    onKeyDown={(e) => {
                                      if (e.key === 'Escape') setRenaming(null)
                                    }}
                                    className="h-7 text-sm"
                                  />
                                </form>
                              ) : (
                                <button
                                  onClick={() => {
                                    onSelect(c.id)
                                    onClose?.()
                                  }}
                                  className="flex flex-1 items-center gap-1.5 truncate px-3 py-2 text-left text-sm"
                                >
                                  {/* Pin badge — visible only when this conversation
                                      is pinned, so unpinned items don't get a hole. */}
                                  {c.pinned && (
                                    <Pin className="size-3 shrink-0 fill-current text-primary" />
                                  )}
                                  <span className="clamp-1">{c.title}</span>
                                </button>
                              )}

                              {!isEditing && (
                                <DropdownMenu>
                                  <DropdownMenuTrigger asChild>
                                    <Button
                                      variant="ghost"
                                      size="icon"
                                      className="mr-1 h-7 w-7 opacity-0 transition-opacity group-hover:opacity-100 data-[state=open]:opacity-100"
                                    >
                                      <MoreHorizontal />
                                    </Button>
                                  </DropdownMenuTrigger>
                                  <DropdownMenuContent align="end">
                                    <DropdownMenuItem onClick={() => togglePin(c)}>
                                      {c.pinned ? (
                                        <>
                                          <PinOff /> Unpin
                                        </>
                                      ) : (
                                        <>
                                          <Pin /> Pin
                                        </>
                                      )}
                                    </DropdownMenuItem>
                                    <DropdownMenuItem onClick={() => openTagEditor(c)}>
                                      <Tag /> Tags…
                                    </DropdownMenuItem>
                                    <DropdownMenuItem onClick={() => openProjectEditor(c)}>
                                      <FolderOpen /> Project…
                                    </DropdownMenuItem>
                                    <DropdownMenuSeparator />
                                    <DropdownMenuItem
                                      onClick={() =>
                                        setRenaming({ id: c.id, value: c.title })
                                      }
                                    >
                                      <Pencil /> Rename
                                    </DropdownMenuItem>
                                    <DropdownMenuItem
                                      onClick={() => setPendingDelete(c)}
                                      className="text-destructive focus:text-destructive"
                                    >
                                      <Trash2 /> Delete
                                    </DropdownMenuItem>
                                  </DropdownMenuContent>
                                </DropdownMenu>
                              )}
                            </div>

                            {/* Tag chips — render inline below the title when present.
                                Kept compact so they don't dominate the list. */}
                            {!isEditing && (c.tags?.length ?? 0) > 0 && (
                              <div className="mb-1 ml-3 flex flex-wrap gap-1">
                                {c.tags.map((t) => (
                                  <span
                                    key={t}
                                    className="rounded bg-secondary px-1.5 py-0.5 text-[10px] text-secondary-foreground"
                                  >
                                    {t}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>
                        </li>
                      )
                    })}
                  </ul>
                </div>
              )
            })}
          </>
        )}
      </div>

      {/* Footer — Settings hosts model / memories / hooks / MCP tabs;
          Notifications stays separate because push permission is a
          per-device concern rather than a shared preference. */}
      <Separator />
      <div className="space-y-1 p-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setSettingsOpen(true)}
          className="w-full justify-start gap-2 text-xs text-muted-foreground"
        >
          <SettingsIcon className="h-4 w-4" />
          Settings
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setNotificationsOpen(true)}
          className="w-full justify-start gap-2 text-xs text-muted-foreground"
        >
          <Bell className="h-4 w-4" />
          Notifications
        </Button>
      </div>

      <Dialog
        open={!!pendingDelete}
        onOpenChange={(o) => !o && setPendingDelete(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete conversation?</DialogTitle>
            <DialogDescription>
              "{pendingDelete?.title}" and all its messages will be permanently removed.
              This cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPendingDelete(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Tag editor dialog — comma-separated, free-form. */}
      <Dialog
        open={!!taggingConv}
        onOpenChange={(o) => {
          if (!o) {
            setTaggingConv(null)
            setTagDraft('')
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Tags</DialogTitle>
            <DialogDescription>
              Comma-separated labels for organising and searching conversations.
            </DialogDescription>
          </DialogHeader>
          <Input
            autoFocus
            value={tagDraft}
            onChange={(e) => setTagDraft(e.target.value)}
            placeholder="work, experiments, important"
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                commitTags()
              }
            }}
          />
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setTaggingConv(null)
                setTagDraft('')
              }}
            >
              Cancel
            </Button>
            <Button onClick={commitTags}>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Project editor dialog — free-text grouping label. Suggestion chips
          below the input let the user snap into an existing project name
          with one click instead of retyping (which would create a near-dupe
          project on a typo). Empty input clears the project. */}
      <Dialog
        open={!!projectingConv}
        onOpenChange={(o) => {
          if (!o) {
            setProjectingConv(null)
            setProjectDraft('')
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Project</DialogTitle>
            <DialogDescription>
              Group this conversation under a project name. Leave blank to
              ungroup.
            </DialogDescription>
          </DialogHeader>
          <Input
            autoFocus
            value={projectDraft}
            onChange={(e) => setProjectDraft(e.target.value)}
            placeholder="e.g. Website redesign"
            maxLength={80}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                commitProject()
              }
            }}
          />
          {existingProjects.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {existingProjects.map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => setProjectDraft(p)}
                  className={cn(
                    'rounded-full border px-2 py-0.5 text-xs transition-colors',
                    projectDraft === p
                      ? 'border-primary bg-primary text-primary-foreground'
                      : 'border-border bg-background hover:bg-accent',
                  )}
                >
                  {p}
                </button>
              ))}
            </div>
          )}
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setProjectingConv(null)
                setProjectDraft('')
              }}
            >
              Cancel
            </Button>
            <Button onClick={commitProject}>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* New-chat dialog — forces the user to pick a cwd before the
          conversation row is created. cwd is immutable afterwards, so
          there is no "change later" fallback; a wrong pick means delete
          and retry. Browse uses the native OS folder picker via tkinter
          on the backend. */}
      <Dialog
        open={newChatOpen}
        onOpenChange={(o) => {
          if (!o) setNewChatOpen(false)
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>New chat — pick a working directory</DialogTitle>
            <DialogDescription>
              Commands run from this folder; file paths are resolved
              relative to it. This can&apos;t be changed later — if you need
              a different folder, start another chat.
            </DialogDescription>
          </DialogHeader>
          <div className="flex gap-2">
            <Input
              autoFocus
              value={newChatCwd}
              onChange={(e) => setNewChatCwd(e.target.value)}
              placeholder="C:\Users\you\projects\my-app"
              className="font-mono text-xs"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !creatingChat) {
                  e.preventDefault()
                  createNewChat()
                }
              }}
            />
            <Button
              variant="outline"
              size="icon"
              onClick={browseForNewChatCwd}
              title="Browse for folder"
              aria-label="Browse for folder"
            >
              <FolderOpen className="size-4" />
            </Button>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setNewChatOpen(false)}
              disabled={creatingChat}
            >
              Cancel
            </Button>
            <Button
              onClick={createNewChat}
              disabled={creatingChat || !newChatCwd.trim()}
            >
              {creatingChat ? 'Creating…' : 'Create chat'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <NotificationsPanel
        open={notificationsOpen}
        onClose={() => setNotificationsOpen(false)}
      />
      <SettingsPanel
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      />
    </aside>
  )
}

/**
 * SemanticResults — renders the top cross-conversation message matches.
 *
 * Each hit shows:
 *   - conversation title (bold, 1 line)
 *   - role + truncated snippet from the matched message
 *   - relative timestamp and match score
 *
 * Clicking a hit calls `onJump(conversationId, messageId)` so the parent can
 * switch conversations AND scroll to the exact message inside the chat pane.
 */
function SemanticResults({ hits, loading, onJump }) {
  if (loading && hits === null) {
    return (
      <p className="mt-8 px-3 text-center text-xs text-muted-foreground">
        Searching…
      </p>
    )
  }
  if (hits === null) return null
  if (hits.length === 0) {
    return (
      <p className="mt-8 px-3 text-center text-xs text-muted-foreground">
        No semantically-similar messages found.
      </p>
    )
  }
  return (
    <ul className="space-y-1">
      {hits.map((h) => (
        <li key={h.message_id}>
          <button
            type="button"
            onClick={() => onJump(h.conversation_id, h.message_id)}
            className="group flex w-full flex-col gap-1 rounded-md px-3 py-2 text-left hover:bg-accent/60"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="clamp-1 text-xs font-medium">
                {h.conversation_title || '(untitled)'}
              </span>
              <span
                className="shrink-0 rounded bg-secondary px-1 py-[1px] text-[9px] text-secondary-foreground"
                title={`Similarity score: ${h.score}`}
              >
                {Math.round(h.score * 100)}%
              </span>
            </div>
            <div className="flex gap-1 text-[11px] text-muted-foreground">
              <span className="shrink-0 uppercase tracking-wider text-[9px] mt-[2px]">
                {h.role === 'user' ? 'you' : 'ai'}
              </span>
              <span className="clamp-2">{h.snippet}</span>
            </div>
          </button>
        </li>
      ))}
    </ul>
  )
}
