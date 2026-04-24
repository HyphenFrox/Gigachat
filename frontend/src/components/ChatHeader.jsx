import React, { useEffect, useState } from 'react'
import { toast } from 'sonner'
import {
  Menu,
  Folder,
  ChevronDown,
  Eye,
  ShieldCheck,
  Zap,
  ClipboardList,
  Loader2,
  Database,
  CircleAlert,
  RefreshCw,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Separator } from '@/components/ui/separator'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import TokenUsage from './TokenUsage'
import ConversationActions from './ConversationActions'
import { api } from '@/lib/api'

/**
 * ChatHeader — top bar of the chat view.
 *
 * Controls:
 *   - Hamburger button (mobile only) to open the Sidebar.
 *   - Conversation title (click to rename inline).
 *   - Working directory chip (read-only; set at chat creation and fixed).
 *   - Token usage indicator (hidden on small screens) — rough context-window
 *     estimate so the user can see when auto-compaction is likely to fire.
 *   - Model picker.
 *   - Auto-approve switch (per conversation) — mirrors Claude Code's
 *     permission-mode toggle. When OFF, every tool call pauses for approval.
 *
 * All mutations go through PATCH /api/conversations/:id.
 */
export default function ChatHeader({
  conv,
  messages,
  onUpdate,
  onOpenSidebar,
  onScrollToMessage,
}) {
  const [editingTitle, setEditingTitle] = useState(false)
  const [titleDraft, setTitleDraft] = useState(conv.title)

  async function patch(body) {
    try {
      const { conversation } = await api.updateConversation(conv.id, body)
      onUpdate(conversation)
    } catch (e) {
      toast.error('Update failed', { description: e.message })
    }
  }

  async function commitTitle() {
    setEditingTitle(false)
    const t = titleDraft.trim()
    if (t && t !== conv.title) await patch({ title: t })
    else setTitleDraft(conv.title)
  }

  return (
    <header className="flex flex-wrap items-center gap-2 border-b border-border bg-background/80 px-3 py-2 backdrop-blur md:px-4">
      <Button
        variant="ghost"
        size="icon"
        className="md:hidden"
        onClick={onOpenSidebar}
        aria-label="Open conversation list"
      >
        <Menu />
      </Button>

      {editingTitle ? (
        <form
          className="flex-1"
          onSubmit={(e) => {
            e.preventDefault()
            commitTitle()
          }}
        >
          <Input
            autoFocus
            value={titleDraft}
            onChange={(e) => setTitleDraft(e.target.value)}
            onBlur={commitTitle}
            onKeyDown={(e) => {
              if (e.key === 'Escape') {
                setEditingTitle(false)
                setTitleDraft(conv.title)
              }
            }}
            className="h-8"
          />
        </form>
      ) : (
        <button
          onClick={() => setEditingTitle(true)}
          className="flex-1 min-w-0 truncate rounded px-2 py-1 text-left text-sm font-medium hover:bg-accent"
          title="Click to rename"
        >
          {conv.title}
        </button>
      )}

      <Separator orientation="vertical" className="mx-1 h-6 hidden sm:block" />

      {/* cwd is read-only after creation: the whole conversation — checkpoints,
          codebase index rows, hook outputs — is keyed by this path. Swapping
          mid-flight would orphan every artefact. The tooltip calls this out
          so a user clicking around for an "edit" button gets the explanation. */}
      <div
        className="hidden max-w-[220px] items-center gap-2 rounded-md px-2 py-1 font-mono text-xs text-muted-foreground sm:flex"
        title={`${conv.cwd}\n(fixed for the life of the conversation — start a new chat to change)`}
      >
        <Folder className="size-4 shrink-0" />
        <span className="truncate">{conv.cwd}</span>
      </div>

      <CodebaseIndexChip conv={conv} />

      <TokenUsage messages={messages} conv={conv} />

      <PermissionModePicker conv={conv} onPatch={patch} />

      <ConversationActions
        conv={conv}
        onUpdate={onUpdate}
        onScrollToMessage={onScrollToMessage}
      />
    </header>
  )
}

/**
 * PermissionModePicker — replaces the old binary "Auto-approve" switch with a
 * three-state selector mirroring the backend's `permission_mode`:
 *
 *   - read_only      Eye icon    — writes are refused with no approval UI
 *   - approve_edits  ShieldCheck — writes pause for manual approval (default)
 *   - allow_all      Zap         — nothing pauses
 *
 * The label collapses to just the icon + short word on narrow screens. Hover
 * tooltip explains the semantics so new users know what "approve edits"
 * actually gates (writes only, reads run silently).
 */
function PermissionModePicker({ conv, onPatch }) {
  const mode =
    conv.permission_mode ||
    (conv.auto_approve ? 'allow_all' : 'approve_edits')

  const config = {
    read_only: {
      label: 'Read only',
      short: 'Read',
      Icon: Eye,
      color: 'text-sky-400',
      description:
        'The agent can only read — files, screenshots, web pages. Any write, click, shell, or edit is refused without an approval prompt.',
    },
    plan: {
      label: 'Plan mode',
      short: 'Plan',
      Icon: ClipboardList,
      color: 'text-purple-400',
      description:
        'Research + propose, then approve. Writes are refused and the agent is instructed to investigate freely and produce an approvable step-by-step plan ending with [PLAN READY]. Click Execute plan to switch modes and replay the plan for execution.',
    },
    approve_edits: {
      label: 'Approve edits',
      short: 'Approve',
      Icon: ShieldCheck,
      color: 'text-emerald-400',
      description:
        'Reads run silently. Writes (shell, edits, clicks, network mutation) pause for manual approval. Default — the safest way to let an agent help without friction on harmless lookups.',
    },
    allow_all: {
      label: 'Allow all',
      short: 'Allow',
      Icon: Zap,
      color: 'text-amber-400',
      description:
        'Nothing pauses. The agent runs every tool call immediately — shell, edits, desktop control, everything. Use only for trusted automation.',
    },
  }
  const active = config[mode] || config.approve_edits
  const ActiveIcon = active.Icon

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className="ml-1 gap-1.5 text-xs"
          title={active.description}
        >
          <ActiveIcon className={`size-3.5 ${active.color}`} />
          <span className="hidden sm:inline">{active.label}</span>
          <span className="sm:hidden">{active.short}</span>
          <ChevronDown className="size-3 opacity-60" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-72">
        <DropdownMenuLabel className="text-xs">Permission mode</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {Object.entries(config).map(([key, meta]) => {
          const Icon = meta.Icon
          const selected = key === mode
          return (
            <DropdownMenuItem
              key={key}
              onClick={() => !selected && onPatch({ permission_mode: key })}
              className="items-start gap-2"
            >
              <Icon className={`size-4 mt-0.5 shrink-0 ${meta.color}`} />
              <div className="flex-1">
                <div className="flex items-center gap-2 text-sm">
                  {meta.label}
                  {selected && (
                    <span className="ml-auto text-[10px] text-muted-foreground">
                      current
                    </span>
                  )}
                </div>
                <div className="mt-0.5 text-[11px] leading-snug text-muted-foreground">
                  {meta.description}
                </div>
              </div>
            </DropdownMenuItem>
          )
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

/**
 * CodebaseIndexChip — tiny pill showing whether the semantic index for this
 * conversation's cwd is ready, indexing, or errored. Clicking the chip
 * forces a fresh background re-index; the status auto-refreshes on a short
 * interval while the build is running so the user sees progress without
 * having to poke the button.
 *
 * Hidden entirely on screens narrower than `sm` — the header gets busy and
 * the index status isn't load-bearing for the core chat UX.
 */
function CodebaseIndexChip({ conv }) {
  const [idx, setIdx] = useState(null)
  const [refreshing, setRefreshing] = useState(false)

  async function refresh() {
    try {
      const { index } = await api.getCodebaseIndex(conv.id)
      setIdx(index)
    } catch {
      // Non-fatal — leave prior state. A failed poll shouldn't mask the
      // previous snapshot with a confusing "null" display.
    }
  }

  // Initial fetch + while-indexing poll. We only spin up the interval when
  // a build is actually running so idle conversations don't hammer the API.
  useEffect(() => {
    refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conv.id])

  useEffect(() => {
    if (!idx || (idx.status !== 'indexing' && idx.status !== 'pending')) return
    const t = setInterval(refresh, 2000)
    return () => clearInterval(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idx?.status])

  async function triggerReindex(e) {
    e.stopPropagation() // parent header has click handlers too
    if (refreshing) return
    setRefreshing(true)
    try {
      const { index } = await api.reindexCodebase(conv.id)
      setIdx(index)
      toast.success('Reindex started')
    } catch (err) {
      toast.error('Reindex failed', { description: err.message })
    } finally {
      setRefreshing(false)
    }
  }

  const status = idx?.status || 'none'
  const meta = {
    none: {
      Icon: Database,
      color: 'text-muted-foreground',
      label: 'No index',
      title: 'Codebase semantic index not built yet — click to build it now.',
    },
    pending: {
      Icon: Loader2,
      color: 'text-amber-400 animate-spin',
      label: 'Queued',
      title: 'Index build queued — it will start shortly.',
    },
    indexing: {
      Icon: Loader2,
      color: 'text-amber-400 animate-spin',
      label: 'Indexing',
      title: 'Building the semantic index…',
    },
    ready: {
      Icon: Database,
      color: 'text-emerald-400',
      label: `${idx?.file_count ?? 0} files`,
      title: `Indexed ${idx?.file_count ?? 0} files / ${idx?.chunk_count ?? 0} chunks. Click to rebuild.`,
    },
    error: {
      Icon: CircleAlert,
      color: 'text-destructive',
      label: 'Index error',
      title: idx?.error || 'Index build failed — click to retry.',
    },
  }[status]
  const { Icon, color, label, title } = meta
  return (
    <Button
      variant="ghost"
      size="sm"
      className="hidden gap-1.5 text-xs text-muted-foreground sm:flex"
      onClick={triggerReindex}
      disabled={refreshing}
      title={title}
    >
      <Icon className={`size-4 ${color}`} />
      <span>{label}</span>
      {status === 'ready' && (
        <RefreshCw className="size-3 opacity-60" />
      )}
    </Button>
  )
}
