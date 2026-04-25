import React, { memo, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { toast } from 'sonner'
import {
  User,
  Pin,
  PinOff,
  Pencil,
  Check,
  X as XIcon,
  Hourglass,
  SquareDashedBottomCode,
} from 'lucide-react'
import BrandLogo from './BrandLogo'
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { Textarea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { languageToArtifactKind, CODE_ARTIFACT_MIN_LINES } from './ArtifactPanel'

/**
 * Message — renders one user or assistant message.
 *
 * We deliberately do NOT render tool messages here; they are displayed
 * inside the assistant message that contains the tool_calls, as ToolCall
 * cards (wiring happens in ChatView).
 *
 * Props:
 *   - role: 'user' | 'assistant'
 *   - content: string (markdown for assistant, plain for user)
 *   - images: optional array of filenames (served via /api/uploads/<name>)
 *             that the user attached to this row
 *   - pinned: boolean — whether this message is pinned (exempt from
 *             auto-compaction). When true the pin badge stays visible.
 *   - queued: boolean — message is queued behind the running turn and
 *             hasn't been picked up by the agent yet. We render a small
 *             hourglass + dim style so the user sees it landed.
 *   - onTogglePin: optional function() — called when the user clicks the
 *                  pin button. Omit to hide the button entirely (e.g. for
 *                  optimistic temp rows that have no DB id yet).
 *   - onEdit: optional function(newContent) — when present, clicking the
 *             Edit pencil swaps in an inline textarea. Save calls this with
 *             the new text. Pass undefined for rows that have no persisted
 *             id (temp optimistic rows, assistant messages).
 *   - canEdit: boolean — whether the edit pencil should be OFFERED right now.
 *             Kept separate from `onEdit` so the callback stays available
 *             once the user enters edit mode, even if the parent's
 *             "can they start an edit?" conditions flip mid-edit (e.g.
 *             `busy` flipping true while the user is typing). Without this
 *             split, Save & regenerate silently no-ops when onEdit
 *             disappears under the user.
 *   - children: extra nodes (e.g. ToolCall cards) rendered under the content
 *
 * Wrapped in `React.memo` (see export at the bottom) so a `setMessages`
 * call in ChatView — fired on every tool event and delta — only re-renders
 * the rows whose props actually changed. Without this, a long transcript
 * pays the full markdown/render cost on every SSE tick.
 */
function Message({
  role,
  content,
  images,
  pinned = false,
  queued = false,
  onTogglePin,
  onEdit,
  canEdit = false,
  onOpenArtifact,
  children,
}) {
  const isUser = role === 'user'
  const hasImages = isUser && Array.isArray(images) && images.length > 0

  // Local edit-mode state. When the user clicks the pencil we copy `content`
  // into a draft buffer and let them edit; pressing Save calls onEdit(draft)
  // which dispatches the server-side regenerate. Cancel reverts and exits.
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(content)
  const editRef = useRef(null)

  useEffect(() => {
    // Reset the draft if the canonical content changes from outside (e.g.
    // SSE refresh after a regenerate completes). Keeps Save/Cancel honest.
    if (!editing) setDraft(content)
  }, [content, editing])

  useEffect(() => {
    // Auto-grow the editor textarea so it doesn't open at a single line.
    const el = editRef.current
    if (!editing || !el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 320) + 'px'
  }, [editing, draft])

  function startEdit() {
    setDraft(content)
    setEditing(true)
  }

  function cancelEdit() {
    setEditing(false)
    setDraft(content)
  }

  function saveEdit() {
    const next = (draft || '').trim()
    // Empty draft → cancel; an empty user prompt would just confuse the
    // model. But unchanged text is FINE — "regenerate from the same
    // prompt" is a valid action (the previous response was bad, try
    // again). We deliberately do NOT bail on `next === content`; every
    // mainstream chat UI lets you regenerate without editing.
    if (!next) {
      cancelEdit()
      return
    }
    // If onEdit has vanished under us (shouldn't happen now that ChatView
    // passes it unconditionally, but defensive against future regressions),
    // surface the failure instead of silently swallowing the click.
    if (typeof onEdit !== 'function') {
      toast.error('Could not save edit', {
        description: 'The regenerate handler is unavailable. Try reopening this chat.',
      })
      return
    }
    setEditing(false)
    onEdit(next)
  }

  function handleEditKey(e) {
    // Ctrl/Cmd+Enter saves; Escape cancels. Plain Enter inserts a newline so
    // multi-line edits work.
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      saveEdit()
    } else if (e.key === 'Escape') {
      e.preventDefault()
      cancelEdit()
    }
  }

  return (
    // `group` lets the action buttons fade in on hover. When pinned/queued
    // we keep the indicator visible even without hover so the state is
    // visible at a glance.
    //
    // Chat-app layout: user messages flow right-to-left (avatar on right,
    // bubble hugging the right edge) and assistant messages flow normally
    // left-to-right. This mirrors WhatsApp / iMessage conventions so a
    // quick glance at the rail tells you who said what without reading.
    <div
      className={cn(
        'group relative flex gap-3 py-3',
        isUser && 'flex-row-reverse',
        queued && 'opacity-70',
      )}
    >
      {isUser ? (
        // Generic user glyph — keeps "you vs Gigachat" visually distinct at
        // a glance without needing to read the role label.
        <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-primary text-xs text-primary-foreground">
          <User className="size-4" />
        </div>
      ) : (
        // Assistant avatar = the brand logo. Same image as the sidebar mark
        // so the user instantly associates the avatar with "this is Gigachat".
        <BrandLogo alt="Gigachat assistant" />
      )}
      <div
        className={cn(
          'min-w-0',
          // User side: shrink to fit + constrain so a short "hi" doesn't
          // stretch into a full-width bar. `items-end` right-aligns the
          // bubble within the column on the rare case the bubble is shorter
          // than its children (header row, image strip).
          // Assistant side: fill the column so markdown prose can flow
          // naturally left-to-right.
          isUser ? 'flex max-w-[85%] flex-col items-end' : 'flex-1',
        )}
      >
        <div
          className={cn(
            'mb-1 flex items-center gap-2 text-xs font-medium text-muted-foreground',
            // Mirror the action-button row for user messages so the label
            // sits next to the avatar (which is now on the right).
            isUser && 'flex-row-reverse',
          )}
        >
          <span>{isUser ? 'You' : 'Gigachat'}</span>
          {queued && (
            <span
              className="inline-flex items-center gap-1 rounded bg-secondary px-1.5 py-0.5 text-[10px] text-secondary-foreground"
              title="Queued — will be sent to the agent after the current step finishes"
            >
              <Hourglass className="size-3" />
              queued
            </span>
          )}
          {onTogglePin && (
            <button
              type="button"
              onClick={onTogglePin}
              aria-pressed={pinned}
              aria-label={pinned ? 'Unpin message' : 'Pin message'}
              title={
                pinned
                  ? 'Unpin — message will become eligible for auto-compaction again'
                  : 'Pin — keep this message in context permanently'
              }
              className={cn(
                'inline-flex size-5 items-center justify-center rounded transition-opacity',
                'hover:bg-accent hover:text-accent-foreground',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                pinned
                  ? 'text-primary opacity-100'
                  : 'opacity-0 group-hover:opacity-70 focus-visible:opacity-100',
              )}
            >
              {pinned ? (
                <Pin className="size-3.5 fill-current" />
              ) : (
                <PinOff className="size-3.5" />
              )}
            </button>
          )}
          {/* Pencil visibility is gated on `canEdit` (not on `onEdit`) so the
              callback can stay available across re-renders and survive a
              mid-edit state flip. See `saveEdit` above for the failsafe
              toast when `onEdit` is missing anyway. */}
          {canEdit && !editing && (
            <button
              type="button"
              onClick={startEdit}
              aria-label="Edit message"
              title="Edit and regenerate"
              className={cn(
                'inline-flex size-5 items-center justify-center rounded text-muted-foreground transition-opacity',
                'hover:bg-accent hover:text-foreground',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                'opacity-0 group-hover:opacity-70 focus-visible:opacity-100',
              )}
            >
              <Pencil className="size-3.5" />
            </button>
          )}
        </div>
        {hasImages && <UserImageStrip names={images} />}

        {editing ? (
          // Inline edit mode — textarea + Save/Cancel. Save calls onEdit
          // which triggers the server-side regenerate flow in ChatView.
          // `type="button"` on both buttons prevents any ambient form
          // ancestor from intercepting the click as a submit (defensive).
          <div className="w-full space-y-2">
            <Textarea
              ref={editRef}
              autoFocus
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={handleEditKey}
              className="text-sm leading-relaxed"
            />
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <Button type="button" size="sm" onClick={saveEdit}>
                <Check className="size-3.5" />
                Save & regenerate
              </Button>
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={cancelEdit}
              >
                <XIcon className="size-3.5" />
                Cancel
              </Button>
              <span className="text-muted-foreground">
                Cmd/Ctrl+Enter to save · Esc to cancel
              </span>
            </div>
          </div>
        ) : (
          content && (
            isUser ? (
              // User bubble — iMessage-style: primary-tinted pill that hugs
              // the right edge. `rounded-br-sm` slightly clips the bottom-
              // right corner as a subtle "tail" nod without going full
              // chat-tail SVG. `inline-block` so short messages shrink to
              // fit their content instead of spanning the column.
              <div className="inline-block max-w-full rounded-2xl rounded-br-sm bg-primary px-3.5 py-2 text-primary-foreground shadow-sm">
                <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">
                  {content}
                </p>
              </div>
            ) : (
              <div className="prose-chat text-sm">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    code: ({ node, inline, className, children, ...props }) => {
                      // Inline (backtick-style) code is rendered as-is.
                      if (inline) {
                        return (
                          <code className={className} {...props}>
                            {children}
                          </code>
                        )
                      }
                      // Fenced block: language hint lives in the className as
                      // `language-<lang>`. We use it to detect "artifact-
                      // eligible" blocks and overlay a small "Open preview"
                      // button. Specialised kinds (html/svg/mermaid/markdown)
                      // always qualify; generic code blocks only qualify
                      // when they're long enough to benefit from a
                      // dedicated, line-numbered viewer.
                      const lang = (className || '')
                        .replace(/language-/, '')
                        .trim()
                      const kind = languageToArtifactKind(lang)
                      const source = String(children || '').replace(/\n$/, '')
                      const lineCount = source ? source.split('\n').length : 0
                      const qualifies =
                        kind === 'code'
                          ? lineCount >= CODE_ARTIFACT_MIN_LINES
                          : !!kind
                      return (
                        <div className="group relative">
                          {qualifies && onOpenArtifact && (
                            <button
                              type="button"
                              onClick={() =>
                                onOpenArtifact({
                                  kind,
                                  source,
                                  language: kind === 'code' ? lang : undefined,
                                  title:
                                    kind === 'code'
                                      ? `${lang || 'code'} (${lineCount} lines)`
                                      : `${kind} preview`,
                                })
                              }
                              className="absolute right-2 top-2 z-10 flex items-center gap-1 rounded border border-border bg-background/90 px-2 py-0.5 text-[10px] text-muted-foreground opacity-80 shadow-sm transition-all hover:bg-accent hover:text-foreground"
                              title="Open in preview pane"
                            >
                              <SquareDashedBottomCode className="size-3" />
                              Open preview
                            </button>
                          )}
                          <code className={className} {...props}>
                            {children}
                          </code>
                        </div>
                      )
                    },
                  }}
                >
                  {content}
                </ReactMarkdown>
              </div>
            )
          )
        )}
        {!editing && children}
      </div>
    </div>
  )
}

/**
 * UserImageStrip — small row of thumbnails for images the user attached.
 *
 * Each thumbnail is clickable and opens the full-size version in a modal,
 * reusing the shadcn Dialog we already rely on for screenshots. For the
 * optimistic temp-row (rendered before the server echoes the user message
 * back), the filename is actually a blob URL prefix; we fall back to the
 * /api/uploads path otherwise.
 */
function UserImageStrip({ names }) {
  return (
    <div className="mb-2 flex flex-wrap gap-2">
      {names.map((name) => {
        const src = name.startsWith('blob:') || name.startsWith('data:')
          ? name
          : `/api/uploads/${name}`
        return (
          <Dialog key={name}>
            <DialogTrigger asChild>
              <button
                type="button"
                className="block overflow-hidden rounded-md border border-border bg-black/10 hover:border-primary/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                aria-label="Expand attached image"
              >
                <img
                  src={src}
                  alt="User attachment"
                  loading="lazy"
                  className="h-24 w-auto object-contain"
                />
              </button>
            </DialogTrigger>
            <DialogContent className="max-w-[95vw] p-2 sm:max-w-5xl">
              <DialogTitle className="sr-only">Attachment</DialogTitle>
              <img
                src={src}
                alt="User attachment (full size)"
                className="h-auto max-h-[85vh] w-full object-contain"
              />
            </DialogContent>
          </Dialog>
        )
      })}
    </div>
  )
}

export default memo(Message)
