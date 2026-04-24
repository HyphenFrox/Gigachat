import React from 'react'
import { Circle, Loader2, CheckCircle2, ListChecks } from 'lucide-react'
import { cn } from '@/lib/utils'

/**
 * TodoPanel — pinned task list the agent maintains via the `todo_write` tool.
 *
 * Shows the most recently published list as a compact, stickied strip above
 * the chat composer. Collapses into a single-line summary on mobile so it
 * doesn't crowd the screen. When no todos have been set for the current
 * conversation, the panel renders nothing at all.
 *
 * Props:
 *   - todos: Array<{ content, activeForm, status }>
 *
 * Status strings (must match the backend):
 *   'pending' | 'in_progress' | 'completed'
 */
export default function TodoPanel({ todos }) {
  if (!todos || !todos.length) return null

  const inProgress = todos.find((t) => t.status === 'in_progress')
  const completed = todos.filter((t) => t.status === 'completed').length

  return (
    <details
      // Open by default on desktop so the user sees the full plan; the
      // summary line keeps mobile tidy.
      open
      className="border-t border-border bg-muted/30 px-3 py-2 text-xs md:px-4"
    >
      <summary className="flex cursor-pointer select-none items-center gap-2">
        <ListChecks className="size-3.5 text-muted-foreground" />
        <span className="font-medium text-muted-foreground">
          Task list
        </span>
        <span className="text-muted-foreground/70">
          · {completed}/{todos.length} done
          {inProgress ? ` · now: ${inProgress.activeForm}` : ''}
        </span>
      </summary>

      <ul className="mt-2 space-y-1">
        {todos.map((t, i) => (
          <li key={i} className="flex items-start gap-2">
            <TodoStatusIcon status={t.status} />
            <span
              className={cn(
                'min-w-0 flex-1 break-words',
                t.status === 'completed' && 'text-muted-foreground line-through',
                t.status === 'in_progress' && 'text-foreground',
              )}
            >
              {t.status === 'in_progress' ? t.activeForm : t.content}
            </span>
          </li>
        ))}
      </ul>
    </details>
  )
}

/** Tiny status icon for a single todo row. */
function TodoStatusIcon({ status }) {
  if (status === 'completed') {
    return <CheckCircle2 className="mt-0.5 size-3.5 shrink-0 text-emerald-400" />
  }
  if (status === 'in_progress') {
    return <Loader2 className="mt-0.5 size-3.5 shrink-0 animate-spin text-primary" />
  }
  return <Circle className="mt-0.5 size-3.5 shrink-0 text-muted-foreground/50" />
}
