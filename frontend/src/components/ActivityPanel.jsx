import React, { useMemo } from 'react'
import {
  Activity,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Sparkles,
} from 'lucide-react'
import { cn } from '@/lib/utils'

/**
 * ActivityPanel — right-side strip inside ChatView showing what the agent
 * is doing right now plus a small history of the most recent tool calls.
 *
 * The panel reads from ChatView's existing `toolStates` map and `messages`
 * array — no new state lifting required. The intent is to give the user a
 * persistent at-a-glance view of *what's happening* without having to scroll
 * through the transcript: which tool is in flight, what its arguments are,
 * and a quick log of recent calls and their outcomes.
 *
 * Three top-level visual states:
 *   1. **Awaiting approval** — amber accent, "needs your input" framing.
 *      Distinct from "running" because the user is the blocker, not the agent.
 *   2. **Running** — green pulsing accent, current tool name + reason.
 *   3. **Idle (or thinking)** — muted, "Generating reply…" if the assistant
 *      is mid-stream, otherwise a quiet "Idle" with the last completed tool.
 *
 * Below the active section we render a compact "Recent" list — last few tool
 * calls with a status icon and label so the user can quickly review the
 * trajectory of the turn without scrolling the chat back up.
 *
 * Props:
 *   - toolStates: { [callId]: { status, name, label, reason, args, result, ... } }
 *   - messages: full conversation history (used to recover ordered tool ids)
 *   - busy: whether a turn is currently streaming
 *   - liveContent: the assistant's in-flight text (used to detect "thinking")
 *   - liveThinking: optional reasoning tokens, same purpose
 */
export default function ActivityPanel({
  toolStates,
  messages,
  busy,
  liveContent,
  liveThinking,
}) {
  // Recover the ordered list of tool calls from the conversation history.
  // `toolStates` is a plain object keyed by call id and doesn't preserve
  // chronological order on its own; pulling from `messages` gives us a
  // dependable timeline we can use both to find the active tool and to
  // build the "recent" history list.
  const orderedCalls = useMemo(() => {
    const out = []
    for (const m of messages) {
      if (m.role !== 'assistant' || !m.tool_calls?.length) continue
      for (const tc of m.tool_calls) {
        out.push({ id: tc.id, name: tc.name, label: tc.label })
      }
    }
    return out
  }, [messages])

  // The active call, if any: the most recent entry whose state is still
  // running or awaiting approval. We scan back-to-front so the *latest*
  // in-flight call wins (in practice only one is ever running at a time,
  // but being explicit guards against future parallel-tool changes).
  const active = useMemo(() => {
    for (let i = orderedCalls.length - 1; i >= 0; i--) {
      const tc = orderedCalls[i]
      const st = toolStates[tc.id]
      if (st && (st.status === 'running' || st.status === 'await')) {
        return { ...tc, ...st }
      }
    }
    return null
  }, [orderedCalls, toolStates])

  // Last few completed (or rejected) calls, newest first. Cap to 6 so the
  // panel doesn't get crowded — the full history is always available in
  // the transcript itself.
  const recent = useMemo(() => {
    const out = []
    for (let i = orderedCalls.length - 1; i >= 0 && out.length < 6; i--) {
      const tc = orderedCalls[i]
      const st = toolStates[tc.id]
      if (!st) continue
      if (st.status === 'done' || st.status === 'rejected') {
        out.push({ ...tc, ...st })
      }
    }
    return out
  }, [orderedCalls, toolStates])

  // "Thinking" state: agent is busy and streaming text but no tool is in
  // flight. Distinct from "Idle" so the user can see the model is alive
  // even when there's nothing tool-shaped to display.
  const thinking = busy && !active && (liveContent || liveThinking || true)

  return (
    <aside className="hidden h-full w-72 shrink-0 flex-col border-l border-border bg-card/40 lg:flex">
      <header className="flex items-center gap-2 border-b border-border px-4 py-3">
        <Activity className="size-4 text-muted-foreground" />
        <h2 className="text-sm font-medium tracking-tight">Activity</h2>
      </header>

      <div className="flex-1 overflow-y-auto p-3 text-xs">
        {/* ----- Active section ---------------------------------------- */}
        {active ? (
          <ActiveCard active={active} />
        ) : thinking ? (
          <ThinkingCard hasContent={!!liveContent} />
        ) : (
          <IdleCard />
        )}

        {/* ----- Recent section --------------------------------------- */}
        {recent.length > 0 && (
          <section className="mt-4">
            <h3 className="mb-2 px-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Recent
            </h3>
            <ul className="space-y-1">
              {recent.map((r) => (
                <RecentRow key={r.id} call={r} />
              ))}
            </ul>
          </section>
        )}
      </div>
    </aside>
  )
}

/** Card shown when a tool is actively running or awaiting approval. */
function ActiveCard({ active }) {
  const isAwait = active.status === 'await'
  return (
    <div
      className={cn(
        'rounded-md border px-3 py-3',
        isAwait
          ? 'border-amber-500/40 bg-amber-500/5'
          : 'border-emerald-500/40 bg-emerald-500/5',
      )}
    >
      <div className="flex items-center gap-2">
        {isAwait ? (
          <AlertCircle className="size-4 shrink-0 text-amber-500" />
        ) : (
          <Loader2 className="size-4 shrink-0 animate-spin text-emerald-500" />
        )}
        <span
          className={cn(
            'text-[11px] font-semibold uppercase tracking-wider',
            isAwait ? 'text-amber-600 dark:text-amber-400' : 'text-emerald-600 dark:text-emerald-400',
          )}
        >
          {isAwait ? 'Awaiting approval' : 'Running'}
        </span>
      </div>
      <div className="mt-2 break-words font-mono text-[11px] text-foreground">
        {active.label || active.name}
      </div>
      {active.reason && (
        <p className="mt-2 text-[11px] leading-snug text-muted-foreground">
          {active.reason}
        </p>
      )}
      {active.args && Object.keys(active.args).length > 0 && (
        <ArgsSummary args={active.args} />
      )}
    </div>
  )
}

/** Card shown when the agent is busy but no tool is in flight (drafting text). */
function ThinkingCard({ hasContent }) {
  return (
    <div className="rounded-md border border-primary/30 bg-primary/5 px-3 py-3">
      <div className="flex items-center gap-2">
        <Sparkles className="size-4 shrink-0 animate-pulse text-primary" />
        <span className="text-[11px] font-semibold uppercase tracking-wider text-primary">
          {hasContent ? 'Generating reply' : 'Thinking'}
        </span>
      </div>
      <p className="mt-2 text-[11px] leading-snug text-muted-foreground">
        The agent is preparing its next step. Tool calls (if any) will appear
        here as they start.
      </p>
    </div>
  )
}

/** Card shown when nothing is happening. */
function IdleCard() {
  return (
    <div className="rounded-md border border-border/60 bg-muted/20 px-3 py-3">
      <div className="flex items-center gap-2">
        <span className="size-2 shrink-0 rounded-full bg-muted-foreground/40" />
        <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          Idle
        </span>
      </div>
      <p className="mt-2 text-[11px] leading-snug text-muted-foreground">
        Send a message to start the agent.
      </p>
    </div>
  )
}

/** One row in the "Recent" list. */
function RecentRow({ call }) {
  const ok = call.status === 'done' && call.result?.ok !== false
  const Icon = call.status === 'rejected'
    ? XCircle
    : ok
      ? CheckCircle2
      : XCircle
  const iconColor = call.status === 'rejected'
    ? 'text-amber-500'
    : ok
      ? 'text-emerald-500'
      : 'text-destructive'

  return (
    <li className="flex items-start gap-2 rounded-md px-2 py-1.5 hover:bg-accent/50">
      <Icon className={cn('mt-0.5 size-3.5 shrink-0', iconColor)} />
      <div className="min-w-0 flex-1">
        <div className="break-words font-mono text-[11px]">
          {call.label || call.name}
        </div>
        {call.status === 'rejected' && (
          <div className="text-[10px] text-amber-500/80">rejected</div>
        )}
        {call.status === 'done' && call.result?.error && (
          <div className="truncate text-[10px] text-destructive/80">
            {call.result.error}
          </div>
        )}
      </div>
    </li>
  )
}

/**
 * Compact rendering of a tool call's args. We show up to 3 keys, truncate
 * each value to ~60 chars, and pretty-print arrays/objects via JSON.
 * Goal: enough visibility to recognise *what* the tool is doing without
 * dominating the panel width.
 */
function ArgsSummary({ args }) {
  // Strip the `reason` arg — every tool schema includes one and we already
  // render it separately above; duplicating it just wastes space.
  const entries = Object.entries(args).filter(([k]) => k !== 'reason').slice(0, 3)
  if (entries.length === 0) return null
  return (
    <div className="mt-2 space-y-1 border-t border-border/60 pt-2">
      {entries.map(([k, v]) => (
        <div key={k} className="font-mono text-[10px]">
          <span className="text-muted-foreground">{k}:</span>{' '}
          <span className="break-all text-foreground/80">
            {formatArgValue(v)}
          </span>
        </div>
      ))}
    </div>
  )
}

/** Render a single arg value as a short string for the args summary. */
function formatArgValue(v) {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'string') {
    return v.length > 60 ? v.slice(0, 60) + '…' : v
  }
  if (typeof v === 'number' || typeof v === 'boolean') return String(v)
  // Objects / arrays — JSON-stringify with a reasonable cap.
  try {
    const s = JSON.stringify(v)
    return s.length > 60 ? s.slice(0, 60) + '…' : s
  } catch {
    return '[unserialisable]'
  }
}
