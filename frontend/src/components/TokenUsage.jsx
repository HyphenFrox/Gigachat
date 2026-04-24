import React, { useEffect, useMemo, useState } from 'react'
import { Gauge, Wallet } from 'lucide-react'
import { cn } from '@/lib/utils'
import { api } from '@/lib/api'

// CHARS_PER_TOKEN and COMPACTION_THRESHOLD still mirror backend/agent.py,
// but NUM_CTX is now fetched from /api/system/config at mount — the backend
// auto-tunes it based on the host's RAM/VRAM, so hardcoding here would just
// drift out of sync. Until the fetch completes we fall back to a safe 16K
// default so the gauge still renders.
const CHARS_PER_TOKEN = 4
const COMPACTION_THRESHOLD = 0.75
const DEFAULT_NUM_CTX = 16384

// Promise is shared across component instances so multiple mounts (e.g.
// tab re-render) don't each hit the endpoint — one request, one cache.
let _configPromise = null
function loadSystemConfig() {
  if (!_configPromise) {
    _configPromise = api.getSystemConfig().catch(() => ({ num_ctx: DEFAULT_NUM_CTX }))
  }
  return _configPromise
}

/**
 * TokenUsage — header chips showing context-window fill AND (when set) the
 * per-conversation soft budget.
 *
 * Two gauges:
 *   1. Compaction gauge (always shown): tokens vs. num_ctx — tells the user
 *      how close they are to auto-compaction.
 *   2. Budget gauge (only shown if conv has budget_turns or budget_tokens):
 *      whichever budget is closer to its cap drives the percentage. Hover
 *      tooltip breaks down both numbers.
 *
 * The compaction gauge is estimated client-side from the messages prop
 * (cheap, updates on every token). The budget gauge pulls a server-side
 * snapshot from /api/conversations/:id/usage so it matches what the backend
 * will actually compare against when deciding whether to refuse a turn.
 *
 * Props:
 *   - messages: array of {content, tool_calls, images}. Same shape as the
 *               ChatView passes around.
 *   - conv:     optional current conversation. Required for the budget
 *               gauge; if omitted, only the compaction gauge renders.
 */
export default function TokenUsage({ messages, conv }) {
  // numCtx is null while we're waiting on /api/system/config; fall back to
  // DEFAULT_NUM_CTX during that short window so the gauge doesn't flicker.
  const [config, setConfig] = useState(null)
  useEffect(() => {
    let alive = true
    loadSystemConfig().then((c) => {
      if (alive) setConfig(c)
    })
    return () => {
      alive = false
    }
  }, [])

  const numCtx = config?.num_ctx || DEFAULT_NUM_CTX

  const { tokens, pct, level } = useMemo(() => {
    let chars = 0
    for (const m of messages || []) {
      chars += (m.content || '').length
      // Tool-call JSON roughly doubles the cost of the parent assistant msg.
      if (m.tool_calls?.length) chars += JSON.stringify(m.tool_calls).length
      // User-attached images are cheap in tokens when rescaled but the
      // base64 payload dominates — treat each as ~1000 effective chars.
      if (m.images?.length) chars += m.images.length * 1000
    }
    const t = Math.floor(chars / CHARS_PER_TOKEN)
    const p = Math.min(100, Math.floor((t / numCtx) * 100))
    let lvl = 'ok'
    if (p >= 100) lvl = 'full'
    else if (p >= COMPACTION_THRESHOLD * 100) lvl = 'warn'
    else if (p >= 60) lvl = 'mid'
    return { tokens: t, pct: p, level: lvl }
  }, [messages, numCtx])

  // Hover tooltip includes the detected hardware so the user can see *why*
  // they got a particular context window (e.g. "8 GB VRAM RTX 3060 Ti").
  const hostLine = config?.gpu_name
    ? `${config.gpu_name} · ${config.vram_gb} GB VRAM · ${config.ram_gb} GB RAM`
    : config
      ? `${config.ram_gb} GB RAM (CPU only)`
      : ''
  const title =
    `~${tokens.toLocaleString()} tokens of ${numCtx.toLocaleString()} used. ` +
    `Auto-compaction kicks in around ${Math.round(COMPACTION_THRESHOLD * 100)}%.` +
    (hostLine ? `\n${hostLine}` : '')

  return (
    <>
      <div
        className={cn(
          'hidden items-center gap-1.5 rounded-md border border-border px-2 py-1 text-[10px] font-mono lg:flex',
          level === 'ok' && 'text-muted-foreground',
          level === 'mid' && 'text-amber-300/90',
          level === 'warn' && 'text-amber-400',
          level === 'full' && 'text-destructive',
        )}
        title={title}
      >
        <Gauge className="size-3" />
        <span>{pct}%</span>
        <span className="hidden xl:inline text-muted-foreground">
          ({tokens.toLocaleString()}/{numCtx.toLocaleString()})
        </span>
      </div>
      {conv && (conv.budget_turns || conv.budget_tokens) ? (
        <BudgetChip conv={conv} messages={messages} />
      ) : null}
    </>
  )
}

/**
 * BudgetChip — renders the conversation's soft budget as a second gauge.
 *
 * Re-fetches /usage on every message change so the numbers stay in sync with
 * the transcript. We invalidate on `messages.length` (cheap) rather than on
 * the full `messages` array (would refetch on every streaming token).
 */
function BudgetChip({ conv, messages }) {
  const [usage, setUsage] = useState(null)
  useEffect(() => {
    let alive = true
    api
      .getConversationUsage(conv.id)
      .then((u) => {
        if (alive) setUsage(u)
      })
      .catch(() => {
        /* non-fatal — chip just won't populate this tick */
      })
    return () => {
      alive = false
    }
    // Only refresh when the transcript grows or the conversation/id changes
    // or the user edited the budget caps. Streaming tokens don't qualify.
  }, [conv.id, conv.budget_turns, conv.budget_tokens, messages?.length])

  const { pct, level, tooltip } = useMemo(() => {
    if (!usage) return { pct: 0, level: 'ok', tooltip: 'loading…' }
    const parts = []
    let worst = 0
    if (conv.budget_turns) {
      const p = Math.min(
        100,
        Math.floor((usage.assistant_turns / conv.budget_turns) * 100),
      )
      worst = Math.max(worst, p)
      parts.push(
        `Turns: ${usage.assistant_turns.toLocaleString()} / ${conv.budget_turns.toLocaleString()} (${p}%)`,
      )
    }
    if (conv.budget_tokens) {
      const p = Math.min(
        100,
        Math.floor((usage.tokens_estimate / conv.budget_tokens) * 100),
      )
      worst = Math.max(worst, p)
      parts.push(
        `Tokens: ~${usage.tokens_estimate.toLocaleString()} / ${conv.budget_tokens.toLocaleString()} (${p}%)`,
      )
    }
    let lvl = 'ok'
    if (worst >= 100) lvl = 'full'
    else if (worst >= 80) lvl = 'warn'
    else if (worst >= 60) lvl = 'mid'
    return {
      pct: worst,
      level: lvl,
      tooltip:
        `Conversation budget — the agent refuses to start a new turn once ` +
        `either cap is hit.\n` +
        parts.join('\n'),
    }
  }, [usage, conv.budget_turns, conv.budget_tokens])

  return (
    <div
      className={cn(
        'hidden items-center gap-1.5 rounded-md border border-border px-2 py-1 text-[10px] font-mono lg:flex',
        level === 'ok' && 'text-muted-foreground',
        level === 'mid' && 'text-amber-300/90',
        level === 'warn' && 'text-amber-400',
        level === 'full' && 'text-destructive',
      )}
      title={tooltip}
    >
      <Wallet className="size-3" />
      <span>{pct}%</span>
    </div>
  )
}
