import React, { lazy, memo, Suspense, useMemo, useState } from 'react'
// react-diff-viewer-continued is ~50KB gzipped and only needed when the
// user expands a pending write_file / edit_file approval card. Load it on
// demand so it doesn't ship in the main bundle. We wrap the default export
// in a small component that reads `DiffMethod` from the same module — this
// keeps both the component and the enum inside the lazy chunk instead of
// pulling the enum into the main bundle.
const LazyDiffViewer = lazy(() =>
  import('react-diff-viewer-continued').then((mod) => ({
    default: (props) => (
      <mod.default compareMethod={mod.DiffMethod.WORDS} {...props} />
    ),
  })),
)
import {
  Terminal,
  TerminalSquare,
  FileText,
  FilePlus2,
  FilePen,
  FolderTree,
  Wrench,
  ChevronDown,
  ChevronRight,
  Check,
  X as XIcon,
  Loader2,
  ShieldAlert,
  Camera,
  MousePointer2,
  Keyboard,
  ArrowUpDown,
  Move,
  Search,
  Globe,
  Clipboard,
  ClipboardCopy,
  FileSearch,
  Files,
  Play,
  Square,
  ListChecks,
  Workflow,
  Info,
  Columns,
  AlignLeft,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { cn } from '@/lib/utils'

/**
 * Map a tool name to an icon component. Kept alongside the component that uses it.
 * Covers every tool the backend exposes in backend/prompts.py:TOOL_SCHEMAS.
 */
const TOOL_ICONS = {
  bash: Terminal,
  bash_bg: TerminalSquare,
  bash_output: Terminal,
  kill_shell: Square,
  read_file: FileText,
  write_file: FilePlus2,
  edit_file: FilePen,
  list_dir: FolderTree,
  glob: Files,
  grep: FileSearch,
  clipboard_read: Clipboard,
  clipboard_write: ClipboardCopy,
  screenshot: Camera,
  computer_click: MousePointer2,
  computer_type: Keyboard,
  computer_key: Keyboard,
  computer_scroll: ArrowUpDown,
  computer_mouse_move: Move,
  web_search: Search,
  fetch_url: Globe,
  todo_write: ListChecks,
  delegate: Workflow,
}

/**
 * ToolCall — collapsible card showing a single agent tool invocation.
 *
 * States (driven by props):
 *   - awaiting approval  → auto-expanded + shows Approve / Reject buttons
 *                          + shows the reason, full args, and (for file-
 *                          writing tools) a unified diff of the pending
 *                          change so the user can decide with full context.
 *   - running            → spinner + "Running..."
 *   - done (ok)          → green check + expandable output
 *   - done (error)       → red X + error text
 *   - rejected           → muted "Rejected by user"
 *
 * Props:
 *   - call: {id, name, args, label, reason?, preview?}
 *   - status: 'await' | 'running' | 'done' | 'rejected'
 *   - result: optional {ok, output, error}
 *   - imagePath: optional string — if set, the tool produced a screenshot
 *                (filename only; served by /api/screenshots/<name>).
 *   - onApprove / onReject: called when manual-mode buttons are clicked
 */
function ToolCall({
  call,
  status,
  result,
  imagePath,
  subagents,
  onApprove,
  onReject,
}) {
  // Default-open policy: any card that needs the user's decision should
  // show its details without an extra click. Computer-use tools stay open
  // so the user can watch the model drive their screen. Delegate tools
  // with in-flight subagents also open so the user sees live progress.
  const isComputerUse =
    call.name === 'screenshot' || call.name?.startsWith('computer_')
  const isDelegate =
    call.name === 'delegate' || call.name === 'delegate_parallel'
  const hasSubagents =
    subagents && Object.keys(subagents).length > 0
  const [open, setOpen] = useState(
    status === 'await' || isComputerUse || (isDelegate && hasSubagents),
  )
  const Icon = TOOL_ICONS[call.name] || Wrench

  // Separate the `reason` from the rest of args so the UI can surface it
  // prominently. The backend also emits `call.reason` on tool_call events,
  // which takes precedence if present — but for rows reconstructed from
  // history we fall back to pulling it out of args.
  const reason = (call.reason || call.args?.reason || '').trim()
  const argsWithoutReason = useMemo(() => {
    if (!call.args) return {}
    const { reason: _omit, ...rest } = call.args
    return rest
  }, [call.args])

  const header = (() => {
    if (status === 'await')
      return (
        <span className="flex items-center gap-2 text-amber-400">
          <ShieldAlert className="size-4" />
          Awaiting approval
        </span>
      )
    if (status === 'running')
      return (
        <span className="flex items-center gap-2 text-muted-foreground">
          <Loader2 className="size-4 animate-spin" />
          Running…
        </span>
      )
    if (status === 'rejected')
      return (
        <span className="flex items-center gap-2 text-muted-foreground">
          <XIcon className="size-4" />
          Rejected
        </span>
      )
    if (result?.ok === false)
      return (
        <span className="flex items-center gap-2 text-destructive">
          <XIcon className="size-4" />
          Failed
        </span>
      )
    return (
      <span className="flex items-center gap-2 text-emerald-400">
        <Check className="size-4" />
        Done
      </span>
    )
  })()

  const isAwait = status === 'await'

  return (
    <div
      className={cn(
        'my-2 overflow-hidden rounded-md border bg-card',
        isAwait
          ? 'border-amber-500/60 ring-1 ring-amber-500/30 shadow-sm shadow-amber-500/10'
          : 'border-border',
      )}
    >
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-accent/40"
      >
        <Icon className="size-4 shrink-0 text-muted-foreground" />
        <span className="font-mono text-xs text-muted-foreground truncate flex-1">
          {call.label || call.name}
        </span>
        <div className="shrink-0">{header}</div>
        <span className="shrink-0 text-muted-foreground">
          {open ? <ChevronDown className="size-4" /> : <ChevronRight className="size-4" />}
        </span>
      </button>

      {open && (
        <div className="border-t border-border bg-background/60 p-3 text-xs">
          {reason && <ReasonBlock reason={reason} />}
          <ArgsBlock args={argsWithoutReason} toolName={call.name} />
          {isAwait && call.preview?.kind === 'diff' && (
            <DiffBlock preview={call.preview} />
          )}
          {subagents && Object.keys(subagents).length > 0 && (
            <SubagentProgressBlock subagents={subagents} />
          )}
          {result && <ResultBlock result={result} />}
          {imagePath && <ScreenshotBlock name={imagePath} />}
        </div>
      )}

      {isAwait && (
        <div className="flex flex-col gap-2 border-t border-border bg-amber-500/5 px-3 py-2 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-[11px] text-amber-300/90">
            Review the command and reason above, then approve or reject.
          </p>
          <div className="flex items-center justify-end gap-2">
            <Button variant="outline" size="sm" onClick={onReject}>
              Reject
            </Button>
            <Button size="sm" onClick={onApprove}>
              Approve & run
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

/**
 * ReasonBlock — highlighted explanation from the model.
 * This is what the user reads first before deciding to approve.
 */
function ReasonBlock({ reason }) {
  return (
    <div className="mb-2 flex gap-2 rounded-md border border-amber-500/20 bg-amber-500/5 p-2">
      <Info className="mt-0.5 size-3.5 shrink-0 text-amber-400" />
      <div className="min-w-0">
        <p className="text-[10px] uppercase tracking-wide text-amber-400/80">
          Reason
        </p>
        <p className="whitespace-pre-wrap break-words text-[12px] text-foreground">
          {reason}
        </p>
      </div>
    </div>
  )
}

/**
 * ArgsBlock — render tool arguments as labeled key/value rows.
 *
 * For bash/bash_bg the command is emphasised as a full-width code block so
 * the user can see exactly what will execute. For other tools, args are
 * rendered in a compact key:value grid.
 */
function ArgsBlock({ args, toolName }) {
  if (!args || !Object.keys(args).length) {
    return <p className="text-muted-foreground">(no arguments)</p>
  }
  // Prominent command preview for shell tools.
  if ((toolName === 'bash' || toolName === 'bash_bg') && args.command) {
    const rest = Object.entries(args).filter(([k]) => k !== 'command')
    return (
      <div className="space-y-2">
        <div>
          <p className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">
            Command
          </p>
          <pre className="whitespace-pre-wrap break-all rounded border border-border bg-black/40 p-2 font-mono text-[12px] leading-snug">
            {args.command}
          </pre>
        </div>
        {rest.length > 0 && <KeyValueRows entries={rest} />}
      </div>
    )
  }
  return <KeyValueRows entries={Object.entries(args)} />
}

/** Small helper to render a flat list of key/value argument rows. */
function KeyValueRows({ entries }) {
  return (
    <div className="space-y-1">
      {entries.map(([k, v]) => (
        <div key={k} className="flex gap-2">
          <span className="shrink-0 text-muted-foreground">{k}:</span>
          <pre className="whitespace-pre-wrap break-all font-mono text-[11px]">
            {typeof v === 'string' ? v : JSON.stringify(v, null, 2)}
          </pre>
        </div>
      ))}
    </div>
  )
}

/**
 * Tailwind's `dark` theme tokens mapped to react-diff-viewer's style slots.
 * Keeps the diff viewer visually consistent with the rest of the app.
 */
const DIFF_DARK_STYLES = {
  variables: {
    dark: {
      diffViewerBackground: 'transparent',
      diffViewerColor: 'hsl(var(--foreground))',
      addedBackground: 'rgba(16, 185, 129, 0.12)',
      addedColor: '#a7f3d0',
      removedBackground: 'rgba(244, 63, 94, 0.12)',
      removedColor: '#fecaca',
      wordAddedBackground: 'rgba(16, 185, 129, 0.28)',
      wordRemovedBackground: 'rgba(244, 63, 94, 0.28)',
      addedGutterBackground: 'rgba(16, 185, 129, 0.18)',
      removedGutterBackground: 'rgba(244, 63, 94, 0.18)',
      gutterBackground: 'rgba(255,255,255,0.02)',
      gutterBackgroundDark: 'rgba(255,255,255,0.02)',
      highlightBackground: 'rgba(250, 204, 21, 0.08)',
      highlightGutterBackground: 'rgba(250, 204, 21, 0.12)',
      codeFoldGutterBackground: 'rgba(255,255,255,0.04)',
      codeFoldBackground: 'rgba(255,255,255,0.03)',
      emptyLineBackground: 'transparent',
      gutterColor: 'rgba(255,255,255,0.35)',
      addedGutterColor: '#a7f3d0',
      removedGutterColor: '#fecaca',
      codeFoldContentColor: 'rgba(255,255,255,0.5)',
      diffViewerTitleBackground: 'transparent',
      diffViewerTitleColor: 'hsl(var(--muted-foreground))',
      diffViewerTitleBorderColor: 'hsl(var(--border))',
    },
  },
  contentText: {
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: '11px',
    lineHeight: '1.5',
  },
  gutter: {
    minWidth: '28px',
    padding: '0 6px',
  },
  line: {
    padding: 0,
  },
}

/**
 * DiffBlock — render a pending file change.
 *
 * When the backend supplies `before` and `after` strings (new preview shape)
 * we show a rich react-diff-viewer with split/unified toggle and syntax
 * highlighting. Falls back to the plain unified-diff renderer for history
 * rows that pre-date the richer payload.
 */
function DiffBlock({ preview }) {
  const hasSplit =
    typeof preview.before === 'string' && typeof preview.after === 'string'
  // Default to split view on wide screens; users can toggle to unified for a
  // narrower diff that's easier to scan on mobile.
  const [splitView, setSplitView] = useState(true)

  return (
    <div className="mt-2 overflow-hidden rounded border border-border bg-black/30">
      <div className="flex items-center justify-between gap-2 border-b border-border px-2 py-1 text-[10px] uppercase tracking-wide text-muted-foreground">
        <div className="flex min-w-0 items-center gap-2">
          <span>Pending change</span>
          {preview.path && (
            <span className="truncate font-mono normal-case text-muted-foreground/80">
              {preview.path}
            </span>
          )}
          {preview.truncated && (
            <span
              className="shrink-0 rounded bg-amber-500/20 px-1 py-[1px] text-[9px] text-amber-300"
              title="File is larger than the 200 KB preview cap; the diff is still accurate but long sections may be clipped"
            >
              truncated
            </span>
          )}
        </div>
        {hasSplit && (
          <button
            type="button"
            onClick={() => setSplitView((v) => !v)}
            className="flex shrink-0 items-center gap-1 rounded border border-border bg-background/40 px-1.5 py-[2px] text-[10px] normal-case text-muted-foreground hover:text-foreground"
            title={splitView ? 'Switch to unified view' : 'Switch to split view'}
          >
            {splitView ? (
              <>
                <AlignLeft className="size-3" /> Unified
              </>
            ) : (
              <>
                <Columns className="size-3" /> Split
              </>
            )}
          </button>
        )}
      </div>

      {preview.note && (
        <p className="border-b border-border bg-amber-500/5 px-2 py-1 text-[11px] text-amber-300">
          {preview.note}
        </p>
      )}

      {hasSplit ? (
        <div className="max-h-96 overflow-auto">
          <Suspense
            fallback={
              <div className="flex items-center justify-center gap-2 p-4 text-[11px] text-muted-foreground">
                <Loader2 className="size-3 animate-spin" />
                Loading diff viewer…
              </div>
            }
          >
            <LazyDiffViewer
              oldValue={preview.before}
              newValue={preview.after}
              splitView={splitView}
              useDarkTheme
              styles={DIFF_DARK_STYLES}
            />
          </Suspense>
        </div>
      ) : (
        <UnifiedFallback diff={preview.diff || ''} />
      )}
    </div>
  )
}

/**
 * Legacy unified-diff renderer. Kept for tool_call rows loaded from history
 * that don't carry `before`/`after` text in their preview payload.
 */
function UnifiedFallback({ diff }) {
  const lines = diff.split('\n')
  return (
    <pre className="max-h-96 overflow-auto p-2 font-mono text-[11px] leading-snug">
      {lines.map((ln, i) => (
        <span
          key={i}
          className={cn(
            'block whitespace-pre-wrap break-all',
            ln.startsWith('+') && !ln.startsWith('+++') && 'text-emerald-300',
            ln.startsWith('-') && !ln.startsWith('---') && 'text-rose-300',
            (ln.startsWith('@@') ||
              ln.startsWith('---') ||
              ln.startsWith('+++')) &&
              'text-muted-foreground',
          )}
        >
          {ln || '\u200b'}
        </span>
      ))}
    </pre>
  )
}

/** Render a tool's output or error. */
function ResultBlock({ result }) {
  return (
    <div
      className={cn(
        'mt-2 rounded border border-border p-2',
        result.ok === false ? 'bg-destructive/10' : 'bg-black/20',
      )}
    >
      {result.error && (
        <p className="mb-1 text-destructive">{result.error}</p>
      )}
      {result.output && (
        <pre className="whitespace-pre-wrap break-all font-mono text-[11px] leading-relaxed max-h-96 overflow-auto">
          {result.output}
        </pre>
      )}
      {!result.error && !result.output && (
        <p className="text-muted-foreground">(no output)</p>
      )}
    </div>
  )
}

/**
 * Inline screenshot thumbnail that expands to a full-screen zoom dialog
 * when clicked. Rendered only when a tool result carries `image_path`.
 *
 * The <img> element's `loading="lazy"` and explicit sizing keep the chat
 * performant even with dozens of screenshots in a single conversation.
 */
function ScreenshotBlock({ name }) {
  const src = `/api/screenshots/${name}`
  return (
    <div className="mt-2">
      <p className="mb-1 text-muted-foreground">screenshot</p>
      <Dialog>
        <DialogTrigger asChild>
          <button
            type="button"
            className="block overflow-hidden rounded border border-border bg-black/20 hover:border-primary/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            aria-label="Expand screenshot"
          >
            <img
              src={src}
              alt="Agent screenshot"
              loading="lazy"
              className="max-h-64 w-auto object-contain"
            />
          </button>
        </DialogTrigger>
        <DialogContent className="max-w-[95vw] p-2 sm:max-w-5xl">
          <DialogTitle className="sr-only">Screenshot</DialogTitle>
          <img
            src={src}
            alt="Agent screenshot (full size)"
            className="h-auto max-h-[85vh] w-full object-contain"
          />
        </DialogContent>
      </Dialog>
    </div>
  )
}

/**
 * SubagentProgressBlock — live per-subagent activity for a `delegate` /
 * `delegate_parallel` card.
 *
 * Each entry in `subagents` is keyed by the subagent_id and carries:
 *   { status, subagentType, task, steps: [{id, name, label, status, error}],
 *     stepCount?, summary?, error? }
 *
 * We render one collapsible block per subagent with:
 *   header   — "explorer • 4 steps • done" (or running / failed)
 *   task     — the prompt the parent gave this branch
 *   steps    — the tool calls the subagent made, with running/done/failed
 *              status dots. Updates live as events arrive.
 *   summary  — the subagent's final reply (only shown when done).
 */
function SubagentProgressBlock({ subagents }) {
  const entries = Object.entries(subagents)
  return (
    <div className="mt-3 space-y-2">
      <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
        Subagents ({entries.length})
      </p>
      {entries.map(([sid, sa]) => (
        <SubagentRow key={sid} subagent={sa} />
      ))}
    </div>
  )
}

function SubagentRow({ subagent }) {
  const steps = subagent.steps || []
  const running = subagent.status === 'running'
  const failed = subagent.status === 'failed'
  const badgeColor = running
    ? 'text-muted-foreground'
    : failed
      ? 'text-destructive'
      : 'text-emerald-400'
  const badge = running ? (
    <Loader2 className="size-3 animate-spin" />
  ) : failed ? (
    <XIcon className="size-3" />
  ) : (
    <Check className="size-3" />
  )
  const totalSteps = subagent.stepCount ?? steps.length
  return (
    <div className="rounded border border-border bg-background/40 p-2">
      <div className="flex items-center gap-2 text-[11px]">
        <span className={cn('flex items-center gap-1', badgeColor)}>
          {badge}
          <span className="font-medium">
            {subagent.subagentType || 'general'}
          </span>
        </span>
        <span className="text-muted-foreground">
          • {totalSteps} step{totalSteps === 1 ? '' : 's'}
        </span>
        <span className={cn('ml-auto', badgeColor)}>
          {running ? 'running…' : failed ? 'failed' : 'done'}
        </span>
      </div>
      {subagent.task && (
        <p className="mt-1 whitespace-pre-wrap break-words text-[11px] text-muted-foreground/90 line-clamp-3">
          {subagent.task}
        </p>
      )}
      {steps.length > 0 && (
        <ul className="mt-2 space-y-1">
          {steps.map((s, i) => (
            <li
              key={s.id || i}
              className="flex items-center gap-2 text-[11px] font-mono"
            >
              {s.status === 'running' ? (
                <Loader2 className="size-3 shrink-0 animate-spin text-muted-foreground" />
              ) : s.status === 'failed' ? (
                <XIcon className="size-3 shrink-0 text-destructive" />
              ) : (
                <Check className="size-3 shrink-0 text-emerald-400" />
              )}
              <span className="truncate text-muted-foreground">
                {s.label || s.name}
              </span>
              {s.error && (
                <span className="shrink-0 text-destructive/80">
                  — {String(s.error).slice(0, 60)}
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
      {subagent.summary && (
        <p className="mt-2 whitespace-pre-wrap break-words text-[11px] text-foreground/80 line-clamp-4">
          {subagent.summary}
        </p>
      )}
      {failed && subagent.error && (
        <p className="mt-2 text-[11px] text-destructive">{subagent.error}</p>
      )}
    </div>
  )
}

/**
 * Wrap the default export in React.memo so a card that hasn't changed doesn't
 * re-render when a sibling tool call updates. ChatView mounts one <ToolCall>
 * per call id; as streaming progresses and `toolStates` mutates, only the
 * rows whose props actually change should re-render. Shallow prop equality is
 * sufficient because ChatView rebuilds object references (call, result,
 * subagents) only when their underlying fields change.
 */
export default memo(ToolCall)
