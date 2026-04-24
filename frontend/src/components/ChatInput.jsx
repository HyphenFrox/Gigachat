import React, { useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import {
  SendHorizonal,
  StopCircle,
  Paperclip,
  X as XIcon,
  Image as ImageIcon,
  FileText,
  Plus,
  File as FileIcon,
  AtSign,
  AlertTriangle,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'
import { api } from '@/lib/api'
import {
  VoiceDictateButton,
  VoiceModeToggle,
  VoiceModeDriver,
} from './VoiceControls'
import ModelPicker from './ModelPicker'

/**
 * Slash-command shortcuts.
 *
 * These are expansion templates, not server-side commands — typing `/plan`
 * pushes the associated phrase into the composer so the user doesn't have
 * to re-type common instructions. The template may contain `$1` marking
 * where the cursor should land after expansion.
 */
const SLASH_COMMANDS = [
  {
    name: '/plan',
    description: 'Ask the model to draft a plan with todo_write before acting.',
    template:
      'Before doing anything, use the todo_write tool to draft a short plan for: $1',
  },
  {
    name: '/bug',
    description: 'Report a bug and ask for a fix with tests.',
    template:
      'I have a bug: $1\n\nInvestigate the root cause, fix it, and run the relevant tests to confirm.',
  },
  {
    name: '/review',
    description: 'Ask for a critical review of recent changes.',
    template:
      'Please review my recent changes in this repo. Check: $1 Report findings clearly.',
  },
  {
    name: '/explain',
    description: 'Ask for a deep-dive explanation of a file or symbol.',
    template: 'Read and explain how $1 works. Walk me through the key parts.',
  },
  {
    name: '/search',
    description: 'Web-search and summarise with citations.',
    template:
      'Use web_search and fetch_url to research: $1\n\nGive me a short summary with links to the sources you used.',
  },
  {
    name: '/desktop',
    description: 'Control the desktop to accomplish a task.',
    template:
      'Take a screenshot first, then help me do this on my desktop: $1',
  },
]

/**
 * ChatInput — composer at the bottom of the chat view.
 *
 * Features:
 *   - Auto-resizing textarea (caps at ~12 lines to avoid eating the viewport).
 *   - Enter to send, Shift+Enter for newline.
 *   - Stays editable while the agent is streaming — pressing Enter (or the
 *     queue button) appends the new message to the in-flight turn's input
 *     queue rather than starting a parallel run. The Stop button is shown
 *     alongside so the user can still interrupt.
 *   - Drag-drop / paste of image OR document files (pdf/txt/md/csv) — routed
 *     to the parent via onImages (kept as the historic prop name; it handles
 *     both kinds internally).
 *   - Pending image thumbnails shown above the field with a remove button.
 *   - Slash-command autocomplete: typing `/` at the start of the composer
 *     surfaces a menu of template expansions.
 */
export default function ChatInput({
  value,
  onChange,
  onSend,
  busy,
  onStop,
  pendingImages = [],
  onImages,
  onRemoveImage,
  // Model-picker wiring — lives next to the composer (not the header) so the
  // "what model answers the next message?" decision is right where the user
  // is looking. Optional: when omitted, the picker row is hidden entirely.
  conv,
  models,
  showAllModels,
  onToggleShowAllModels,
  onPatch,
}) {
  const ref = useRef(null)
  const fileInputRef = useRef(null)
  const [isDragOver, setIsDragOver] = useState(false)

  // Voice-mode state. When ON, a VoiceModeDriver is mounted that streams
  // transcribed speech into the composer and auto-submits after a ~1.4s
  // silence. The push-to-talk dictate button is disabled while mode is on
  // so the two features don't fight over the mic.
  const [voiceMode, setVoiceMode] = useState(false)
  // Interim transcript surfaced from the recognizer — rendered as a dim chip
  // above the composer so the user sees what's being heard before it lands.
  const [interimTranscript, setInterimTranscript] = useState('')

  // Whether the Send button is enabled (truthy text or at least one image).
  // Mirrored in keypress handler so Enter behaves the same as the button.
  const canSubmit = (value || '').trim().length > 0 || pendingImages.length > 0

  // Auto-size: reset height, then grow up to a max (px).
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 260) + 'px'
  }, [value])

  // Slash-command popup state. The popup appears whenever the user types
  // something that starts with `/` and doesn't yet contain whitespace/newline
  // — i.e. they're still typing the command name.
  const slashQuery = useMemo(() => {
    if (!value || !value.startsWith('/')) return null
    const firstSpace = value.search(/[\s\n]/)
    if (firstSpace === -1) return value
    return null
  }, [value])
  const slashMatches = useMemo(() => {
    if (!slashQuery) return []
    const q = slashQuery.toLowerCase()
    return SLASH_COMMANDS.filter(
      (c) =>
        c.name.toLowerCase().startsWith(q) ||
        c.description.toLowerCase().includes(q.slice(1)),
    )
  }, [slashQuery])

  function expandSlash(cmd) {
    const template = cmd.template
    // If the template carries a $1 placeholder we leave the cursor there;
    // otherwise we just append a trailing space so the user types right
    // after the expansion.
    const marker = '$1'
    let next = template.includes(marker) ? template : template + ' '
    next = next.replace(marker, '')
    onChange(next)
    // Refocus and push the caret to where $1 was (simple heuristic: end).
    requestAnimationFrame(() => {
      const el = ref.current
      if (!el) return
      el.focus()
      const caret = template.indexOf(marker)
      if (caret !== -1) {
        el.setSelectionRange(caret, caret)
      } else {
        const pos = next.length
        el.setSelectionRange(pos, pos)
      }
    })
  }

  // ----- @-mention autocomplete ----------------------------------------
  // Detect when the caret is inside an `@partial` token so we can surface
  // a file-picker menu. An `@` is eligible only when it's at the start of
  // the input or preceded by whitespace — that way literal `@` inside
  // email addresses or code doesn't trigger the menu.
  //
  // `mentionQuery` is null when no mention is active, or the substring
  // typed after `@` otherwise (empty string allowed — an empty query
  // surfaces the most recent files in the cwd).
  //
  // Menu highlights the first result; arrow keys / Enter navigate. On
  // pick the `@partial` token is replaced with `@<relative/path>` and the
  // caret moves past the inserted mention.
  const [mentionQuery, setMentionQuery] = useState(null)
  const [mentionStart, setMentionStart] = useState(-1)
  const [mentionResults, setMentionResults] = useState([])
  const [mentionActive, setMentionActive] = useState(0)

  // Re-evaluate mention state on every value change. We also re-run on
  // caret move via onSelect, because the user can escape a mention by
  // moving the caret with arrow keys.
  function updateMentionState() {
    const el = ref.current
    if (!el) return
    const cursor = el.selectionStart ?? (value || '').length
    const before = (value || '').slice(0, cursor)
    // Match an @-token touching the caret. We require the `@` to sit at
    // the start of the input or right after whitespace — literal `@`
    // inside code or emails shouldn't pop the menu.
    const m = before.match(/(?:^|[\s\n])(@[^\s@\n]*)$/)
    if (!m) {
      setMentionQuery(null)
      setMentionResults([])
      return
    }
    const token = m[1]
    const atPos = before.length - token.length
    setMentionStart(atPos)
    setMentionQuery(token.slice(1))
    setMentionActive(0)
  }

  // Debounced fetch of file suggestions whenever the query changes.
  useEffect(() => {
    if (mentionQuery === null || !conv?.id) {
      setMentionResults([])
      return
    }
    const id = setTimeout(async () => {
      try {
        const { files } = await api.searchConversationFiles(
          conv.id, mentionQuery, 10,
        )
        setMentionResults(files || [])
      } catch {
        // Swallow — a failed lookup shouldn't block the user's typing.
        setMentionResults([])
      }
    }, 120)
    return () => clearTimeout(id)
  }, [mentionQuery, conv?.id])

  function pickMention(file) {
    if (mentionStart < 0) return
    const tokenLen = 1 + (mentionQuery || '').length
    const before = (value || '').slice(0, mentionStart)
    const after = (value || '').slice(mentionStart + tokenLen)
    const insert = `@${file.rel_path}`
    const sep = after.startsWith(' ') || after.length === 0 ? '' : ' '
    const next = before + insert + sep + after
    onChange(next)
    setMentionQuery(null)
    setMentionResults([])
    const newCursor = before.length + insert.length + sep.length
    requestAnimationFrame(() => {
      const el = ref.current
      if (!el) return
      el.focus()
      el.setSelectionRange(newCursor, newCursor)
    })
  }

  function handleKey(e) {
    // Mention menu takes precedence over slash / send when open — the user
    // is clearly navigating the suggestions, not composing a message.
    if (mentionQuery !== null && mentionResults.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setMentionActive((i) => (i + 1) % mentionResults.length)
        return
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault()
        setMentionActive((i) =>
          i <= 0 ? mentionResults.length - 1 : i - 1,
        )
        return
      }
      if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault()
        pickMention(mentionResults[mentionActive])
        return
      }
      if (e.key === 'Escape') {
        e.preventDefault()
        setMentionQuery(null)
        setMentionResults([])
        return
      }
    }

    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      // If the slash menu is showing and there's exactly one match, Enter
      // expands it instead of sending an empty turn.
      if (slashMatches.length === 1 && value.startsWith('/')) {
        e.preventDefault()
        expandSlash(slashMatches[0])
        return
      }
      e.preventDefault()
      // Always call onSend — the parent decides whether to start a new turn
      // or queue against the running one, based on its `busy` state.
      if (canSubmit) onSend()
    }
    if (e.key === 'Escape' && slashMatches.length) {
      // Nothing to close explicitly; the menu disappears on whitespace.
    }
  }

  // ----- vision-capability probe ----------------------------------------
  // Fetch the active model's capabilities so we can warn the user before
  // they attach an image to a non-vision model (the agent would otherwise
  // silently drop the image). Result is cached per-model in the component
  // so switching back to a previously-seen model doesn't re-fetch.
  const [visionByModel, setVisionByModel] = useState({})
  const activeModel = conv?.model || ''
  const modelSupportsVision = activeModel ? visionByModel[activeModel] : null

  useEffect(() => {
    if (!activeModel) return
    if (activeModel in visionByModel) return
    let cancelled = false
    api
      .getModelCapabilities(activeModel)
      .then((caps) => {
        if (cancelled) return
        setVisionByModel((m) => ({ ...m, [activeModel]: !!caps?.vision }))
      })
      .catch(() => {
        // On error leave the entry unset so a retry on next attach can
        // try again — we don't want a single 404 to permanently hide the
        // warning.
      })
    return () => {
      cancelled = true
    }
  }, [activeModel, visionByModel])

  // True iff at least one of the currently-pending attachments is an image.
  // The parent tags documents with `kind === 'document'`; everything else
  // in `pendingImages` is a rendered image chip. Drives the composer
  // warning chip below.
  const hasPendingImages = useMemo(
    () => (pendingImages || []).some((att) => att && att.kind !== 'document'),
    [pendingImages],
  )
  const showVisionWarning =
    hasPendingImages && modelSupportsVision === false && activeModel

  // ----- attachment intake: paste / drop / file-picker --------------------
  // Accepts images AND documents. The server rejects anything outside its
  // allowlist with a 415, so we don't filter aggressively here — surfacing
  // the server error gives the user a better hint than a silent drop.
  function isAllowedAttachment(f) {
    const t = f.type || ''
    return (
      t.startsWith('image/') ||
      t === 'application/pdf' ||
      t === 'text/plain' ||
      t === 'text/markdown' ||
      t === 'text/csv'
    )
  }

  // Single intake point for every attachment path (paste / drop / picker).
  // Warns once via toast if the user is attaching an image to a non-vision
  // model — the backend strips images from non-vision model requests, so
  // surfacing this at attach time saves a confusing "model ignored my
  // screenshot" moment later.
  function emitAttachments(files) {
    if (!files || !files.length || !onImages) return
    const imageFiles = files.filter((f) => (f.type || '').startsWith('image/'))
    if (imageFiles.length && activeModel && modelSupportsVision === false) {
      toast.warning(
        `${activeModel} doesn't support images`,
        {
          description:
            'Switch to a vision-capable model (e.g. llama3.2-vision, gemma4) — otherwise the image will be dropped from the request.',
          duration: 6000,
        },
      )
    }
    onImages(files)
  }

  function handlePaste(e) {
    if (!onImages) return
    const items = e.clipboardData?.items
    if (!items) return
    const files = []
    for (const it of items) {
      if (it.kind === 'file') {
        const f = it.getAsFile()
        if (f && isAllowedAttachment(f)) files.push(f)
      }
    }
    if (files.length) {
      e.preventDefault()
      emitAttachments(files)
    }
  }

  function handleDrop(e) {
    e.preventDefault()
    setIsDragOver(false)
    if (!onImages) return
    const files = Array.from(e.dataTransfer?.files || []).filter(isAllowedAttachment)
    if (files.length) emitAttachments(files)
  }

  function handleFilePicker(e) {
    const files = Array.from(e.target.files || []).filter(isAllowedAttachment)
    if (files.length) emitAttachments(files)
    // Reset so picking the same file again still fires onChange.
    e.target.value = ''
  }

  return (
    <div
      className={cn(
        'border-t border-border bg-background p-3 md:p-4',
        isDragOver && 'ring-2 ring-primary/60',
      )}
      onDragOver={(e) => {
        e.preventDefault()
        if (e.dataTransfer.types?.includes('Files')) setIsDragOver(true)
      }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={handleDrop}
    >
      <div className="mx-auto flex max-w-3xl flex-col gap-2">
        {pendingImages.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {pendingImages.map((im) =>
              im.kind === 'document' ? (
                <PendingDocumentChip
                  key={im.name}
                  doc={im}
                  onRemove={() => onRemoveImage?.(im.name)}
                />
              ) : (
                <PendingImageChip
                  key={im.name}
                  image={im}
                  onRemove={() => onRemoveImage?.(im.name)}
                />
              ),
            )}
          </div>
        )}

        {/* Persistent warning chip — shown alongside the thumbnails whenever
            the active model can't consume images. The initial toast fires
            once on attach; this chip stays visible until the user either
            drops the attachments or switches to a vision-capable model. */}
        {showVisionWarning && (
          <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <div>
              <strong>{activeModel}</strong> is not a vision model. The
              attached image will be dropped before reaching the model —
              switch to a vision-capable model (e.g. llama3.2-vision,
              gemma4) to use it.
            </div>
          </div>
        )}

        {slashMatches.length > 0 && (
          <SlashMenu
            matches={slashMatches}
            onPick={expandSlash}
            query={slashQuery || ''}
          />
        )}

        {/* @-mention picker — shown when the caret is inside an @token.
            Filtering happens server-side; the UI just renders whatever the
            backend returns. Active row is highlighted via mentionActive. */}
        {mentionQuery !== null && mentionResults.length > 0 && (
          <MentionMenu
            matches={mentionResults}
            query={mentionQuery}
            active={mentionActive}
            onPick={pickMention}
            onHover={setMentionActive}
          />
        )}

        {/* Interim (non-final) transcript chip. Dimmed so the user sees what
            the engine is currently guessing without mistaking it for typed
            text. Disappears as soon as the phrase finalises or the mic stops. */}
        {interimTranscript && (
          <p className="italic text-xs text-muted-foreground">
            <span className="opacity-60">…</span> {interimTranscript}
          </p>
        )}

        {/* Model picker sits just above the composer so the user can verify
            (and swap) which model will handle the next turn. The change takes
            effect on the *next* message — any in-flight stream keeps running
            on whichever model started it. */}
        {conv && onPatch && (
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
              Model
            </span>
            <ModelPicker
              conv={conv}
              models={models || []}
              showAllModels={showAllModels}
              onToggleShowAllModels={onToggleShowAllModels}
              onPatch={onPatch}
            />
          </div>
        )}

        <div className="flex items-end gap-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => fileInputRef.current?.click()}
            title="Attach image or document"
          >
            <Paperclip />
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp,image/gif,application/pdf,text/plain,text/markdown,text/csv,.pdf,.txt,.md,.csv"
            multiple
            className="hidden"
            onChange={handleFilePicker}
          />

          {/* Push-to-talk dictation — appends transcribed speech into the
              composer. Disabled while continuous voice mode owns the mic. */}
          <VoiceDictateButton
            value={value}
            onChange={onChange}
            onInterim={setInterimTranscript}
            disabled={voiceMode}
          />

          {/* Continuous voice mode — transcribes speech AND auto-submits on
              silence. The toggle lives next to the dictate button; the driver
              is mounted (below) only while active. */}
          <VoiceModeToggle
            active={voiceMode}
            onToggle={setVoiceMode}
          />

          <Textarea
            ref={ref}
            value={value}
            onChange={(e) => {
              onChange(e.target.value)
              // Re-scan for a mention token after the value mutates. We do
              // this in a microtask so the textarea's selection already
              // reflects the new character.
              queueMicrotask(updateMentionState)
            }}
            onSelect={updateMentionState}
            onKeyUp={updateMentionState}
            onKeyDown={handleKey}
            onPaste={handlePaste}
            placeholder={
              voiceMode
                ? '🎙 Voice mode on — speak; pause to send automatically.'
                : busy
                  ? 'Queue another message…  (delivered after the current turn finishes its current step)'
                  : 'Message Gigachat…  (Shift+Enter for newline, / for commands, @ to mention a file, paste or drop an image)'
            }
            rows={1}
            className="max-h-[260px] text-sm"
          />
          {/* Send / Queue button — same handler either way; ChatView reads the
              busy flag to decide whether to start a turn or enqueue. The icon
              and tooltip swap so the user knows which behaviour they'll get. */}
          <Button
            size="icon"
            onClick={onSend}
            disabled={!canSubmit}
            title={busy ? 'Queue (Enter)' : 'Send (Enter)'}
            variant={busy ? 'secondary' : 'default'}
          >
            {busy ? <Plus /> : <SendHorizonal />}
          </Button>
          {/* Stop button is shown alongside (not in place of) the send button
              while the turn is running, so the user can interrupt without
              losing the ability to compose the next message. */}
          {busy && (
            <Button variant="destructive" size="icon" onClick={onStop} title="Stop">
              <StopCircle />
            </Button>
          )}
        </div>

        {/* Invisible driver that owns the mic while voice mode is on. We keep
            it outside any conditional that might re-mount it — the `active`
            prop gates mounting via the `&&` below. */}
        {voiceMode && (
          <VoiceModeDriver
            onInterim={setInterimTranscript}
            onFinalChunk={(phrase) => {
              setInterimTranscript('')
              // Same append logic as the push-to-talk button: leading space
              // when the composer isn't already terminated by whitespace.
              const current = value || ''
              const sep =
                current.length === 0 || /\s$/.test(current) ? '' : ' '
              onChange(current + sep + phrase)
            }}
            onAutoSubmit={() => {
              // Only auto-send if there's actually content — guards against
              // spurious empty submissions from brief mic bumps.
              const trimmed = (value || '').trim()
              if (trimmed.length > 0 || pendingImages.length > 0) {
                onSend()
              }
            }}
            onStop={() => setVoiceMode(false)}
          />
        )}
      </div>

      <p className="mx-auto mt-2 max-w-3xl text-center text-[10px] text-muted-foreground">
        Gigachat runs locally on your PC. With Auto-approve off, every command waits for your OK.
      </p>
    </div>
  )
}

/** Card-style chip for a pending document (PDF / text). Shows name + page
 *  count + a truncation warning when the extracted body hit the size cap. */
function PendingDocumentChip({ doc, onRemove }) {
  const label = doc.original_name || doc.name
  const pageNote = doc.page_count ? ` · ${doc.page_count}p` : ''
  return (
    <div
      className="relative flex items-center gap-2 rounded-md border border-border bg-muted/40 px-2 py-1.5 text-xs"
      title={doc.truncated ? 'Contents were truncated before sending' : label}
    >
      <FileText className="size-4 text-muted-foreground" />
      <span className="max-w-[180px] truncate font-medium">{label}</span>
      <span className="text-[10px] text-muted-foreground">
        {formatBytes(doc.size)}{pageNote}{doc.truncated ? ' · truncated' : ''}
      </span>
      <button
        type="button"
        onClick={onRemove}
        className="ml-1 rounded-full bg-background p-0.5 text-muted-foreground shadow ring-1 ring-border hover:text-destructive"
        aria-label="Remove attachment"
        title="Remove"
      >
        <XIcon className="size-3" />
      </button>
    </div>
  )
}

function formatBytes(n) {
  if (!n || n < 1024) return `${n || 0} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

/** Thumbnail + remove button for one pending-upload image. */
function PendingImageChip({ image, onRemove }) {
  const src = image.previewUrl || `/api/uploads/${image.name}`
  return (
    <div className="relative">
      <img
        src={src}
        alt="pending attachment"
        className="h-16 w-16 rounded-md border border-border object-cover"
      />
      <button
        type="button"
        onClick={onRemove}
        className="absolute -right-1.5 -top-1.5 rounded-full bg-background p-0.5 text-muted-foreground shadow ring-1 ring-border hover:text-destructive"
        aria-label="Remove attachment"
        title="Remove"
      >
        <XIcon className="size-3" />
      </button>
    </div>
  )
}

/** Inline popup that lists slash-command matches. */
function SlashMenu({ matches, onPick, query }) {
  return (
    <div className="rounded-md border border-border bg-card p-1 text-xs shadow-lg">
      <div className="px-2 py-1 text-[10px] uppercase tracking-wide text-muted-foreground">
        Quick commands — matches for “{query}”
      </div>
      <ul className="max-h-48 overflow-auto">
        {matches.map((m) => (
          <li key={m.name}>
            <button
              type="button"
              onClick={() => onPick(m)}
              className="flex w-full items-start gap-2 rounded px-2 py-1.5 text-left hover:bg-accent"
            >
              <span className="w-20 shrink-0 font-mono text-[11px] text-primary">
                {m.name}
              </span>
              <span className="flex-1 text-muted-foreground">{m.description}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}

/**
 * MentionMenu — dropdown of @-mention file suggestions.
 *
 * Highlights the `active` row (kept in sync with arrow keys in the parent).
 * Clicking a row picks it; hovering updates `active` so keyboard + mouse
 * stay consistent. A tiny source chip ("name" vs "semantic") lets users
 * see at a glance whether a match came from the filename or the semantic
 * index.
 */
function MentionMenu({ matches, onPick, onHover, query, active }) {
  return (
    <div className="rounded-md border border-border bg-card p-1 text-xs shadow-lg">
      <div className="flex items-center gap-1 px-2 py-1 text-[10px] uppercase tracking-wide text-muted-foreground">
        <AtSign className="size-3" />
        File — {query ? `matches for “${query}”` : 'start typing to filter'}
      </div>
      <ul className="max-h-56 overflow-auto">
        {matches.map((m, i) => {
          const isActive = i === active
          return (
            <li key={m.path}>
              <button
                type="button"
                onMouseEnter={() => onHover?.(i)}
                onClick={() => onPick(m)}
                className={cn(
                  'flex w-full items-start gap-2 rounded px-2 py-1.5 text-left',
                  isActive ? 'bg-accent' : 'hover:bg-accent/50',
                )}
              >
                <FileIcon className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />
                <div className="min-w-0 flex-1">
                  <div className="truncate font-mono text-[11px]">
                    {m.rel_path}
                  </div>
                  {m.snippet && (
                    <div className="truncate text-[10px] text-muted-foreground">
                      {m.snippet}
                    </div>
                  )}
                </div>
                {m.source === 'semantic' && (
                  <span className="shrink-0 rounded bg-muted px-1 text-[9px] text-muted-foreground">
                    sem
                  </span>
                )}
              </button>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
