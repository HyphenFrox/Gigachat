import React, { useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { X as XIcon, Copy, Download, Code2, Eye, AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

/**
 * ArtifactPanel — slide-over pane that renders an assistant-produced artifact.
 *
 * Supported artifact kinds (detected from the fenced-code block's language):
 *   - html       → sandboxed <iframe srcdoc=…> so scripts can't touch our page
 *   - svg        → rendered inside a sandboxed iframe as well; SVG allows
 *                   <script> and event handlers, so we don't dangerouslySetInnerHTML
 *   - mermaid    → compiled to SVG via the `mermaid` library (lazy-loaded)
 *   - markdown   → react-markdown with GFM (same engine as chat)
 *
 * Two tabs: Preview (rendered) and Source (raw text). Copy and Download
 * buttons are available in either view. The pane slides in from the right
 * edge and is dismissed via the close button, Escape, or clicking outside.
 *
 * Security note: the HTML/SVG iframe uses `sandbox=""` (no allowances) so
 * the embedded content cannot run top-level navigation, access localStorage,
 * use the parent's cookies, or call the browser's permissions APIs. Scripts
 * still execute inside the iframe itself but are fully confined.
 */
export default function ArtifactPanel({ artifact, onClose }) {
  const [tab, setTab] = useState('preview') // 'preview' | 'source'

  useEffect(() => {
    if (!artifact) return
    function onKey(e) {
      if (e.key === 'Escape') onClose?.()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [artifact, onClose])

  if (!artifact) return null

  async function copySource() {
    try {
      await navigator.clipboard.writeText(artifact.source || '')
      toast.success('Copied source to clipboard')
    } catch (e) {
      toast.error('Copy failed', { description: e.message })
    }
  }

  function downloadSource() {
    // Code artifacts get a language-aware extension so the download is
    // actually openable in an editor; everything else uses the kind map.
    let ext = ARTIFACT_EXTENSIONS[artifact.kind] || 'txt'
    if (artifact.kind === 'code' && artifact.language) {
      const lang = String(artifact.language).toLowerCase()
      ext = CODE_LANGUAGE_EXTENSIONS[lang] || 'txt'
    }
    const blob = new Blob([artifact.source || ''], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `artifact-${artifact.id || 'snippet'}.${ext}`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  return (
    <>
      {/* Backdrop — click to dismiss. Doesn't block the activity panel below
          because it's wrapped inside the artifact overlay only. */}
      <div
        className="fixed inset-0 z-30 bg-black/40 md:hidden"
        onClick={onClose}
      />

      {/* Slide-over container. On desktop it occupies a wide right column
          (clamped by min-width); on mobile it fills the screen. */}
      <aside
        className={cn(
          'fixed inset-y-0 right-0 z-40 flex w-full max-w-[720px] flex-col border-l border-border bg-background shadow-xl',
          'md:w-[620px]',
        )}
        role="dialog"
        aria-label={`Artifact preview: ${artifact.kind}`}
      >
        <header className="flex items-center justify-between gap-2 border-b border-border px-4 py-2">
          <div className="flex min-w-0 items-center gap-2">
            <span className="rounded bg-secondary px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-secondary-foreground">
              {artifact.kind}
            </span>
            <h2 className="truncate text-sm font-medium">
              {artifact.title || 'Artifact preview'}
            </h2>
          </div>
          <div className="flex items-center gap-1">
            <div className="flex overflow-hidden rounded-md border border-border text-[11px]">
              <button
                type="button"
                onClick={() => setTab('preview')}
                className={cn(
                  'flex items-center gap-1 px-2 py-1 transition-colors',
                  tab === 'preview'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:bg-accent/60',
                )}
              >
                <Eye className="size-3" /> Preview
              </button>
              <button
                type="button"
                onClick={() => setTab('source')}
                className={cn(
                  'flex items-center gap-1 px-2 py-1 transition-colors',
                  tab === 'source'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:bg-accent/60',
                )}
              >
                <Code2 className="size-3" /> Source
              </button>
            </div>
            <Button variant="ghost" size="icon" onClick={copySource} title="Copy source">
              <Copy className="size-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={downloadSource} title="Download">
              <Download className="size-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={onClose} title="Close">
              <XIcon />
            </Button>
          </div>
        </header>

        <div className="flex-1 overflow-auto">
          {tab === 'preview' ? (
            <ArtifactRenderer artifact={artifact} />
          ) : (
            <pre className="whitespace-pre-wrap break-all p-4 font-mono text-[12px] leading-relaxed text-foreground">
              {artifact.source}
            </pre>
          )}
        </div>
      </aside>
    </>
  )
}

const ARTIFACT_EXTENSIONS = {
  html: 'html',
  svg: 'svg',
  mermaid: 'mmd',
  markdown: 'md',
  code: 'txt',
}

// Language -> file extension for "code" artifacts. Anything not listed
// falls back to .txt so the download still works; the preview itself
// doesn't change based on this list.
const CODE_LANGUAGE_EXTENSIONS = {
  python: 'py', py: 'py',
  javascript: 'js', js: 'js',
  typescript: 'ts', ts: 'ts',
  jsx: 'jsx', tsx: 'tsx',
  java: 'java',
  kotlin: 'kt', kt: 'kt',
  swift: 'swift',
  c: 'c', cpp: 'cpp', 'c++': 'cpp',
  rust: 'rs', rs: 'rs',
  go: 'go',
  php: 'php',
  ruby: 'rb', rb: 'rb',
  shell: 'sh', bash: 'sh', sh: 'sh',
  powershell: 'ps1', ps1: 'ps1',
  sql: 'sql',
  yaml: 'yml', yml: 'yml',
  json: 'json',
  toml: 'toml',
  css: 'css', scss: 'scss',
}

/** Dispatches to the right renderer for the artifact kind. */
function ArtifactRenderer({ artifact }) {
  switch (artifact.kind) {
    case 'html':
      return <SandboxedFrame srcdoc={artifact.source} />
    case 'svg':
      // Wrap the raw SVG in a tiny HTML document so the iframe sandbox still
      // applies — keeps malicious SVG scripts fully quarantined. Also centers
      // the drawing so it doesn't render flush to the top-left.
      return (
        <SandboxedFrame
          srcdoc={`<!doctype html><html><head><style>html,body{margin:0;height:100%;background:transparent;display:flex;align-items:center;justify-content:center;overflow:hidden}svg{max-width:100%;max-height:100%}</style></head><body>${artifact.source}</body></html>`}
        />
      )
    case 'mermaid':
      return <MermaidRenderer source={artifact.source} />
    case 'markdown':
      return (
        <div className="prose-chat max-w-none p-4 text-sm">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {artifact.source}
          </ReactMarkdown>
        </div>
      )
    case 'code':
      return <CodeRenderer source={artifact.source} language={artifact.language} />
    default:
      return <p className="p-4 text-sm text-muted-foreground">Unknown artifact.</p>
  }
}

/**
 * CodeRenderer — gutter-numbered `<pre>` for long code blocks.
 *
 * We deliberately don't ship a full syntax-highlighter library just for
 * the preview pane — keeps the bundle lean (no ~300 KB prism/shiki) and
 * the user's own editor is always one Download click away. Line numbers
 * alone make long code a whole lot easier to scan vs. the inline
 * chat-view rendering.
 */
function CodeRenderer({ source, language }) {
  const lines = useMemo(() => (source || '').split('\n'), [source])
  return (
    <div className="h-full overflow-auto bg-muted/20 font-mono text-[12px] leading-relaxed">
      {language ? (
        <div className="sticky top-0 border-b border-border bg-background/90 px-4 py-1 text-[10px] uppercase tracking-wider text-muted-foreground backdrop-blur">
          {language}
        </div>
      ) : null}
      <div className="flex">
        <pre
          aria-hidden="true"
          className="select-none border-r border-border px-3 py-4 text-right text-muted-foreground/70"
        >
          {lines.map((_, i) => (
            <div key={i}>{i + 1}</div>
          ))}
        </pre>
        <pre className="flex-1 overflow-x-auto whitespace-pre px-4 py-4 text-foreground">
          {source || ''}
        </pre>
      </div>
    </div>
  )
}

/**
 * SandboxedFrame — iframe with an empty `sandbox` attribute (no allowances).
 *
 * `sandbox=""` means:
 *   - scripts still run, but same-origin is treated as a unique origin
 *   - cannot access parent window / localStorage / cookies
 *   - cannot submit forms, open popups, or navigate the top frame
 *   - cannot request permissions (mic/camera/clipboard/etc.)
 *
 * This is the right level for untrusted model output: rich enough to render
 * a working demo, locked down enough that a prompt-injection payload can't
 * exfiltrate data or escalate privilege.
 */
function SandboxedFrame({ srcdoc }) {
  return (
    <iframe
      title="Artifact preview"
      sandbox=""
      srcDoc={srcdoc}
      className="h-full w-full border-0 bg-white"
    />
  )
}

/**
 * MermaidRenderer — lazy-loads the mermaid library and renders the diagram
 * as inline SVG. The library is ~500 KB so we only pay for it when the user
 * actually opens a mermaid artifact.
 */
function MermaidRenderer({ source }) {
  const [svg, setSvg] = useState(null)
  const [error, setError] = useState(null)
  const idRef = useRef(`mmd-${Math.random().toString(36).slice(2)}`)

  useEffect(() => {
    let cancelled = false
    async function render() {
      try {
        const mermaid = (await import('mermaid')).default
        mermaid.initialize({
          startOnLoad: false,
          theme: 'dark',
          securityLevel: 'strict',
        })
        const { svg } = await mermaid.render(idRef.current, source)
        if (!cancelled) {
          setSvg(svg)
          setError(null)
        }
      } catch (e) {
        if (!cancelled) {
          setError(e.message || String(e))
          setSvg(null)
        }
      }
    }
    render()
    return () => {
      cancelled = true
    }
  }, [source])

  if (error) {
    return (
      <div className="flex items-start gap-2 p-4 text-sm text-destructive">
        <AlertTriangle className="mt-0.5 size-4 shrink-0" />
        <div>
          <p className="font-medium">Mermaid render failed</p>
          <p className="mt-1 text-xs opacity-80">{error}</p>
        </div>
      </div>
    )
  }
  if (!svg) {
    return <p className="p-4 text-sm text-muted-foreground">Rendering diagram…</p>
  }
  return (
    <div
      className="flex h-full items-center justify-center p-4"
      // Mermaid's output is static SVG with no <script> tags; the library
      // also applies `securityLevel: 'strict'` above, which sanitizes text.
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  )
}

/**
 * Build the supported languages map in a single place so detection and
 * rendering never drift. Exposed for use by the inline code-block button.
 */
export const ARTIFACT_KINDS = new Set(['html', 'svg', 'mermaid', 'markdown', 'code'])

/**
 * Normalise a fenced-code language hint to one of our artifact kinds, or
 * null if the language isn't something we render.
 *
 * Specialised kinds (html/svg/mermaid/markdown) always win — they have
 * dedicated renderers. Anything else recognised as "code" gets the
 * generic line-numbered viewer iff the block is long enough; that
 * threshold is enforced by the caller (Message.jsx) so short snippets
 * don't pick up a pointless "Open preview" button.
 */
export function languageToArtifactKind(lang) {
  if (!lang) return null
  const l = lang.toLowerCase().trim()
  if (l === 'html') return 'html'
  if (l === 'svg') return 'svg'
  if (l === 'mermaid') return 'mermaid'
  if (l === 'markdown' || l === 'md') return 'markdown'
  // Every other non-empty language becomes a generic code artifact.
  // We still return a kind so the caller can opt in based on length.
  return 'code'
}

/**
 * Minimum source length (in lines) before a *generic* code block is worth
 * popping out into the preview pane. Specialised kinds ignore this — a
 * 3-line SVG still renders beautifully, but a 3-line Python snippet is
 * already fine inline.
 */
export const CODE_ARTIFACT_MIN_LINES = 20
