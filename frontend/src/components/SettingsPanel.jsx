import React, { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'
import {
  Settings,
  Cpu,
  HardDrive,
  Loader2,
  RefreshCw,
  Sliders,
  Brain,
  Webhook,
  Plug,
  CalendarClock,
  KeyRound,
  Wrench,
  Sparkles,
  BookOpen,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { cn } from '@/lib/utils'
import { api } from '@/lib/api'
import MemoriesSection from './MemoriesPanel'
import HooksSection from './HooksPanel'
import MCPSection from './MCPServersPanel'
import SchedulesSection from './SchedulesPanel'
import SecretsSection from './SecretsPanel'
import UserToolsSection from './UserToolsPanel'
import DocsSection from './DocsPanel'

/**
 * SettingsPanel — consolidated settings drawer.
 *
 * Tabs:
 *   - General — default chat model, hardware summary, pull status.
 *   - Memories — global memories injected into every system prompt.
 *   - Secrets — named credentials referenced via {{secret:NAME}} in tool calls.
 *   - Schedules — recurring / one-shot agent runs.
 *   - Tools — user-defined Python tools the agent can call.
 *   - Hooks — lifecycle hooks that fire on agent events.
 *   - MCP — external Model Context Protocol servers.
 *
 * The "Notifications" button stays separately in the sidebar footer because
 * browser push permission and device registration feel more like a
 * per-device toggle than a preference shared across tabs.
 *
 * Each tab body lives in its own file so the refactor stays modular and the
 * nested add/edit/delete dialogs for memories/hooks can still float above
 * the settings drawer.
 *
 * Props:
 *   - open: boolean
 *   - onClose: () => void
 */
const TABS = [
  { id: 'general', label: 'General', Icon: Sliders },
  { id: 'memories', label: 'Memories', Icon: Brain },
  { id: 'secrets', label: 'Secrets', Icon: KeyRound },
  { id: 'schedules', label: 'Schedules', Icon: CalendarClock },
  { id: 'tools', label: 'Tools', Icon: Sparkles },
  { id: 'hooks', label: 'Hooks', Icon: Webhook },
  { id: 'docs', label: 'Docs', Icon: BookOpen },
  { id: 'mcp', label: 'MCP', Icon: Plug },
]

export default function SettingsPanel({ open, onClose }) {
  const [tab, setTab] = useState('general')

  // Reset to General every time the drawer reopens so the user lands in a
  // predictable place rather than wherever they left off last time.
  useEffect(() => {
    if (open) setTab('general')
  }, [open])

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose?.()}>
      <DialogContent className="flex max-h-[90vh] w-[95vw] flex-col overflow-hidden p-0 sm:max-w-2xl">
        <DialogHeader className="border-b border-border px-6 pb-3 pt-4">
          <DialogTitle className="flex items-center gap-2">
            <Settings className="size-4" />
            Settings
          </DialogTitle>
          <DialogDescription>
            Preferences, memories, secrets, schedules, tools, hooks, docs, and MCP integrations — one place.
          </DialogDescription>
        </DialogHeader>

        {/* Tab bar — horizontal, icon + label, active tab gets an underline.
            With 8 tabs, the dialog's max width (sm:2xl, 672 px) plus 95vw on
            phones isn't always enough to render every tab inline, so the row
            scrolls horizontally instead of clipping the rightmost tabs.
            `flex-shrink-0` on each tab keeps labels readable; `whitespace-
            nowrap` stops "Schedules" from wrapping into two lines mid-tab.
            `scrollbar-thin scrollbar-track-transparent` keeps the scrollbar
            visually subtle, native-mobile-app style, while still discoverable. */}
        <div
          className="flex items-center gap-1 overflow-x-auto whitespace-nowrap border-b border-border px-4 pt-1 [scrollbar-width:thin]"
          role="tablist"
        >
          {TABS.map(({ id, label, Icon }) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={tab === id}
              onClick={() => setTab(id)}
              className={cn(
                'flex shrink-0 items-center gap-1.5 border-b-2 px-3 py-2 text-xs font-medium transition-colors -mb-px',
                tab === id
                  ? 'border-primary text-foreground'
                  : 'border-transparent text-muted-foreground hover:text-foreground',
              )}
            >
              <Icon className="size-3.5" />
              {label}
            </button>
          ))}
        </div>

        {/* Tab body — scrollable, padded. Conditional render so each section's
            useEffect refresh runs on tab switch (matches the previous
            per-panel open/close behavior). */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {tab === 'general' && <GeneralSection />}
          {tab === 'memories' && <MemoriesSection />}
          {tab === 'secrets' && <SecretsSection />}
          {tab === 'schedules' && <SchedulesSection />}
          {tab === 'tools' && <UserToolsSection />}
          {tab === 'hooks' && <HooksSection />}
          {tab === 'docs' && <DocsSection />}
          {tab === 'mcp' && <MCPSection />}
        </div>
      </DialogContent>
    </Dialog>
  )
}

/**
 * GeneralSection — default model picker, hardware summary, pull status.
 *
 * Previously lived at the top level of SettingsPanel; extracted here so the
 * settings drawer can host multiple tabs without one of them having a
 * privileged "outer" position.
 */
function GeneralSection() {
  const [config, setConfig] = useState(null) // /api/system/config payload
  const [settings, setSettings] = useState(null) // /api/settings.settings
  const [models, setModels] = useState([])
  const [choice, setChoice] = useState('') // '' = auto-detected
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  // Filter the dropdown to tool-capable models by default. Flipping this
  // re-runs `refresh()` which re-fetches `/api/models` with the new query.
  const [showAllModels, setShowAllModels] = useState(false)

  // Single refresh that fetches the three pieces the panel depends on in
  // parallel. The installed-models list drives the dropdown, the settings
  // payload tells us what the user already picked, and the system config
  // feeds the auto-tune / pull-status banner.
  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const [cfg, set, mods] = await Promise.all([
        api.getSystemConfig(),
        api.getSettings(),
        api.listModels({ all: showAllModels }),
      ])
      setConfig(cfg)
      setSettings(set.settings || {})
      setChoice(set.settings?.default_chat_model || '')
      setModels(Array.isArray(mods.models) ? mods.models : [])
    } catch (e) {
      toast.error('Could not load settings', { description: e.message })
    } finally {
      setLoading(false)
    }
  }, [showAllModels])

  useEffect(() => {
    refresh()
  }, [refresh])

  async function save() {
    setSaving(true)
    try {
      // Empty string => clear the override. Backend accepts either null or ''.
      const patch = { default_chat_model: choice ? choice : null }
      const res = await api.updateSettings(patch)
      setSettings(res.settings || {})
      if (choice) {
        toast.success('Default model saved', {
          description: `New conversations will start with ${choice}.`,
        })
      } else {
        toast.success('Using auto-detected default', {
          description: `Backend will pick the best model for your hardware (currently ${res.effective_chat_model}).`,
        })
      }
      // Re-read the system config so the "effective model" label stays in sync.
      try {
        const cfg = await api.getSystemConfig()
        setConfig(cfg)
      } catch {
        /* non-fatal */
      }
    } catch (e) {
      toast.error('Save failed', { description: e.message })
    } finally {
      setSaving(false)
    }
  }

  const effective = config?.effective_chat_model || ''
  const recommended = config?.recommended_chat_model || ''
  const hwRam = config?.ram_gb
  const hwVram = config?.vram_gb
  const hwGpu = config?.gpu_name
  const pulling = !!config?.model_pulling
  const pullStatus = config?.pull_status || ''
  const pullError = config?.pull_error || ''

  // Dirty = user edited the picker since the last save.
  const currentStored = settings?.default_chat_model || ''
  const dirty = choice !== currentStored

  return (
    <div className="space-y-4">
      {/* Default model picker */}
      <section className="space-y-2">
        <div className="flex items-center justify-between">
          <label
            htmlFor="default-model"
            className="text-sm font-medium text-foreground"
          >
            Default chat model
          </label>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => setShowAllModels((v) => !v)}
              className={cn(
                'flex items-center gap-1 text-[11px] transition-colors',
                showAllModels
                  ? 'text-amber-400 hover:text-amber-300'
                  : 'text-muted-foreground hover:text-foreground',
              )}
              title={
                showAllModels
                  ? 'Showing every installed model. Some may 400 the agent loop if their Ollama template does not declare tool support. Click to filter back to tool-capable only.'
                  : 'Showing only models whose Ollama `capabilities` include `tools`. Click to show every installed model (including embedding / non-tool ones).'
              }
              disabled={loading}
            >
              <Wrench className="size-3" />
              {showAllModels ? 'All models' : 'Tool-capable'}
            </button>
            <button
              type="button"
              onClick={refresh}
              className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
              title="Re-read installed models and auto-tune status"
              disabled={loading}
            >
              <RefreshCw className={cn('size-3', loading && 'animate-spin')} />
              Refresh
            </button>
          </div>
        </div>
        <select
          id="default-model"
          value={choice}
          onChange={(e) => setChoice(e.target.value)}
          disabled={loading || models.length === 0}
          className={cn(
            'flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm',
            'ring-offset-background placeholder:text-muted-foreground',
            'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
            'disabled:cursor-not-allowed disabled:opacity-50',
          )}
        >
          <option value="">
            Auto-detected{recommended ? ` (${recommended})` : ''}
          </option>
          {models.map((m) => (
            <option key={m} value={m}>
              {m}
              {m === recommended ? ' — recommended' : ''}
            </option>
          ))}
        </select>
        <p className="text-xs text-muted-foreground">
          {currentStored ? (
            <>Currently using <span className="font-medium text-foreground">{currentStored}</span> for new chats.</>
          ) : (
            <>
              Auto-detected pick: <span className="font-medium text-foreground">{effective || '—'}</span>. Change
              the dropdown above to override for every new conversation.
            </>
          )}
        </p>
        {models.length === 0 && !loading && (
          <p className="text-xs text-amber-500">
            No models installed yet. Ollama may still be pulling — try the
            Refresh button in a minute, or pull one manually with{' '}
            <code className="rounded bg-muted px-1">ollama pull gemma4:e4b</code>.
          </p>
        )}
        <div className="flex justify-end pt-1">
          <Button size="sm" onClick={save} disabled={!dirty || saving || loading}>
            {saving ? 'Saving…' : 'Save'}
          </Button>
        </div>
      </section>

      {/* Pull status banner while the auto-tuner is downloading. */}
      {(pulling || pullError) && (
        <section
          className={cn(
            'flex items-start gap-2 rounded-md border p-3 text-xs',
            pullError
              ? 'border-destructive/40 bg-destructive/10 text-destructive'
              : 'border-border bg-muted/40 text-muted-foreground',
          )}
        >
          {pulling ? (
            <Loader2 className="mt-0.5 size-4 shrink-0 animate-spin text-primary" />
          ) : (
            <HardDrive className="mt-0.5 size-4 shrink-0" />
          )}
          <div>
            <p className="font-medium text-foreground">
              {pulling ? 'Downloading model…' : 'Auto-pull reported an issue'}
            </p>
            <p>{pullError || pullStatus || '—'}</p>
          </div>
        </section>
      )}

      {/* Hardware summary — helps the user understand the auto-tuner's pick. */}
      <section className="rounded-md border border-border bg-card p-3 text-xs">
        <div className="mb-1 flex items-center gap-2 text-[10px] uppercase tracking-wide text-muted-foreground">
          <Cpu className="size-3" />
          Detected hardware
        </div>
        <ul className="space-y-0.5">
          <li>
            <span className="text-muted-foreground">RAM:</span>{' '}
            <span className="font-medium text-foreground">
              {typeof hwRam === 'number' ? `${hwRam.toFixed(1)} GB` : '—'}
            </span>
          </li>
          <li>
            <span className="text-muted-foreground">GPU:</span>{' '}
            <span className="font-medium text-foreground">
              {hwGpu
                ? `${hwGpu}${typeof hwVram === 'number' && hwVram > 0 ? ` (${hwVram.toFixed(1)} GB VRAM)` : ''}`
                : 'none detected'}
            </span>
          </li>
          <li>
            <span className="text-muted-foreground">Context window:</span>{' '}
            <span className="font-medium text-foreground">
              {config?.num_ctx?.toLocaleString?.() || config?.num_ctx || '—'} tokens
            </span>
          </li>
        </ul>
      </section>
    </div>
  )
}
