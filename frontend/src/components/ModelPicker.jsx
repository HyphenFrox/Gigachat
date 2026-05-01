import React, { useEffect, useMemo, useState } from 'react'
import {
  ChevronDown, Wrench, Search, Monitor, Wifi, Globe,
  Lock, Loader2, Download,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { cn } from '@/lib/utils'
import { api } from '@/lib/api'

/**
 * ModelPicker — choose the conversation's model from across the
 * compute pool: local Ollama, paired-LAN peers, and (when Public
 * Pool is enabled) the public swarm via the rendezvous.
 *
 * Why the rewrite: with LAN + public sources the catalogue can hold
 * dozens or hundreds of entries. A flat dropdown stops being usable
 * past ~15 items. The new UI:
 *   1. Loads the aggregated inventory in one call (`listModelsAllSources`).
 *   2. Renders three labelled sections with counts.
 *   3. Provides a search box that filters across all sections.
 *   4. Each non-local row carries a badge for its source device
 *      and (where applicable) a lock icon for E2E-encrypted dispatch.
 *   5. Public-pool rows show a "downloads from official source on
 *      the executing machine" subtitle so users understand we don't
 *      pump models through the user's internet.
 *
 * Sources:
 *   • Local       — running on this device's Ollama. Always available.
 *   • LAN         — installed on a paired peer; routed via E2E
 *                    encrypted compute proxy.
 *   • Public Pool — discoverable via rendezvous; routed to a peer
 *                    that has the model OR auto-pulled from the
 *                    OFFICIAL source (Ollama registry / HuggingFace)
 *                    on the executing machine. NOT a P2P file
 *                    transfer — model bytes never traverse another
 *                    user's connection (they'd bottleneck the swarm).
 *
 * Backwards-compat: the old `models` prop is still accepted and used
 * if `listModelsAllSources` fails (e.g. on a backend that hasn't
 * been upgraded). UI degrades to a single Local section in that case.
 */
export default function ModelPicker({
  conv,
  models,            // legacy fallback (flat array of names)
  showAllModels,
  onToggleShowAllModels,
  onPatch,
}) {
  const [sources, setSources] = useState(null)
  const [loading, setLoading] = useState(false)
  const [search, setSearch] = useState('')
  const [open, setOpen] = useState(false)

  // Refetch whenever the picker opens — model lists change as users
  // pull new models, peers come online, or the public-pool toggle
  // flips. Cheap (~50 ms backend roundtrip).
  useEffect(() => {
    if (!open) return
    let cancelled = false
    setLoading(true)
    api.listModelsAllSources({ toolsOnly: !showAllModels })
      .then((data) => {
        if (!cancelled) setSources(data)
      })
      .catch(() => {
        if (!cancelled) setSources(null)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [open, showAllModels])

  // Compose filtered sections. `search` is matched against name +
  // family + source label so a user typing "llama" surfaces both
  // local Llama variants and any LAN/public peers offering them.
  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    function match(m, extra = '') {
      if (!q) return true
      const hay = `${m.name || ''} ${m.family || ''} ${m.source_label || ''} ${extra}`
        .toLowerCase()
      return hay.includes(q)
    }
    if (!sources) {
      // Legacy fallback to the flat `models` array.
      const list = (models || [])
        .map((name) => ({ name }))
        .filter((m) => match(m))
      return { local: list, lan: [], public: [], public_pool_enabled: false }
    }
    return {
      local: (sources.local || []).filter((m) => match(m)),
      lan: (sources.lan || []).filter((m) => match(m, 'lan paired')),
      public: (sources.public || []).filter((m) => match(m, 'public pool swarm')),
      public_pool_enabled: !!sources.public_pool_enabled,
    }
  }, [sources, models, search])

  const totalCount =
    filtered.local.length + filtered.lan.length + filtered.public.length

  function pick(name) {
    if (!name || name === conv?.model) return
    onPatch({ model: name })
    setOpen(false)
  }

  if (!conv) return null

  return (
    <DropdownMenu open={open} onOpenChange={setOpen}>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className="h-7 gap-1 px-2 text-xs"
          title={conv.model}
        >
          <span className="max-w-[160px] truncate sm:max-w-[240px]">
            {conv.model}
          </span>
          <ChevronDown className="size-3 shrink-0" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="start"
        className="w-[min(95vw,30rem)] max-h-[70vh] overflow-hidden p-0"
      >
        {/* Search bar — sticky at top */}
        <div className="border-b border-border bg-background p-2">
          <div className="relative">
            <Search className="absolute left-2 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              autoFocus
              placeholder="Search models…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="h-7 pl-7 text-xs"
            />
          </div>
          <div className="mt-1.5 flex items-center justify-between text-[10px] text-muted-foreground">
            <span>
              {loading
                ? 'Loading inventory…'
                : `${totalCount} model${totalCount === 1 ? '' : 's'} across pool`}
            </span>
            {onToggleShowAllModels && (
              <button
                type="button"
                onClick={(e) => {
                  e.preventDefault()
                  onToggleShowAllModels()
                }}
                title={
                  showAllModels
                    ? 'Hide models without tool-calling support.'
                    : 'Show every installed model, even those without tool support.'
                }
                className="inline-flex items-center gap-1 rounded px-1 py-0.5 hover:bg-accent"
              >
                <Wrench
                  className={cn(
                    'size-3',
                    showAllModels ? 'text-muted-foreground/50' : 'text-emerald-400',
                  )}
                />
                {showAllModels ? 'All models' : 'Tool-capable'}
              </button>
            )}
          </div>
        </div>

        {/* Sections */}
        <div className="max-h-[55vh] overflow-y-auto py-1">
          {/* LOCAL */}
          {filtered.local.length > 0 && (
            <Section
              icon={Monitor}
              label="On this device"
              tone="text-foreground"
              count={filtered.local.length}
            >
              {filtered.local.map((m) => (
                <ModelRow
                  key={`local-${m.name}`}
                  m={m}
                  current={m.name === conv.model}
                  onPick={() => pick(m.name)}
                  badge={null}
                />
              ))}
            </Section>
          )}

          {/* LAN */}
          {filtered.lan.length > 0 && (
            <Section
              icon={Wifi}
              label="On paired devices (LAN)"
              tone="text-sky-400"
              count={filtered.lan.length}
            >
              {filtered.lan.map((m, i) => (
                <ModelRow
                  key={`lan-${m.source_device_id}-${m.name}-${i}`}
                  m={m}
                  current={m.name === conv.model}
                  onPick={() => pick(m.name)}
                  badge={
                    <SourceBadge
                      label={m.source_label}
                      encrypted={m.encrypted !== false}
                    />
                  }
                />
              ))}
            </Section>
          )}

          {/* PUBLIC POOL */}
          {filtered.public_pool_enabled && (
            <Section
              icon={Globe}
              label="On public pool"
              tone="text-amber-400"
              count={filtered.public.length}
            >
              {filtered.public.length === 0 ? (
                <div className="px-3 pb-2 pt-1 text-[11px] leading-snug text-muted-foreground">
                  No public-pool peers advertising models yet (or rendezvous
                  unreachable). When peers are online, the routing layer
                  will pick one with the chosen model — and if none has it,
                  download from the model's <strong>official source</strong>{' '}
                  (Ollama registry / HuggingFace) on the executing machine.
                  Model bytes never flow peer-to-peer over the swarm.
                </div>
              ) : (
                filtered.public.map((m, i) => (
                  <ModelRow
                    key={`pub-${m.source_device_id || i}-${m.name}`}
                    m={m}
                    current={m.name === conv.model}
                    onPick={() => pick(m.name)}
                    badge={
                      <SourceBadge
                        label={m.source_label || 'public peer'}
                        encrypted
                      />
                    }
                  />
                ))
              )}
            </Section>
          )}

          {/* Empty state */}
          {totalCount === 0 && !loading && (
            <div className="px-4 py-6 text-center text-xs text-muted-foreground">
              {search.trim()
                ? `No models match "${search}".`
                : 'No models found. Pull one with `ollama pull <name>` or pair a device that has one.'}
            </div>
          )}
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

/* ----------------------------- helpers ----------------------------- */

function Section({ icon: Icon, label, tone, count, children }) {
  return (
    <div>
      <div
        className={cn(
          'sticky top-0 z-10 flex items-center gap-2 bg-background/95 px-3 py-1 text-[10px] font-medium uppercase tracking-wide backdrop-blur',
          tone,
        )}
      >
        <Icon className="size-3" />
        <span>{label}</span>
        <span className="ml-auto text-muted-foreground">{count}</span>
      </div>
      {children}
    </div>
  )
}

function ModelRow({ m, current, onPick, badge }) {
  return (
    <button
      type="button"
      onClick={onPick}
      title={m.name}
      className={cn(
        'flex w-full items-start gap-2 px-3 py-1.5 text-left',
        'hover:bg-accent focus:bg-accent focus:outline-none',
        current && 'bg-accent/50',
      )}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 text-xs">
          <span className="break-all leading-snug">{m.name}</span>
          {current && (
            <span className="shrink-0 text-[9px] uppercase text-muted-foreground">
              current
            </span>
          )}
        </div>
        <div className="mt-0.5 flex flex-wrap items-center gap-1.5 text-[10px] text-muted-foreground">
          {m.parameter_size && <span>{m.parameter_size}</span>}
          {m.quantization_level && <span>{m.quantization_level}</span>}
          {m.family && <span>{m.family}</span>}
          {typeof m.size_bytes === 'number' && m.size_bytes > 0 && (
            <span>{(m.size_bytes / 1e9).toFixed(1)} GB</span>
          )}
          {badge}
        </div>
      </div>
    </button>
  )
}

function SourceBadge({ label, encrypted }) {
  return (
    <span className="inline-flex items-center gap-1 rounded bg-muted px-1.5 py-0.5 text-[9px]">
      {encrypted && (
        <Lock
          className="size-2.5 text-emerald-500"
          title="End-to-end encrypted"
        />
      )}
      <span className="max-w-[120px] truncate">{label}</span>
    </span>
  )
}
