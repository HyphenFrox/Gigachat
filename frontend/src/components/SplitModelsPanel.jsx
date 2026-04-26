import React, { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'
import {
  Layers,
  Download,
  AlertTriangle,
  Loader2,
  CircleCheck,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { api } from '@/lib/api'

/**
 * AutoSplitInstallSection — Phase 2 (auto-routing version).
 *
 * The previous iteration of this panel exposed a full "register split
 * models" UI. That's been removed in favor of a fully automatic routing
 * decision in the backend (`compute_pool.route_chat_for`). The user now
 * just picks any model from the chat picker and the backend transparently
 * decides whether to use Ollama on the host (small-enough models) or to
 * spawn `llama-server` with `--rpc <worker>:<port>` flags so the model's
 * layers fan across the host's VRAM/RAM and every connected worker's
 * GPU/CPU/RAM (big models). The user never sees a "split model" tab.
 *
 * What this component does is the only manual step that's left:
 * making sure llama.cpp's binaries are present on the host. It's a
 * one-time ~150 MB download — surfaced as a small banner near the top
 * of the Compute tab so the user can install it on a fresh machine
 * without leaving Settings.
 */
export default function AutoSplitInstallSection() {
  const [info, setInfo] = useState(null)
  const [installing, setInstalling] = useState(false)
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.listSplitModels()
      setInfo(res.llama_cpp || null)
    } catch (e) {
      // Silent — this banner is best-effort. The user can still register
      // workers and use Phase 1 routing if the install state can't be
      // queried for whatever reason.
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  async function doInstall() {
    setInstalling(true)
    toast.info('Downloading llama.cpp…', {
      description: 'About 150 MB — this can take a minute.',
    })
    try {
      await api.installLlamaCpp('host')
      toast.success('llama.cpp installed', {
        description:
          'Big models that exceed your host VRAM will now auto-route ' +
          'across the compute pool.',
      })
    } catch (e) {
      toast.error('Install failed', { description: e.message })
    } finally {
      setInstalling(false)
      refresh()
    }
  }

  if (loading || !info) return null

  // Installed: show a tight, satisfied row + brief explanation.
  if (info.installed) {
    return (
      <div className="mt-3 flex items-start gap-2 rounded-md border border-emerald-500/20 bg-emerald-500/5 p-2.5 text-[11px]">
        <CircleCheck className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-500" />
        <div className="flex-1 text-muted-foreground">
          <span className="font-medium text-foreground">llama.cpp {info.version}</span>{' '}
          installed. Models too big for your host VRAM auto-split across
          this device + every worker that has{' '}
          <code className="rounded bg-muted px-1">rpc-server</code>{' '}
          reachable.
        </div>
      </div>
    )
  }

  // Not installed: amber-tinted nudge, install button.
  return (
    <div className="mt-3 flex items-start gap-3 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-xs">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />
      <div className="flex-1">
        <div className="flex items-center gap-1.5 font-medium text-foreground">
          <Layers className="h-3.5 w-3.5" />
          llama.cpp not installed
        </div>
        <p className="mt-0.5 text-muted-foreground">
          Without it, the compute pool can only route whole-request work
          (chat / embeddings / subagents per machine, Phase 1). Install
          to also enable big-model layer splitting — when you pick a
          model that exceeds host VRAM, the backend will spawn
          llama-server with{' '}
          <code className="rounded bg-muted px-1">--rpc</code> flags
          pointing at every reachable worker. About 150 MB; one-time.
        </p>
        {!info.platform_supported && (
          <p className="mt-1 text-destructive">
            Auto-install unsupported here: {info.platform_reason}
          </p>
        )}
      </div>
      <Button
        size="sm"
        onClick={doInstall}
        disabled={installing || !info.platform_supported}
        className="gap-1.5"
      >
        {installing ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
        ) : (
          <Download className="h-3.5 w-3.5" />
        )}
        {installing ? 'Installing…' : 'Install llama.cpp'}
      </Button>
    </div>
  )
}
