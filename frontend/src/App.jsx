import React, { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Toaster } from '@/components/ui/sonner'
import Sidebar from './components/Sidebar'
import ChatView from './components/ChatView'
import { api } from '@/lib/api'
import { onPushClick } from '@/lib/pwa'

/**
 * App — top-level layout.
 *
 * Layout:
 *   - Desktop (md+): 2-column — Sidebar (fixed 288px) + ChatView (fills).
 *   - Mobile (<md): ChatView full-width; Sidebar slides in from the left as
 *     a drawer when the hamburger is tapped. This gives the "native mobile
 *     app" feel CLAUDE.md asks for.
 *
 * No auth flow: the chat UI is loopback-only at the backend (the
 * AuthMiddleware in app.py rejects non-loopback requests for non-P2P
 * paths with a clear "loopback only" 403). Cross-device chat from
 * another laptop's browser isn't a supported use case — install
 * Gigachat on the other device too and pair them via Compute pool.
 */
export default function App() {
  const [conversations, setConversations] = useState([])
  const [activeId, setActiveId] = useState(null)
  const [models, setModels] = useState([])
  // When true, the model picker shows every installed model — including ones
  // that will 400 the agent loop because their Ollama template doesn't declare
  // `{{ .Tools }}`. Default stays false so the dropdown is safe out of the box;
  // the user can flip it per-session from the picker footer.
  const [showAllModels, setShowAllModels] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  // When the user clicks a semantic-search hit, the sidebar sets both a
  // conversation id and a target message id. ChatView scrolls to the message
  // after it finishes loading the conversation and clears the target via
  // `onJumpHandled` so a second click on the same hit re-triggers the jump.
  const [jumpTarget, setJumpTarget] = useState({ convId: null, messageId: null })

  const reloadConversations = useCallback(async () => {
    try {
      const { conversations } = await api.listConversations()
      setConversations(conversations)
      // If the active conversation was deleted elsewhere, deselect.
      if (activeId && !conversations.find((c) => c.id === activeId)) {
        setActiveId(null)
      }
    } catch (e) {
      toast.error('Failed to load conversations', { description: e.message })
    }
  }, [activeId])

  const reloadModels = useCallback(async () => {
    try {
      const { models, error } = await api.listModels({ all: showAllModels })
      setModels(models || [])
      if (error) {
        toast.warning('Ollama not reachable', {
          description: 'Is `ollama serve` running on http://localhost:11434?',
        })
      }
    } catch (e) {
      toast.error('Failed to list models', { description: e.message })
    }
  }, [showAllModels])

  // Initial load — fires on mount.
  useEffect(() => {
    reloadConversations()
    reloadModels()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Re-fetch the model list when the tool-capable filter flips.
  useEffect(() => {
    reloadModels()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showAllModels])

  // When the user taps a push notification, the service worker posts a
  // message back to the focused tab with the conversation id. Route that
  // to the same jump flow the semantic-search hit uses — selecting the
  // conversation puts ChatView in charge of loading + scrolling.
  useEffect(() => {
    onPushClick(({ conversation_id }) => {
      if (!conversation_id) return
      setActiveId(conversation_id)
      setJumpTarget({ convId: conversation_id, messageId: null })
    })
  }, [])

  // Cold-start path: the service worker's notificationclick opens a fresh
  // window with `?conv=<id>`. Consume the hint once, then strip it from the
  // URL so a reload doesn't keep re-jumping.
  useEffect(() => {
    if (typeof window === 'undefined') return
    const params = new URLSearchParams(window.location.search)
    const convFromUrl = params.get('conv')
    if (convFromUrl) {
      setActiveId(convFromUrl)
      params.delete('conv')
      const next =
        window.location.pathname +
        (params.toString() ? `?${params.toString()}` : '')
      window.history.replaceState({}, '', next)
    }
  }, [])

  // When a conversation's metadata changes (rename, updated_at), refresh list.
  const onConversationUpdated = useCallback((_updated) => {
    reloadConversations()
  }, [reloadConversations])

  return (
    <div className="flex h-full w-full">
      {/* Mobile overlay drawer */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/60 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <div
        className={[
          'fixed inset-y-0 left-0 z-40 w-72 border-r border-border bg-background shadow-xl transition-transform duration-200 md:static md:z-0 md:shadow-none',
          sidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0',
        ].join(' ')}
      >
        <Sidebar
          conversations={conversations}
          activeId={activeId}
          models={models}
          onSelect={(id) => {
            setActiveId(id)
            setSidebarOpen(false)
          }}
          onJumpToMessage={(convId, messageId) => {
            setActiveId(convId)
            setJumpTarget({ convId, messageId })
            setSidebarOpen(false)
          }}
          onReload={reloadConversations}
          onClose={() => setSidebarOpen(false)}
        />
      </div>

      <main className="flex min-w-0 flex-1 flex-col">
        <ChatView
          id={activeId}
          models={models}
          showAllModels={showAllModels}
          onToggleShowAllModels={() => setShowAllModels((v) => !v)}
          onConversationUpdated={onConversationUpdated}
          onOpenSidebar={() => setSidebarOpen(true)}
          jumpToMessageId={
            jumpTarget.convId === activeId ? jumpTarget.messageId : null
          }
          onJumpHandled={() => setJumpTarget({ convId: null, messageId: null })}
        />
      </main>

      <Toaster />
    </div>
  )
}
