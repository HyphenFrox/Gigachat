import React, { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Toaster } from '@/components/ui/sonner'
import Sidebar from './components/Sidebar'
import ChatView from './components/ChatView'
import LoginView from './components/LoginView'
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
 */
export default function App() {
  // Auth state. `null` = still checking; {requires_password, authenticated, host}
  // once the initial /api/auth/status call returns. We render the login page
  // when requires_password && !authenticated, and the main app otherwise.
  // A mid-session expiry (via the `gigachat:unauthorized` window event from
  // api.js) flips authenticated back to false so the login gate reappears.
  const [authState, setAuthState] = useState(null)
  const [conversations, setConversations] = useState([])
  const [activeId, setActiveId] = useState(null)
  const [models, setModels] = useState([])
  // When true, the model picker shows every installed model — including ones
  // that will 400 the agent loop because their Ollama template doesn't declare
  // `{{ .Tools }}`. Default stays false so the dropdown is safe out of the box;
  // the user can flip it per-session from the picker footer.
  const [showAllModels, setShowAllModels] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const refreshAuth = useCallback(async () => {
    try {
      const s = await api.getAuthStatus()
      setAuthState(s)
    } catch {
      // If the status call itself fails, assume we need a login rather than
      // rendering a blank screen — the user can retry from the form.
      setAuthState({ requires_password: true, authenticated: false, host: '' })
    }
  }, [])

  useEffect(() => {
    refreshAuth()
  }, [refreshAuth])

  // Any API call returning 401 flips back to the login screen. This covers
  // the session-expired case (the cookie outlives 30 days) and the
  // secret-rotated case (admin wiped data/auth_secret.key).
  useEffect(() => {
    function handler() {
      setAuthState((prev) =>
        prev ? { ...prev, authenticated: false } : prev,
      )
    }
    window.addEventListener('gigachat:unauthorized', handler)
    return () => window.removeEventListener('gigachat:unauthorized', handler)
  }, [])
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

  // Initial load — runs once the user is past the auth gate so we don't
  // fire a wall of requests that would all 401 on a fresh remote visit.
  useEffect(() => {
    if (!authState?.authenticated) return
    reloadConversations()
    reloadModels()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authState?.authenticated])

  // Re-fetch the model list when the tool-capable filter flips. Skipped until
  // the user is authenticated — same rationale as the initial load effect.
  useEffect(() => {
    if (!authState?.authenticated) return
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

  const handleLogout = useCallback(async () => {
    try {
      await api.logout()
    } catch {
      /* already signed out — proceed to the login screen anyway */
    }
    setAuthState((prev) =>
      prev ? { ...prev, authenticated: false } : prev,
    )
    setActiveId(null)
    setConversations([])
  }, [])

  // Still probing the server for its auth posture — render a neutral
  // skeleton (just the toaster) rather than a misleading login prompt.
  if (authState === null) {
    return (
      <div className="flex h-full w-full items-center justify-center bg-background">
        <Toaster />
      </div>
    )
  }

  // Remote install + no cookie → gate.
  if (authState.requires_password && !authState.authenticated) {
    return (
      <>
        <LoginView host={authState.host} onAuthenticated={refreshAuth} />
        <Toaster />
      </>
    )
  }

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
          authRequired={!!authState?.requires_password}
          onLogout={handleLogout}
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
