import React, { useCallback, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { BrainCircuit, Play, Clock, HelpCircle, Lightbulb, Repeat, X } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import Message from './Message'
import ToolCall from './ToolCall'
import ChatHeader from './ChatHeader'
import ChatInput from './ChatInput'
import TodoPanel from './TodoPanel'
import ActivityPanel from './ActivityPanel'
import ArtifactPanel from './ArtifactPanel'
import BrandLogo from './BrandLogo'
import { api } from '@/lib/api'
import { postEventStream } from '@/lib/sse'
import { cn } from '@/lib/utils'

// Selector-safe escape for message ids (UUIDs are already safe but we
// defend against any future id scheme that might include punctuation).
const cssEscape = (s) =>
  typeof window !== 'undefined' && window.CSS && window.CSS.escape
    ? window.CSS.escape(s)
    : String(s).replace(/[^\w-]/g, '\\$&')

/**
 * ChatView — the main conversation pane.
 *
 * Responsibilities:
 *   1. Load the conversation metadata and message history when `id` changes.
 *   2. Send a new user message and stream the agent's reply as SSE events.
 *   3. Maintain per-tool-call UI state (await / running / done / rejected)
 *      in a single map keyed by tool call id.
 *   4. Handle approval button clicks while the turn is paused.
 *   5. Track the agent's todo list and render it in a pinned panel.
 *   6. Handle pasted / dropped images — upload to the backend, show chips
 *      in the composer, and attach their filenames to the next user turn.
 *   7. Auto-scroll to the latest activity.
 *   8. After the first user message, auto-title the conversation.
 *
 * Event contract (see backend/agent.py for the producer side):
 *   delta            -> append to `liveContent`
 *   thinking         -> accumulate into `liveThinking`
 *   assistant_message-> canonical row persisted; append to `messages`
 *   tool_call        -> toolStates[id] = {status:'running', reason, args}
 *   await_approval   -> toolStates[id] = {status:'await', reason, args, preview}
 *   tool_result      -> toolStates[id] = {status:'done', result}
 *   todos_updated    -> replace the current todo list
 *   turn_done        -> clear live buffers, mark not busy
 *   error            -> toast.error, mark not busy
 */
export default function ChatView({
  id,
  models,
  showAllModels,
  onToggleShowAllModels,
  onConversationUpdated,
  onOpenSidebar,
  jumpToMessageId,
  onJumpHandled,
}) {
  const [conv, setConv] = useState(null)
  const [messages, setMessages] = useState([])
  // Per-tool-call UI state. Shape:
  //   { [callId]: {
  //       status: 'await'|'running'|'done'|'rejected',
  //       result?: {ok, output, error},
  //       imagePath?: string,
  //       reason?: string,
  //       args?: object,
  //       preview?: {kind:'diff', path, diff, ...},
  //     }
  //   }
  const [toolStates, setToolStates] = useState({})
  const [liveContent, setLiveContent] = useState('') // streaming assistant text for current iteration
  const [liveThinking, setLiveThinking] = useState('') // streaming reasoning tokens (if the model emits them)
  const [busy, setBusy] = useState(false)
  const [input, setInput] = useState('')
  const [error, setError] = useState(null)
  // Tasks the agent is currently tracking via `todo_write`. Reset to [] on
  // conversation change; stays stable across turns within the same convo.
  const [todos, setTodos] = useState([])
  // Pending attachments (images + documents) for the next user turn.
  // Shape for images:    { kind:'image', name, size, content_type, previewUrl, original_name }
  // Shape for documents: { kind:'document', name, size, content_type, original_name,
  //                        extracted_text, truncated, page_count }
  // Both forms live in one list so the composer can render a mixed chip strip
  // while the send handler still separates them at submit time.
  const [pendingImages, setPendingImages] = useState([])
  // Currently-open artifact preview. Shape: {kind, source, title} or null.
  // Opened when the user clicks the "Open preview" chip inside a fenced-code
  // block of an assistant message. See ArtifactPanel for supported kinds.
  const [activeArtifact, setActiveArtifact] = useState(null)
  // Pending AskUserQuestion prompt — the agent yielded an `await_user_answer`
  // event and is blocking for a click. Shape: { id, question, options } or null.
  const [pendingQuestion, setPendingQuestion] = useState(null)
  // Side-task chips flagged via the spawn_task tool mid-turn. We keep a local
  // copy so new chips appear instantly from the SSE stream; on reload we
  // reconcile against the server via api.listSideTasks().
  const [sideTasks, setSideTasks] = useState([])
  // A "wakeup scheduled" toast message the agent just emitted — surfaced as a
  // small banner in the status strip so the user knows the chat will reopen.
  const [scheduledWakeup, setScheduledWakeup] = useState(null)
  // Active autonomous-loop row for this conversation (null when idle). We
  // poll this on mount + after every turn so the Stop-loop banner appears
  // and disappears without needing a dedicated SSE event. Shape mirrors the
  // scheduled_tasks row: {id, prompt, interval_seconds, next_run_at, ...}.
  const [activeLoop, setActiveLoop] = useState(null)

  // Map of convId → AbortController for every in-flight turn. Using a map
  // (rather than a single ref) lets the Stop button target the turn belonging
  // to the chat the user is currently looking at, even when another chat's
  // turn is still running in the background.
  const abortControllersRef = useRef(new Map())
  // Map of convId → { content, thinking } holding the in-progress assistant
  // output / reasoning for every live turn. We mutate these buffers from the
  // stream handler REGARDLESS of which chat the user is currently viewing —
  // otherwise switching away from an ongoing turn and back loses everything
  // streamed while the user was elsewhere (since React state is cleared on
  // conversation change). Cleared on `assistant_message` (iteration commit)
  // and in the stream `finally` (turn ended any way). React state is
  // re-hydrated from this map whenever the user switches INTO a conv that
  // has an in-flight turn, so the thinking block and partial prose persist
  // across chat switches.
  const liveBuffersRef = useRef(new Map())
  const scrollRef = useRef(null)
  // Always reflects the id prop of the latest render. Callbacks that outlive
  // a conversation switch (e.g. `send` awaiting an SSE stream) read this to
  // tell whether their results still apply to the chat the user is on.
  const currentIdRef = useRef(id)
  // Re-attach poller. SSE is per-turn — when the connection dies (server
  // reload, network blip, browser tab change while the agent kept going)
  // there's no way to reopen the same stream. The agent loop keeps writing
  // assistant + tool messages to the DB though, so we poll the conversation
  // every couple seconds while its state is `running` and refresh the UI
  // from the persisted history. Stops automatically when state flips to
  // `idle` / `error`. Map keyed by conv id so the poller for chat A keeps
  // ticking even when the user has switched to chat B.
  const reattachPollersRef = useRef(new Map())
  useEffect(() => {
    currentIdRef.current = id
  }, [id])

  // ----- load conversation + history ----------------------------------------
  useEffect(() => {
    // Switching chats should NOT cancel any in-flight turn — the previous
    // conversation's backend run_turn loop keeps going, and its SSE stream
    // stays open so completion still triggers the post-turn refresh. We
    // hydrate transient live-state from `liveBuffersRef` so that an ongoing
    // turn's thinking block and partial prose persist when the user switches
    // back to it; the stream handler updates those buffers regardless of
    // which chat is active. If the chat we're switching TO has its own
    // in-flight turn (its controller is still registered), restore `busy`
    // so the pending bubble and Stop button reappear immediately.
    const hasInflight = Boolean(id && abortControllersRef.current.get(id))
    setBusy(hasInflight)
    const buf = (id && liveBuffersRef.current.get(id)) || null
    setLiveContent(buf?.content || '')
    setLiveThinking(buf?.thinking || '')

    if (!id) {
      setConv(null)
      setMessages([])
      setToolStates({})
      setTodos([])
      setPendingImages([])
      setPendingQuestion(null)
      setSideTasks([])
      setScheduledWakeup(null)
      setActiveLoop(null)
      return
    }
    let cancelled = false
    // Clear the transient banners/chips the moment the user switches chats —
    // they're always per-conversation and stale wakeup/question state would
    // otherwise bleed between conversations.
    setPendingQuestion(null)
    setSideTasks([])
    setScheduledWakeup(null)
    setActiveLoop(null)
    ;(async () => {
      try {
        const data = await api.getConversation(id)
        if (cancelled) return
        setConv(data.conversation)
        setMessages(data.messages)
        setToolStates(buildToolStatesFromHistory(data.messages))
        setTodos(extractLatestTodos(data.messages))
        // Conversation already running on load — most commonly because
        // the crash-resilience resumer re-launched it before we
        // attached, but also: opening a chat from the sidebar that's
        // mid-turn in a different tab. We have no SSE for this turn,
        // so start the reattach poller — it'll surface new messages
        // as the agent produces them and stop itself when state goes
        // back to idle.
        if (
          data.conversation?.state === 'running'
          && !abortControllersRef.current.has(id)
        ) {
          setBusy(true)
          startReattachPoller(id)
        }
        // Hydrate live buffers from the per-conv ref map so an ongoing turn's
        // accumulated prose/reasoning survives chat switches. API history
        // only returns PERSISTED messages — any in-progress live chunk lives
        // only in the ref map until the next `assistant_message` commit.
        const liveBuf = liveBuffersRef.current.get(id) || null
        setLiveContent(liveBuf?.content || '')
        setLiveThinking(liveBuf?.thinking || '')
        setPendingImages([])
        setError(null)
        // Rehydrate side-task chips for this conversation so they survive a
        // page reload. The endpoint only returns chips the user hasn't yet
        // opened or dismissed.
        try {
          const { side_tasks } = await api.listSideTasks(id)
          if (!cancelled && Array.isArray(side_tasks)) {
            setSideTasks(side_tasks)
          }
        } catch {
          /* non-fatal — chips will repopulate on the next stream event */
        }
        // Rehydrate the autonomous-loop banner so a loop started in an
        // earlier session is still visible after a reload.
        try {
          const { loop } = await api.getConversationLoop(id)
          if (!cancelled) setActiveLoop(loop || null)
        } catch {
          /* non-fatal — the banner just won't render until next refresh */
        }
      } catch (e) {
        if (!cancelled) {
          setError(e.message)
          toast.error('Failed to load conversation', { description: e.message })
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [id])

  // Clean up every reattach poller on full unmount so we don't leak
  // setInterval handles. Per-conv pollers are intentionally kept alive
  // across chat switches (the user may navigate away from a chat that's
  // still mid-resumed-turn and come back to find it up to date) — so we
  // only sweep on unmount, not on every conv id change.
  useEffect(() => {
    return () => {
      for (const handle of reattachPollersRef.current.values()) {
        clearInterval(handle)
      }
      reattachPollersRef.current.clear()
    }
  }, [])

  // ----- auto-scroll bookkeeping -------------------------------------------
  // `nearBottomRef` tracks whether the user is pinned to the latest activity
  // or has scrolled up to read earlier output. Two places observe it:
  //   1. The dependency-driven effect below, which scrolls on new messages /
  //      streaming content / tool-state changes.
  //   2. A ResizeObserver on the inner content that fires when anything
  //      INSIDE a message grows (expanding a tool call, a screenshot image
  //      loading in, a long diff rendering). Without #2 the user would stay
  //      put while a newly-expanded card pushes the bottom off-screen.
  const nearBottomRef = useRef(true)
  const contentRef = useRef(null)
  // Set when the user explicitly moves the viewport away from the bottom
  // (wheel up, touch drag, PageUp/Home). While true, auto-scroll is fully
  // suspended — streaming deltas grow the scroll container but we leave the
  // viewport alone so the user can read earlier output. Cleared only when
  // the user scrolls back within ~32px of the bottom on their own, so a
  // fast-streaming response can't yank them forward.
  const userPinnedUpRef = useRef(false)
  // Timestamp of the last user scroll-related gesture (wheel / touch / key).
  // The auto-scroll effect honours a short grace window after this so a
  // mid-flight gesture isn't fought by a streaming-delta render that
  // arrived first. Without the grace window: each delta re-runs the effect
  // and writes scrollTop=scrollHeight, while the user's wheel event is
  // still queued in the event loop — so the view jumps back to bottom
  // before the wheel handler has a chance to set `userPinnedUpRef`.
  const lastUserGestureAtRef = useRef(0)
  const _GESTURE_GRACE_MS = 250
  // Timestamp of the most recent PROGRAMMATIC scrollTop write. onScroll
  // uses this to distinguish "user reached the bottom on their own"
  // (legitimate signal to release the pin) from "we just auto-scrolled
  // to the bottom on the user's behalf" (NOT a release signal — the
  // user may have just wheeled up and the auto-scroll wiped their
  // pin). Without this, every auto-scroll-to-bottom resets pin=false
  // via onScroll's distanceFromBottom<4 branch, and the user's wheel
  // gesture gets fought every time.
  const lastProgScrollAtRef = useRef(0)
  const _PROG_SCROLL_WINDOW_MS = 150

  // Update nearBottomRef whenever the user scrolls. Using a ref (not state)
  // so we don't cause a re-render on every wheel tick.
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    // Track the previous scrollTop so we can detect a DECREASE — the one
    // signal that unambiguously means "user scrolled up." Content growth
    // leaves scrollTop unchanged (only scrollHeight grows), and our own
    // auto-scroll-to-bottom only ever increases scrollTop. So
    // `scrollTop < previous` is a pure user gesture we can trust even on
    // devices that don't fire wheel events (some touchpads, assistive
    // tech, programmatic input like keyboard via native scroll).
    let lastScrollTop = el.scrollTop
    const onScroll = () => {
      const top = el.scrollTop
      const distanceFromBottom = el.scrollHeight - top - el.clientHeight
      nearBottomRef.current = distanceFromBottom < 200
      // If this scroll event was triggered by our own programmatic
      // write (`el.scrollTop = el.scrollHeight` from the auto-scroll
      // effects), skip the pin updates entirely — programmatic
      // scrolls are not "the user did something", they're our own
      // bookkeeping. Without this gate, an auto-scroll-to-bottom
      // wipes the pin via the `distanceFromBottom < 4` branch, and
      // the user's wheel-up gesture gets yanked back the next render.
      const sinceProg = Date.now() - lastProgScrollAtRef.current
      if (sinceProg < _PROG_SCROLL_WINDOW_MS) {
        lastScrollTop = top
        return
      }
      // Any upward movement sets the pin. We keep the wheel/touch/key
      // handlers below as a belt-and-suspenders path (they fire BEFORE
      // the scroll lands so the pin is set atomically with the gesture),
      // but this covers the gap for any input device those don't see.
      if (top < lastScrollTop - 1) userPinnedUpRef.current = true
      lastScrollTop = top
      // Only release the pin when the user is essentially AT the bottom
      // (a few pixels of rounding slack for high-DPI displays). A looser
      // threshold would cancel the pin the user just set: a trackpad /
      // precision-wheel tick can be as small as 10–20px, so ANY value
      // larger than that would immediately fire-and-clear from a single
      // upward nudge, which then lets the next streaming delta yank the
      // viewport back down. The user has to explicitly return to the
      // bottom to resume auto-follow.
      if (distanceFromBottom < 4) userPinnedUpRef.current = false
    }
    // Any explicit scroll-up gesture pins the viewport. We MUST NOT compute
    // distanceFromBottom here — wheel/keydown fire BEFORE the browser has
    // applied the scroll, so the reading is still ~0 when the user initiates
    // an upward gesture from the bottom. Instead, we pin based on intent
    // (deltaY sign for wheel, specific key for keyboard, any touchmove for
    // mobile) and let onScroll — which fires AFTER the scroll lands — be
    // the sole place that clears the pin when the user reaches the bottom.
    const onWheel = (e) => {
      // Stamp every wheel — even downward — so a streaming auto-scroll
      // doesn't fight a downward scrubbing gesture either; the grace
      // window is symmetric.
      lastUserGestureAtRef.current = Date.now()
      if (e.deltaY < 0) userPinnedUpRef.current = true
    }
    const onTouchMove = () => {
      lastUserGestureAtRef.current = Date.now()
      userPinnedUpRef.current = true
    }
    const onKey = (e) => {
      // Only upward-intent keys pin. Down-intent keys (PageDown/ArrowDown/End)
      // will naturally end at the bottom, which onScroll detects and clears.
      if (
        e.key === 'PageUp' ||
        e.key === 'ArrowUp' ||
        e.key === 'Home'
      ) {
        lastUserGestureAtRef.current = Date.now()
        userPinnedUpRef.current = true
      }
    }
    el.addEventListener('scroll', onScroll, { passive: true })
    el.addEventListener('wheel', onWheel, { passive: true })
    el.addEventListener('touchmove', onTouchMove, { passive: true })
    el.addEventListener('keydown', onKey)
    // Seed the ref with the initial state so the first ResizeObserver fire
    // has something sensible to check against.
    onScroll()
    return () => {
      el.removeEventListener('scroll', onScroll)
      el.removeEventListener('wheel', onWheel)
      el.removeEventListener('touchmove', onTouchMove)
      el.removeEventListener('keydown', onKey)
    }
  }, [])

  // Shared gate for both auto-scroll paths below. Returns true when the
  // viewport should be pinned to the bottom right now: the user hasn't
  // scrolled away, was already near the bottom before this growth, and
  // hasn't initiated a gesture in the very recent past.
  //
  // The gesture-grace window is the load-bearing piece. Without it, a
  // streaming-delta render that lands BEFORE the user's queued wheel
  // event re-runs the effect and writes scrollTop=scrollHeight while
  // the gesture is still in-flight — so the view jumps back to bottom
  // and the user has to wheel up over and over. With it, any wheel /
  // touch / key gesture in the last 250 ms suspends auto-scroll long
  // enough for the gesture to land, the wheel handler to set the pin,
  // and onScroll to update nearBottomRef. After the grace window
  // expires, if the user ended up at the bottom the pin is cleared and
  // auto-scroll resumes; if they stopped above, the pin stays and
  // auto-scroll stays off until they scroll back to the bottom.
  const _shouldAutoScroll = () => {
    if (userPinnedUpRef.current) return false
    if (!nearBottomRef.current) return false
    if (Date.now() - lastUserGestureAtRef.current < _GESTURE_GRACE_MS) return false
    return true
  }

  // Re-pin to bottom any time the inner content grows and the user was
  // near the bottom before the growth. This covers tool-card expansion,
  // async image loads, and anything else that changes layout outside of
  // React's message state.
  useEffect(() => {
    const scroll = scrollRef.current
    const content = contentRef.current
    if (!scroll || !content || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => {
      if (_shouldAutoScroll()) {
        // Stamp BEFORE the assignment so the scroll event the
        // browser will fire next finds the marker in place.
        lastProgScrollAtRef.current = Date.now()
        scroll.scrollTop = scroll.scrollHeight
      }
    })
    ro.observe(content)
    return () => ro.disconnect()
  }, [])

  // Dependency-driven scroll: fires on new messages, streaming deltas,
  // tool-state transitions, and busy flips. Complements the
  // ResizeObserver above; together they keep the viewport pinned to the
  // latest activity whenever the user hasn't manually scrolled away.
  //
  // We DO want `liveContent` / `liveThinking` in the deps: the
  // ResizeObserver alone misses streaming-thinking growth, because the
  // ThinkingBlock <pre> is `max-h-60 overflow-auto` — its inner
  // scrollHeight grows but the outer container's size is capped, so the
  // observer never fires. Without this effect ticking on every token
  // the viewport would just sit there while reasoning streams in.
  //
  // The race condition the dep-list previously caused (auto-scroll
  // landing between a user's wheel event and the browser's scroll) is
  // handled by `_shouldAutoScroll`'s gesture-grace window above.
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    if (_shouldAutoScroll()) {
      // Mark as programmatic — see lastProgScrollAtRef comment for why
      // onScroll needs to discriminate this from user-initiated
      // scrolls (otherwise auto-scrolling to the bottom wipes the
      // pin the user just set with a wheel-up gesture).
      lastProgScrollAtRef.current = Date.now()
      el.scrollTop = el.scrollHeight
    }
  }, [messages, liveContent, liveThinking, toolStates, busy])

  // Scroll to a specific message and briefly flash it. Triggered when the
  // user clicks a semantic-search hit in the sidebar — we land inside the
  // conversation at the exact message so they don't have to hunt for it.
  // `highlightedMessageId` drives the CSS class that paints the ring for ~2s.
  const [highlightedMessageId, setHighlightedMessageId] = useState(null)
  useEffect(() => {
    if (!jumpToMessageId || !messages.length) return
    // Let the DOM settle after the conversation load before we hunt for the
    // element — messages render in the same frame, but the outer scroller
    // needs one paint to have a real scrollHeight.
    const raf = requestAnimationFrame(() => {
      const node = document.querySelector(
        `[data-message-id="${cssEscape(jumpToMessageId)}"]`,
      )
      if (node && typeof node.scrollIntoView === 'function') {
        node.scrollIntoView({ behavior: 'smooth', block: 'center' })
        // Suppress the auto-scroll-to-bottom for this jump — the user wants
        // to stay where they landed, not get yanked to the newest message.
        nearBottomRef.current = false
        setHighlightedMessageId(jumpToMessageId)
      }
      onJumpHandled?.()
    })
    return () => cancelAnimationFrame(raf)
  }, [jumpToMessageId, messages, onJumpHandled])

  // Fade the highlight out after a couple seconds so the page doesn't look
  // stuck on a yellow ring forever.
  useEffect(() => {
    if (!highlightedMessageId) return
    const t = setTimeout(() => setHighlightedMessageId(null), 2400)
    return () => clearTimeout(t)
  }, [highlightedMessageId])

  // Scroll helper exposed to ChatHeader → ConversationActions (Pinned dialog).
  // Same DOM contract as the jumpToMessageId effect above, but callable
  // synchronously after the dialog closes.
  const scrollToMessage = useCallback((mid) => {
    if (!mid) return
    requestAnimationFrame(() => {
      const node = document.querySelector(
        `[data-message-id="${cssEscape(mid)}"]`,
      )
      if (node && typeof node.scrollIntoView === 'function') {
        node.scrollIntoView({ behavior: 'smooth', block: 'center' })
        nearBottomRef.current = false
        setHighlightedMessageId(mid)
      }
    })
  }, [])

  // ----- send a message / run a turn ----------------------------------------
  // `url` defaults to the standard send endpoint, but the edit-and-regenerate
  // flow passes a different URL (.../messages/{mid}/edit) — same SSE shape,
  // so we can reuse this whole pipeline. `skipOptimistic` is set by edit-
  // and-regenerate because that caller has ALREADY rewritten the existing
  // user row in place — appending another temp row would duplicate the
  // prompt visibly until the server refresh arrives.
  // ----- reattach poller ----------------------------------------------------
  // Spin up a 2.5 s poll that refreshes the conversation while its state is
  // `running` but we don't have a live SSE connection (server reloaded,
  // network blip, conversation was already mid-turn when we opened it,
  // crash-resilience resumer kicked in). Stops automatically once state
  // flips to idle/error. Idempotent — calling twice for the same conv is a
  // no-op.
  const stopReattachPoller = useCallback((cid) => {
    const handle = reattachPollersRef.current.get(cid)
    if (!handle) return
    clearInterval(handle)
    reattachPollersRef.current.delete(cid)
  }, [])

  const startReattachPoller = useCallback((cid) => {
    if (!cid) return
    if (reattachPollersRef.current.has(cid)) return  // already polling
    let stopped = false
    const tick = async () => {
      if (stopped) return
      try {
        const fresh = await api.getConversation(cid)
        // User switched away — don't push state into the visible chat,
        // but keep polling so the buffer stays warm in case they return.
        const isCurrent = currentIdRef.current === cid
        if (isCurrent) {
          setConv(fresh.conversation)
          setMessages(fresh.messages)
          setToolStates(buildToolStatesFromHistory(fresh.messages))
        }
        const st = fresh.conversation?.state
        if (st !== 'running') {
          stopped = true
          stopReattachPoller(cid)
          if (isCurrent) {
            setBusy(false)
            // Live buffers are partial-streamed prose that the server
            // never persisted (the SSE that produced them died). The
            // refreshed history is now the source of truth, so any
            // partial buffer is stale — clear it.
            setLiveContent('')
            setLiveThinking('')
            liveBuffersRef.current.delete(cid)
          }
        }
      } catch {
        /* transient — keep polling */
      }
    }
    // Fire immediately, then on a 2.5 s cadence. The first tick gives the
    // user near-instant feedback that we noticed the running state; the
    // interval picks up subsequent message commits.
    tick()
    const handle = setInterval(tick, 2500)
    reattachPollersRef.current.set(cid, handle)
  }, [stopReattachPoller])

  const send = useCallback(
    async ({ newUserText, images, url, skipOptimistic = false } = {}) => {
      if (!conv) return
      const controller = new AbortController()
      abortControllersRef.current.set(conv.id, controller)
      setBusy(true)

      // If there's a user message, optimistically append it so the UI feels
      // responsive even before the server echoes it back. It gets a temp id
      // that we reconcile when the next refresh arrives from the server.
      if (!skipOptimistic && (newUserText || (images && images.length))) {
        setMessages((m) => [
          ...m,
          {
            id: 'tmp-' + Date.now(),
            role: 'user',
            content: newUserText || '',
            tool_calls: [],
            images: (images || []).map((im) => im.name),
            created_at: Date.now() / 1000,
          },
        ])
      }

      // Pin the conversation id this stream's events belong to. The guarded
      // handler below drops every event the backend emits while the user is
      // looking at a different chat — so a long-running turn in chat A can
      // keep generating in the background without polluting chat B's view.
      // Live-content buffers, however, are maintained per-convId regardless
      // of which chat is visible, so switching back to an ongoing turn
      // re-hydrates its partial prose and thinking block (see the chat-
      // switch effect above).
      const turnConvId = conv.id
      const bumpBuffer = (kind, text) => {
        if (!text) return
        const map = liveBuffersRef.current
        const buf = map.get(turnConvId) || { content: '', thinking: '' }
        buf[kind] = (buf[kind] || '') + text
        map.set(turnConvId, buf)
      }
      const guardedHandler = (evt) => {
        // Keep the off-screen buffer in sync so a chat switch can restore it.
        // delta → assistant prose token; thinking → reasoning token;
        // assistant_message → iteration's row committed, buffer no longer
        // needed (the persisted row is the source of truth going forward).
        if (evt.type === 'delta') bumpBuffer('content', evt.text)
        else if (evt.type === 'thinking') bumpBuffer('thinking', evt.text)
        else if (evt.type === 'assistant_message') {
          liveBuffersRef.current.delete(turnConvId)
        }
        if (currentIdRef.current !== turnConvId) return
        handleEvent(evt)
      }

      try {
        await postEventStream({
          url: url || `/api/conversations/${conv.id}/messages`,
          body: {
            content: newUserText || '',
            images: (images || []).map((im) => im.name),
          },
          signal: controller.signal,
          onEvent: guardedHandler,
        })
      } catch (e) {
        if (e.name !== 'AbortError') {
          toast.error('Stream error', { description: e.message })
        }
      } finally {
        // If the user switched conversations mid-turn, skip all the state
        // writes below — the new conversation's own load effect is already
        // in charge, and stomping its state with rows from the previous
        // chat would resurrect the cross-chat bleed the guarded handler
        // is meant to prevent.
        const stillActive = conv?.id === currentIdRef.current
        if (stillActive) {
          setLiveContent('')
          setLiveThinking('')
          // `setBusy(false)` is deferred until after we've checked the
          // server's view of the conversation: if the SSE died (server
          // reload, network blip) but the backend kept producing
          // messages — i.e. state is still `running` — we want the busy
          // indicator to stay up while the reattach poller catches up.
        }
        // The stream is over — drop the persistent buffer. If a final
        // `assistant_message` fired it's already gone; this clears the
        // abnormal-exit cases (abort, error, client unmount) too.
        liveBuffersRef.current.delete(conv.id)
        // Only drop the entry if it's still ours — a later turn in the same
        // chat (queue + re-run) may have registered a newer controller under
        // the same id, and we don't want to evict it.
        if (abortControllersRef.current.get(conv.id) === controller) {
          abortControllersRef.current.delete(conv.id)
        }
        if (!stillActive) return
        // Refresh conversation meta so updated_at ordering reflects this turn,
        // and the auto-title handler sees the persisted state.
        let resumedRunning = false
        try {
          const fresh = await api.getConversation(conv.id)
          setConv(fresh.conversation)
          setMessages(fresh.messages)
          setToolStates(buildToolStatesFromHistory(fresh.messages))
          onConversationUpdated?.(fresh.conversation)
          maybeAutoTitle(fresh.conversation, fresh.messages)
          // The SSE loop ended but the server still says the turn is
          // running — that's the "uvicorn reloaded mid-turn, the
          // crash-resilience resumer kicked in, we missed the rest of
          // the stream" case. Spin up the reattach poller so the
          // browser can show new messages as they land in the DB.
          resumedRunning = fresh.conversation?.state === 'running'
          if (resumedRunning) {
            startReattachPoller(conv.id)
          } else {
            setBusy(false)
          }
        } catch {
          // Even on a refetch failure we can't know whether the server
          // is still working, so be conservative and clear busy. The
          // user can refresh to recover.
          setBusy(false)
        }
        // Re-check the autonomous loop so the banner flips on/off when the
        // agent just called start_loop or stop_loop during this turn.
        try {
          const { loop } = await api.getConversationLoop(conv.id)
          setActiveLoop(loop || null)
        } catch {
          /* non-fatal — banner state will settle on next refresh */
        }
      }
    },
    [conv, onConversationUpdated, startReattachPoller],
  )

  // Queue a follow-up message against the currently running turn. Unlike
  // `send`, this does NOT open a new SSE stream — the active one will emit
  // a `user_message_added` event when the agent loop drains the queue.
  // We optimistically render the queued message as a temp row so the user
  // sees it land immediately; the temp row is replaced by the real one on
  // the next history refresh.
  const queue = useCallback(
    async ({ newUserText, images } = {}) => {
      if (!conv) return
      const tmpId = 'tmp-' + Date.now()
      setMessages((m) => [
        ...m,
        {
          id: tmpId,
          role: 'user',
          content: newUserText || '',
          tool_calls: [],
          images: (images || []).map((im) => im.name),
          created_at: Date.now() / 1000,
          _queued: true, // visual hint — see Message render path
        },
      ])
      try {
        await api.queueMessage(conv.id, {
          content: newUserText || '',
          images: (images || []).map((im) => im.name),
        })
      } catch (e) {
        // Roll back the optimistic row and surface the failure.
        setMessages((m) => m.filter((x) => x.id !== tmpId))
        toast.error('Could not queue message', { description: e.message })
      }
    },
    [conv],
  )

  // ----- event handler ------------------------------------------------------
  function handleEvent(evt) {
    switch (evt.type) {
      case 'delta':
        setLiveContent((s) => s + (evt.text || ''))
        break
      case 'thinking':
        // Accumulate reasoning tokens so the UI can show "Thinking…" progress
        // instead of a blank wait. Not all models emit these; for models that
        // don't, the pending bubble still shows a pulse dot so the user at
        // least sees *some* activity during the Ollama round-trip.
        setLiveThinking((s) => s + (evt.text || ''))
        break
      case 'assistant_message':
        // The server has persisted this iteration's assistant row. Commit
        // the live content into a real message bubble and clear the buffer.
        setMessages((m) => [
          ...m,
          {
            id: evt.id,
            role: 'assistant',
            content: evt.content || '',
            tool_calls: evt.tool_calls || [],
            created_at: Date.now() / 1000,
          },
        ])
        setLiveContent('')
        setLiveThinking('')
        break
      case 'user_message_added':
        // A queued follow-up message was just persisted by the agent loop.
        // Drop any matching optimistic temp row (matched on content+role)
        // and append the canonical one with its real DB id. Without the
        // dedupe step we'd briefly show the same message twice — once as
        // the optimistic temp and once as the server-confirmed row.
        setMessages((m) => {
          const idx = m.findIndex(
            (x) =>
              x._queued && x.role === 'user' && x.content === (evt.content || ''),
          )
          const cleaned = idx === -1 ? m : [...m.slice(0, idx), ...m.slice(idx + 1)]
          return [
            ...cleaned,
            {
              id: evt.id,
              role: 'user',
              content: evt.content || '',
              images: evt.images || [],
              tool_calls: [],
              created_at: evt.created_at || Date.now() / 1000,
            },
          ]
        })
        break
      case 'tool_call':
        setToolStates((s) => ({
          ...s,
          [evt.id]: {
            ...(s[evt.id] || {}),
            status: 'running',
            reason: evt.reason || '',
            args: evt.args || {},
            name: evt.name,
            label: evt.label,
          },
        }))
        break
      case 'await_approval':
        setToolStates((s) => ({
          ...s,
          [evt.id]: {
            ...(s[evt.id] || {}),
            status: 'await',
            approvalId: evt.id,
            reason: evt.reason || '',
            args: evt.args || {},
            preview: evt.preview || null,
            name: evt.name,
            label: evt.label,
          },
        }))
        break
      case 'await_user_answer':
        // AskUserQuestion tool — agent is blocked on a multi-choice click.
        setPendingQuestion({
          id: evt.id,
          question: evt.question || '',
          options: Array.isArray(evt.options) ? evt.options : [],
        })
        break
      case 'side_task_flagged':
        // spawn_task tool just flagged a drive-by issue; show the chip live
        // without refetching.
        if (evt.side_task && evt.side_task.id) {
          setSideTasks((prev) => {
            if (prev.some((x) => x.id === evt.side_task.id)) return prev
            return [...prev, evt.side_task]
          })
        }
        break
      case 'wakeup_scheduled':
        setScheduledWakeup({
          id: evt.id,
          message: evt.output || 'Scheduled wakeup.',
        })
        toast.success('Wakeup scheduled', {
          description: evt.output || 'The agent will continue this chat later.',
        })
        break
      case 'worktrees_changed':
        toast.success('Worktrees changed', {
          description: 'The agent created or removed a git worktree.',
        })
        break
      case 'tool_result':
        setToolStates((s) => ({
          ...s,
          [evt.id]: {
            ...(s[evt.id] || {}),
            status: evt.error === 'rejected by user' ? 'rejected' : 'done',
            result: { ok: evt.ok, output: evt.output, error: evt.error },
            imagePath: evt.image_path || null,
          },
        }))
        break
      // ---- subagent progress events -------------------------------------
      // Fired by run_subagent / run_subagents_parallel while a `delegate`
      // tool is in flight. We key them under the parent delegate's tool
      // call id so the ToolCall card can render a nested activity list.
      case 'subagent_started':
      case 'subagent_tool_call':
      case 'subagent_tool_result':
      case 'subagent_done': {
        const parentId = evt.parent_tool_call_id
        if (!parentId) break
        const subId = evt.subagent_id
        if (!subId) break
        setToolStates((s) => {
          const prev = s[parentId] || {}
          const prevSubs = prev.subagents || {}
          const prevSub = prevSubs[subId] || { steps: [] }
          let nextSub = prevSub
          if (evt.type === 'subagent_started') {
            nextSub = {
              ...prevSub,
              status: 'running',
              subagentType: evt.subagent_type || 'general',
              task: evt.task || '',
              steps: prevSub.steps || [],
            }
          } else if (evt.type === 'subagent_tool_call') {
            nextSub = {
              ...prevSub,
              status: 'running',
              steps: [
                ...(prevSub.steps || []),
                {
                  id: evt.tool_call_id,
                  name: evt.name,
                  label: evt.label,
                  status: 'running',
                },
              ],
            }
          } else if (evt.type === 'subagent_tool_result') {
            nextSub = {
              ...prevSub,
              steps: (prevSub.steps || []).map((st) =>
                st.id === evt.tool_call_id
                  ? { ...st, status: evt.ok ? 'done' : 'failed', error: evt.error || null }
                  : st,
              ),
            }
          } else if (evt.type === 'subagent_done') {
            nextSub = {
              ...prevSub,
              status: evt.ok ? 'done' : 'failed',
              stepCount: evt.steps ?? (prevSub.steps?.length || 0),
              summary: evt.summary || '',
              error: evt.error || null,
            }
          }
          return {
            ...s,
            [parentId]: {
              ...prev,
              subagents: { ...prevSubs, [subId]: nextSub },
            },
          }
        })
        break
      }
      case 'todos_updated':
        setTodos(Array.isArray(evt.todos) ? evt.todos : [])
        break
      case 'stream_retry':
        // Server is silently retrying the current reply (e.g. a thinking-
        // only model produced no visible content). Clear the live buffers
        // so the retry's deltas render into a fresh bubble — otherwise
        // any partial text would be prepended to the corrected response
        // and persist forever in the user's transcript.
        setLiveContent('')
        setLiveThinking('')
        break
      case 'turn_done':
        // nothing to do; the finally block handles cleanup
        break
      case 'hook_ran': {
        // A lifecycle hook fired. Surface it as a non-blocking toast so the
        // user can confirm their configured shell command actually ran — and
        // can see the failure if it didn't. We deliberately don't render
        // hook output inline in the transcript; that would clutter the
        // conversation. The backend already injects hook stdout into the
        // model's context via system notes.
        const ev = evt.event || 'hook'
        const label = evt.command ? `${ev}: ${String(evt.command).slice(0, 60)}` : ev
        if (evt.ok) {
          toast.success(`hook ${ev}`, { description: label })
        } else {
          toast.error(`hook ${ev} failed`, {
            description: evt.error || label,
          })
        }
        break
      }
      case 'error':
        toast.error('Agent error', { description: evt.message })
        break
      default:
        break
    }
  }

  // ----- approval buttons ---------------------------------------------------
  // The original SSE stream is still open and blocked inside the server,
  // awaiting this decision. POST /approve resolves the backend's future and
  // the stream resumes on its own — we only optimistically update UI here.
  async function approve(callId, approved) {
    try {
      await api.approve(conv.id, callId, approved)
      setToolStates((s) => ({
        ...s,
        [callId]: {
          ...(s[callId] || {}),
          status: approved ? 'running' : 'rejected',
        },
      }))
    } catch (e) {
      toast.error('Approval failed', { description: e.message })
    }
  }

  // ----- pin / unpin a message ----------------------------------------------
  // Pinned messages are exempt from automatic compaction on the server, so
  // the model keeps seeing them even in very long conversations. We update
  // local state optimistically; any failure toasts and reverts.
  async function togglePin(messageId, nextPinned) {
    // Optimistic update first — users expect instant visual feedback.
    setMessages((m) =>
      m.map((x) => (x.id === messageId ? { ...x, pinned: nextPinned } : x)),
    )
    try {
      await api.pinMessage(conv.id, messageId, nextPinned)
    } catch (e) {
      // Roll back and surface the error via Sonner.
      setMessages((m) =>
        m.map((x) => (x.id === messageId ? { ...x, pinned: !nextPinned } : x)),
      )
      toast.error('Could not update pin', { description: e.message })
    }
  }

  // ----- auto-title ---------------------------------------------------------
  // If the conversation is still called "New chat" and the user has sent at
  // least one message, derive a title from the first 40 chars of that message.
  async function maybeAutoTitle(c, msgs) {
    if (c.title !== 'New chat') return
    const firstUser = msgs.find((m) => m.role === 'user')
    if (!firstUser) return
    const title = firstUser.content.trim().slice(0, 48).replace(/\s+/g, ' ') || 'Chat'
    try {
      const { conversation } = await api.updateConversation(c.id, { title })
      setConv(conversation)
      onConversationUpdated?.(conversation)
    } catch {
      /* non-fatal */
    }
  }

  // ----- submit handler -----------------------------------------------------
  // When the agent is idle this starts a new turn; when it's busy it appends
  // to the in-flight turn's queue instead of opening a parallel stream.
  //
  // Attachments split in two ways at submit time:
  //   - Images   → passed as `images[]` so the backend can attach them as
  //                multimodal input to models that support it.
  //   - Documents → their already-extracted text is prepended to the user's
  //                 message body (wrapped in a visible header block) so
  //                 text-only models can still reason over the contents.
  function onSend() {
    const text = input.trim()
    if (!text && pendingImages.length === 0) return

    const imagesOnly = pendingImages.filter((a) => a.kind !== 'document')
    const documents = pendingImages.filter((a) => a.kind === 'document')

    let finalText = text
    if (documents.length) {
      const blocks = documents.map(formatDocumentBlock).join('\n\n')
      finalText = blocks + (text ? `\n\n${text}` : '')
    }

    setInput('')
    setPendingImages([])
    if (busy) {
      queue({ newUserText: finalText, images: imagesOnly })
    } else {
      send({ newUserText: finalText, images: imagesOnly })
    }
  }

  // Stop the in-flight turn. We do two things in parallel:
  //   1. Tell the backend to set its stop flag. The agent loop polls this
  //      between Ollama chunks / before tool dispatch, so the local model
  //      actually stops generating — just closing the SSE connection isn't
  //      enough, since the server's HTTP request to Ollama keeps running.
  //   2. Abort the local fetch so the SSE reader unblocks and `busy` flips
  //      to false in the finally block of `send`.
  function onStop() {
    if (conv) {
      // Fire-and-forget — the user already sees the effect when the
      // stream ends. Failures are silent (the abort below still happens).
      api.stopTurn(conv.id).catch(() => {
        /* non-fatal — abort below is the safety net */
      })
      // Target this chat's controller specifically — other chats may have
      // in-flight turns of their own, and we don't want Stop here to abort
      // a background turn the user started elsewhere.
      abortControllersRef.current.get(conv.id)?.abort()
    }
  }

  // Shared conversation-patch helper used by both the header (title, cwd,
  // permission mode) and the composer's model picker. Keeping the mutation
  // in one place means state updates feel identical whichever control the
  // user touches.
  async function patchConversation(body) {
    if (!conv) return
    try {
      const { conversation } = await api.updateConversation(conv.id, body)
      setConv(conversation)
      onConversationUpdated?.(conversation)
    } catch (e) {
      toast.error('Update failed', { description: e.message })
    }
  }

  // ----- edit-and-regenerate ------------------------------------------------
  // Hits the dedicated /messages/{mid}/edit endpoint which atomically rewrites
  // the user row, drops everything after it, and starts a new SSE stream.
  // Disabled while the agent is busy — editing in the middle of a tool call
  // would corrupt the in-flight assistant/tool sequence.
  //
  // This is invoked unconditionally from the message rendering loop (see
  // `onEdit={...}` below). We guard `busy` HERE rather than in the prop
  // gate so the Save button in an already-open editor never silently
  // no-ops when the parent's `busy` flips mid-edit.
  async function editUserMessage(messageId, newText) {
    // Surface "no conversation loaded" instead of silently returning. If
    // the user clicks Save & regenerate and the only feedback is "nothing
    // happened", they have no way to know whether the click reached the
    // handler at all — so we toast every refusal path explicitly.
    if (!conv) {
      toast.error('Could not save edit', {
        description: 'No conversation is loaded. Try clicking the conversation in the sidebar again.',
      })
      return
    }
    if (!messageId || typeof messageId !== 'string' || messageId.startsWith('tmp-')) {
      toast.error('Could not save edit', {
        description: 'This message hasn\'t finished saving yet — wait a moment and retry.',
      })
      return
    }
    if (busy) {
      toast.error('Cannot regenerate right now', {
        description:
          'The agent is still running. Stop the current turn (Esc) and try again.',
      })
      return
    }
    // Optimistically rewrite the local row and drop messages after it; the
    // SSE stream will push the new assistant reply into place. If findIndex
    // misses the row (could happen if `messages` raced with a server refresh
    // mid-click) we still go through to the server — the backend has the
    // canonical state and will 404 with a clear error if the id is wrong,
    // which `send`'s catch will surface as a toast.
    let optimisticHit = false
    setMessages((m) => {
      const idx = m.findIndex((x) => x.id === messageId)
      if (idx === -1) return m
      optimisticHit = true
      const updated = { ...m[idx], content: newText }
      return [...m.slice(0, idx), updated]
    })
    if (!optimisticHit) {
      // Not fatal — keep going so the server still gets the edit — but warn
      // so the user sees the row will only update after the post-stream
      // refresh, not instantly.
      // eslint-disable-next-line no-console
      console.warn(
        '[edit-and-regenerate] message id not found in local state',
        { messageId, knownIds: messages.map((m) => m.id) },
      )
    }
    setToolStates({})
    setLiveContent('')
    setLiveThinking('')
    await send({
      newUserText: newText,
      images: [],
      url: `/api/conversations/${conv.id}/messages/${messageId}/edit`,
      // We already rewrote the user row in place — don't let `send` append
      // a duplicate temp row on top.
      skipOptimistic: true,
    })
  }

  // ----- attachment upload handling ----------------------------------------
  /**
   * Upload a user-pasted image or document to the backend and store the
   * response so it can be attached to the next turn. The server branches on
   * the file's Content-Type; we just forward whatever the user drops.
   *
   * Unsupported MIME types surface as a toast instead of a silent drop so the
   * user knows their file wasn't added.
   */
  async function handleImageFiles(files) {
    if (!files || !files.length || !conv) return
    for (const file of files) {
      try {
        const resp = await api.uploadAttachment(conv.id, file)
        // For images we keep a local preview URL so the chip can render the
        // thumbnail without a server round-trip. Documents don't need one.
        const previewUrl =
          resp.kind === 'image' ? URL.createObjectURL(file) : null
        setPendingImages((list) => [
          ...list,
          { ...resp, previewUrl },
        ])
        if (resp.extract_error) {
          toast.warning('Attached file (text extraction failed)', {
            description: resp.extract_error,
          })
        } else if (resp.truncated) {
          toast.warning('Attached document was truncated', {
            description: `Only the first ${Math.round(
              (resp.extracted_text || '').length / 1000,
            )}k characters will be sent to the model.`,
          })
        }
      } catch (e) {
        toast.error('Upload failed', { description: e.message })
      }
    }
  }

  function removePendingImage(name) {
    setPendingImages((list) => {
      const hit = list.find((im) => im.name === name)
      if (hit?.previewUrl) {
        // Revoke the blob URL so we don't leak memory across many pastes.
        try {
          URL.revokeObjectURL(hit.previewUrl)
        } catch {
          /* ignore */
        }
      }
      return list.filter((im) => im.name !== name)
    })
  }

  // ----- AskUserQuestion click handler --------------------------------------
  // The agent is blocked in its own SSE stream on a future keyed by `id`.
  // Posting the answer resolves the future; we only need to clear the
  // pending state here — the stream will continue and surface the agent's
  // next message on its own.
  async function answerQuestion(answerId, value) {
    if (!conv) return
    try {
      await api.answerQuestion(conv.id, answerId, value)
      setPendingQuestion(null)
    } catch (e) {
      toast.error('Could not send answer', { description: e.message })
    }
  }

  // ----- side-task chip actions ---------------------------------------------
  // "Open" converts the chip into a fresh conversation seeded with the
  // spawn_task prompt; we drop the chip locally and notify the outer shell so
  // the sidebar updates with the new conversation.
  async function openSideTask(task) {
    if (!conv) return
    try {
      const { conversation } = await api.openSideTask(task.id, {
        cwd: conv.cwd,
        model: conv.model,
      })
      setSideTasks((prev) => prev.filter((x) => x.id !== task.id))
      toast.success('Side task opened', {
        description: conversation?.title || task.title,
      })
      onConversationUpdated?.(conversation)
    } catch (e) {
      toast.error('Could not open side task', { description: e.message })
    }
  }

  async function dismissSideTask(task) {
    try {
      await api.dismissSideTask(task.id)
      setSideTasks((prev) => prev.filter((x) => x.id !== task.id))
    } catch (e) {
      toast.error('Could not dismiss side task', { description: e.message })
    }
  }

  // ----- autonomous loop control -------------------------------------------
  // The Stop-loop button in the status banner calls the HTTP endpoint which
  // deletes the scheduled-task row. We optimistically clear the local banner
  // so the UI feels instant; a failed request rolls back via the refetch.
  async function stopLoop() {
    if (!conv) return
    const was = activeLoop
    setActiveLoop(null)
    try {
      await api.stopConversationLoop(conv.id)
      toast.success('Loop stopped')
    } catch (e) {
      setActiveLoop(was) // rollback — loop still running on the server
      toast.error('Could not stop loop', { description: e.message })
    }
  }

  // ----- execute plan -------------------------------------------------------
  // Plan mode is a two-phase protocol: the agent produces a plan ending in
  // [PLAN READY], the user clicks this button, the backend flips the conv to
  // approve_edits and enqueues the plan as the next user turn so execution
  // replays with writes enabled.
  async function executePlan() {
    if (!conv) return
    try {
      const { conversation } = await api.executePlan(conv.id)
      setConv(conversation)
      onConversationUpdated?.(conversation)
      toast.success('Executing plan', {
        description: 'Switched to approve_edits and queued the plan for execution.',
      })
    } catch (e) {
      toast.error('Could not execute plan', { description: e.message })
    }
  }

  // ----- render -------------------------------------------------------------
  if (!id) {
    return <EmptyState onOpenSidebar={onOpenSidebar} />
  }

  if (!conv) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        {error ? error : 'Loading…'}
      </div>
    )
  }

  // Only render user + assistant messages. Tool messages are merged into
  // their parent assistant bubble via the toolStates map. Filter is cheap;
  // no memoization needed.
  const visibleMessages = messages.filter(
    (m) => m.role === 'user' || m.role === 'assistant',
  )

  // Find the id of the most recent user message — we only expose the edit
  // affordance there. Editing an arbitrary historical message would orphan
  // the assistant replies that came AFTER it; restricting the action to the
  // latest user row keeps the UX honest about what the button does.
  const lastUserMessageId = (() => {
    for (let i = visibleMessages.length - 1; i >= 0; i--) {
      if (visibleMessages[i].role === 'user' && typeof visibleMessages[i].id === 'string') {
        return visibleMessages[i].id
      }
    }
    return null
  })()

  // "Execute plan" is offered only when the user is in plan mode AND the
  // latest assistant message ends with our `[PLAN READY]` sentinel. The
  // backend checks `endswith("[PLAN READY]")` on the same message before
  // enqueuing the replay, so we mirror the exact rule here — otherwise the
  // button would show up for plans the backend would then refuse to execute.
  const planReady = (() => {
    if (conv?.permission_mode !== 'plan') return false
    if (busy) return false
    for (let i = visibleMessages.length - 1; i >= 0; i--) {
      const m = visibleMessages[i]
      if (m.role === 'assistant') {
        if (typeof m.content !== 'string') return false
        return m.content.trim().endsWith('[PLAN READY]')
      }
    }
    return false
  })()

  return (
    // Outer split: chat column on the left (fills available width),
    // ActivityPanel on the right. Activity panel hides itself on screens
    // below `lg` (its own className), so on mobile the chat takes the
    // full width — same UX as before.
    <div className="flex h-full min-h-0 w-full">
    <div className="flex h-full min-h-0 min-w-0 flex-1 flex-col">
      <ChatHeader
        conv={conv}
        messages={messages}
        onUpdate={(c) => {
          setConv(c)
          onConversationUpdated?.(c)
        }}
        onOpenSidebar={onOpenSidebar}
        onScrollToMessage={scrollToMessage}
      />

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-3 md:px-6"
      >
        <div ref={contentRef} className="mx-auto max-w-3xl py-4">
          {visibleMessages.length === 0 && !liveContent && (
            <WelcomeTips cwd={conv.cwd} />
          )}

          {visibleMessages.map((m) => (
            // data-message-id lets the semantic-search jump logic find this
            // row via querySelector. The ring class flashes for ~2s after
            // a jump so the user's eye lands on the right message.
            <div
              key={m.id}
              data-message-id={m.id}
              className={cn(
                'rounded-md transition-shadow duration-500',
                highlightedMessageId === m.id &&
                  'ring-2 ring-amber-400/70 ring-offset-2 ring-offset-background',
              )}
            >
              <Message
                role={m.role}
                content={m.content}
                images={m.images}
                pinned={!!m.pinned}
                queued={!!m._queued}
                onTogglePin={
                  typeof m.id === 'string' && !m.id.startsWith('tmp-')
                    ? () => togglePin(m.id, !m.pinned)
                    : undefined
                }
                // `onEdit` is passed for ANY persisted user row — the
                // callback itself checks `busy` and toasts if the caller
                // shouldn't be editing right now. Gating it at the prop
                // level (the previous behaviour) meant a `busy` flip while
                // the user was mid-edit would swap the callback to
                // `undefined` under them and turn Save into a silent no-op.
                onEdit={
                  m.role === 'user' &&
                  typeof m.id === 'string' &&
                  !m.id.startsWith('tmp-')
                    ? (next) => editUserMessage(m.id, next)
                    : undefined
                }
                // Whether the Edit pencil should be OFFERED right now.
                // This is the "can they start an edit?" gate — distinct
                // from `onEdit`, which controls whether an in-progress
                // edit can be saved.
                canEdit={
                  m.role === 'user' &&
                  m.id === lastUserMessageId &&
                  !busy &&
                  typeof m.id === 'string' &&
                  !m.id.startsWith('tmp-')
                }
                onOpenArtifact={
                  m.role === 'assistant'
                    ? (artifact) =>
                        setActiveArtifact({ ...artifact, id: m.id })
                    : undefined
                }
              >
                {m.role === 'assistant' &&
                  (m.tool_calls || []).map((tc) => {
                    const st = toolStates[tc.id] || { status: 'done' }
                    const call = {
                      ...tc,
                      reason: st.reason || tc.reason || '',
                      args: st.args || tc.args || {},
                      preview: st.preview || null,
                    }
                    return (
                      <ToolCall
                        key={tc.id}
                        call={call}
                        status={st.status}
                        result={st.result}
                        imagePath={st.imagePath}
                        subagents={st.subagents}
                        onApprove={() => approve(tc.id, true)}
                        onReject={() => approve(tc.id, false)}
                      />
                    )
                  })}
              </Message>
            </div>
          ))}

          {(busy || liveContent || liveThinking) && (
            <PendingAssistantBubble
              content={liveContent}
              thinking={liveThinking}
              busy={busy}
            />
          )}
        </div>
      </div>

      <TodoPanel todos={todos} />

      <StatusStrip
        planReady={planReady}
        onExecutePlan={executePlan}
        scheduledWakeup={scheduledWakeup}
        onDismissWakeup={() => setScheduledWakeup(null)}
        pendingQuestion={pendingQuestion}
        onAnswerQuestion={answerQuestion}
        sideTasks={sideTasks}
        onOpenSideTask={openSideTask}
        onDismissSideTask={dismissSideTask}
        activeLoop={activeLoop}
        onStopLoop={stopLoop}
      />

      <ChatInput
        value={input}
        onChange={setInput}
        onSend={onSend}
        busy={busy}
        onStop={onStop}
        pendingImages={pendingImages}
        onImages={handleImageFiles}
        onRemoveImage={removePendingImage}
        conv={conv}
        models={models}
        showAllModels={showAllModels}
        onToggleShowAllModels={onToggleShowAllModels}
        onPatch={patchConversation}
      />
    </div>

      {/* Right-side activity panel — hides itself on screens below `lg`
          (responsive class lives inside the component) so mobile is
          unaffected. Reads ChatView's existing state, no extra wiring. */}
      <ActivityPanel
        toolStates={toolStates}
        messages={messages}
        busy={busy}
        liveContent={liveContent}
        liveThinking={liveThinking}
      />

      {/* Slide-over artifact preview. Rendered at the top of the DOM so it
          overlays whatever layout is beneath it. `null` when no artifact is
          open, so the panel isn't in the DOM when idle. */}
      <ArtifactPanel
        artifact={activeArtifact}
        onClose={() => setActiveArtifact(null)}
      />
    </div>
  )
}

/**
 * StatusStrip — the thin stack of banners/chips that sits between the
 * TodoPanel and the ChatInput. Holds four kinds of ephemeral UI, top-down:
 *
 *   1. "Execute plan" button — shown when plan mode has produced [PLAN READY].
 *   2. Wakeup banner         — the agent scheduled itself to resume later.
 *   3. AskUserQuestion row   — multi-choice inline prompt blocking the turn.
 *   4. Side-task chips       — drive-by issues the agent flagged with spawn_task.
 *
 * Each section renders nothing when its data is absent, so the strip
 * collapses invisibly when the conversation has no active signals.
 */
/**
 * ActiveLoopBanner — surfaces the autonomous-loop row as a persistent banner
 * with a Stop button. The countdown re-renders once a second so the user can
 * tell at a glance how soon the next tick will fire; that also makes it
 * obvious whether the loop is "alive" (ticking down) or stuck (frozen).
 *
 * Props:
 *   loop   — {id, prompt, interval_seconds, next_run_at, ...}
 *   onStop — () => void; parent handles the DELETE call and optimistic UI.
 */
function ActiveLoopBanner({ loop, onStop }) {
  // Re-render on an interval so the countdown text stays fresh without
  // coupling the banner to the parent's render cadence.
  const [, force] = useState(0)
  useEffect(() => {
    const t = setInterval(() => force((n) => n + 1), 1000)
    return () => clearInterval(t)
  }, [])
  const secsUntil = Math.max(
    0,
    Math.round((loop.next_run_at || 0) - Date.now() / 1000),
  )
  const mins = Math.floor(secsUntil / 60)
  const secs = secsUntil % 60
  const nextLabel =
    secsUntil <= 1
      ? 'firing now…'
      : `next tick in ${mins > 0 ? `${mins}m ` : ''}${secs}s`
  return (
    <div className="flex items-center gap-2 rounded-md border border-emerald-500/40 bg-emerald-500/5 px-3 py-2 text-xs">
      <Repeat className="size-4 text-emerald-400 shrink-0 animate-pulse" />
      <div className="flex-1 min-w-0">
        <div className="font-medium text-foreground">
          Autonomous loop running
        </div>
        <div className="truncate text-muted-foreground">
          <span className="font-mono">{loop.interval_seconds}s interval</span>
          <span className="mx-1">·</span>
          <span>{nextLabel}</span>
          {loop.prompt && (
            <>
              <span className="mx-1">·</span>
              <span className="italic">{loop.prompt.slice(0, 80)}{loop.prompt.length > 80 ? '…' : ''}</span>
            </>
          )}
        </div>
      </div>
      <Button
        variant="outline"
        size="sm"
        className="h-7 gap-1 border-emerald-500/40 text-xs"
        onClick={onStop}
      >
        <X className="size-3.5" />
        Stop loop
      </Button>
    </div>
  )
}

function StatusStrip({
  planReady,
  onExecutePlan,
  scheduledWakeup,
  onDismissWakeup,
  pendingQuestion,
  onAnswerQuestion,
  sideTasks,
  onOpenSideTask,
  onDismissSideTask,
  activeLoop,
  onStopLoop,
}) {
  const hasAnything =
    planReady ||
    scheduledWakeup ||
    pendingQuestion ||
    activeLoop ||
    (sideTasks && sideTasks.length > 0)
  if (!hasAnything) return null

  return (
    <div className="border-t border-border bg-background/60">
      <div className="mx-auto flex max-w-3xl flex-col gap-2 px-3 py-2 md:px-6">
        {planReady && (
          <div className="flex items-center gap-2 rounded-md border border-purple-500/40 bg-purple-500/5 px-3 py-2 text-xs">
            <Play className="size-4 text-purple-400 shrink-0" />
            <div className="flex-1">
              <div className="font-medium text-foreground">Plan ready</div>
              <div className="text-muted-foreground">
                The agent finished its plan. Switch to approve-edits mode and run it.
              </div>
            </div>
            <Button size="sm" onClick={onExecutePlan} className="gap-1">
              <Play className="size-3.5" />
              Execute plan
            </Button>
          </div>
        )}

        {scheduledWakeup && (
          <div className="flex items-center gap-2 rounded-md border border-amber-500/40 bg-amber-500/5 px-3 py-2 text-xs">
            <Clock className="size-4 text-amber-400 shrink-0" />
            <div className="flex-1 text-muted-foreground">
              {scheduledWakeup.message}
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="size-6"
              onClick={onDismissWakeup}
              aria-label="Dismiss wakeup notice"
            >
              <X className="size-3.5" />
            </Button>
          </div>
        )}

        {activeLoop && (
          <ActiveLoopBanner loop={activeLoop} onStop={onStopLoop} />
        )}

        {pendingQuestion && (
          <PendingQuestionCard
            question={pendingQuestion}
            onAnswer={onAnswerQuestion}
          />
        )}

        {sideTasks && sideTasks.length > 0 && (
          <div className="flex flex-wrap items-center gap-2">
            {sideTasks.map((t) => (
              <SideTaskChip
                key={t.id}
                task={t}
                onOpen={() => onOpenSideTask(t)}
                onDismiss={() => onDismissSideTask(t)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

/**
 * PendingQuestionCard — the inline multi-choice prompt the agent uses when
 * it calls ask_user_question. Blocks the agent's stream until the user
 * picks an option; each button posts the choice to the backend, which
 * resolves the awaited future and lets the turn continue.
 */
function PendingQuestionCard({ question, onAnswer }) {
  return (
    <div className="rounded-md border border-sky-500/40 bg-sky-500/5 px-3 py-2 text-xs">
      <div className="flex items-start gap-2">
        <HelpCircle className="size-4 text-sky-400 shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <div className="font-medium text-foreground">
            {question.question || 'The agent is asking for a decision.'}
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {question.options.map((opt, idx) => {
              const value = typeof opt === 'string' ? opt : opt?.value ?? ''
              const label = typeof opt === 'string' ? opt : opt?.label ?? value
              const description =
                typeof opt === 'object' ? opt?.description : null
              return (
                <Button
                  key={idx}
                  variant="outline"
                  size="sm"
                  className="h-auto flex-col items-start gap-0.5 py-1.5 text-left"
                  onClick={() => onAnswer(question.id, value)}
                  title={description || undefined}
                >
                  <span className="text-xs font-medium">{label}</span>
                  {description && (
                    <span className="text-[10px] font-normal text-muted-foreground">
                      {description}
                    </span>
                  )}
                </Button>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

/**
 * SideTaskChip — a single flagged drive-by issue. Clicking "Open" converts
 * it to a fresh conversation seeded with the agent's prompt; dismissing just
 * marks it closed so it won't reappear after a reload.
 */
function SideTaskChip({ task, onOpen, onDismiss }) {
  return (
    <div
      className="group flex max-w-xs items-center gap-1.5 rounded-full border border-border bg-card px-2 py-1 text-[11px]"
      title={task.tldr || task.title}
    >
      <Lightbulb className="size-3 text-amber-400 shrink-0" />
      <button
        type="button"
        onClick={onOpen}
        className="truncate font-medium hover:underline focus:outline-none"
      >
        {task.title}
      </button>
      <button
        type="button"
        onClick={onDismiss}
        className="ml-1 rounded p-0.5 text-muted-foreground opacity-60 hover:bg-accent hover:opacity-100"
        aria-label={`Dismiss ${task.title}`}
      >
        <X className="size-3" />
      </button>
    </div>
  )
}

// ----- helpers --------------------------------------------------------------

/**
 * Wrap a document's extracted text in a clearly-labelled block so the model
 * sees "this is pasted file content, not the user's actual instruction". The
 * marker strings are chosen to survive markdown rendering intact.
 */
function formatDocumentBlock(doc) {
  const name = doc.original_name || doc.name
  const meta = doc.page_count ? ` — ${doc.page_count} page${doc.page_count === 1 ? '' : 's'}` : ''
  const note = doc.truncated ? ' (truncated)' : ''
  const body = (doc.extracted_text || '').trim()
  return [
    `--- attached: ${name}${meta}${note} ---`,
    body,
    `--- end ${name} ---`,
  ].join('\n')
}

/**
 * Build a {callId: {status, result}} map from persisted message history.
 * Tool results live in separate rows (role = 'tool') with JSON-encoded content.
 */
function buildToolStatesFromHistory(messages) {
  const state = {}
  for (const m of messages) {
    if (m.role === 'tool' && m.tool_calls?.length) {
      const id = m.tool_calls[0].id
      const imagePath = m.tool_calls[0].image_path || null
      let parsed = null
      try {
        parsed = JSON.parse(m.content)
      } catch {
        /* ignore */
      }
      if (parsed) {
        state[id] = {
          status: parsed.error === 'rejected by user' ? 'rejected' : 'done',
          result: {
            ok: parsed.ok,
            output: parsed.output,
            error: parsed.error,
          },
          imagePath,
        }
      }
    }
  }
  return state
}

/**
 * Reconstruct the most recent `todo_write` result from persisted history,
 * so the TodoPanel can render the agent's plan immediately after a page
 * reload without waiting for a new `todos_updated` event.
 *
 * Walks messages oldest-to-newest and returns the last tool-row produced
 * by `todo_write`. Parsing is best-effort — malformed rows are skipped.
 */
function extractLatestTodos(messages) {
  let found = []
  for (const m of messages) {
    if (m.role !== 'tool' || !m.tool_calls?.length) continue
    if ((m.tool_calls[0].name || '') !== 'todo_write') continue
    // The tool output we persisted is the JSON-encoded result envelope;
    // the list itself is mirrored into `output`. We re-derive it from the
    // persisted assistant tool_call args for reliability.
    // Fallback: skip if we can't parse the output.
    try {
      const parsed = JSON.parse(m.content)
      const todos = parsed?.todos
      if (Array.isArray(todos) && todos.length) {
        found = todos
      }
    } catch {
      /* ignore */
    }
  }
  // If the persisted output didn't include the list (older rows may
  // predate that field), try mining it from the originating assistant row.
  if (!found.length) {
    for (const m of messages) {
      if (m.role !== 'assistant' || !m.tool_calls?.length) continue
      for (const tc of m.tool_calls) {
        if (tc.name === 'todo_write' && Array.isArray(tc.args?.todos)) {
          found = tc.args.todos
        }
      }
    }
  }
  return found
}

/**
 * PendingAssistantBubble — the streaming placeholder rendered beneath the
 * conversation while an agent turn is in flight.
 *
 * This is the progress indicator the user sees between "I pressed enter" and
 * "the final answer shows up". It intentionally renders even when no tokens
 * have arrived yet (busy=true but nothing in `content` or `thinking`), so the
 * user is never looking at a blank chat waiting for the model.
 *
 * Three visual states, in priority order:
 *   1. Streaming prose (liveContent)     → shows it as markdown + pulse dot
 *   2. Streaming thinking (liveThinking) → collapsible "Thinking…" block
 *   3. Waiting (busy only)               → just a "Working…" indicator
 * States 1 and 2 can co-occur: thinking chip above, prose below.
 */
function PendingAssistantBubble({ content, thinking, busy }) {
  const hasContent = Boolean(content)
  const hasThinking = Boolean(thinking)
  return (
    <div className="flex gap-3 py-3">
      {/* Same brand-logo avatar as the persisted assistant Message rows so
          the in-flight bubble doesn't visually swap when the stream finishes. */}
      <BrandLogo alt="Gigachat assistant" />
      <div className="min-w-0 flex-1">
        <div className="mb-1 text-xs font-medium text-muted-foreground">
          Gigachat
        </div>

        {hasThinking && (
          <ThinkingBlock
            thinking={thinking}
            running={busy && !hasContent}
          />
        )}

        {hasContent && (
          <div className="prose-chat text-sm">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
          </div>
        )}

        {busy && !hasContent && !hasThinking && (
          <div className="flex items-center gap-2 py-1 text-xs text-muted-foreground">
            <span className="pulse-dot" />
            <span>Working…</span>
          </div>
        )}

        {busy && hasContent && (
          <span className="pulse-dot mt-1 inline-block align-middle text-muted-foreground" />
        )}
      </div>
    </div>
  )
}

/**
 * ThinkingBlock — collapsible "Thought process" chip.
 *
 * Uses native <details> so it's accessible and keyboard-friendly without any
 * extra library. While `running` is true the summary pulses so the user can
 * tell the model is still streaming tokens; once the prose starts arriving
 * (or the turn ends) the pulse stops.
 *
 * The <pre> auto-scrolls to the bottom as new reasoning tokens arrive, so the
 * user always sees the most recent thought without having to reach for the
 * scrollbar. If the user deliberately scrolls up to re-read an earlier thought
 * we back off — intent-based, same pattern the outer chat scroller uses.
 */
function ThinkingBlock({ thinking, running }) {
  const preRef = useRef(null)
  // True once the user has scrolled up inside the thinking pre; reset when
  // they scroll back to the bottom. While true we stop auto-pinning.
  const pinnedUpRef = useRef(false)

  // Wire up scroll/wheel/touch listeners once. We detect intent up-front on
  // wheel/touch (the browser hasn't applied the scroll yet at that moment, so
  // a distance check would read ~0 and miss the gesture). The `scroll` handler
  // then handles the release case — if the user scrolls back to the bottom,
  // we re-enable auto-pin.
  useEffect(() => {
    const el = preRef.current
    if (!el) return
    const onScroll = () => {
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 8
      if (atBottom) pinnedUpRef.current = false
    }
    const onWheel = (e) => {
      if (e.deltaY < 0) pinnedUpRef.current = true
    }
    const onTouchMove = () => {
      pinnedUpRef.current = true
    }
    el.addEventListener('scroll', onScroll, { passive: true })
    el.addEventListener('wheel', onWheel, { passive: true })
    el.addEventListener('touchmove', onTouchMove, { passive: true })
    return () => {
      el.removeEventListener('scroll', onScroll)
      el.removeEventListener('wheel', onWheel)
      el.removeEventListener('touchmove', onTouchMove)
    }
  }, [])

  // Pin to bottom on every token unless the user has scrolled up. We also
  // run this when `running` flips so the block lands at the bottom the
  // moment it first opens.
  useEffect(() => {
    const el = preRef.current
    if (!el) return
    if (pinnedUpRef.current) return
    el.scrollTop = el.scrollHeight
  }, [thinking, running])

  return (
    <details
      className="my-2 rounded-md border border-dashed border-border/70 bg-background/40 px-3 py-2 text-xs"
      open={running}
    >
      <summary className="flex cursor-pointer select-none items-center gap-2 text-muted-foreground">
        <BrainCircuit className="size-3.5" />
        <span>{running ? 'Thinking…' : 'Thought process'}</span>
        {running && <span className="pulse-dot ml-1 text-muted-foreground" />}
      </summary>
      <pre
        ref={preRef}
        className="mt-2 max-h-60 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-muted-foreground"
      >
        {thinking}
      </pre>
    </details>
  )
}

/**
 * Rendered when no conversation is selected (initial load, or after deletion).
 */
function EmptyState({ onOpenSidebar }) {
  return (
    <div className="flex h-full flex-col items-center justify-center p-6 text-center">
      <BrandLogo size="size-20" className="mb-4 shadow-lg" />
      <h1 className="text-xl font-semibold">Gigachat</h1>
      <p className="mt-2 max-w-md text-sm text-muted-foreground">
        A Claude-Code-style assistant powered by a local Ollama model running on your PC.
        Pick a conversation from the sidebar or start a new one.
      </p>
      <Button className="mt-6 md:hidden" onClick={onOpenSidebar}>
        Open conversations
      </Button>
    </div>
  )
}

/** Brief tips shown in a new conversation before the first message. */
function WelcomeTips({ cwd }) {
  return (
    <div className="mx-auto max-w-xl py-16 text-center">
      <BrandLogo size="size-16" className="mx-auto mb-3 shadow-md" />
      <h2 className="text-lg font-semibold">Ask Gigachat anything.</h2>
      <p className="mt-1 text-sm text-muted-foreground">
        It can run shell commands, read & write files, take screenshots, and
        control the mouse & keyboard on your PC. Default working folder:
      </p>
      <p className="mt-1 font-mono text-xs text-muted-foreground break-all">{cwd}</p>
      <div className="mt-6 grid grid-cols-1 gap-2 sm:grid-cols-2">
        {SUGGESTIONS.map((s) => (
          <div
            key={s}
            className="rounded-md border border-border bg-card px-3 py-2 text-left text-xs text-muted-foreground"
          >
            {s}
          </div>
        ))}
      </div>
    </div>
  )
}

const SUGGESTIONS = [
  'What files are in this folder?',
  'Write a Python script that prints the current time.',
  'Check my disk usage.',
  'Explain what the largest file here does.',
]
