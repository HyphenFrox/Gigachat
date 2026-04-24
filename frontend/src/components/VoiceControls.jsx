import React, { useCallback, useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { Mic, MicOff, AudioLines } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { useSpeechRecognition } from '@/lib/useSpeechRecognition'

/**
 * VoiceDictateButton — push-to-talk that fills the composer with transcribed
 * speech. User still hits Enter (or Send) to actually submit the message.
 *
 * Click to start, click to stop. On each finalised phrase we append it to
 * the current composer value so the user can start speaking, pause, keep
 * thinking, and keep speaking. Interim (non-final) text is surfaced as a
 * dimmed pill above the composer so the user sees the engine's live guess.
 *
 * Props:
 *   - value: current composer text (needed so we append rather than replace)
 *   - onChange: setter for composer text
 *   - onInterim: callback with the current interim transcript (or '')
 *                so the parent can render it as a hint chip.
 *   - disabled: pass true to lock the button (e.g. when voice mode is on).
 */
export function VoiceDictateButton({
  value,
  onChange,
  onInterim,
  disabled = false,
}) {
  const onFinal = useCallback(
    (phrase) => {
      if (!phrase) return
      // Join with a leading space unless the composer is empty or already
      // ends in whitespace. Avoids "helloworld" when speaking in chunks.
      const current = value || ''
      const sep =
        current.length === 0 || /\s$/.test(current) ? '' : ' '
      onChange(current + sep + phrase)
    },
    [value, onChange],
  )

  const onError = useCallback((ev) => {
    if (ev.error === 'not-allowed' || ev.error === 'service-not-allowed') {
      toast.error('Microphone blocked', {
        description: 'Allow microphone access in the browser address bar to use voice input.',
      })
    } else if (ev.error === 'no-speech') {
      // Quiet failure — the user stopped speaking before any phrase finalised.
      return
    } else if (ev.error === 'not-supported') {
      toast.error('Voice input unsupported', {
        description: 'Your browser does not expose the Web Speech API. Try Chrome or Edge.',
      })
    } else {
      toast.error('Voice input error', { description: String(ev.error || ev) })
    }
  }, [])

  const { supported, listening, interim, start, stop } = useSpeechRecognition({
    continuous: false,
    interimResults: true,
    onInterim,
    onFinalPhrase: onFinal,
    onError,
  })

  // Bubble the interim transcript up so the parent can render a chip.
  useEffect(() => {
    onInterim?.(interim)
  }, [interim, onInterim])

  if (!supported) {
    // Hide entirely rather than render a disabled button — saves horizontal
    // space in the composer, and we don't want to advertise a feature that
    // won't work.
    return null
  }

  return (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      onClick={() => (listening ? stop() : start())}
      disabled={disabled}
      title={listening ? 'Stop dictation' : 'Dictate (push-to-talk)'}
      aria-pressed={listening}
      className={cn(
        'relative',
        listening && 'text-rose-400 animate-pulse',
      )}
    >
      {listening ? <MicOff /> : <Mic />}
    </Button>
  )
}

/**
 * VoiceModeToggle + VoiceModeDriver — continuous voice mode.
 *
 * Mode: user talks; the composer fills with transcribed speech; after ~1.2s
 * of silence following a final phrase we auto-submit. No send button needed.
 *
 * The toggle button flips the mode on/off. When on, the driver component
 * keeps recognition running and owns a silence timer that fires onAutoSubmit
 * when the user has stopped speaking.
 *
 * We keep these separate so the toggle can live anywhere while the driver
 * (which owns side effects) is only mounted when active — this avoids
 * burning the mic and CPU on recognizer events when voice mode is off.
 */
export function VoiceModeToggle({ active, onToggle, disabled }) {
  // Early feature detection so we don't render a button that does nothing.
  const SpeechRecognition =
    typeof window !== 'undefined'
      ? window.SpeechRecognition || window.webkitSpeechRecognition
      : null
  if (!SpeechRecognition) return null

  return (
    <Button
      type="button"
      variant={active ? 'default' : 'ghost'}
      size="icon"
      onClick={() => onToggle(!active)}
      disabled={disabled}
      title={active ? 'Stop voice mode' : 'Start voice mode (auto-sends on silence)'}
      aria-pressed={active}
      className={cn(active && 'bg-primary text-primary-foreground')}
    >
      <AudioLines className={cn(active && 'animate-pulse')} />
    </Button>
  )
}

/**
 * VoiceModeDriver — mounted only while `active` is true. Wires up continuous
 * recognition with silence-based auto-submit.
 *
 * Props:
 *   - onInterim: (string) — live partial transcript
 *   - onFinalChunk: (string) — called each time a final phrase lands; the
 *                   parent appends it to the composer
 *   - onAutoSubmit: () — called after AUTO_SUBMIT_SILENCE_MS of silence
 *                   following a finalised phrase. Parent should trigger
 *                   onSend exactly as a keyboard Enter would.
 *   - onStop: () — notify the parent the engine stopped (e.g. on error)
 *                   so it can flip its `active` state back off.
 */
export function VoiceModeDriver({
  onInterim,
  onFinalChunk,
  onAutoSubmit,
  onStop,
}) {
  // Silence window: long enough that the user has time to breathe, short
  // enough to feel conversational. Measured from the last *finalised* phrase.
  const AUTO_SUBMIT_SILENCE_MS = 1400

  const timerRef = useRef(null)
  const hasSpokenRef = useRef(false)
  const submittedRef = useRef(false)

  // Route callbacks through refs so the silence timer, which may fire up to
  // AUTO_SUBMIT_SILENCE_MS after the phrase arrived, always invokes the
  // freshest handler (which in turn reads the freshest composer state).
  const onInterimRef = useRef(onInterim)
  const onFinalChunkRef = useRef(onFinalChunk)
  const onAutoSubmitRef = useRef(onAutoSubmit)
  useEffect(() => {
    onInterimRef.current = onInterim
    onFinalChunkRef.current = onFinalChunk
    onAutoSubmitRef.current = onAutoSubmit
  }, [onInterim, onFinalChunk, onAutoSubmit])

  const clearTimer = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
  }

  const handleInterim = useCallback((text) => {
    onInterimRef.current?.(text)
    // Any new interim text means the user is still talking — cancel pending
    // auto-submit until they pause again.
    if (text) clearTimer()
  }, [])

  const handleFinal = useCallback((phrase) => {
    if (!phrase) return
    hasSpokenRef.current = true
    submittedRef.current = false
    onFinalChunkRef.current?.(phrase)
    // Reset silence timer so the next pause (not the next phrase) triggers
    // the send.
    clearTimer()
    timerRef.current = setTimeout(() => {
      if (!submittedRef.current) {
        submittedRef.current = true
        onAutoSubmitRef.current?.()
      }
    }, AUTO_SUBMIT_SILENCE_MS)
  }, [])

  const handleError = useCallback(
    (ev) => {
      // Permission errors are terminal — stop voice mode and toast.
      if (ev.error === 'not-allowed' || ev.error === 'service-not-allowed') {
        toast.error('Microphone blocked', {
          description: 'Allow microphone access to use voice mode.',
        })
        onStop?.()
        return
      }
      // `no-speech` and `audio-capture` are transient; the recognizer's onend
      // will fire right after and we'll auto-restart below.
    },
    [onStop],
  )

  const { supported, listening, start, stop } = useSpeechRecognition({
    continuous: true,
    interimResults: true,
    onInterim: handleInterim,
    onFinalPhrase: handleFinal,
    onError: handleError,
  })

  // Start on mount; abort on unmount. Auto-restart on end because some
  // browsers (looking at you, Chrome) terminate continuous recognition after
  // ~60s of audio — we need to kick it back on so voice mode feels permanent.
  useEffect(() => {
    if (!supported) {
      toast.error('Voice mode unsupported', {
        description: 'Your browser does not expose the Web Speech API.',
      })
      onStop?.()
      return
    }
    start()
    return () => {
      clearTimer()
      stop()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Relaunch recognition whenever it ends (browser tab still in the foreground)
  // so one continuous "mode" survives the engine's internal reset windows.
  useEffect(() => {
    if (!listening) {
      // Small delay — starting immediately inside an onend handler throws
      // InvalidStateError on Chromium. 120ms is enough for cleanup.
      const t = setTimeout(() => start(), 120)
      return () => clearTimeout(t)
    }
  }, [listening, start])

  return null
}
