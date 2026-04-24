import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * useSpeechRecognition — thin React wrapper around the Web Speech API.
 *
 * We lean on the browser's native SpeechRecognition (`webkitSpeechRecognition`
 * on Chromium/Edge/Safari) because it's free, offline-capable on modern
 * browsers, and avoids shipping audio to any server. Firefox currently has
 * no implementation — callers should check `supported` and hide the UI
 * gracefully when it's false.
 *
 * Two listening modes are exposed via options:
 *   - continuous=false (default) — single-shot: stops after the first
 *     utterance. Used for the push-to-talk button in the composer.
 *   - continuous=true — keeps the mic open across pauses and fires
 *     `onFinalPhrase` every time a chunk becomes final. Used by "voice mode"
 *     which auto-sends on silence.
 *
 * Callbacks:
 *   - onInterim(text)      : called as the engine refines the partial guess
 *   - onFinalPhrase(text)  : called once per finalised utterance (continuous)
 *                            OR once on stop (non-continuous)
 *   - onError(event)       : bubble recognition errors so the UI can toast
 *
 * All callbacks are read from refs, so consumers can pass fresh closures on
 * every render without causing the recognizer to be torn down and rebuilt.
 */
export function useSpeechRecognition({
  continuous = false,
  interimResults = true,
  lang,
  onInterim,
  onFinalPhrase,
  onError,
} = {}) {
  // Browser support check — stable across the hook's lifetime.
  const SpeechRecognition =
    typeof window !== 'undefined'
      ? window.SpeechRecognition || window.webkitSpeechRecognition
      : null
  const supported = !!SpeechRecognition

  const [listening, setListening] = useState(false)
  const [interim, setInterim] = useState('')
  const [error, setError] = useState(null)

  const recognitionRef = useRef(null)
  const interimCb = useRef(onInterim)
  const finalCb = useRef(onFinalPhrase)
  const errorCb = useRef(onError)

  // Keep callback refs fresh without triggering effect re-runs.
  useEffect(() => {
    interimCb.current = onInterim
    finalCb.current = onFinalPhrase
    errorCb.current = onError
  }, [onInterim, onFinalPhrase, onError])

  // Build a SpeechRecognition instance on demand. We lazy-construct inside
  // start() rather than in useEffect so options changes take effect at the
  // moment listening begins, not when they're passed.
  const start = useCallback(() => {
    if (!supported) {
      setError('not-supported')
      errorCb.current?.({ error: 'not-supported' })
      return false
    }
    if (recognitionRef.current) return true

    const rec = new SpeechRecognition()
    rec.continuous = !!continuous
    rec.interimResults = !!interimResults
    if (lang) rec.lang = lang

    rec.onstart = () => {
      setListening(true)
      setError(null)
    }
    rec.onend = () => {
      setListening(false)
      setInterim('')
      recognitionRef.current = null
    }
    rec.onerror = (ev) => {
      // `aborted` fires whenever we call stop() ourselves; it's not an error.
      if (ev.error === 'aborted') return
      setError(ev.error || 'unknown')
      errorCb.current?.(ev)
    }
    rec.onresult = (ev) => {
      // Event carries a list of SpeechRecognitionResults — some final, some
      // interim depending on the engine's confidence. We accumulate across
      // the batch so a single result may contain multiple finalised phrases
      // in quick-speech scenarios.
      let interimText = ''
      let finalText = ''
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const chunk = ev.results[i]
        const transcript = chunk[0]?.transcript || ''
        if (chunk.isFinal) {
          finalText += transcript
        } else {
          interimText += transcript
        }
      }
      if (interimText) {
        setInterim(interimText)
        interimCb.current?.(interimText)
      }
      if (finalText) {
        setInterim('')
        finalCb.current?.(finalText.trim())
      }
    }

    recognitionRef.current = rec
    try {
      rec.start()
      return true
    } catch (e) {
      // Chrome throws InvalidStateError if start() is called twice in quick
      // succession. Surface gracefully — the caller can retry on the next tick.
      setError(e?.message || 'start-failed')
      errorCb.current?.({ error: e?.message || 'start-failed' })
      recognitionRef.current = null
      return false
    }
  }, [supported, SpeechRecognition, continuous, interimResults, lang])

  const stop = useCallback(() => {
    const rec = recognitionRef.current
    if (!rec) return
    try {
      rec.stop()
    } catch {
      // Already stopped — ignore.
    }
  }, [])

  // Make absolutely sure the mic is released if the component unmounts mid-listen.
  useEffect(() => {
    return () => {
      const rec = recognitionRef.current
      if (rec) {
        try {
          rec.abort()
        } catch {
          /* ignore */
        }
        recognitionRef.current = null
      }
    }
  }, [])

  return { supported, listening, interim, error, start, stop }
}
