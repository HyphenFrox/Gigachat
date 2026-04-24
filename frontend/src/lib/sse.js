/**
 * Fetch-based Server-Sent Events reader.
 *
 * We use `fetch` instead of the built-in EventSource because:
 *   1) EventSource can only send GET requests; we need POST with a JSON body.
 *   2) EventSource can't carry custom headers or be aborted mid-stream.
 *
 * The server emits plain "data: {json}\n\n" frames (see backend/app.py::_sse).
 * This reader buffers the byte stream, splits on "\n\n", strips the "data: "
 * prefix from each frame, parses the JSON, and hands each event to `onEvent`.
 */

/**
 * Start an SSE POST request and dispatch events.
 *
 * @param {object} opts
 * @param {string} opts.url - URL to POST to.
 * @param {object} [opts.body] - JSON body (optional).
 * @param {(event:any) => void} opts.onEvent - called once per parsed event.
 * @param {AbortSignal} [opts.signal] - abort the stream early.
 * @returns {Promise<void>} resolves when the stream ends cleanly.
 */
export async function postEventStream({ url, body, onEvent, signal }) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: body ? JSON.stringify(body) : undefined,
    signal,
  })

  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => '')
    throw new Error(`SSE request failed: ${res.status} ${res.statusText} ${text}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // An SSE frame is terminated by a blank line (\n\n). Process all complete
    // frames currently in the buffer; the trailing partial frame stays.
    let boundary
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)

      // A frame can have multiple "field: value" lines; we only use `data:`.
      const dataLines = frame
        .split('\n')
        .filter((l) => l.startsWith('data: '))
        .map((l) => l.slice(6))
      if (!dataLines.length) continue

      const json = dataLines.join('\n')
      try {
        onEvent(JSON.parse(json))
      } catch {
        // Best-effort — silently drop malformed frames rather than tearing
        // the whole stream down for one bad event.
      }
    }
  }
}
