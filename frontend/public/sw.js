/**
 * Gigachat service worker.
 *
 * Responsibilities:
 *   1. Offline shell — cache the app shell (index.html + icons) with a
 *      network-first strategy so the app still opens if the user launches
 *      the PWA while offline.
 *   2. Web Push handler — when the server sends an encrypted payload via
 *      pushManager.subscribe(), this worker turns it into an OS notification.
 *   3. Notification click — focus the already-open app (and, if a
 *      conversation id was attached to the payload, post a message so the
 *      page can jump to it). Falls back to opening a new window.
 *
 * All logic lives in this single file so the page only has to register a
 * single URL (/sw.js). The file is served by FastAPI at the root scope
 * (Service-Worker-Allowed: /) so the worker can control the whole app.
 */

// Bump the version when app-shell resources change. The old cache is purged
// on activate so users don't get stale shell HTML after a redeploy.
const CACHE_VERSION = 'gigachat-shell-v2'

// Files we want available offline. We deliberately keep this list TINY — the
// heavy JS bundles come from /assets/* with content-hashed filenames, so
// there's no safe way to pre-cache them without chasing hashes. A
// network-first strategy for "/" covers the common "kicked offline briefly"
// case without needing a bundle map.
const SHELL_URLS = ['/', '/manifest.webmanifest', '/icon-192.png', '/icon-512.png']

self.addEventListener('install', (event) => {
  // Pre-cache the shell so first-offline-launch works. Failing here is
  // non-fatal — the feature simply degrades to "app only works online".
  event.waitUntil(
    caches.open(CACHE_VERSION).then((cache) =>
      cache.addAll(SHELL_URLS).catch(() => undefined),
    ),
  )
  // Take over immediately instead of waiting for all tabs to close —
  // speeds up the install→active transition during development.
  self.skipWaiting()
})

self.addEventListener('activate', (event) => {
  // Purge old shell caches from previous versions so we don't accumulate
  // dead storage on every deploy.
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(
        names
          .filter((name) => name.startsWith('gigachat-shell-') && name !== CACHE_VERSION)
          .map((name) => caches.delete(name)),
      ),
    ),
  )
  self.clients.claim()
})

/**
 * fetch — lightweight routing.
 *
 * Navigation requests (top-level HTML loads) go network-first with a cache
 * fallback so offline users still get the shell. Everything else we
 * explicitly handle as a pass-through so a previous buggy worker version
 * can't leave a `respondWith(Response.error())` cached in the registration
 * — the observed symptom is `net::ERR_CONNECTION_RESET` on hashed
 * /assets/* URLs even though the server is serving them fine.
 *
 * Same-origin API calls (especially SSE under /api/conversations/*/messages)
 * MUST be passed through untouched — wrapping them in the worker's fetch
 * breaks streaming.
 */
self.addEventListener('fetch', (event) => {
  const { request } = event

  // GET-only worker surface. Anything else (POST/PUT/DELETE) we never touch.
  if (request.method !== 'GET') return

  // Non-navigation GETs (CSS, JS, images, fonts, /api/*) — don't intercept.
  // Returning without calling respondWith lets the browser handle it natively.
  if (request.mode !== 'navigate') return

  event.respondWith(
    fetch(request)
      .then((resp) => {
        // Cache a fresh copy of "/" so the next offline launch has it.
        if (resp && resp.ok && request.url.endsWith('/')) {
          const copy = resp.clone()
          caches.open(CACHE_VERSION).then((cache) => cache.put('/', copy))
        }
        return resp
      })
      .catch(() => caches.match('/').then((cached) => cached || Response.error())),
  )
})

/**
 * push — display the incoming payload as a notification.
 *
 * The server always sends JSON via pywebpush. We parse defensively so a
 * malformed payload still surfaces *something* rather than silently dropping.
 * The payload shape matches backend/push.py: {title, body, tag?, conversation_id?}.
 */
self.addEventListener('push', (event) => {
  let payload = {}
  if (event.data) {
    try {
      payload = event.data.json()
    } catch {
      try {
        payload = { title: 'Gigachat', body: event.data.text() }
      } catch {
        payload = { title: 'Gigachat', body: 'New activity' }
      }
    }
  }
  const title = payload.title || 'Gigachat'
  const options = {
    body: payload.body || '',
    // Collapsing tag — successive notifications with the same tag replace
    // the previous one instead of stacking (avoids an inbox of "scheduled
    // run completed" pings after a busy day).
    tag: payload.tag || 'gigachat',
    icon: '/icon-192.png',
    badge: '/icon-192.png',
    // Stash the conversation id so the click handler below can focus the
    // right tab. We intentionally only round-trip a handful of fields.
    data: {
      conversation_id: payload.conversation_id || null,
      kind: payload.kind || 'notification',
    },
    renotify: false,
  }
  event.waitUntil(self.registration.showNotification(title, options))
})

/**
 * notificationclick — focus an existing tab (and tell it where to jump) or
 * open a new one if no window is running.
 */
self.addEventListener('notificationclick', (event) => {
  event.notification.close()
  const convId = event.notification.data?.conversation_id || null

  event.waitUntil(
    (async () => {
      const clients = await self.clients.matchAll({
        type: 'window',
        includeUncontrolled: true,
      })
      // Prefer a tab that's already scoped to our app — focus it and tell
      // the page which conversation to open via postMessage.
      for (const client of clients) {
        if (client.url.includes(self.registration.scope)) {
          await client.focus()
          if (convId) {
            client.postMessage({
              type: 'push-click',
              conversation_id: convId,
            })
          }
          return
        }
      }
      // No window open — cold start into the app. We append a ?conv= hint
      // that main.jsx can read and redirect to the right conversation.
      const target = convId
        ? `${self.registration.scope}?conv=${encodeURIComponent(convId)}`
        : self.registration.scope
      await self.clients.openWindow(target)
    })(),
  )
})
