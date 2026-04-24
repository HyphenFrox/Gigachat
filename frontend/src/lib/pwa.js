/**
 * PWA + Web Push glue code.
 *
 * Two concerns in one module:
 *
 *   1. `registerServiceWorker()` — kicks off the /sw.js registration on page
 *      load and listens for "push-click" messages the worker posts back when
 *      the user taps a notification. The message carries the conversation id
 *      so the app can jump straight to it.
 *
 *   2. `push.*` — a tiny wrapper around the browser's PushManager that talks
 *      to our backend's /api/push/* endpoints. Handles the fiddly
 *      base64url → Uint8Array conversion that `applicationServerKey` needs.
 *
 * All functions are no-ops (with clear return values) when the browser
 * doesn't support the APIs — the caller can still render a sensible UI.
 */

let pushClickHandler = null

/**
 * Register a handler to be called when the service worker posts a
 * `push-click` message (user tapped a notification). The payload is
 * `{ type: 'push-click', conversation_id: string | null }`. Only one
 * handler is kept — calling again replaces the previous one.
 */
export function onPushClick(handler) {
  pushClickHandler = handler
}

/**
 * Register /sw.js at the root scope. Called once from main.jsx. Idempotent:
 * if there's already an active registration, this is a cheap noop.
 */
export async function registerServiceWorker() {
  if (typeof navigator === 'undefined' || !('serviceWorker' in navigator)) return null
  try {
    const reg = await navigator.serviceWorker.register('/sw.js', { scope: '/' })
    // Relay the worker's "push-click" postMessage to whichever handler the
    // app installed via onPushClick(). Keeping this in the library (not in
    // App.jsx) means the handler stays wired even across hot reloads.
    navigator.serviceWorker.addEventListener('message', (event) => {
      const data = event.data
      if (!data || data.type !== 'push-click') return
      if (typeof pushClickHandler === 'function') {
        try {
          pushClickHandler(data)
        } catch (e) {
          console.error('push-click handler threw', e)
        }
      }
    })
    return reg
  } catch (e) {
    console.warn('Service worker registration failed:', e)
    return null
  }
}

// -----------------------------------------------------------------------------
// Push subscription helpers
// -----------------------------------------------------------------------------
function urlBase64ToUint8Array(base64String) {
  // VAPID keys arrive base64url-encoded (no padding). PushManager.subscribe
  // wants a raw Uint8Array, so this unpacks the RFC 4648 §5 form.
  const padding = '='.repeat((4 - (base64String.length % 4)) % 4)
  const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/')
  const bin = atob(base64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

async function getRegistration() {
  if (typeof navigator === 'undefined' || !('serviceWorker' in navigator)) return null
  // `ready` resolves to the active registration; if no SW is installed yet it
  // hangs forever, so we race against a short timeout and fall back to null.
  return Promise.race([
    navigator.serviceWorker.ready,
    new Promise((resolve) => setTimeout(() => resolve(null), 2500)),
  ])
}

export const push = {
  /**
   * Capability report consumed by the settings UI so it can either render
   * the enable button or explain why the feature is unavailable.
   */
  supported() {
    return (
      typeof window !== 'undefined' &&
      'serviceWorker' in navigator &&
      'PushManager' in window &&
      'Notification' in window
    )
  },

  /**
   * Current browser permission state — one of 'default' | 'granted' | 'denied'.
   * Returns 'unsupported' when the Notifications API is missing entirely.
   */
  permission() {
    if (typeof Notification === 'undefined') return 'unsupported'
    return Notification.permission
  },

  /** Whether we already have an active PushSubscription for this browser. */
  async isSubscribed() {
    const reg = await getRegistration()
    if (!reg) return false
    const sub = await reg.pushManager.getSubscription()
    return !!sub
  },

  /**
   * Subscribe the current browser and POST the subscription to the server.
   * Returns the PushSubscription on success, or throws with a user-visible
   * message on failure (the caller surfaces it via Sonner).
   */
  async subscribe() {
    if (!this.supported()) throw new Error('Push notifications are not supported in this browser.')

    const perm = await Notification.requestPermission()
    if (perm !== 'granted') throw new Error('Notification permission denied.')

    const reg = await getRegistration()
    if (!reg) throw new Error('Service worker is not active yet — try again in a moment.')

    // Reuse the existing subscription if the user already enabled this once —
    // avoids asking the push service for a new endpoint on every toggle.
    let sub = await reg.pushManager.getSubscription()
    if (!sub) {
      const res = await fetch('/api/push/vapid-key')
      if (!res.ok) throw new Error('Could not fetch VAPID key.')
      const { public_key: publicKey } = await res.json()
      sub = await reg.pushManager.subscribe({
        userVisibleOnly: true, // all major browsers require this to be true
        applicationServerKey: urlBase64ToUint8Array(publicKey),
      })
    }

    const json = sub.toJSON()
    const body = {
      endpoint: json.endpoint,
      keys: {
        p256dh: json.keys?.p256dh || '',
        auth: json.keys?.auth || '',
      },
      user_agent: navigator.userAgent || null,
    }
    const res = await fetch('/api/push/subscribe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) throw new Error(`Server rejected subscription: ${res.status}`)
    return sub
  },

  /**
   * Unsubscribe this browser locally AND tell the server to delete its row.
   * Both legs are best-effort so even a partial failure leaves the user in a
   * sensible state (e.g. server row gone but browser permission still granted).
   */
  async unsubscribe() {
    const reg = await getRegistration()
    if (!reg) return
    const sub = await reg.pushManager.getSubscription()
    if (!sub) return
    const endpoint = sub.endpoint
    try {
      await sub.unsubscribe()
    } catch {
      /* ignore — we still want to clear the server side */
    }
    await fetch('/api/push/unsubscribe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ endpoint }),
    }).catch(() => undefined)
  },

  /** Ask the server to fan a test notification out to all registered browsers. */
  async sendTest() {
    const res = await fetch('/api/push/test', { method: 'POST' })
    if (!res.ok) throw new Error(`Test push failed: ${res.status}`)
    return res.json()
  },
}
