import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

/**
 * shadcn's canonical `cn` helper.
 * Joins conditional class lists and resolves Tailwind conflicts
 * (so `cn("p-2", condition && "p-4")` yields just `p-4`).
 *
 * @param {...any} inputs - class names, arrays, or falsy values
 * @returns {string}
 */
export function cn(...inputs) {
  return twMerge(clsx(inputs))
}

// Resolved IANA time zone for the user's browser (e.g. "America/Los_Angeles",
// "Asia/Kolkata"). All `formatX` helpers below pin to this zone explicitly
// so the rendered output unambiguously reflects local wall-clock time —
// not UTC, not the server's TZ. Cached at module load because
// `resolvedOptions()` is non-trivial and the value never changes within
// a tab session.
const USER_TIMEZONE = (() => {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || undefined
  } catch {
    // Older browsers may not expose `timeZone`. Falling back to
    // `undefined` makes Intl pick the system default — same effective
    // result, just without the assertion in the source.
    return undefined
  }
})()

/**
 * Get the user's IANA time zone (e.g. "America/Los_Angeles") if the
 * platform exposes it. Useful when callers want to add the zone to
 * their own log lines or status text.
 *
 * @returns {string | undefined}
 */
export function getUserTimezone() {
  return USER_TIMEZONE
}

/**
 * Format a Unix-epoch-seconds timestamp in the user's LOCAL time zone,
 * the way a chat app does:
 *   - same day  → time only (e.g. "14:32")
 *   - within a week → weekday + time (e.g. "Tue 14:32")
 *   - same year → month-day + time (e.g. "Mar 4 14:32")
 *   - older → full date + time (e.g. "2024-11-09 14:32")
 *
 * Returns "" for falsy / invalid input so callers can safely render
 * the result as text without an extra null-check.
 *
 * @param {number} epochSec - seconds since 1970-01-01 UTC (e.g. message.created_at)
 * @returns {string}
 */
export function formatMessageTime(epochSec) {
  if (!epochSec || typeof epochSec !== 'number' || !isFinite(epochSec)) {
    return ''
  }
  const d = new Date(epochSec * 1000)
  if (isNaN(d.getTime())) return ''
  const now = new Date()
  // Day comparison via getFullYear/getMonth/getDate uses LOCAL time,
  // so "same day" / "days ago" reflect the user's wall clock — not UTC.
  const sameDay =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate()
  const time = d.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    timeZone: USER_TIMEZONE,
  })
  if (sameDay) return time

  // Within the last 7 days → weekday + time. Compares calendar days,
  // not raw seconds, so a message at 23:50 yesterday counts as one
  // day ago even when the clock is past midnight.
  const dayMs = 24 * 60 * 60 * 1000
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const startOfThat = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
  const daysAgo = Math.round((startOfToday - startOfThat) / dayMs)
  if (daysAgo > 0 && daysAgo < 7) {
    const weekday = d.toLocaleDateString(undefined, {
      weekday: 'short', timeZone: USER_TIMEZONE,
    })
    return `${weekday} ${time}`
  }

  if (d.getFullYear() === now.getFullYear()) {
    const md = d.toLocaleDateString(undefined, {
      month: 'short', day: 'numeric', timeZone: USER_TIMEZONE,
    })
    return `${md} ${time}`
  }
  // Older than this year → full ISO-ish date + time.
  const ymd = d.toLocaleDateString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
    timeZone: USER_TIMEZONE,
  })
  return `${ymd} ${time}`
}

/**
 * Full timestamp for a `title` tooltip — locale-formatted, includes
 * seconds + weekday + the time zone abbreviation (e.g. "PST" / "GMT+5:30")
 * so the user can confirm the rendering is in their LOCAL time zone.
 * Used for hover tooltips on the chat-bubble timestamp.
 *
 * @param {number} epochSec
 * @returns {string}
 */
export function formatFullTimestamp(epochSec) {
  if (!epochSec || typeof epochSec !== 'number' || !isFinite(epochSec)) {
    return ''
  }
  const d = new Date(epochSec * 1000)
  if (isNaN(d.getTime())) return ''
  return d.toLocaleString(undefined, {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    // `timeZoneName: 'short'` appends the abbreviation after the time
    // (e.g. "Tuesday, March 4, 2025, 14:32:01 PST"). Visible confirmation
    // that the timestamp is in the viewer's local zone.
    timeZoneName: 'short',
    timeZone: USER_TIMEZONE,
  })
}

/**
 * Return true when two Unix-epoch-second timestamps fall on the same
 * calendar day in the local time zone. Used by the chat transcript
 * to decide where to insert a day-separator pill — only above the
 * first message of each new day, exactly like a messaging app.
 *
 * @param {number} a - first epoch (seconds)
 * @param {number} b - second epoch (seconds)
 * @returns {boolean}
 */
export function sameCalendarDay(a, b) {
  if (!a || !b) return false
  const da = new Date(a * 1000)
  const db = new Date(b * 1000)
  if (isNaN(da.getTime()) || isNaN(db.getTime())) return false
  return (
    da.getFullYear() === db.getFullYear() &&
    da.getMonth() === db.getMonth() &&
    da.getDate() === db.getDate()
  )
}

/**
 * Day-separator label for grouping messages by calendar day in the
 * chat transcript, computed in the user's LOCAL time zone:
 *   today / yesterday / Mon-Sun (this week) / Mar 4 / Mar 4 2023.
 *
 * @param {number} epochSec
 * @returns {string}
 */
export function formatDayLabel(epochSec) {
  if (!epochSec || typeof epochSec !== 'number' || !isFinite(epochSec)) {
    return ''
  }
  const d = new Date(epochSec * 1000)
  if (isNaN(d.getTime())) return ''
  const now = new Date()
  const dayMs = 24 * 60 * 60 * 1000
  // getFullYear/getMonth/getDate read LOCAL time, so a message at
  // 23:50 yesterday still counts as one local day ago even when the
  // clock just crossed midnight.
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const startOfThat = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
  const daysAgo = Math.round((startOfToday - startOfThat) / dayMs)
  if (daysAgo === 0) return 'Today'
  if (daysAgo === 1) return 'Yesterday'
  if (daysAgo > 1 && daysAgo < 7) {
    return d.toLocaleDateString(undefined, {
      weekday: 'long', timeZone: USER_TIMEZONE,
    })
  }
  if (d.getFullYear() === now.getFullYear()) {
    return d.toLocaleDateString(undefined, {
      month: 'short', day: 'numeric', timeZone: USER_TIMEZONE,
    })
  }
  return d.toLocaleDateString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
    timeZone: USER_TIMEZONE,
  })
}
