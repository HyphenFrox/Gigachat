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

/**
 * Format a Unix-epoch-seconds timestamp the way a chat app does:
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
  const sameDay =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate()
  const time = d.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
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
    const weekday = d.toLocaleDateString(undefined, { weekday: 'short' })
    return `${weekday} ${time}`
  }

  if (d.getFullYear() === now.getFullYear()) {
    const md = d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
    return `${md} ${time}`
  }
  // Older than this year → full ISO-ish date + time.
  const ymd = d.toLocaleDateString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
  })
  return `${ymd} ${time}`
}

/**
 * Full timestamp for a `title` tooltip — locale-formatted, includes
 * seconds and weekday. Used for hover tooltips on the chat-bubble
 * timestamp so users can see the exact moment without cluttering
 * the inline display.
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
 * chat transcript. Same logic as messaging apps:
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
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const startOfThat = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
  const daysAgo = Math.round((startOfToday - startOfThat) / dayMs)
  if (daysAgo === 0) return 'Today'
  if (daysAgo === 1) return 'Yesterday'
  if (daysAgo > 1 && daysAgo < 7) {
    return d.toLocaleDateString(undefined, { weekday: 'long' })
  }
  if (d.getFullYear() === now.getFullYear()) {
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  }
  return d.toLocaleDateString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
  })
}
