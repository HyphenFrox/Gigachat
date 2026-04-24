import React from 'react'
import logoUrl from '@/assets/gigachat-logo.jpg'
import { cn } from '@/lib/utils'

/**
 * BrandLogo — single source of truth for the Gigachat avatar / brand mark.
 *
 * Same image is reused as:
 *   - the small badge in the Sidebar header next to the app name
 *   - the assistant message avatar (replaces the generic Sparkles glyph)
 *   - the large hero glyph on the empty / no-conversation-selected screens
 *
 * Centralising it means a future logo swap touches one file. The image is
 * imported as a JS module so Vite hashes the asset URL at build time and
 * the `<img>` cache-busts cleanly across deploys.
 *
 * Props:
 *   - size: tailwind size class, e.g. "size-7" (default) or "size-16".
 *           Pass any class string — we just append it. Use this to scale
 *           between the tiny avatar and the giant hero variant.
 *   - className: extra classes appended after `size`. Common use: ring,
 *                shadow, or border styling specific to the call site.
 *   - alt: accessible label override. Defaults to "Gigachat" because the
 *          icon doubles as the brand mark; override only when context
 *          warrants a different a11y label.
 */
export default function BrandLogo({ size = 'size-7', className, alt = 'Gigachat' }) {
  return (
    <img
      src={logoUrl}
      alt={alt}
      // `object-cover` keeps the image filling the circle without letterboxing
      // when the source aspect ratio doesn't match a square.
      className={cn(
        'shrink-0 rounded-full object-cover',
        size,
        className,
      )}
      // Hint the browser to decode off the main thread — keeps the chat
      // scroll feeling buttery on first paint when many messages mount.
      decoding="async"
      loading="eager"
    />
  )
}
