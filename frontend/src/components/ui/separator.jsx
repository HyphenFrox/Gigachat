import * as React from 'react'
import { cn } from '@/lib/utils'

/**
 * Separator — thin divider line, horizontal or vertical.
 * Kept minimal; no Radix dep needed because we only use it as a visual rule.
 */
const Separator = React.forwardRef(
  ({ className, orientation = 'horizontal', ...props }, ref) => (
    <div
      ref={ref}
      role="separator"
      className={cn(
        'shrink-0 bg-border',
        orientation === 'horizontal' ? 'h-px w-full' : 'h-full w-px',
        className,
      )}
      {...props}
    />
  ),
)
Separator.displayName = 'Separator'

export { Separator }
