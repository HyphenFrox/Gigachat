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
