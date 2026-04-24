import { Toaster as SonnerToaster } from 'sonner'

/**
 * Toaster — wrapper around Sonner with shadcn theming applied.
 * Mount once at the root (see App.jsx). All errors/info across the app
 * go through `toast.error(...)` / `toast.success(...)` imported from 'sonner'.
 *
 * CLAUDE.md explicitly requires Sonner for all user-visible errors.
 */
const Toaster = (props) => (
  <SonnerToaster
    theme="dark"
    position="bottom-right"
    richColors
    closeButton
    toastOptions={{
      classNames: {
        toast:
          'group toast group-[.toaster]:bg-background group-[.toaster]:text-foreground group-[.toaster]:border-border group-[.toaster]:shadow-lg',
        description: 'group-[.toast]:text-muted-foreground',
        actionButton:
          'group-[.toast]:bg-primary group-[.toast]:text-primary-foreground',
        cancelButton:
          'group-[.toast]:bg-muted group-[.toast]:text-muted-foreground',
      },
    }}
    {...props}
  />
)

export { Toaster }
