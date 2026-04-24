import React, { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Bell, BellOff, Send, ShieldAlert } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { api } from '@/lib/api'
import { push } from '@/lib/pwa'

/**
 * NotificationsPanel — Settings → Notifications drawer.
 *
 * Lets the user:
 *   - Enable Web Push on the current browser (subscribes via the service
 *     worker, asks the OS for permission, POSTs the subscription to our
 *     backend so future scheduled-task completions can reach them).
 *   - Disable Web Push on this browser (removes the subscription locally
 *     and tells the backend to drop the row so it's not retried).
 *   - Send a test notification to confirm the round-trip works.
 *
 * When the browser doesn't support Web Push (e.g. most iOS Safari versions
 * older than 16.4 in non-installed mode, private-mode Firefox), we render
 * a read-only explanation rather than an input the user can't operate.
 *
 * Props:
 *   - open: boolean
 *   - onClose: () => void
 */
export default function NotificationsPanel({ open, onClose }) {
  // Tri-state: null = not yet checked, true = subscribed here, false = not.
  const [subscribedHere, setSubscribedHere] = useState(null)
  const [permission, setPermission] = useState('default')
  const [deviceCount, setDeviceCount] = useState(0)
  const [busy, setBusy] = useState(false)
  const supported = push.supported()

  const refresh = useCallback(async () => {
    if (!supported) {
      setSubscribedHere(false)
      setPermission(push.permission())
      return
    }
    try {
      const [here, perm, status] = await Promise.all([
        push.isSubscribed(),
        Promise.resolve(push.permission()),
        api.pushStatus().catch(() => ({ count: 0 })),
      ])
      setSubscribedHere(here)
      setPermission(perm)
      setDeviceCount(status.count || 0)
    } catch (e) {
      toast.error('Could not read notification state', {
        description: e.message,
      })
    }
  }, [supported])

  useEffect(() => {
    if (open) refresh()
  }, [open, refresh])

  async function enableHere() {
    setBusy(true)
    try {
      await push.subscribe()
      toast.success('Notifications enabled on this device')
      await refresh()
    } catch (e) {
      toast.error('Enable failed', { description: e.message })
    } finally {
      setBusy(false)
    }
  }

  async function disableHere() {
    setBusy(true)
    try {
      await push.unsubscribe()
      toast.success('Notifications disabled on this device')
      await refresh()
    } catch (e) {
      toast.error('Disable failed', { description: e.message })
    } finally {
      setBusy(false)
    }
  }

  async function sendTest() {
    setBusy(true)
    try {
      const res = await push.sendTest()
      if (res.sent > 0) {
        toast.success(`Test sent to ${res.sent} device(s)`)
      } else {
        toast.warning('No devices received the test', {
          description:
            res.pruned > 0
              ? `${res.pruned} stale subscription(s) were removed — re-enable on this device.`
              : 'Enable notifications on this or another device first.',
        })
      }
      await refresh()
    } catch (e) {
      toast.error('Test push failed', { description: e.message })
    } finally {
      setBusy(false)
    }
  }

  // Permission-denied branch: the user (or an enterprise policy) has
  // explicitly blocked notifications. We can't re-prompt — the only remedy
  // is unblocking the site in the browser's site-settings dialog.
  const permBlocked = permission === 'denied'

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose?.()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Bell className="size-4" />
            Notifications
          </DialogTitle>
          <DialogDescription>
            Get a native push alert when a scheduled task finishes — even when
            Gigachat is not the active tab.
          </DialogDescription>
        </DialogHeader>

        {!supported ? (
          <div className="rounded-md border border-dashed border-border bg-muted/30 p-4 text-sm text-muted-foreground">
            <p className="mb-1 font-medium text-foreground">Not available</p>
            <p>
              This browser does not expose the Web Push API. Install the app to
              your home screen (PWA) or try the latest Chrome, Edge, or
              Firefox.
            </p>
          </div>
        ) : permBlocked ? (
          <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 p-4 text-sm">
            <ShieldAlert className="mt-0.5 size-4 shrink-0 text-destructive" />
            <div>
              <p className="font-medium text-destructive">
                Notifications blocked
              </p>
              <p className="mt-1 text-muted-foreground">
                You previously denied permission for this site. Re-enable it in
                your browser's site-settings (look for the lock icon in the
                address bar), then reload this page.
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-3 text-sm">
            <div className="rounded-md border border-border bg-card p-3">
              <div className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">
                This device
              </div>
              <div className="flex items-center justify-between gap-2">
                <p className="text-sm">
                  {subscribedHere === null
                    ? 'Checking…'
                    : subscribedHere
                      ? 'Notifications are enabled on this browser.'
                      : 'Notifications are not enabled on this browser.'}
                </p>
                {subscribedHere ? (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={disableHere}
                    disabled={busy}
                  >
                    <BellOff className="size-4" />
                    Disable
                  </Button>
                ) : (
                  <Button size="sm" onClick={enableHere} disabled={busy}>
                    <Bell className="size-4" />
                    Enable
                  </Button>
                )}
              </div>
            </div>

            <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
              <span>
                Active across{' '}
                <span className="font-medium text-foreground">
                  {deviceCount}
                </span>{' '}
                device{deviceCount === 1 ? '' : 's'} (browsers + installed PWAs).
              </span>
              <Button
                size="sm"
                variant="ghost"
                onClick={sendTest}
                disabled={busy || deviceCount === 0}
                title="Send a test notification to every enabled device"
              >
                <Send className="size-4" />
                Send test
              </Button>
            </div>
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
