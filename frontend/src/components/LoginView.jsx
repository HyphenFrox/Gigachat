import React, { useState } from 'react'
import { toast } from 'sonner'
import { Lock, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { api } from '@/lib/api'

/**
 * LoginView — full-viewport password gate shown when the backend is bound
 * to a non-loopback address and the user has no valid session cookie.
 *
 * This is the ONE screen that renders before the main app on a LAN
 * install (the only non-loopback bind mode supported). On a localhost
 * install the gate is skipped entirely by App.jsx, so the UX cost of
 * having this page is zero for the default setup.
 *
 * The page deliberately says "Gigachat" in plain text and shows the host
 * the server believes it's bound to — no sneaky re-branding, no hidden
 * redirect. If a link somehow puts you on the wrong server you can tell
 * by the banner before typing a password.
 */
export default function LoginView({ host, onAuthenticated }) {
  const [password, setPassword] = useState('')
  const [submitting, setSubmitting] = useState(false)

  async function submit(e) {
    e.preventDefault()
    if (!password || submitting) return
    setSubmitting(true)
    try {
      await api.login(password)
      toast.success('Signed in')
      setPassword('')
      onAuthenticated()
    } catch (err) {
      toast.error('Sign-in failed', {
        description: err.message.includes('401')
          ? 'Wrong password.'
          : err.message,
      })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="flex h-full w-full items-center justify-center bg-background p-4">
      <div className="w-full max-w-sm rounded-xl border border-border bg-card p-6 shadow-lg">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex size-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <Lock className="size-5" />
          </div>
          <div>
            <h1 className="text-lg font-semibold">Gigachat</h1>
            {host && (
              <p className="font-mono text-[11px] text-muted-foreground">
                {host}
              </p>
            )}
          </div>
        </div>
        <p className="mb-4 text-sm text-muted-foreground">
          This installation is exposed beyond localhost. Enter the password
          set in <code className="font-mono text-xs">data/auth.json</code> or
          the <code className="font-mono text-xs">GIGACHAT_PASSWORD</code>{' '}
          environment variable.
        </p>
        <form onSubmit={submit} className="space-y-3">
          <Input
            type="password"
            autoFocus
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            autoComplete="current-password"
            required
          />
          <Button type="submit" className="w-full" disabled={submitting}>
            {submitting ? (
              <>
                <Loader2 className="mr-2 size-4 animate-spin" />
                Signing in…
              </>
            ) : (
              'Sign in'
            )}
          </Button>
        </form>
      </div>
    </div>
  )
}
