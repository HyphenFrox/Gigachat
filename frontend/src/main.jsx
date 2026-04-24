import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.scss'
import { registerServiceWorker } from './lib/pwa'

// Single React root — standard Vite template. StrictMode is disabled on
// purpose: it double-invokes effects in dev, which would open two SSE
// connections per turn and make the UI state harder to reason about.
ReactDOM.createRoot(document.getElementById('root')).render(<App />)

// Kick off the PWA service worker once the app has mounted. Failures are
// logged to the console — the app still works without SW, it just loses
// offline-shell caching and push notifications.
if (typeof window !== 'undefined') {
  window.addEventListener('load', () => {
    registerServiceWorker()
  })
}
