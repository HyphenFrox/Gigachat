import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

/**
 * Vite config.
 *
 * Dev-mode setup:
 *   - React Fast Refresh via @vitejs/plugin-react.
 *   - Path alias "@" -> "./src" (matches shadcn/ui convention).
 *   - All /api/* requests are proxied to the FastAPI backend on :8000.
 *     The proxy preserves streaming so Server-Sent Events from the agent
 *     flush incrementally rather than buffering the full response.
 *
 * Production build:
 *   - `npm run build` emits to ./dist which FastAPI serves as static files.
 */
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // Keep SSE connections alive and unbuffered.
        ws: false,
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            // Tell any intermediate gzip layer to back off for event streams.
            if ((proxyRes.headers['content-type'] || '').includes('text/event-stream')) {
              proxyRes.headers['cache-control'] = 'no-cache'
              proxyRes.headers['x-accel-buffering'] = 'no'
            }
          })
        },
      },
    },
  },
})
