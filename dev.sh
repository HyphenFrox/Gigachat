#!/usr/bin/env bash
# ----------------------------------------------------------------------
# dev.sh  -  start Gigachat in development mode.
#
# POSIX counterpart to dev.bat. Starts:
#   1) FastAPI backend with --reload on http://localhost:8000
#   2) Vite dev server with HMR on http://localhost:5173
#
# Open http://localhost:5173 in your browser. The dev server proxies
# /api/* to the backend automatically (see frontend/vite.config.js).
#
# Both processes attach to the current terminal as background jobs;
# Ctrl+C kills both via the trap below.
# ----------------------------------------------------------------------
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Pick the right Python: prefer .venv/, fall back to global python3.
if [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="$(command -v python3 || command -v python)"
  if [ -z "$PY" ]; then
    echo " [!] python3 not found on PATH and no .venv/ in this dir." >&2
    echo "     Run ./install.sh first." >&2
    exit 1
  fi
fi

echo
echo " Gigachat dev servers launching..."
echo "   Backend:   http://localhost:8000"
echo "   Frontend:  http://localhost:5173   (open this one)"
echo "   Python:    $PY"
echo
echo " (Ollama auto-starts in the background when the backend launches.)"
echo " Ctrl+C to stop both servers."
echo

# Spawn the backend in the background.
"$PY" -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Spawn the frontend dev server in the background.
( cd "$ROOT/frontend" && npm run dev ) &
FRONTEND_PID=$!

# Make sure both children die when the script does.
cleanup() {
  echo
  echo " Shutting down..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# Wait on the children. Returns when either exits.
wait -n "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
