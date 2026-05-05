#!/usr/bin/env bash
# ----------------------------------------------------------------------
# start.sh  -  run Gigachat in production mode (single port).
#
# POSIX counterpart to start.bat. Picks the local .venv/ if present,
# falls back to whatever `python3` is on PATH. Builds the frontend
# bundle on the fly if it's missing.
# ----------------------------------------------------------------------
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Pick the right Python: prefer the project's local .venv/, fall back
# to the global python3. Same isolation rationale as the .bat.
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

if [ ! -f "$ROOT/frontend/dist/index.html" ]; then
  echo " [!] Frontend not built. Running build first..."
  ( cd "$ROOT/frontend" && npm run build )
fi

exec "$PY" -m backend.server
