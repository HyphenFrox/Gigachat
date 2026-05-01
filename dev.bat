@echo off
REM ----------------------------------------------------------------------
REM dev.bat  -  start Gigachat in development mode.
REM
REM Opens two new console windows:
REM   1) FastAPI backend with --reload on http://localhost:8000
REM   2) Vite dev server with HMR on http://localhost:5173
REM
REM Open http://localhost:5173 in your browser. The dev server proxies
REM /api/* to the backend automatically (see frontend/vite.config.js).
REM
REM Requires: Python 3.12+, Node 20+. Ollama is auto-started by the backend
REM if it's installed; otherwise install from https://ollama.com/download.
REM
REM Virtualenv-aware: when .venv\Scripts\python.exe exists at the project
REM root we use it instead of the global `python`. Avoids the mixed-install
REM permission errors that hit fresh Windows setups when the system + user
REM site-packages dirs get tangled. Create one with:
REM     python -m venv .venv
REM     .\.venv\Scripts\activate
REM     python -m pip install -r backend\requirements.txt
REM ----------------------------------------------------------------------

setlocal
set ROOT=%~dp0

REM Pick the right Python: prefer the project's local venv, fall back to
REM whatever `python` resolves to on PATH. The path comparison is just a
REM file existence check — no `where` invocation, no shell-out cost.
set PY=python
if exist "%ROOT%.venv\Scripts\python.exe" set PY="%ROOT%.venv\Scripts\python.exe"

REM Force loopback mode for dev regardless of what data/auth.json says. The
REM env var wins over the file, so running dev.bat skips the password prompt
REM even if auth.json is configured for "lan". The production launcher
REM (start.bat) still honors auth.json.
set GIGACHAT_HOST=127.0.0.1

start "Gigachat - backend" cmd /k "cd /d %ROOT% && set GIGACHAT_HOST=127.0.0.1&& %PY% -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload"
start "Gigachat - frontend" cmd /k "cd /d %ROOT%frontend && npm run dev"

echo.
echo  Gigachat dev servers launching...
echo    Backend:   http://localhost:8000
echo    Frontend:  http://localhost:5173   (open this one)
if exist "%ROOT%.venv\Scripts\python.exe" (
  echo    Python:    .venv\Scripts\python.exe
) else (
  echo    Python:    global ^(no .venv detected; create one to isolate deps^)
)
echo.
echo  (Ollama auto-starts in the background when the backend launches.)
endlocal
