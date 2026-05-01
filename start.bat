@echo off
REM ----------------------------------------------------------------------
REM start.bat  -  run Gigachat.
REM
REM Reads data/auth.json (or the GIGACHAT_HOST env var) to decide what
REM interface to bind:
REM   - "host": "127.0.0.1" / unset / "lan"  -> backend only
REM
REM ``lan`` mode binds to 0.0.0.0 with a source-IP allowlist (loopback +
REM private RFC1918 ranges) so other devices on the same Wi-Fi/Ethernet
REM can reach the app once they enter the password. The app is never
REM exposed to the public internet — that mode used to be wired through
REM cloudflared and was removed deliberately.
REM
REM Ollama is auto-started by the backend when it's installed; install
REM from https://ollama.com/download if you haven't yet.
REM ----------------------------------------------------------------------

setlocal
set ROOT=%~dp0
cd /d %ROOT%

REM Pick the right Python: prefer the project's local venv at .venv\,
REM fall back to whatever `python` resolves to on PATH. Avoids the
REM mixed-system+user-site PermissionError class that hits Windows
REM users on fresh installs.
set PY=python
if exist "%ROOT%.venv\Scripts\python.exe" set PY="%ROOT%.venv\Scripts\python.exe"

if not exist "%ROOT%frontend\dist\index.html" (
  echo [!] Frontend not built. Running build first...
  call "%ROOT%build.bat"
)

%PY% -m backend.server

endlocal
