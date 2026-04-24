@echo off
REM ----------------------------------------------------------------------
REM start.bat  -  run Gigachat.
REM
REM Reads data/auth.json to decide what to do:
REM   - "host": "public"   -> backend + cloudflared tunnel (public HTTPS)
REM   - anything else      -> backend only (loopback / tailscale)
REM
REM In public mode the backend serves on http://127.0.0.1:8000 and
REM cloudflared forwards https://<your-hostname> onto it. The tunnel
REM name and hostname are set once via:
REM     cloudflared tunnel login
REM     cloudflared tunnel create gigachat
REM     cloudflared tunnel route dns gigachat <your-hostname>
REM and this script just runs it afterwards. See README "Remote access".
REM
REM Ollama is auto-started by the backend when it's installed; install
REM from https://ollama.com/download if you haven't yet.
REM ----------------------------------------------------------------------

setlocal
set ROOT=%~dp0
cd /d %ROOT%

if not exist "%ROOT%frontend\dist\index.html" (
  echo [!] Frontend not built. Running build first...
  call "%ROOT%build.bat"
)

REM Detect public mode by grepping auth.json. The exact string match is
REM intentional: we only flip into public mode when the operator has
REM explicitly opted in, never as a silent default.
set PUBLIC_MODE=0
if exist "%ROOT%data\auth.json" (
  findstr /c:"\"host\": \"public\"" "%ROOT%data\auth.json" >nul 2>&1
  if not errorlevel 1 set PUBLIC_MODE=1
)

if "%PUBLIC_MODE%"=="1" (
  REM Sanity-check: the tunnel must exist before cloudflared can run it.
  where cloudflared >nul 2>&1
  if errorlevel 1 (
    echo [!] public mode is configured but cloudflared is not installed.
    echo     winget install --id Cloudflare.cloudflared
    pause
    exit /b 1
  )
  cloudflared tunnel list 2>nul | findstr /c:"gigachat" >nul
  if errorlevel 1 (
    echo [!] No cloudflared tunnel named "gigachat" was found.
    echo     Run these once ^(they need your Cloudflare account^):
    echo         cloudflared tunnel login
    echo         cloudflared tunnel create gigachat
    echo         cloudflared tunnel route dns gigachat YOUR_HOSTNAME
    pause
    exit /b 1
  )

  REM Launch the backend in its own window so each process's logs stay
  REM readable. Closing either window only kills that one process;
  REM close both to fully shut down the public endpoint.
  echo Starting backend on http://127.0.0.1:8000 ...
  start "Gigachat backend" cmd /k python -m backend.server

  REM Give uvicorn a moment to bind the socket before cloudflared starts
  REM forwarding. cloudflared will retry on its own if the backend is
  REM slow, but this avoids the initial 502s showing up in the logs.
  timeout /t 3 /nobreak >nul

  echo Starting cloudflared tunnel "gigachat" ...
  cloudflared tunnel run --url http://127.0.0.1:8000 gigachat
) else (
  python -m backend.server
)

endlocal
