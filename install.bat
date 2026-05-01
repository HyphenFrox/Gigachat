@echo off
REM ----------------------------------------------------------------------
REM install.bat  -  one-shot Gigachat setup. Run this once and you're done.
REM
REM Does end-to-end:
REM   1. Self-elevates to Administrator (UAC prompt) if not already —
REM      needed only for the firewall rule in step 4.
REM   2. Creates an isolated Python virtualenv at .venv\ and installs
REM      the backend dependencies into it.
REM   3. npm-installs the frontend and builds the production bundle
REM      (so start.bat can serve it directly).
REM   4. Adds an inbound Windows Firewall rule for TCP 8000 (Private
REM      profile only) so other Gigachat installs on the same Wi-Fi /
REM      Ethernet can reach this device's P2P endpoints (encrypted
REM      compute proxy + pair handshake). Without the rule, pairing
REM      from another device fails with a silent connection timeout.
REM
REM No background scheduled task, no auto-start. After install.bat
REM finishes, the user explicitly launches the backend with one of:
REM
REM   dev.bat    development mode (Vite hot-reload + uvicorn --reload).
REM              Two console windows; visit http://localhost:5173.
REM   start.bat  production single-port (FastAPI serves the built
REM              frontend at http://localhost:8000). One window.
REM
REM Re-runnable: every step is idempotent. Run this again after
REM `git pull` to refresh deps + rebuild the frontend bundle.
REM
REM Uninstall: run `uninstall.bat`.
REM
REM Requires:  Python 3.12+, Node 20+. Ollama is auto-started by the
REM            backend if it's installed; otherwise install it from
REM            https://ollama.com/download.
REM ----------------------------------------------------------------------

setlocal
set ROOT=%~dp0

REM -- Self-elevate to Administrator -------------------------------------
REM The firewall rule needs elevation. `net session` is the canonical
REM "am I admin" check (returns 0 only as admin). If not, PowerShell
REM re-launches THIS script via UAC; the original (non-admin) cmd
REM window exits and a new admin one takes over.
net session >nul 2>&1
if errorlevel 1 (
  echo  Requesting Administrator privileges...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b 0
)

cd /d %ROOT%

echo.
echo  ====================================================
echo   Gigachat - one-shot install
echo  ====================================================
echo.

REM -- 1. Python venv ---------------------------------------------------
if not exist "%ROOT%.venv\Scripts\python.exe" (
  echo  [1/4] Creating Python virtualenv at .venv\ ...
  python -m venv .venv
  if errorlevel 1 (
    echo.
    echo  [!] Failed to create .venv. Make sure Python 3.12+ is on PATH:
    echo        python --version
    pause
    exit /b 1
  )
) else (
  echo  [1/4] Reusing existing .venv\ ...
)

REM -- 2. Backend deps --------------------------------------------------
echo  [2/4] Installing backend dependencies into .venv\ ...
call "%ROOT%.venv\Scripts\python.exe" -m pip install --upgrade pip
call "%ROOT%.venv\Scripts\python.exe" -m pip install -r backend\requirements.txt
if errorlevel 1 (
  echo.
  echo  [!] Backend deps install failed. Common causes:
  echo        - Antivirus scanning the venv mid-write. Add %ROOT%.venv\ to
  echo          your AV's exclusion list and retry.
  echo        - Network blocked. Try `python -m pip install --upgrade pip`
  echo          first to confirm PyPI is reachable.
  pause
  exit /b 1
)

REM -- 3. Frontend deps + production build ------------------------------
echo  [3/4] Installing + building frontend ...
cd /d %ROOT%frontend
call npm install
if errorlevel 1 (
  echo.
  echo  [!] npm install failed. Make sure Node 20+ is on PATH:
  echo        node --version
  pause
  exit /b 1
)
call npm run build
if errorlevel 1 (
  echo.
  echo  [!] Frontend build failed. See output above.
  pause
  exit /b 1
)
cd /d %ROOT%

REM -- 4. Firewall rule -------------------------------------------------
echo  [4/4] Adding Windows Firewall rule ^(TCP 8000, Private profile^) ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'Stop';" ^
  "Get-NetFirewallRule -DisplayName 'Gigachat backend (port 8000)' -ErrorAction SilentlyContinue | Remove-NetFirewallRule;" ^
  "Get-NetFirewallRule -DisplayName 'Gigachat backend (port 8000, Private)' -ErrorAction SilentlyContinue | Remove-NetFirewallRule;" ^
  "New-NetFirewallRule -DisplayName 'Gigachat backend (port 8000)' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -Profile Private -Description 'Allows P2P endpoints (encrypted compute proxy + pair handshake) to be reached from other Gigachat installs on the same physical network.' | Out-Null;"

if errorlevel 1 (
  echo.
  echo  [!] Firewall rule setup failed. See PowerShell output above.
  pause
  exit /b 1
)

REM -- Done -------------------------------------------------------------
echo.
echo  ====================================================
echo   Setup complete.
echo.
echo   Now run ONE of:
echo     dev.bat    development ^(Vite hot-reload, port 5173^)
echo     start.bat  production single-port ^(http://localhost:8000^)
echo.
echo   To remove everything ^(firewall rule, .venv, node_modules,
echo   frontend build^), run:
echo     uninstall.bat
echo  ====================================================
echo.
pause
endlocal
