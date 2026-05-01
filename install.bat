@echo off
REM ----------------------------------------------------------------------
REM install.bat  -  one-shot Gigachat setup. Run this once and you're done.
REM
REM Does everything end-to-end:
REM   1. Self-elevates to Administrator (UAC prompt) if not already.
REM   2. Creates an isolated Python virtualenv at .venv\ and installs
REM      the backend dependencies into it.
REM   3. npm-installs the frontend and builds the production bundle
REM      (so start.bat / the scheduled task can serve it directly).
REM   4. Adds an inbound Windows Firewall rule for TCP 8000 (Private
REM      profile only) so other Gigachat installs on the same Wi-Fi /
REM      Ethernet can reach this device's P2P endpoints.
REM   5. Registers a Scheduled Task "Gigachat" that auto-starts the
REM      backend at user logon and restarts on crash.
REM   6. Starts the task so the backend is up immediately.
REM
REM Re-runnable: every step is idempotent. Run this again after
REM `git pull` to refresh deps + frontend bundle without breaking the
REM running task.
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
REM Firewall + Scheduled Task changes need elevation. `net session` is
REM the canonical "am I admin" check (returns 0 only as admin). If not,
REM PowerShell re-launches THIS script via UAC; the original (non-admin)
REM cmd window exits and a new admin one takes over.
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
  echo  [1/6] Creating Python virtualenv at .venv\ ...
  python -m venv .venv
  if errorlevel 1 (
    echo.
    echo  [!] Failed to create .venv. Make sure Python 3.12+ is on PATH:
    echo        python --version
    pause
    exit /b 1
  )
) else (
  echo  [1/6] Reusing existing .venv\ ...
)

REM -- 2. Backend deps --------------------------------------------------
echo  [2/6] Installing backend dependencies into .venv\ ...
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
echo  [3/6] Installing + building frontend ...
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

REM -- 4. Firewall rule + 5. Scheduled task -----------------------------
REM Both run via one PowerShell invocation so we don't pay the launch
REM cost twice and so a partial failure leaves a clean state. Both
REM operations are idempotent (Remove-* before Register-*).
echo  [4/6] Adding Windows Firewall rule ^(TCP 8000, Private profile^) ...
echo  [5/6] Registering Scheduled Task "Gigachat" ^(auto-start at logon^) ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'Stop';" ^
  "Get-NetFirewallRule -DisplayName 'Gigachat backend (port 8000)' -ErrorAction SilentlyContinue | Remove-NetFirewallRule;" ^
  "Get-NetFirewallRule -DisplayName 'Gigachat backend (port 8000, Private)' -ErrorAction SilentlyContinue | Remove-NetFirewallRule;" ^
  "New-NetFirewallRule -DisplayName 'Gigachat backend (port 8000)' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -Profile Private -Description 'Allows P2P endpoints (encrypted compute proxy + pair handshake) to be reached from other Gigachat installs on the same physical network.' | Out-Null;" ^
  "$root = '%ROOT:~0,-1%';" ^
  "$py = Join-Path $root '.venv\Scripts\python.exe';" ^
  "$user = $env:USERDOMAIN + '\\' + $env:USERNAME;" ^
  "$action = New-ScheduledTaskAction -Execute $py -Argument '-m backend.server' -WorkingDirectory $root;" ^
  "$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user;" ^
  "$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1);" ^
  "$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited;" ^
  "Unregister-ScheduledTask -TaskName 'Gigachat' -Confirm:$false -ErrorAction SilentlyContinue;" ^
  "Register-ScheduledTask -TaskName 'Gigachat' -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'Gigachat backend (auto-starts at logon).' | Out-Null;"

if errorlevel 1 (
  echo.
  echo  [!] Firewall / Scheduled Task setup failed. See PowerShell output above.
  pause
  exit /b 1
)

REM -- 6. Start the backend now ----------------------------------------
echo  [6/6] Starting Gigachat ...
powershell -NoProfile -Command "Start-ScheduledTask -TaskName 'Gigachat'" >nul
REM Give uvicorn a moment to bind the port before we report success.
timeout /t 5 /nobreak >nul

REM -- Done -------------------------------------------------------------
echo.
echo  ====================================================
echo   Gigachat is running.
echo.
echo   Open in your browser:  http://localhost:8000
echo.
echo   The backend auto-starts at logon and restarts on
echo   crash, so once installed you don't need to think
echo   about it. To stop the backend now:
echo     Stop-ScheduledTask -TaskName Gigachat
echo.
echo   To remove everything ^(firewall rule, scheduled task,
echo   .venv, node_modules, frontend build^), run:
echo     uninstall.bat
echo  ====================================================
echo.
pause
endlocal
