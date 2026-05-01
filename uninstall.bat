@echo off
REM ----------------------------------------------------------------------
REM uninstall.bat  -  remove everything install.bat created.
REM
REM Symmetric counterpart to install.bat. Removes:
REM   1. Scheduled Task "Gigachat" (auto-start)
REM   2. Windows Firewall rule for TCP 8000
REM   3. .venv\                     (Python virtualenv + every backend dep)
REM   4. frontend\node_modules\     (every npm dep)
REM   5. frontend\dist\             (production build output)
REM
REM Optionally also removes:
REM   6. data\                      (chats, identity.json, memories,
REM                                  uploads, screenshots, etc.)
REM      ^-- Prompted before removal. Keeping it is the default so you can
REM          re-install later without losing your conversations + your
REM          P2P device identity.
REM
REM Does NOT touch:
REM   - The source tree (.py / .jsx / docs / scripts).
REM   - Anything outside this folder. No global Python / Node packages
REM     are removed.
REM
REM Re-runnable: every step tolerates "already gone" so you can run it
REM after a partial uninstall.
REM
REM Requires: Administrator (the firewall rule + scheduled task removal
REM           need elevation).
REM ----------------------------------------------------------------------

setlocal
set ROOT=%~dp0

REM -- Self-elevate to Administrator -------------------------------------
net session >nul 2>&1
if errorlevel 1 (
  echo  Requesting Administrator privileges...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b 0
)

cd /d %ROOT%

echo.
echo  ====================================================
echo   Gigachat - uninstall
echo  ====================================================
echo.

REM -- 1 + 2: scheduled task + firewall rule (admin-only operations) ----
echo  [1/5] Stopping + unregistering Scheduled Task "Gigachat" ...
echo  [2/5] Removing Windows Firewall rule for TCP 8000 ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'SilentlyContinue';" ^
  "Stop-ScheduledTask -TaskName 'Gigachat';" ^
  "Unregister-ScheduledTask -TaskName 'Gigachat' -Confirm:$false;" ^
  "Get-NetFirewallRule -DisplayName 'Gigachat backend (port 8000)' | Remove-NetFirewallRule;" ^
  "Get-NetFirewallRule -DisplayName 'Gigachat backend (port 8000, Private)' | Remove-NetFirewallRule;"

REM Give the OS a beat to release any file locks the running backend
REM might still hold on .venv\ files (the scheduled-task stop above
REM signals shutdown but Python releases handles asynchronously).
timeout /t 2 /nobreak >nul

REM Defence-in-depth: kill any python.exe still holding files inside
REM .venv\ — typically the just-stopped backend. Without this, the
REM rmdir below can fail with "file in use" on busy systems.
powershell -NoProfile -Command ^
  "Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '%ROOT:\=\\%.venv\\*' } | Stop-Process -Force -ErrorAction SilentlyContinue"
timeout /t 1 /nobreak >nul

REM -- 3: .venv ---------------------------------------------------------
echo  [3/5] Removing .venv\ ...
if exist "%ROOT%.venv" (
  rmdir /s /q "%ROOT%.venv"
)

REM -- 4: frontend\node_modules ----------------------------------------
echo  [4/5] Removing frontend\node_modules\ ...
if exist "%ROOT%frontend\node_modules" (
  rmdir /s /q "%ROOT%frontend\node_modules"
)

REM -- 5: frontend\dist ------------------------------------------------
echo  [5/5] Removing frontend\dist\ ...
if exist "%ROOT%frontend\dist" (
  rmdir /s /q "%ROOT%frontend\dist"
)

REM -- Optional: data\ wipe (interactive prompt) -----------------------
echo.
if exist "%ROOT%data" (
  echo  Optional: also delete user data?
  echo    .\data\  contains chat history, paired-device identity,
  echo             memories, uploads, screenshots, audit log.
  echo    Keeping it ^(default^) lets you re-install later without
  echo    losing conversations or the P2P device identity peers
  echo     paired with.
  echo.
  set /p WIPEDATA="  Delete data\ too? [y/N] "
  if /i "%WIPEDATA%"=="y" (
    rmdir /s /q "%ROOT%data"
    echo  [+] data\ removed.
  ) else (
    echo  [+] data\ kept ^(re-install later picks up where you left off^).
  )
)

echo.
echo  ====================================================
echo   Uninstall complete.
echo.
echo   The source tree is intact. To re-install at any
echo   time, run:
echo     install.bat
echo  ====================================================
echo.
pause
endlocal
