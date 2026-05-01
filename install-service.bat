@echo off
REM ----------------------------------------------------------------------
REM install-service.bat  -  set up Gigachat to auto-start + be reachable.
REM
REM Two one-shot setup steps that need Administrator:
REM
REM   1. Adds an inbound Windows Firewall rule for TCP 8000 (Private
REM      profile only). Without this, other Gigachat installs on the
REM      same Wi-Fi/Ethernet can't reach this device's P2P endpoints
REM      (encrypted compute proxy + pair handshake) — pairing fails
REM      silently with a connection timeout.
REM
REM   2. Registers a Scheduled Task "Gigachat" that launches the backend
REM      at user logon. Survives SSH session ends, login/logout cycles,
REM      and reboots — once you set it up you don't have to think about
REM      `start.bat` again.
REM
REM Re-runnable: both steps are idempotent (existing firewall rule and
REM scheduled task get replaced cleanly).
REM
REM Uninstall: run `uninstall-service.bat`.
REM
REM Requires: Administrator. Right-click → Run as administrator, OR
REM elevate from an admin terminal. Setup.bat must have been run first
REM so .venv\ exists.
REM ----------------------------------------------------------------------

setlocal
set ROOT=%~dp0
cd /d %ROOT%

REM Confirm we're elevated. `net session` returns 0 only when running
REM as Administrator; non-admin gets "Access is denied".
net session >nul 2>&1
if errorlevel 1 (
  echo.
  echo  [!] This script needs to run as Administrator.
  echo      Right-click install-service.bat and choose "Run as administrator",
  echo      OR open an elevated PowerShell / Command Prompt and re-run it.
  exit /b 1
)

REM Confirm setup.bat has been run (otherwise the scheduled task points
REM at a non-existent .venv\ python that would silently fail on every
REM logon).
if not exist "%ROOT%.venv\Scripts\python.exe" (
  echo.
  echo  [!] No .venv\ found. Run setup.bat first to create the Python
  echo      virtualenv and install dependencies, THEN re-run this.
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'Stop';" ^
  "Write-Host '[1/2] Adding Windows Firewall rule (TCP 8000, Private profile)...';" ^
  "Get-NetFirewallRule -DisplayName 'Gigachat backend (port 8000)' -ErrorAction SilentlyContinue | Remove-NetFirewallRule;" ^
  "New-NetFirewallRule -DisplayName 'Gigachat backend (port 8000)' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -Profile Private -Description 'Allows P2P endpoints (encrypted compute proxy + pair handshake) to be reached from other Gigachat installs on the same physical network. Created by install-service.bat.' | Out-Null;" ^
  "Write-Host '       OK';" ^
  "$root = '%ROOT:~0,-1%';" ^
  "$py = Join-Path $root '.venv\Scripts\python.exe';" ^
  "$user = $env:USERDOMAIN + '\\' + $env:USERNAME;" ^
  "Write-Host '[2/2] Registering Scheduled Task ''Gigachat'' (auto-start at logon)...';" ^
  "$action = New-ScheduledTaskAction -Execute $py -Argument '-m backend.server' -WorkingDirectory $root;" ^
  "$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user;" ^
  "$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1);" ^
  "$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited;" ^
  "Unregister-ScheduledTask -TaskName 'Gigachat' -Confirm:$false -ErrorAction SilentlyContinue;" ^
  "Register-ScheduledTask -TaskName 'Gigachat' -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'Gigachat backend (auto-starts at logon).' | Out-Null;" ^
  "Start-ScheduledTask -TaskName 'Gigachat';" ^
  "Start-Sleep -Seconds 3;" ^
  "Write-Host '       OK (started; verify with `Get-ScheduledTask -TaskName Gigachat`)';"

if errorlevel 1 (
  echo.
  echo  [!] Setup failed. Re-run from an elevated console and check the error above.
  exit /b 1
)

echo.
echo  ====================================================
echo   Gigachat is now set up to auto-start at logon.
echo.
echo   Chat UI:        http://localhost:8000  (this machine)
echo   P2P endpoints:  reachable on this machine's LAN IP
echo                   from other Gigachat installs on the
echo                   same Wi-Fi/Ethernet.
echo.
echo   To stop the backend NOW (without uninstalling):
echo     Stop-ScheduledTask -TaskName Gigachat
echo.
echo   To remove the firewall rule + scheduled task entirely:
echo     uninstall-service.bat  (also as Administrator)
echo  ====================================================
echo.

endlocal
