@echo off
REM ----------------------------------------------------------------------
REM uninstall-service.bat  -  remove the auto-start scheduled task and
REM                           the inbound-port-8000 firewall rule.
REM
REM Symmetric counterpart to install-service.bat. After this, the backend
REM no longer auto-launches at logon and other devices on the LAN can no
REM longer reach this device's P2P endpoints. The .venv\, the source
REM tree, and any stored data (data/app.db, identity.json, etc.) are
REM untouched — uninstall the service, not the app.
REM
REM Requires: Administrator.
REM ----------------------------------------------------------------------

setlocal

net session >nul 2>&1
if errorlevel 1 (
  echo.
  echo  [!] This script needs to run as Administrator.
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'SilentlyContinue';" ^
  "Write-Host '[1/2] Stopping + unregistering Scheduled Task ''Gigachat''...';" ^
  "Stop-ScheduledTask -TaskName 'Gigachat';" ^
  "Unregister-ScheduledTask -TaskName 'Gigachat' -Confirm:$false;" ^
  "Write-Host '       OK';" ^
  "Write-Host '[2/2] Removing Windows Firewall rule (TCP 8000)...';" ^
  "Get-NetFirewallRule -DisplayName 'Gigachat backend (port 8000)' | Remove-NetFirewallRule;" ^
  "Get-NetFirewallRule -DisplayName 'Gigachat backend (port 8000, Private)' | Remove-NetFirewallRule;" ^
  "Write-Host '       OK';"

echo.
echo  ====================================================
echo   Gigachat service removed. The app itself is intact:
echo     - .venv\, source code, data\ all untouched
echo     - Manual start still works:  start.bat
echo     - To re-enable auto-start:   install-service.bat
echo  ====================================================
echo.

endlocal
