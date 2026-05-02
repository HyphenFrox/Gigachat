# Detached-process launcher for the Gigachat backend.
#
# Why this exists: when run via SSH, `Start-Process -WindowStyle Hidden`
# silently kills the spawned process when the SSH session disconnects,
# even with -PassThru. Win32_Process.Create via WMI doesn't have that
# problem — the spawned process is a true top-level process owned by
# the running user, completely detached from the SSH session.
#
# This is a one-shot helper for testing / remote re-deploys; the
# normal way to run the backend is `start.bat` or `dev.bat` from a
# console session, neither of which has the detach problem.

# Derive root from the script's own location so the launcher works on
# any machine without hard-coding a username (it was previously
# pinned to `gauta`, which only matched two of the three boxes).
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$py = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $py)) {
    Write-Host "ERROR: $py does not exist. Run install.bat first."
    exit 1
}

# Stop any existing backend before launching a new one.
Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "$root*" } |
    ForEach-Object {
        Write-Host "stopping existing backend pid=$($_.Id)"
        Stop-Process -Id $_.Id -Force
    }
Start-Sleep -Seconds 2

Write-Host "launching $py -m backend.server in $root"
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "`"$py`" -m backend.server"
    CurrentDirectory = $root
}
Write-Host "WMI returncode=$($r.ReturnValue) pid=$($r.ProcessId)"

if ($r.ReturnValue -ne 0) { exit 1 }

Start-Sleep -Seconds 6
$p = Get-Process -Id $r.ProcessId -ErrorAction SilentlyContinue
if ($p) {
    Write-Host "alive after 6s: pid=$($p.Id)"
} else {
    Write-Host "ERROR: pid $($r.ProcessId) died within 6s"
    exit 1
}
