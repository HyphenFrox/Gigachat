# Deploy the most-likely-to-change backend modules to a remote peer
# over SSH and bounce the backend via the WMI launcher.
#
# Why this exists: most "fix doesn't work yet" debugging cycles come
# down to "this box has the new code, that box doesn't." This bundles
# the four files we churn on most + the launcher, scp's them in one
# shot, then restarts via WMI so the spawned process survives the
# SSH session disconnect.
#
# Usage:
#   .\scripts\deploy_to_peer.ps1 -SshTarget gauta@192.168.1.9
#   .\scripts\deploy_to_peer.ps1 -SshTarget gautam-fbs       # uses ssh config alias
#
# Optional -RemoteRoot parameter (defaults to the standard install path).
param(
    [Parameter(Mandatory=$true)]
    [string]$SshTarget,
    [string]$RemoteRoot = 'C:/Users/gauta/Downloads/Gigachat'
)

$ErrorActionPreference = 'Stop'

# Resolve repo root from the script's own location so this works on
# any box without hard-coding usernames.
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

# Files we routinely touch when hacking on the P2P pair / discovery /
# secure-proxy paths. Add more here if you start editing modules
# outside this set.
$backendFiles = @(
    'backend\identity.py',
    'backend\p2p_secure_proxy.py',
    'backend\p2p_lan_client.py',
    'backend\p2p_discovery.py',
    'backend\p2p_rpc_server.py',
    'backend\p2p_binary_fetch.py',
    'backend\compute_pool.py',
    'backend\split_lifecycle.py',
    'backend\app.py'
)
$scriptFiles = @(
    'scripts\spawn_backend_via_wmi.ps1'
)

Write-Host "deploying to $SshTarget (root=$RemoteRoot)"

# Helper: scp a fixed list of files into one remote dir. We avoid
# `& scp @args` splatting because Windows scp mis-parses a single-
# element array (the destination ends up as a second source). Loop
# instead — one small extra round-trip but reliable.
function Send-Files {
    param([string[]]$Files, [string]$Dest)
    foreach ($f in $Files) {
        & scp $f $Dest
        if ($LASTEXITCODE -ne 0) { throw "scp $f -> $Dest failed" }
    }
}

$backendSrcs = $backendFiles | ForEach-Object { Join-Path $root $_ }
Send-Files -Files $backendSrcs -Dest "${SshTarget}:${RemoteRoot}/backend/"

$scriptSrcs = $scriptFiles | ForEach-Object { Join-Path $root $_ }
Send-Files -Files $scriptSrcs -Dest "${SshTarget}:${RemoteRoot}/scripts/"

Write-Host "files deployed; restarting remote backend"
$remoteCmd = "powershell -NoProfile -ExecutionPolicy Bypass -File ${RemoteRoot}/scripts/spawn_backend_via_wmi.ps1"
& ssh $SshTarget $remoteCmd
if ($LASTEXITCODE -ne 0) { throw "remote restart failed" }

Write-Host "done -- peer $SshTarget is on the latest code"
