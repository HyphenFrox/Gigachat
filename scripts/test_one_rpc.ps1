# Test llama-server pointing at a single rpc-server endpoint to see
# what device it enumerates. Used to diagnose whether CPU-only
# rpc-servers are usable or get silently dropped.
param(
    [string]$RpcEndpoint = "192.168.1.9:50053",
    [string]$Model = "C:\Users\gauta\.ollama\models\blobs\sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29",
    [int]$Port = 11550,
    [int]$WaitSec = 25
)
$ErrorActionPreference = 'Continue'
$logPath = Join-Path $env:TEMP "llama-test-$Port.log"
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

$proc = Start-Process -FilePath "$env:USERPROFILE\.gigachat\llama-cpp\llama-server.exe" `
    -ArgumentList @(
        "--model", $Model,
        "--rpc", $RpcEndpoint,
        "--port", $Port,
        "-c", "1024",
        "--no-warmup",
        "-ngl", "99"
    ) `
    -RedirectStandardOutput $logPath `
    -RedirectStandardError "$logPath.err" `
    -WindowStyle Hidden -PassThru

Write-Host "spawned pid=$($proc.Id), log=$logPath, RPC=$RpcEndpoint"
Start-Sleep -Seconds $WaitSec
Get-Process -Id $proc.Id -ErrorAction SilentlyContinue | Stop-Process -Force

Write-Host "=== stdout filtered ==="
Get-Content $logPath -ErrorAction SilentlyContinue | Select-String -Pattern "using device|model buffer|RPC\d|memory_breakdown|failed|abort|server is listening|model loaded|register" | Select-Object -First 30 | ForEach-Object { Write-Host $_.Line }
Write-Host "=== stderr ==="
Get-Content "$logPath.err" -ErrorAction SilentlyContinue | Select-Object -First 10 | ForEach-Object { Write-Host $_ }
