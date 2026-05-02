# Pull the latest llama.cpp binary bundle and overlay it onto the
# install at C:\Users\<user>\.gigachat\llama-cpp\.
#
# Usage:
#   .\update_llamacpp.ps1 -Variant cuda  # NVIDIA host
#   .\update_llamacpp.ps1 -Variant sycl  # Intel iGPU host
#   .\update_llamacpp.ps1 -Variant vulkan
#
# Stops any running llama-server / rpc-server first so file replacement
# isn't blocked by handles. Idempotent.
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('cuda', 'sycl', 'vulkan')]
    [string]$Variant,
    [string]$Build = 'b9002'
)
$ErrorActionPreference = 'Stop'
$dir = Join-Path $env:USERPROFILE '.gigachat\llama-cpp'
$tmp = Join-Path $env:TEMP "llama-$Build-$Variant.zip"
$out = Join-Path $env:TEMP "llama-$Build-$Variant"

# Variant -> release asset name
$asset = switch ($Variant) {
    'cuda'   { "llama-$Build-bin-win-cuda-12.4-x64.zip" }
    'sycl'   { "llama-$Build-bin-win-sycl-x64.zip" }
    'vulkan' { "llama-$Build-bin-win-vulkan-x64.zip" }
}
$url = "https://github.com/ggml-org/llama.cpp/releases/download/$Build/$asset"

try {
    $oldVer = (cmd /c "`"$dir\llama-server.exe`" --version 2>&1") | Select-String 'version:' | Select-Object -First 1
    Write-Host "old version: $oldVer"
} catch {
    Write-Host "old version: (could not query)"
}

# Stop running processes so file overlay isn't blocked.
Get-Process llama-server,rpc-server -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "stopping pid=$($_.Id) ($($_.Name))"
    Stop-Process -Id $_.Id -Force
}
Start-Sleep -Seconds 2

# Clean temp from prior runs.
if (Test-Path $out) { Remove-Item $out -Recurse -Force }
if (Test-Path $tmp) { Remove-Item $tmp -Force }

Write-Host "downloading $asset"
Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing
Write-Host "extracting"
Expand-Archive -Path $tmp -DestinationPath $out -Force

# Overlay just executables + DLLs + SPIR-V kernels, leave logs/ etc.
# alone. -ErrorAction SilentlyContinue skips files that don't exist
# in the source bundle for THIS variant (e.g. SYCL bundle lacks
# .spv files; that's fine).
Copy-Item -Path "$out\*.exe" -Destination $dir -Force -ErrorAction SilentlyContinue
Copy-Item -Path "$out\*.dll" -Destination $dir -Force -ErrorAction SilentlyContinue
Copy-Item -Path "$out\*.spv" -Destination $dir -Force -ErrorAction SilentlyContinue

try {
    $newVer = (cmd /c "`"$dir\llama-server.exe`" --version 2>&1") | Select-String 'version:' | Select-Object -First 1
    Write-Host "new version: $newVer"
} catch {
    Write-Host "new version: (could not query)"
}
Get-ChildItem $dir -Filter ggml-*.dll | Where-Object { $_.Name -match 'rpc|sycl|vulkan|cuda' } | Format-Table Name, @{N='MB';E={[math]::Round($_.Length/1MB,1)}} -AutoSize | Out-String | Write-Host
