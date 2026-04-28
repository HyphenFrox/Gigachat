# ----------------------------------------------------------------------
# install-worker-tools.ps1 — one-time setup on a compute-pool worker.
#
# Installs the optional Python libraries the host's distributed-tool
# dispatchers need to route work onto this worker. Without these, the
# host's `read_doc` and `web_search` tools quietly fall back to
# running on host alone (no error, no break — just no offload).
#
# Run once per worker, in PowerShell, with internet access:
#
#     powershell -ExecutionPolicy Bypass -File install-worker-tools.ps1
#
# Idempotent — re-running just upgrades the packages to whatever
# `requirements.txt` pins on the host.
#
# Worker-side requirements (assumed already present):
#   - OpenSSH server (Windows 10+ ships it; enable via
#     `Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0`)
#   - PowerShell 5.1+ (default on Windows 10+)
#   - Python 3.11+ on PATH (install from https://python.org or via
#     `winget install Python.Python.3.12`)
# ----------------------------------------------------------------------

$ErrorActionPreference = "Stop"

# Resolve the worker's Python interpreter — `py.exe` first (Python
# launcher, robust to PATH ordering); fall back to bare `python.exe`.
$python = (Get-Command py.exe -ErrorAction SilentlyContinue).Source
if (-not $python) {
    $python = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
}
if (-not $python) {
    Write-Error "Python is not installed or not on PATH. Install from https://python.org and re-run."
    exit 1
}

Write-Host "Using Python: $python"

# pymupdf, python-docx, openpyxl — read_doc parsers
# ddgs                                 — web_search backend
$packages = @(
    "pymupdf",
    "python-docx",
    "openpyxl",
    "ddgs"
)

Write-Host "Installing/upgrading: $($packages -join ', ')"
& $python -m pip install --user --upgrade @packages
if ($LASTEXITCODE -ne 0) {
    Write-Error "pip install failed (exit $LASTEXITCODE). Check the output above for the failing package."
    exit $LASTEXITCODE
}

# Verify each library imports cleanly so the host's probe sees them
# the next time it sweeps capabilities.
& $python -c @"
import sys
ok = True
for mod in ('pymupdf', 'docx', 'openpyxl', 'ddgs'):
    try:
        __import__(mod)
        print(f'  OK  {mod}')
    except ImportError as e:
        print(f'  FAIL {mod}: {e}')
        ok = False
sys.exit(0 if ok else 2)
"@
if ($LASTEXITCODE -ne 0) {
    Write-Error "Verification failed — one or more libraries didn't import after install."
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Worker tools installed. The host's next capability probe (~5 min) picks up the new libs;" -ForegroundColor Green
Write-Host "after that, distributed read_doc and web_search will route through this worker." -ForegroundColor Green
