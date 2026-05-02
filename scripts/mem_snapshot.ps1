# Quick system memory snapshot + the top 10 processes by working set.
# Used during P2P load testing to figure out which side is hitting
# the memory ceiling (peer running the model usually shows >50%
# of physical RAM in `ollama.exe` working set).
$os = Get-CimInstance Win32_OperatingSystem
$totalMB = [math]::Round($os.TotalVisibleMemorySize / 1024)
$freeMB = [math]::Round($os.FreePhysicalMemory / 1024)
$usedMB = $totalMB - $freeMB
$pct = [math]::Round(($usedMB / $totalMB) * 100, 1)
Write-Host ("hostname:  " + $env:COMPUTERNAME)
Write-Host ("total:     {0} MB" -f $totalMB)
Write-Host ("used:      {0} MB ({1}%)" -f $usedMB, $pct)
Write-Host ("free:      {0} MB" -f $freeMB)
Write-Host ""
Write-Host "--- top 10 processes by working set ---"
Get-Process | Sort-Object -Property WorkingSet64 -Descending |
    Select-Object -First 10 -Property `
        @{Name="MB"; Expression={[math]::Round($_.WorkingSet64/1MB)}},
        ProcessName,
        Id |
    Format-Table -AutoSize | Out-String | Write-Host
