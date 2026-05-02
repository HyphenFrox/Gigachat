# Test TCP reachability of every rpc-server endpoint we expect to be
# usable. Runs from any peer; reports per-endpoint reachable=true/false.
$endpoints = @(
    "192.168.1.4:50052",  # local iGPU
    "192.168.1.4:50053",  # local CPU
    "192.168.1.9:50052",  # naresh iGPU
    "192.168.1.9:50053"   # naresh CPU
)
foreach ($ep in $endpoints) {
    $parts = $ep.Split(":")
    $tcp = New-Object System.Net.Sockets.TcpClient
    try {
        $task = $tcp.ConnectAsync($parts[0], [int]$parts[1])
        $ok = $task.Wait(3000) -and $tcp.Connected
        Write-Host ("{0,-22} reachable={1}" -f $ep, $ok)
        $tcp.Close()
    } catch {
        Write-Host ("{0,-22} ERR: {1}" -f $ep, $_.Exception.Message)
    }
}
