# START_MT5.ps1
# Script to start MetaTrader 5 terminal on Windows VPS
# This script finds and launches the MT5 terminal automatically

Write-Host "[INFO] Starting MetaTrader 5 terminal..."

# Common MT5 installation paths on Windows
$mt5_paths = @(
    "C:\Program Files\MetaTrader 5\terminal64.exe",
    "C:\Program Files\MetaTrader 5\terminal.exe",
    "C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
    "C:\Program Files (x86)\MetaTrader 5\terminal.exe"
)

$terminal_found = $false

foreach ($path in $mt5_paths) {
    if (Test-Path $path) {
        Write-Host "[INFO] Found MT5 at: $path"
        Write-Host "[INFO] Starting terminal..."
        
        try {
            Start-Process -FilePath $path -WindowStyle Normal
            Write-Host "[SUCCESS] MetaTrader 5 terminal started."
            Write-Host "[INFO] Please log in to your trading account in the terminal."
            Write-Host "[INFO] Once logged in, your AUTO_TRADER.py can connect."
            $terminal_found = $true
            break
        } catch {
            Write-Host "[ERROR] Failed to start terminal at $path : $_"
        }
    }
}

if (-not $terminal_found) {
    Write-Host "[ERROR] MetaTrader 5 terminal not found in any of these locations:"
    foreach ($path in $mt5_paths) {
        Write-Host "  - $path"
    }
    Write-Host "[ERROR] Please install MetaTrader 5 or check the installation path."
    Write-Host "[INFO] You can also manually specify the path in your config.json"
    exit 1
}
