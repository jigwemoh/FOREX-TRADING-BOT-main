# SETUP_MT5_AUTOSTART.ps1
# Script to set up MetaTrader 5 to auto-start on Windows VPS boot
# Run this script as Administrator

Write-Host "[INFO] Setting up MetaTrader 5 auto-start on system boot..."
Write-Host "[WARNING] This script requires Administrator privileges."

# Check if running as Administrator
$isAdmin = [bool]([System.Security.Principal.WindowsIdentity]::GetCurrent().Groups -match "S-1-5-32-544")
if (-not $isAdmin) {
    Write-Host "[ERROR] This script must be run as Administrator!"
    Write-Host "[INFO] Right-click PowerShell and select 'Run as Administrator'"
    exit 1
}

# Common MT5 installation paths on Windows
$mt5_paths = @(
    "C:\Program Files\MetaTrader 5\terminal64.exe",
    "C:\Program Files\MetaTrader 5\terminal.exe",
    "C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
    "C:\Program Files (x86)\MetaTrader 5\terminal.exe"
)

$terminal_found = $false
$mt5_path = ""

foreach ($path in $mt5_paths) {
    if (Test-Path $path) {
        Write-Host "[SUCCESS] Found MT5 at: $path"
        $terminal_found = $true
        $mt5_path = $path
        break
    }
}

if (-not $terminal_found) {
    Write-Host "[ERROR] MetaTrader 5 terminal not found in any of these locations:"
    foreach ($path in $mt5_paths) {
        Write-Host "  - $path"
    }
    Write-Host "[ERROR] Please install MetaTrader 5 first."
    exit 1
}

# Create a Windows Task Scheduler task to start MT5 at boot
$taskName = "StartMetaTrader5"
$taskDescription = "Automatically start MetaTrader 5 terminal at system boot"

# Check if task already exists
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "[INFO] Task '$taskName' already exists. Removing it..."
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

# Create a new scheduled task
Write-Host "[INFO] Creating scheduled task..."

# Task action: Start MT5 terminal
$action = New-ScheduledTaskAction -Execute $mt5_path -WorkingDirectory (Split-Path $mt5_path)

# Task trigger: At system startup
$trigger = New-ScheduledTaskTrigger -AtStartup

# Task settings
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable

# Register the task
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

Register-ScheduledTask -TaskName $taskName `
    -Description $taskDescription `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Force

Write-Host "[SUCCESS] Windows Task Scheduler task created!"
Write-Host "[INFO] MetaTrader 5 will now automatically start when your VPS boots up."
Write-Host "[INFO] To manually test, run: powershell -ExecutionPolicy Bypass -File .\START_MT5.ps1"
Write-Host "[INFO] You can manage the task in Task Scheduler (taskschd.msc)"
