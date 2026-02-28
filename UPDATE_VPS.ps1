# UPDATE_VPS.ps1
# PowerShell script to update the FOREX-TRADING-BOT pipeline on your VPS
# Usage: Run this script from the project root directory

Write-Host "[INFO] Stopping any running Python processes..."
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host "[INFO] Pulling latest code from GitHub..."
if (Get-Command git -ErrorAction SilentlyContinue) {
    git pull origin main
} else {
    Write-Host "[ERROR] Git is not installed or not in PATH. Please install Git for Windows: https://git-scm.com/download/win"
    exit 1
}

Write-Host "[INFO] Updating Python dependencies..."
if (Test-Path ".venv") {
    Write-Host "[INFO] Activating virtual environment..."
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
} else {
    Write-Host "[INFO] No virtual environment found. Installing dependencies globally."
    python -m pip install --upgrade pip
    pip install -r requirements.txt
}

Write-Host "[INFO] Update complete. You can now restart your trading bot."
