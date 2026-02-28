#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Complete automated setup for Forex Trading Bot on Windows VPS
.DESCRIPTION
    This script automates the entire setup process:
    - Checks Python installation
    - Creates virtual environment
    - Installs all dependencies
    - Verifies MetaTrader5
    - Creates configuration
    - Tests the system
.NOTES
    Run this script AFTER transferring files to VPS
#>

# Colors for output
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

# Banner
Write-Host "`n========================================" -ForegroundColor Magenta
Write-Host "  FOREX TRADING BOT - VPS SETUP" -ForegroundColor Magenta
Write-Host "========================================`n" -ForegroundColor Magenta

# Step 1: Check Python installation
Write-Info "[1/8] Checking Python installation..."
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "Python not found! Please install Python 3.9+ first."
    Write-Info "Download from: https://www.python.org/downloads/"
    exit 1
}

$pythonVersion = & python --version
Write-Success "✓ Found: $pythonVersion"

# Verify Python version
$versionMatch = [regex]::Match($pythonVersion, '(\d+)\.(\d+)')
$major = [int]$versionMatch.Groups[1].Value
$minor = [int]$versionMatch.Groups[2].Value
if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 9)) {
    Write-Error "Python 3.9+ required. You have Python $major.$minor"
    exit 1
}

# Step 2: Check if we're in the right directory
Write-Info "`n[2/8] Verifying project structure..."
if (-not (Test-Path "PY_FILES")) {
    Write-Error "PY_FILES directory not found!"
    Write-Info "Please run this script from the FOREX-TRADING-BOT-main directory."
    exit 1
}
if (-not (Test-Path "requirements.txt")) {
    Write-Error "requirements.txt not found!"
    exit 1
}
Write-Success "✓ Project structure verified"

# Step 3: Create virtual environment
Write-Info "`n[3/8] Creating Python virtual environment..."
if (Test-Path ".venv") {
    Write-Warning "Virtual environment already exists. Skipping creation."
} else {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment!"
        exit 1
    }
    Write-Success "✓ Virtual environment created"
}

# Step 4: Activate virtual environment and upgrade pip
Write-Info "`n[4/8] Activating virtual environment..."
$activateScript = ".\.venv\Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Error "Activation script not found at: $activateScript"
    exit 1
}
& $activateScript
Write-Success "✓ Virtual environment activated"

Write-Info "Upgrading pip..."
& python -m pip install --upgrade pip --quiet
Write-Success "✓ Pip upgraded"

# Step 5: Install dependencies
Write-Info "`n[5/8] Installing Python dependencies..."
Write-Info "This may take several minutes..."
& pip install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install dependencies!"
    exit 1
}
Write-Success "✓ Dependencies installed"

# Step 6: Verify MetaTrader5 installation
Write-Info "`n[6/8] Verifying MetaTrader5 package..."
$mt5Test = & python -c "import MetaTrader5 as mt5; print('OK')" 2>&1
if ($mt5Test -match "OK") {
    Write-Success "✓ MetaTrader5 package working"
} else {
    Write-Error "MetaTrader5 package not working properly!"
    Write-Info "Error: $mt5Test"
    Write-Warning "Note: You still need to install MT5 terminal from your broker."
    exit 1
}

# Step 7: Verify ML models exist
Write-Info "`n[7/8] Checking ML models..."
if (-not (Test-Path "ALL_MODELS")) {
    Write-Warning "ALL_MODELS directory not found!"
    Write-Info "You need to transfer trained models from your Mac."
} else {
    $modelCount = (Get-ChildItem -Path "ALL_MODELS" -Recurse -Filter "*.joblib" | Measure-Object).Count
    if ($modelCount -gt 0) {
        Write-Success "✓ Found $modelCount model files"
    } else {
        Write-Warning "No .joblib model files found in ALL_MODELS/"
        Write-Info "Transfer models from Mac: ALL_MODELS/*/*.joblib"
    }
}

# Step 8: Create configuration
Write-Info "`n[8/8] Setting up trading configuration..."
Write-Host "`nDo you want to create trading configuration now? (y/n): " -NoNewline -ForegroundColor Yellow
$response = Read-Host
if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Info "`nStarting configuration wizard..."
    Set-Location PY_FILES
    & python CONFIG_MANAGER.py create
    Set-Location ..
} else {
    Write-Warning "Skipped configuration. Run later with: python PY_FILES\CONFIG_MANAGER.py create"
}

# Summary
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  SETUP COMPLETE!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green

Write-Info "Next Steps:"
Write-Host "1. " -NoNewline; Write-Success "Install MetaTrader5 terminal from your broker (if not done)"
Write-Host "2. " -NoNewline; Write-Success "Transfer models: " -NoNewline; Write-Host "Copy ALL_MODELS folder from Mac"
Write-Host "3. " -NoNewline; Write-Success "Configure bot: " -NoNewline; Write-Host "python PY_FILES\CONFIG_MANAGER.py create"
Write-Host "4. " -NoNewline; Write-Success "Test connection: " -NoNewline; Write-Host "python PY_FILES\AUTO_TRADER.py (test mode)"
Write-Host "5. " -NoNewline; Write-Success "Start trading: " -NoNewline; Write-Host "python PY_FILES\AUTO_TRADER.py"

Write-Host "`nTo run bot in background:" -ForegroundColor Cyan
Write-Host "  Start-Process python -ArgumentList 'PY_FILES\AUTO_TRADER.py' -WindowStyle Hidden" -ForegroundColor Gray

Write-Host "`nTo setup as Windows Service:" -ForegroundColor Cyan
Write-Host "  See VPS_DEPLOYMENT.md - Section: Windows Service Setup (NSSM)" -ForegroundColor Gray

Write-Host "`n"
