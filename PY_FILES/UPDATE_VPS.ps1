#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Self-updating script for VPS (run directly on Windows VPS)
.DESCRIPTION
    Pull latest changes from Git or download from cloud storage
.NOTES
    Place this in PY_FILES/ and run on VPS
#>

# Colors for output
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

Write-Host "`n========================================" -ForegroundColor Magenta
Write-Host "  FOREX BOT - VPS AUTO UPDATE" -ForegroundColor Magenta
Write-Host "========================================`n" -ForegroundColor Magenta

# Change to project root
Set-Location ..

# Check if bot is running
Write-Info "[1/6] Checking if bot is running..."
$botProcess = Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.Path -like "*FOREX-TRADING-BOT-main*"}
if ($botProcess) {
    Write-Warning "Bot is currently running (PID: $($botProcess.Id))"
    Write-Host "Stop the bot? (y/n): " -NoNewline -ForegroundColor Yellow
    $response = Read-Host
    if ($response -eq 'y' -or $response -eq 'Y') {
        Stop-Process -Id $botProcess.Id -Force
        Write-Success "✓ Bot stopped"
    } else {
        Write-Error "Cannot update while bot is running. Exiting."
        exit 1
    }
} else {
    Write-Success "✓ No running bot detected"
}

# Backup current version
Write-Info "`n[2/6] Creating backup..."
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$backupDir = "C:\FOREX-BOT-BACKUPS"
if (!(Test-Path $backupDir)) { New-Item -ItemType Directory -Path $backupDir | Out-Null }

$backupPath = "$backupDir\PY_FILES_$timestamp"
Copy-Item "PY_FILES" -Destination $backupPath -Recurse -Force
Write-Success "✓ Backup created: $backupPath"

# Update method selection
Write-Info "`n[3/6] Select update method:"
Write-Host "  1. Git pull (if using Git)"
Write-Host "  2. Download from URL (Dropbox/Google Drive link)"
Write-Host "  3. Manual (files already in C:\FOREX-BOT-UPDATE)"
Write-Host "`nChoice (1/2/3): " -NoNewline -ForegroundColor Yellow
$updateMethod = Read-Host

switch ($updateMethod) {
    "1" {
        # Git pull
        Write-Info "`n[4/6] Pulling latest changes from Git..."
        if (!(Test-Path ".git")) {
            Write-Error "Not a git repository. Use method 2 or 3."
            exit 1
        }
        
        git pull
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Git pull failed!"
            exit 1
        }
        Write-Success "✓ Git pull complete"
    }
    
    "2" {
        # Download from URL
        Write-Info "`n[4/6] Downloading update package..."
        Write-Host "Enter download URL: " -NoNewline
        $downloadUrl = Read-Host
        
        $updatePackage = "$env:TEMP\forex-bot-update.tar.gz"
        try {
            Invoke-WebRequest -Uri $downloadUrl -OutFile $updatePackage
            Write-Success "✓ Download complete"
            
            Write-Info "Extracting update..."
            tar -xzf $updatePackage -C .
            Remove-Item $updatePackage
            Write-Success "✓ Update extracted"
        } catch {
            Write-Error "Download failed: $_"
            exit 1
        }
    }
    
    "3" {
        # Manual update
        Write-Info "`n[4/6] Applying manual update..."
        $updateDir = "C:\FOREX-BOT-UPDATE"
        if (!(Test-Path $updateDir)) {
            Write-Error "Update directory not found: $updateDir"
            Write-Info "Place updated files in $updateDir and try again"
            exit 1
        }
        
        Copy-Item "$updateDir\*" -Destination . -Recurse -Force
        Write-Success "✓ Manual update applied"
    }
    
    default {
        Write-Error "Invalid choice"
        exit 1
    }
}

# Update dependencies
Write-Info "`n[5/6] Updating Python dependencies..."
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt --upgrade --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Some dependencies failed to update, but continuing..."
}
Write-Success "✓ Dependencies updated"

# Verify installation
Write-Info "`n[6/6] Verifying installation..."
$testImports = @"
import sys
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
import lightgbm
print('OK')
"@

$result = & python -c $testImports 2>&1
if ($result -match "OK") {
    Write-Success "✓ Installation verified"
} else {
    Write-Warning "Verification failed, but update is complete"
    Write-Info "Error: $result"
}

# Summary
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  UPDATE COMPLETE!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green

Write-Info "What's next:"
Write-Host "1. " -NoNewline; Write-Success "Review changes: " -NoNewline; Write-Host "check PY_FILES/"
Write-Host "2. " -NoNewline; Write-Success "Update config (if needed): " -NoNewline; Write-Host "python PY_FILES\CONFIG_MANAGER.py"
Write-Host "3. " -NoNewline; Write-Success "Restart bot: " -NoNewline; Write-Host "python PY_FILES\AUTO_TRADER.py"

Write-Host "`nBackup location: " -ForegroundColor Cyan -NoNewline
Write-Host $backupPath

Write-Host "`nTo rollback:" -ForegroundColor Yellow
Write-Host "  Remove-Item PY_FILES -Recurse -Force" -ForegroundColor Gray
Write-Host "  Copy-Item $backupPath -Destination PY_FILES -Recurse" -ForegroundColor Gray

Write-Host "`n"
