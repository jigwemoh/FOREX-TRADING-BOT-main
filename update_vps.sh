#!/bin/bash
# Auto-update VPS deployment from Mac

set -e

if [ -z "$1" ]; then
    echo "Usage: ./update_vps.sh YOUR_VPS_IP"
    exit 1
fi

VPS_IP=$1
VPS_USER="Administrator"
VPS_PATH="C:/FOREX-TRADING-BOT-main"

echo "=================================="
echo "  VPS UPDATE - FOREX BOT"
echo "=================================="

# Step 1: Create update package (lightweight - code only)
echo ""
echo "[1/5] Creating update package..."
tar -czf forex-bot-update.tar.gz \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='*.log' \
  --exclude='ALL_MODELS' \
  --exclude='CSV_FILES' \
  --exclude='catboost_info' \
  --exclude='forex-bot-complete.tar.gz' \
  --exclude='forex-bot-update.tar.gz' \
  PY_FILES/ \
  requirements.txt \
  SETUP_VPS.ps1 \
  VPS_DEPLOYMENT.md \
  SETUP_AUTO_TRADING.md \
  PACKAGE_FOR_VPS.md \
  README.md

UPDATE_SIZE=$(du -h forex-bot-update.tar.gz | cut -f1)
echo "✓ Update package created: $UPDATE_SIZE"

# Step 2: Transfer to VPS
echo ""
echo "[2/5] Transferring to VPS..."
scp forex-bot-update.tar.gz ${VPS_USER}@${VPS_IP}:${VPS_PATH}/

echo "✓ Transfer complete"

# Step 3: Backup current version on VPS
echo ""
echo "[3/5] Backing up current version on VPS..."
ssh ${VPS_USER}@${VPS_IP} "powershell -Command \"
    \\\$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    \\\$backupDir = 'C:/FOREX-BOT-BACKUPS'
    if (!(Test-Path \\\$backupDir)) { New-Item -ItemType Directory -Path \\\$backupDir }
    Copy-Item '${VPS_PATH}/PY_FILES' -Destination \\\"\\\$backupDir/PY_FILES_\\\$timestamp\\\" -Recurse -Force
    Write-Host '✓ Backup created: PY_FILES_'\\\$timestamp
\""

# Step 4: Extract update on VPS
echo ""
echo "[4/5] Applying update on VPS..."
ssh ${VPS_USER}@${VPS_IP} "powershell -Command \"
    cd ${VPS_PATH}
    tar -xzf forex-bot-update.tar.gz
    Remove-Item forex-bot-update.tar.gz
    Write-Host '✓ Update extracted'
\""

# Step 5: Update dependencies on VPS
echo ""
echo "[5/5] Updating dependencies on VPS..."
ssh ${VPS_USER}@${VPS_IP} "powershell -Command \"
    cd ${VPS_PATH}
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt --upgrade --quiet
    Write-Host '✓ Dependencies updated'
\""

# Cleanup local update package
rm forex-bot-update.tar.gz

echo ""
echo "=================================="
echo "  UPDATE COMPLETE!"
echo "=================================="
echo ""
echo "Next steps on VPS:"
echo "  1. If config changed: python PY_FILES\CONFIG_MANAGER.py"
echo "  2. Restart bot: python PY_FILES\AUTO_TRADER.py"
echo ""
echo "Or restart Windows Service:"
echo "  Restart-Service ForexTradingBot"
echo ""
