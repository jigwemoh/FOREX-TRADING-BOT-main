#!/bin/bash
# Deploy Forex Trading Bot to Windows VPS
# Run from Mac: ./deploy_to_vps.sh YOUR_VPS_IP

set -e

# Configuration
VPS_IP="${1}"
VPS_USER="Administrator"
VPS_PATH="C:/FOREX-TRADING-BOT-main"
LOCAL_PATH="/Users/igwemoh/Downloads/FOREX-TRADING-BOT-main"

if [ -z "$VPS_IP" ]; then
    echo "Usage: ./deploy_to_vps.sh YOUR_VPS_IP"
    echo "Example: ./deploy_to_vps.sh 192.168.1.100"
    exit 1
fi

echo "=========================================="
echo "Deploying Forex Trading Bot to VPS"
echo "VPS IP: $VPS_IP"
echo "=========================================="

# Step 1: Create deployment package
echo ""
echo "[1/5] Creating deployment package..."
cd "$LOCAL_PATH"

# Exclude unnecessary files
tar -czf forex-bot-deploy.tar.gz \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='*.log' \
    PY_FILES/ \
    ALL_MODELS/ \
    CSV_FILES/ \
    requirements.txt \
    SETUP_AUTO_TRADING.md \
    VPS_DEPLOYMENT.md

echo "✓ Package created: forex-bot-deploy.tar.gz"

# Step 2: Transfer to VPS
echo ""
echo "[2/5] Transferring files to VPS..."
scp forex-bot-deploy.tar.gz "${VPS_USER}@${VPS_IP}:C:/"

echo "✓ Files transferred"

# Step 3: Extract on VPS
echo ""
echo "[3/5] Extracting files on VPS..."
ssh "${VPS_USER}@${VPS_IP}" "powershell -Command '
    if (Test-Path C:\FOREX-TRADING-BOT-main) {
        Remove-Item -Recurse -Force C:\FOREX-TRADING-BOT-main
    }
    New-Item -ItemType Directory -Path C:\FOREX-TRADING-BOT-main
    tar -xzf C:\forex-bot-deploy.tar.gz -C C:\FOREX-TRADING-BOT-main
    Write-Host \"Files extracted to C:\FOREX-TRADING-BOT-main\"
'"

echo "✓ Files extracted"

# Step 4: Setup Python environment
echo ""
echo "[4/5] Setting up Python environment on VPS..."
ssh "${VPS_USER}@${VPS_IP}" "powershell -Command '
    cd C:\FOREX-TRADING-BOT-main
    
    # Create virtual environment
    python -m venv venv
    
    # Activate and install dependencies
    .\venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install pandas numpy scikit-learn lightgbm joblib MetaTrader5
    
    Write-Host \"Python environment ready\"
'"

echo "✓ Python environment configured"

# Step 5: Verify installation
echo ""
echo "[5/5] Verifying installation..."
ssh "${VPS_USER}@${VPS_IP}" "powershell -Command '
    cd C:\FOREX-TRADING-BOT-main
    .\venv\Scripts\Activate.ps1
    
    # Test MT5 import
    python -c \"import MetaTrader5 as mt5; print('✓ MetaTrader5 imported successfully')\"
    
    # Check directory structure
    Write-Host \"\"
    Write-Host \"Directory structure:\"
    Get-ChildItem -Recurse -Directory | Select-Object FullName
'"

echo ""
echo "=========================================="
echo "✓ Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps on VPS:"
echo "1. RDP to $VPS_IP"
echo "2. Open PowerShell"
echo "3. cd C:\FOREX-TRADING-BOT-main\PY_FILES"
echo "4. python CONFIG_MANAGER.py create"
echo "5. python AUTO_TRADER.py"
echo ""
echo "Monitor from Mac:"
echo "ssh ${VPS_USER}@${VPS_IP}"
echo ""

# Cleanup
rm forex-bot-deploy.tar.gz
