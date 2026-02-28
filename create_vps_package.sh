#!/bin/bash
# Create complete VPS package from Mac

set -e

echo "=================================="
echo "  Creating VPS Package"
echo "=================================="

# Get project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Package name
PACKAGE_NAME="forex-bot-complete.tar.gz"

echo ""
echo "[1/3] Preparing files..."

# Check required directories
if [ ! -d "PY_FILES" ]; then
    echo "Error: PY_FILES directory not found!"
    exit 1
fi

if [ ! -d "ALL_MODELS" ]; then
    echo "Warning: ALL_MODELS directory not found!"
    echo "Models will need to be transferred separately."
fi

echo "✓ Project structure verified"

# Create package
echo ""
echo "[2/3] Creating compressed package..."
echo "This may take a minute..."

tar -czf "$PACKAGE_NAME" \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='*.log' \
  --exclude='catboost_info' \
  --exclude='deploy_to_vps.sh' \
  --exclude='monitor_vps.sh' \
  --exclude='create_vps_package.sh' \
  PY_FILES/ \
  ALL_MODELS/ \
  CSV_FILES/ \
  requirements.txt \
  SETUP_VPS.ps1 \
  VPS_DEPLOYMENT.md \
  SETUP_AUTO_TRADING.md \
  PACKAGE_FOR_VPS.md \
  README.md

PACKAGE_SIZE=$(du -h "$PACKAGE_NAME" | cut -f1)
echo "✓ Package created: $PACKAGE_NAME ($PACKAGE_SIZE)"

# Instructions
echo ""
echo "[3/3] Package ready!"
echo ""
echo "=================================="
echo "  Next Steps"
echo "=================================="
echo ""
echo "Option 1: Transfer via SCP (if VPS has SSH)"
echo "  scp $PACKAGE_NAME Administrator@YOUR_VPS_IP:C:/"
echo ""
echo "Option 2: Transfer via RDP"
echo "  1. Connect to VPS with Remote Desktop"
echo "  2. Enable local drive sharing in RDP settings"
echo "  3. Copy $PACKAGE_NAME to C:\\ on VPS"
echo ""
echo "Option 3: Use cloud storage"
echo "  1. Upload to Dropbox/Google Drive/OneDrive"
echo "  2. Download on VPS from browser"
echo ""
echo "On VPS (PowerShell):"
echo "  cd C:\\"
echo "  tar -xzf $PACKAGE_NAME"
echo "  cd FOREX-TRADING-BOT-main"
echo "  powershell -ExecutionPolicy Bypass -File .\\SETUP_VPS.ps1"
echo ""
echo "Package location: $SCRIPT_DIR/$PACKAGE_NAME"
echo ""
