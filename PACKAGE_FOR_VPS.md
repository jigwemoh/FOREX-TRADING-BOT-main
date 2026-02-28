# Complete VPS Package - Standalone Setup

This guide shows how to package the entire trading bot for easy deployment to your Windows VPS **without needing your Mac**.

## Method 1: Transfer Complete Package (Recommended)

### On Your Mac (One-Time Packaging)

```bash
cd /Users/igwemoh/Downloads/FOREX-TRADING-BOT-main

# Create complete package
tar -czf forex-bot-complete.tar.gz \
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
  SETUP_VPS.ps1 \
  VPS_DEPLOYMENT.md \
  SETUP_AUTO_TRADING.md \
  README.md

# Transfer to VPS
scp forex-bot-complete.tar.gz Administrator@YOUR_VPS_IP:C:/
```

### On Your Windows VPS

**Option A: Using PowerShell (Recommended)**
```powershell
# Extract package
cd C:\
tar -xzf forex-bot-complete.tar.gz

# Use extracted project folder
cd FOREX-TRADING-BOT-main

# Run automated setup
powershell -ExecutionPolicy Bypass -File .\SETUP_VPS.ps1
```

**Option B: Using WinRAR/7-Zip GUI**
1. Download and install 7-Zip from: https://www.7-zip.org/
2. Right-click `forex-bot-complete.tar.gz` → 7-Zip → Extract Here
3. Open PowerShell in the extracted folder
4. Run: `powershell -ExecutionPolicy Bypass -File .\SETUP_VPS.ps1`

---

## Method 2: Manual File Transfer (No Terminal)

### Step 1: Copy Files via RDP

1. **Connect to VPS via Remote Desktop**
   - Windows: Press `Win+R`, type `mstsc`, enter VPS IP
   - Mac: Use Microsoft Remote Desktop app

2. **Enable Local Drive Sharing**
   - In RDP connection settings
   - Check "Local Resources" → "More" → Select your Mac drive
   - Connect to VPS

3. **Copy Files**
   - On VPS, open File Explorer
   - You'll see your Mac drive under "This PC"
   - Copy these folders/files to `C:\FOREX-TRADING-BOT-main\`:
     ```
     PY_FILES/
     ALL_MODELS/
     CSV_FILES/
     requirements.txt
     SETUP_VPS.ps1
     VPS_DEPLOYMENT.md
     SETUP_AUTO_TRADING.md
     README.md
     ```

### Step 2: Run Setup on VPS

```powershell
cd C:\FOREX-TRADING-BOT-main
powershell -ExecutionPolicy Bypass -File .\SETUP_VPS.ps1
```

---

## Method 3: GitHub Repository (Best for Updates)

### On Your Mac (One-Time Setup)

```bash
cd /Users/igwemoh/Downloads/FOREX-TRADING-BOT-main

# Initialize git (if not already)
git init
git add .
git commit -m "Initial forex bot setup"

# Push to private GitHub repo
git remote add origin https://github.com/YOUR_USERNAME/forex-trading-bot.git
git push -u origin main
```

### On Your Windows VPS

```powershell
# Install Git for Windows (if not installed)
# Download from: https://git-scm.com/download/win

# Clone repository
cd C:\
git clone https://github.com/YOUR_USERNAME/forex-trading-bot.git FOREX-TRADING-BOT-main
cd FOREX-TRADING-BOT-main

# Run setup
powershell -ExecutionPolicy Bypass -File .\SETUP_VPS.ps1
```

**Benefits:**
- Easy updates: `git pull` on VPS
- Version control
- Backup in cloud
- Share across multiple VPS instances

---

## What SETUP_VPS.ps1 Does (Fully Automated)

The PowerShell setup script handles everything:

✅ **[1/8]** Checks Python 3.9+ installation  
✅ **[2/8]** Verifies project structure  
✅ **[3/8]** Creates Python virtual environment  
✅ **[4/8]** Activates venv and upgrades pip  
✅ **[5/8]** Installs all dependencies from requirements.txt  
✅ **[6/8]** Verifies MetaTrader5 package  
✅ **[7/8]** Checks for ML models  
✅ **[8/8]** Optionally runs configuration wizard  

---

## After Setup: Running the Bot

### 1. Configure Trading Settings

```powershell
cd C:\FOREX-TRADING-BOT-main\PY_FILES
python CONFIG_MANAGER.py create
```

You'll be prompted for:
- MT5 login credentials (account number, password, server)
- Trading symbol (default: EURUSD)
- Risk percentage per trade (default: 2%)
- Stop loss/take profit in pips
- Trading hours

### 2. Test the Connection

```powershell
# Quick test
python -c "import MetaTrader5 as mt5; print('MT5 Available' if mt5.initialize() else 'MT5 Not Working')"
```

### 3. Run the Bot (Foreground)

```powershell
python AUTO_TRADER.py
```

Press `Ctrl+C` to stop.

### 4. Run 24/7 (Background)

**Simple Background Process:**
```powershell
Start-Process python -ArgumentList "AUTO_TRADER.py" -WindowStyle Hidden -WorkingDirectory "C:\FOREX-TRADING-BOT-main\PY_FILES"
```

**As Windows Service (Production):**
See `VPS_DEPLOYMENT.md` → Section "Windows Service Setup (NSSM)"

---

## Monitoring the Bot

### Check if Bot is Running

```powershell
# Find Python process
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,CPU,WS

# Check logs
Get-Content C:\FOREX-TRADING-BOT-main\PY_FILES\trading_log.txt -Tail 20 -Wait
```

### Monitor from Your Mac (SSH)

```bash
# View live logs
ssh Administrator@YOUR_VPS_IP "type C:\FOREX-TRADING-BOT-main\PY_FILES\trading_log.txt"

# Or use the monitoring script
./monitor_vps.sh YOUR_VPS_IP
```

---

## Updating the Bot

### If Using Git

```powershell
cd C:\FOREX-TRADING-BOT-main
git pull
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt --upgrade
```

> Important: If you update with `git`, do **not** use GitHub "Download ZIP" for updates. ZIP downloads create separate extracted folders and are not a `git` working copy.

### If Using Manual Transfer

1. Create new package on Mac with updated files
2. Transfer to VPS
3. Replace old files
4. Restart the bot

---

## File Sizes (Approximate)

| Component | Size | Required? |
|-----------|------|-----------|
| PY_FILES/ | 500 KB | ✅ Yes |
| ALL_MODELS/ | 50-100 MB | ✅ Yes (for ML predictions) |
| CSV_FILES/ | 100-200 MB | ⚠️ Optional (for backtesting) |
| requirements.txt | 1 KB | ✅ Yes |
| Scripts | 50 KB | ✅ Yes |

**Total Package Size:** ~150-300 MB

---

## Troubleshooting

### "Python not found"

**Install Python on VPS:**
```powershell
# Download Python 3.11 (recommended)
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe" -OutFile "$env:TEMP\python-installer.exe"

# Install (add to PATH)
& "$env:TEMP\python-installer.exe" /quiet InstallAllUsers=1 PrependPath=1

# Verify
python --version
```

### "MetaTrader5 not working"

1. **Install MT5 Terminal:** Download from your broker's website
2. **Login manually once:** Open MT5, login with credentials
3. **Keep MT5 open:** The terminal must be running for API to work

### "Models not found"

**Transfer models from Mac:**
```bash
# On Mac
cd /Users/igwemoh/Downloads/FOREX-TRADING-BOT-main
tar -czf models-only.tar.gz ALL_MODELS/

# Transfer
scp models-only.tar.gz Administrator@YOUR_VPS_IP:C:/FOREX-TRADING-BOT-main/

# On VPS
cd C:\FOREX-TRADING-BOT-main
tar -xzf models-only.tar.gz
```

### "Execution Policy Error"

```powershell
# Allow script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or run with bypass
powershell -ExecutionPolicy Bypass -File .\SETUP_VPS.ps1
```

---

## Security Best Practices

1. **Use Private GitHub Repo** - Never use public repos for trading bots
2. **Encrypt Config** - `trading_config.json` contains your MT5 password
3. **Enable Windows Firewall** - Only allow necessary ports
4. **Use VPN** - Some brokers require specific IP regions
5. **Regular Backups** - Backup `trading_config.json` and logs

---

## Complete Checklist

Before starting on VPS:

- [ ] Python 3.9+ installed
- [ ] MetaTrader5 terminal installed and logged in
- [ ] All files transferred (PY_FILES, ALL_MODELS, requirements.txt, SETUP_VPS.ps1)
- [ ] Run `SETUP_VPS.ps1` successfully
- [ ] Configuration created (`CONFIG_MANAGER.py create`)
- [ ] MT5 connection tested
- [ ] Demo account tested first
- [ ] Understand emergency stop procedure

Emergency Stop:
```powershell
# Stop all Python processes
Stop-Process -Name python -Force

# Or close specific bot process
Get-Process python | Where-Object {$_.Path -like "*FOREX-TRADING-BOT-main*"} | Stop-Process
```

---

## Getting Help

1. Check `VPS_DEPLOYMENT.md` for detailed Windows setup
2. Check `SETUP_AUTO_TRADING.md` for bot configuration
3. Review `trading_log.txt` for error messages
4. Test on demo account first before live trading

**Always start with demo account trading to verify everything works correctly!**
