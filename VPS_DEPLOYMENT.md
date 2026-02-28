# Windows VPS Deployment Guide - Real MT5 Auto Trading

## Prerequisites

- Windows VPS (minimum 2GB RAM, 2 vCPU)
- MT5 account with broker
- SSH/RDP access to VPS
- This trading bot repository

## Step 1: VPS Setup

### Recommended VPS Providers
- **Forex VPS**: ForexVPS.net, FXVM, Vultr (Windows)
- **Specs**: Windows Server 2019/2022, 2GB RAM, 50GB SSD
- **Location**: Near your broker's server (lower latency)

### Initial VPS Configuration

1. Connect via RDP:
   ```
   Remote Desktop → your-vps-ip:3389
   Username: Administrator
   Password: [your VPS password]
   ```

2. Disable Windows Firewall (or allow Python):
   ```
   Control Panel → Windows Defender Firewall → Turn off
   ```

3. Set timezone to broker's timezone

## Step 2: Install Python on VPS

Download and install Python 3.9+:

1. Open PowerShell as Administrator
2. Download Python:
   ```powershell
   Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe" -OutFile "C:\python-installer.exe"
   ```

3. Install silently:
   ```powershell
   C:\python-installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
   ```

4. Verify installation:
   ```powershell
   python --version
   pip --version
   ```

## Step 3: Install MetaTrader 5 on VPS

1. Download MT5 from your broker's website
2. Install MT5 (default location: `C:\Program Files\MetaTrader 5\`)
3. Login to your account
4. Tools → Options → Expert Advisors:
   - ✓ Allow automated trading
   - ✓ Allow DLL imports
   - ✓ Allow WebRequest to listed URLs

## Step 4: Transfer Trading Bot to VPS

### Option A: Using Git (Recommended)

On VPS PowerShell:
```powershell
cd C:\
git clone https://github.com/yourusername/FOREX-TRADING-BOT.git
cd FOREX-TRADING-BOT
```

### Option B: Using RDP File Transfer

1. Connect to VPS via RDP
2. Enable clipboard sharing in RDP settings
3. Copy files from Mac to VPS desktop
4. Move to `C:\FOREX-TRADING-BOT`

### Option C: Using SCP from Mac

On your Mac terminal:
```bash
# Zip the project
cd /Users/igwemoh/Downloads
tar -czf forex-bot.tar.gz FOREX-TRADING-BOT-main/

# Transfer to VPS (requires OpenSSH on VPS)
scp forex-bot.tar.gz Administrator@YOUR_VPS_IP:C:/

# On VPS PowerShell, extract:
# tar -xzf C:\forex-bot.tar.gz
```

## Step 5: Setup Python Environment on VPS

On VPS PowerShell:

```powershell
cd C:\FOREX-TRADING-BOT-main

# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install pandas numpy scikit-learn lightgbm joblib MetaTrader5

# Verify MT5 installation
python -c "import MetaTrader5 as mt5; print('MT5 version:', mt5.__version__)"
```

## Step 6: Transfer Trained Models to VPS

From your Mac:

```bash
# Create models archive
cd /Users/igwemoh/Downloads/FOREX-TRADING-BOT-main
tar -czf models.tar.gz ALL_MODELS/

# Transfer to VPS
scp models.tar.gz Administrator@YOUR_VPS_IP:C:/FOREX-TRADING-BOT-main/
```

On VPS:
```powershell
cd C:\FOREX-TRADING-BOT-main
tar -xzf models.tar.gz
```

## Step 7: Configure Auto Trader on VPS

On VPS:

```powershell
cd C:\FOREX-TRADING-BOT-main\PY_FILES
python CONFIG_MANAGER.py create
```

Enter your configuration:
- MT5 credentials
- Trading parameters
- Risk settings

## Step 8: Test Connection

```powershell
python -c "
import MetaTrader5 as mt5
mt5.initialize()
print('MT5 initialized:', mt5.terminal_info())
print('Account info:', mt5.account_info())
mt5.shutdown()
"
```

## Step 9: Run Auto Trader on VPS

### Test Run (foreground):
```powershell
cd C:\FOREX-TRADING-BOT-main\PY_FILES
python AUTO_TRADER.py
```

Monitor logs:
```powershell
Get-Content ..\CSV_FILES\trading_log.txt -Wait -Tail 50
```

### Production Run (background as Windows Service):

Create `run_trader.bat`:
```batch
@echo off
cd C:\FOREX-TRADING-BOT-main\PY_FILES
C:\FOREX-TRADING-BOT-main\venv\Scripts\python.exe AUTO_TRADER.py
```

**Option A: Run on Startup (Task Scheduler)**

1. Open Task Scheduler
2. Create Basic Task:
   - Name: "Forex Auto Trader"
   - Trigger: At startup
   - Action: Start a program
   - Program: `C:\FOREX-TRADING-BOT-main\run_trader.bat`
   - Start in: `C:\FOREX-TRADING-BOT-main\PY_FILES`

**Option B: Run as Windows Service (NSSM)**

```powershell
# Download NSSM
Invoke-WebRequest -Uri "https://nssm.cc/release/nssm-2.24.zip" -OutFile "C:\nssm.zip"
Expand-Archive C:\nssm.zip C:\nssm

# Install service
C:\nssm\nssm-2.24\win64\nssm.exe install ForexTrader "C:\FOREX-TRADING-BOT-main\venv\Scripts\python.exe" "C:\FOREX-TRADING-BOT-main\PY_FILES\AUTO_TRADER.py"

# Set working directory
C:\nssm\nssm-2.24\win64\nssm.exe set ForexTrader AppDirectory "C:\FOREX-TRADING-BOT-main\PY_FILES"

# Start service
C:\nssm\nssm-2.24\win64\nssm.exe start ForexTrader

# Check status
C:\nssm\nssm-2.24\win64\nssm.exe status ForexTrader
```

## Step 10: Monitor from Mac

### SSH into VPS and check logs:

```bash
ssh Administrator@YOUR_VPS_IP
```

On VPS:
```powershell
cd C:\FOREX-TRADING-BOT-main\CSV_FILES
Get-Content trading_log.txt -Wait -Tail 50
```

### Create Monitoring Script on Mac:

```bash
# monitor_vps.sh
#!/bin/bash
ssh Administrator@YOUR_VPS_IP "powershell -Command 'Get-Content C:\FOREX-TRADING-BOT-main\CSV_FILES\trading_log.txt -Tail 100'"
```

## Troubleshooting

### MT5 Connection Failed
1. Ensure MT5 terminal is running on VPS
2. Check MT5 login credentials
3. Verify "Allow automated trading" is enabled
4. Check firewall isn't blocking MT5

### Python Script Crashes
1. Check logs: `C:\FOREX-TRADING-BOT-main\CSV_FILES\trading_log.txt`
2. Verify all models are in `ALL_MODELS/` directory
3. Check virtual environment is activated
4. Ensure dependencies are installed

### High Latency
- Move VPS closer to broker's server location
- Use broker's recommended VPS provider

### Service Won't Start
1. Check Task Scheduler logs
2. Verify paths in batch file
3. Run manually first to catch errors
4. Check Windows Event Viewer

## VPS Maintenance

### Daily Checks
```powershell
# Check if service is running
Get-Service ForexTrader

# View recent logs
Get-Content C:\FOREX-TRADING-BOT-main\CSV_FILES\trading_log.txt -Tail 100

# Check account balance
python -c "import MetaTrader5 as mt5; mt5.initialize(); print(mt5.account_info().balance); mt5.shutdown()"
```

### Weekly Tasks
- Review trading performance
- Update models if needed
- Check VPS disk space
- Verify backup strategy

### Monthly Tasks
- Update Python packages
- Review and optimize parameters
- Analyze trading logs
- Check VPS performance metrics

## Backup Strategy

### Automated Backup Script (PowerShell)

Save as `backup_config.ps1`:
```powershell
$BackupPath = "C:\Backups\ForexBot"
$SourcePath = "C:\FOREX-TRADING-BOT-main"
$Date = Get-Date -Format "yyyy-MM-dd"

# Create backup directory
New-Item -ItemType Directory -Force -Path "$BackupPath\$Date"

# Backup configuration
Copy-Item "$SourcePath\config.json" "$BackupPath\$Date\"

# Backup logs
Copy-Item "$SourcePath\CSV_FILES\*.txt" "$BackupPath\$Date\"
Copy-Item "$SourcePath\CSV_FILES\*.csv" "$BackupPath\$Date\"

# Backup models (if retrained)
Copy-Item -Recurse "$SourcePath\ALL_MODELS" "$BackupPath\$Date\"

Write-Host "Backup completed: $BackupPath\$Date"
```

Schedule daily via Task Scheduler.

## Security Best Practices

1. **Change default RDP port** (from 3389 to custom)
2. **Enable Windows Defender**
3. **Use strong passwords**
4. **Restrict RDP access** to your IP only
5. **Keep config.json secure** (contains credentials)
6. **Regular Windows Updates**
7. **Monitor unauthorized access attempts**
8. **Use VPN** for RDP connection

## Performance Optimization

### VPS Settings
- Disable unnecessary Windows services
- Disable Windows visual effects
- Keep only essential programs running
- Schedule Windows Updates during non-trading hours

### Python Optimization
- Use compiled Python (.pyc) files
- Enable Python optimizations: `python -OO AUTO_TRADER.py`
- Limit logging verbosity in production

### MT5 Optimization
- Close unnecessary charts
- Disable visual notifications
- Limit history data loaded
- Use SSD storage for faster data access

## Cost Estimation

| Service | Monthly Cost |
|---------|-------------|
| Windows VPS (2GB RAM) | $15-30 |
| Domain (optional) | $10-15 |
| Backup storage (optional) | $5-10 |
| **Total** | **$15-55/month** |

## Quick Reference Commands

### Start Trading Bot
```powershell
cd C:\FOREX-TRADING-BOT-main\PY_FILES
python AUTO_TRADER.py
```

### Stop Trading Bot
```powershell
# Press Ctrl+C in terminal
# Or kill process:
Stop-Process -Name python -Force
```

### Check Status
```powershell
Get-Process python
Get-Content ..\CSV_FILES\trading_log.txt -Tail 20
```

### Restart Service
```powershell
C:\nssm\nssm-2.24\win64\nssm.exe restart ForexTrader
```

### Update Code
```powershell
cd C:\FOREX-TRADING-BOT-main
git pull origin main
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt --upgrade
```

## Emergency Procedures

### Close All Positions Immediately
```powershell
python -c "
import MetaTrader5 as mt5
mt5.initialize()
positions = mt5.positions_get()
for pos in positions:
    mt5.Close(pos.ticket)
mt5.shutdown()
"
```

### Disable Auto Trading
1. Open MT5 terminal
2. Tools → Options → Expert Advisors
3. Uncheck "Allow automated trading"
4. Restart MT5

### Emergency Contact Script
Save your broker's support number and have manual override ready.
