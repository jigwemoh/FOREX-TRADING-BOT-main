# Automatic Trading Setup Guide

## Prerequisites

1. **MetaTrader 5 Terminal** installed and running
2. **Live or Demo MT5 Account** with your broker
3. **Python Environment** with all dependencies installed
4. **Trained ML Models** in `ALL_MODELS/` directory

## Step-by-Step Setup

### 1. Install MetaTrader5 Python Package

```bash
pip install MetaTrader5
```

### 2. Create Configuration File

Run the configuration manager to set up your credentials:

```bash
cd PY_FILES
python CONFIG_MANAGER.py create
```

This will prompt you for:
- MT5 account number
- MT5 password (hidden input)
- MT5 server name
- Trading parameters (symbol, timeframe, risk %)
- Risk management settings
- Trading hours

Configuration is saved to `config.json` (keep this file secure!).

### 3. Test Connection

Before going live, test your MT5 connection:

```python
import MetaTrader5 as mt5

# Initialize
if not mt5.initialize():
    print("Failed to initialize MT5")
    exit()

# Login
login = 12345678  # Your account number
password = "your_password"
server = "Your-Broker-Server"

if mt5.login(login, password=password, server=server):
    print("✓ Connected successfully")
    print(mt5.account_info())
else:
    print("✗ Login failed:", mt5.last_error())

mt5.shutdown()
```

### 4. Verify ML Models

Ensure your models are trained and saved:

```bash
ls -la ALL_MODELS/EURUSD/
# Should show: T_5M.joblib, T_5M_scaler.joblib, etc.
```

If models are missing, train them first:

```bash
python SIMPLE_TRAIN.py
```

### 5. Start on Demo Account First

**CRITICAL: Always test on demo account first!**

Edit `AUTO_TRADER.py` to use demo credentials:

```python
# Demo account configuration
MT5_LOGIN = 12345678  # Demo account number
MT5_PASSWORD = "demo_password"
MT5_SERVER = "Demo-Server"

# Conservative settings for testing
RISK_PERCENT = 0.5  # Very small risk
MAX_POSITIONS = 1   # Only one position
CHECK_INTERVAL = 60  # Check every minute for testing
```

### 6. Run the Auto Trader

```bash
python AUTO_TRADER.py
```

Monitor the logs in real-time:
```bash
tail -f ../CSV_FILES/trading_log.txt
```

### 7. Monitor Performance

The bot will log:
- Connection status
- Market data retrieval
- ML predictions and confidence
- Trade execution
- Position management
- Risk checks

Example log output:
```
2026-02-26 10:00:00 - INFO - Connected to MT5: AccountInfo(...)
2026-02-26 10:00:01 - INFO - Loaded model: T_5M
2026-02-26 10:00:01 - INFO - Starting auto trader for EURUSD
2026-02-26 10:05:00 - INFO - ML Signal: 1 (conf: 0.68) | SMC Signal: 1
2026-02-26 10:05:01 - INFO - Order opened: 123456 | Type: BUY | Lots: 0.1 | Confidence: 0.68
```

### 8. Risk Management Features

The bot includes multiple safety features:

- **Daily Loss Limit**: Stops trading if daily loss exceeds threshold
- **Max Drawdown**: Stops if total drawdown exceeds limit
- **Position Limits**: Maximum number of concurrent positions
- **Spread Check**: Avoids trading during high spread conditions
- **Trading Hours**: Only trades during specified hours
- **Correlation Check**: Limits correlated positions
- **Trailing Stop**: Automatically moves stop loss to protect profits

### 9. Running 24/7

#### Option A: Keep Terminal Open (Simple)
Just leave the Python script running in a terminal window.

#### Option B: Background Process (Linux/Mac)
```bash
nohup python AUTO_TRADER.py > auto_trader.log 2>&1 &
```

Check if running:
```bash
ps aux | grep AUTO_TRADER
```

Stop the bot:
```bash
pkill -f AUTO_TRADER.py
```

#### Option C: System Service (Linux)

Create `/etc/systemd/system/forex-trader.service`:
```ini
[Unit]
Description=Forex Auto Trading Bot
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/FOREX-TRADING-BOT-main/PY_FILES
ExecStart=/usr/bin/python3 AUTO_TRADER.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable forex-trader
sudo systemctl start forex-trader
sudo systemctl status forex-trader
```

View logs:
```bash
sudo journalctl -u forex-trader -f
```

#### Option D: Task Scheduler (Windows)

1. Open Task Scheduler
2. Create Basic Task
3. Trigger: At startup
4. Action: Start a program
   - Program: `python.exe`
   - Arguments: `C:\path\to\AUTO_TRADER.py`
   - Start in: `C:\path\to\PY_FILES`

### 10. Monitoring Dashboard (Optional)

Create a simple monitoring script:

```python
# MONITOR_TRADER.py
import MetaTrader5 as mt5
import time
from datetime import datetime

mt5.initialize()
mt5.login(login, password, server)

while True:
    account = mt5.account_info()
    positions = mt5.positions_get()
    
    print(f"\n{'='*50}")
    print(f"Time: {datetime.now()}")
    print(f"Balance: ${account.balance:.2f}")
    print(f"Equity: ${account.equity:.2f}")
    print(f"Profit: ${account.profit:.2f}")
    print(f"Open Positions: {len(positions)}")
    
    if positions:
        for pos in positions:
            print(f"  {pos.symbol}: {pos.type_str} | Lots: {pos.volume} | P/L: ${pos.profit:.2f}")
    
    time.sleep(30)  # Update every 30 seconds
```

## Safety Checklist Before Going Live

- [ ] Tested on demo account for at least 1 week
- [ ] Verified ML models are performing well
- [ ] Risk parameters are conservative (1% or less)
- [ ] Daily loss limit is set
- [ ] Max positions is limited
- [ ] Spread check is enabled
- [ ] Trading hours are appropriate
- [ ] Monitoring system is in place
- [ ] You understand all bot functionality
- [ ] Emergency stop procedure is known

## Emergency Stop

To immediately stop trading:

1. **Press Ctrl+C** in the terminal running the bot
2. **Or close all positions manually** in MT5
3. **Or run**: `pkill -f AUTO_TRADER.py`

## Configuration Parameters Explained

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `risk_percent` | % of balance risked per trade | 0.5-2.0% |
| `max_positions` | Maximum concurrent trades | 1-3 |
| `stop_loss_pips` | Stop loss distance | 30-100 pips |
| `take_profit_pips` | Take profit distance | 2x stop loss |
| `trailing_stop_pips` | Trailing stop distance | 20-50 pips |
| `check_interval` | Seconds between checks | 300-3600 |
| `max_daily_loss_percent` | Daily loss limit | 3-5% |
| `max_daily_trades` | Max trades per day | 5-15 |

## Advanced Features

### Multi-Symbol Trading

Run multiple instances for different symbols:

```bash
# Terminal 1: EURUSD
python AUTO_TRADER.py --symbol EURUSD

# Terminal 2: GBPUSD
python AUTO_TRADER.py --symbol GBPUSD
```

### Telegram Notifications

Add to `AUTO_TRADER.py`:

```python
import requests

def send_telegram(message):
    bot_token = "YOUR_BOT_TOKEN"
    chat_id = "YOUR_CHAT_ID"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    requests.post(url, data={"chat_id": chat_id, "text": message})

# Call after each trade
send_telegram(f"Order opened: {result.order} | {symbol} {signal_type}")
```

### Email Alerts

```python
import smtplib
from email.mime.text import MIMEText

def send_email(subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = "your_email@gmail.com"
    msg['To'] = "your_email@gmail.com"
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login("your_email@gmail.com", "app_password")
        smtp.send_message(msg)
```

## Troubleshooting

### Connection Issues
- Ensure MT5 terminal is running
- Check internet connection
- Verify credentials are correct
- Confirm broker allows automated trading

### No Trades Executing
- Check ML model confidence thresholds
- Verify risk manager allows trading
- Check spread conditions
- Confirm trading hours settings

### High Loss Rate
- Reduce position size
- Increase ML confidence threshold
- Review model performance
- Check market conditions

## Disclaimer

**IMPORTANT**: Automated trading carries significant risk. 
- Past performance does not guarantee future results
- Use only risk capital you can afford to lose
- Start with demo account
- Monitor continuously
- Understand all functionality before going live

## Support

For issues:
1. Check logs in `CSV_FILES/trading_log.txt`
2. Review MT5 terminal logs
3. Test individual components separately
4. Use demo account for debugging
