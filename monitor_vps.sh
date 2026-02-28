#!/bin/bash
# Monitor Forex Trading Bot on Windows VPS from Mac
# Usage: ./monitor_vps.sh YOUR_VPS_IP

VPS_IP="${1}"
VPS_USER="Administrator"

if [ -z "$VPS_IP" ]; then
    echo "Usage: ./monitor_vps.sh YOUR_VPS_IP"
    echo "Example: ./monitor_vps.sh 192.168.1.100"
    exit 1
fi

echo "=========================================="
echo "Forex Trading Bot - VPS Monitor"
echo "VPS: $VPS_IP"
echo "=========================================="

while true; do
    clear
    echo "=========================================="
    echo "Forex Trading Bot Status - $(date)"
    echo "=========================================="
    
    # Check if bot is running
    echo ""
    echo "Process Status:"
    ssh "${VPS_USER}@${VPS_IP}" "powershell -Command '
        \$process = Get-Process -Name python -ErrorAction SilentlyContinue
        if (\$process) {
            Write-Host \"✓ Bot is RUNNING (PID: \$(\$process.Id))\"
            Write-Host \"  CPU: \$(\$process.CPU)\"
            Write-Host \"  Memory: \$([math]::Round(\$process.WorkingSet64/1MB, 2)) MB\"
        } else {
            Write-Host \"✗ Bot is NOT running\"
        }
    '"
    
    # Get account info
    echo ""
    echo "Account Status:"
    ssh "${VPS_USER}@${VPS_IP}" "powershell -Command '
        cd C:\FOREX-TRADING-BOT-main\PY_FILES
        ..\venv\Scripts\python.exe -c \"
import MetaTrader5 as mt5
import sys
if not mt5.initialize():
    print('✗ MT5 not connected')
    sys.exit(1)
    
account = mt5.account_info()
if account:
    print(f'Balance: \${account.balance:.2f}')
    print(f'Equity: \${account.equity:.2f}')
    print(f'Profit: \${account.profit:.2f}')
    print(f'Margin: \${account.margin:.2f}')
    print(f'Free Margin: \${account.margin_free:.2f}')
    
    # Get positions
    positions = mt5.positions_get()
    print(f'\\nOpen Positions: {len(positions) if positions else 0}')
    if positions:
        for pos in positions:
            print(f'  {pos.symbol}: {pos.type_str} | Lots: {pos.volume} | P/L: \${pos.profit:.2f}')
else:
    print('✗ Account info unavailable')
    
mt5.shutdown()
\"
    '" 2>/dev/null || echo "✗ Unable to get account info"
    
    # Show recent log entries
    echo ""
    echo "Recent Logs (last 10 lines):"
    ssh "${VPS_USER}@${VPS_IP}" "powershell -Command '
        Get-Content C:\FOREX-TRADING-BOT-main\CSV_FILES\trading_log.txt -Tail 10 -ErrorAction SilentlyContinue
    '" 2>/dev/null || echo "✗ Unable to read logs"
    
    echo ""
    echo "=========================================="
    echo "Press Ctrl+C to exit | Refreshing in 30s..."
    echo "=========================================="
    
    sleep 30
done
