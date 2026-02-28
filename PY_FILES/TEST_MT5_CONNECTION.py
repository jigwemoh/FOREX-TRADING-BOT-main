#!/usr/bin/env python3
"""
Test MT5 connection and diagnose issues
"""

try:
    import MetaTrader5 as mt5  # type: ignore
except ImportError:
    import MT5_MOCK as mt5  # type: ignore

import json
from pathlib import Path
import time

def test_mt5_connection():
    """Test MT5 connection and market data access"""
    
    print("[INFO] Testing MT5 connection...")
    
    # Load config
    config_path = Path("../config.json")
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return False
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    trading_cfg = config.get("trading", {})
    
    # MT5 credentials are already loaded in the terminal
    # Ensure symbol is uppercase
    SYMBOL = str(trading_cfg.get("symbol", "EURUSD")).upper()
    
    # Check if MT5 is initialized
    if not mt5.initialize():
        print("[ERROR] MT5 not initialized. Make sure terminal is running and logged in.")
        return False
    
    print("[SUCCESS] MT5 initialized")
    
    # Check account info
    account = mt5.account_info()
    if account is None:
        print("[ERROR] Could not get account info")
        mt5.shutdown()
        return False
    
    print(f"[SUCCESS] Connected to account: {account.login}")
    print(f"  Server: {account.server}")
    print(f"  Balance: {account.balance}")
    print(f"  Equity: {account.equity}")
    
    # Test symbol subscription
    print(f"\n[INFO] Testing symbol: {SYMBOL}")
    
    # Enable symbol in terminal
    if not mt5.symbol_select(SYMBOL, True):
        print(f"[WARNING] Could not select {SYMBOL}, trying anyway...")
    else:
        print(f"[SUCCESS] Symbol {SYMBOL} selected")
    
    # Wait for symbol to be ready
    print("[INFO] Waiting for market data to be ready...")
    time.sleep(2)
    
    # Try to get market data
    print(f"[INFO] Fetching 10 bars of {SYMBOL} 1H data...")
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 10)
    
    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        print(f"[ERROR] Failed to get market data: {error}")
        print(f"[INFO] Possible reasons:")
        print(f"  1. Symbol {SYMBOL} is not available on your account")
        print(f"  2. Market is closed")
        print(f"  3. Terminal connection is unstable")
        print(f"\n[ACTION] Try one of these:")
        print(f"  1. Check that {SYMBOL} is available in your account")
        print(f"  2. Change symbol in config.json (e.g., EURUSD, GBPUSD, etc.)")
        print(f"  3. Restart MetaTrader 5 terminal")
        mt5.shutdown()
        return False
    
    print(f"[SUCCESS] Got {len(rates)} bars of market data")
    print(f"\nLast bar:")
    print(f"  Time: {rates[-1][0]}")
    print(f"  Open: {rates[-1][1]}")
    print(f"  High: {rates[-1][2]}")
    print(f"  Low: {rates[-1][3]}")
    print(f"  Close: {rates[-1][4]}")
    print(f"  Volume: {rates[-1][5]}")
    
    # Test different symbols
    print(f"\n[INFO] Testing symbol availability...")
    test_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD"]
    available = []
    
    for sym in test_symbols:
        if mt5.symbol_select(sym, True):
            time.sleep(0.5)
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, 1)
            if rates is not None and len(rates) > 0:
                available.append(sym)
    
    if available:
        print(f"[SUCCESS] Available symbols: {', '.join(available)}")
    else:
        print(f"[WARNING] Could not fetch data for any test symbols")
    
    mt5.shutdown()
    return True

if __name__ == "__main__":
    test_mt5_connection()
