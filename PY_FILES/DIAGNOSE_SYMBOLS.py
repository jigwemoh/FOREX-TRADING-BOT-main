#!/usr/bin/env python3
"""
Diagnose available symbols on MT5 account
"""

try:
    import MetaTrader5 as mt5  # type: ignore
except ImportError:
    import MT5_MOCK as mt5  # type: ignore

import time
import json
from pathlib import Path

def diagnose_symbols():
    """Test all common trading symbols to find what's available"""
    
    print("[INFO] Diagnosing MT5 symbol availability...\n")
    
    # Initialize MT5
    if not mt5.initialize():
        print("[ERROR] Failed to initialize MT5")
        return
    
    # Get account info
    account = mt5.account_info()
    print(f"[ACCOUNT] Login: {account.login}")
    print(f"[ACCOUNT] Server: {account.server}")
    print(f"[ACCOUNT] Balance: ${account.balance:,.2f}\n")
    
    # Test symbols organized by category
    test_symbols = {
        "Forex": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"],
        "Metals": ["XAUUSD", "XAGUSD"],
        "Crypto": ["BTCUSD", "ETHUSD", "LTCUSD"],
        "Indices": ["US500", "US100", "US30"],
    }
    
    available_symbols = {}
    
    for category, symbols in test_symbols.items():
        print(f"[{category.upper()}]")
        available_symbols[category] = []
        
        for sym in symbols:
            # Try to select symbol
            if mt5.symbol_select(sym, True):
                time.sleep(0.3)
                
                # Try to get tick
                tick = mt5.symbol_info_tick(sym)
                if tick is not None:
                    available_symbols[category].append(sym)
                    print(f"  ✓ {sym:10} | Bid: {tick.bid:>10.5f} | Ask: {tick.ask:>10.5f}")
                else:
                    print(f"  ⚠ {sym:10} | Selected but no tick data")
            else:
                print(f"  ✗ {sym:10} | Not available")
        
        print()
    
    # Summary
    print("[SUMMARY]")
    total_available = sum(len(syms) for syms in available_symbols.values())
    print(f"Total available symbols: {total_available}\n")
    
    for category, symbols in available_symbols.items():
        if symbols:
            print(f"{category}: {', '.join(symbols)}")
    
    # Save to config
    if available_symbols:
        recommended = []
        for syms in available_symbols.values():
            if syms:
                recommended.append(syms[0])
        
        if recommended:
            config = {
                "trading": {
                    "symbol": recommended[0],
                    "symbols": recommended,
                    "timeframe": "H1",
                    "risk_per_trade": 0.02,
                    "max_trades": 5
                },
                "mt5": {
                    "login": account.login,
                    "server": account.server
                },
                "available_symbols": available_symbols
            }
            
            config_path = Path("../config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            
            print(f"\n[SUCCESS] Updated config.json with available symbols")
            print(f"[RECOMMENDED] Start with: {', '.join(recommended)}")
    
    mt5.shutdown()

if __name__ == "__main__":
    diagnose_symbols()
