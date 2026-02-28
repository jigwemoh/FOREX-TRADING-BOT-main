#!/usr/bin/env python3
"""
Configuration Manager for Auto Trading
Securely manage MT5 credentials and trading parameters
"""

import json
from pathlib import Path
from typing import Dict, Any
import getpass

class ConfigManager:
    """Manage trading bot configuration"""
    
    def __init__(self, config_file: str = "../config.json"):
        self.config_file = Path(config_file)
        self.config: Dict[str, Any] = {}
        
    def create_config(self):
        """Interactive configuration setup"""
        print("=" * 50)
        print("MT5 AUTO TRADER CONFIGURATION")
        print("=" * 50)
        
        # MT5 credentials
        print("\n[MT5 Connection]")
        mt5_login = input("MT5 Account Number: ")
        mt5_password = getpass.getpass("MT5 Password: ")
        mt5_server = input("MT5 Server: ")
        mt5_terminal_path = input("MT5 Terminal Path (optional, e.g. C:\\Program Files\\MetaTrader 5\\terminal64.exe): ")
        
        # Trading parameters
        print("\n[Trading Parameters]")
        symbol = input("Symbol (default: EURUSD): ") or "EURUSD"
        timeframe = input("Timeframe (1M/5M/15M/30M/1H/4H/1D, default: 1H): ") or "1H"
        risk_percent = float(input("Risk per trade % (default: 1.0): ") or "1.0")
        max_positions = int(input("Max open positions (default: 3): ") or "3")
        use_ml = input("Use ML predictions? (yes/no, default: yes): ").lower() != "no"
        
        # Risk management
        print("\n[Risk Management]")
        stop_loss_pips = float(input("Stop loss in pips (default: 50): ") or "50")
        take_profit_pips = float(input("Take profit in pips (default: 100): ") or "100")
        trailing_stop_pips = float(input("Trailing stop in pips (default: 30): ") or "30")
        
        # Execution
        print("\n[Execution]")
        check_interval = int(input("Check interval in seconds (default: 300): ") or "300")
        
        # Trading hours
        print("\n[Trading Hours (24h format)]")
        start_hour = int(input("Start hour (default: 0 = always on): ") or "0")
        end_hour = int(input("End hour (default: 0 = always on): ") or "0")
        
        # Build config
        self.config = {
            "mt5": {
                "login": int(mt5_login),
                "password": mt5_password,
                "server": mt5_server,
                "terminal_path": mt5_terminal_path
            },
            "trading": {
                "symbol": symbol,
                "timeframe": timeframe,
                "risk_percent": risk_percent,
                "max_positions": max_positions,
                "use_ml": use_ml
            },
            "risk_management": {
                "stop_loss_pips": stop_loss_pips,
                "take_profit_pips": take_profit_pips,
                "trailing_stop_pips": trailing_stop_pips
            },
            "execution": {
                "check_interval": check_interval,
                "trading_hours": {
                    "start": start_hour,
                    "end": end_hour
                }
            },
            "notifications": {
                "telegram_enabled": False,
                "email_enabled": False
            }
        }
        
        # Save
        self.save_config()
        print(f"\n✓ Configuration saved to {self.config_file}")
        
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
            
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.config_file.exists():
            print(f"Configuration file not found: {self.config_file}")
            print("Creating new configuration...")
            self.create_config()
            
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)
            
        return self.config
        
    def update_config(self, section: str, key: str, value: Any):
        """Update a specific configuration value"""
        if section in self.config:
            self.config[section][key] = value
            self.save_config()
            
    def display_config(self):
        """Display current configuration (hide password)"""
        display_config = self.config.copy()
        if 'mt5' in display_config and 'password' in display_config['mt5']:
            display_config['mt5']['password'] = '***HIDDEN***'
            
        print(json.dumps(display_config, indent=4))


if __name__ == "__main__":
    manager = ConfigManager()
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "create":
        manager.create_config()
    else:
        config = manager.load_config()
        manager.display_config()
