#!/usr/bin/env python3
"""
Advanced Risk Management Module
Monitors account health and enforces trading rules
"""

import MetaTrader5 as mt5  # type: ignore
from datetime import datetime
from typing import Dict
import logging

class RiskManager:
    """Advanced risk management for auto trading"""
    
    def __init__(
        self,
        max_daily_loss_percent: float = 5.0,
        max_daily_trades: int = 10,
        max_drawdown_percent: float = 20.0,
        max_correlation_positions: int = 2
    ):
        self.max_daily_loss_percent = max_daily_loss_percent
        self.max_daily_trades = max_daily_trades
        self.max_drawdown_percent = max_drawdown_percent
        self.max_correlation_positions = max_correlation_positions
        
        self.daily_trades = 0
        self.daily_start_balance = 0.0
        self.peak_balance = 0.0
        self.last_reset = datetime.now().date()
        
    def reset_daily_counters(self):
        """Reset daily tracking counters"""
        account_info = mt5.account_info()
        if account_info:
            self.daily_start_balance = account_info.balance
            self.peak_balance = max(self.peak_balance, account_info.balance)
            
        self.daily_trades = 0
        self.last_reset = datetime.now().date()
        logging.info(f"Daily counters reset. Starting balance: {self.daily_start_balance}")
        
    def check_daily_reset(self):
        """Check if we need to reset daily counters"""
        if datetime.now().date() > self.last_reset:
            self.reset_daily_counters()
            
    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is exceeded"""
        self.check_daily_reset()
        
        account_info = mt5.account_info()
        if not account_info:
            return False
            
        current_balance = account_info.balance
        daily_loss = self.daily_start_balance - current_balance
        daily_loss_percent = (daily_loss / self.daily_start_balance) * 100
        
        if daily_loss_percent >= self.max_daily_loss_percent:
            logging.warning(f"Daily loss limit exceeded: {daily_loss_percent:.2f}% (limit: {self.max_daily_loss_percent}%)")
            return False
            
        return True
        
    def check_max_drawdown(self) -> bool:
        """Check if maximum drawdown is exceeded"""
        account_info = mt5.account_info()
        if not account_info:
            return False
            
        current_balance = account_info.balance
        drawdown = ((self.peak_balance - current_balance) / self.peak_balance) * 100
        
        if drawdown >= self.max_drawdown_percent:
            logging.warning(f"Max drawdown exceeded: {drawdown:.2f}% (limit: {self.max_drawdown_percent}%)")
            return False
            
        return True
        
    def check_daily_trade_limit(self) -> bool:
        """Check if daily trade limit is exceeded"""
        self.check_daily_reset()
        
        if self.daily_trades >= self.max_daily_trades:
            logging.warning(f"Daily trade limit reached: {self.daily_trades}/{self.max_daily_trades}")
            return False
            
        return True
        
    def increment_trade_count(self):
        """Increment daily trade counter"""
        self.daily_trades += 1
        logging.info(f"Daily trades: {self.daily_trades}/{self.max_daily_trades}")
        
    def check_correlated_positions(self, symbol: str) -> bool:
        """Check if too many correlated positions are open"""
        # Define correlated pairs
        correlations = {
            "EURUSD": ["GBPUSD", "AUDUSD", "NZDUSD"],
            "GBPUSD": ["EURUSD", "GBPJPY", "EURGBP"],
            "USDJPY": ["EURJPY", "GBPJPY"],
            "AUDUSD": ["NZDUSD", "EURUSD", "AUDNZD"],
            "NZDUSD": ["AUDUSD", "EURUSD", "AUDNZD"]
        }
        
        related_pairs = correlations.get(symbol, [])
        related_pairs.append(symbol)
        
        # Count positions in correlated pairs
        correlated_count = 0
        positions = mt5.positions_get()
        
        if positions:
            for pos in positions:
                if pos.symbol in related_pairs:
                    correlated_count += 1
                    
        if correlated_count >= self.max_correlation_positions:
            logging.warning(f"Too many correlated positions: {correlated_count}/{self.max_correlation_positions}")
            return False
            
        return True
        
    def check_spread(self, symbol: str, max_spread_pips: float = 3.0) -> bool:
        """Check if current spread is acceptable"""
        tick = mt5.symbol_info_tick(symbol)
        symbol_info = mt5.symbol_info(symbol)
        
        if not tick or not symbol_info:
            return False
            
        spread = (tick.ask - tick.bid) / (symbol_info.point * 10)  # Convert to pips
        
        if spread > max_spread_pips:
            logging.warning(f"Spread too high: {spread:.1f} pips (max: {max_spread_pips})")
            return False
            
        return True
        
    def check_market_hours(self, start_hour: int = 0, end_hour: int = 0) -> bool:
        """Check if current time is within trading hours"""
        if start_hour == 0 and end_hour == 0:
            return True  # Always trade
            
        current_hour = datetime.now().hour
        
        if start_hour < end_hour:
            # Normal hours (e.g., 9-17)
            if not (start_hour <= current_hour < end_hour):
                logging.info(f"Outside trading hours: {current_hour} (allowed: {start_hour}-{end_hour})")
                return False
        else:
            # Overnight hours (e.g., 22-6)
            if not (current_hour >= start_hour or current_hour < end_hour):
                logging.info(f"Outside trading hours: {current_hour} (allowed: {start_hour}-{end_hour})")
                return False
                
        return True
        
    def can_trade(
        self,
        symbol: str,
        start_hour: int = 0,
        end_hour: int = 0,
        max_spread_pips: float = 3.0
    ) -> bool:
        """
        Comprehensive check if trading is allowed
        Returns True if all risk checks pass
        """
        checks = [
            ("Daily Loss Limit", self.check_daily_loss_limit()),
            ("Max Drawdown", self.check_max_drawdown()),
            ("Daily Trade Limit", self.check_daily_trade_limit()),
            ("Correlated Positions", self.check_correlated_positions(symbol)),
            ("Spread Check", self.check_spread(symbol, max_spread_pips)),
            ("Trading Hours", self.check_market_hours(start_hour, end_hour))
        ]
        
        for check_name, result in checks:
            if not result:
                logging.warning(f"Risk check failed: {check_name}")
                return False
                
        return True
        
    def get_account_summary(self) -> Dict[str, float]:
        """Get current account risk summary"""
        account_info = mt5.account_info()
        if not account_info:
            return {}
            
        self.check_daily_reset()
        
        current_balance = account_info.balance
        equity = account_info.equity
        margin = account_info.margin
        free_margin = account_info.margin_free
        
        daily_pnl = current_balance - self.daily_start_balance
        daily_pnl_percent = (daily_pnl / self.daily_start_balance) * 100 if self.daily_start_balance > 0 else 0
        
        drawdown = ((self.peak_balance - current_balance) / self.peak_balance) * 100 if self.peak_balance > 0 else 0
        
        return {
            "balance": current_balance,
            "equity": equity,
            "margin": margin,
            "free_margin": free_margin,
            "daily_pnl": daily_pnl,
            "daily_pnl_percent": daily_pnl_percent,
            "daily_trades": self.daily_trades,
            "drawdown_percent": drawdown,
            "margin_level": (equity / margin * 100) if margin > 0 else 0
        }


if __name__ == "__main__":
    # Test risk manager
    if mt5.initialize():
        rm = RiskManager(
            max_daily_loss_percent=5.0,
            max_daily_trades=10,
            max_drawdown_percent=20.0
        )
        
        summary = rm.get_account_summary()
        print("Account Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
            
        can_trade = rm.can_trade("EURUSD")
        print(f"\nCan trade: {can_trade}")
        
        mt5.shutdown()
