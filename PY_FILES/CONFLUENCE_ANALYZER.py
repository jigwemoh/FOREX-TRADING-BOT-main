#!/usr/bin/env python3
"""
Confluence Analyzer for Phase 2 Entry Filtering
Evaluates multiple technical signals to improve entry quality
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class ConfluenceAnalyzer:
    """Analyzes multiple signals for high-confidence entries"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.enabled = self.config.get("confluence_filtering", True)
        self.min_signals = self.config.get("min_confluence_signals", 2)
        self.rsi_oversold = self.config.get("rsi_oversold_threshold", 30)
        self.rsi_overbought = self.config.get("rsi_overbought_threshold", 70)
    
    def analyze(self, df: pd.DataFrame, direction: int = 1) -> Dict[str, int]:
        """
        Analyze confluence signals for a direction (1=BUY, -1=SELL)
        Returns dict with signal counts and details
        """
        signals = {
            "ema_trend": 0,
            "rsi_extreme": 0,
            "macd_momentum": 0,
            "bb_extreme": 0,
            "total_confluence": 0,
        }
        
        if df.empty or len(df) < 2:
            return signals
        
        try:
            last = df.iloc[-1]
            
            # =====================
            # EMA TREND SIGNAL
            # =====================
            if "EMA_20" in df.columns and "EMA_50" in df.columns:
                ema20 = last.get("EMA_20", 0)
                ema50 = last.get("EMA_50", 0)
                close = last.get("Close", 0)
                
                if direction == 1:  # BUY
                    # Close > EMA20 > EMA50 = uptrend
                    if close > ema20 and ema20 > ema50:
                        signals["ema_trend"] = 1
                else:  # SELL
                    # Close < EMA20 < EMA50 = downtrend
                    if close < ema20 and ema20 < ema50:
                        signals["ema_trend"] = 1
            
            # =====================
            # RSI EXTREME SIGNAL
            # =====================
            if "RSI" in df.columns:
                rsi = last.get("RSI", 50)
                
                if direction == 1:  # BUY - RSI oversold
                    if rsi < self.rsi_oversold:
                        signals["rsi_extreme"] = 1
                else:  # SELL - RSI overbought
                    if rsi > self.rsi_overbought:
                        signals["rsi_extreme"] = 1
            
            # =====================
            # MACD MOMENTUM SIGNAL
            # =====================
            if "MACD" in df.columns and "MACD_Signal" in df.columns:
                macd = last.get("MACD", 0)
                macd_signal = last.get("MACD_Signal", 0)
                
                if direction == 1:  # BUY
                    if macd > macd_signal:
                        signals["macd_momentum"] = 1
                else:  # SELL
                    if macd < macd_signal:
                        signals["macd_momentum"] = 1
            
            # =====================
            # BOLLINGER BANDS SIGNAL
            # =====================
            if "BB_L" in df.columns and "BB_H" in df.columns and "Close" in df.columns:
                close = last.get("Close", 0)
                bb_low = last.get("BB_L", 0)
                bb_high = last.get("BB_H", 0)
                bb_mid = last.get("BB_Mid", close)
                
                if bb_low > 0 and bb_high > 0:
                    bb_range = bb_high - bb_low
                    
                    if direction == 1:  # BUY - close near lower band
                        if close < bb_low + (bb_range * 0.3):
                            signals["bb_extreme"] = 1
                    else:  # SELL - close near upper band
                        if close > bb_high - (bb_range * 0.3):
                            signals["bb_extreme"] = 1
            
            # Calculate total confluence
            signals["total_confluence"] = sum([
                signals["ema_trend"],
                signals["rsi_extreme"],
                signals["macd_momentum"],
                signals["bb_extreme"],
            ])
            
        except Exception as e:
            logger.debug(f"Confluence analysis error: {e}")
        
        return signals
    
    def is_confluent(self, df: pd.DataFrame, direction: int = 1) -> bool:
        """Check if entry has minimum confluence signals"""
        if not self.enabled:
            return True
        
        signals = self.analyze(df, direction)
        return signals["total_confluence"] >= self.min_signals
    
    def get_signal_summary(self, signals: Dict) -> str:
        """Get human-readable summary of signals"""
        parts = []
        if signals.get("ema_trend"):
            parts.append("EMA↑")
        if signals.get("rsi_extreme"):
            parts.append("RSI✓")
        if signals.get("macd_momentum"):
            parts.append("MACD↑")
        if signals.get("bb_extreme"):
            parts.append("BB✓")
        
        return f"[{signals['total_confluence']}/4: {' '.join(parts)}]"


def get_confluence_config(base_config: Dict) -> Dict:
    """Extract confluence config from base config"""
    trading_config = base_config.get("trading", {})
    scalping_config = trading_config.get("scalping", {})
    
    return {
        "confluence_filtering": scalping_config.get("confluence_filtering", True),
        "min_confluence_signals": scalping_config.get("min_confluence_signals", 2),
        "rsi_oversold_threshold": 30,
        "rsi_overbought_threshold": 70,
    }
