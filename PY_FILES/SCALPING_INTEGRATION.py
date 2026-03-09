#!/usr/bin/env python3
"""
Scalping Integration Module
Bridges SCALPING_ENGINE with AUTO_TRADER_MULTI for hybrid M1/M5 + H1/4H trading
Enables quick entry/exit on intraday signals while maintaining long-term ML edge
"""

import logging
from typing import Dict, Tuple, Any
from datetime import datetime, timezone
import json

logger = logging.getLogger("SCALPING_INTEGRATION")


class ScalpingIntegration:
    """
    Manages integration between ML signals and scalping setups
    Enables time-based strategy selection:
    - M1/M5 scalping during high liquidity (London/NY)
    - H1/4H swing trading during other times
    """
    
    def __init__(self, config_path: str = "../config.json"):
        """Initialize integration with config"""
        self.config = self._load_config(config_path)
        self.scalping_config = self.config.get("trading", {}).get("scalping", {})
        self.microstructure_config = self.config.get("trading", {}).get("microstructure", {})
        self.session_config = self.config.get("trading", {}).get("session_filters", {})
        
        self.scalping_enabled = self.scalping_config.get("enabled", True)
        self.session_cache: Dict[str, str] = {}  # symbol -> current_session
        
        logger.info(f"[INIT] Scalping Integration | Enabled: {self.scalping_enabled}")

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def get_strategy_for_time(self) -> Tuple[str, str, Dict[str, Any]]:
        """
        Determine optimal strategy and timeframe based on current time
        
        Returns:
            (strategy_name, recommended_timeframe, config_dict)
        """
        now = datetime.now(timezone.utc)
        hour = now.hour
        
        london_hours = self.session_config.get("london_hours", [8, 16])
        ny_hours = self.session_config.get("ny_hours", [13, 21])
        asia_hours = self.session_config.get("asia_hours", [0, 8])
        
        london_active = london_hours[0] <= hour < london_hours[1]
        ny_active = ny_hours[0] <= hour < ny_hours[1]
        asia_active = asia_hours[0] <= hour < asia_hours[1]
        
        # Overlap: London + NY (highest liquidity)
        if london_active and ny_active:
            strategy = "scalping_aggressive"
            timeframe = "M1"
            session = "OVERLAP"
        # London session: Moderate scalping
        elif london_active:
            strategy = "scalping_moderate"
            timeframe = "M5"
            session = "LONDON"
        # NY session: Moderate scalping
        elif ny_active:
            strategy = "scalping_moderate"
            timeframe = "M5"
            session = "NY"
        # Asia session: Light scalping or swing
        elif asia_active:
            strategy = "swing_conservative"
            timeframe = "1H"
            session = "ASIA"
        else:
            strategy = "swing_conservative"
            timeframe = "4H"
            session = "OFF_HOURS"
        
        return strategy, timeframe, {
            "session": session,
            "london_active": london_active,
            "ny_active": ny_active,
            "asia_active": asia_active
        }

    def get_scalping_parameters(self) -> Dict[str, Any]:
        """Get current scalping parameters for this session"""
        strategy, timeframe, session_info = self.get_strategy_for_time()
        
        # Adjust parameters based on session
        params = {
            "strategy": strategy,
            "timeframe": timeframe,
            "session": session_info["session"],
            
            # Base parameters
            "min_confidence": self.scalping_config.get("min_confidence", 0.65),
            "risk_per_trade": self.scalping_config.get("risk_per_trade", 0.02),
            "max_consecutive_losses": self.scalping_config.get("max_consecutive_losses", 4),
            "reward_ratio": self.scalping_config.get("reward_ratio", 1.2),
            
            # Microstructure detection
            "detect_liquidity_sweeps": self.microstructure_config.get("detect_liquidity_sweeps", True),
            "detect_rejection_candles": self.microstructure_config.get("detect_rejection_candles", True),
            "vwap_lookback": self.microstructure_config.get("vwap_lookback", 50),
            "atr_period": self.microstructure_config.get("atr_period", 14),
            "ema_fast": self.microstructure_config.get("ema_fast", 20),
            "ema_slow": self.microstructure_config.get("ema_slow", 50),
        }
        
        # Session-specific adjustments
        if session_info["session"] == "OVERLAP":
            # Most aggressive during overlap
            params["risk_per_trade"] = self.scalping_config.get("risk_per_trade", 0.02)
            params["min_confidence"] = 0.60  # Lower threshold, more trades
        elif session_info["session"] in ["LONDON", "NY"]:
            # Moderate scalping
            params["risk_per_trade"] = self.scalping_config.get("risk_per_trade", 0.02) * 0.8
            params["min_confidence"] = 0.63
        else:
            # Conservative during low liquidity
            params["risk_per_trade"] = self.scalping_config.get("risk_per_trade", 0.02) * 0.5
            params["min_confidence"] = 0.68
        
        return params

    def should_scalp_symbol(self, symbol: str) -> bool:
        """Check if symbol is suitable for scalping right now"""
        if not self.scalping_enabled:
            return False
        
        # Major forex pairs are preferred for scalping
        major_forex = {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD"}
        
        # Can scalp majors, limit scalping on altcoins/minors
        if symbol in major_forex:
            return True
        
        # Allow scalping on some crypto with caution
        crypto_ok = {"BTCUSD", "ETHUSD", "XRPUSD"}
        if symbol in crypto_ok:
            # Only during high liquidity periods
            _, _, session_info = self.get_strategy_for_time()
            return session_info["london_active"] or session_info["ny_active"]
        
        return False

    def calculate_setup_score(
        self,
        setup_type: str,
        ml_confidence: float,
        volatility_regime: str,
        spread_pips: float
    ) -> float:
        """
        Score a setup for trading quality
        Higher score = better trade opportunity
        
        Args:
            setup_type: Type of setup (EMA_BREAKOUT, VWAP_BOUNCE, etc)
            ml_confidence: ML model confidence (0.0-1.0)
            volatility_regime: Market regime (LOW, NORMAL, HIGH, EXTREME)
            spread_pips: Current bid-ask spread
            
        Returns:
            Quality score (0.0-1.0)
        """
        base_score = 0.5
        
        # Setup type bonuses
        setup_scores = {
            "LIQUIDITY_SWEEP": 0.85,  # Best for scalping
            "REJECTION_BULLISH": 0.80,
            "REJECTION_BEARISH": 0.80,
            "EMA_BREAKOUT_LONG": 0.70,
            "EMA_BREAKOUT_SHORT": 0.70,
            "VWAP_BOUNCE_LONG": 0.65,
            "VWAP_BOUNCE_SHORT": 0.65,
        }
        base_score = setup_scores.get(setup_type, 0.5)
        
        # ML confidence multiplier
        ml_multiplier = 0.5 + (ml_confidence * 0.5)  # Range: 0.5 to 1.0
        
        # Volatility adjustment
        regime_multiplier = {
            "NORMAL_VOLATILITY": 1.0,
            "HIGH_VOLATILITY": 0.85,
            "LOW_VOLATILITY": 0.7,
            "EXTREME_VOLATILITY": 0.4,
        }
        volatility_mult = regime_multiplier.get(volatility_regime, 0.8)
        
        # Spread penalty (wide spreads reduce profit potential)
        spread_penalty = max(0.6, 1.0 - (spread_pips / 2.0))  # 2 pips = breakeven
        
        # Final score
        final_score = base_score * ml_multiplier * volatility_mult * spread_penalty
        
        return min(1.0, max(0.0, final_score))

    def get_time_based_exit_bars(self) -> int:
        """
        Get number of bars to hold scalp before time-based exit
        Prevents holding losers too long
        
        Returns:
            Number of bars (for M1 = minutes, M5 = 5-minute intervals)
        """
        _, timeframe, _ = self.get_strategy_for_time()
        
        if timeframe == "M1":
            return 5  # 5 minutes max
        elif timeframe == "M5":
            return 3  # 15 minutes max
        else:
            return 30  # 30+ bars for swing
    
    def log_scalping_stats(self, symbol: str, stats: Dict[str, Any]):
        """Log scalping session statistics"""
        logger.info(
            f"[SCALP STATS] {symbol} | "
            f"Trades: {stats.get('trade_count', 0)} | "
            f"Wins: {stats.get('win_count', 0)} | "
            f"PnL: {stats.get('daily_pnl', 0):+.2f} | "
            f"Win Rate: {stats.get('win_rate', 0):.1%} | "
            f"Max DD: {stats.get('max_dd', 0):.1%}"
        )

    @staticmethod
    def format_setup_report(symbol: str, setup_data: Dict[str, Any]) -> str:
        """Format scalping setup for logging/reporting"""
        return (
            f"[{symbol}] {setup_data.get('setup_type', 'UNKNOWN')} | "
            f"Dir: {'+1' if setup_data.get('direction', 0) > 0 else '-1'} | "
            f"Conf: {setup_data.get('confidence', 0):.2f} | "
            f"SL: {setup_data.get('stop_loss_pips', 0):.1f}p | "
            f"TP: {setup_data.get('target_pips', 0):.1f}p"
        )


class RiskAdjustmentEngine:
    """
    Dynamically adjusts trading risk based on:
    - Account volatility
    - Drawdown severity
    - Win rate trends
    - Portfolio concentration
    """
    
    def __init__(self, initial_risk: float = 0.02):
        self.initial_risk = initial_risk
        self.current_risk = initial_risk
        self.adjustment_history = []
    
    def adjust_for_drawdown(self, current_dd: float, max_dd: float) -> float:
        """
        Reduce risk when drawdown is severe
        
        Args:
            current_dd: Current daily drawdown (e.g., -0.03 for -3%)
            max_dd: Max allowed daily drawdown (e.g., -0.06 for -6%)
            
        Returns:
            Adjusted risk multiplier (0.3 to 1.0)
        """
        if current_dd > -0.02:  # Safe zone
            return 1.0
        elif current_dd > -0.04:  # Caution zone
            return 0.7
        elif current_dd > -0.05:  # High caution
            return 0.5
        else:  # Critical
            return 0.3
    
    def adjust_for_win_rate(self, win_rate: float, min_expectancy: float = 0.55) -> float:
        """
        Reduce risk if win rate drops
        
        Args:
            win_rate: Current win rate (e.g., 0.55 for 55%)
            min_expectancy: Minimum acceptable win rate
            
        Returns:
            Adjusted risk multiplier (0.5 to 1.0)
        """
        if win_rate >= 0.55:
            return 1.0
        elif win_rate >= 0.52:
            return 0.8
        elif win_rate >= 0.50:
            return 0.6
        else:
            return 0.5
    
    def adjust_for_position_concentration(self, active_symbols: int, total_symbols: int) -> float:
        """
        Reduce risk if too concentrated in few symbols
        
        Args:
            active_symbols: Number of symbols with open trades
            total_symbols: Total available symbols
            
        Returns:
            Adjusted risk multiplier
        """
        concentration = active_symbols / max(total_symbols, 1)
        
        if concentration <= 0.25:  # Trading 1-2 of 4+ symbols
            return 1.0
        elif concentration <= 0.5:
            return 0.8
        elif concentration <= 0.75:
            return 0.6
        else:
            return 0.4  # Trading most symbols - reduce risk significantly
    
    def get_adjusted_risk(
        self,
        current_dd: float,
        max_dd: float,
        win_rate: float,
        active_symbols: int,
        total_symbols: int
    ) -> float:
        """Calculate final adjusted risk"""
        drawdown_mult = self.adjust_for_drawdown(current_dd, max_dd)
        wr_mult = self.adjust_for_win_rate(win_rate)
        concentration_mult = self.adjust_for_position_concentration(active_symbols, total_symbols)
        
        final_risk = self.initial_risk * drawdown_mult * wr_mult * concentration_mult
        
        self.current_risk = final_risk
        self.adjustment_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "risk": final_risk,
            "drawdown_mult": drawdown_mult,
            "wr_mult": wr_mult,
            "concentration_mult": concentration_mult
        })
        
        logger.info(
            f"[RISK ADJUST] DD: {drawdown_mult:.2f}x | WR: {wr_mult:.2f}x | "
            f"Conc: {concentration_mult:.2f}x | Final: {final_risk:.4f}"
        )
        
        return final_risk
