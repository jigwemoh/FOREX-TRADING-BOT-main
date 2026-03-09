#!/usr/bin/env python3
"""
Production-Grade Scalping Engine
Handles M1/M5 intraday trading with adaptive risk sizing and microstructure detection
Integration with AUTO_TRADER_MULTI.py for multi-timeframe hybrid execution
"""

import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SCALPING_ENGINE")


class MarketRegime(Enum):
    """Market volatility regimes"""
    LOW_VOLATILITY = 1      # Tight range, wide spreads relative to movement
    NORMAL_VOLATILITY = 2   # Balanced conditions
    HIGH_VOLATILITY = 3     # Wide movement, news/economic events
    EXTREME_VOLATILITY = 4  # Market breaking, stop hunts likely


class OrderFlowSignal(Enum):
    """Order flow microstructure signals"""
    LIQUIDITY_SWEEP = 1     # Stop hunt detection
    FALSE_BREAKOUT = 2      # Rejection after breakout attempt
    VOLUME_SPIKE = 3        # Sudden volume increase
    INSTITUTIONAL_BID = 4   # Significant bid/ask imbalance


@dataclass
class ScalpingSetup:
    """Scalping setup detection result"""
    setup_type: str  # 'EMA_BREAKOUT', 'VWAP_BOUNCE', 'LIQUIDITY_SWEEP', 'REJECTION_CANDLE'
    direction: int   # 1 for long, -1 for short, 0 for none
    confidence: float  # 0.0-1.0 confidence score
    entry_price: float
    stop_loss_pips: float
    target_pips: float
    order_flow_signal: Optional[OrderFlowSignal] = None
    volatility_regime: MarketRegime = MarketRegime.NORMAL_VOLATILITY
    atr_value: float = 0.0
    spread_pips: float = 0.0


@dataclass
class RiskMetrics:
    """Track trading risk across session"""
    daily_pnl: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    daily_drawdown: float = 0.0
    max_daily_drawdown: float = 0.0
    active_trades: int = 0
    winning_trades: int = 0


class ScalpingEngine:
    """Core scalping logic for M1/M5 timeframes"""

    def __init__(
        self,
        symbol: str,
        timeframe: str = "M1",
        atr_period: int = 14,
        vwap_lookback: int = 50,
        ema_fast: int = 20,
        ema_slow: int = 50
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.atr_period = atr_period
        self.vwap_lookback = vwap_lookback
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        
        # Risk management
        self.risk_per_trade = 0.02  # 2% max per trade
        self.max_daily_loss = 0.06  # -6% stop trading
        self.max_consecutive_losses = 4
        self.kelly_fraction = 0.5
        
        # Scalping parameters
        self.min_confidence = 0.65  # AI must be 65%+ confident
        self.win_rate_threshold = 0.55  # Expect 55% win rate
        self.reward_ratio = 1.2  # Risk 1R to make 1.2R
        
        # Trading hours
        self.london_open = 8  # 8 AM GMT
        self.london_close = 16  # 4 PM GMT
        self.ny_open = 13  # 1 PM GMT
        self.ny_close = 21  # 9 PM GMT
        self.asia_open = 0  # 12 AM GMT
        self.asia_close = 8  # 8 AM GMT
        
        # State tracking
        self.risk_metrics: Dict[str, RiskMetrics] = {}
        self.session_start_time = None
        self.last_setup: Optional[ScalpingSetup] = None
        self.news_window_active = False
        self.spread_baseline = 1.5  # Default 1.5 pips for major pairs
        
        logger.info(f"[INIT] Scalping Engine for {symbol} on {timeframe}")

    def detect_market_regime(self, df: pd.DataFrame, atr_value: float, spread_pips: float) -> MarketRegime:
        """
        Detect current market regime based on volatility and spread
        
        Args:
            df: OHLC data with ATR calculated
            atr_value: Current ATR value
            spread_pips: Current bid-ask spread in pips
            
        Returns:
            MarketRegime classification
        """
        if len(df) < 20:
            return MarketRegime.NORMAL_VOLATILITY
        
        # ATR median over last 20 bars
        recent_atr = df['ATR'].tail(20).median() if 'ATR' in df else atr_value
        
        # Volatility classification
        atr_ratio = atr_value / recent_atr if recent_atr > 0 else 1.0
        
        # Spread as proxy for volatility
        normal_spread = self.spread_baseline
        spread_ratio = spread_pips / normal_spread if normal_spread > 0 else 1.0
        
        # Decision logic
        if atr_ratio > 1.8 or spread_ratio > 2.5:
            return MarketRegime.EXTREME_VOLATILITY
        elif atr_ratio > 1.4 or spread_ratio > 1.8:
            return MarketRegime.HIGH_VOLATILITY
        elif atr_ratio < 0.7 or spread_ratio < 0.6:
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.NORMAL_VOLATILITY

    def calculate_ema_setup(
        self,
        df: pd.DataFrame,
        current_price: float,
        atr_value: float,
        spread_pips: float
    ) -> Optional[ScalpingSetup]:
        """
        EMA Breakout Setup
        Entry: Price > EMA20 > EMA50 (long) with RSI pullback
        
        Args:
            df: OHLC dataframe with technical indicators
            current_price: Latest price
            atr_value: Current ATR
            spread_pips: Current spread
            
        Returns:
            ScalpingSetup if valid entry found, None otherwise
        """
        if len(df) < 51:
            return None
        
        df = df.copy()
        
        # Calculate EMAs
        df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['RSI'] = self._calculate_rsi(df['close'], period=7)
        
        latest = df.iloc[-1]
        ema20 = latest['EMA20']
        ema50 = latest['EMA50']
        rsi = latest['RSI']
        
        # Long signal: Price > EMA20 > EMA50, RSI in 40-50 range (pullback)
        if current_price > ema20 > ema50 and 40 <= rsi <= 50:
            target_pips = 3  # 3 pips target for M1
            stop_loss_pips = atr_value / 10**4 * 10000  # ATR in pips
            
            return ScalpingSetup(
                setup_type='EMA_BREAKOUT_LONG',
                direction=1,
                confidence=0.68,
                entry_price=current_price,
                stop_loss_pips=max(stop_loss_pips, 2),
                target_pips=target_pips,
                volatility_regime=self.detect_market_regime(df, atr_value, spread_pips),
                atr_value=atr_value,
                spread_pips=spread_pips
            )
        
        # Short signal: Price < EMA20 < EMA50, RSI in 50-60 range
        if current_price < ema20 < ema50 and 50 <= rsi <= 60:
            target_pips = 3
            stop_loss_pips = atr_value / 10**4 * 10000
            
            return ScalpingSetup(
                setup_type='EMA_BREAKOUT_SHORT',
                direction=-1,
                confidence=0.68,
                entry_price=current_price,
                stop_loss_pips=max(stop_loss_pips, 2),
                target_pips=target_pips,
                volatility_regime=self.detect_market_regime(df, atr_value, spread_pips),
                atr_value=atr_value,
                spread_pips=spread_pips
            )
        
        return None

    def calculate_vwap_setup(
        self,
        df: pd.DataFrame,
        current_price: float,
        atr_value: float,
        spread_pips: float
    ) -> Optional[ScalpingSetup]:
        """
        VWAP Bounce Setup
        Entry: Price touches VWAP and bounces with momentum
        
        Args:
            df: OHLC dataframe with volume
            current_price: Latest price
            atr_value: Current ATR
            spread_pips: Current spread
            
        Returns:
            ScalpingSetup if valid entry found, None otherwise
        """
        if len(df) < 50:
            return None
        
        df = df.copy()
        
        # Calculate VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['CumPV'] = typical_price * df['volume']
        df['CumV'] = df['volume'].cumsum()
        df['VWAP'] = df['CumPV'].rolling(self.vwap_lookback).sum() / df['CumV'].rolling(self.vwap_lookback).sum()
        
        # Calculate momentum
        df['Momentum'] = df['close'].diff(3)
        df['RSI'] = self._calculate_rsi(df['close'], period=7)
        
        latest = df.iloc[-1]
        vwap = latest['VWAP']
        momentum = latest['Momentum']
        rsi = latest['RSI']
        
        # Long: Price bounced above VWAP with positive momentum and RSI < 50
        vwap_dist = abs(current_price - vwap) / current_price * 10000  # In pips
        if (vwap_dist < 2 and  # Within 2 pips of VWAP
            current_price > vwap and  # Bouncing above
            momentum > 0 and  # Positive momentum
            rsi < 50):  # Not overbought
            
            return ScalpingSetup(
                setup_type='VWAP_BOUNCE_LONG',
                direction=1,
                confidence=0.62,
                entry_price=current_price,
                stop_loss_pips=2.5,
                target_pips=2.0,
                volatility_regime=self.detect_market_regime(df, atr_value, spread_pips),
                atr_value=atr_value,
                spread_pips=spread_pips
            )
        
        # Short: Price bounced below VWAP with negative momentum and RSI > 50
        if (vwap_dist < 2 and
            current_price < vwap and
            momentum < 0 and
            rsi > 50):
            
            return ScalpingSetup(
                setup_type='VWAP_BOUNCE_SHORT',
                direction=-1,
                confidence=0.62,
                entry_price=current_price,
                stop_loss_pips=2.5,
                target_pips=2.0,
                volatility_regime=self.detect_market_regime(df, atr_value, spread_pips),
                atr_value=atr_value,
                spread_pips=spread_pips
            )
        
        return None

    def detect_liquidity_sweep(
        self,
        df: pd.DataFrame,
        current_price: float,
        atr_value: float,
        spread_pips: float
    ) -> Optional[ScalpingSetup]:
        """
        Liquidity Sweep Detection
        Pattern: Price runs through a level, then reverses sharply
        High probability reversal trade
        
        Args:
            df: OHLC dataframe
            current_price: Latest price
            atr_value: Current ATR
            spread_pips: Current spread
            
        Returns:
            ScalpingSetup if sweep detected
        """
        if len(df) < 10:
            return None
        
        df = df.copy()
        df['Range'] = df['high'] - df['low']
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        prev_prev = df.iloc[-3]
        
        # Sweep detected: Previous bars had tight range, latest breaks and reverses
        avg_range = df['Range'].tail(5).mean()
        latest_range = latest['Range']
        
        # Long sweep: Previous high was broken, now we're reversing below it
        if (prev['high'] > prev_prev['high'] and  # Higher high
            current_price < prev['high'] and  # But closed below
            latest_range > avg_range * 1.5):  # Large range bar
            
            return ScalpingSetup(
                setup_type='LIQUIDITY_SWEEP_SHORT',
                direction=-1,
                confidence=0.70,
                entry_price=current_price,
                stop_loss_pips=3.0,
                target_pips=4.0,
                order_flow_signal=OrderFlowSignal.LIQUIDITY_SWEEP,
                volatility_regime=self.detect_market_regime(df, atr_value, spread_pips),
                atr_value=atr_value,
                spread_pips=spread_pips
            )
        
        # Short sweep: Previous low was broken, now we're reversing above it
        if (prev['low'] < prev_prev['low'] and  # Lower low
            current_price > prev['low'] and  # But closed above
            latest_range > avg_range * 1.5):  # Large range bar
            
            return ScalpingSetup(
                setup_type='LIQUIDITY_SWEEP_LONG',
                direction=1,
                confidence=0.70,
                entry_price=current_price,
                stop_loss_pips=3.0,
                target_pips=4.0,
                order_flow_signal=OrderFlowSignal.LIQUIDITY_SWEEP,
                volatility_regime=self.detect_market_regime(df, atr_value, spread_pips),
                atr_value=atr_value,
                spread_pips=spread_pips
            )
        
        return None

    def detect_rejection_candle(
        self,
        df: pd.DataFrame,
        current_price: float,
        atr_value: float,
        spread_pips: float
    ) -> Optional[ScalpingSetup]:
        """
        Rejection Candle Detection
        Pattern: Candle with long wick showing price rejection at support/resistance
        High probability setup
        
        Args:
            df: OHLC dataframe
            current_price: Latest price
            atr_value: Current ATR
            spread_pips: Current spread
            
        Returns:
            ScalpingSetup if rejection detected
        """
        if len(df) < 5:
            return None
        
        df = df.copy()
        
        latest = df.iloc[-1]
        body_size = abs(latest['close'] - latest['open'])
        upper_wick = latest['high'] - max(latest['open'], latest['close'])
        lower_wick = min(latest['open'], latest['close']) - latest['low']
        
        # Rejection bullish: Long lower wick, small body, closes higher
        if lower_wick > body_size * 3 and latest['close'] > latest['open']:
            return ScalpingSetup(
                setup_type='REJECTION_BULLISH',
                direction=1,
                confidence=0.66,
                entry_price=current_price,
                stop_loss_pips=max(lower_wick / 10**4 * 10000, 2),
                target_pips=2.5,
                volatility_regime=self.detect_market_regime(df, atr_value, spread_pips),
                atr_value=atr_value,
                spread_pips=spread_pips
            )
        
        # Rejection bearish: Long upper wick, small body, closes lower
        if upper_wick > body_size * 3 and latest['close'] < latest['open']:
            return ScalpingSetup(
                setup_type='REJECTION_BEARISH',
                direction=-1,
                confidence=0.66,
                entry_price=current_price,
                stop_loss_pips=max(upper_wick / 10**4 * 10000, 2),
                target_pips=2.5,
                volatility_regime=self.detect_market_regime(df, atr_value, spread_pips),
                atr_value=atr_value,
                spread_pips=spread_pips
            )
        
        return None

    def evaluate_setup_quality(
        self,
        setup: ScalpingSetup,
        ai_probability: float = 0.65
    ) -> Tuple[bool, float]:
        """
        Evaluate if setup meets entry criteria
        
        Args:
            setup: ScalpingSetup to evaluate
            ai_probability: ML model confidence (0.0-1.0)
            
        Returns:
            (should_trade, final_confidence)
        """
        # Check AI confidence gate
        if ai_probability < self.min_confidence:
            return False, 0.0
        
        # Check reward/risk ratio
        if setup.target_pips < setup.stop_loss_pips * self.reward_ratio:
            return False, 0.0
        
        # Check volatility regime suitability
        if setup.volatility_regime == MarketRegime.EXTREME_VOLATILITY:
            return False, 0.0  # Too risky for scalping
        
        # Reduce confidence in low volatility
        regime_confidence = setup.confidence
        if setup.volatility_regime == MarketRegime.LOW_VOLATILITY:
            regime_confidence *= 0.8
        
        # Combine AI + setup confidence
        combined_confidence = (ai_probability + regime_confidence) / 2.0
        
        return True, combined_confidence

    def calculate_adaptive_lot_size(
        self,
        setup: ScalpingSetup,
        account_balance: float,
        current_positions: int,
        symbol: str
    ) -> float:
        """
        Calculate adaptive lot size using Kelly Criterion
        
        Args:
            setup: Scalping setup with risk parameters
            account_balance: Account balance in USD
            current_positions: Number of open positions
            symbol: Trading symbol
            
        Returns:
            Lot size (as fraction of balance)
        """
        if symbol not in self.risk_metrics:
            self.risk_metrics[symbol] = RiskMetrics()
        
        metrics = self.risk_metrics[symbol]
        
        # Get base risk
        base_risk = self.risk_per_trade * account_balance
        
        # Adjust for consecutive losses
        loss_penalty = max(0.1, 1.0 - (metrics.consecutive_losses * 0.25))
        
        # Adjust for daily drawdown
        if metrics.daily_drawdown < -0.04:
            loss_penalty *= 0.5  # 50% reduction at -4%
        
        # Position count penalty
        position_penalty = max(0.3, 1.0 - (current_positions * 0.15))
        
        # Kelly fraction application
        win_rate = (metrics.win_count / max(metrics.trade_count, 1))
        loss_rate = 1.0 - win_rate
        reward_ratio = setup.target_pips / setup.stop_loss_pips
        
        kelly_value = (win_rate * reward_ratio - loss_rate) / reward_ratio
        kelly_fraction_limited = min(self.kelly_fraction, max(0.1, kelly_value))
        
        # Final lot size
        adjusted_risk = base_risk * loss_penalty * position_penalty * kelly_fraction_limited
        
        logger.info(
            f"[LOT SIZE] {symbol} | Base: {base_risk:.2f} | "
            f"Loss Penalty: {loss_penalty:.2f} | Position Penalty: {position_penalty:.2f} | "
            f"Kelly: {kelly_fraction_limited:.3f} | Final: {adjusted_risk:.2f}"
        )
        
        return adjusted_risk

    def should_trade_now(self) -> bool:
        """
        Check if current time is suitable for scalping
        Avoid news windows and low liquidity times
        
        Returns:
            True if conditions suitable for trading
        """
        now = datetime.now(timezone.utc)
        hour = now.hour
        
        # High liquidity windows
        london_active = self.london_open <= hour < self.london_close
        ny_active = self.ny_open <= hour < self.ny_close
        overlap = (self.london_open <= hour < self.london_close) and (self.ny_open <= hour < self.ny_close)
        
        # Prefer overlap or high liquidity
        if overlap or london_active or ny_active:
            return not self.news_window_active
        
        return False

    def update_trade_outcome(
        self,
        symbol: str,
        pnl: float,
        is_win: bool,
        position_size: float
    ):
        """Track trade outcome for adaptive sizing"""
        if symbol not in self.risk_metrics:
            self.risk_metrics[symbol] = RiskMetrics()
        
        metrics = self.risk_metrics[symbol]
        metrics.daily_pnl += pnl
        metrics.trade_count += 1
        
        if is_win:
            metrics.win_count += 1
            metrics.consecutive_losses = 0
        else:
            metrics.loss_count += 1
            metrics.consecutive_losses += 1
            metrics.max_consecutive_losses = max(
                metrics.max_consecutive_losses,
                metrics.consecutive_losses
            )
        
        # Track drawdown
        metrics.daily_drawdown = min(0, metrics.daily_pnl)
        metrics.max_daily_drawdown = min(metrics.max_daily_drawdown, metrics.daily_drawdown)
        
        logger.info(
            f"[TRADE RESULT] {symbol} | PnL: {pnl:+.2f} | "
            f"Win: {metrics.win_count}/{metrics.trade_count} | "
            f"Consecutive Losses: {metrics.consecutive_losses} | "
            f"Daily PnL: {metrics.daily_pnl:+.2f}"
        )

    def should_stop_trading_session(self, symbol: str) -> bool:
        """
        Stop trading based on risk thresholds
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if should stop trading session
        """
        if symbol not in self.risk_metrics:
            return False
        
        metrics = self.risk_metrics[symbol]
        
        # Stop after 4 consecutive losses
        if metrics.consecutive_losses >= self.max_consecutive_losses:
            logger.warning(
                f"[STOP] {symbol} | Max consecutive losses ({self.max_consecutive_losses}) reached"
            )
            return True
        
        # Stop after -6% daily loss
        if metrics.daily_drawdown <= -self.max_daily_loss:
            logger.warning(
                f"[STOP] {symbol} | Max daily loss ({-self.max_daily_loss:.1%}) reached | "
                f"Current: {metrics.daily_drawdown:.1%}"
            )
            return True
        
        return False

    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def reset_session_metrics(self, symbol: str):
        """Reset daily metrics at session start"""
        self.risk_metrics[symbol] = RiskMetrics()
        self.session_start_time = datetime.now(timezone.utc)
        logger.info(f"[SESSION START] {symbol} | Metrics reset")


# Integration helper function
def create_scalping_setup_from_ml(
    symbol: str,
    ml_signal: int,
    ml_confidence: float,
    df: pd.DataFrame,
    current_price: float,
    atr_value: float,
    spread_pips: float,
    scalping_engine: ScalpingEngine
) -> Optional[Tuple[ScalpingSetup, float]]:
    """
    Create scalping setup combining ML signals with microstructure detection
    
    Args:
        symbol: Trading symbol
        ml_signal: ML model signal (1=buy, -1=sell, 0=hold)
        ml_confidence: ML model confidence (0.0-1.0)
        df: OHLC dataframe
        current_price: Current market price
        atr_value: Current ATR value
        spread_pips: Current bid-ask spread
        scalping_engine: ScalpingEngine instance
        
    Returns:
        (ScalpingSetup, final_confidence) tuple if valid setup, None otherwise
    """
    
    # Generate setup candidates from microstructure
    setups = []
    
    ema_setup = scalping_engine.calculate_ema_setup(df, current_price, atr_value, spread_pips)
    if ema_setup and ema_setup.direction == ml_signal:
        setups.append(ema_setup)
    
    vwap_setup = scalping_engine.calculate_vwap_setup(df, current_price, atr_value, spread_pips)
    if vwap_setup and vwap_setup.direction == ml_signal:
        setups.append(vwap_setup)
    
    sweep_setup = scalping_engine.detect_liquidity_sweep(df, current_price, atr_value, spread_pips)
    if sweep_setup and sweep_setup.direction == ml_signal:
        setups.append(sweep_setup)
    
    rejection_setup = scalping_engine.detect_rejection_candle(df, current_price, atr_value, spread_pips)
    if rejection_setup and rejection_setup.direction == ml_signal:
        setups.append(rejection_setup)
    
    if not setups:
        return None
    
    # Use strongest setup
    best_setup = max(setups, key=lambda x: x.confidence)  # type: ignore
    
    # Evaluate quality
    should_trade, final_confidence = scalping_engine.evaluate_setup_quality(
        best_setup,
        ai_probability=ml_confidence
    )
    
    if should_trade:
        logger.info(
            f"[SCALP SETUP] {symbol} | Type: {best_setup.setup_type} | "
            f"Direction: {'LONG' if best_setup.direction == 1 else 'SHORT'} | "
            f"Confidence: {final_confidence:.3f} | "
            f"SL: {best_setup.stop_loss_pips:.1f}p | TP: {best_setup.target_pips:.1f}p | "
            f"Regime: {best_setup.volatility_regime.name}"
        )
        return (best_setup, final_confidence)
    
    return None
