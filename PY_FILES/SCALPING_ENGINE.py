#!/usr/bin/env python3
"""
Production-Grade Scalping Engine
Handles M1/M5 intraday trading with adaptive risk sizing and microstructure detection
Integration with AUTO_TRADER_MULTI.py for multi-timeframe hybrid execution
"""

import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timezone
import numpy as np
from pathlib import Path
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SCALPING_ENGINE")

# ML imports for setup quality prediction
ml_available = False
try:
    from catboost import CatBoostClassifier, Pool  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.metrics import accuracy_score, precision_score, recall_score  # type: ignore
    ml_available = True
except ImportError:
    CatBoostClassifier = None  # type: ignore
    Pool = None  # type: ignore
    train_test_split = None  # type: ignore
    accuracy_score = None  # type: ignore
    precision_score = None  # type: ignore
    recall_score = None  # type: ignore
    pass

ML_AVAILABLE = ml_available  # type: ignore

if not ML_AVAILABLE:
    logger.warning("ML libraries not available - setup quality prediction disabled")


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
    HAMMER_REVERSAL = 5     # Hammer/shooting star pattern
    ENGULFING_BREAK = 6     # Engulfing candle breakout
    FRACTAL_BREAK = 7       # Williams fractal breakout
    ICEBERG_ORDER = 8       # Large hidden order detection


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


@dataclass
class SetupQualityFeatures:
    """Features for ML setup quality prediction"""
    setup_type: str
    direction: int
    confidence: float
    volatility_regime: str
    atr_value: float
    spread_pips: float
    hour_of_day: int
    day_of_week: int
    rsi_value: float
    volume_ratio: float
    trend_strength: float
    support_resistance_distance: float
    consecutive_losses: int
    win_rate_recent: float
    market_regime_score: float


class MLSetupQualityPredictor:
    """Machine learning model to predict setup success probability"""

    def __init__(self, model_path: str = "../ALL_MODELS/setup_quality_model.cbm"):
        self.model_path = Path(model_path)
        self.model: Any = None
        self.feature_columns = [
            'setup_type', 'direction', 'confidence', 'volatility_regime', 'atr_value',
            'spread_pips', 'hour_of_day', 'day_of_week', 'rsi_value', 'volume_ratio',
            'trend_strength', 'support_resistance_distance', 'consecutive_losses',
            'win_rate_recent', 'market_regime_score'
        ]
        self.categorical_features = ['setup_type', 'volatility_regime']
        self.is_trained = False
        self.training_data_path = Path("../CSV_FILES/setup_quality_training.csv")

        if ML_AVAILABLE:
            self.load_or_train_model()
        else:
            logger.warning("ML libraries not available - using rule-based quality prediction")

    def load_or_train_model(self):
        """Load existing model or train new one from historical data"""
        if not ML_AVAILABLE:
            logger.info("[ML] ML libraries not available - using rule-based prediction")
            return
            
        try:
            if self.model_path.exists():
                self.model = CatBoostClassifier()  # type: ignore
                self.model.load_model(str(self.model_path))
                self.is_trained = True
                logger.info(f"[ML] Loaded setup quality model from {self.model_path}")
            else:
                logger.info("[ML] No existing model found - will use rule-based prediction until trained")
        except Exception as e:
            logger.error(f"[ML] Error loading model: {e}")
            self.model = None

    def extract_features(self, setup: ScalpingSetup, market_data: Dict[str, Any],
                        risk_metrics: RiskMetrics) -> SetupQualityFeatures:
        """Extract features for ML prediction from setup and market context"""

        # Basic setup features
        setup_type = setup.setup_type
        direction = setup.direction
        confidence = setup.confidence

        # Market context features
        volatility_regime = setup.volatility_regime.name
        atr_value = setup.atr_value
        spread_pips = setup.spread_pips

        # Time features
        now = datetime.now(timezone.utc)
        hour_of_day = now.hour
        day_of_week = now.weekday()

        # Technical indicators (would be calculated from market_data)
        rsi_value = market_data.get('rsi', 50.0)
        volume_ratio = market_data.get('volume_ratio', 1.0)
        trend_strength = market_data.get('trend_strength', 0.0)
        support_resistance_distance = market_data.get('sr_distance', 0.5)

        # Risk metrics
        consecutive_losses = risk_metrics.consecutive_losses
        win_rate_recent = (risk_metrics.win_count / max(risk_metrics.trade_count, 1))

        # Market regime score (0-1 scale)
        regime_scores = {
            'LOW_VOLATILITY': 0.2,
            'NORMAL_VOLATILITY': 0.5,
            'HIGH_VOLATILITY': 0.8,
            'EXTREME_VOLATILITY': 0.1  # Low score due to unpredictability
        }
        market_regime_score = regime_scores.get(volatility_regime, 0.5)

        return SetupQualityFeatures(
            setup_type=setup_type,
            direction=direction,
            confidence=confidence,
            volatility_regime=volatility_regime,
            atr_value=atr_value,
            spread_pips=spread_pips,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            rsi_value=rsi_value,
            volume_ratio=volume_ratio,
            trend_strength=trend_strength,
            support_resistance_distance=support_resistance_distance,
            consecutive_losses=consecutive_losses,
            win_rate_recent=win_rate_recent,
            market_regime_score=market_regime_score
        )

    def predict_quality_probability(self, features: SetupQualityFeatures) -> float:
        """Predict setup success probability using ML model"""

        if not ML_AVAILABLE or not self.is_trained or self.model is None:
            # Rule-based fallback
            return self._rule_based_prediction(features)

        try:
            # Convert features to DataFrame
            feature_dict = {
                'setup_type': features.setup_type,
                'direction': features.direction,
                'confidence': features.confidence,
                'volatility_regime': features.volatility_regime,
                'atr_value': features.atr_value,
                'spread_pips': features.spread_pips,
                'hour_of_day': features.hour_of_day,
                'day_of_week': features.day_of_week,
                'rsi_value': features.rsi_value,
                'volume_ratio': features.volume_ratio,
                'trend_strength': features.trend_strength,
                'support_resistance_distance': features.support_resistance_distance,
                'consecutive_losses': features.consecutive_losses,
                'win_rate_recent': features.win_rate_recent,
                'market_regime_score': features.market_regime_score
            }

            df = pd.DataFrame([feature_dict])

            # Create Pool for prediction
            pool = Pool(df, cat_features=self.categorical_features)  # type: ignore

            # Get probability of positive class (success)
            probabilities = self.model.predict_proba(pool)
            success_probability = probabilities[0][1]  # Probability of class 1 (success)

            logger.info(f"[ML] Setup quality prediction: {success_probability:.3f}")
            return success_probability

        except Exception as e:
            logger.error(f"[ML] Prediction error: {e}")
            return self._rule_based_prediction(features)

    def _rule_based_prediction(self, features: SetupQualityFeatures) -> float:
        """Rule-based quality prediction when ML model unavailable"""

        base_score = features.confidence

        # Adjust for volatility regime
        regime_multipliers = {
            'LOW_VOLATILITY': 1.1,    # More predictable
            'NORMAL_VOLATILITY': 1.0, # Baseline
            'HIGH_VOLATILITY': 0.9,   # Less predictable
            'EXTREME_VOLATILITY': 0.7 # Very unpredictable
        }
        base_score *= regime_multipliers.get(features.volatility_regime, 1.0)

        # Adjust for recent performance
        if features.consecutive_losses > 2:
            base_score *= 0.8  # Reduce confidence after losses
        elif features.win_rate_recent > 0.6:
            base_score *= 1.1  # Increase confidence after wins

        # Adjust for time of day (prefer active sessions)
        if 8 <= features.hour_of_day <= 16:  # London session
            base_score *= 1.05
        elif 13 <= features.hour_of_day <= 21:  # NY session
            base_score *= 1.05

        # Adjust for spread (wider spreads reduce quality)
        if features.spread_pips > 3.0:
            base_score *= 0.9

        return min(0.95, max(0.1, base_score))  # Clamp between 10% and 95%

    def record_setup_outcome(self, features: SetupQualityFeatures, outcome: int):
        """Record setup outcome for future model training"""

        if not self.training_data_path.exists():
            # Create header
            with open(self.training_data_path, 'w') as f:
                f.write(','.join(self.feature_columns + ['outcome']) + '\n')

        # Append outcome data
        feature_values = [
            features.setup_type, features.direction, features.confidence,
            features.volatility_regime, features.atr_value, features.spread_pips,
            features.hour_of_day, features.day_of_week, features.rsi_value,
            features.volume_ratio, features.trend_strength,
            features.support_resistance_distance, features.consecutive_losses,
            features.win_rate_recent, features.market_regime_score, outcome
        ]

        with open(self.training_data_path, 'a') as f:
            f.write(','.join(map(str, feature_values)) + '\n')

    def retrain_model(self, min_samples: int = 1000):
        """Retrain ML model from accumulated training data"""

        if not ML_AVAILABLE or not self.training_data_path.exists():
            logger.warning("[ML] Cannot retrain - ML libraries or training data unavailable")
            return False

        try:
            # Load training data
            df = pd.read_csv(self.training_data_path)

            if len(df) < min_samples:
                logger.info(f"[ML] Not enough training samples ({len(df)} < {min_samples})")
                return False

            # Prepare features and target
            X = df[self.feature_columns]
            y = df['outcome']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(  # type: ignore
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train model
            self.model = CatBoostClassifier(  # type: ignore
                iterations=1000,
                learning_rate=0.1,
                depth=6,
                cat_features=self.categorical_features,
                verbose=False,
                random_state=42
            )

            train_pool = Pool(X_train, y_train, cat_features=self.categorical_features)  # type: ignore
            test_pool = Pool(X_test, y_test, cat_features=self.categorical_features)  # type: ignore

            self.model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)  # type: ignore
            precision = precision_score(y_test, y_pred)  # type: ignore
            recall = recall_score(y_test, y_pred)  # type: ignore

            logger.info(
                f"[ML] Model retrained - Accuracy: {accuracy:.3f}, "
                f"Precision: {precision:.3f}, Recall: {recall:.3f}"
            )

            # Save model
            self.model.save_model(str(self.model_path))
            self.is_trained = True

            return True

        except Exception as e:
            logger.error(f"[ML] Retraining failed: {e}")
            return False


class AdaptiveParameterManager:
    """Dynamically adjust trading parameters based on performance and market conditions"""

    def __init__(self, config_path: str = "../config.json"):
        self.config_path = Path(config_path)
        self.base_config = self._load_config()
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.market_regime_history: List[str] = []
        self.parameter_adjustments: Dict[str, Dict[str, Any]] = {}

        # Adaptive thresholds
        self.min_performance_samples = 10
        self.adjustment_cooldown = 300  # 5 minutes between adjustments
        self.last_adjustment_time = 0

        logger.info("[ADAPTIVE] Parameter manager initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load base configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[ADAPTIVE] Error loading config: {e}")
            return {}

    def record_performance(self, symbol: str, trade_result: Dict[str, Any]):
        """Record trade performance for adaptive adjustments"""

        if symbol not in self.performance_history:
            self.performance_history[symbol] = []

        self.performance_history[symbol].append(trade_result)

        # Keep only recent performance (last 100 trades)
        if len(self.performance_history[symbol]) > 100:
            self.performance_history[symbol] = self.performance_history[symbol][-100:]

        logger.debug(f"[ADAPTIVE] Recorded performance for {symbol}: {len(self.performance_history[symbol])} trades")

    def record_market_regime(self, regime: str):
        """Record current market regime"""

        self.market_regime_history.append(regime)

        # Keep only recent regimes (last 50 observations)
        if len(self.market_regime_history) > 50:
            self.market_regime_history = self.market_regime_history[-50:]

    def get_adaptive_parameters(self, symbol: str, current_regime: str) -> Dict[str, Any]:
        """Get dynamically adjusted parameters based on performance and conditions"""

        base_params = self.base_config.get('scalping', {}).copy()
        if not isinstance(base_params, dict):
            base_params = {}

        # Check if we have enough performance data
        if symbol not in self.performance_history or len(self.performance_history[symbol]) < self.min_performance_samples:
            logger.debug(f"[ADAPTIVE] Using base parameters - insufficient performance data for {symbol}")
            return base_params

        # Check cooldown
        current_time = datetime.now(timezone.utc).timestamp()
        if current_time - self.last_adjustment_time < self.adjustment_cooldown:
            return self.parameter_adjustments.get(symbol, base_params)

        # Calculate performance metrics
        performance = self.performance_history[symbol][-self.min_performance_samples:]
        win_rate = sum(1 for p in performance if p.get('pnl', 0) > 0) / len(performance)
        avg_win = np.mean([p['pnl'] for p in performance if p['pnl'] > 0]) if any(p['pnl'] > 0 for p in performance) else 0
        avg_loss = abs(np.mean([p['pnl'] for p in performance if p['pnl'] < 0])) if any(p['pnl'] < 0 for p in performance) else 0
        profit_factor = (avg_win * win_rate) / (avg_loss * (1 - win_rate)) if avg_loss > 0 else float('inf')

        # Adjust parameters based on performance
        adjusted_params = base_params.copy()

        # Adjust confidence threshold based on win rate
        if win_rate > 0.65:  # Good performance
            adjusted_params['min_confidence'] = min(0.8, base_params.get('min_confidence', 0.65) + 0.05)
            adjusted_params['trailing_stop_activation'] = max(0.5, base_params.get('trailing_stop_activation', 1.0) - 0.1)
        elif win_rate < 0.45:  # Poor performance
            adjusted_params['min_confidence'] = max(0.7, base_params.get('min_confidence', 0.65) + 0.05)
            adjusted_params['trailing_stop_activation'] = min(1.5, base_params.get('trailing_stop_activation', 1.0) + 0.1)

        # Adjust position sizing based on profit factor
        if profit_factor > 1.5:  # Good risk-reward
            adjusted_params['risk_per_trade'] = min(0.03, base_params.get('risk_per_trade', 0.02) + 0.005)
        elif profit_factor < 0.8:  # Poor risk-reward
            adjusted_params['risk_per_trade'] = max(0.01, base_params.get('risk_per_trade', 0.02) - 0.005)

        # Adjust scale-out levels based on market regime
        regime_multipliers = {
            'LOW_VOLATILITY': 0.9,    # Tighter exits in low vol
            'NORMAL_VOLATILITY': 1.0, # Baseline
            'HIGH_VOLATILITY': 1.1,   # Wider exits in high vol
            'EXTREME_VOLATILITY': 1.2 # Much wider exits in extreme vol
        }

        regime_multiplier = regime_multipliers.get(current_regime, 1.0)
        base_levels = base_params.get('scale_out_levels', [1.0, 2.0, 3.0])
        adjusted_params['scale_out_levels'] = [level * regime_multiplier for level in base_levels]

        # Adjust time-based exits based on recent performance
        recent_performance = performance[-5:]  # Last 5 trades
        recent_win_rate = sum(1 for p in recent_performance if p.get('pnl', 0) > 0) / len(recent_performance)

        if recent_win_rate < 0.4:  # Recent poor performance
            # Reduce holding time to cut losses faster
            base_max_bars = base_params.get('max_bars_held', {})
            adjusted_params['max_bars_held'] = {}
            for timeframe, bars in base_max_bars.items():
                adjusted_params['max_bars_held'][timeframe] = max(2, int(bars * 0.8))

        self.parameter_adjustments[symbol] = adjusted_params
        self.last_adjustment_time = current_time

        logger.info(
            f"[ADAPTIVE] {symbol} | Win Rate: {win_rate:.1%} | Profit Factor: {profit_factor:.2f} | "
            f"Regime: {current_regime} | Params adjusted"
        )

        return adjusted_params

    def get_regime_based_parameters(self, regime: str) -> Dict[str, Any]:
        """Get parameters optimized for specific market regime"""

        base_params = self.base_config.get('scalping', {})

        regime_params = {
            'LOW_VOLATILITY': {
                'min_confidence': base_params.get('min_confidence', 0.65) * 0.9,  # Lower threshold
                'trailing_stop_activation': base_params.get('trailing_stop_activation', 1.0) * 0.8,  # Tighter trailing
                'scale_out_levels': [level * 0.9 for level in base_params.get('scale_out_levels', [1.0, 2.0, 3.0])],
                'max_bars_held': {tf: max(2, int(bars * 0.8)) for tf, bars in base_params.get('max_bars_held', {}).items()}
            },
            'NORMAL_VOLATILITY': base_params,  # Use base parameters
            'HIGH_VOLATILITY': {
                'min_confidence': base_params.get('min_confidence', 0.65) * 1.1,  # Higher threshold
                'trailing_stop_activation': base_params.get('trailing_stop_activation', 1.0) * 1.2,  # Looser trailing
                'scale_out_levels': [level * 1.2 for level in base_params.get('scale_out_levels', [1.0, 2.0, 3.0])],
                'max_bars_held': {tf: int(bars * 1.3) for tf, bars in base_params.get('max_bars_held', {}).items()}
            },
            'EXTREME_VOLATILITY': {
                'min_confidence': base_params.get('min_confidence', 0.65) * 1.3,  # Much higher threshold
                'trailing_stop_activation': base_params.get('trailing_stop_activation', 1.0) * 1.5,  # Much looser trailing
                'scale_out_levels': [level * 1.5 for level in base_params.get('scale_out_levels', [1.0, 2.0, 3.0])],
                'max_bars_held': {tf: int(bars * 0.7) for tf, bars in base_params.get('max_bars_held', {}).items()}  # Shorter holds
            }
        }

        return regime_params.get(regime, base_params)

    def reset_performance_history(self, symbol: str):
        """Reset performance history for a symbol"""
        if symbol in self.performance_history:
            self.performance_history[symbol] = []
            logger.info(f"[ADAPTIVE] Reset performance history for {symbol}")


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

    def detect_hammer_reversal(
        self,
        df: pd.DataFrame,
        current_price: float,
        atr_value: float,
        spread_pips: float
    ) -> Optional[ScalpingSetup]:
        """
        Hammer/Shooting Star Reversal Detection
        Pattern: Small body with long wick, indicates potential reversal

        Args:
            df: OHLC dataframe
            current_price: Latest price
            atr_value: Current ATR
            spread_pips: Current spread

        Returns:
            ScalpingSetup if hammer/shooting star detected
        """
        if len(df) < 5:
            return None

        df = df.copy()
        latest = df.iloc[-1]

        body_size = abs(latest['close'] - latest['open'])
        upper_wick = latest['high'] - max(latest['open'], latest['close'])
        lower_wick = min(latest['open'], latest['close']) - latest['low']
        total_range = latest['high'] - latest['low']

        # Hammer: Long lower wick, small body, bullish close
        if (lower_wick > body_size * 2 and
            lower_wick > total_range * 0.6 and
            latest['close'] > latest['open'] and
            upper_wick < body_size):

            return ScalpingSetup(
                setup_type='HAMMER_REVERSAL_LONG',
                direction=1,
                confidence=0.72,
                entry_price=current_price,
                stop_loss_pips=max(lower_wick / 10**4 * 10000, 2.5),
                target_pips=3.5,
                order_flow_signal=OrderFlowSignal.HAMMER_REVERSAL,
                volatility_regime=self.detect_market_regime(df, atr_value, spread_pips),
                atr_value=atr_value,
                spread_pips=spread_pips
            )

        # Shooting Star: Long upper wick, small body, bearish close
        if (upper_wick > body_size * 2 and
            upper_wick > total_range * 0.6 and
            latest['close'] < latest['open'] and
            lower_wick < body_size):

            return ScalpingSetup(
                setup_type='SHOOTING_STAR_SHORT',
                direction=-1,
                confidence=0.72,
                entry_price=current_price,
                stop_loss_pips=max(upper_wick / 10**4 * 10000, 2.5),
                target_pips=3.5,
                order_flow_signal=OrderFlowSignal.HAMMER_REVERSAL,
                volatility_regime=self.detect_market_regime(df, atr_value, spread_pips),
                atr_value=atr_value,
                spread_pips=spread_pips
            )

        return None

    def detect_engulfing_breakout(
        self,
        df: pd.DataFrame,
        current_price: float,
        atr_value: float,
        spread_pips: float
    ) -> Optional[ScalpingSetup]:
        """
        Engulfing Candle Breakout Detection
        Pattern: Larger candle completely engulfs previous candle

        Args:
            df: OHLC dataframe
            current_price: Latest price
            atr_value: Current ATR
            spread_pips: Current spread

        Returns:
            ScalpingSetup if engulfing breakout detected
        """
        if len(df) < 3:
            return None

        df = df.copy()
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Bullish engulfing: Current candle opens below prev close, closes above prev open
        if (latest['open'] < prev['close'] and
            latest['close'] > prev['open'] and
            latest['close'] > latest['open'] and
            abs(latest['close'] - latest['open']) > abs(prev['close'] - prev['open']) * 1.5):

            return ScalpingSetup(
                setup_type='ENGULFING_BREAKOUT_LONG',
                direction=1,
                confidence=0.75,
                entry_price=current_price,
                stop_loss_pips=max(atr_value / 10**4 * 10000, 3.0),
                target_pips=4.0,
                order_flow_signal=OrderFlowSignal.ENGULFING_BREAK,
                volatility_regime=self.detect_market_regime(df, atr_value, spread_pips),
                atr_value=atr_value,
                spread_pips=spread_pips
            )

        # Bearish engulfing: Current candle opens above prev close, closes below prev open
        if (latest['open'] > prev['close'] and
            latest['close'] < prev['open'] and
            latest['close'] < latest['open'] and
            abs(latest['close'] - latest['open']) > abs(prev['close'] - prev['open']) * 1.5):

            return ScalpingSetup(
                setup_type='ENGULFING_BREAKOUT_SHORT',
                direction=-1,
                confidence=0.75,
                entry_price=current_price,
                stop_loss_pips=max(atr_value / 10**4 * 10000, 3.0),
                target_pips=4.0,
                order_flow_signal=OrderFlowSignal.ENGULFING_BREAK,
                volatility_regime=self.detect_market_regime(df, atr_value, spread_pips),
                atr_value=atr_value,
                spread_pips=spread_pips
            )

        return None

    def detect_fractal_breakout(
        self,
        df: pd.DataFrame,
        current_price: float,
        atr_value: float,
        spread_pips: float
    ) -> Optional[ScalpingSetup]:
        """
        Williams Fractal Breakout Detection
        Pattern: Price breaks above/below Williams fractal level

        Args:
            df: OHLC dataframe
            current_price: Latest price
            atr_value: Current ATR
            spread_pips: Current spread

        Returns:
            ScalpingSetup if fractal breakout detected
        """
        if len(df) < 5:
            return None

        df = df.copy()

        # Calculate Williams fractals (simplified)
        for i in range(2, len(df) - 2):
            # Bullish fractal: High is higher than 2 bars on each side
            if (df.iloc[i]['high'] > df.iloc[i-2]['high'] and
                df.iloc[i]['high'] > df.iloc[i-1]['high'] and
                df.iloc[i]['high'] > df.iloc[i+1]['high'] and
                df.iloc[i]['high'] > df.iloc[i+2]['high']):

                fractal_level = df.iloc[i]['high']
                # Check if current price breaks above this fractal
                if current_price > fractal_level and df.iloc[-1]['close'] > fractal_level:
                    return ScalpingSetup(
                        setup_type='FRACTAL_BREAKOUT_LONG',
                        direction=1,
                        confidence=0.68,
                        entry_price=current_price,
                        stop_loss_pips=max(atr_value / 10**4 * 10000, 2.5),
                        target_pips=3.5,
                        order_flow_signal=OrderFlowSignal.FRACTAL_BREAK,
                        volatility_regime=self.detect_market_regime(df, atr_value, spread_pips),
                        atr_value=atr_value,
                        spread_pips=spread_pips
                    )

            # Bearish fractal: Low is lower than 2 bars on each side
            if (df.iloc[i]['low'] < df.iloc[i-2]['low'] and
                df.iloc[i]['low'] < df.iloc[i-1]['low'] and
                df.iloc[i]['low'] < df.iloc[i+1]['low'] and
                df.iloc[i]['low'] < df.iloc[i+2]['low']):

                fractal_level = df.iloc[i]['low']
                # Check if current price breaks below this fractal
                if current_price < fractal_level and df.iloc[-1]['close'] < fractal_level:
                    return ScalpingSetup(
                        setup_type='FRACTAL_BREAKOUT_SHORT',
                        direction=-1,
                        confidence=0.68,
                        entry_price=current_price,
                        stop_loss_pips=max(atr_value / 10**4 * 10000, 2.5),
                        target_pips=3.5,
                        order_flow_signal=OrderFlowSignal.FRACTAL_BREAK,
                        volatility_regime=self.detect_market_regime(df, atr_value, spread_pips),
                        atr_value=atr_value,
                        spread_pips=spread_pips
                    )

        return None

    def detect_volume_spike(
        self,
        df: pd.DataFrame,
        current_price: float,
        atr_value: float,
        spread_pips: float
    ) -> Optional[ScalpingSetup]:
        """
        Volume Spike Detection
        Pattern: Sudden increase in volume with price movement

        Args:
            df: OHLC dataframe with volume
            current_price: Latest price
            atr_value: Current ATR
            spread_pips: Current spread

        Returns:
            ScalpingSetup if volume spike detected
        """
        if len(df) < 10 or 'volume' not in df.columns:
            return None

        df = df.copy()
        latest = df.iloc[-1]

        # Calculate average volume over last 10 bars
        avg_volume = df['volume'].tail(10).mean()
        volume_ratio = latest['volume'] / avg_volume if avg_volume > 0 else 1.0

        # Volume spike with price action
        if volume_ratio > 2.0:  # 2x average volume
            price_change = latest['close'] - latest['open']
            body_size = abs(price_change)

            # Bullish volume spike
            if price_change > 0 and body_size > atr_value * 0.5:
                return ScalpingSetup(
                    setup_type='VOLUME_SPIKE_LONG',
                    direction=1,
                    confidence=0.70,
                    entry_price=current_price,
                    stop_loss_pips=max(atr_value / 10**4 * 10000, 2.5),
                    target_pips=3.5,
                    order_flow_signal=OrderFlowSignal.VOLUME_SPIKE,
                    volatility_regime=self.detect_market_regime(df, atr_value, spread_pips),
                    atr_value=atr_value,
                    spread_pips=spread_pips
                )

            # Bearish volume spike
            elif price_change < 0 and body_size > atr_value * 0.5:
                return ScalpingSetup(
                    setup_type='VOLUME_SPIKE_SHORT',
                    direction=-1,
                    confidence=0.70,
                    entry_price=current_price,
                    stop_loss_pips=max(atr_value / 10**4 * 10000, 2.5),
                    target_pips=3.5,
                    order_flow_signal=OrderFlowSignal.VOLUME_SPIKE,
                    volatility_regime=self.detect_market_regime(df, atr_value, spread_pips),
                    atr_value=atr_value,
                    spread_pips=spread_pips
                )

        return None


class DynamicExitManager:
    """
    Advanced exit strategies with trailing stops and scaling
    """

    def __init__(self, initial_stop_pips: float, target_pips: float):
        self.initial_stop = initial_stop_pips
        self.target_pips = target_pips
        self.trailing_activated = False
        self.best_price = 0.0
        self.scale_out_levels = []
        self.time_based_exit_bars = 0

    def calculate_dynamic_stop(
        self,
        current_price: float,
        entry_price: float,
        direction: int,
        atr_value: float,
        bars_held: int
    ) -> float:
        """
        Calculate dynamic stop loss with trailing and time-based adjustments

        Args:
            current_price: Current market price
            entry_price: Entry price
            direction: 1 for long, -1 for short
            atr_value: Current ATR value
            bars_held: Number of bars since entry

        Returns:
            Stop loss price
        """
        # Base ATR stop
        atr_stop = atr_value / 10**4 * 10000  # ATR in pips

        if direction == 1:  # Long position
            profit_pips = (current_price - entry_price) / 10**4 * 10000

            # Activate trailing stop after 1R profit
            if profit_pips >= self.target_pips:
                if not self.trailing_activated:
                    self.trailing_activated = True
                    self.best_price = current_price

                # Trail behind best price
                trail_distance = max(atr_stop * 0.8, self.target_pips * 0.5)
                trail_price = self.best_price - (trail_distance / 10000 * 10**4)

                # Update best price
                if current_price > self.best_price:
                    self.best_price = current_price

                return max(trail_price, entry_price + (self.initial_stop / 10000 * 10**4))

            else:
                # Initial stop with time decay
                time_decay = min(0.3, bars_held * 0.05)  # Loosen stop over time
                return entry_price - ((self.initial_stop * (1 + time_decay)) / 10000 * 10**4)

        else:  # Short position
            profit_pips = (entry_price - current_price) / 10**4 * 10000

            # Activate trailing stop after 1R profit
            if profit_pips >= self.target_pips:
                if not self.trailing_activated:
                    self.trailing_activated = True
                    self.best_price = current_price

                # Trail above best price
                trail_distance = max(atr_stop * 0.8, self.target_pips * 0.5)
                trail_price = self.best_price + (trail_distance / 10000 * 10**4)

                # Update best price
                if current_price < self.best_price:
                    self.best_price = current_price

                return min(trail_price, entry_price - (self.initial_stop / 10000 * 10**4))

            else:
                # Initial stop with time decay
                time_decay = min(0.3, bars_held * 0.05)
                return entry_price + ((self.initial_stop * (1 + time_decay)) / 10000 * 10**4)

    def should_scale_out(
        self,
        current_price: float,
        entry_price: float,
        direction: int,
        position_size: float
    ) -> Tuple[bool, float]:
        """
        Determine if position should be partially closed for profit

        Returns:
            (should_scale, size_to_close)
        """
        if direction == 1:
            profit_pips = (current_price - entry_price) / 10**4 * 10000
        else:
            profit_pips = (entry_price - current_price) / 10**4 * 10000

        # Scale out at 1R, 2R, 3R levels
        scale_levels = [1.0, 2.0, 3.0]
        scale_sizes = [0.3, 0.3, 0.4]  # Scale out 30%, 30%, 40%

        for i, level in enumerate(scale_levels):
            if profit_pips >= level * self.target_pips and not self._level_hit(level):
                self.scale_out_levels.append(level)
                return True, position_size * scale_sizes[i]

        return False, 0.0

    def should_time_exit(self, bars_held: int, volatility_regime: str) -> bool:
        """
        Time-based exit based on holding time and market conditions
        """
        # Base time limits by volatility
        time_limits = {
            "LOW_VOLATILITY": 12,      # 1 hour on M5
            "NORMAL_VOLATILITY": 8,    # 40 minutes on M5
            "HIGH_VOLATILITY": 6,      # 30 minutes on M5
            "EXTREME_VOLATILITY": 4    # 20 minutes on M5
        }

        max_bars = time_limits.get(volatility_regime, 8)

        # Exit if held too long
        if bars_held >= max_bars:
            return True

        # Exit if no progress after half time
        if bars_held >= max_bars // 2 and not self.trailing_activated:
            return True

        return False

    def _level_hit(self, level: float) -> bool:
        """Check if scale level already hit"""
        return level in self.scale_out_levels

    def reset(self):
        """Reset exit manager for new trade"""
        self.trailing_activated = False
        self.best_price = 0.0
        self.scale_out_levels = []


class AdvancedRiskManager:
    """
    Enhanced risk management with portfolio-level controls
    """

    def __init__(self):
        self.portfolio_risk_limit = 0.05  # 5% max portfolio risk
        self.correlation_limits = {
            'EURUSD-GBPUSD': 0.8,
            'EURUSD-USDJPY': 0.6,
            'GBPUSD-USDJPY': 0.7
        }
        self.sector_exposure = {}  # Track exposure by currency/sector

    def calculate_portfolio_var(
        self,
        positions: List[Dict[str, Any]],
        volatility_data: Dict[str, float]
    ) -> float:
        """
        Calculate Value at Risk for entire portfolio

        Args:
            positions: List of position dictionaries
            volatility_data: Symbol -> volatility mapping

        Returns:
            Portfolio VaR as percentage
        """
        if not positions:
            return 0.0

        total_risk = 0.0

        for position in positions:
            symbol = position['symbol']
            size = position['size']
            stop_pips = position['stop_pips']

            # Base risk for this position
            position_risk = size * (stop_pips / 10000)  # Risk in price units

            # Adjust for volatility
            vol_multiplier = volatility_data.get(symbol, 1.0)
            adjusted_risk = position_risk * vol_multiplier

            total_risk += adjusted_risk

        # Apply correlation diversification (simplified)
        correlation_discount = 0.85  # Assume 15% diversification benefit
        portfolio_var = total_risk * correlation_discount

        return portfolio_var

    def check_correlation_risk(
        self,
        new_symbol: str,
        existing_positions: List[str]
    ) -> bool:
        """
        Check if adding new symbol exceeds correlation limits

        Returns:
            True if correlation risk acceptable
        """
        for existing in existing_positions:
            pair_key = f"{new_symbol}-{existing}"
            reverse_key = f"{existing}-{new_symbol}"

            correlation_limit = self.correlation_limits.get(
                pair_key,
                self.correlation_limits.get(reverse_key, 0.9)
            )

            # Simplified correlation check - in real implementation,
            # would calculate actual correlation from price data
            if correlation_limit < 0.8:  # High correlation
                return False

        return True

    def update_sector_exposure(
        self,
        symbol: str,
        position_size: float,
        action: str  # 'add' or 'remove'
    ):
        """Track exposure by currency sector"""
        # Extract base currencies
        if symbol.endswith('USD'):
            sector = symbol[:-3]  # EUR, GBP, etc.
        else:
            sector = symbol[:3]   # For JPY pairs

        if action == 'add':
            self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + position_size
        elif action == 'remove':
            self.sector_exposure[sector] = max(0, self.sector_exposure.get(sector, 0) - position_size)

    def get_sector_concentration_penalty(self, symbol: str) -> float:
        """Calculate penalty for over-concentration in sector"""
        if symbol.endswith('USD'):
            sector = symbol[:-3]
        else:
            sector = symbol[:3]

        sector_exposure = self.sector_exposure.get(sector, 0)
        total_exposure = sum(self.sector_exposure.values())

        if total_exposure == 0:
            return 1.0

        concentration = sector_exposure / total_exposure

        # Penalty for >40% concentration in one sector
        if concentration > 0.4:
            return 0.7
        elif concentration > 0.3:
            return 0.85
        else:
            return 1.0


class PerformanceAnalytics:
    """
    Real-time performance tracking and optimization
    """

    def __init__(self):
        self.trade_history = []
        self.setup_performance = {}
        self.time_performance = {}
        self.regime_performance = {}

    def record_trade(
        self,
        symbol: str,
        setup_type: str,
        direction: int,
        entry_time: datetime,
        exit_time: datetime,
        pnl: float,
        bars_held: int,
        volatility_regime: str,
        session: str
    ):
        """Record completed trade for analysis"""
        trade = {
            'symbol': symbol,
            'setup_type': setup_type,
            'direction': direction,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'pnl': pnl,
            'bars_held': bars_held,
            'volatility_regime': volatility_regime,
            'session': session,
            'hour': entry_time.hour,
            'weekday': entry_time.weekday()
        }

        self.trade_history.append(trade)
        self._update_performance_metrics(trade)

    def _update_performance_metrics(self, trade: Dict[str, Any]):
        """Update rolling performance statistics"""
        setup = trade['setup_type']
        regime = trade['volatility_regime']
        session = trade['session']
        hour = trade['hour']

        # Setup performance
        if setup not in self.setup_performance:
            self.setup_performance[setup] = {'wins': 0, 'losses': 0, 'total_pnl': 0}

        if trade['pnl'] > 0:
            self.setup_performance[setup]['wins'] += 1
        else:
            self.setup_performance[setup]['losses'] += 1

        self.setup_performance[setup]['total_pnl'] += trade['pnl']

        # Time-based performance
        time_key = f"{session}_{hour}"
        if time_key not in self.time_performance:
            self.time_performance[time_key] = {'wins': 0, 'losses': 0, 'total_pnl': 0}

        if trade['pnl'] > 0:
            self.time_performance[time_key]['wins'] += 1
        else:
            self.time_performance[time_key]['losses'] += 1

        self.time_performance[time_key]['total_pnl'] += trade['pnl']

        # Regime performance
        if regime not in self.regime_performance:
            self.regime_performance[regime] = {'wins': 0, 'losses': 0, 'total_pnl': 0}

        if trade['pnl'] > 0:
            self.regime_performance[regime]['wins'] += 1
        else:
            self.regime_performance[regime]['losses'] += 1

        self.regime_performance[regime]['total_pnl'] += trade['pnl']

    def get_setup_win_rate(self, setup_type: str) -> float:
        """Get win rate for specific setup type"""
        if setup_type not in self.setup_performance:
            return 0.5

        stats = self.setup_performance[setup_type]
        total = stats['wins'] + stats['losses']
        return stats['wins'] / total if total > 0 else 0.5

    def get_optimal_trading_hours(self) -> List[int]:
        """Identify best performing trading hours"""
        hour_performance = {}

        for time_key, stats in self.time_performance.items():
            if '_' in time_key:
                _, hour_str = time_key.split('_', 1)
                try:
                    hour = int(hour_str)
                    total = stats['wins'] + stats['losses']
                    if total >= 5:  # Minimum sample size
                        win_rate = stats['wins'] / total
                        hour_performance[hour] = win_rate
                except ValueError:
                    continue

        # Return top 6 hours
        sorted_hours = sorted(hour_performance.items(), key=lambda x: x[1], reverse=True)  # type: ignore
        return [hour for hour, _ in sorted_hours[:6]]

    def get_regime_adjustments(self) -> Dict[str, float]:
        """Get performance-based regime adjustments"""
        adjustments = {}

        for regime, stats in self.regime_performance.items():
            total = stats['wins'] + stats['losses']
            if total >= 10:  # Minimum sample size
                win_rate = stats['wins'] / total
                # Adjust confidence based on historical performance
                adjustments[regime] = win_rate - 0.5  # Deviation from 50%

        return adjustments

    def generate_performance_report(self) -> str:
        """Generate detailed performance report"""
        report = ["=== Scalping Performance Report ===\n"]

        # Overall statistics
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in self.trade_history)

        report.append(f"Total Trades: {total_trades}")
        report.append(f"Win Rate: {winning_trades/total_trades:.1%}" if total_trades > 0 else "Win Rate: N/A")
        report.append(f"Total P&L: ${total_pnl:.2f}\n")

        # Setup performance
        report.append("Setup Performance:")
        for setup, stats in sorted(self.setup_performance.items(), key=lambda x: x[1]['total_pnl'], reverse=True):  # type: ignore
            total = stats['wins'] + stats['losses']
            if total > 0:
                win_rate = stats['wins'] / total
                report.append(f"  {setup}: {win_rate:.1%} win rate, ${stats['total_pnl']:.2f} P&L")

        # Best trading hours
        best_hours = self.get_optimal_trading_hours()
        report.append(f"\nBest Trading Hours (UTC): {best_hours}")

        return "\n".join(report)


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
    
    # New enhanced setup detections
    hammer_setup = scalping_engine.detect_hammer_reversal(df, current_price, atr_value, spread_pips)
    if hammer_setup and hammer_setup.direction == ml_signal:
        setups.append(hammer_setup)
    
    engulfing_setup = scalping_engine.detect_engulfing_breakout(df, current_price, atr_value, spread_pips)
    if engulfing_setup and engulfing_setup.direction == ml_signal:
        setups.append(engulfing_setup)
    
    fractal_setup = scalping_engine.detect_fractal_breakout(df, current_price, atr_value, spread_pips)
    if fractal_setup and fractal_setup.direction == ml_signal:
        setups.append(fractal_setup)
    
    volume_setup = scalping_engine.detect_volume_spike(df, current_price, atr_value, spread_pips)
    if volume_setup and volume_setup.direction == ml_signal:
        setups.append(volume_setup)
    
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
