#!/usr/bin/env python3
"""
Order Flow Signal Generator
Combines order flow signals with microstructure analysis and ML predictions
to generate high-confidence trading signals.

Key Features:
- 6 core order flow signal types with confidence scoring
- Confluence filtering (cross-timeframe signal validation)
- Regime-aware signal weighting
- Integration with Phase 1 microstructure and Phase 2 ML quality
"""

import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timezone
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ORDER_FLOW_SIGNAL_GENERATOR")


class OrderFlowSignalType(Enum):
    """Core order flow signal types"""
    IMBALANCE_BUY = "IMBALANCE_BUY"
    IMBALANCE_SELL = "IMBALANCE_SELL"
    SWEEP_PATTERN = "SWEEP_PATTERN"
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    MOMENTUM_DIVERGENCE = "MOMENTUM_DIVERGENCE"
    VOLUME_EXPLOSION = "VOLUME_EXPLOSION"
    INSTITUTIONAL_FOOTPRINT = "INSTITUTIONAL_FOOTPRINT"


class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING = "TRENDING"           # Consistent directional order flow
    RANGING = "RANGING"             # Oscillating order flow
    INSTITUTIONAL_ACTIVITY = "INSTITUTIONAL_ACTIVITY"  # Large hidden orders
    LOW_LIQUIDITY = "LOW_LIQUIDITY"  # Insufficient order flow


@dataclass
class OrderFlowSignal:
    """Complete order flow signal with all context"""
    signal_type: OrderFlowSignalType
    symbol: str
    direction: int  # 1 for buy, -1 for sell
    confidence: float  # 0-1, from order flow analysis
    imbalance_ratio: float  # Buy/sell ratio
    momentum_aligned: bool  # Does price follow volume?
    institutional_score: float  # 0-1 institutional activity probability
    regime: MarketRegime
    timestamp: datetime
    
    # Integration metrics
    microstructure_score: float = 0.0  # From Phase 1 setup detection
    ml_quality_score: float = 0.0  # From Phase 2 ML predictor
    confluence_level: int = 1  # 1-3 (single to triple timeframe confirmation)
    
    # Final blended confidence
    final_confidence: float = 0.0


@dataclass
class ConfluenceScore:
    """Confluence analysis result"""
    tick_level_signal: bool
    one_sec_signal: bool
    five_sec_signal: bool
    total_confirmations: int
    confluence_strength: float  # 0-1


class OrderFlowSignalGenerator:
    """
    Generate order flow signals with multi-layer confluence checking.
    Blends order flow with microstructure and ML predictions.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol

        # Signal weights (configurable)
        self.signal_weights = {
            OrderFlowSignalType.IMBALANCE_BUY: 0.40,
            OrderFlowSignalType.IMBALANCE_SELL: 0.40,
            OrderFlowSignalType.SWEEP_PATTERN: 0.30,
            OrderFlowSignalType.ACCUMULATION: 0.25,
            OrderFlowSignalType.DISTRIBUTION: 0.25,
            OrderFlowSignalType.MOMENTUM_DIVERGENCE: 0.35,
            OrderFlowSignalType.VOLUME_EXPLOSION: 0.45,
            OrderFlowSignalType.INSTITUTIONAL_FOOTPRINT: 0.50
        }

        # Regime weights
        self.regime_weights = {
            MarketRegime.TRENDING: 1.0,
            MarketRegime.RANGING: 0.7,
            MarketRegime.INSTITUTIONAL_ACTIVITY: 1.2,
            MarketRegime.LOW_LIQUIDITY: 0.4
        }

        # Confluence boost multipliers
        self.confluence_multipliers = {
            1: 1.0,  # Single timeframe
            2: 1.2,  # Dual confirmation
            3: 1.5   # Triple confirmation
        }

        self.recent_signals: List[OrderFlowSignal] = []

        logger.info(f"[INIT] Order flow signal generator for {symbol}")

    def generate_signal(
        self,
        signal_type: OrderFlowSignalType,
        imbalance_ratio: float,
        momentum_aligned: bool,
        institutional_score: float,
        regime: MarketRegime,
        microstructure_score: float = 0.0,
        ml_quality_score: float = 0.0,
        confluence_level: int = 1
    ) -> OrderFlowSignal:
        """
        Generate complete order flow signal.

        Args:
            signal_type: Type of order flow signal detected
            imbalance_ratio: Buy/sell volume ratio (>1 = buy bias)
            momentum_aligned: Does price movement confirm volume?
            institutional_score: Institutional activity probability (0-1)
            regime: Current market regime
            microstructure_score: Phase 1 setup quality (0-1)
            ml_quality_score: Phase 2 ML quality prediction (0-1)
            confluence_level: 1-3 for cross-timeframe confirmation

        Returns:
            Complete OrderFlowSignal with blended confidence
        """
        
        # Determine direction from signal type and imbalance
        if signal_type in [OrderFlowSignalType.IMBALANCE_BUY, OrderFlowSignalType.ACCUMULATION]:
            direction = 1
        elif signal_type in [OrderFlowSignalType.IMBALANCE_SELL, OrderFlowSignalType.DISTRIBUTION]:
            direction = -1
        else:
            # For mixed signals, use imbalance ratio
            direction = 1 if imbalance_ratio > 1.0 else -1

        # Calculate base confidence from order flow
        order_flow_confidence = self._calculate_order_flow_confidence(
            signal_type,
            imbalance_ratio,
            momentum_aligned,
            institutional_score
        )

        # Apply regime weighting
        regime_weight = self.regime_weights.get(regime, 1.0)
        regime_adjusted_confidence = order_flow_confidence * regime_weight

        # Apply confluence boost
        confluence_multiplier = self.confluence_multipliers.get(confluence_level, 1.0)
        confluence_boosted_confidence = min(1.0, regime_adjusted_confidence * confluence_multiplier)

        # Blend with microstructure and ML scores
        final_confidence = self._blend_signals(
            confluence_boosted_confidence,
            microstructure_score,
            ml_quality_score
        )

        signal = OrderFlowSignal(
            signal_type=signal_type,
            symbol=self.symbol,
            direction=direction,
            confidence=order_flow_confidence,
            imbalance_ratio=imbalance_ratio,
            momentum_aligned=momentum_aligned,
            institutional_score=institutional_score,
            regime=regime,
            timestamp=datetime.now(timezone.utc),
            microstructure_score=microstructure_score,
            ml_quality_score=ml_quality_score,
            confluence_level=confluence_level,
            final_confidence=final_confidence
        )

        self.recent_signals.append(signal)
        if len(self.recent_signals) > 1000:
            self.recent_signals = self.recent_signals[-1000:]

        return signal

    def _calculate_order_flow_confidence(
        self,
        signal_type: OrderFlowSignalType,
        imbalance_ratio: float,
        momentum_aligned: bool,
        institutional_score: float
    ) -> float:
        """
        Calculate confidence purely from order flow metrics.
        Range: 0.0 to 1.0
        """
        
        base_score = 0.5

        # Imbalance strength
        if signal_type in [OrderFlowSignalType.IMBALANCE_BUY, OrderFlowSignalType.IMBALANCE_SELL]:
            # Convert ratio to 0-1 score
            # 1.0:1 = 0.5, 1.5:1 = 0.7, 2.0:1 = 0.85
            ratio_normalized = (imbalance_ratio - 1.0) / (2.0 - 1.0)
            base_score = 0.5 + (ratio_normalized * 0.35)

        # Momentum alignment boost
        if momentum_aligned:
            base_score = min(1.0, base_score + 0.15)
        else:
            base_score = max(0.3, base_score - 0.2)

        # Institutional score boost
        base_score = base_score * 0.8 + (institutional_score * 0.2)

        # Signal-specific adjustments
        signal_weight = self.signal_weights.get(signal_type, 0.3)
        base_score = base_score * (signal_weight / 0.5)  # Normalize to 0.5 baseline

        return min(1.0, max(0.1, base_score))  # Clamp to 0.1-1.0

    def _blend_signals(
        self,
        order_flow_score: float,
        microstructure_score: float,
        ml_quality_score: float
    ) -> float:
        """
        Blend order flow with microstructure and ML scores.
        Weights: OF=0.50, Microstructure=0.30, ML=0.20
        """
        
        blended = (
            order_flow_score * 0.50 +
            microstructure_score * 0.30 +
            ml_quality_score * 0.20
        )

        return min(1.0, max(0.1, blended))

    def detect_regime(
        self,
        imbalance_ratio: float,
        volume_trend: float,
        volatility: float,
        recent_signals_count: int
    ) -> MarketRegime:
        """
        Classify market regime based on order flow patterns.

        Args:
            imbalance_ratio: Current bid/ask ratio
            volume_trend: Volume increasing (>0) or decreasing (<0)
            volatility: Current market volatility (0-1)
            recent_signals_count: Count of signals in last minute

        Returns:
            MarketRegime classification
        """
        
        # Check for institutional activity
        if 0.9 < imbalance_ratio < 1.1 and volume_trend > 0.3:
            return MarketRegime.INSTITUTIONAL_ACTIVITY

        # Check for trending regime
        if abs(imbalance_ratio - 1.0) > 0.15 or recent_signals_count > 5:
            return MarketRegime.TRENDING

        # Check for low liquidity
        if volume_trend < -0.3 or recent_signals_count < 1:
            return MarketRegime.LOW_LIQUIDITY

        # Default: ranging
        return MarketRegime.RANGING

    def get_signal_performance(self) -> Dict[str, Any]:
        """
        Analyze performance of generated signals.
        """
        if not self.recent_signals:
            return {}

        signals_df = pd.DataFrame([
            {
                'type': s.signal_type.value,
                'confidence': s.final_confidence,
                'regime': s.regime.value,
                'timestamp': s.timestamp
            }
            for s in self.recent_signals
        ])

        return {
            'total_signals': len(self.recent_signals),
            'avg_confidence': signals_df['confidence'].mean(),
            'max_confidence': signals_df['confidence'].max(),
            'by_type': signals_df.groupby('type').size().to_dict(),
            'by_regime': signals_df.groupby('regime').size().to_dict()
        }


class ConfluenceAnalyzer:
    """
    Analyze signal confluence across timeframes.
    Higher confluence = higher probability signal.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.tick_signals: deque[OrderFlowSignal] = deque(maxlen=100)
        self.one_sec_signals: deque[OrderFlowSignal] = deque(maxlen=100)
        self.five_sec_signals: deque[OrderFlowSignal] = deque(maxlen=100)

    def add_signal(
        self,
        signal: OrderFlowSignal,
        timeframe: str
    ):
        """Record signal at specific timeframe"""
        
        if timeframe == 'tick':
            self.tick_signals.append(signal)
        elif timeframe == '1s':
            self.one_sec_signals.append(signal)
        elif timeframe == '5s':
            self.five_sec_signals.append(signal)

        # Keep only recent signals (last 10 per timeframe)
        max_size = 10
        if len(self.tick_signals) > max_size:
            self.tick_signals.pop(0)
        if len(self.one_sec_signals) > max_size:
            self.one_sec_signals.pop(0)
        if len(self.five_sec_signals) > max_size:
            self.five_sec_signals.pop(0)

    def check_confluence(self, signal_type: OrderFlowSignalType, direction: int) -> ConfluenceScore:
        """
        Check if signal is confirmed across multiple timeframes.

        Returns:
            ConfluenceScore with confirmation count and strength
        """
        
        def has_matching_signal(signals: List[OrderFlowSignal]) -> bool:
            """Check if any recent signal matches type and direction"""
            for s in signals[-3:]:  # Check last 3 signals
                if s.signal_type == signal_type and s.direction == direction:
                    return True
            return False

        tick_signal = has_matching_signal(self.tick_signals)
        one_sec_signal = has_matching_signal(self.one_sec_signals)
        five_sec_signal = has_matching_signal(self.five_sec_signals)

        total_confirmations = sum([tick_signal, one_sec_signal, five_sec_signal])
        confluence_strength = total_confirmations / 3.0

        return ConfluenceScore(
            tick_level_signal=tick_signal,
            one_sec_signal=one_sec_signal,
            five_sec_signal=five_sec_signal,
            total_confirmations=total_confirmations,
            confluence_strength=confluence_strength
        )

    def filter_by_confluence(
        self,
        signal: OrderFlowSignal,
        min_confluence: int = 2
    ) -> bool:
        """
        Filter signal based on confluence requirement.

        Args:
            signal: Signal to evaluate
            min_confluence: Minimum timeframe confirmations (1-3)

        Returns:
            True if signal meets confluence requirement
        """
        
        confluence = self.check_confluence(signal.signal_type, signal.direction)
        return confluence.total_confirmations >= min_confluence


class RegimeAdaptiveWeighter:
    """
    Adjust signal weights based on current market regime.
    Different regimes have different optimal signal types.
    """

    def __init__(self):
        self.regime_signal_weights = {
            MarketRegime.TRENDING: {
                OrderFlowSignalType.IMBALANCE_BUY: 1.0,
                OrderFlowSignalType.IMBALANCE_SELL: 1.0,
                OrderFlowSignalType.MOMENTUM_DIVERGENCE: 0.6,
                OrderFlowSignalType.SWEEP_PATTERN: 0.8,
                OrderFlowSignalType.VOLUME_EXPLOSION: 1.1,
                OrderFlowSignalType.ACCUMULATION: 0.7,
                OrderFlowSignalType.DISTRIBUTION: 0.7,
                OrderFlowSignalType.INSTITUTIONAL_FOOTPRINT: 0.9,
            },
            MarketRegime.RANGING: {
                OrderFlowSignalType.IMBALANCE_BUY: 0.9,
                OrderFlowSignalType.IMBALANCE_SELL: 0.9,
                OrderFlowSignalType.MOMENTUM_DIVERGENCE: 1.1,
                OrderFlowSignalType.SWEEP_PATTERN: 0.7,
                OrderFlowSignalType.VOLUME_EXPLOSION: 0.6,
                OrderFlowSignalType.ACCUMULATION: 1.0,
                OrderFlowSignalType.DISTRIBUTION: 1.0,
                OrderFlowSignalType.INSTITUTIONAL_FOOTPRINT: 0.8,
            },
            MarketRegime.INSTITUTIONAL_ACTIVITY: {
                OrderFlowSignalType.IMBALANCE_BUY: 1.1,
                OrderFlowSignalType.IMBALANCE_SELL: 1.1,
                OrderFlowSignalType.MOMENTUM_DIVERGENCE: 0.8,
                OrderFlowSignalType.SWEEP_PATTERN: 1.2,
                OrderFlowSignalType.VOLUME_EXPLOSION: 1.3,
                OrderFlowSignalType.ACCUMULATION: 1.2,
                OrderFlowSignalType.DISTRIBUTION: 1.2,
                OrderFlowSignalType.INSTITUTIONAL_FOOTPRINT: 1.4,
            },
            MarketRegime.LOW_LIQUIDITY: {
                OrderFlowSignalType.IMBALANCE_BUY: 0.5,
                OrderFlowSignalType.IMBALANCE_SELL: 0.5,
                OrderFlowSignalType.MOMENTUM_DIVERGENCE: 0.4,
                OrderFlowSignalType.SWEEP_PATTERN: 0.4,
                OrderFlowSignalType.VOLUME_EXPLOSION: 0.3,
                OrderFlowSignalType.ACCUMULATION: 0.3,
                OrderFlowSignalType.DISTRIBUTION: 0.3,
                OrderFlowSignalType.INSTITUTIONAL_FOOTPRINT: 0.2,
            },
        }

    def get_regime_adjusted_confidence(
        self,
        signal: OrderFlowSignal
    ) -> float:
        """
        Get confidence adjusted for current regime.
        """
        
        regime_weights = self.regime_signal_weights.get(signal.regime, {})
        signal_weight = regime_weights.get(signal.signal_type, 1.0)

        return min(1.0, signal.final_confidence * signal_weight)

    def suggest_strategy_for_regime(self, regime: MarketRegime) -> Dict[str, Any]:
        """
        Suggest trading strategy for current regime.
        """
        
        strategies = {
            MarketRegime.TRENDING: {
                'strategy': 'TREND_FOLLOWING',
                'recommended_signals': [
                    OrderFlowSignalType.IMBALANCE_BUY,
                    OrderFlowSignalType.IMBALANCE_SELL,
                    OrderFlowSignalType.VOLUME_EXPLOSION
                ],
                'position_sizing': 'AGGRESSIVE',
                'stop_loss_pips': 20,
                'take_profit_ratio': 1.5,
            },
            MarketRegime.RANGING: {
                'strategy': 'MEAN_REVERSION',
                'recommended_signals': [
                    OrderFlowSignalType.MOMENTUM_DIVERGENCE,
                    OrderFlowSignalType.ACCUMULATION,
                    OrderFlowSignalType.DISTRIBUTION
                ],
                'position_sizing': 'CONSERVATIVE',
                'stop_loss_pips': 15,
                'take_profit_ratio': 1.2,
            },
            MarketRegime.INSTITUTIONAL_ACTIVITY: {
                'strategy': 'BREAKOUT_FOLLOWING',
                'recommended_signals': [
                    OrderFlowSignalType.SWEEP_PATTERN,
                    OrderFlowSignalType.VOLUME_EXPLOSION,
                    OrderFlowSignalType.INSTITUTIONAL_FOOTPRINT
                ],
                'position_sizing': 'MODERATE_AGGRESSIVE',
                'stop_loss_pips': 25,
                'take_profit_ratio': 1.8,
            },
            MarketRegime.LOW_LIQUIDITY: {
                'strategy': 'WAIT',
                'recommended_signals': [],
                'position_sizing': 'MINIMAL',
                'stop_loss_pips': 10,
                'take_profit_ratio': 0.5,
            },
        }

        return strategies.get(regime, {})
