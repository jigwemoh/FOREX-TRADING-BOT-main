#!/usr/bin/env python3
"""
Order Flow Analysis Engine
Real-time tick-level analysis for detecting institutional order patterns,
volume imbalances, and microstructure opportunities.

Key Features:
- Bid/ask volume tracking with imbalance detection
- Volume profile analysis (POC, VA, HVN/LVN)
- Sweep pattern identification
- Cumulative delta and on-balance volume tracking
- Sub-millisecond processing latency
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any, Sequence
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timezone
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ORDER_FLOW_ANALYZER")


class OrderFlowSignal(Enum):
    """Order flow microstructure signals"""
    IMBALANCE_BUY = 1      # Buy volume > sell volume significantly
    IMBALANCE_SELL = 2     # Sell volume > buy volume significantly
    SWEEP_PATTERN = 3      # Large order hitting multiple levels
    ACCUMULATION = 4       # Volume increasing with consolidation
    DISTRIBUTION = 5       # Volume decreasing with consolidation
    MOMENTUM_DIVERGENCE = 6  # Price up but volume weak (or vice versa)
    VOLUME_EXPLOSION = 7    # Extreme volume spike detected
    INSTITUTIONAL_FOOTPRINT = 8  # Large hidden order pattern


@dataclass
class Tick:
    """Represents a single price tick"""
    timestamp: datetime
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last_price: float
    last_size: float


@dataclass
class VolumeNode:
    """Volume aggregated at price level"""
    price: float
    buy_volume: float
    sell_volume: float
    total_volume: float


@dataclass
class VolumeProfile:
    """Complete volume profile for aggregated ticks"""
    ticks_count: int
    price_range: Tuple[float, float]
    point_of_control: float  # Price with most volume
    value_area: Tuple[float, float]  # 70% of volume range
    high_volume_nodes: List[float]  # Top 5 price levels
    low_volume_nodes: List[float]  # Bottom 5 price levels
    total_volume: float
    buy_volume: float
    sell_volume: float
    imbalance_ratio: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OrderFlowMetrics:
    """Real-time order flow metrics"""
    cumulative_delta: float
    imbalance_ratio: float
    buy_volume_total: float
    sell_volume_total: float
    bid_ask_ratio: float
    momentum_strength: float  # -1 to +1 (sell to buy)
    volume_trend: float  # -1 to +1 (decreasing to increasing)
    institutional_probability: float  # 0-1 likelihood of institutional activity


class RingBuffer:
    """
    Fast circular buffer for O(1) tick storage and retrieval.
    Prevents memory allocation overhead and GC pauses.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.size = 0

    def append(self, tick: Tick):
        """Add tick to buffer in O(1) time"""
        self.buffer[self.head] = tick
        self.head = (self.head + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def get_all(self) -> Sequence[Optional[Tick]]:
        """Get all ticks in chronological order"""
        if self.size == 0:
            return []
        
        if self.size < self.capacity:
            return self.buffer[:self.size]
        
        # Reconstruct order from circular buffer
        return self.buffer[self.head:] + self.buffer[:self.head]

    def get_last(self, n: int) -> Sequence[Optional[Tick]]:
        """Get last n ticks"""
        n = min(n, self.size)
        if n == 0:
            return []
        
        start = (self.head - n) % self.capacity
        if start + n <= self.capacity:
            return self.buffer[start:start + n]
        else:
            return self.buffer[start:] + self.buffer[:n - (self.capacity - start)]

    def __len__(self):
        return self.size


class OrderFlowAnalyzer:
    """
    Core order flow analysis with tick-level processing.
    Processes ticks in <1ms and generates order flow signals.
    """

    def __init__(
        self,
        symbol: str,
        tick_buffer_size: int = 1000,
        imbalance_buy_threshold: float = 1.5,
        imbalance_sell_threshold: float = 0.67,
        volume_profile_bins: int = 100,
        sweep_volume_multiplier: float = 2.0
    ):
        self.symbol = symbol
        self.tick_buffer = RingBuffer(tick_buffer_size)

        # Configuration
        self.imbalance_buy_threshold = imbalance_buy_threshold
        self.imbalance_sell_threshold = imbalance_sell_threshold
        self.volume_profile_bins = volume_profile_bins
        self.sweep_volume_multiplier = sweep_volume_multiplier

        # Order flow metrics
        self.bid_volume = 0.0
        self.ask_volume = 0.0
        self.cumulative_delta = 0.0
        self.imbalance_ratio = 1.0
        self.on_balance_volume = 0.0
        self.last_close = 0.0
        self.last_volume = 0.0

        # Institutional detection
        self.volume_sum_20 = deque(maxlen=20)
        self.sweep_candidates: Dict[float, float] = {}  # price -> volume
        self.institutional_score = 0.0

        logger.info(f"[INIT] Order flow analyzer for {symbol} with {tick_buffer_size} tick buffer")

    def process_tick(self, tick: Tick) -> List[OrderFlowSignal]:
        """
        Process single tick and detect order flow signals.
        Must complete in <1ms.

        Args:
            tick: Individual price tick with volumes

        Returns:
            List of detected signals (empty if no signals)
        """
        signals = []

        # Add to buffer
        self.tick_buffer.append(tick)

        # Update volumes
        if tick.bid_size > 0:
            self.bid_volume += tick.bid_size
        if tick.ask_size > 0:
            self.ask_volume += tick.ask_size

        # Calculate metrics
        self._update_metrics(tick)

        # Detect signals
        if imbalance_signal := self._detect_imbalance():
            signals.append(imbalance_signal)

        if sweep_signal := self._detect_sweep_pattern(tick):
            signals.append(sweep_signal)

        if momentum_signal := self._detect_momentum_divergence():
            signals.append(momentum_signal)

        if volume_signal := self._detect_volume_explosion():
            signals.append(volume_signal)

        if inst_signal := self._detect_institutional_footprint():
            signals.append(inst_signal)

        return signals

    def _update_metrics(self, tick: Tick):
        """Update all order flow metrics from tick data"""
        # Imbalance ratio
        total_volume = self.bid_volume + self.ask_volume
        if total_volume > 0:
            self.imbalance_ratio = self.bid_volume / max(self.ask_volume, 0.001)
            self.cumulative_delta = self.bid_volume - self.ask_volume
        else:
            self.imbalance_ratio = 1.0

        # On-balance volume
        if tick.last_price > self.last_close:
            self.on_balance_volume += tick.last_size
        elif tick.last_price < self.last_close:
            self.on_balance_volume -= tick.last_size

        self.last_close = tick.last_price
        self.last_volume = tick.last_size

        # Track volume for institutional detection
        self.volume_sum_20.append(tick.last_size)

    def _detect_imbalance(self) -> Optional[OrderFlowSignal]:
        """
        Detect significant buy/sell imbalance.
        Buy imbalance: bid_volume / ask_volume > threshold
        """
        if self.imbalance_ratio > self.imbalance_buy_threshold:
            return OrderFlowSignal.IMBALANCE_BUY
        elif self.imbalance_ratio < self.imbalance_sell_threshold:
            return OrderFlowSignal.IMBALANCE_SELL

        return None

    def _detect_sweep_pattern(self, tick: Tick) -> Optional[OrderFlowSignal]:
        """
        Detect large orders hitting multiple price levels (sweep pattern).
        Indicates institutional buying/selling.
        """
        if tick.last_size < 1.0:  # Ignore tiny sizes
            return None

        avg_volume = np.mean(list(self.volume_sum_20)) if self.volume_sum_20 else 0
        if avg_volume == 0:
            return None

        # Spike detection
        if tick.last_size > (avg_volume * self.sweep_volume_multiplier):
            # Track price levels for multi-level pattern
            self.sweep_candidates[tick.last_price] = tick.last_size

            # Check if we have multiple levels (indicates sweep)
            if len(self.sweep_candidates) >= 3:
                total_sweep_volume = sum(self.sweep_candidates.values())
                if total_sweep_volume > (avg_volume * 3):
                    self.sweep_candidates.clear()
                    return OrderFlowSignal.SWEEP_PATTERN

        return None

    def _detect_momentum_divergence(self) -> Optional[OrderFlowSignal]:
        """
        Detect momentum divergence: price trending but volume weakening.
        Often precedes reversal.
        """
        if len(self.volume_sum_20) < 10:
            return None

        # Recent volume trend
        recent_volumes = list(self.volume_sum_20)[-5:]
        avg_recent = np.mean(recent_volumes)
        avg_older = np.mean(list(self.volume_sum_20)[:-5]) if len(self.volume_sum_20) > 5 else avg_recent

        if avg_older == 0:
            return None

        volume_ratio = avg_recent / avg_older

        # Price momentum
        recent_ticks = self.tick_buffer.get_last(5)
        if len(recent_ticks) < 2:
            return None

        price_momentum = (recent_ticks[-1].last_price - recent_ticks[0].last_price) / recent_ticks[0].last_price

        # Divergence: price up but volume down (or vice versa)
        if abs(price_momentum) > 0.001:  # Meaningful price move
            if (price_momentum > 0 and volume_ratio < 0.8) or (price_momentum < 0 and volume_ratio < 0.8):
                return OrderFlowSignal.MOMENTUM_DIVERGENCE

        return None

    def _detect_volume_explosion(self) -> Optional[OrderFlowSignal]:
        """
        Detect extreme volume spike (>3x average).
        Indicates significant institutional activity.
        """
        if len(self.volume_sum_20) < 5:
            return None

        avg_volume = np.mean(list(self.volume_sum_20))
        last_volume = self.last_volume

        if avg_volume > 0 and last_volume > (avg_volume * 3):
            return OrderFlowSignal.VOLUME_EXPLOSION

        return None

    def _detect_institutional_footprint(self) -> Optional[OrderFlowSignal]:
        """
        Detect institutional activity patterns:
        - Large orders at support/resistance
        - Accumulation (volume up, price flat)
        - Distribution (volume down, price flat)
        """
        # Check for accumulation/distribution pattern
        recent_ticks = self.tick_buffer.get_last(10)
        if len(recent_ticks) < 5:
            return None

        price_range = max(t.last_price for t in recent_ticks) - min(t.last_price for t in recent_ticks)
        
        # Price is relatively flat
        if price_range < (recent_ticks[-1].last_price * 0.002):  # <0.2% range
            volume_trend = self.on_balance_volume
            
            # OBV increasing = accumulation
            if volume_trend > 0:
                self.institutional_score = min(1.0, self.institutional_score + 0.2)
                if self.institutional_score > 0.5:
                    return OrderFlowSignal.ACCUMULATION
            # OBV decreasing = distribution
            elif volume_trend < 0:
                self.institutional_score = max(0.0, self.institutional_score - 0.2)
                if self.institutional_score < -0.5:
                    return OrderFlowSignal.DISTRIBUTION
        else:
            # Reset score if price moves
            self.institutional_score *= 0.9

        return None

    def get_volume_profile(self) -> Optional[VolumeProfile]:
        """
        Build volume profile from buffered ticks.
        Groups volume by price level and identifies key zones.

        Returns:
            VolumeProfile with POC, VA, HVN/LVN
        """
        ticks = self.tick_buffer.get_all()
        if len(ticks) < 1:
            return None

        # Create price bins
        prices = [t.last_price for t in ticks]
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price

        if price_range == 0:
            return None

        # Aggregate volume by price bin
        bin_width = price_range / self.volume_profile_bins
        volume_by_price: Dict[int, VolumeNode] = {}

        for tick in ticks:
            bin_idx = int((tick.last_price - min_price) / bin_width)
            bin_idx = min(bin_idx, self.volume_profile_bins - 1)
            bin_price = min_price + (bin_idx * bin_width)

            if bin_idx not in volume_by_price:
                volume_by_price[bin_idx] = VolumeNode(
                    price=bin_price,
                    buy_volume=0.0,
                    sell_volume=0.0,
                    total_volume=0.0
                )

            # Estimate buy/sell based on price movement
            node = volume_by_price[bin_idx]
            if tick.bid_size > tick.ask_size:
                node.buy_volume += tick.bid_size
                node.sell_volume += tick.ask_size
            else:
                node.buy_volume += tick.ask_size
                node.sell_volume += tick.bid_size
            node.total_volume += tick.bid_size + tick.ask_size

        if not volume_by_price:
            return None

        # Find POC (price with most volume)
        poc_bin = max(volume_by_price.keys(), key=lambda x: volume_by_price[x].total_volume)
        poc_price = volume_by_price[poc_bin].price

        # Find value area (70% of volume)
        sorted_bins = sorted(volume_by_price.keys(), key=lambda x: volume_by_price[x].total_volume, reverse=True)
        total_volume = sum(node.total_volume for node in volume_by_price.values())
        cumulative_volume = 0
        va_bins = []

        for bin_idx in sorted_bins:
            cumulative_volume += volume_by_price[bin_idx].total_volume
            va_bins.append(bin_idx)
            if cumulative_volume >= total_volume * 0.7:
                break

        va_min = min(volume_by_price[b].price for b in va_bins)
        va_max = max(volume_by_price[b].price for b in va_bins)

        # High/Low volume nodes (top and bottom 5)
        top_5_bins = sorted_bins[:5]
        bottom_5_bins = sorted_bins[-5:] if len(sorted_bins) > 5 else []

        hvn = [volume_by_price[b].price for b in top_5_bins]
        lvn = [volume_by_price[b].price for b in bottom_5_bins]

        buy_total = sum(node.buy_volume for node in volume_by_price.values())
        sell_total = sum(node.sell_volume for node in volume_by_price.values())
        imbalance = buy_total / max(sell_total, 0.001)

        return VolumeProfile(
            ticks_count=len(ticks),
            price_range=(min_price, max_price),
            point_of_control=poc_price,
            value_area=(va_min, va_max),
            high_volume_nodes=hvn,
            low_volume_nodes=lvn,
            total_volume=total_volume,
            buy_volume=buy_total,
            sell_volume=sell_total,
            imbalance_ratio=imbalance
        )

    def get_metrics(self) -> OrderFlowMetrics:
        """Get current order flow metrics"""
        recent_ticks = self.tick_buffer.get_last(20)
        
        if len(recent_ticks) > 1:
            price_change = recent_ticks[-1].last_price - recent_ticks[0].last_price
            momentum = 1.0 if price_change > 0 else -1.0 if price_change < 0 else 0.0
        else:
            momentum = 0.0

        # Volume trend (increasing or decreasing)
        if len(self.volume_sum_20) > 1:
            volume_trend = np.mean(list(self.volume_sum_20)[-5:]) - np.mean(list(self.volume_sum_20)[:-5])
            volume_trend = np.tanh(volume_trend / np.mean(list(self.volume_sum_20)))  # Normalize to -1 to 1
        else:
            volume_trend = 0.0

        return OrderFlowMetrics(
            cumulative_delta=self.cumulative_delta,
            imbalance_ratio=self.imbalance_ratio,
            buy_volume_total=self.bid_volume,
            sell_volume_total=self.ask_volume,
            bid_ask_ratio=self.bid_volume / max(self.ask_volume, 0.001),
            momentum_strength=momentum,
            volume_trend=volume_trend,
            institutional_probability=max(0.0, min(1.0, abs(self.institutional_score)))
        )

    def reset_cycle(self):
        """Reset metrics for new analysis cycle"""
        self.bid_volume = 0.0
        self.ask_volume = 0.0
        self.cumulative_delta = 0.0
        self.imbalance_ratio = 1.0
        self.sweep_candidates.clear()

        logger.debug(f"[{self.symbol}] Reset order flow metrics")


class TickAggregator:
    """
    Aggregate individual ticks into micro-candles for faster analysis.
    Supports volume-based, time-based, and imbalance-based aggregation.
    """

    def __init__(
        self,
        symbol: str,
        aggregation_type: str = "volume",
        aggregation_threshold: int = 10,
        time_threshold_ms: int = 100
    ):
        self.symbol = symbol
        self.aggregation_type = aggregation_type
        self.aggregation_threshold = aggregation_threshold
        self.time_threshold_ms = time_threshold_ms

        self.tick_buffer: List[Tick] = []
        self.aggregated_candles: deque[Dict[str, Any]] = deque(maxlen=100)

    def add_tick(self, tick: Tick) -> Optional[Dict[str, Any]]:
        """
        Add tick and return aggregated candle if threshold reached.

        Returns:
            Candle dict if aggregation threshold met, None otherwise
        """
        self.tick_buffer.append(tick)

        should_aggregate = False

        if self.aggregation_type == "volume":
            # Volume-based aggregation
            total_volume = sum(t.bid_size + t.ask_size for t in self.tick_buffer)
            should_aggregate = total_volume >= self.aggregation_threshold

        elif self.aggregation_type == "count":
            # Tick-count aggregation
            should_aggregate = len(self.tick_buffer) >= self.aggregation_threshold

        elif self.aggregation_type == "time":
            # Time-based aggregation
            if len(self.tick_buffer) > 1:
                time_diff = (self.tick_buffer[-1].timestamp - self.tick_buffer[0].timestamp).total_seconds() * 1000
                should_aggregate = time_diff >= self.time_threshold_ms

        if should_aggregate and self.tick_buffer:
            candle = self._aggregate_ticks()
            self.aggregated_candles.append(candle)
            self.tick_buffer.clear()
            return candle

        return None

    def _aggregate_ticks(self) -> Dict[str, Any]:
        """Convert buffer of ticks into single candle"""
        if not self.tick_buffer:
            return {}

        prices = [t.last_price for t in self.tick_buffer]
        volumes = [t.last_size for t in self.tick_buffer]

        return {
            'timestamp': self.tick_buffer[-1].timestamp,
            'open': self.tick_buffer[0].last_price,
            'high': max(prices),
            'low': min(prices),
            'close': self.tick_buffer[-1].last_price,
            'volume': sum(volumes),
            'tick_count': len(self.tick_buffer),
            'bid_volume': sum(t.bid_size for t in self.tick_buffer),
            'ask_volume': sum(t.ask_size for t in self.tick_buffer)
        }

    def get_last_candles(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get last n aggregated candles"""
        return list(self.aggregated_candles)[-n:]
