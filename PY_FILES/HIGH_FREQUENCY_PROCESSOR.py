#!/usr/bin/env python3
"""
High-Frequency Processing System
Real-time async tick processing with sub-millisecond latency optimization.
Enables multi-scale analysis and immediate signal generation.

Key Features:
- Non-blocking async tick event processing
- Ring buffer for O(1) tick storage
- Multi-scale concurrent analysis (tick, 1-sec, 5-sec)
- Cross-timeframe confluence filtering
- Latency monitoring and optimization
"""

import asyncio
import numpy as np
from typing import Dict, Callable, Optional, List, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timezone
from collections import deque
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HIGH_FREQUENCY_PROCESSOR")


@dataclass
class LatencyMetrics:
    """Track processing latency for optimization"""
    tick_arrival_time: float
    processing_start: float
    processing_end: float
    total_latency_ms: float = field(default=0.0)
    queue_depth: int = 0

    def __post_init__(self):
        self.total_latency_ms = (self.processing_end - self.tick_arrival_time) * 1000


@dataclass
class MultiScaleCandle:
    """Candle at specific timeframe"""
    timeframe: str  # 'tick', '1s', '5s'
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float
    sell_volume: float
    tick_count: int


class SignalConfluence(Enum):
    """Confluence level for cross-timeframe signals"""
    SINGLE_TIMEFRAME = 1    # Signal on only one scale
    DUAL_CONFIRMATION = 2   # Confirmed on two scales
    TRIPLE_CONFIRMATION = 3 # Confirmed on all three scales
    CONTRADICTION = -1      # Signals conflict across scales


class AsyncTickHandler:
    """
    Non-blocking async handler for tick events.
    Processes ticks with latency <5ms target.
    """

    def __init__(
        self,
        symbol: str,
        max_queue_size: int = 10000,
        max_latency_ms: float = 100.0,
        enable_profiling: bool = True
    ):
        self.symbol = symbol
        self.tick_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=max_queue_size)
        self.max_latency_ms = max_latency_ms
        self.enable_profiling = enable_profiling

        # Callbacks for different tick processing stages
        self.on_tick_callbacks: List[Callable[..., Any]] = []
        self.on_signal_callbacks: List[Callable[..., Any]] = []
        self.on_error_callbacks: List[Callable[..., Any]] = []

        # Performance tracking
        self.latency_history: deque[float] = deque(maxlen=1000)
        self.processed_ticks = 0
        self.dropped_ticks = 0
        self.exceeded_latency_count = 0

        logger.info(f"[INIT] Async tick handler for {symbol} with {max_queue_size} queue size")

    def register_callback(self, callback: Callable[..., Any], callback_type: str = "tick"):
        """
        Register callback for tick/signal/error events.

        Args:
            callback: Async function to call on event
            callback_type: 'tick', 'signal', or 'error'
        """
        if callback_type == "tick":
            self.on_tick_callbacks.append(callback)
        elif callback_type == "signal":
            self.on_signal_callbacks.append(callback)
        elif callback_type == "error":
            self.on_error_callbacks.append(callback)

    def add_tick(self, tick: Any) -> bool:
        """
        Add tick to processing queue (non-blocking).
        Returns immediately - actual processing happens async.

        Args:
            tick: Tick data from MT5

        Returns:
            True if queued successfully, False if queue full
        """
        try:
            # Add timestamp for latency tracking
            tick.arrival_time = time.perf_counter()
            
            # Try to add without blocking
            self.tick_queue.put_nowait(tick)
            return True
        except asyncio.QueueFull:
            self.dropped_ticks += 1
            logger.warning(f"[{self.symbol}] Tick queue full - dropped tick")
            return False

    async def process_ticks(self):
        """
        Main async loop for processing queued ticks.
        Runs continuously, processing ticks as they arrive.
        """
        logger.info(f"[{self.symbol}] Starting tick processing loop")

        while True:
            try:
                # Get tick from queue (blocks until available)
                tick = await asyncio.wait_for(
                    self.tick_queue.get(),
                    timeout=1.0
                )

                # Execute tick callbacks
                for callback in self.on_tick_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(tick)
                        else:
                            callback(tick)
                    except Exception as e:
                        await self._handle_error(e, "tick callback")

                # Track latency
                processing_end = time.perf_counter()
                latency = (processing_end - tick.arrival_time) * 1000  # ms

                if self.enable_profiling:
                    self.latency_history.append(latency)
                    self.processed_ticks += 1

                    if latency > self.max_latency_ms:
                        self.exceeded_latency_count += 1
                        logger.warning(
                            f"[{self.symbol}] Latency exceeded: {latency:.2f}ms "
                            f"(max: {self.max_latency_ms}ms)"
                        )

                self.tick_queue.task_done()

            except asyncio.TimeoutError:
                # No ticks for 1 second - normal during low volume
                continue
            except Exception as e:
                await self._handle_error(e, "main loop")
                await asyncio.sleep(0.1)  # Brief pause on error

    async def _handle_error(self, error: Exception, context: str):
        """Handle errors during processing"""
        logger.error(f"[{self.symbol}] Error in {context}: {error}")

        for callback in self.on_error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error, context)
                else:
                    callback(error, context)
            except Exception as e:
                logger.error(f"[{self.symbol}] Error in error callback: {e}")

    def get_latency_stats(self) -> Dict[str, Any]:
        """Get latency statistics"""
        if not self.latency_history:
            return {}

        latencies = list(self.latency_history)
        return {
            'mean_ms': np.mean(latencies),
            'median_ms': np.median(latencies),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'max_ms': np.max(latencies),
            'min_ms': np.min(latencies),
            'total_processed': self.processed_ticks,
            'total_dropped': self.dropped_ticks,
            'exceeded_max_latency': self.exceeded_latency_count
        }

    def get_recent_ticks(self, limit: int = 100) -> List[Any]:
        """
        Get recent ticks from the queue (non-destructive).
        Used for order flow analysis.

        Args:
            limit: Maximum number of recent ticks to return

        Returns:
            List of recent tick objects
        """
        try:
            # Get items from queue without removing them
            temp_queue = list(self.tick_queue._queue) if hasattr(self.tick_queue, '_queue') else []
            
            # Return most recent ticks up to limit
            return temp_queue[-limit:] if temp_queue else []
        except Exception as e:
            logger.warning(f"[{self.symbol}] Error getting recent ticks: {e}")
            return []


class MultiScaleAnalyzer:
    """
    Concurrent analysis at multiple timeframes.
    - Tick level: Raw order flow signals
    - 1-second candles: Momentum confirmation
    - 5-second candles: Trend strength
    """

    def __init__(self, symbol: str):
        self.symbol = symbol

        # Candle buffers for each timeframe
        self.tick_candles: deque[MultiScaleCandle] = deque(maxlen=100)
        self.one_sec_candles: deque[MultiScaleCandle] = deque(maxlen=300)  # 5 min history
        self.five_sec_candles: deque[MultiScaleCandle] = deque(maxlen=60)  # 5 min history

        # Current candle in progress
        self.current_tick_candle = None
        self.current_1s_candle = None
        self.current_5s_candle = None

        # Timing for candle creation
        self.last_tick_time = None
        self.last_1s_time = None
        self.last_5s_time = None

        logger.info(f"[INIT] Multi-scale analyzer for {symbol}")

    def update_with_tick(self, tick: Any) -> Dict[str, MultiScaleCandle]:
        """
        Update candles with new tick and return completed candles.

        Returns:
            Dict mapping timeframe -> completed candle (if any)
        """
        completed_candles = {}
        current_time = datetime.now(timezone.utc)

        # Update tick-level candle (aggregate 10 ticks)
        if self._update_candle(
            self.current_tick_candle,
            tick,
            'tick',
            candle_size=10
        ):
            self.current_tick_candle = self._create_candle(tick, 'tick')
            if self.current_tick_candle:
                completed_candles['tick'] = self.tick_candles[-1] if self.tick_candles else None

        # Update 1-second candle
        if self.last_1s_time is None:
            self.last_1s_time = current_time
            self.current_1s_candle = self._create_candle(tick, '1s')
        elif (current_time - self.last_1s_time).total_seconds() >= 1.0:
            if self.current_1s_candle:
                self.one_sec_candles.append(self.current_1s_candle)
                completed_candles['1s'] = self.current_1s_candle
            self.current_1s_candle = self._create_candle(tick, '1s')
            self.last_1s_time = current_time
        else:
            self._update_candle(self.current_1s_candle, tick, '1s')

        # Update 5-second candle
        if self.last_5s_time is None:
            self.last_5s_time = current_time
            self.current_5s_candle = self._create_candle(tick, '5s')
        elif (current_time - self.last_5s_time).total_seconds() >= 5.0:
            if self.current_5s_candle:
                self.five_sec_candles.append(self.current_5s_candle)
                completed_candles['5s'] = self.current_5s_candle
            self.current_5s_candle = self._create_candle(tick, '5s')
            self.last_5s_time = current_time
        else:
            self._update_candle(self.current_5s_candle, tick, '5s')

        return completed_candles

    def _create_candle(self, tick: Any, timeframe: str) -> MultiScaleCandle:
        """Create new candle from tick"""
        price = getattr(tick, 'last_price', 0.0)
        bid_size = getattr(tick, 'bid_size', 0.0)
        ask_size = getattr(tick, 'ask_size', 0.0)

        return MultiScaleCandle(
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=bid_size + ask_size,
            buy_volume=bid_size,
            sell_volume=ask_size,
            tick_count=1
        )

    def _update_candle(self, candle: Optional[MultiScaleCandle], tick: Any,
                       timeframe: str, candle_size: int = 1) -> bool:
        """
        Update existing candle with new tick.
        Returns True if candle should be finalized.
        """
        if candle is None:
            return False

        price = getattr(tick, 'last_price', 0.0)
        bid_size = getattr(tick, 'bid_size', 0.0)
        ask_size = getattr(tick, 'ask_size', 0.0)

        candle.high = max(candle.high, price)
        candle.low = min(candle.low, price)
        candle.close = price
        candle.volume += bid_size + ask_size
        candle.buy_volume += bid_size
        candle.sell_volume += ask_size
        candle.tick_count += 1

        # Check if candle complete (for tick timeframe)
        if timeframe == 'tick' and candle.tick_count >= candle_size:
            if self.current_tick_candle:
                self.tick_candles.append(self.current_tick_candle)
            return True

        return False

    def check_confluence(
        self,
        signal_type: str,
        tick_signal: bool,
        one_sec_signal: bool,
        five_sec_signal: bool
    ) -> SignalConfluence:
        """
        Check signal confluence across timeframes.

        Returns:
            Confluence level for this signal
        """
        signal_count = sum([tick_signal, one_sec_signal, five_sec_signal])

        if signal_count == 0:
            return SignalConfluence.CONTRADICTION

        if signal_count == 3:
            return SignalConfluence.TRIPLE_CONFIRMATION
        elif signal_count == 2:
            return SignalConfluence.DUAL_CONFIRMATION
        else:
            return SignalConfluence.SINGLE_TIMEFRAME

    def get_candle_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get current candle status across all timeframes"""
        return {
            'tick': {
                'count': len(self.tick_candles),
                'current_open': self.current_tick_candle.open if self.current_tick_candle else None,
                'current_close': self.current_tick_candle.close if self.current_tick_candle else None,
                'current_volume': self.current_tick_candle.volume if self.current_tick_candle else 0
            },
            '1s': {
                'count': len(self.one_sec_candles),
                'current_open': self.current_1s_candle.open if self.current_1s_candle else None,
                'current_close': self.current_1s_candle.close if self.current_1s_candle else None,
                'current_volume': self.current_1s_candle.volume if self.current_1s_candle else 0
            },
            '5s': {
                'count': len(self.five_sec_candles),
                'current_open': self.current_5s_candle.open if self.current_5s_candle else None,
                'current_close': self.current_5s_candle.close if self.current_5s_candle else None,
                'current_volume': self.current_5s_candle.volume if self.current_5s_candle else 0
            }
        }


class LatencyOptimizer:
    """Monitor and optimize processing latency"""

    def __init__(self, target_latency_ms: float = 5.0):
        self.target_latency_ms = target_latency_ms
        self.optimization_history: deque[Dict[str, Any]] = deque(maxlen=100)

    def analyze_latency(self, handler: AsyncTickHandler) -> Dict[str, Any]:
        """Analyze latency and suggest optimizations"""
        stats = handler.get_latency_stats()

        if not stats:
            return {}

        self.optimization_history.append(stats)

        # Generate optimization suggestions
        suggestions = []

        if stats['p99_ms'] > self.target_latency_ms * 1.5:
            suggestions.append("High P99 latency - consider reducing callback complexity")

        if stats['total_dropped'] > stats['total_processed'] * 0.01:
            suggestions.append("High drop rate (>1%) - increase queue size or reduce processing time")

        if stats['exceeded_max_latency'] > stats['total_processed'] * 0.05:
            suggestions.append("5%+ of ticks exceed max latency - optimize hot paths")

        return {
            'stats': stats,
            'suggestions': suggestions,
            'is_optimized': len(suggestions) == 0
        }

    def get_optimization_report(self) -> str:
        """Generate optimization report"""
        if not self.optimization_history:
            return "No latency data collected yet"

        latest = self.optimization_history[-1]
        report = (
            f"Latency Report:\n"
            f"  Mean: {latest['mean_ms']:.2f}ms\n"
            f"  P99: {latest['p99_ms']:.2f}ms\n"
            f"  Max: {latest['max_ms']:.2f}ms\n"
            f"  Processed: {latest['total_processed']}\n"
            f"  Dropped: {latest['total_dropped']}\n"
            f"  Exceeded Max: {latest['exceeded_max_latency']}"
        )
        return report
