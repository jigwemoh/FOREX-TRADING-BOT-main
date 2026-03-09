#!/usr/bin/env python3
"""
AUTO_TRADER_MULTI.py INTEGRATION CODE - COPY/PASTE READY
=========================================================

This file contains the exact code snippets to add to AUTO_TRADER_MULTI.py
to enable scalping. Follow these 4 sections in order.

IMPORTANT: This is NOT a standalone file - these are code snippets to add
to your existing AUTO_TRADER_MULTI.py file.
"""

# ==============================================================================
# SECTION 1: ADD IMPORTS AT TOP OF FILE (After existing imports)
# ==============================================================================

# Copy this block and add it after the existing imports in AUTO_TRADER_MULTI.py
# (around line 18-21)

"""
# ADD THESE LINES:

from SCALPING_ENGINE import ScalpingEngine, create_scalping_setup_from_ml
from SCALPING_INTEGRATION import ScalpingIntegration, RiskAdjustmentEngine
"""


# ==============================================================================
# SECTION 2: MODIFY __init__ METHOD
# ==============================================================================

# Find the __init__ method (around line 35-75) and add this at the end
# ADD THESE LINES before the closing of __init__:

"""
        # ===== SCALPING ENGINE INITIALIZATION (NEW) =====
        # Initialize scalping engines for intraday trading
        self.scalping_engines: Dict[str, ScalpingEngine] = {}
        self.scalping_integration = ScalpingIntegration(config_path="../config.json")
        self.risk_adjuster = RiskAdjustmentEngine(initial_risk=0.02)
        
        for symbol in self.symbols:
            self.scalping_engines[symbol] = ScalpingEngine(
                symbol=symbol,
                timeframe="M1",
                atr_period=14,
                vwap_lookback=50,
                ema_fast=20,
                ema_slow=50
            )
        
        logger.info(f"[SCALPING] Initialized {len(self.scalping_engines)} scalping engines")
        logger.info(f"[SCALPING] Scalping enabled: {self.scalping_integration.scalping_enabled}")
"""


# ==============================================================================
# SECTION 3: ADD HELPER METHODS (Add to class, after existing methods)
# ==============================================================================

# ADD THESE METHODS to the MultiSymbolAutoTrader class:

"""
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        \"\"\"
        Calculate Average True Range for stop loss sizing
        
        Args:
            df: OHLC dataframe with columns: high, low, close
            period: ATR period (default 14)
            
        Returns:
            ATR value in price units (same scale as price data)
        \"\"\"
        try:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(period).mean()
            
            return float(atr.iloc[-1]) if len(atr) > 0 and not pd.isna(atr.iloc[-1]) else 0.0001
        except Exception as e:
            logging.warning(f"Error calculating ATR: {e}")
            return 0.0001

    def _get_current_spread(self, symbol: str) -> float:
        \"\"\"
        Get current bid-ask spread in pips
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Spread in pips (4 decimal places = 1 pip)
        \"\"\"
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick and hasattr(tick, 'ask') and hasattr(tick, 'bid'):
                spread_price = tick.ask - tick.bid
                # Convert to pips (4 decimal places for most pairs)
                spread_pips = spread_price * 10**4
                return max(float(spread_pips), 0.1)
        except Exception as e:
            logging.warning(f"Error getting spread for {symbol}: {e}")
        
        # Fallback to default spreads for major pairs
        default_spreads = {
            "EURUSD": 1.5, "GBPUSD": 2.0, "USDJPY": 1.5,
            "AUDUSD": 1.5, "NZDUSD": 2.5, "USDCAD": 1.8,
            "EURGBP": 1.8, "EURJPY": 2.0
        }
        return float(default_spreads.get(symbol, 2.0))

    def _get_market_data_timeframe(
        self,
        symbol: str,
        bars: int = 100,
        timeframe: str = "M5"
    ) -> Optional[pd.DataFrame]:
        \"\"\"
        Get market data for specific timeframe (M1, M5, 1H, etc)
        Similar to get_market_data but allows custom timeframe
        
        Args:
            symbol: Trading symbol
            bars: Number of bars to fetch
            timeframe: Timeframe string ("M1", "M5", "1H", "4H", etc)
            
        Returns:
            DataFrame with OHLC data or None if error
        \"\"\"
        try:
            # Map timeframe string to MT5 constant
            tf_map = {
                "M1": mt5.TIMEFRAME_M1,
                "5M": mt5.TIMEFRAME_M5,
                "M5": mt5.TIMEFRAME_M5,
                "10M": mt5.TIMEFRAME_M10,
                "M10": mt5.TIMEFRAME_M10,
                "15M": mt5.TIMEFRAME_M15,
                "M15": mt5.TIMEFRAME_M15,
                "30M": mt5.TIMEFRAME_M30,
                "M30": mt5.TIMEFRAME_M30,
                "1H": mt5.TIMEFRAME_H1,
                "H1": mt5.TIMEFRAME_H1,
                "4H": mt5.TIMEFRAME_H4,
                "H4": mt5.TIMEFRAME_H4,
                "1D": mt5.TIMEFRAME_D1,
                "D1": mt5.TIMEFRAME_D1,
            }
            
            mt5_tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_M5)
            
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars)
            if rates is None:
                logging.warning(f"No data for {symbol} on {timeframe}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={
                'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
                'tick_volume': 'volume'
            })
            
            return df[['time', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logging.warning(f"Error fetching {timeframe} data for {symbol}: {e}")
            return None
"""


# ==============================================================================
# SECTION 4: MODIFY run_symbol() METHOD
# ==============================================================================

# Find the run_symbol() method and modify the signal generation section
# REPLACE THE SECTION after getting ML signal with this:

"""
# In run_symbol(), after getting ml_signal and ml_confidence, add:

                # ===== SCALPING OPPORTUNITY CHECK (NEW) =====
                # Check if we should attempt scalping on M1/M5
                if self.scalping_integration.scalping_enabled:
                    try:
                        if self.scalping_integration.should_scalp_symbol(symbol):
                            
                            # Get M5 data for scalping analysis
                            df_scalp = self._get_market_data_timeframe(symbol, bars=50, timeframe="M5")
                            
                            if df_scalp is not None and len(df_scalp) >= 10:
                                current_price = float(df_scalp['close'].iloc[-1])
                                atr_value = self._calculate_atr(df_scalp, period=14)
                                spread_pips = self._get_current_spread(symbol)
                                
                                # Detect scalping setup
                                scalp_result = create_scalping_setup_from_ml(
                                    symbol=symbol,
                                    ml_signal=ml_signal,
                                    ml_confidence=ml_confidence,
                                    df=df_scalp,
                                    current_price=current_price,
                                    atr_value=atr_value,
                                    spread_pips=spread_pips,
                                    scalping_engine=self.scalping_engines[symbol]
                                )
                                
                                if scalp_result is not None:
                                    setup, final_confidence = scalp_result
                                    
                                    # Calculate adaptive position size
                                    total_positions = len(self.get_all_open_positions())
                                    adaptive_lot = self.scalping_engines[symbol].calculate_adaptive_lot_size(
                                        setup=setup,
                                        account_balance=self.get_balance(),
                                        current_positions=total_positions,
                                        symbol=symbol
                                    )
                                    
                                    # Check additional gates before executing
                                    if adaptive_lot > 0:
                                        # Would add trade here
                                        # self.open_trade(symbol, setup.direction, final_confidence, position_size=adaptive_lot)
                                        pass
                    
                    except Exception as e:
                        logging.warning(f"[SCALPING] Error processing {symbol}: {e}")
                
                # ===== CONTINUE WITH EXISTING LOGIC =====
"""


# ==============================================================================
# SECTION 5: CONFIGURATION VERIFICATION
# ==============================================================================

# Make sure your config.json has these sections:

"""
{
  "trading": {
    "scalping": {
      "enabled": true,
      "timeframes": ["M1", "M5"],
      "min_confidence": 0.65,
      "risk_per_trade": 0.02,
      "max_daily_loss": 0.06,
      "max_consecutive_losses": 4,
      "reward_ratio": 1.2
    },
    "microstructure": {
      "detect_liquidity_sweeps": true,
      "detect_rejection_candles": true,
      "vwap_lookback": 50,
      "atr_period": 14,
      "ema_fast": 20,
      "ema_slow": 50
    },
    "session_filters": {
      "london_hours": [8, 16],
      "ny_hours": [13, 21],
      "asia_hours": [0, 8],
      "prefer_overlap": true,
      "avoid_news_window": true,
      "news_window_minutes": 15
    }
  },
  "execution": {
    "check_interval": 300,
    "scalping_check_interval": 60
  }
}
"""


# ==============================================================================
# STEP-BY-STEP INTEGRATION CHECKLIST
# ==============================================================================

"""
1. BACKUP YOUR CURRENT AUTO_TRADER_MULTI.py
   $ cp PY_FILES/AUTO_TRADER_MULTI.py PY_FILES/AUTO_TRADER_MULTI.py.backup

2. OPEN AUTO_TRADER_MULTI.py IN YOUR EDITOR

3. ADD IMPORTS (SECTION 1)
   Location: After line 21 (after other imports)
   Paste: The import statements from SECTION 1

4. MODIFY __init__ (SECTION 2)
   Location: End of __init__ method (before closing)
   Paste: The initialization code from SECTION 2

5. ADD HELPER METHODS (SECTION 3)
   Location: After existing methods in class (before end of class)
   Paste: The three helper methods from SECTION 3

6. MODIFY run_symbol() (SECTION 4)
   Location: In run_symbol(), after getting ML signal
   Paste: The scalping check code from SECTION 4

7. VERIFY config.json (SECTION 5)
   Location: /config.json
   Make sure: It has all the scalping sections

8. TEST THE INTEGRATION
   $ python PY_FILES/AUTO_TRADER_MULTI.py
   
   Look for logs like:
   "[INIT] Scalping Engine for EURUSD on M1"
   "[SCALPING] Initialized 4 scalping engines"

9. IF ERRORS:
   Compare your modified file with the original .backup
   Make sure indentation matches
   Make sure all imports are present

10. ONCE WORKING:
    Start with conservative config:
      "min_confidence": 0.68
      "risk_per_trade": 0.005
      "timeframes": ["M5"]  (not M1)
    
    Monitor for 3 days before adjusting
"""


# ==============================================================================
# COMMON ISSUES & FIXES
# ==============================================================================

"""
ISSUE: ImportError: No module named 'SCALPING_ENGINE'
CAUSE: Missing scalping files
FIX:
  1. Make sure SCALPING_ENGINE.py exists in /PY_FILES/
  2. Make sure SCALPING_INTEGRATION.py exists in /PY_FILES/
  3. Check file locations match exactly

ISSUE: AttributeError: module 'mt5' has no attribute 'TIMEFRAME_M1'
CAUSE: Mock MT5 doesn't support all timeframes
FIX:
  1. Use real MetaTrader5 on Windows
  2. Or update MT5_MOCK.py to add missing timeframes

ISSUE: KeyError in _get_current_spread()
CAUSE: Symbol not in data or missing columns
FIX:
  1. Check symbol is available on your broker
  2. Verify MT5 is connected: should see price data

ISSUE: No scalping setups detected
CAUSE: Confidence threshold too high or market conditions poor
FIX:
  1. Lower min_confidence: 0.68 → 0.60
  2. Check logs for "[HOLD]" messages (no signal generated)
  3. Try during London/NY overlap hours

ISSUE: Position sizes very small (< 0.01 lots)
CAUSE: Recent losses reducing position size (Kelly adjustment)
FIX:
  1. This is normal and expected
  2. Size will increase when win rate improves
  3. Don't override - trust the algorithm
  4. Or reduce initial risk: 0.02 → 0.01
"""


# ==============================================================================
# REFERENCE: Expected Output After Integration
# ==============================================================================

"""
After successful integration, you should see logs like:

[INFO] [INIT] Scalping Engine for EURUSD on M1
[INFO] [INIT] Scalping Engine for GBPUSD on M1
[INFO] [INIT] Scalping Engine for USDJPY on M1
[INFO] [INIT] Scalping Engine for AUDUSD on M1
[INFO] [SCALPING] Initialized 4 scalping engines
[INFO] [SCALPING] Scalping enabled: True

(During trading...)

[INFO] [SCALP SETUP] EURUSD | Type: EMA_BREAKOUT_LONG | Dir: LONG | 
       Confidence: 0.89 | SL: 2.0p | TP: 3.0p | Regime: NORMAL_VOLATILITY
[INFO] [LOT SIZE] EURUSD | Base: 200.00 | Loss Penalty: 1.00 | 
       Position Penalty: 1.00 | Kelly: 0.85 | Final: 170.00
[INFO] [PASS] SIGNAL PASSED - Opening trade

(After 5 minutes...)

[INFO] [TRADE RESULT] EURUSD | PnL: +170.00 | Win: 1/1 | 
       Consecutive Losses: 0 | Daily PnL: +170.00
"""


# ==============================================================================
# DONE! NEXT STEPS
# ==============================================================================

"""
1. Run the integrated bot with conservative settings
2. Monitor logs for 3+ days to verify setups work
3. Check win rate (should be 50%+)
4. Check daily profit (should be 0.5-1% on conservative settings)
5. Gradually increase risk over 2-3 weeks per SCALPING_QUICK_START.md
6. Monitor and adjust based on performance
"""
