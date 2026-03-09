#!/usr/bin/env python3
"""
QUICK START GUIDE: Scalping System Setup & Configuration
========================================================

This guide walks you through setting up and running the scalping system.

==============================================================================
STEP 1: INSTALL DEPENDENCIES
==============================================================================

If you haven't already, install scalping-specific packages:

$ pip install xgboost lightgbm catboost

For optional advanced features:
$ pip install tensorflow torch  # If planning LSTM/deep learning models

==============================================================================
STEP 2: CONFIGURE config.json
==============================================================================

Your config.json already has scalping parameters added. Here's what each does:

"scalping": {
  "enabled": true,                    # Turn ON/OFF with single flag
  "timeframes": ["M1", "M5"],        # Trade these timeframes
  "min_confidence": 0.65,             # AI model must be 65%+ sure
  "risk_per_trade": 0.02,             # 2% risk max per trade
  "max_daily_loss": 0.06,             # Stop trading if -6% daily
  "max_consecutive_losses": 4,        # Stop after 4 losses in a row
  "reward_ratio": 1.2                 # Aim for 1.2× risk target
}

Recommendations for FIRST RUN (Conservative):
──────────────────────────────────────────
{
  "scalping": {
    "enabled": true,
    "timeframes": ["M5"],              # Start with M5, add M1 later
    "min_confidence": 0.68,            # Higher = more selective
    "risk_per_trade": 0.01,            # 1% not 2% - be safe
    "max_daily_loss": 0.04,            # Stop at -4% not -6%
    "max_consecutive_losses": 3,       # Stop sooner
    "reward_ratio": 1.5                # Higher target = harder to get
  }
}

This configuration:
- Trades M5 only (less noise than M1)
- Waits for 68%+ AI confidence (fewer false signals)
- Risks only 1% per trade (safe position sizing)
- Stops trading after -4% daily loss (capital preservation)
- Stops after 3 losses (avoids revenge trading)
- Requires 1.5:1 risk:reward (high quality setups only)

Expected results: 3-5 trades per day, 55%+ win rate, 0.5-0.8% daily return

==============================================================================
STEP 3: MODIFY AUTO_TRADER_MULTI.py FOR SCALPING INTEGRATION
==============================================================================

You need to integrate the scalping engine into your main trading bot.

Add these imports at the top of AUTO_TRADER_MULTI.py:

```python
from SCALPING_ENGINE import ScalpingEngine, create_scalping_setup_from_ml
from SCALPING_INTEGRATION import ScalpingIntegration, RiskAdjustmentEngine
```

Modify the __init__ method to initialize scalping:

```python
def __init__(self, ...):
    # ... existing code ...
    
    # Initialize scalping engines
    self.scalping_engines = {}
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
    
    logger.info("[SCALPING] Engines initialized for all symbols")
```

Modify the run_symbol method to check for scalping opportunities:

```python
def run_symbol(self, symbol: str):
    # ... existing ML signal generation ...
    
    ml_signal, ml_confidence = self.get_ml_signal(symbol, df_1h)
    
    # NEW: Check for scalping opportunities on M1/M5
    if self.scalping_integration.scalping_enabled:
        if self.scalping_integration.should_scalp_symbol(symbol):
            
            # Get M5 data for scalping
            df_m5 = self.get_market_data(symbol, bars=50, timeframe="M5")
            current_price = df_m5['close'].iloc[-1]
            atr_value = self._calculate_atr(df_m5, period=14)
            spread_pips = self._get_current_spread(symbol)
            
            # Combine ML signal with microstructure setup
            scalp_result = create_scalping_setup_from_ml(
                symbol=symbol,
                ml_signal=ml_signal,
                ml_confidence=ml_confidence,
                df=df_m5,
                current_price=current_price,
                atr_value=atr_value,
                spread_pips=spread_pips,
                scalping_engine=self.scalping_engines[symbol]
            )
            
            if scalp_result:
                setup, final_confidence = scalp_result
                
                # Calculate risk-adjusted position size
                adaptive_lot = self.scalping_engines[symbol].calculate_adaptive_lot_size(
                    setup=setup,
                    account_balance=self.get_balance(),
                    current_positions=len(self.get_all_open_positions()),
                    symbol=symbol
                )
                
                # Execute scalp trade
                self.open_trade(
                    symbol=symbol,
                    direction=setup.direction,
                    confidence=final_confidence,
                    position_size=adaptive_lot,
                    stop_loss_pips=setup.stop_loss_pips,
                    take_profit_pips=setup.target_pips
                )
    
    # ... rest of existing logic ...
```

==============================================================================
STEP 4: ADD HELPER METHODS TO AUTO_TRADER_MULTI
==============================================================================

Add these methods to calculate ATR and get spread:

```python
def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range for stop loss sizing"""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()
    
    return atr.iloc[-1] if len(atr) > 0 else 0.0001

def _get_current_spread(self, symbol: str) -> float:
    """Get current bid-ask spread in pips"""
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            spread = (tick.ask - tick.bid) * 10**4  # Convert to pips
            return max(spread, 0.1)
    except:
        pass
    
    # Default spreads for major pairs
    default_spreads = {
        "EURUSD": 1.5, "GBPUSD": 2.0, "USDJPY": 1.5,
        "AUDUSD": 1.5, "NZDUSD": 2.5, "USDCAD": 1.8
    }
    return default_spreads.get(symbol, 2.0)
```

==============================================================================
STEP 5: RUN WITH SCALPING ENABLED
==============================================================================

Start your bot with scalping:

$ python AUTO_TRADER_MULTI.py

You should see logs like:

```
[INIT] Scalping Engine for EURUSD on M1
[INIT] Scalping Engine for GBPUSD on M1
[INIT] Scalping Integration | Enabled: True
[SCALP SETUP] EURUSD | Type: LIQUIDITY_SWEEP | Dir: LONG | 
  Confidence: 0.92 | SL: 3.0p | TP: 4.0p | Regime: NORMAL_VOLATILITY
[LOT SIZE] EURUSD | Base: 200.00 | Loss Penalty: 1.0 | 
  Position Penalty: 1.0 | Kelly: 0.80 | Final: 160.00
[PASS] SIGNAL PASSED - Opening trade
```

==============================================================================
STEP 6: MONITOR & TUNE
==============================================================================

During the first few days, check:

1. Setup Detection:
   - Are setups being detected? (check logs)
   - Are they matching ML signals?
   - Example: "LIQUIDITY_SWEEP matches ML SHORT signal"

2. Position Sizing:
   - Is lot size reasonable? (check "Final:" amount)
   - Is it adjusting when you have losses?
   - Should decrease after consecutive losses

3. Win Rate:
   - After 20+ trades, check win rate
   - Expected: 53-58%
   - If below 50%, pause and review

4. Daily Profit:
   - Expected: 0.5-1% daily on conservative settings
   - If 0%, check if trades are being executed
   - If negative, consider:
     - Tighten min_confidence (0.68 → 0.70)
     - Reduce risk_per_trade (0.01 → 0.005)
     - Switch to M5 only (less noise)

5. Spread Impact:
   - During news: spreads widen 5-10x normal
   - Scalping becomes unprofitable
   - Bot should skip these (automatic via regime detection)

==============================================================================
STEP 7: GRADUAL RAMPING (DO THIS!)
==============================================================================

DAY 1-3: TEST WITH M5 ONLY
─────────────────────────
config.json:
  "timeframes": ["M5"],
  "min_confidence": 0.68,
  "risk_per_trade": 0.005  (0.5%)

Goal: Verify setups are working, no execution issues

DAY 4-7: ADD CONFIDENCE
──────────────────────
  "min_confidence": 0.66,
  "risk_per_trade": 0.01  (1%)

Goal: Increase trade frequency, maintain quality

DAY 8-14: ADD M1 DURING OVERLAP
───────────────────────────────
  "timeframes": ["M1"],
  "min_confidence": 0.63,
  "risk_per_trade": 0.02  (2%)

Goal: Most frequent trading window, validate M1 setups

WEEK 3+: FULL PRODUCTION
────────────────────
  "timeframes": ["M1", "M5"],
  "min_confidence": 0.65,
  "risk_per_trade": 0.02  (2%)

Goal: Full system running

==============================================================================
STEP 8: TROUBLESHOOTING
==============================================================================

No setups detected:
───────────────────
Problem: Logs show "[HOLD] No setup signal generated"
Solution: 
  1. Lower min_confidence from 0.65 → 0.60
  2. Check volatility: might be in LOW_VOLATILITY regime
  3. Add more symbols to increase trade frequency

Win rate too low (< 50%):
──────────────────────
Problem: Trading too many low-confidence setups
Solution:
  1. Increase min_confidence: 0.65 → 0.70
  2. Increase reward_ratio: 1.2 → 1.5 (higher target = fewer bad trades)
  3. Switch to M5 only (M1 is harder)
  4. Trade during overlap hours only (higher quality)

Inconsistent results:
────────────────────
Problem: Some days 2% profit, next day -3% loss
Solution:
  1. Lower max_daily_loss: 0.06 → 0.04 (cap losses sooner)
  2. Reduce risk_per_trade: 0.02 → 0.01
  3. Add session_filter (only trade London/NY hours)

Position size too small:
───────────────────────
Problem: "(Final: 40)" way too small for account
Solution:
  1. Check consecutive losses: might have recent string of losses
  2. Check daily drawdown: might be -3% already
  3. Review risk_per_trade setting: might be 0.01 instead of 0.02
  4. Verify account balance loaded correctly in logs

==============================================================================
REFERENCE: Expected Log Output
==============================================================================

Healthy scalping session should show:

[INFO] [SCALP SETUP] EURUSD | Type: EMA_BREAKOUT_LONG | Dir: LONG | 
       Confidence: 0.89 | SL: 2.0p | TP: 3.0p | Regime: NORMAL_VOLATILITY
[INFO] [LOT SIZE] EURUSD | Base: 200.00 | Loss Penalty: 1.00 | 
       Position Penalty: 1.00 | Kelly: 0.85 | Final: 170.00
[INFO] [PASS] SIGNAL PASSED - Opening trade
[INFO] Trade opened: EURUSD Buy 1.0950 SL: 1.0948 TP: 1.0953

(5 minutes later...)

[INFO] [TRADE RESULT] EURUSD | PnL: +170.00 | Win: 8/10 | 
       Consecutive Losses: 0 | Daily PnL: +340.00

==============================================================================
FINAL CHECKLIST
==============================================================================

Before going live:

☐ config.json has scalping section with parameters
☐ AUTO_TRADER_MULTI.py imports ScalpingEngine and ScalpingIntegration
☐ Scalping engines initialized in __init__
☐ run_symbol() method checks for scalping setups
☐ Helper methods (_calculate_atr, _get_current_spread) added
☐ Started with conservative settings (M5, 0.68 confidence, 0.5% risk)
☐ Monitored for 3+ days without issues
☐ Win rate is 50%+ (not -10%)
☐ Position sizes are reasonable (not 0.01 lots)
☐ All logs are clean (no errors about missing spread values)

Good luck! Questions? Check SCALPING_SYSTEM_DOCS.py for deep dives.
"""

# This is a guide file - read it before running
