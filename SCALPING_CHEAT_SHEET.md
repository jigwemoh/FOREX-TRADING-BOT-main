# SCALPING SYSTEM CHEAT SHEET & REFERENCE

## Quick Reference: Setup Detection

### 1. EMA BREAKOUT
```
Condition:    Price > EMA20 > EMA50 with RSI pullback
Long Entry:   RSI(7) = 40-50 (oversold bounce)
Short Entry:  RSI(7) = 50-60 (overbought reaction)
Target:       3 pips (M1) / 5 pips (M5)
Stop Loss:    1× ATR(14)
Confidence:   0.68
Probability:  55-60% win rate expected
```

### 2. VWAP BOUNCE
```
Condition:    Price within 2 pips of VWAP
Bounce Type:  Positive momentum above VWAP, RSI < 50 (long)
              Negative momentum below VWAP, RSI > 50 (short)
Target:       2.5 pips
Stop Loss:    2.5 pips
Confidence:   0.62
Probability:  52-55% win rate expected
```

### 3. LIQUIDITY SWEEP ⭐
```
Pattern:      Previous high/low broken, sharp reversal
Example:      High 1.0955 → Close 1.0948 (bears hunted stops)
Target:       4 pips (excellent 1.3:1 R:R)
Stop Loss:    3 pips
Confidence:   0.70 (HIGHEST - use this often!)
Probability:  60-70% win rate expected
Order Flow:   LIQUIDITY_SWEEP signal
```

### 4. REJECTION CANDLE
```
Pattern:      Long wick (3× body size) + small close
Long Signal:  Low wick < body × 3 (buyers rejected bears)
Short Signal: High wick > body × 3 (sellers rejected buyers)
Target:       2.5 pips
Stop Loss:    Based on wick length (usually 2.5-3 pips)
Confidence:   0.66
Probability:  56-60% win rate expected
Location:     Works best at round numbers & support/resistance
```

---

## Setup Quality Scoring Formula

```
Base Score = 0.5 (neutral)

Setup Type Bonuses:
  LIQUIDITY_SWEEP:     +0.35 (0.85 total)
  REJECTION:           +0.30 (0.80 total)
  EMA_BREAKOUT:        +0.20 (0.70 total)
  VWAP_BOUNCE:         +0.15 (0.65 total)

ML Confidence Multiplier:
  0.50 to 1.00 range (0% confidence = 0.5x, 100% = 1.0x)

Volatility Multiplier:
  NORMAL:    ×1.00 (optimal)
  LOW:       ×0.70 (tight range = worse)
  HIGH:      ×0.85 (whipsaw risk)
  EXTREME:   ×0.40 (too risky, usually skip)

Spread Penalty:
  = 1.0 - (current_spread / 2.0)
  (at 2 pips spread = breakeven point)

FINAL SCORE = Base × ML_mult × Volatility_mult × Spread_penalty
```

Example Calculation:
```
Setup: LIQUIDITY_SWEEP
  Base: 0.85
  ML Confidence: 0.70 → Multiplier: 0.85
  Volatility: NORMAL → ×1.00
  Spread: 1.2 pips → Penalty: 0.4

Score = 0.85 × 0.85 × 1.00 × 0.4 = 0.289 (low due to wide spread)

vs. Same setup with 1.2 pips spread:
  Spread: 1.2 pips → Penalty: 0.40
Score = 0.85 × 0.85 × 1.00 × 0.40 = 0.289
```

---

## Position Sizing Formula (Kelly Criterion)

```
Step 1: Base Risk
  Risk = Account Balance × 2%
  Example: $10,000 × 0.02 = $200

Step 2: Consecutive Loss Penalty
  For each loss: ×(1 - 0.25) = ×0.75
  After 1 loss: $200 × 0.75 = $150
  After 2 losses: $200 × 0.75 × 0.75 = $112.50
  After 3 losses: $200 × 0.75³ = $84.38
  After 4 losses: STOP TRADING

Step 3: Drawdown Penalty
  If DD > -2%:  ×1.0
  If DD > -4%:  ×0.5
  If DD > -5%:  ×0.3
  If DD < -6%:  STOP TRADING

Step 4: Position Count Penalty
  0 positions: ×1.0
  1 position:  ×0.95
  2 positions: ×0.90
  3 positions: ×0.85
  4+ positions: ×0.70

Step 5: Kelly Fraction
  Kelly% = (WinRate × RR - LossRate) / RR
  Applied at 0.5 Kelly (conservative)
  
  Example (55% win rate, 1.2R target):
  Kelly% = (0.55 × 1.2 - 0.45) / 1.2 = 17.5%
  Capped at 50% Kelly = 8.75% max
  But hardcap at 2% per trade

Step 6: Final Calculation
  Final Risk = Base × Loss_Penalty × DD_Penalty × Position_Penalty × Kelly_Fraction
  Lot Size = Final Risk / (Stop Loss Pips × Pip Value)
```

Real Example:
```
Account: $10,000
Recent: 1 loss, DD -1%, 1 position open

Base Risk:        $200
Loss Penalty:     ×0.75 = $150
DD Penalty:       ×1.0 = $150
Position Penalty: ×0.95 = $142.50
Kelly Fraction:   ×0.80 = $114

EURUSD Trade:
  Stop Loss: 2 pips
  Pip Value: $10/pip
  
  Lot Size = $114 / (2 × $10) = 0.57 micro lots
  Risk: $114 (exactly as calculated)
  Reward: $114 × 1.2 = $136.80 (1.2R target)
```

---

## Market Regime Classification

```
EXTREME VOLATILITY 🔴
├─ ATR > 1.8× normal
├─ Spread > 2.5× normal
├─ Action: SKIP TRADING
└─ Examples: Brexit vote, emergency Fed statement

HIGH VOLATILITY 🟠
├─ ATR 1.4-1.8× normal
├─ Spread 1.8-2.5× normal
├─ Action: Reduce position by 20%, skip M1
└─ Examples: NFP release aftermath

NORMAL VOLATILITY 🟢
├─ ATR 0.7-1.4× normal
├─ Spread 0.6-1.8× normal
├─ Action: Full position size, all setups OK
└─ Examples: Regular London/NY trading

LOW VOLATILITY 🟡
├─ ATR < 0.7× normal
├─ Spread < 0.6× normal
├─ Action: Reduce confidence by 20%, skip tight range trades
└─ Examples: Post-news consolidation, overnight Asian hours
```

---

## Time-Based Strategy Matrix

```
                 OVERLAP        LONDON         NY             ASIA           OFF-HOURS
                 13:00-16:00    08:00-16:00    13:00-21:00    00:00-08:00    Other
────────────────────────────────────────────────────────────────────────────────────
Strategy:        SCALP_AGG      SCALP_MOD      SCALP_MOD      SWING_CONS     SWING_CONS
Timeframe:       M1             M5             M5             1H             4H
Risk:            2.0%           1.6%           1.6%           1.0%           1.0%
Min Confidence:  0.60           0.63           0.63           0.68           0.68
Expected Trades: 5-10/day       3-5/day        3-5/day        1-2/day        1/session
Ideal Symbols:   EURUSD, GBPUSD All             All            All            Top 4 only
####────────────────────────────────────────────────────────────────────────────────
Spread Quality:  Tightest       Tight          Tight          Wide           Widest
Volume:          Highest        Very High      High           Low            Very Low
Volatility:      Elevated       Normal         Normal         Low            Variable
Slippage Risk:   Minimal        Low            Low            Moderate       High
```

---

## Risk Management Circuit Breakers

```
✅ TRADING ALLOWED IF:
├─ Consecutive losses < 4
├─ Daily loss > -6%
├─ Volatility ≠ EXTREME
├─ Active positions < 5 total
├─ Position per symbol < 3
└─ Spread < 5 pips (tighten if news)

❌ STOP TRADING IF:
├─ [LOSS BREAKER] 4 consecutive losses
│  └─ Action: Pause 1-2 hours, review setup quality
├─ [DRAWDOWN BREAKER] Daily loss < -6%
│  └─ Action: Market stop, don't trade rest of day
├─ [VOLATILITY BREAKER] Volatility in EXTREME regime
│  └─ Action: Skip until back to NORMAL or HIGH
└─ [POSITION BREAKER] 5+ positions open OR 3+ per symbol
   └─ Action: Close smallest loss or oldest position
```

---

## Session-Based Risk Scaling

```
Starting Point: 2% risk per trade

OVERLAP (Best):           2.0% ← Max risk allowed
London (Good):            1.6% ← 20% reduction
NY (Good):                1.6% ← 20% reduction
Asia (Caution):           1.0% ← 50% reduction
Off-Hours (Avoid):        1.0% ← 50% reduction, consider skipping

Applying to Position Size:
  If system calculates $200 risk in normal mode
  During Asia: $200 × 0.5 = $100 risk max
  
  This means:
  ✓ Fewer trades (higher quality bar)
  ✓ Smaller lot sizes (less slippage impact)
  ✓ Tighter stops (less affected by wide spreads)
```

---

## Entry & Exit Checklist

### BEFORE ENTERING
- [ ] All 10 entry conditions met?
- [ ] Setup direction matches ML signal?
- [ ] Confidence score > 0.60?
- [ ] R:R ratio ≥ 1.2?
- [ ] Spread < 3 pips? (skip if > 3)
- [ ] Session appropriate? (not off-hours)
- [ ] No news in next 2 minutes?
- [ ] Position limit not exceeded?
- [ ] ATR value reasonable? (not spike)

### DURING TRADE
- [ ] Monitor breakeven (move stop at 0.8R)
- [ ] Watch for breakeven exit trigger
- [ ] Check spread (widen = exit early)
- [ ] News within 2 min? (close if yes)
- [ ] Time-based exit (5 bars M1, 3 bars M5)?
- [ ] Trail stops active?

### EXIT CONDITIONS (ANY TRIGGER)
- [ ] TP hit? → Close 100% (profit take)
- [ ] SL hit? → Close 100% (stop loss)
- [ ] Time-based? → Close if 5 bars no movement (M1)
- [ ] Breakeven? → Move SL to cost, trail from there
- [ ] News? → Close 2 min before event
- [ ] Spread spike? → Close if > 5 pips

---

## Performance Metrics to Track

### Daily
```
[DAILY SUMMARY]
Trades Taken:     8
Winning Trades:   5 (62.5% win rate)
Losing Trades:    3 (37.5%)
Consecutive Loss: 1 (OK, <4)
Average Win:      +$45
Average Loss:     -$30
Daily Profit:     +$135 (+1.35% of account)
Max Drawdown:     -0.3% (from peak daily)
Setup Types:      3× EMA, 2× VWAP, 2× Sweep, 1× Rejection
Regime:           NORMAL throughout
```

### Weekly
```
[WEEKLY SUMMARY]
Total Trades:     40
Win Rate:         57% (target: 53-58%)
Expectancy:       +0.18% per trade
Weekly Profit:    +6.5% (40 × 0.18%)
Max Daily DD:     -2.1% (Day 3)
Max Weekly DD:    -2.1% (acceptable)
Best Setup:       Liquidity Sweep (70% win rate)
Worst Setup:      EMA at Asia hours (48% win rate)
Recommendation:   Focus on Sweeps, avoid EMA in Asia
```

### Monthly
```
[MONTHLY SUMMARY]
Total Trades:     160
Win Rate:         56%
Total PnL:        +$1,380 (+13.8%)
Max Drawdown:     -4.2% (acceptable, <6%)
Days Trading:     20
Days +0%:         12 (60%)
Days -0%:         8 (40%)
Best Day:         +2.1%
Worst Day:        -2.8%
Sharpe Ratio:     ~1.8 (good)
Profit Factor:    1.85 (good, >1.3 is profitable)
```

---

## Troubleshooting Quick Reference

### Problem: No Setups Detected
```
Symptom:  Logs show "[HOLD] No setup signal"
Cause:    Market conditions too quiet or setup parameters too strict
Fix:
  1. Lower min_confidence: 0.65 → 0.60
  2. Check volatility regime (might be LOW)
  3. Add more symbols
  4. Verify bars are loading correctly (echo df.shape)
```

### Problem: Win Rate < 50%
```
Symptom:  Losing more than winning
Cause:    Entry quality too low, too many marginal setups
Fix:
  1. Increase min_confidence: 0.65 → 0.70
  2. Increase reward_ratio: 1.2 → 1.5 (harder targets)
  3. Trade during overlap only (best liquidity)
  4. Focus on LIQUIDITY_SWEEP setups (highest quality)
  5. Skip EMA setups in Asia (lowest quality there)
```

### Problem: Inconsistent Results (±3% daily)
```
Symptom:  Some days +2%, next day -3%
Cause:    Trading too many different conditions
Fix:
  1. Reduce daily loss trigger: 0.06 → 0.04
  2. Reduce max_consecutive_losses: 4 → 2 (stop quicker)
  3. Add session filter (only 08:00-21:00 GMT)
  4. Reduce position size: 0.02 → 0.01 (smoother equity)
```

### Problem: Positions Too Small
```
Symptom:  "Final: 10" (way too small for account)
Cause:    Recent losses or high DD reducing position size
Check:
  1. How many consecutive losses? (reduces by 25% each)
  2. What's daily DD? (-4% → 50% reduction)
  3. How many positions open? (multiple = penalty)
  4. Is Kelly fraction very low? (might be <0.5)
Solution:  Wait for trades to become winning again
  - Kelly automatically increases with wins
  - Loss multiplier decays with each new win
```

---

## Session Trading Strategy

### LONDON OVERLAP (13:00-16:00 GMT) - PRIME TIME
```
✓ Best conditions:
  - Tightest spreads (EURUSD ~1.2 pips)
  - Highest volume (most liquidity)
  - Most setups detected (active price action)
  - All 4 setup types work well
  - M1 scalping optimal

Trading Plan:
  1. Trade M1 aggressively (0.65 confidence OK)
  2. Target: 5-10 trades in 3 hours
  3. Expected: +0.5-1% daily
  4. Risk: Full 2% per trade
  
Notable: Some best days will be during this window
```

### LONDON SESSION (08:00-16:00 GMT)
```
✓ Good conditions:
  - Tight spreads (EURUSD ~1.5 pips)
  - Steady trends (European news)
  - M5 better than M1
  
Trading Plan:
  1. Trade M5 moderate setup (0.63 confidence)
  2. Target: 3-5 trades per day
  3. Expected: +0.3-0.6% daily
  4. Risk: 1.6% per trade (20% less aggressive)
```

### NY SESSION (13:00-21:00 GMT)
```
✓ Good conditions:
  - Moderate spreads (EURUSD ~1.5-2 pips)
  - Strong volume (US participation)
  - Trends clear (macro news)
  
Trading Plan:
  1. Trade M5 moderate setup
  2. Often overlaps with London (even better)
  3. Target: 3-5 trades
  4. Expected: +0.3-0.6% daily
```

### ASIA SESSION (00:00-08:00 GMT)
```
⚠ Challenging conditions:
  - Wide spreads (EURUSD ~2-3 pips)
  - Lower volume (fewer participants)
  - Ranging market (no clear direction)
  
Trading Plan:
  1. Avoid M1 (slippage too high)
  2. Trade 1H only (gives more time)
  3. Higher quality gate (0.68 confidence)
  4. Risk: 1% only (half normal)
  5. Target: 1-2 trades if any
  6. Expected: +0.1-0.3% daily if lucky
  
Alternative: SKIP ASIA (most traders do)
```

### OFF-HOURS (Other)
```
❌ Avoid trading:
  - Lowest volume
  - Widest spreads
  - No institutional participation
  
If you must trade:
  1. Only 4H timeframe
  2. Major levels only (0.68 confidence)
  3. Risk: 1% max
  4. Usually skip entirely
```

---

## Quick Decision Tree

```
START
  │
  ├─ Is scalping ENABLED? 
  │  NO  → Run normal ML signals only
  │  YES → Continue
  │
  ├─ Is volatility EXTREME?
  │  YES → SKIP (too risky)
  │  NO  → Continue
  │
  ├─ Did we have 4+ consecutive losses?
  │  YES → STOP (circuit breaker)
  │  NO  → Continue
  │
  ├─ Is daily loss < -6%?
  │  YES → STOP (capital preservation)
  │  NO  → Continue
  │
  ├─ Is spread > 5 pips?
  │  YES → SKIP (slippage kills profit)
  │  NO  → Continue
  │
  ├─ Setup detected + ML signal matches?
  │  NO  → HOLD (wait for next signal)
  │  YES → Continue
  │
  ├─ Setup quality score > 0.60?
  │  NO  → SKIP (low quality)
  │  YES → Continue
  │
  ├─ R:R ratio ≥ 1.2?
  │  NO  → SKIP (bad risk/reward)
  │  YES → Continue
  │
  ├─ Calculate adaptive position size
  │  (Kelly fraction + drawdown adjustments)
  │
  └─ EXECUTE TRADE ✓
     │
     ├─ Take Profit hit? → CLOSE
     ├─ Stop Loss hit?   → CLOSE
     ├─ 5 bars no move?  → CLOSE (M1)
     ├─ 3 bars no move?  → CLOSE (M5)
     └─ Breakeven + trail
```

---

## Confidence Gate Reference

```
Current Settings:
  Overlap:  0.60 confidence minimum
  London:   0.63 confidence minimum
  NY:       0.63 confidence minimum
  Asia:     0.68 confidence minimum

Translation:
  0.60 = 60% AI sure, more trades, lower quality
  0.63 = 63% AI sure, balanced (RECOMMENDED)
  0.68 = 68% AI sure, fewer trades, higher quality
  0.70 = 70% AI sure, very selective, best win rate

If you're struggling:
  Win Rate < 50% → Increase to 0.70 (too many bad trades)
  No Trades     → Decrease to 0.60 (too strict)
  Sweet Spot    → Keep at 0.65-0.67
```

---

## Gold Standards

### Best Case Scenario
```
Time: 13:30 GMT (London/NY Overlap)
Setup: LIQUIDITY_SWEEP (0.85 base confidence)
ML: BUY signal (0.75 confidence) 
Volatility: NORMAL
Spread: 1.2 pips (tight!)
RR Ratio: 1.3 (4 pips target / 3 pips stop)
Result: +135 pips = 100% win

Probability: 70%+ win rate expected
```

### Worst Case Scenario
```
Time: 03:00 GMT (Asia low liquidity)
Setup: EMA_BREAKOUT (0.68 base confidence)
ML: SELL signal (0.59 confidence - marginal)
Volatility: HIGH (post-news)
Spread: 4 pips (wide!)
RR Ratio: 0.9 (not even 1:1 reward/risk)
Result: Stop hit, -180 pips

Probability: Avoid - circuit breaker should skip
```

---

End of Cheat Sheet
