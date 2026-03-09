# FOREX Trading Bot - Two Versions

This repository contains two versions of the FOREX trading bot:

## 1. AUTO_TRADER_MULTI.py (Original)
- **Purpose**: Traditional ML-driven trading on H1/4H timeframes
- **Strategy**: Uses machine learning models for longer-term trades
- **Risk**: Higher risk per trade (2%)
- **Frequency**: Fewer trades (1-3 per day per symbol)
- **Timeframes**: 1H, 4H
- **Features**: SMC analysis, ML predictions, trailing stops

## 2. AUTO_TRADER_MULTI_SCALPING.py (New)
- **Purpose**: High-frequency scalping with microstructure analysis
- **Strategy**: Combines ML signals with intraday price action setups
- **Risk**: Lower risk per trade (0.5% conservative, up to 2%)
- **Frequency**: More trades (3-8 per day per symbol)
- **Timeframes**: M1, M5 (intraday)
- **Features**: All original features PLUS scalping engine with:
  - 4 microstructure setups (EMA breakouts, VWAP bounces, liquidity sweeps, rejection candles)
  - Adaptive Kelly criterion position sizing
  - Session-based trading filters
  - Market regime detection
  - Risk circuit breakers

## Quick Start

### For Traditional Trading:
```bash
cd PY_FILES
python AUTO_TRADER_MULTI.py
```

### For Scalping Trading:
```bash
cd PY_FILES
python AUTO_TRADER_MULTI_SCALPING.py
```

## Configuration

Both bots use the same `config.json` file. The scalping version automatically detects and uses scalping settings when enabled.

For conservative scalping testing:
- `scalping.enabled`: true
- `scalping.timeframes`: ["M5"] (start with M5 only)
- `scalping.min_confidence`: 0.68 (higher than default)
- `scalping.risk_per_trade`: 0.005 (0.5% risk)

## Documentation

- `SCALPING_QUICK_START.md`: Complete scalping setup guide
- `SCALPING_SYSTEM_DOCS.py`: Technical documentation
- `SCALPING_CHEAT_SHEET.md`: Daily reference
- `README_SCALPING.md`: Main scalping guide

## Comparison Testing

Run both versions simultaneously to compare performance:
- Use different MT5 accounts or demo accounts
- Monitor win rates, daily returns, and drawdowns
- The scalping version should show more frequent but smaller trades
- Expected scalping performance: 53-58% win rate, 0.5-1% daily return

## Safety Notes

- Start with paper trading/demo accounts
- Use conservative settings for first 3-5 days
- Monitor closely for the first week
- Both versions include comprehensive risk management
- Never risk more than you can afford to lose</content>
<parameter name="filePath">/Users/igwemoh/Downloads/FOREX-TRADING-BOT-main/README_TRADERS.md