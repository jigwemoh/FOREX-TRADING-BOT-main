# XGBoost Trade Quality Model

## Overview

This implementation follows systematic trading best practices for ML-powered trade filtering. Instead of predicting price direction, the model estimates **P(TP hit before SL | valid SMC setup)** — trade quality.

## Key Principles

### 1. Problem Formulation
- **Type**: Binary classification
- **Label**: 
  - `1` → TP hit before SL
  - `0` → SL hit before SL
- **Why**: Matches real trading outcomes, stable across regimes

### 2. Model Choice
- **XGBoost** (tree-based gradient boosting)
- Handles non-linear SMC feature interactions
- Interpretable via SHAP
- Robust on medium datasets

### 3. Feature Engineering (SMC-Focused)
**Included:**
- Order Block detection (bullish/bearish OB)
- Fair Value Gaps (FVG size, direction)
- Liquidity sweeps (depth, frequency)
- HTF alignment (trend context)
- Session detection (London/NY/Asia)
- ATR-based volatility
- Volume analysis

**Explicitly Excluded:**
- Post-trade features (exit_price, etc.)
- Raw OHLC (use ratios/structure instead)
- Future-looking data

### 4. Data Splitting (Time-Based)
```
Train:      Oldest 70%  (e.g., 2022-01 to 2024-06)
Validation: Next 15%    (e.g., 2024-06 to 2024-09)
Test:       Recent 15%  (e.g., 2024-09 to 2025-01)
```
**Critical**: No random shuffle — prevents lookahead bias.

### 5. Class Imbalance
- Expected win rate: 45-60%
- Uses `scale_pos_weight` in XGBoost (NOT oversampling)
- Preserves probability calibration

### 6. XGBoost Configuration
```python
XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=auto,  # Based on class ratio
    objective="binary:logistic",
    eval_metric="logloss",
    early_stopping_rounds=30
)
```

### 7. Metrics That Matter
| Metric | Why |
|--------|-----|
| **ROC-AUC** | Ranking quality (must be > 0.55) |
| **Log Loss** | Probability calibration |
| **Expectancy** | Real money impact |

**Ignore**: Accuracy, F1 (misleading in trading)

### 8. Threshold Optimization
The model doesn't trade at 0.50 probability. It finds the **profit-maximizing threshold**.

**Example thresholds:**
- `0.55` → Too many trades, noisy
- `0.60` → Decent
- `0.65` → Selective (recommended)
- `0.70+` → Rare, very high quality

**Expectancy formula:**
```
expectancy = (win_rate × avg_win) − (loss_rate × avg_loss)
```

Optimal threshold = max(expectancy) with acceptable trade frequency.

### 9. Live Trading Decision Logic
```python
if (
    smc_rules_passed
    and model_proba >= optimal_threshold  # From training
    and rr_ratio >= 1.8
    and session in ["London", "NY"]
):
    execute_trade()
```

**ML never overrides SMC** — it filters.

### 10. SHAP Explainability
```python
import shap

explainer = shap.Explainer(model)
shap_values = explainer(X_sample)
```

**Expected top features:**
1. Liquidity sweep depth
2. OB distance
3. FVG size
4. Session timing
5. HTF alignment

If not → feature logic needs review.

### 11. Retraining Policy
**When to retrain:**
- Every 2-4 weeks (scheduled)
- After >20% drawdown
- After macro regime change (rates spike, volatility shift)

**Store metadata:**
- Model version
- Training date
- Dataset size
- Optimal threshold

### 12. Failure Modes (Stop Trading If)
- AUC < 0.55 (model worse than random)
- SHAP shows random feature dominance
- Live win rate deviates >15% from backtest
- Trade frequency spikes suddenly

## Usage

### Training
```bash
cd PY_FILES
python TRAIN_XGBOOST_QUALITY.py
```

**Output:**
- XGBoost models saved to `ALL_MODELS/{symbol}/T_*.joblib`
- Feature list: `ALL_MODELS/{symbol}/features.joblib`
- Metadata: `ALL_MODELS/{symbol}/model_metadata.json`

### Live Trading
The live trader (`AUTO_TRADER_MULTI.py`) automatically:
1. Loads XGBoost models
2. Calculates SMC features
3. Gets probability from model
4. Filters trades: `if prob >= ml_threshold and smc_valid`

**Config (`config.json`):**
```json
{
  "trading": {
    "timeframe": "1H",
    "use_ml": true,
    "ml_threshold": 0.65
  }
}
```

### Monitoring
Check logs for:
```
[ML CONFIG] BTCUSD | timeframe=1H (T_1H) | threshold=0.65 | required_features=76
[ML FEATURE GAP] EURUSD missing 12 features; filled with 0.0
```

## Model Performance Benchmarks

**Good model:**
- AUC: 0.60 - 0.75
- Log Loss: < 0.65
- Expectancy: > 0.10
- Win Rate: 52-65%
- Optimal Threshold: 0.60-0.70

**Poor model (retrain):**
- AUC: < 0.55
- Expectancy: < 0
- Trade count: 0 at any reasonable threshold

## Feature Importance (Expected Ranking)

Based on SMC principles, SHAP should rank:

1. **Liquidity_Sweep_High/Low** (key SMC concept)
2. **Sweep_Depth** (magnitude matters)
3. **FVG_Size** (gap significance)
4. **Bullish_OB / Bearish_OB** (structure)
5. **HTF_Bullish / HTF_Bearish** (alignment)
6. **Session_London / Session_NY** (timing)
7. **ATR_Pct** (volatility regime)
8. **Volume_Ratio** (confirmation)

If **Hour** or **Body_Range** dominate → model is overfitting noise.

## References
- **Marcos López de Prado** – *Advances in Financial Machine Learning*
- **CFA Institute** – Time-series ML best practices
- **SHAP Documentation** – https://shap.readthedocs.io

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                       │
├─────────────────────────────────────────────────────────┤
│ Historical OHLCV Data (2+ years)                        │
│           ↓                                              │
│ SMC Feature Engineering (OB, FVG, Liquidity Sweeps)     │
│           ↓                                              │
│ Create TP-before-SL Target (forward scan 50 candles)    │
│           ↓                                              │
│ Time-based Split (70% train / 15% val / 15% test)       │
│           ↓                                              │
│ XGBoost Training (early stopping on val set)            │
│           ↓                                              │
│ Threshold Optimization (max expectancy on test set)     │
│           ↓                                              │
│ SHAP Explainability (validate feature logic)            │
│           ↓                                              │
│ Save: model + scaler + features + metadata              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     LIVE TRADING                        │
├─────────────────────────────────────────────────────────┤
│ MT5 Live Candles (real-time OHLCV)                      │
│           ↓                                              │
│ Calculate SMC Features (same as training)                │
│           ↓                                              │
│ Load XGBoost model + scaler + features                  │
│           ↓                                              │
│ Align features to saved list (fill missing with 0)      │
│           ↓                                              │
│ Scale features (using saved scaler)                     │
│           ↓                                              │
│ Get probability: model.predict_proba()                  │
│           ↓                                              │
│ Trade Decision:                                          │
│   IF prob >= optimal_threshold (e.g., 0.65)             │
│   AND smc_rules_passed                                  │
│   AND rr_ratio >= 1.8                                   │
│   AND session in ["London", "NY"]                       │
│   THEN execute_trade()                                  │
└─────────────────────────────────────────────────────────┘
```

## Why This Works

1. **ML ranks setup quality** (not direction)
2. **SMC controls context** (structure + liquidity)
3. **Risk rules cap damage** (SL/TP via ATR)
4. **Threshold creates asymmetry** (only high-quality setups)

**Result**: Converts SMC from discretionary art to systematic edge.
