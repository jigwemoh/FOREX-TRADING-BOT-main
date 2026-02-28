"""
Hybrid Strategy: SMC + Machine Learning
Combines Smart Money Concept signals with ML model predictions
for better entry confirmation and higher win rate.
"""

import pandas as pd
from typing import Any

import joblib  # type: ignore[import-untyped]

from func import AnalysisResults, Trade, apply_features, create_targets, analyze_results, SYMBOL
from SMC_Strategy import smc_strategy_features, smc_signal_generator


def backtest_hybrid_smc_ml(df: pd.DataFrame, models_dict: dict[str, Any], feature_cols_dict: dict[str, list[str]], threshold: float = 55, atr_sl: float = 1.5, atr_tp: float = 4.5, spread_pips: float = 1.2, slippage_pips: float = 0.2, pip_value: float = 0.0001) -> list[tuple[str, str, int, int]]:
    """
    Hybrid backtest using SMC signals filtered by ML confidence.
    
    models_dict: dict of {target_name: model}
    feature_cols_dict: dict of {target_name: feature_columns}
    """
    trades: list[Trade] = []
    spread = spread_pips * pip_value
    slippage = slippage_pips * pip_value
    i = 0
    
    while i < len(df) - 1:
        row = df.iloc[i]
        smc_signal = row['SMC_Signal']
        
        if smc_signal == 0:  # No SMC signal
            i += 1
            continue
        
        # Get ML confirmation for T_5M target
        target: str = 'T_5M'
        model: Any = models_dict[target]
        feature_cols: list[str] = feature_cols_dict[target]
        
        X = row[feature_cols].to_numpy().reshape(1, -1)
        proba = model.predict_proba(X)[0]
        up_conf, down_conf = proba[1] * 100, proba[0] * 100
        
        # Require ML confirmation with SMC signal
        # SMC signal 1 (LONG) + ML up prediction (up_conf > down_conf)
        # SMC signal -1 (SHORT) + ML down prediction (down_conf > up_conf)
        if smc_signal == 1 and up_conf > down_conf and max(up_conf, down_conf) >= threshold:
            direction = "BUY"
        elif smc_signal == -1 and down_conf > up_conf and max(up_conf, down_conf) >= threshold:
            direction = "SELL"
        else:
            i += 1
            continue
        
        # Trade setup
        next_row = df.iloc[i + 1]
        atr = row["ATR"]
        
        if direction == "BUY":
            entry = next_row["Open"] + spread + slippage
            sl = entry - (atr_sl * atr)
            tp = entry + (atr_tp * atr)
        else:  # SELL
            entry = next_row["Open"] - slippage
            sl = entry + (atr_sl * atr)
            tp = entry - (atr_tp * atr)
        
        # Look for exit
        for j in range(i + 1, min(i + 200, len(df))):
            candle = df.iloc[j]
            
            if direction == "BUY":
                if candle["Low"] <= sl and candle["High"] >= tp:
                    trades.append(("LOSS", direction, i, j))
                    i = j
                    break
                elif candle["Low"] <= sl:
                    trades.append(("LOSS", direction, i, j))
                    i = j
                    break
                elif candle["High"] >= tp:
                    trades.append(("WIN", direction, i, j))
                    i = j
                    break
            else:  # SELL
                candle_high_ask = candle["High"] + spread
                candle_low_ask = candle["Low"] + spread
                
                if candle_high_ask >= sl and candle_low_ask <= tp:
                    trades.append(("LOSS", direction, i, j))
                    i = j
                    break
                elif candle_high_ask >= sl:
                    trades.append(("LOSS", direction, i, j))
                    i = j
                    break
                elif candle_low_ask <= tp:
                    trades.append(("WIN", direction, i, j))
                    i = j
                    break
        else:
            i += 1
    
    return trades


if __name__ == "__main__":
    print("\n" + "="*80)
    print("HYBRID STRATEGY: SMC + MACHINE LEARNING")
    print("="*80 + "\n")
    
    backtest_data: pd.DataFrame = pd.read_csv(f"CSV_FILES/MT5_5M_BT_{SYMBOL}_Dataset.csv")  # type: ignore[reportUnknownMemberType]
    
    # Apply features
    backtest_df = apply_features(backtest_data)
    backtest_df = create_targets(backtest_df)
    
    # Apply SMC features
    backtest_df = smc_strategy_features(backtest_df)
    
    # Generate SMC signals
    backtest_df = smc_signal_generator(backtest_df)
    
    # Fill NaN values for SMC columns
    smc_columns = ['liquidity_level', 'bullish_fvg', 'bearish_fvg', 'bos_up', 'bos_down', 
                   'ob_mitigated_bullish', 'ob_mitigated_bearish', 'SMC_Signal']
    for col in smc_columns:
        if col in backtest_df.columns:
            backtest_df[col] = backtest_df[col].fillna(0)  # type: ignore[reportUnknownMemberType]
    
    # Remove NaN values from required columns
    backtest_df = backtest_df.dropna(subset=['ATR', 'High', 'Low', 'Open', 'Close'])  # type: ignore[reportUnknownMemberType]
    
    # Load trained ML models
    print("Loading trained ML models...\n")
    models_dict: dict[str, Any] = {}
    feature_cols_dict: dict[str, list[str]] = {}
    
    targets = ['T_5M', 'T_10M', 'T_15M', 'T_20M', 'T_30M']
    
    for target in targets:
        try:
            bundle: dict[str, Any] = joblib.load(f"ALL_MODELS/{SYMBOL}_lgbm_{target}.pkl")  # type: ignore[reportUnknownMemberType]
            models_dict[target] = bundle["model"]
            feature_cols_dict[target] = bundle["features"]
            print(f"✓ Loaded {SYMBOL}_lgbm_{target}.pkl")
        except FileNotFoundError:
            print(f"✗ Could not load {SYMBOL}_lgbm_{target}.pkl")
    
    print(f"\nBacktesting Hybrid Strategy on {SYMBOL}")
    print(f"Total candles: {len(backtest_df)}")
    print(f"Backtest period: {backtest_df.index[0]} to {backtest_df.index[-1]}\n")
    
    # Run backtest
    results: list[Trade] = backtest_hybrid_smc_ml(backtest_df, models_dict, feature_cols_dict, threshold=55)
    
    print(f"\nHybrid Strategy Results (SMC + ML)")
    print("="*80)
    analysis: AnalysisResults = analyze_results(results)
    print("="*80)
    
    # Detailed analysis
    if len(results) > 0:
        long_trades = [t for t in results if t[1] == "BUY"]
        short_trades = [t for t in results if t[1] == "SELL"]
        
        long_wins = sum(1 for t in long_trades if t[0] == "WIN")
        short_wins = sum(1 for t in short_trades if t[0] == "WIN")
        
        print(f"\nDetailed Stats:")
        print(f"Long (BUY) trades:  {len(long_trades):3d} | Wins: {long_wins:3d} | Win Rate: {(long_wins/len(long_trades)*100 if long_trades else 0):6.2f}%")
        print(f"Short (SELL) trades: {len(short_trades):3d} | Wins: {short_wins:3d} | Win Rate: {(short_wins/len(short_trades)*100 if short_trades else 0):6.2f}%")
        print(f"\nTotal SMC signals: {sum(backtest_df['SMC_Signal'] != 0)}")
        print(f"Confirmed trades: {len(results)}")
        print(f"Confirmation rate: {(len(results) / max(sum(backtest_df['SMC_Signal'] != 0), 1) * 100):.2f}%")
    else:
        print("\nNo trades generated!")
    
    # Comparison with pure SMC
    print("\n" + "="*80)
    print("COMPARISON: Pure SMC vs Hybrid SMC+ML")
    print("="*80)
    print("\nPure SMC Strategy:")
    print("  Total Trades: 251")
    print("  Win Rate: 13.15%")
    print("\nHybrid SMC+ML Strategy:")
    win_rate_value: float = analysis.get('win_rate', 0.0)
    print(f"  Total Trades: {analysis.get('total_trades', 0)}")
    print(f"  Win Rate: {win_rate_value}%")
    
    improvement: float = win_rate_value - 13.15
    print(f"\nImprovement: {improvement:+.2f}% win rate")
    print("="*80)
