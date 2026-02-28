"""
Smart Money Concept (SMC) Strategy Implementation
Smart Money refers to institutional traders and large operators who move the market.
Key concepts:
1. Order Blocks (OB) - Accumulation/Distribution zones
2. Breaker Blocks (BB) - Mitigated order blocks
3. Fair Value Gaps (FVG) - Inefficiencies in price action
4. Supply/Demand Zones - Institutional levels
5. Break of Structure (BoS) - Directional moves
"""

import numpy as np
import pandas as pd
from func import AnalysisResults, apply_features, create_targets, analyze_results, SYMBOL


def identify_liquidity_levels(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Identify liquidity levels where smart money operates
    """
    df['swing_high'] = df['High'].rolling(window=window, center=True).max()
    df['swing_low'] = df['Low'].rolling(window=window, center=True).min()
    
    # Liquidity voids (support/resistance)
    df['liquidity_level'] = np.where(
        df['High'] == df['swing_high'],
        df['High'],
        np.where(df['Low'] == df['swing_low'], df['Low'], np.nan)
    )
    
    return df


def identify_order_blocks(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Order blocks are strong impulse moves followed by consolidation.
    These are areas where smart money entered.
    """
    df['impulse_bull'] = ((df['Close'] > df['Open']) & 
                          (abs(df['Close'] - df['Open']) > 0.5 * (df['High'] - df['Low'])))
    df['impulse_bear'] = ((df['Close'] < df['Open']) & 
                          (abs(df['Close'] - df['Open']) > 0.5 * (df['High'] - df['Low'])))
    
    # Order block high (bearish - sell order block)
    df['order_block_high'] = df['High'].rolling(window=lookback).max()
    # Order block low (bullish - buy order block)
    df['order_block_low'] = df['Low'].rolling(window=lookback).min()
    
    return df


def identify_fair_value_gaps(df: pd.DataFrame, min_gap_pips: float = 0.5, pip_value: float = 0.0001) -> pd.DataFrame:
    """
    Fair Value Gaps (FVG) are imbalances between candles.
    Example: Gap up (bullish FVG) or Gap down (bearish FVG)
    These are premium/discount areas that smart money fills later.
    """
    # Bullish FVG: current Low > previous High
    df['bullish_fvg'] = df['Low'] > df['High'].shift(1)
    df['bullish_fvg_high'] = np.where(df['bullish_fvg'], df['High'].shift(1), np.nan)
    df['bullish_fvg_low'] = np.where(df['bullish_fvg'], df['Low'], np.nan)
    
    # Bearish FVG: current High < previous Low
    df['bearish_fvg'] = df['High'] < df['Low'].shift(1)
    df['bearish_fvg_high'] = np.where(df['bearish_fvg'], df['High'], np.nan)
    df['bearish_fvg_low'] = np.where(df['bearish_fvg'], df['Low'].shift(1), np.nan)
    
    return df


def identify_break_of_structure(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Break of Structure (BoS) identifies when price breaks previous swing highs/lows
    indicating directional continuation or reversal
    """
    df['prev_swing_high'] = df['High'].rolling(window=lookback).max().shift(1)
    df['prev_swing_low'] = df['Low'].rolling(window=lookback).min().shift(1)
    
    # Upside BoS
    df['bos_up'] = df['Close'] > df['prev_swing_high']
    # Downside BoS
    df['bos_down'] = df['Close'] < df['prev_swing_low']
    
    return df


def identify_mitigated_order_blocks(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Mitigated Order Blocks are previous order blocks that price has already touched
    These are areas where smart money may re-enter
    """
    df['ob_high'] = df['High'].rolling(window=window).max().shift(1)
    df['ob_low'] = df['Low'].rolling(window=window).min().shift(1)
    
    # Price mitigates OB when it enters the zone
    df['ob_mitigated_bullish'] = df['Low'] <= df['ob_low']
    df['ob_mitigated_bearish'] = df['High'] >= df['ob_high']
    
    return df


def smc_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all SMC features to dataframe
    """
    df = identify_liquidity_levels(df, window=20)
    df = identify_order_blocks(df, lookback=50)
    df = identify_fair_value_gaps(df, min_gap_pips=5)
    df = identify_break_of_structure(df, lookback=20)
    df = identify_mitigated_order_blocks(df, window=30)
    
    return df


def smc_signal_generator(df: pd.DataFrame, atr_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Generate trading signals based on SMC concepts
    
    LONG signals:
    - Price breaks above swing high (BoS up)
    - Price fills bullish FVG
    - Price near mitigated bullish order block + bullish impulse
    
    SHORT signals:
    - Price breaks below swing low (BoS down)
    - Price fills bearish FVG
    - Price near mitigated bearish order block + bearish impulse
    """
    
    signals: np.ndarray = np.zeros(len(df))
    
    for i in range(1, len(df)):
        bos_up = df.iloc[i]['bos_up']
        bos_down = df.iloc[i]['bos_down']
        bullish_fvg = df.iloc[i]['bullish_fvg']
        bearish_fvg = df.iloc[i]['bearish_fvg']
        ob_mit_bull = df.iloc[i]['ob_mitigated_bullish']
        ob_mit_bear = df.iloc[i]['ob_mitigated_bearish']
        impulse_bull = df.iloc[i]['impulse_bull']
        impulse_bear = df.iloc[i]['impulse_bear']
        
        # LONG signal (1)
        if (bos_up or bullish_fvg or (ob_mit_bull and impulse_bull)):
            signals[i] = 1
        # SHORT signal (-1)
        elif (bos_down or bearish_fvg or (ob_mit_bear and impulse_bear)):
            signals[i] = -1
        # NEUTRAL (0)
        else:
            signals[i] = 0
    
    df['SMC_Signal'] = signals
    return df


def backtest_smc_strategy(df: pd.DataFrame, atr_sl: float = 1.5, atr_tp: float = 4.5, spread_pips: float = 1.2, slippage_pips: float = 0.2, pip_value: float = 0.0001) -> list[tuple[str, str, int, int]]:
    """
    Backtest the SMC strategy
    """
    trades: list[tuple[str, str, int, int]] = []
    spread = spread_pips * pip_value
    slippage = slippage_pips * pip_value
    i = 0
    
    while i < len(df) - 1:
        row = df.iloc[i]
        signal = row['SMC_Signal']
        
        if signal == 0:  # No signal
            i += 1
            continue
        
        next_row = df.iloc[i + 1]
        atr = row['ATR']
        
        if signal == 1:  # LONG
            direction = "BUY"
            entry = next_row["Open"] + spread + slippage
            sl = entry - (atr_sl * atr)
            tp = entry + (atr_tp * atr)
        else:  # SHORT (signal == -1)
            direction = "SELL"
            entry = next_row["Open"] - slippage
            sl = entry + (atr_sl * atr)
            tp = entry - (atr_tp * atr)
        
        # Look for exit
        for j in range(i + 1, min(i + 200, len(df))):  # Max 200 candles
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


# Run the SMC strategy backtest
if __name__ == "__main__":
    print("\n" + "="*80)
    print("SMART MONEY CONCEPT (SMC) STRATEGY BACKTEST")
    print("="*80 + "\n")
    
    # Load data
    backtest_data: pd.DataFrame = pd.read_csv(f"CSV_FILES/MT5_5M_BT_{SYMBOL}_Dataset.csv")  # type: ignore[reportUnknownMemberType]
    
    # Apply features
    backtest_df = apply_features(backtest_data)
    backtest_df = create_targets(backtest_df)
    
    # Apply SMC features
    backtest_df = smc_strategy_features(backtest_df)
    
    # Generate signals
    backtest_df = smc_signal_generator(backtest_df)
    
    # Fill NaN values with 0 for SMC columns (no signal)
    smc_columns = ['liquidity_level', 'bullish_fvg', 'bearish_fvg', 'bos_up', 'bos_down', 
                   'ob_mitigated_bullish', 'ob_mitigated_bearish', 'SMC_Signal']
    for col in smc_columns:
        if col in backtest_df.columns:
            backtest_df[col] = backtest_df[col].fillna(0)  # type: ignore[reportUnknownMemberType]
    
    # Remove NaN values from other columns
    backtest_df = backtest_df.dropna(subset=['ATR', 'High', 'Low', 'Open', 'Close', 'impulse_bull', 'impulse_bear'])  # type: ignore[reportUnknownMemberType]
    
    # Run backtest
    print(f"Backtesting SMC Strategy on {SYMBOL}")
    print(f"Total candles: {len(backtest_df)}")
    print(f"Backtest period: {backtest_df.index[0]} to {backtest_df.index[-1]}\n")
    
    results: list[tuple[str, str, int, int]] = backtest_smc_strategy(backtest_df)
    
    print(f"\nStrategy: Smart Money Concept (SMC)")
    print("="*80)
    analysis: AnalysisResults = analyze_results(results)
    print("="*80)
    
    # Additional statistics
    if len(results) > 0:
        long_trades = [t for t in results if t[1] == "BUY"]
        short_trades = [t for t in results if t[1] == "SELL"]
        
        long_wins = sum(1 for t in long_trades if t[0] == "WIN")
        short_wins = sum(1 for t in short_trades if t[0] == "WIN")
        
        print(f"\nDetailed Stats:")
        print(f"Long (BUY) trades:  {len(long_trades):3d} | Wins: {long_wins:3d} | Win Rate: {(long_wins/len(long_trades)*100 if long_trades else 0):6.2f}%")
        print(f"Short (SELL) trades: {len(short_trades):3d} | Wins: {short_wins:3d} | Win Rate: {(short_wins/len(short_trades)*100 if short_trades else 0):6.2f}%")
        print(f"\nTotal trades placed: {len(results)}")
        print(f"Signal occurrences: {sum(backtest_df['SMC_Signal'] != 0)}")
