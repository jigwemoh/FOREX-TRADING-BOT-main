#!/usr/bin/env python3
"""
Simple ML Model Training for All Forex Pairs
Standalone version with minimal dependencies on func.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings

warnings.filterwarnings('ignore')

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    import ta
except ImportError:
    import os
    print("Installing required packages...")
    os.system("pip install scikit-learn lightgbm ta -q")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    import ta


def create_simple_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create essential features without NaN issues"""
    
    df = df.copy()
    
    # Basic OHLC features
    df['hl_diff'] = df['High'] - df['Low']
    df['oc_diff'] = df['Close'] - df['Open']
    df['body_range_ratio'] = df['oc_diff'] / (df['hl_diff'] + 1e-8)
    df['volume_ma5'] = df['Volume'].rolling(5, min_periods=1).mean()
    df['volume_ratio'] = df['Volume'] / (df['volume_ma5'] + 1e-8)
    
    # Price-based features
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1).fillna(df['Close']))
    df['high_low_ratio'] = df['High'] / df['Low']
    df['close_open_ratio'] = df['Close'] / df['Open']
    
    # Moving averages
    df['sma5'] = df['Close'].rolling(5, min_periods=1).mean()
    df['sma20'] = df['Close'].rolling(20, min_periods=1).mean()
    df['sma50'] = df['Close'].rolling(50, min_periods=1).mean()
    
    # Momentum indicators
    try:
        df['rsi14'] = ta.momentum.rsi(df['Close'], window=14)
    except:
        df['rsi14'] = 50  # Default if ta fails
    
    try:
        df['macd'] = ta.trend.macd_diff(df['Close'])
    except:
        df['macd'] = 0
    
    # ATR
    try:
        df['atr14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    except:
        df['atr14'] = (df['High'] - df['Low']).rolling(14, min_periods=1).mean()
    
    # Fill NaN with forward fill then backward fill
    df = df.ffill().fillna(method='bfill').fillna(0)
    
    # Get feature columns
    feature_cols = [c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    
    return df, feature_cols


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create prediction targets for all timeframes"""
    
    # Multi-horizon targets matching original system
    # T_5M, T_10M, T_15M, T_20M, T_30M
    horizons = {
        "T_5M": 1,   # 1 period ahead (5 minutes)
        "T_10M": 2,  # 2 periods ahead (10 minutes)
        "T_15M": 3,  # 3 periods ahead (15 minutes)
        "T_20M": 4,  # 4 periods ahead (20 minutes)
        "T_30M": 6   # 6 periods ahead (30 minutes)
    }
    
    for target_name, shift in horizons.items():
        future_return = (df['Close'].shift(-shift) - df['Close']) / df['Close']
        df[target_name] = (future_return > 0).astype(int)
    
    # Remove last 6 rows (no future data for 30M target)
    df = df.iloc[:-6].copy()
    
    return df


def train_pair_model(symbol: str, data_file: str) -> bool:
    """Train models for a single pair"""
    
    print(f"\n{'='*60}")
    print(f"{symbol}")
    print('='*60)
    
    try:
        # Load data
        print(f"  Loading {len(pd.read_csv(data_file))} records...", end=" ")
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        print(f"✓")
        
        # Create features
        print(f"  Creating features...", end=" ")
        df, feature_cols = create_simple_features(df)
        print(f"✓ ({len(feature_cols)} features)")
        
        # Create targets
        print(f"  Creating targets...", end=" ")
        df = create_targets(df)
        print(f"✓")
        
        # Check data
        print(f"  Data check: {len(df)} rows...", end=" ")
        if len(df) < 500:
            print(f"✗ (insufficient)")
            return False
        print(f"✓")
        
        # Prepare storage
        models_dir = Path("ALL_MODELS") / symbol
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Train models
        targets = ['T_5M', 'T_10M', 'T_15M', 'T_20M', 'T_30M']
        trained = 0
        
        print(f"  Training models: ", end="")
        for target in targets:
            try:
                # Prepare data
                X = df[feature_cols].astype(np.float32)
                y = df[target].astype(int)
                
                # Need both classes
                if len(y.unique()) < 2:
                    continue
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Scale
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    verbose=-1,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train_scaled, y_train)
                
                # Save with original naming convention
                joblib.dump(model, models_dir / f"{target}.joblib")
                joblib.dump(scaler, models_dir / f"{target}_scaler.joblib")
                
                score = model.score(X_test_scaled, y_test)
                print(f"{target}={score:.3f} ", end="")
                trained += 1
                
            except Exception as e:
                continue
        
        # Save features
        with open(models_dir / "features.txt", 'w') as f:
            f.write('\n'.join(feature_cols))
        
        print(f"\n  Result: ✓ Trained {trained} models")
        return trained > 0
        
    except Exception as e:
        print(f"\n  Error: {str(e)}")
        return False


def main():
    """Train all pair models"""
    
    pairs = [
        ("EURUSD", "CSV_FILES/MT5_5M_BT_EURUSD_Dataset.csv"),
        ("GBPUSD", "CSV_FILES/MT5_5M_BT_GBPUSD_Dataset.csv"),
        ("USDJPY", "CSV_FILES/MT5_5M_BT_USDJPY_Dataset.csv"),
        ("AUDUSD", "CSV_FILES/MT5_5M_BT_AUDUSD_Dataset.csv"),
        ("NZDUSD", "CSV_FILES/MT5_5M_BT_NZDUSD_Dataset.csv"),
        ("USDCAD", "CSV_FILES/MT5_5M_BT_USDCAD_Dataset.csv"),
        ("USDHKD", "CSV_FILES/MT5_5M_BT_USDHKD_Dataset.csv"),
        ("EURGBP", "CSV_FILES/MT5_5M_BT_EURGBP_Dataset.csv"),
        ("EURJPY", "CSV_FILES/MT5_5M_BT_EURJPY_Dataset.csv"),
        ("GBPJPY", "CSV_FILES/MT5_5M_BT_GBPJPY_Dataset.csv"),
        ("AUDNZD", "CSV_FILES/MT5_5M_BT_AUDNZD_Dataset.csv")
    ]
    
    print("\n" + "="*80)
    print("MULTI-PAIR ML MODEL TRAINING")
    print("="*80)
    print(f"Training {len(pairs)} forex pairs with LightGBM\n")
    
    successful = 0
    for i, (symbol, data_file) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}]", end=" ")
        if train_pair_model(symbol, data_file):
            successful += 1
    
    # Summary
    print("\n" + "="*80)
    print(f"RESULTS: {successful}/{len(pairs)} pairs trained successfully")
    print("="*80 + "\n")
    
    # List models
    models_dir = Path("ALL_MODELS")
    if models_dir.exists():
        print("Trained Models:")
        for pair_dir in sorted(models_dir.iterdir()):
            if pair_dir.is_dir():
                models = list(pair_dir.glob("T_*.joblib"))
                if models:
                    print(f"  ✓ {pair_dir.name}: {len(models)} timeframe models")


if __name__ == "__main__":
    main()
