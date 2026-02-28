#!/usr/bin/env python3
"""
Train ML models for all forex pairs - Simplified Version
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
import sys

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from func import apply_features

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
except ImportError:
    import os
    os.system("pip install scikit-learn lightgbm -q")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb


def train_pair_models(symbol: str, data_file: str) -> bool:
    """Train all timeframe models for a specific pair"""
    
    print(f"\n{'='*60}")
    print(f"Training Models for {symbol}")
    print('='*60)
    
    try:
        # Load data
        print(f"  📖 Loading data...")
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        
        if len(df) < 1000:
            print(f"    ✗ Insufficient data: {len(df)} rows")
            return False
        
        print(f"    ✓ Loaded {len(df)} records")
        
        # Apply features
        print(f"  🔧 Preparing features...")
        df = apply_features(df)
        print(f"    ✓ Applied features and targets")
        
        # Get feature and target columns
        target_cols = ['T_5M', 'T_10M', 'T_15M', 'T_20M', 'T_30M']
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [c for c in df.columns if c not in target_cols + ohlcv_cols + ['Date']]
        
        # Remove NaN rows
        valid_data = df.dropna()
        
        if len(valid_data) < 500:
            print(f"    ✗ Insufficient valid data after NaN removal: {len(valid_data)} rows")
            return False
        
        print(f"    ✓ Using {len(feature_cols)} features, {len(valid_data)} valid rows")
        
        # Prepare storage
        models_dir = Path("ALL_MODELS") / symbol
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Train models
        print(f"  🤖 Training models for {len(target_cols)} timeframes...")
        successful = 0
        
        for target in target_cols:
            try:
                # Prepare data
                X = valid_data[feature_cols].astype(np.float32)
                y = valid_data[target].astype(int)
                
                # Check if we have both classes
                if len(y.unique()) < 2:
                    print(f"    ⚠️  {target}: Insufficient class diversity, skipping")
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = lgb.LGBMClassifier(
                    n_estimators=150,
                    max_depth=7,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1,
                    n_jobs=-1
                )
                
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                # Save model and scaler
                joblib.dump(model, models_dir / f"{target}.joblib")
                joblib.dump(scaler, models_dir / f"{target}_scaler.joblib")
                
                print(f"    ✓ {target}: Train={train_score:.3f}, Test={test_score:.3f}")
                successful += 1
                
            except Exception as e:
                print(f"    ✗ {target}: {str(e)}")
        
        # Save feature columns
        with open(models_dir / "features.txt", 'w') as f:
            f.write('\n'.join(feature_cols))
        
        if successful > 0:
            print(f"  ✓ Successfully trained {successful}/{len(target_cols)} models")
            return True
        else:
            print(f"  ✗ Failed to train any models")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Train models for all pairs"""
    
    pairs = {
        "EURUSD": "CSV_FILES/MT5_5M_BT_EURUSD_Dataset.csv",
        "GBPUSD": "CSV_FILES/MT5_5M_BT_GBPUSD_Dataset.csv",
        "USDJPY": "CSV_FILES/MT5_5M_BT_USDJPY_Dataset.csv",
        "AUDUSD": "CSV_FILES/MT5_5M_BT_AUDUSD_Dataset.csv",
        "NZDUSD": "CSV_FILES/MT5_5M_BT_NZDUSD_Dataset.csv",
        "USDCAD": "CSV_FILES/MT5_5M_BT_USDCAD_Dataset.csv",
        "USDHKD": "CSV_FILES/MT5_5M_BT_USDHKD_Dataset.csv",
        "EURGBP": "CSV_FILES/MT5_5M_BT_EURGBP_Dataset.csv",
        "EURJPY": "CSV_FILES/MT5_5M_BT_EURJPY_Dataset.csv",
        "GBPJPY": "CSV_FILES/MT5_5M_BT_GBPJPY_Dataset.csv",
        "AUDNZD": "CSV_FILES/MT5_5M_BT_AUDNZD_Dataset.csv"
    }
    
    print("\n" + "="*80)
    print("MULTI-PAIR ML MODEL TRAINING")
    print("="*80)
    print(f"Training {len(pairs)} forex pairs | LightGBM Classifiers")
    print("="*80)
    
    successful = 0
    failed = 0
    
    for i, (symbol, data_file) in enumerate(pairs.items(), 1):
        print(f"\n[{i}/{len(pairs)}]")
        
        if train_pair_models(symbol, data_file):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"✓ Successful: {successful}/{len(pairs)}")
    print(f"✗ Failed: {failed}/{len(pairs)}")
    
    # List all models
    print("\nTrained Models by Pair:")
    models_dir = Path("ALL_MODELS")
    if models_dir.exists():
        for pair_dir in sorted(models_dir.iterdir()):
            if pair_dir.is_dir():
                models = list(pair_dir.glob("T_*.joblib"))
                if models:
                    print(f"  ✓ {pair_dir.name}: {len(models)} timeframe models")
    
    print("\n" + "="*80)
    print("✓ Model training complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
