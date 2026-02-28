#!/usr/bin/env python3
"""
Train ML models for all forex pairs
Trains LightGBM classifiers for each pair and timeframe
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
import sys

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from func import apply_features, create_targets

# Try importing required ML libraries
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    _ml_libraries_available = True
except ImportError:
    _ml_libraries_available = False
    print("Installing ML libraries...")
    import os
    os.system("pip install scikit-learn lightgbm -q")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    _ml_libraries_available = True


class PairModelTrainer:
    """Train and save ML models for a specific forex pair"""
    
    def __init__(self, symbol: str, data_file: str):
        self.symbol = symbol
        self.data_file = data_file
        self.df = None
        self.models = {}
        self.scalers = {}
        self.feature_cols = None
        
    def load_data(self) -> bool:
        """Load data from CSV"""
        try:
            print(f"  📖 Loading data from {self.data_file}...")
            self.df = pd.read_csv(self.data_file, index_col=0, parse_dates=True)
            
            if len(self.df) < 1000:
                print(f"    ✗ Insufficient data: {len(self.df)} rows (need at least 1000)")
                return False
            
            print(f"    ✓ Loaded {len(self.df)} records")
            return True
        except Exception as e:
            print(f"    ✗ Error loading data: {str(e)}")
            return False
    
    def prepare_features_targets(self) -> bool:
        """Apply features and create targets"""
        try:
            print(f"  🔧 Preparing features and targets...")
            
            # Apply technical indicators
            self.df = apply_features(self.df)
            print(f"    ✓ Applied {len([c for c in self.df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume']])} features")
            
            # Create multi-timeframe targets
            self.df = create_targets(self.df)
            print(f"    ✓ Created targets for 5 timeframes")
            
            # Get feature columns (all except OHLCV and targets)
            target_cols = ['T_5M', 'T_10M', 'T_15M', 'T_20M', 'T_30M']
            ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            self.feature_cols = [c for c in self.df.columns if c not in target_cols + ohlcv_cols]
            
            print(f"    ✓ Using {len(self.feature_cols)} features for training")
            
            return True
        except Exception as e:
            print(f"    ✗ Error preparing features: {str(e)}")
            return False
    
    def train_model_for_target(self, target_name: str, test_size: float = 0.2) -> bool:
        """Train LightGBM model for specific target"""
        try:
            print(f"    Training {target_name}...")
            
            # Remove NaN rows
            valid_data = self.df.dropna(subset=self.feature_cols + [target_name])
            
            if len(valid_data) < 500:
                print(f"      ✗ Insufficient valid data: {len(valid_data)} rows")
                return False
            
            # Prepare data
            X = valid_data[self.feature_cols].astype(np.float32)
            y = valid_data[target_name].astype(int)
            
            # Handle class imbalance
            class_weights = {}
            unique_classes = np.unique(y)
            total = len(y)
            for cls in unique_classes:
                class_weights[cls] = total / (len(unique_classes) * (y == cls).sum())
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train LightGBM
            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            print(f"      ✓ Train Acc: {train_score:.4f} | Test Acc: {test_score:.4f}")
            
            # Store model and scaler
            self.models[target_name] = model
            self.scalers[target_name] = scaler
            
            return True
        except Exception as e:
            print(f"      ✗ Error training {target_name}: {str(e)}")
            return False
    
    def train_all_models(self) -> bool:
        """Train all timeframe models"""
        try:
            print(f"  🤖 Training models for all timeframes...")
            
            targets = ['T_5M', 'T_10M', 'T_15M', 'T_20M', 'T_30M']
            successful = 0
            
            for target in targets:
                if self.train_model_for_target(target):
                    successful += 1
            
            if successful == 0:
                print(f"    ✗ Failed to train any models")
                return False
            
            print(f"    ✓ Successfully trained {successful}/{len(targets)} models")
            return True
        except Exception as e:
            print(f"    ✗ Error in training: {str(e)}")
            return False
    
    def save_models(self) -> bool:
        """Save trained models and scalers"""
        try:
            print(f"  💾 Saving models...")
            
            models_dir = Path("ALL_MODELS") / self.symbol
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Save models
            for target_name, model in self.models.items():
                model_path = models_dir / f"{target_name}.joblib"
                joblib.dump(model, model_path)
            
            # Save scalers
            for target_name, scaler in self.scalers.items():
                scaler_path = models_dir / f"{target_name}_scaler.joblib"
                joblib.dump(scaler, scaler_path)
            
            # Save feature columns
            feature_path = models_dir / "features.txt"
            with open(feature_path, 'w') as f:
                f.write('\n'.join(self.feature_cols))
            
            print(f"    ✓ Saved {len(self.models)} models and scalers to {models_dir}")
            return True
        except Exception as e:
            print(f"    ✗ Error saving models: {str(e)}")
            return False
    
    def train_and_save(self) -> bool:
        """Complete training pipeline"""
        print(f"\n{'='*60}")
        print(f"Training Models for {self.symbol}")
        print('='*60)
        
        if not self.load_data():
            return False
        
        if not self.prepare_features_targets():
            return False
        
        if not self.train_all_models():
            return False
        
        if not self.save_models():
            return False
        
        print(f"✓ Training complete for {self.symbol}")
        return True


def train_all_pair_models():
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
    print(f"Training LightGBM models for {len(pairs)} forex pairs")
    print("Target timeframes: 5M, 10M, 15M, 20M, 30M")
    print("="*80 + "\n")
    
    successful_pairs = 0
    failed_pairs = 0
    
    for i, (symbol, data_file) in enumerate(pairs.items(), 1):
        print(f"\n[{i}/{len(pairs)}] {symbol}")
        
        trainer = PairModelTrainer(symbol, data_file)
        
        if trainer.train_and_save():
            successful_pairs += 1
        else:
            failed_pairs += 1
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"✓ Successful: {successful_pairs}/{len(pairs)}")
    print(f"✗ Failed: {failed_pairs}/{len(pairs)}")
    
    # List all models
    print("\nTrained Models:")
    models_dir = Path("ALL_MODELS")
    if models_dir.exists():
        for pair_dir in sorted(models_dir.iterdir()):
            if pair_dir.is_dir():
                models = list(pair_dir.glob("T_*.joblib"))
                if models:
                    print(f"  ✓ {pair_dir.name}: {len(models)} models")
    
    print("\n" + "="*80)
    print("✓ Model training complete!")
    print("You can now run multi-pair backtests with: python PY_FILES/MULTI_PAIR_BACKTEST.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    train_all_pair_models()
