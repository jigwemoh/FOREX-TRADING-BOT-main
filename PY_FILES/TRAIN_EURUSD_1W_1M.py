#!/usr/bin/env python3
"""
EURUSD 1W and 1M XGBoost Training
Uses real yfinance data for weekly and monthly
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import warnings
import json
from typing import Dict, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
import xgboost as xgb
import shap

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))


class EURUSDHighTFTrainer:
    """Train 1W and 1M models from real yfinance data"""
    
    def __init__(self):
        script_dir = Path(__file__).parent.parent
        self.csv_dir = script_dir / "CSV_FILES"
        self.models_dir = script_dir / "ALL_MODELS"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.files = {
            '1W': "EURUSD_Weekly.csv",
            '1M': "EURUSD_Monthly.csv",
        }
    
    def load_data(self, timeframe: str) -> Optional[pd.DataFrame]:
        """Load timeframe data"""
        filepath = self.csv_dir / self.files[timeframe]
        
        if not filepath.exists():
            print(f"  ✗ File not found: {filepath}")
            return None
        
        print(f"  Loading {filepath.name}...", end=" ", flush=True)
        
        try:
            # Read with header and skip metadata row if present
            df = pd.read_csv(filepath)
            
            # Remove metadata rows (where Open column contains non-numeric values)
            df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
            df = df.dropna(subset=['Open'])
            
            # Ensure proper columns
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col not in df.columns:
                    matching = [c for c in df.columns if c.lower() == col.lower()]
                    if matching:
                        df = df.rename(columns={matching[0]: col})
                    else:
                        df[col] = 0
            
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Close', 'Date'])
            
            print(f"✓ Loaded {len(df):,} bars")
            return df
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def apply_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply SMC features"""
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure Date is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        df = df.fillna(0)
        
        # Momentum
        df['Returns'] = df['Close'].pct_change()
        df['EMA_20'] = df['Close'].ewm(span=20, min_periods=1).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, min_periods=1).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, min_periods=1).mean()
        
        # Volatility
        df['ATR'] = (df['High'] - df['Low']).rolling(14, min_periods=1).mean()
        df['ATR_Pct'] = (df['ATR'] / df['Close']).fillna(0)
        df['BB_Mid'] = df['Close'].rolling(20, min_periods=1).mean()
        df['BB_Std'] = df['Close'].rolling(20, min_periods=1).std().fillna(0)
        df['BB_H'] = df['BB_Mid'] + (2 * df['BB_Std'])
        df['BB_L'] = df['BB_Mid'] - (2 * df['BB_Std'])
        
        # Volume
        df['Volume_MA'] = df['Volume'].rolling(20, min_periods=1).mean()
        df['Volume_Ratio'] = (df['Volume'] / df['Volume_MA']).fillna(1)
        
        # Time
        if 'Date' in df.columns:
            df['Hour'] = df['Date'].dt.hour
            df['Weekday'] = df['Date'].dt.weekday
            df['Month'] = df['Date'].dt.month
            df['Date_ordinal'] = df['Date'].astype(np.int64) // 10**9
        
        # Structure
        df['Swing_High'] = df['High'].rolling(5, center=True, min_periods=1).max()
        df['Swing_Low'] = df['Low'].rolling(5, center=True, min_periods=1).min()
        df['Distance_to_High'] = (df['Swing_High'] - df['Close']).fillna(0)
        df['Distance_to_Low'] = (df['Close'] - df['Swing_Low']).fillna(0)
        
        return df.fillna(0)
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create TP-before-SL target"""
        df = df.copy()
        df['Target'] = 0
        
        close_vals = df['Close'].values
        atr_vals = df['ATR'].values
        high_vals = df['High'].values
        low_vals = df['Low'].values
        
        # For longer timeframes, use smaller lookahead
        lookahead = min(30, len(df) // 10)
        
        for i in range(len(df) - lookahead):
            current_close = float(close_vals[i])
            atr = max(float(atr_vals[i]), current_close * 0.001)
            
            tp = current_close + (atr * 2.0)
            sl = current_close - (atr * 1.0)
            
            for j in range(i + 1, min(i + lookahead + 1, len(df))):
                high = float(high_vals[j])
                low = float(low_vals[j])
                
                if high >= tp:
                    df.loc[i, 'Target'] = 1
                    break
                elif low <= sl:
                    df.loc[i, 'Target'] = 0
                    break
        
        return df
    
    def train_timeframe(self, tf: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Train model for timeframe"""
        print(f"\n{'='*60}")
        print(f"EURUSD {tf} - XGBoost Trade Quality")
        print(f"{'='*60}")
        
        try:
            # Check data
            if len(df) < 50:
                print(f"✗ Insufficient data: {len(df)} bars < 50 minimum")
                return None
            
            # Prepare
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Target' in numeric_cols:
                numeric_cols.remove('Target')
            
            X = df[numeric_cols].fillna(0)
            y = df['Target'].astype(int)
            
            print(f"Bars: {len(df)}")
            print(f"Features: {len(numeric_cols)}")
            target_counts = y.value_counts().to_dict()
            print(f"Target: {target_counts}")
            
            # Check class balance
            if len(target_counts) < 2:
                print(f"✗ Only one class in target")
                return None
            
            # Split (larger test for smaller datasets)
            test_pct = 0.3 if len(df) < 100 else 0.15
            val_pct = 0.15 if len(df) > 100 else 0.10
            train_pct = 1.0 - test_pct - val_pct
            
            train_size = int(len(df) * train_pct)
            val_size = int(len(df) * val_pct)
            
            X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
            X_val, y_val = X.iloc[train_size:train_size+val_size], y.iloc[train_size:train_size+val_size]
            X_test, y_test = X.iloc[train_size+val_size:], y.iloc[train_size+val_size:]
            
            print(f"Train/Val/Test: {len(X_train)} / {len(X_val)} / {len(X_test)}")
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Class weight
            pos_count = y_train.sum()
            neg_count = len(y_train) - pos_count
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            
            # Train
            print("Training...", end=" ", flush=True)
            model = xgb.XGBClassifier(
                n_estimators=150,  # Reduced for smaller datasets
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=20
            )
            
            model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
            print("✓")
            
            # Evaluate
            if len(X_test) > 0:
                y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                try:
                    auc = roc_auc_score(y_test, y_test_proba)
                except:
                    auc = 0.5
                
                try:
                    ll = log_loss(y_test, y_test_proba, labels=[0, 1])
                except:
                    ll = -np.mean(y_test * np.log(y_test_proba + 1e-6) + (1-y_test) * np.log(1-y_test_proba + 1e-6))
                
                print(f"AUC: {auc:.4f} | LogLoss: {ll:.4f}")
            else:
                auc = 0.5
                ll = 0.7
                print(f"✗ Insufficient test data")
            
            # Threshold optimization
            best_threshold = 0.50
            best_expectancy = 0
            
            if len(X_test) > 10:
                for threshold in np.arange(0.50, 0.76, 0.05):
                    y_pred = (y_test_proba >= threshold).astype(int)
                    if y_pred.sum() == 0:
                        continue
                    
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                    win_rate = tp / (tp + fp) if (tp + fp) > 0 else 0
                    loss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
                    expectancy = (win_rate * 2.0) - (loss_rate * 1.0)
                    
                    if expectancy > best_expectancy:
                        best_expectancy = expectancy
                        best_threshold = threshold
            
            print(f"Optimal threshold: {best_threshold:.2f} (expectancy {best_expectancy:+.4f})")
            
            # Save
            symbol_dir = self.models_dir / "EURUSD"
            symbol_dir.mkdir(exist_ok=True)
            
            tf_map = {'1W': 'T_1W', '1M': 'T_1M'}
            tf_name = tf_map[tf]
            
            joblib.dump(model, symbol_dir / f"{tf_name}.joblib")
            joblib.dump(scaler, symbol_dir / f"{tf_name}_scaler.joblib")
            joblib.dump(numeric_cols, symbol_dir / f"{tf_name}_features.joblib")
            
            result = {
                'auc_score': float(auc),
                'log_loss': float(ll),
                'optimal_threshold': float(best_threshold),
                'expectancy': float(best_expectancy),
                'n_features': len(numeric_cols),
                'n_bars': len(df)
            }
            
            return result
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    print("\n" + "="*80)
    print("EURUSD 1W AND 1M XGBOOST TRAINING")
    print("Real yfinance data")
    print("="*80)
    
    trainer = EURUSDHighTFTrainer()
    results = {}
    
    for tf in ['1W', '1M']:
        df = trainer.load_data(tf)
        if df is None:
            continue
        
        df = trainer.apply_features(df)
        df = trainer.create_target(df)
        
        result = trainer.train_timeframe(tf, df)
        if result:
            results[tf] = result
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    
    if results:
        summary_df = pd.DataFrame(results).T
        print(summary_df[['auc_score', 'optimal_threshold', 'expectancy', 'n_bars']].to_string())
    else:
        print("✗ No models trained")
    
    # Merge with existing metadata
    symbol_dir = Path(__file__).parent.parent / "ALL_MODELS" / "EURUSD"
    metadata_file = symbol_dir / "model_metadata.json"
    
    if metadata_file.exists():
        import json
        with open(metadata_file) as f:
            existing = json.load(f)
        existing.update(results)
        results = existing
    
    with open(metadata_file, 'w') as f:
        import json
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Metadata saved to {metadata_file}")
    print("✓ Training complete!")


if __name__ == "__main__":
    main()
