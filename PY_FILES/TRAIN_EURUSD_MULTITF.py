#!/usr/bin/env python3
"""
EURUSD Multi-Timeframe XGBoost Training
Aggregates real 5M data into 1H, 4H, 1D, 1W, 1M timeframes
Trains separate models for each timeframe
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import warnings
import json
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
import xgboost as xgb
import shap

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))


class EURUSDMultiTimeframeTrainer:
    """Train EURUSD models for all timeframes from real 5M data"""
    
    def __init__(self):
        script_dir = Path(__file__).parent.parent
        self.data_file = script_dir / "CSV_FILES/MT5_5M_EURUSD_Exchange_Rate_Dataset.csv"
        self.models_dir = script_dir / "ALL_MODELS"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Timeframe aggregation rules
        self.timeframe_agg = {
            '5M': None,  # Use raw 5M data
            '1H': 12,    # 12 * 5M = 1H
            '4H': 48,    # 48 * 5M = 4H
            '1D': 288,   # 288 * 5M = 1D
            '1W': 2016,  # 2016 * 5M = 1W
            '1M': 8640,  # 8640 * 5M = 1M (30 days)
        }
    
    def load_raw_data(self) -> Optional[pd.DataFrame]:
        """Load raw 5M EURUSD data"""
        if not self.data_file.exists():
            print(f"✗ File not found: {self.data_file}")
            return None
        
        print(f"Loading {self.data_file.name}...", end=" ", flush=True)
        df = pd.read_csv(self.data_file, header=None)
        
        if len(df.columns) == 6:
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        elif len(df.columns) == 5:
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close']
            df['Volume'] = 0
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Close'])
        df = df.reset_index(drop=True)
        
        print(f"✓ Loaded {len(df):,} bars")
        return df
    
    def aggregate_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Aggregate 5M data to target timeframe"""
        df = df.copy()
        
        if timeframe == '5M':
            return df
        
        agg_size = self.timeframe_agg[timeframe]
        
        # Resample using groupby with aggregation
        df['Group'] = df.index // agg_size
        
        agg_df = df.groupby('Group').agg({
            'Date': 'first',
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).reset_index(drop=True)
        
        return agg_df.dropna()
    
    def apply_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply SMC features"""
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df.columns:
                df[col] = 0.0
        
        # Momentum
        df['Returns'] = df['Close'].pct_change()
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        df['EMA_200'] = df['Close'].ewm(span=200).mean()
        
        # Volatility
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        df['ATR_Pct'] = df['ATR'] / df['Close']
        df['BB_Mid'] = df['Close'].rolling(20).mean()
        df['BB_Std'] = df['Close'].rolling(20).std()
        df['BB_H'] = df['BB_Mid'] + (2 * df['BB_Std'])
        df['BB_L'] = df['BB_Mid'] - (2 * df['BB_Std'])
        
        # Volume
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA'].fillna(1)
        
        # Time
        if 'Date' in df.columns:
            df['Hour'] = df['Date'].dt.hour
            df['Weekday'] = df['Date'].dt.weekday
            df['Date_ordinal'] = df['Date'].astype(np.int64) // 10**9
        
        # Structure
        df['Swing_High'] = df['High'].rolling(window=5, center=True).max()
        df['Swing_Low'] = df['Low'].rolling(window=5, center=True).min()
        df['Distance_to_High'] = df['Swing_High'] - df['Close']
        df['Distance_to_Low'] = df['Close'] - df['Swing_Low']
        
        return df.bfill().fillna(0)
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create TP-before-SL target"""
        df = df.copy()
        df['Target'] = 0
        
        close_vals = df['Close'].values
        atr_vals = df['ATR'].values
        high_vals = df['High'].values
        low_vals = df['Low'].values
        
        lookahead = 50
        
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
    
    def train_timeframe(self, tf: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Train model for single timeframe"""
        print(f"\n{'='*60}")
        print(f"EURUSD {tf} - XGBoost Trade Quality")
        print(f"{'='*60}")
        
        try:
            # Prepare
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Target' in numeric_cols:
                numeric_cols.remove('Target')
            
            X = df[numeric_cols].fillna(0)
            y = df['Target'].astype(int)
            
            print(f"Bars: {len(df):,}")
            print(f"Features: {len(numeric_cols)}")
            print(f"Target dist: {y.value_counts().to_dict()}")
            
            # Split
            train_size = int(len(df) * 0.70)
            val_size = int(len(df) * 0.15)
            
            X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
            X_val, y_val = X.iloc[train_size:train_size+val_size], y.iloc[train_size:train_size+val_size]
            X_test, y_test = X.iloc[train_size+val_size:], y.iloc[train_size+val_size:]
            
            print(f"Train/Val/Test: {len(X_train):,} / {len(X_val):,} / {len(X_test):,}")
            
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
                n_estimators=300,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=30
            )
            
            model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
            print("✓")
            
            # Evaluate
            y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Safe metric calculation
            try:
                auc = roc_auc_score(y_test, y_test_proba)
            except:
                auc = 0.5
            
            try:
                ll = log_loss(y_test, y_test_proba, labels=[0, 1])
            except:
                # Fallback if one label missing
                ll = -np.mean(y_test * np.log(y_test_proba + 1e-6) + (1-y_test) * np.log(1-y_test_proba + 1e-6))
            
            print(f"AUC: {auc:.4f} | LogLoss: {ll:.4f}")
            
            # Threshold optimization
            best_threshold = 0.50
            best_expectancy = -999
            
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
            
            # SHAP (top 5)
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_train_scaled[:min(1000, len(X_train_scaled))])
                importance = np.abs(shap_values).mean(axis=0)
                top_idx = np.argsort(importance)[-5:][::-1]
                print("Top features:", ", ".join([numeric_cols[i] for i in top_idx if i < len(numeric_cols)]))
            except:
                pass
            
            # Save
            symbol_dir = self.models_dir / "EURUSD"
            symbol_dir.mkdir(exist_ok=True)
            
            tf_map = {'5M': 'T_5M', '1H': 'T_1H', '4H': 'T_4H', '1D': 'T_1D', '1W': 'T_1W', '1M': 'T_1M'}
            tf_name = tf_map.get(tf, f'T_{tf}')
            
            joblib.dump(model, symbol_dir / f"{tf_name}.joblib")
            joblib.dump(scaler, symbol_dir / f"{tf_name}_scaler.joblib")
            joblib.dump(numeric_cols, symbol_dir / f"{tf_name}_features.joblib")
            
            # Metadata
            metadata = {
                f'{tf_name}': {
                    'auc_score': float(auc),
                    'log_loss': float(ll),
                    'optimal_threshold': float(best_threshold),
                    'expectancy': float(best_expectancy),
                    'n_features': len(numeric_cols),
                    'n_bars': len(df)
                }
            }
            
            return metadata[tf_name]
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_all_timeframes(self) -> Dict[str, Dict[str, Any]]:
        """Train models for all timeframes"""
        print("\n" + "="*80)
        print("EURUSD MULTI-TIMEFRAME XGBOOST TRAINING")
        print("Real data only - no simulation")
        print("="*80)
        
        df_raw = self.load_raw_data()
        if df_raw is None:
            return {}
        
        all_results = {}
        
        for tf in ['5M', '1H', '4H', '1D', '1W', '1M']:
            # Aggregate
            df = self.aggregate_timeframe(df_raw, tf)
            
            if len(df) < 200:
                print(f"\n⚠️  {tf}: Insufficient data ({len(df)} bars < 200 minimum)")
                continue
            
            # Features
            df = self.apply_features(df)
            
            # Target
            df = self.create_target(df)
            
            # Train
            result = self.train_timeframe(tf, df)
            if result:
                all_results[tf] = result
        
        # Summary
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        summary_df = pd.DataFrame(all_results).T
        if not summary_df.empty:
            print(summary_df[['auc_score', 'optimal_threshold', 'expectancy']].to_string())
        
        # Consolidate metadata
        symbol_dir = self.models_dir / "EURUSD"
        metadata_file = symbol_dir / "model_metadata.json"
        
        # Try to merge with existing metadata
        if metadata_file.exists():
            with open(metadata_file) as f:
                existing = json.load(f)
            existing.update(all_results)
            all_results = existing
        
        with open(metadata_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✓ Metadata saved to {metadata_file}")
        return all_results


if __name__ == "__main__":
    trainer = EURUSDMultiTimeframeTrainer()
    results = trainer.train_all_timeframes()
    print("\n✓ Training complete!")
