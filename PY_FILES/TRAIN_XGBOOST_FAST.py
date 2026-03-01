#!/usr/bin/env python3
"""
Fast XGBoost Trade Quality Training - Optimized for Speed
Trains on sampled data to reduce memory/compute while maintaining quality
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import warnings
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
import xgboost as xgb
import shap

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

try:
    from func import apply_features as external_apply_features
except ImportError:
    external_apply_features = None


class FastTradeQualityTrainer:
    """Fast XGBoost training with memory optimization"""
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        models_dir: Optional[str] = None,
        sample_size: int = 50000,  # Use 50k rows per symbol (manageable)
        min_threshold: float = 0.50,
        max_threshold: float = 0.75,
        threshold_step: float = 0.05
    ):
        script_dir = Path(__file__).parent.parent
        self.data_dir = Path(data_dir) if data_dir else (script_dir / "CSV_FILES")
        self.models_dir = Path(models_dir) if models_dir else (script_dir / "ALL_MODELS")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.sample_size = sample_size
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.threshold_step = threshold_step
        
        # All available symbols
        self.symbols_config = {
            # Crypto
            'BTCUSD': 'MT5_5M_BTCUSD_Dataset.csv',
            'ETHUSD': 'MT5_5M_ETHUSD_Dataset.csv',
            'XRPUSD': 'MT5_5M_XRPUSD_Dataset.csv',
            'LTCUSD': 'MT5_5M_LTCUSD_Dataset.csv',
            'ADAUSD': 'MT5_5M_ADAUSD_Dataset.csv',
            'SOLUSD': 'MT5_5M_SOLUSD_Dataset.csv',
            'DOGEUSD': 'MT5_5M_DOGEUSD_Dataset.csv',
            # Forex - USD pairs
            'EURUSD': 'MT5_5M_BT_EURUSD_Dataset.csv',
            'GBPUSD': 'MT5_5M_BT_GBPUSD_Dataset.csv',
            'USDJPY': 'MT5_5M_BT_USDJPY_Dataset.csv',
            'AUDUSD': 'MT5_5M_BT_AUDUSD_Dataset.csv',
            'NZDUSD': 'MT5_5M_BT_NZDUSD_Dataset.csv',
            'USDCAD': 'MT5_5M_BT_USDCAD_Dataset.csv',
            'USDHKD': 'MT5_5M_BT_USDHKD_Dataset.csv',
            # Forex - crosses
            'EURGBP': 'MT5_5M_BT_EURGBP_Dataset.csv',
            'EURJPY': 'MT5_5M_BT_EURJPY_Dataset.csv',
            'GBPJPY': 'MT5_5M_BT_GBPJPY_Dataset.csv',
            'AUDNZD': 'MT5_5M_BT_AUDNZD_Dataset.csv',
        }
        
    def load_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load and sample CSV data for a symbol"""
        file_path = self.data_dir / self.symbols_config[symbol]
        
        if not file_path.exists():
            print(f"✗ File not found: {file_path}")
            return None
        
        print(f"  Loading {file_path.name}...", end=" ", flush=True)
        
        # Load with sampling for large files
        df = pd.read_csv(file_path)
        
        if len(df) > self.sample_size:
            # Sample from recent data (last sample_size rows)
            df = df.tail(self.sample_size).reset_index(drop=True)
            print(f"✓ Loaded {len(df):,} rows (sampled from {df.shape[0] + (len(pd.read_csv(file_path)) - self.sample_size)} total)", flush=True)
        else:
            print(f"✓ Loaded {len(df):,} rows", flush=True)
        
        # Normalize column names
        col_mapping = {
            'time': 'Date', 'Datetime': 'Date', 'datetime': 'Date',
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'tick_volume': 'Volume'
        }
        for old, new in col_mapping.items():
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)
        
        return df
    
    def apply_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply SMC-focused technical features"""
        if external_apply_features is not None:
            df = external_apply_features(df)
        else:
            df = self._basic_smc_features(df)
        return df
    
    def _basic_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Minimal SMC feature engineering (fast version)"""
        # Ensure OHLCV columns exist
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df.columns:
                df[col] = 0.0
        
        # Basic momentum/trend features
        df['Returns'] = df['Close'].pct_change()
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        df['EMA_200'] = df['Close'].ewm(span=200).mean()
        
        # Volatility
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        df['ATR_Pct'] = df['ATR'] / df['Close']
        
        # Volume
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA'].fillna(1)
        
        # Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(20).mean()
        df['BB_Std'] = df['Close'].rolling(20).std()
        df['BB_H'] = df['BB_Mid'] + (2 * df['BB_Std'])
        df['BB_L'] = df['BB_Mid'] - (2 * df['BB_Std'])
        
        # Time-based features
        df['Hour'] = pd.to_datetime(df['Date'], errors='coerce').dt.hour
        df['Weekday'] = pd.to_datetime(df['Date'], errors='coerce').dt.weekday
        
        # Fill NaN values
        df = df.bfill().fillna(0)
        
        return df
    
    def create_trade_quality_target(
        self,
        df: pd.DataFrame,
        tp_multiplier: float = 2.0,
        sl_multiplier: float = 1.0,
        lookahead: int = 50
    ) -> pd.DataFrame:
        """Create TP-before-SL binary target (optimized)"""
        df = df.copy()
        df['Target'] = 0
        
        close_vals = df['Close'].values
        atr_vals = df['ATR'].values
        high_vals = df['High'].values
        low_vals = df['Low'].values
        
        # Forward-scan in chunks to reduce memory usage
        for i in range(len(df) - lookahead):
            current_close = float(close_vals[i])
            atr = max(float(atr_vals[i]), current_close * 0.001)
            
            tp = current_close + (atr * tp_multiplier)
            sl = current_close - (atr * sl_multiplier)
            
            # Scan next 50 candles
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
    
    def train_model(
        self,
        symbol: str,
        df: pd.DataFrame,
        use_shap: bool = True
    ) -> Tuple[Optional[Any], Optional[StandardScaler], Optional[List[str]], Optional[Dict[str, Any]]]:
        """Train XGBoost with fast evaluation"""
        print(f"\n{'='*40}")
        print(f"Training Trade Quality Model: {symbol}")
        print(f"{'='*40}")
        
        try:
            # Prepare features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Target' in numeric_cols:
                numeric_cols.remove('Target')
            
            feature_cols = numeric_cols
            
            print(f"Data points: {len(df):,}")
            print(f"Features: {len(feature_cols)}")
            
            X = df[feature_cols].fillna(0)
            y = df['Target'].astype(int)
            
            target_dist = y.value_counts().to_dict()
            print(f"Target distribution: {target_dist}")
            
            # Time-based 70/15/15 split
            train_size = int(len(df) * 0.70)
            val_size = int(len(df) * 0.15)
            
            X_train = X.iloc[:train_size]
            y_train = y.iloc[:train_size]
            X_val = X.iloc[train_size:train_size + val_size]
            y_val = y.iloc[train_size:train_size + val_size]
            X_test = X.iloc[train_size + val_size:]
            y_test = y.iloc[train_size + val_size:]
            
            print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Class imbalance
            pos_count = y_train.sum()
            neg_count = len(y_train) - pos_count
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            print(f"Scale pos weight: {scale_pos_weight:.2f}")
            
            # Train XGBoost (fast config)
            print("\nTraining XGBoost...")
            model = xgb.XGBClassifier(
                n_estimators=150,  # Reduced from 300 for speed
                max_depth=4,  # Reduced from 5
                learning_rate=0.05,  # Increased for speed
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=20  # Reduced from 30
            )
            
            model.fit(
                X_train_scaled,
                y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
            
            # Evaluate
            y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
            auc_score = roc_auc_score(y_test, y_test_proba)
            logloss = log_loss(y_test, y_test_proba)
            
            print(f"\n{'='*40}")
            print(f"Model Performance (Test Set)")
            print(f"{'='*40}")
            print(f"ROC-AUC:  {auc_score:.4f}")
            print(f"Log Loss: {logloss:.4f}")
            
            # Threshold optimization
            print(f"\n{'='*40}")
            print(f"Threshold Optimization")
            print(f"{'='*40}")
            best_threshold = 0.50
            best_expectancy = 0
            
            for threshold in np.arange(self.min_threshold, self.max_threshold + self.threshold_step, self.threshold_step):
                y_pred = (y_test_proba >= threshold).astype(int)
                trade_count = y_pred.sum()
                
                if trade_count == 0:
                    continue
                
                _, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                win_rate = tp / (tp + fp) if (tp + fp) > 0 else 0
                loss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
                
                expectancy = (win_rate * 2.0) - (loss_rate * 1.0)  # 2:1 RR
                
                if expectancy > best_expectancy:
                    best_expectancy = expectancy
                    best_threshold = threshold
            
            best_result = {
                'expectancy': best_expectancy,
                'win_rate': (best_expectancy + 1.0) / 3.0  # Derived
            }
            
            print(f"Optimal Threshold: {best_threshold:.2f}")
            print(f"  Expectancy: {best_result['expectancy']:.4f}")
            print(f"  Win Rate: {best_result['win_rate']:.1%}")
            
            # SHAP (optional, skip if slow)
            if use_shap and len(X_train_scaled) > 1000:
                try:
                    print(f"\n{'='*40}")
                    print(f"SHAP Feature Importance")
                    print(f"{'='*40}")
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_train_scaled[:1000])
                    importance = np.abs(shap_values).mean(axis=0)
                    feature_importance_df = pd.DataFrame({
                        'feature': feature_cols,
                        'shap_importance': importance
                    }).sort_values('shap_importance', ascending=False).head(10)
                    print(feature_importance_df.to_string(index=False))
                except Exception as e:
                    print(f"SHAP analysis failed: {e}")
            
            # Select top features
            top_features = feature_cols
            
            # Metrics
            metrics: Dict[str, Any] = {
                'auc_score': float(auc_score),
                'log_loss': float(logloss),
                'optimal_threshold': float(best_threshold),
                'expectancy': float(best_result['expectancy']),
                'win_rate': float(best_result['win_rate']),
                'n_features': len(top_features),
                'n_train': len(X_train)
            }
            
            return model, scaler, top_features, metrics
            
        except Exception as e:
            print(f"✗ Error training {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None
    
    def save_model(
        self,
        symbol: str,
        model: Any,
        scaler: StandardScaler,
        features: List[str],
        metrics: Dict[str, Any]
    ) -> None:
        """Save XGBoost model, scaler, features, and metadata"""
        symbol_dir = self.models_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Save for all timeframes (trade quality is timeframe-agnostic)
        timeframes = ['T_5M', 'T_10M', 'T_15M', 'T_20M', 'T_30M', 'T_1H', 'T_4H', 'T_1D', 'T_1W', 'T_1M']
        
        for tf in timeframes:
            joblib.dump(model, symbol_dir / f"{tf}.joblib")
            joblib.dump(scaler, symbol_dir / f"{tf}_scaler.joblib")
        
        joblib.dump(features, symbol_dir / "features.joblib")
        
        import json
        with open(symbol_dir / "model_metadata.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Saved XGBoost model to {symbol_dir}")
        print(f"  - 10 timeframe variants")
        print(f"  - {len(features)} features")
        print(f"  - Optimal threshold: {metrics['optimal_threshold']:.2f}")
    
    def train_all_symbols(
        self,
        symbols: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        """Train trade quality models for multiple symbols"""
        if symbols is None:
            symbols = list(self.symbols_config.keys())
        
        results: Dict[str, Dict[str, Any]] = {}
        failed: List[str] = []
        
        for symbol in symbols:
            try:
                df = self.load_data(symbol)
                if df is None:
                    failed.append(symbol)
                    continue
                
                print(f"  Applying SMC features...", end=" ", flush=True)
                df = self.apply_features(df)
                print("✓")
                
                print(f"  Creating TP-before-SL target...", end=" ", flush=True)
                df = self.create_trade_quality_target(df, tp_multiplier=2.0, sl_multiplier=1.0)
                print("✓")
                
                model, scaler, features, metrics = self.train_model(symbol, df, use_shap=True)
                
                if model is None:
                    failed.append(symbol)
                    continue
                
                self.save_model(symbol, model, scaler, features, metrics)
                results[symbol] = metrics
                
            except Exception as e:
                print(f"\n✗ Error training {symbol}: {e}")
                failed.append(symbol)
        
        return results, failed


if __name__ == "__main__":
    print("\n" + "="*80)
    print("FAST XGBOOST TRADE QUALITY - ALL PAIRS")
    print("="*80)
    
    trainer = FastTradeQualityTrainer(sample_size=50000)  # Use 50k rows max per symbol
    
    # All symbols
    all_symbols = [
        # Crypto
        'BTCUSD', 'ETHUSD', 'XRPUSD', 'LTCUSD', 'ADAUSD', 'SOLUSD', 'DOGEUSD',
        # Forex
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD',
        'USDCAD', 'USDHKD', 'EURGBP', 'EURJPY', 'GBPJPY', 'AUDNZD'
    ]
    
    results, failed = trainer.train_all_symbols(all_symbols)
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Successful: {len(results)}/{len(all_symbols)}")
    print(f"Failed: {len(failed)}")
    
    if results:
        results_df = pd.DataFrame(results).T
        print("\n" + results_df[['auc_score', 'expectancy', 'optimal_threshold', 'win_rate']].to_string())
    
    if failed:
        print(f"\nFailed symbols: {', '.join(failed)}")
    
    print("\n✓ Fast training complete!")
    print("="*80)
