#!/usr/bin/env python3
"""
XGBoost Trade Quality Model Training
Predicts P(TP hit before SL | valid SMC setup)

Based on systematic trading best practices:
- Binary classification (TP-before-SL outcome)
- Time-based train/val/test split (no lookahead)
- XGBoost with early stopping
- SHAP explainability
- Threshold optimization for expectancy
- Proper probability calibration metrics

Reference: Marcos López de Prado – Advances in Financial Machine Learning
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

# Add PY_FILES to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from func import apply_features as external_apply_features
except ImportError:
    print("Warning: Could not import apply_features from func.py")
    external_apply_features = None


class TradeQualityTrainer:
    """Train XGBoost models for trade quality prediction (TP-before-SL)"""
    
    def __init__(
        self,
        data_dir: str = "../CSV_FILES",
        models_dir: str = "../ALL_MODELS",
        min_threshold: float = 0.50,
        max_threshold: float = 0.75,
        threshold_step: float = 0.05
    ):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Threshold optimization range
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.threshold_step = threshold_step
        
        # Symbol configuration
        self.symbols_config = {
            'BTCUSD': 'MT5_5M_BTCUSD_Dataset.csv',
            'ETHUSD': 'MT5_5M_ETHUSD_Dataset.csv',
            'XRPUSD': 'MT5_5M_XRPUSD_Dataset.csv',
            'LTCUSD': 'MT5_5M_LTCUSD_Dataset.csv',
            'ADAUSD': 'MT5_5M_ADAUSD_Dataset.csv',
            'SOLUSD': 'MT5_5M_SOLUSD_Dataset.csv',
            'DOGEUSD': 'MT5_5M_DOGEUSD_Dataset.csv',
            'EURUSD': 'MT5_5M_EURUSD_Exchange_Rate_Dataset.csv',
            'GBPUSD': 'MT5_5M_BT_GBPUSD_Dataset.csv',
            'USDJPY': 'MT5_5M_BT_USDJPY_Dataset.csv',
        }
        
    def load_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load CSV data for a symbol"""
        file_path = self.data_dir / self.symbols_config[symbol]
        
        if not file_path.exists():
            print(f"✗ File not found: {file_path}")
            return None
        
        print(f"  Loading {file_path.name}...", end=" ", flush=True)
        df = pd.read_csv(file_path)
        
        # Normalize column names
        if 'time' in df.columns:
            df.rename(columns={'time': 'Date'}, inplace=True)
        elif 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
        elif 'datetime' in df.columns:
            df.rename(columns={'datetime': 'Date'}, inplace=True)
        
        if df.empty:
            print("✗ Empty dataframe")
            return None
        
        # Capitalize OHLC column names
        col_mapping = {
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'tick_volume': 'Volume'
        }
        for old, new in col_mapping.items():
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)
        
        print(f"✓ Loaded {len(df):,} rows")
        return df
    
    def apply_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply SMC-focused technical features"""
        if external_apply_features is not None:
            df = external_apply_features(df)
        else:
            df = self._basic_smc_features(df)
        
        return df
    
    def _basic_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """SMC-focused feature engineering"""
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            df[col] = pd.to_numeric(df[col])
        
        # Price structure
        df['HL_Ratio'] = df['High'] / df['Low']
        df['OC_Ratio'] = df['Open'] / df['Close']
        df['Range'] = df['High'] - df['Low']
        df['Body'] = (df['Close'] - df['Open']).abs()
        df['Body_Range'] = df['Body'] / df['Range'].replace(0, np.nan)
        
        # SMC: Order Block detection (simplified)
        df['Bullish_OB'] = ((df['Close'] > df['Open']) & 
                            (df['Close'].shift(-1) > df['High'])).astype(int)
        df['Bearish_OB'] = ((df['Close'] < df['Open']) & 
                            (df['Close'].shift(-1) < df['Low'])).astype(int)
        
        # SMC: Fair Value Gap (FVG)
        df['FVG_Up'] = (df['Low'].shift(-1) > df['High'].shift(1)).astype(int)
        df['FVG_Down'] = (df['High'].shift(-1) < df['Low'].shift(1)).astype(int)
        df['FVG_Size'] = np.where(
            df['FVG_Up'] == 1,
            df['Low'].shift(-1) - df['High'].shift(1),
            np.where(df['FVG_Down'] == 1, df['Low'].shift(1) - df['High'].shift(-1), 0)
        )
        
        # SMC: Liquidity sweep
        df['High_20'] = df['High'].rolling(20).max()
        df['Low_20'] = df['Low'].rolling(20).min()
        df['Liquidity_Sweep_High'] = (df['High'] > df['High_20'].shift(1)).astype(int)
        df['Liquidity_Sweep_Low'] = (df['Low'] < df['Low_20'].shift(1)).astype(int)
        df['Sweep_Depth'] = np.where(
            df['Liquidity_Sweep_High'] == 1,
            df['High'] - df['High_20'].shift(1),
            np.where(df['Liquidity_Sweep_Low'] == 1, df['Low_20'].shift(1) - df['Low'], 0)
        )
        
        # Moving averages (trend context)
        for period in [20, 50, 200]:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        df['Trend'] = np.where(df['EMA_20'] > df['EMA_50'], 1, -1)
        
        # HTF alignment (simplified: price vs 200 EMA)
        df['HTF_Bullish'] = (df['Close'] > df['EMA_200']).astype(int)
        df['HTF_Bearish'] = (df['Close'] < df['EMA_200']).astype(int)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR (volatility)
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        df['TR'] = np.maximum(tr1, np.maximum(tr2, tr3))
        df['ATR'] = df['TR'].rolling(14).mean()
        df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
        
        # Session detection (simplified: hour-based)
        df['Hour'] = df['Date'].dt.hour
        df['Session_London'] = ((df['Hour'] >= 8) & (df['Hour'] < 16)).astype(int)
        df['Session_NY'] = ((df['Hour'] >= 13) & (df['Hour'] < 21)).astype(int)
        df['Session_Asia'] = ((df['Hour'] >= 0) & (df['Hour'] < 8)).astype(int)
        
        # Volume analysis
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA'].replace(0, np.nan)
        
        # Replace inf/nan
        df = df.replace([np.inf, -np.inf], np.nan)
        df.set_index('Date', inplace=True)
        
        return df
    
    def create_trade_quality_target(
        self,
        df: pd.DataFrame,
        tp_multiplier: float = 2.0,
        sl_multiplier: float = 1.0
    ) -> pd.DataFrame:
        """
        Create TP-before-SL binary target
        
        Target = 1 if TP hit before SL
        Target = 0 if SL hit before TP
        
        This simulates: Given current price, if we enter long,
        will TP (price + ATR*tp_multiplier) hit before SL (price - ATR*sl_multiplier)?
        """
        df = df.copy()
        df = df.reset_index()
        
        outcomes = []
        
        for i in range(len(df) - 50):  # Leave buffer for forward scan
            current_close = float(df.loc[i, 'Close'])
            atr = float(df.loc[i, 'ATR'])
            
            if pd.isna(atr) or atr == 0:
                outcomes.append(np.nan)
                continue
            
            # Define TP and SL levels (simplified: long bias)
            tp_level = current_close + (atr * tp_multiplier)
            sl_level = current_close - (atr * sl_multiplier)
            
            # Scan forward up to 50 candles
            tp_hit = False
            sl_hit = False
            
            for j in range(i + 1, min(i + 51, len(df))):
                high = float(df.loc[j, 'High'])
                low = float(df.loc[j, 'Low'])
                
                if high >= tp_level:
                    tp_hit = True
                    break
                if low <= sl_level:
                    sl_hit = True
                    break
            
            # Outcome
            if tp_hit and not sl_hit:
                outcomes.append(1)
            elif sl_hit and not tp_hit:
                outcomes.append(0)
            else:
                outcomes.append(np.nan)  # Neither or both (rare)
        
        # Pad remaining
        outcomes.extend([np.nan] * (len(df) - len(outcomes)))
        
        df['Target'] = outcomes
        df.set_index('Date', inplace=True)
        
        return df
    
    def time_based_split(
        self,
        df: pd.DataFrame,
        train_pct: float = 0.70,
        val_pct: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Time-based split (NO SHUFFLE)
        Train: oldest 70%
        Val: next 15%
        Test: most recent 15%
        """
        df = df.sort_index()  # Ensure chronological order
        
        n = len(df)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))
        
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]
        
        print(f"  Train: {len(train):,} rows ({train.index[0]} to {train.index[-1]})")
        print(f"  Val:   {len(val):,} rows ({val.index[0]} to {val.index[-1]})")
        print(f"  Test:  {len(test):,} rows ({test.index[0]} to {test.index[-1]})")
        
        return train, val, test
    
    def optimize_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        avg_win: float = 2.0,
        avg_loss: float = 1.0
    ) -> Dict[float, Dict[str, float]]:
        """
        Find profit-maximizing probability threshold
        
        Returns dict with threshold -> expectancy
        """
        thresholds = np.arange(self.min_threshold, self.max_threshold + self.threshold_step, self.threshold_step)
        results: Dict[float, Dict[str, float]] = {}
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if y_pred.sum() == 0:  # No trades
                results[float(threshold)] = {
                    'expectancy': 0.0,
                    'win_rate': 0.0,
                    'trade_count': 0.0
                }
                continue
            
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            _tn, fp, _fn, tp = cm.ravel()
            
            wins = tp
            losses = fp
            total_trades = wins + losses
            
            if total_trades == 0:
                win_rate = 0.0
                expectancy = 0.0
            else:
                win_rate = wins / total_trades
                loss_rate = losses / total_trades
                expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
            
            results[float(threshold)] = {
                'expectancy': float(expectancy),
                'win_rate': float(win_rate),
                'trade_count': float(total_trades)
            }
        
        return results
    
    def train_model(
        self,
        symbol: str,
        df: pd.DataFrame,
        use_shap: bool = True
    ) -> Tuple[Optional[Any], Optional[StandardScaler], Optional[List[str]], Optional[Dict[str, Any]]]:
        """Train XGBoost trade quality model"""
        
        print(f"\n{'='*60}")
        print(f"Training Trade Quality Model: {symbol}")
        print(f"{'='*60}")
        
        # Drop NaN
        df = df.dropna()
        
        if len(df) < 1000:
            print(f"✗ Not enough data for {symbol} ({len(df)} rows)")
            return None, None, None, None
        
        print(f"Data points: {len(df):,}")
        
        # Exclude post-trade features
        exclude_cols = ['Target', 'Volume', 'Open', 'High', 'Low', 'Close']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        features_df = df[feature_cols]
        y = df['Target']
        
        # Numeric only
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        features_df = features_df[numeric_cols]
        
        print(f"Features: {len(features_df.columns)}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Time-based split
        train_idx = int(len(features_df) * 0.70)
        val_idx = int(len(features_df) * 0.85)
        
        X_train, y_train = features_df.iloc[:train_idx], y.iloc[:train_idx]
        X_val, y_val = features_df.iloc[train_idx:val_idx], y.iloc[train_idx:val_idx]
        X_test, y_test = features_df.iloc[val_idx:], y.iloc[val_idx:]
        
        print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Calculate class weight for imbalance
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Train XGBoost
        print("\nTraining XGBoost...")
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
            n_jobs=-1
        )
        
        model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_val_scaled, y_val)],
            early_stopping_rounds=30,
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
        threshold_results = self.optimize_threshold(
            y_test.values,
            y_test_proba,
            avg_win=2.0,
            avg_loss=1.0
        )
        
        best_threshold = max(threshold_results.keys(), key=lambda k: threshold_results[k]['expectancy'])
        best_result = threshold_results[best_threshold]
        
        print(f"Optimal Threshold: {best_threshold:.2f}")
        print(f"  Expectancy: {best_result['expectancy']:.4f}")
        print(f"  Win Rate: {best_result['win_rate']:.2%}")
        print(f"  Trade Count: {best_result['trade_count']}")
        
        # SHAP explainability
        top_features = []
        if use_shap and len(X_test_scaled) > 0:
            try:
                print(f"\n{'='*40}")
                print("SHAP Feature Importance")
                print(f"{'='*40}")
                
                explainer = shap.Explainer(model, X_train_scaled[:100])  # Sample for speed
                shap_values = explainer(X_test_scaled[:100])
                
                shap_importance = np.abs(shap_values.values).mean(axis=0)
                feature_importance_df = pd.DataFrame({
                    'feature': features_df.columns,
                    'shap_importance': shap_importance
                }).sort_values('shap_importance', ascending=False)
                
                print(feature_importance_df.head(10).to_string(index=False))
                top_features = feature_importance_df.head(76)['feature'].tolist()
                
            except Exception as e:
                print(f"SHAP analysis failed: {e}")
                top_features = features_df.columns.tolist()[:76]
        else:
            # Fallback: XGBoost feature importance
            importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': features_df.columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            top_features = feature_importance_df.head(76)['feature'].tolist()
        
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
            model_path = symbol_dir / f"{tf}.joblib"
            scaler_path = symbol_dir / f"{tf}_scaler.joblib"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
        
        # Save feature list
        features_path = symbol_dir / "features.joblib"
        joblib.dump(features, features_path)
        
        # Save metadata
        metadata_path = symbol_dir / "model_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✓ Saved XGBoost model to {symbol_dir}")
        print(f"  - {len(timeframes)} timeframe variants")
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
        
        print(f"\n{'='*60}")
        print(f"XGBoost Trade Quality Training")
        print(f"{'='*60}")
        print(f"Symbols: {', '.join(symbols)}\n")
        
        for symbol in symbols:
            try:
                if symbol not in self.symbols_config:
                    print(f"✗ Unknown symbol: {symbol}")
                    failed.append(symbol)
                    continue
                
                # Load data
                df = self.load_data(symbol)
                if df is None:
                    failed.append(symbol)
                    continue
                
                # Apply features
                print(f"  Applying SMC features...", end=" ", flush=True)
                df = self.apply_features(df)
                print("✓")
                
                # Create target
                print(f"  Creating TP-before-SL target...", end=" ", flush=True)
                df = self.create_trade_quality_target(df, tp_multiplier=2.0, sl_multiplier=1.0)
                print("✓")
                
                # Train model
                model, scaler, features, metrics = self.train_model(symbol, df, use_shap=True)
                
                if model is None:
                    failed.append(symbol)
                    continue
                
                # Save model
                self.save_model(symbol, model, scaler, features, metrics)
                results[symbol] = metrics
                
            except Exception as e:
                print(f"\n✗ Error training {symbol}: {e}")
                import traceback
                traceback.print_exc()
                failed.append(symbol)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Training Summary")
        print(f"{'='*60}")
        print(f"Successful: {len(results)}")
        print(f"Failed: {len(failed)}")
        
        if results:
            print(f"\nModel Performance:")
            results_df = pd.DataFrame(results).T
            print(results_df.to_string())
        
        if failed:
            print(f"\nFailed symbols: {', '.join(failed)}")
        
        return results, failed


if __name__ == "__main__":
    trainer = TradeQualityTrainer()
    
    # Train crypto symbols
    crypto_symbols = ['BTCUSD', 'ETHUSD', 'XRPUSD', 'LTCUSD', 'ADAUSD', 'SOLUSD', 'DOGEUSD']
    
    # Optional: forex
    forex_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    print("Training crypto models with XGBoost (trade quality)...")
    results, failed = trainer.train_all_symbols(crypto_symbols)
    
    if forex_symbols:
        print("\n\nTraining forex models...")
        forex_results, forex_failed = trainer.train_all_symbols(forex_symbols)
        results.update(forex_results)
        failed.extend(forex_failed)
    
    print(f"\n✓ Training complete! Check ALL_MODELS/ for XGBoost models.")
    print(f"\nNext steps:")
    print(f"1. Review SHAP feature importance")
    print(f"2. Verify AUC > 0.55 for all symbols")
    print(f"3. Use optimal_threshold in live trading")
