#!/usr/bin/env python3
"""
Multi-Symbol ML Model Training
Trains LightGBM models on 2-year historical crypto and forex data
Creates models for each symbol with feature selection and cross-validation
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Add PY_FILES to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from func import apply_features
except ImportError:
    print("Warning: Could not import apply_features from func.py, using basic features")
    apply_features = None

warnings.filterwarnings('ignore')

class CryptoModelTrainer:
    """Train ML models for multiple symbols"""
    
    def __init__(self, data_dir: str = "../CSV_FILES", models_dir: str = "../ALL_MODELS"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Define available symbols and their data files
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
        
    def load_data(self, symbol: str) -> pd.DataFrame:
        """Load CSV data for a symbol"""
        file_path = self.data_dir / self.symbols_config[symbol]
        
        if not file_path.exists():
            print(f"✗ File not found: {file_path}")
            return None
        
        print(f"  Loading {file_path.name}...", end=" ", flush=True)
        df = pd.read_csv(file_path)
        
        # Rename columns to match expected format
        if 'time' in df.columns:
            df.rename(columns={'time': 'Date'}, inplace=True)
        
        if df.empty:
            print("✗ Empty dataframe")
            return None
        
        # Capitalize OHLC column names for feature engineering
        col_mapping = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}
        for old, new in col_mapping.items():
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)
        
        print(f"✓ Loaded {len(df):,} rows")
        return df
    
    def apply_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply technical features to dataframe"""
        if apply_features is not None:
            return apply_features(df)
        else:
            # Fallback: basic feature engineering
            return self._basic_features(df)
    
    def _basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic feature engineering fallback"""
        df = df.copy()
        
        # Ensure proper column names and types
        df['Date'] = pd.to_datetime(df['Date'])
        
        df['Close'] = pd.to_numeric(df['Close'])
        df['High'] = pd.to_numeric(df['High'])
        df['Low'] = pd.to_numeric(df['Low'])
        df['Open'] = pd.to_numeric(df['Open'])
        df['Volume'] = pd.to_numeric(df['Volume'])
        
        # Basic indicators
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['Histogram'] = df['MACD'] - df['Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        df['BB_Std'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        
        # ATR
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['ATR'] = df['TR'].rolling(14).mean()
        
        # Price patterns
        df['HL_Ratio'] = df['High'] / df['Low']
        df['OC_Ratio'] = df['Open'] / df['Close']
        df['Close_SMA5'] = df['Close'] / df['SMA_5']
        df['Close_SMA20'] = df['Close'] / df['SMA_20']
        df['Range'] = df['High'] - df['Low']
        df['Body'] = abs(df['Close'] - df['Open'])
        df['Body_Range'] = df['Body'] / df['Range'].replace(0, 1)
        
        # Volume
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA'].replace(0, 1)
        
        # Trend
        df['Trend'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
        df['Price_Above_SMA'] = (df['Close'] > df['SMA_50']).astype(int)
        
        # Returns
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Return_MA'] = df['Log_Return'].rolling(5).mean()
        df['Return_Std'] = df['Log_Return'].rolling(5).std()
        
        df.set_index('Date', inplace=True)
        return df
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable (1 = price goes up, 0 = price goes down)"""
        df = df.copy()
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        return df
    
    def train_model(self, symbol: str, df: pd.DataFrame) -> Tuple[LGBMClassifier, StandardScaler, List[str], Dict]:
        """Train LightGBM model for a symbol"""
        
        print(f"\n{'='*60}")
        print(f"Training {symbol} Model")
        print(f"{'='*60}")
        
        # Drop NaN rows
        df = df.dropna()
        
        if len(df) < 1000:
            print(f"✗ Not enough data for {symbol} ({len(df)} rows)")
            return None, None, None, None
        
        print(f"Data points: {len(df):,}")
        
        # Prepare features and target
        X = df.drop(columns=['Target', 'Volume'])
        y = df['Target']
        
        # Select numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols]
        
        print(f"Features: {len(X.columns)}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Split data: 70% train, 30% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Train set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        print("\nTraining LightGBM classifier...")
        model = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=7,
            num_leaves=31,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top 76 features (like original preprocessing)
        top_features = feature_importance.head(76)['feature'].tolist()
        
        print(f"\nTop 10 features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Retrain with top features
        print(f"\nRetraining with top 76 features...")
        X_train_top = X_train_scaled[:, [list(X.columns).index(f) for f in top_features]]
        X_test_top = X_test_scaled[:, [list(X.columns).index(f) for f in top_features]]
        
        model_top = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=7,
            num_leaves=31,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        model_top.fit(X_train_top, y_train)
        
        # Evaluate
        y_pred = model_top.predict(X_test_top)
        y_pred_proba = model_top.predict_proba(X_test_top)[:, 1]
        
        accuracy = (y_pred == y_test).mean()
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\nModel Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC Score: {auc_score:.4f}")
        
        report = classification_report(y_test, y_pred, output_dict=True)
        print(f"\nClassification Report:")
        print(f"  Precision (Class 1): {report['1']['precision']:.4f}")
        print(f"  Recall (Class 1): {report['1']['recall']:.4f}")
        print(f"  F1-Score (Class 1): {report['1']['f1-score']:.4f}")
        
        metrics = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'n_features': len(top_features),
            'n_samples': len(X_train)
        }
        
        return model_top, scaler, top_features, metrics
    
    def save_model(self, symbol: str, model: LGBMClassifier, scaler: StandardScaler, features: List[str]):
        """Save model and scaler to disk"""
        symbol_dir = self.models_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model for different timeframes
        for tf in ['T_5M', 'T_10M', 'T_15M', 'T_20M', 'T_30M']:
            model_path = symbol_dir / f"{tf}.joblib"
            scaler_path = symbol_dir / f"{tf}_scaler.joblib"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
        
        # Save feature list
        features_path = symbol_dir / "features.joblib"
        joblib.dump(features, features_path)
        
        print(f"  ✓ Saved to {symbol_dir}")
    
    def train_all_symbols(self, symbols: List[str] = None):
        """Train models for multiple symbols"""
        if symbols is None:
            symbols = list(self.symbols_config.keys())
        
        results = {}
        failed = []
        
        print(f"\n{'='*60}")
        print(f"Multi-Symbol ML Model Training")
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
                print(f"  Applying features...", end=" ", flush=True)
                df = self.apply_features(df)
                print("✓")
                
                # Create target
                df = self.create_target(df)
                
                # Train model
                model, scaler, features, metrics = self.train_model(symbol, df)
                
                if model is None:
                    failed.append(symbol)
                    continue
                
                # Save model
                self.save_model(symbol, model, scaler, features)
                results[symbol] = metrics
                
            except Exception as e:
                print(f"\n✗ Error training {symbol}: {e}")
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
    # Train on all available symbols
    trainer = CryptoModelTrainer()
    
    # Crypto pairs (main focus)
    crypto_symbols = ['BTCUSD', 'ETHUSD', 'XRPUSD', 'LTCUSD', 'ADAUSD', 'SOLUSD', 'DOGEUSD']
    
    # Optional: also train forex pairs if available
    forex_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    print("Training crypto models...")
    results, failed = trainer.train_all_symbols(crypto_symbols)
    
    if forex_symbols:
        print("\n\nTraining forex models...")
        forex_results, forex_failed = trainer.train_all_symbols(forex_symbols)
        results.update(forex_results)
        failed.extend(forex_failed)
    
    print(f"\n✓ All done! Check ALL_MODELS/ directory for trained models.")
