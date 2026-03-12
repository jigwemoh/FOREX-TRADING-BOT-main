#!/usr/bin/env python3
"""
Comprehensive Diagnostic Backtest
Analyzes current ML + Scalping system performance
Outputs: Win rate, Sharpe ratio, drawdown, avg win/loss, setup effectiveness
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiagnosticBacktester:
    """Comprehensive backtest analyzer for current system"""
    
    def __init__(self, symbols: List[str] = None, config_path: str = "../config.json"):
        self.symbols = symbols or ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results: Dict[str, Any] = {}
        self.trades: List[Dict[str, Any]] = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load trading config"""
        try:
            with open(self.config_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def load_backtest_data(self, symbol: str) -> pd.DataFrame:
        """Load 5M backtest data for symbol"""
        data_paths = [
            Path(f"../CSV_FILES/MT5_5M_BT_{symbol}_Dataset.csv"),
            Path(f"../CSV_FILES/{symbol}_Exchange_Rate_Dataset.csv"),
        ]
        
        for path in data_paths:
            if path.exists():
                try:
                    logger.info(f"[{symbol}] Loading {path}")
                    df = pd.read_csv(path)
                    # Prepare data for func.apply_features
                    # It needs either DatetimeIndex or Date column
                    if 'Datetime' in df.columns:
                        df['Date'] = pd.to_datetime(df['Datetime'])
                        df.set_index('Date', inplace=True)
                    elif 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                    return df
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")
        
        logger.warning(f"[{symbol}] No backtest data found")
        return pd.DataFrame()
    
    def load_ml_models(self, symbol: str) -> Dict[str, Any]:
        """Load all ML models for a symbol"""
        models_dict = {}
        feature_dict = {}
        timeframes = ["T_5M", "T_10M", "T_15M", "T_20M", "T_30M", "T_1H"]
        
        for tf in timeframes:
            # Try flat file first: EURUSD_lgbm_T_5M.pkl
            model_path = Path(f"../ALL_MODELS/{symbol}_lgbm_{tf}.pkl")
            if model_path.exists():
                try:
                    bundle = joblib.load(model_path)
                    models_dict[tf] = bundle.get("model")
                    feature_dict[tf] = bundle.get("features", [])
                    logger.info(f"[{symbol}] Loaded {tf}: {len(feature_dict[tf])} features")
                except Exception as e:
                    logger.warning(f"[{symbol}] Failed to load {tf}: {e}")
            else:
                # Try directory: ALL_MODELS/EURUSD/T_5M
                model_path = Path(f"../ALL_MODELS/{symbol}/{tf}")
                if model_path.exists():
                    try:
                        bundle = joblib.load(model_path)
                        models_dict[tf] = bundle.get("model")
                        feature_dict[tf] = bundle.get("features", [])
                        logger.info(f"[{symbol}] Loaded {tf}: {len(feature_dict[tf])} features")
                    except Exception as e:
                        logger.warning(f"[{symbol}] Failed to load {tf}: {e}")
        
        return {"models": models_dict, "features": feature_dict}
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using your feature pipeline"""
        if df.empty:
            return df
        
        try:
            # Use your trained feature engineering pipeline
            from func import apply_features as func_apply_features
            df_copy = df.copy()
            logger.debug(f"calculate_features input: index_type={type(df_copy.index).__name__}, "
                        f"is_datetimeindex={isinstance(df_copy.index, pd.DatetimeIndex)}, "
                        f"columns={df_copy.columns.tolist()[:5]}")
            result = func_apply_features(df_copy)
            logger.debug(f"apply_features successful: output shape={result.shape}")
            return result
        except Exception as e:
            logger.warning(f"Failed to use func.apply_features: {e}, using fallback")
            # Fallback basic features
            if df.empty:
                return df
            
            df_copy = df.copy()
            if isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy = df_copy.reset_index()
            
            # Basic features
            close_col = df_copy.get('Close', df_copy.get('close'))
            if close_col is None:
                return df_copy.fillna(0)
            
            df_copy['Returns'] = close_col.pct_change()
            df_copy['EMA_20'] = close_col.ewm(span=20).mean()
            df_copy['EMA_50'] = close_col.ewm(span=50).mean()
            df_copy['TREND'] = np.where(df_copy['EMA_20'] > df_copy['EMA_50'], 1, -1)
            
            return df_copy.fillna(0)
    
    def get_ml_signal(self, df: pd.DataFrame, model: Any, features: List[str], threshold: float = 0.52) -> np.ndarray:
        """Get ML signal (1=BUY, -1=SELL, 0=HOLD)"""
        try:
            # Only use features that exist in dataframe
            available_features = [f for f in features if f in df.columns]
            
            if not available_features:
                logger.warning(f"No features available. Required: {features[:5]}... Found: {list(df.columns[:5])}")
                return np.zeros(len(df))
            
            # Prepare features
            X = df[available_features].fillna(0).values
            
            # Get probabilities
            proba = model.predict_proba(X)
            prob_up = proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(X))
            
            # Generate signal
            signals = np.zeros(len(X))
            signals[prob_up >= threshold] = 1
            signals[prob_up <= (1 - threshold)] = -1
            
            logger.info(f"Generated {(signals != 0).sum()} signals out of {len(df)} bars")
            return signals
        except Exception as e:
            logger.warning(f"ML signal error: {e}")
            return np.zeros(len(df))
    
    def simulate_trades(self, symbol: str, df: pd.DataFrame, models_dict: Dict[str, Any],
                       features_dict: Dict[str, Any], risk_pct: float = 0.01) -> List[Dict[str, Any]]:
        """Simulate trades based on ML signals"""
        trades = []
        
        if df.empty or not models_dict:
            return trades
        
        # Use T_5M model for signals (highest frequency available)
        tf_signal = "T_5M"
        if tf_signal not in models_dict or not models_dict[tf_signal]:
            logger.warning(f"[{symbol}] No {tf_signal} model available, trying T_10M")
            tf_signal = "T_10M"
            if tf_signal not in models_dict or not models_dict[tf_signal]:
                logger.warning(f"[{symbol}] No models available")
                return trades
        
        model = models_dict[tf_signal]
        features = features_dict.get(tf_signal, [])
        threshold = self.config.get("trading", {}).get("ml_threshold", 0.52)
        
        logger.info(f"[{symbol}] Using {tf_signal} model with threshold {threshold}")
        
        # Get signals
        signals = self.get_ml_signal(df, model, features, threshold)
        
        # Simulate entries/exits
        position = None
        entry_price = 0
        entry_idx = 0
        entry_signal = 0
        
        for i in range(1, len(df)):
            if signals[i] == 0:
                continue
            
            # Exit if signal flips
            if position and signals[i] == -entry_signal:
                exit_price = df.iloc[i]['close']
                pnl = (exit_price - entry_price) * entry_signal
                pnl_pct = (pnl / entry_price) * 100
                
                trades.append({
                    'symbol': symbol,
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'direction': entry_signal,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'bars_held': i - entry_idx,
                    'win': 1 if pnl > 0 else 0
                })
                position = None
            
            # New entry if no position
            elif not position:
                entry_price = df.iloc[i]['close']
                entry_idx = i
                entry_signal = signals[i]
                position = True
        
        return trades
    
    def calculate_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics from trades"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'consecutive_losses': 0
            }
        
        df_trades = pd.DataFrame(trades)
        wins = df_trades[df_trades['win'] == 1]
        losses = df_trades[df_trades['win'] == 0]
        
        total_pnl = df_trades['pnl'].sum()
        total_pnl_pct = df_trades['pnl_pct'].sum()
        
        # Sharpe ratio (assuming 0% risk-free rate)
        pnl_series = df_trades['pnl_pct'].values
        sharpe = np.mean(pnl_series) / (np.std(pnl_series) + 1e-9) * np.sqrt(252)
        
        # Max drawdown
        cumsum = df_trades['pnl'].cumsum()
        running_max = cumsum.expanding().max()
        drawdown = (cumsum - running_max) / (running_max + 1e-9)
        max_dd = drawdown.min()
        
        # Consecutive losses
        max_consecutive = 0
        current_consecutive = 0
        for win in df_trades['win']:
            if win == 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return {
            'total_trades': len(df_trades),
            'win_rate': (len(wins) / len(df_trades)) * 100 if len(df_trades) > 0 else 0,
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
            'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd * 100,
            'consecutive_losses': max_consecutive,
            'avg_bars_held': df_trades['bars_held'].mean()
        }
    
    def run_diagnostic(self):
        """Run full diagnostic backtest"""
        logger.info("=" * 70)
        logger.info("DIAGNOSTIC BACKTEST - CURRENT SYSTEM ANALYSIS")
        logger.info("=" * 70)
        
        all_trades = []
        
        for symbol in self.symbols:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"SYMBOL: {symbol}")
            logger.info(f"{'=' * 70}")
            
            # Load data
            df = self.load_backtest_data(symbol)
            if df.empty:
                logger.warning(f"[{symbol}] Skipped - no data")
                continue
            
            # Calculate features BEFORE lowercasing (apply_features needs proper case)
            df = self.calculate_features(df)
            
            # Add lowercase 'close' column for simulation logic (keep original case for model features)
            if 'Close' in df.columns:
                df['close'] = df['Close']
            if 'Open' in df.columns:
                df['open'] = df['Open']
            
            if 'close' not in df.columns:
                logger.warning(f"[{symbol}] Invalid data format")
                continue
            
            # Load models
            models_dict = self.load_ml_models(symbol)
            if not models_dict['models']:
                logger.warning(f"[{symbol}] No models found")
                continue
            
            # Simulate trades (df still has proper case feature columns)
            trades = self.simulate_trades(symbol, df, models_dict['models'], 
                                         models_dict['features'])
            
            if not trades:
                logger.warning(f"[{symbol}] No trades generated")
                continue
            
            # Calculate metrics
            metrics = self.calculate_metrics(trades)
            
            # Store results
            self.results[symbol] = metrics
            all_trades.extend(trades)
            
            # Print results
            logger.info(f"\n📊 Performance Metrics ({len(trades)} trades):")
            logger.info(f"  Win Rate:           {metrics['win_rate']:.2f}%")
            logger.info(f"  Avg Win:            ${metrics['avg_win']:.2f}")
            logger.info(f"  Avg Loss:           ${metrics['avg_loss']:.2f}")
            logger.info(f"  Profit Factor:      {metrics['profit_factor']:.2f}")
            logger.info(f"  Total P&L:          ${metrics['total_pnl']:.2f} ({metrics['total_pnl_pct']:.2f}%)")
            logger.info(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
            logger.info(f"  Max Drawdown:       {metrics['max_drawdown']:.2f}%")
            logger.info(f"  Consecutive Losses: {metrics['consecutive_losses']}")
            logger.info(f"  Avg Bars Held:      {metrics['avg_bars_held']:.0f}")
        
        # Overall summary
        if all_trades:
            logger.info(f"\n{'=' * 70}")
            logger.info("OVERALL SUMMARY")
            logger.info(f"{'=' * 70}")
            overall_metrics = self.calculate_metrics(all_trades)
            logger.info(f"Total Trades (all symbols):  {overall_metrics['total_trades']}")
            logger.info(f"Win Rate:                    {overall_metrics['win_rate']:.2f}%")
            logger.info(f"Profit Factor:               {overall_metrics['profit_factor']:.2f}")
            logger.info(f"Sharpe Ratio:                {overall_metrics['sharpe_ratio']:.2f}")
            logger.info(f"Max Drawdown:                {overall_metrics['max_drawdown']:.2f}%")
            logger.info(f"Total P&L:                   ${overall_metrics['total_pnl']:.2f}")
            
            # Interpretation
            logger.info(f"\n{'=' * 70}")
            logger.info("INTERPRETATION & RECOMMENDATIONS")
            logger.info(f"{'=' * 70}")
            
            wr = overall_metrics['win_rate']
            pf = overall_metrics['profit_factor']
            sharpe = overall_metrics['sharpe_ratio']
            dd = overall_metrics['max_drawdown']
            
            if wr < 45:
                logger.warning(f"⚠️  Win rate {wr:.1f}% is LOW - system may need threshold tuning")
            elif wr >= 55:
                logger.info(f"✅ Win rate {wr:.1f}% is GOOD - consider scaling up")
            else:
                logger.info(f"⚠️  Win rate {wr:.1f}% is MARGINAL - needs improvement")
            
            if pf < 1.0:
                logger.warning(f"⚠️  Profit factor {pf:.2f} < 1.0 - avg losses exceed wins")
            elif pf >= 1.5:
                logger.info(f"✅ Profit factor {pf:.2f} is STRONG")
            else:
                logger.info(f"⚠️  Profit factor {pf:.2f} is WEAK - consider tighter stops")
            
            if sharpe < 1.0:
                logger.warning(f"⚠️  Sharpe ratio {sharpe:.2f} is LOW - poor risk-adjusted returns")
            elif sharpe >= 1.5:
                logger.info(f"✅ Sharpe ratio {sharpe:.2f} is GOOD")
            
            if dd > 10:
                logger.warning(f"⚠️  Max drawdown {dd:.1f}% is HIGH - consider reducing position size")
            elif dd < 5:
                logger.info(f"✅ Max drawdown {dd:.1f}% is acceptable")

if __name__ == "__main__":
    backtest = DiagnosticBacktester()
    backtest.run_diagnostic()
