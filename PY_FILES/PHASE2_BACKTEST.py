#!/usr/bin/env python3
"""
PHASE 2 BACKTEST: Tighter Stops + Entry Confluence Filters
Tests improvements to boost profit factor from 1.17 to 1.5+
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from func import apply_features


class Phase2BacktestImproved:
    """Phase 2 backtest with tighter stops and entry confluence"""
    
    def __init__(self):
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
        self.config_path = "../config.json"
        self.config = self.load_config()
        
        # PHASE 2 IMPROVEMENTS
        self.improvements = {
            "baseline": {
                "name": "Baseline (Current)",
                "stop_loss_pips": 50,
                "take_profit_pips": 100,
                "confluence_required": False,
                "min_confluence_signals": 0,
            },
            "tight_stops": {
                "name": "Tighter Stops Only",
                "stop_loss_pips": 25,  # Reduced from 50
                "take_profit_pips": 75,  # Reduced proportionally
                "confluence_required": False,
                "min_confluence_signals": 0,
            },
            "confluence_filter": {
                "name": "Confluence Filter (2+ signals)",
                "stop_loss_pips": 50,
                "take_profit_pips": 100,
                "confluence_required": True,
                "min_confluence_signals": 2,  # EMA + RSI
            },
            "tight_stops_confluence": {
                "name": "Tight Stops + Confluence (2+ signals)",
                "stop_loss_pips": 25,
                "take_profit_pips": 75,
                "confluence_required": True,
                "min_confluence_signals": 2,
            },
            "aggressive_confluence": {
                "name": "Tight Stops + Strong Confluence (3+ signals)",
                "stop_loss_pips": 20,
                "take_profit_pips": 60,
                "confluence_required": True,
                "min_confluence_signals": 3,  # EMA + RSI + MACD
            },
        }
    
    def load_config(self) -> Dict[str, Any]:
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
        timeframes = ["T_5M", "T_10M", "T_15M", "T_20M", "T_30M"]
        
        for tf in timeframes:
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
                # Try directory structure
                dir_path = Path(f"../ALL_MODELS/{symbol}/{tf}")
                if dir_path.exists():
                    try:
                        model_file = dir_path / "model.pkl"
                        if model_file.exists():
                            bundle = joblib.load(model_file)
                            models_dict[tf] = bundle.get("model")
                            feature_dict[tf] = bundle.get("features", [])
                    except Exception as e:
                        logger.warning(f"[{symbol}] Failed to load from {dir_path}: {e}")
        
        return {"models": models_dict, "features": feature_dict}
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using feature pipeline"""
        if df.empty:
            return df
        
        try:
            df_copy = df.copy()
            result = apply_features(df_copy)
            return result
        except Exception as e:
            logger.warning(f"Failed to use apply_features: {e}")
            return df.copy()
    
    def get_confluence_signals(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Generate confluence signals for entry filtering
        Returns dict with signals: ema_bullish, rsi_oversold, macd_bullish, etc.
        """
        signals = {
            "ema_bullish": 0,
            "rsi_oversold": 0,
            "macd_bullish": 0,
            "bb_bounce": 0,
        }
        
        if df.empty or len(df) < 2:
            return signals
        
        try:
            last = df.iloc[-1]
            
            # EMA signal: Close above EMA20 and EMA20 above EMA50
            if 'EMA_20' in df.columns and 'EMA_50' in df.columns:
                if last.get('Close', last.get('close', 0)) > last.get('EMA_20', 0):
                    if last.get('EMA_20', 0) > last.get('EMA_50', 0):
                        signals["ema_bullish"] = 1
            
            # RSI signal: Oversold (RSI < 30) for buy confluence
            if 'RSI' in df.columns:
                if last.get('RSI', 50) < 30:
                    signals["rsi_oversold"] = 1
            
            # MACD signal: MACD above signal line
            if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                if last.get('MACD', 0) > last.get('MACD_Signal', 0):
                    signals["macd_bullish"] = 1
            
            # Bollinger Bands signal: Close near lower band
            if 'BB_L' in df.columns and 'Close' in df.columns:
                close = last.get('Close', 0)
                bb_l = last.get('BB_L', 0)
                bb_mid = last.get('BB_Mid', close)
                if bb_l > 0 and bb_mid > 0:
                    bb_range = bb_mid - bb_l
                    if close < bb_l + (bb_range * 0.2):  # Within 20% of lower band
                        signals["bb_bounce"] = 1
        
        except Exception as e:
            logger.debug(f"Confluence calculation error: {e}")
        
        return signals
    
    def get_ml_signal(self, df: pd.DataFrame, model: Any, features: List[str], threshold: float = 0.52) -> np.ndarray:
        """Get ML signal with feature matching"""
        try:
            available_features = [f for f in features if f in df.columns]
            if not available_features:
                return np.zeros(len(df))
            
            X = df[available_features].fillna(0).values
            if X.shape[1] != len(features):
                return np.zeros(len(df))
            
            y_proba = model.predict_proba(X)[:, 1]
            signals = np.zeros(len(df))
            signals[y_proba >= threshold] = 1
            signals[y_proba < (1 - threshold)] = -1
            
            return signals
        except Exception as e:
            logger.debug(f"ML signal error: {e}")
            return np.zeros(len(df))
    
    def simulate_trades(self, symbol: str, df: pd.DataFrame, models_dict: Dict[str, Any],
                       features_dict: Dict[str, Any], improvement_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate trades with Phase 2 improvements"""
        trades = []
        
        if df.empty or not models_dict:
            return trades
        
        # Add lowercase close for simulation
        if 'Close' in df.columns:
            df['close'] = df['Close']
        
        # Calculate features
        df = self.calculate_features(df)
        
        # Get ML signals
        tf_signal = "T_5M"
        model = models_dict.get(tf_signal)
        features = features_dict.get(tf_signal, [])
        threshold = self.config.get("trading", {}).get("ml_threshold", 0.52)
        
        if model is None:
            logger.warning(f"[{symbol}] No model for {tf_signal}")
            return trades
        
        signals = self.get_ml_signal(df, model, features, threshold)
        
        # Simulation parameters
        stop_pips = improvement_config["stop_loss_pips"]
        target_pips = improvement_config["take_profit_pips"]
        require_confluence = improvement_config["confluence_required"]
        min_confluence = improvement_config["min_confluence_signals"]
        
        position = None
        entry_price = 0
        entry_idx = 0
        entry_signal = 0
        
        for i in range(1, len(df)):
            if signals[i] == 0:
                continue
            
            # Check confluence filter if required
            if require_confluence and signals[i] != 0:
                confluence = self.get_confluence_signals(df.iloc[:i+1])
                confluent_signals = sum(confluence.values())
                if confluent_signals < min_confluence:
                    continue  # Skip entry - not enough confluence
            
            # Exit signal
            if position and signals[i] == -entry_signal:
                exit_price = df.iloc[i]['close']
                pnl = (exit_price - entry_price) * entry_signal
                pnl_pips = pnl * 10000
                pnl_pct = (pnl / entry_price) * 100
                
                trades.append({
                    'symbol': symbol,
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pips': pnl_pips,
                    'pnl_pct': pnl_pct,
                    'bars_held': i - entry_idx,
                    'exit_reason': 'signal_flip',
                })
                position = None
            
            # Entry signal
            if not position and signals[i] != 0:
                entry_price = df.iloc[i]['close']
                entry_idx = i
                entry_signal = signals[i]
                position = entry_signal
            
            # Stop loss
            if position:
                current_price = df.iloc[i]['close']
                loss_pips = (entry_price - current_price) * entry_signal * 10000
                profit_pips = (current_price - entry_price) * entry_signal * 10000
                
                if loss_pips >= stop_pips:
                    trades.append({
                        'symbol': symbol,
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': (current_price - entry_price) * entry_signal,
                        'pnl_pips': -stop_pips,
                        'pnl_pct': (-stop_pips / (entry_price * 10000)) * 100,
                        'bars_held': i - entry_idx,
                        'exit_reason': 'stop_loss',
                    })
                    position = None
                
                # Take profit
                elif profit_pips >= target_pips:
                    trades.append({
                        'symbol': symbol,
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': (current_price - entry_price) * entry_signal,
                        'pnl_pips': target_pips,
                        'pnl_pct': (target_pips / (entry_price * 10000)) * 100,
                        'bars_held': i - entry_idx,
                        'exit_reason': 'take_profit',
                    })
                    position = None
        
        return trades
    
    def calculate_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'total_pnl_pct': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'consecutive_losses': 0,
                'avg_bars_held': 0,
            }
        
        pnls = [t['pnl'] for t in trades]
        pnl_pcts = [t['pnl_pct'] for t in trades]
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        win_rate = (len(wins) / len(pnls)) * 100 if pnls else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        total_profit = sum(wins)
        total_loss = abs(sum(losses))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        cumulative_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        sharpe = np.mean(pnl_pcts) / np.std(pnl_pcts) if np.std(pnl_pcts) > 0 else 0
        
        consecutive = 0
        max_consecutive = 0
        for p in pnls:
            if p < 0:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        bars_held = [t['bars_held'] for t in trades]
        avg_bars_held = np.mean(bars_held) if bars_held else 0
        
        return {
            'total_trades': len(pnls),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_pnl': sum(pnls),
            'total_pnl_pct': np.mean(pnl_pcts) * len(pnls) if pnl_pcts else 0,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'consecutive_losses': max_consecutive,
            'avg_bars_held': avg_bars_held,
        }
    
    def run_comparison(self):
        """Run backtest comparing all improvement scenarios"""
        logger.info("=" * 80)
        logger.info("PHASE 2 BACKTEST: Tighter Stops + Entry Confluence Filters")
        logger.info("=" * 80)
        
        all_results = {}
        
        for improvement_name, improvement_config in self.improvements.items():
            logger.info(f"\n{'=' * 80}")
            logger.info(f"TESTING: {improvement_config['name']}")
            logger.info(f"  Stop Loss: {improvement_config['stop_loss_pips']} pips")
            logger.info(f"  Take Profit: {improvement_config['take_profit_pips']} pips")
            logger.info(f"  Confluence Required: {improvement_config['confluence_required']}")
            logger.info(f"{'=' * 80}")
            
            all_trades = []
            
            for symbol in self.symbols:
                df = self.load_backtest_data(symbol)
                if df.empty:
                    logger.warning(f"[{symbol}] Skipped - no data")
                    continue
                
                models_dict = self.load_ml_models(symbol)
                if not models_dict['models']:
                    logger.warning(f"[{symbol}] Skipped - no models")
                    continue
                
                trades = self.simulate_trades(symbol, df, models_dict['models'],
                                            models_dict['features'], improvement_config)
                
                if trades:
                    metrics = self.calculate_metrics(trades)
                    logger.info(f"\n[{symbol}] {len(trades)} trades")
                    logger.info(f"  Win Rate: {metrics['win_rate']:.2f}%")
                    logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
                    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                    logger.info(f"  P&L: ${metrics['total_pnl']:.2f}")
                    all_trades.extend(trades)
                else:
                    logger.warning(f"[{symbol}] No trades generated")
            
            # Overall metrics
            if all_trades:
                overall = self.calculate_metrics(all_trades)
                all_results[improvement_name] = overall
                
                logger.info(f"\n{'─' * 80}")
                logger.info(f"OVERALL RESULTS ({improvement_config['name']})")
                logger.info(f"{'─' * 80}")
                logger.info(f"Total Trades: {overall['total_trades']}")
                logger.info(f"Win Rate: {overall['win_rate']:.2f}%")
                logger.info(f"Profit Factor: {overall['profit_factor']:.2f} ⭐")
                logger.info(f"Sharpe Ratio: {overall['sharpe_ratio']:.2f}")
                logger.info(f"Max Drawdown: {overall['max_drawdown']:.2f}")
                logger.info(f"Total P&L: ${overall['total_pnl']:.2f}")
        
        # Summary comparison
        self.print_comparison_summary(all_results)
    
    def print_comparison_summary(self, all_results: Dict[str, Any]):
        """Print comparison of all scenarios"""
        logger.info(f"\n{'=' * 80}")
        logger.info("PHASE 2 IMPROVEMENT SUMMARY")
        logger.info(f"{'=' * 80}")
        
        # Find best improvements
        best_pf = max(all_results.items(), key=lambda x: x[1]['profit_factor'])
        best_wr = max(all_results.items(), key=lambda x: x[1]['win_rate'])
        best_sharpe = max(all_results.items(), key=lambda x: x[1]['sharpe_ratio'])
        
        logger.info("\n📊 METRIC COMPARISON:")
        logger.info(f"{'Scenario':<35} {'Profit Factor':<15} {'Win Rate':<15} {'Sharpe':<10}")
        logger.info("─" * 75)
        
        for name, metrics in all_results.items():
            config = self.improvements[name]
            logger.info(f"{config['name']:<35} {metrics['profit_factor']:<15.2f} "
                       f"{metrics['win_rate']:<15.2f}% {metrics['sharpe_ratio']:<10.2f}")
        
        logger.info("\n🎯 BEST RESULTS:")
        logger.info(f"✅ Best Profit Factor: {self.improvements[best_pf[0]]['name']} ({best_pf[1]['profit_factor']:.2f})")
        logger.info(f"✅ Best Win Rate: {self.improvements[best_wr[0]]['name']} ({best_wr[1]['win_rate']:.2f}%)")
        logger.info(f"✅ Best Sharpe: {self.improvements[best_sharpe[0]]['name']} ({best_sharpe[1]['sharpe_ratio']:.2f})")
        
        logger.info("\n💡 RECOMMENDATIONS:")
        baseline_pf = all_results.get('baseline', {}).get('profit_factor', 1.17)
        for name, metrics in all_results.items():
            if metrics['profit_factor'] > baseline_pf:
                pf_improvement = ((metrics['profit_factor'] - baseline_pf) / baseline_pf) * 100
                logger.info(f"  → {self.improvements[name]['name']}: "
                           f"+{pf_improvement:.1f}% profit factor improvement")


if __name__ == "__main__":
    backtest = Phase2BacktestImproved()
    backtest.run_comparison()
