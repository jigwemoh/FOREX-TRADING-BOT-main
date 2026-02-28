"""
Multi-Pair Backtest Runner
Execute backtests across all major forex pairs simultaneously
"""

import os
from typing import Any

import pandas as pd

from MULTI_PAIR_CONFIG import MAJOR_PAIRS, CONFIGS, get_pair_config
from func import AnalysisResults, Trade, apply_features, create_targets, analyze_results
from SMC_Strategy import smc_strategy_features, smc_signal_generator, backtest_smc_strategy
from Hybrid_SMC_ML import backtest_hybrid_smc_ml


class MultiPairBacktester:
    """Execute backtests for multiple pairs"""
    
    def __init__(self, config_name: str = "balanced"):
        self.config = CONFIGS[config_name]
        self.results: dict[str, AnalysisResults] = {}
        self.trade_details: dict[str, list[Trade]] = {}
    
    def backtest_pair(self, symbol: str, strategy: str = "hybrid_smc_ml") -> bool:
        """Run backtest for a single pair"""
        try:
            pair_config = get_pair_config(symbol)
            if not pair_config:
                print(f"❌ {symbol}: Configuration not found")
                return False
            
            # Check if data file exists
            if not os.path.exists(pair_config["data_file_5m"]):
                print(f"⏭️  {symbol}: Data file not found ({pair_config['data_file_5m']})")
                return False
            
            print(f"\n📊 Backtesting {symbol}...", end=" ", flush=True)
            
            # Load data
            data: pd.DataFrame = pd.read_csv(pair_config["data_file_5m"])  # type: ignore[reportUnknownMemberType]
            
            # Apply features
            df: pd.DataFrame = apply_features(data)
            df = create_targets(df)
            
            # Apply SMC features for any strategy
            df = smc_strategy_features(df)
            df = smc_signal_generator(df)
            
            # Fill NaN values
            smc_cols = ['liquidity_level', 'bullish_fvg', 'bearish_fvg', 'bos_up', 
                       'bos_down', 'ob_mitigated_bullish', 'ob_mitigated_bearish', 'SMC_Signal']
            for col in smc_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)  # type: ignore[reportUnknownMemberType]
            
            # Clean data
            df = df.dropna(subset=['ATR', 'High', 'Low', 'Open', 'Close'])  # type: ignore[reportUnknownMemberType]
            
            if len(df) == 0:
                print("⚠️  No valid data")
                return False
            
            # Run backtest based on strategy
            if strategy == "hybrid_smc_ml":
                # Load ML models
                models_dict: dict[str, Any] = {}
                feature_cols_dict: dict[str, list[str]] = {}
                
                for target in pair_config["model_targets"][:1]:  # Start with T_5M
                    try:
                        import joblib  # type: ignore[import-untyped]
                        bundle: dict[str, Any] = joblib.load(f"ALL_MODELS/{symbol}_lgbm_{target}.pkl")  # type: ignore[reportUnknownMemberType]
                        models_dict[target] = bundle["model"]
                        feature_cols_dict[target] = bundle["features"]
                    except FileNotFoundError:
                        print(f"⏭️  {symbol}: ML model not found")
                        return False
                
                trades: list[Trade] = backtest_hybrid_smc_ml(
                    df, models_dict, feature_cols_dict,
                    threshold=self.config["confidence_threshold"],
                    atr_sl=self.config["atr_sl_multiplier"],
                    atr_tp=self.config["atr_tp_multiplier"],
                    spread_pips=self.config["spread_pips"],
                    slippage_pips=self.config["slippage_pips"]
                )
            else:
                # Pure SMC strategy
                trades = backtest_smc_strategy(df)
            
            # Store results
            analysis: AnalysisResults = analyze_results(trades)
            self.results[symbol] = analysis
            self.trade_details[symbol] = trades
            
            # Print summary
            print(f"✓ {analysis['total_trades']} trades | WR: {analysis['win_rate']:.2f}%")
            return True
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            return False
    
    def backtest_all_pairs(self) -> None:
        """Run backtest for all configured pairs"""
        print("\n" + "="*80)
        print(f"MULTI-PAIR BACKTEST - {self.config['strategy_type'].upper()}")
        print("="*80)
        
        successful = 0
        failed = 0
        
        for symbol in self.config["pairs"]:
            result = self.backtest_pair(symbol, self.config["strategy_type"])
            if result:
                successful += 1
            else:
                failed += 1
        
        # Print summary report
        self._print_summary_report(successful, failed, 0)
    
    def _print_summary_report(self, successful: int, failed: int, skipped: int) -> None:
        """Print comprehensive summary report"""
        print("\n" + "="*80)
        print("MULTI-PAIR BACKTEST SUMMARY")
        print("="*80)
        
        print(f"\n📈 RESULTS:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Skipped: {skipped}")
        
        if not self.results:
            print("\n⚠️  No results to display")
            return
        
        # Create summary table
        summary_data: dict[str, list[Any]] = {
            'Pair': [],
            'Total Trades': [],
            'Wins': [],
            'Losses': [],
            'Win Rate %': [],
        }
        
        for symbol in sorted(self.results.keys()):
            result = self.results[symbol]
            summary_data['Pair'].append(symbol)
            summary_data['Total Trades'].append(result['total_trades'])
            summary_data['Wins'].append(result['wins'])
            summary_data['Losses'].append(result['losses'])
            summary_data['Win Rate %'].append(f"{result['win_rate']:.2f}")
        
        summary_df: pd.DataFrame = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))  # type: ignore[reportUnknownMemberType]
        
        # Overall statistics
        total_trades = sum(r['total_trades'] for r in self.results.values())
        total_wins = sum(r['wins'] for r in self.results.values())
        overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        print("\n" + "="*80)
        print(f"PORTFOLIO STATISTICS:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Total Wins: {total_wins}")
        print(f"  Overall Win Rate: {overall_wr:.2f}%")
        print(f"  Average Win Rate per Pair: {sum(r['win_rate'] for r in self.results.values()) / len(self.results):.2f}%")
        print("="*80)
    
    def save_results(self, filename: str = "multi_pair_backtest_results.csv") -> None:
        """Save results to CSV"""
        if not self.results:
            print("⚠️  No results to save")
            return
        
        data: dict[str, list[Any]] = {
            'Pair': [],
            'Total Trades': [],
            'Wins': [],
            'Losses': [],
            'Win Rate': [],
        }
        
        for symbol in sorted(self.results.keys()):
            result = self.results[symbol]
            data['Pair'].append(symbol)
            data['Total Trades'].append(result['total_trades'])
            data['Wins'].append(result['wins'])
            data['Losses'].append(result['losses'])
            data['Win Rate'].append(result['win_rate'])
        
        df: pd.DataFrame = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"\n✓ Results saved to {filename}")


def run_multi_pair_backtest(config: str = "balanced", pairs: list[str] | None = None) -> None:
    """Convenience function to run multi-pair backtest"""
    backtester = MultiPairBacktester(config)
    
    if pairs:
        backtester.config['pairs'] = pairs
    
    backtester.backtest_all_pairs()
    backtester.save_results(f"CSV_FILES/multi_pair_backtest_{config}.csv")


if __name__ == "__main__":
    # Run balanced configuration on major pairs
    run_multi_pair_backtest("balanced", list(MAJOR_PAIRS.keys()))
