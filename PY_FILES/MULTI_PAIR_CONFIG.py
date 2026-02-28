"""
Multi-Pair Configuration for FOREX Trading Bot
Supports all major forex pairs with their characteristics
"""

from typing import TypedDict

class PairConfig(TypedDict):
    """Configuration for a forex pair"""
    symbol: str
    description: str
    pip_value: float
    volatility_rank: int  # 1=highest, 7=lowest (among major pairs)
    correlation_group: str  # For diversification
    data_file_5m: str
    model_targets: list[str]


# Major forex pairs configuration
MAJOR_PAIRS: dict[str, PairConfig] = {
    "EURUSD": {
        "symbol": "EURUSD",
        "description": "Euro vs US Dollar - Most liquid, tight spreads",
        "pip_value": 0.0001,
        "volatility_rank": 4,
        "correlation_group": "EUR_GROUP",
        "data_file_5m": "CSV_FILES/MT5_5M_BT_EURUSD_Dataset.csv",
        "model_targets": ["T_5M", "T_10M", "T_15M", "T_20M", "T_30M"]
    },
    "GBPUSD": {
        "symbol": "GBPUSD",
        "description": "British Pound vs US Dollar - Volatile, wider swings",
        "pip_value": 0.0001,
        "volatility_rank": 1,  # Most volatile
        "correlation_group": "GBP_GROUP",
        "data_file_5m": "CSV_FILES/MT5_5M_BT_GBPUSD_Dataset.csv",
        "model_targets": ["T_5M", "T_10M", "T_15M", "T_20M", "T_30M"]
    },
    "USDJPY": {
        "symbol": "USDJPY",
        "description": "US Dollar vs Japanese Yen - Carry trade, safe haven",
        "pip_value": 0.01,  # Note: JPY has different pip value
        "volatility_rank": 3,
        "correlation_group": "JPY_GROUP",
        "data_file_5m": "CSV_FILES/MT5_5M_BT_USDJPY_Dataset.csv",
        "model_targets": ["T_5M", "T_10M", "T_15M", "T_20M", "T_30M"]
    },
    "AUDUSD": {
        "symbol": "AUDUSD",
        "description": "Australian Dollar vs US Dollar - Commodity correlated",
        "pip_value": 0.0001,
        "volatility_rank": 5,
        "correlation_group": "AUD_GROUP",
        "data_file_5m": "CSV_FILES/MT5_5M_BT_AUDUSD_Dataset.csv",
        "model_targets": ["T_5M", "T_10M", "T_15M", "T_20M", "T_30M"]
    },
    "NZDUSD": {
        "symbol": "NZDUSD",
        "description": "New Zealand Dollar vs US Dollar - Commodity correlated",
        "pip_value": 0.0001,
        "volatility_rank": 6,
        "correlation_group": "NZD_GROUP",
        "data_file_5m": "CSV_FILES/MT5_5M_BT_NZDUSD_Dataset.csv",
        "model_targets": ["T_5M", "T_10M", "T_15M", "T_20M", "T_30M"]
    },
    "USDCAD": {
        "symbol": "USDCAD",
        "description": "US Dollar vs Canadian Dollar - Oil correlated",
        "pip_value": 0.0001,
        "volatility_rank": 7,  # Least volatile
        "correlation_group": "CAD_GROUP",
        "data_file_5m": "CSV_FILES/MT5_5M_BT_USDCAD_Dataset.csv",
        "model_targets": ["T_5M", "T_10M", "T_15M", "T_20M", "T_30M"]
    },
    "USDHKD": {
        "symbol": "USDHKD",
        "description": "US Dollar vs Hong Kong Dollar - Pegged currency",
        "pip_value": 0.0001,
        "volatility_rank": 7,
        "correlation_group": "HKD_GROUP",
        "data_file_5m": "CSV_FILES/MT5_5M_BT_USDHKD_Dataset.csv",
        "model_targets": ["T_5M", "T_10M", "T_15M", "T_20M", "T_30M"]
    },
}

# Cross pairs (non-USD pairs) for advanced users
CROSS_PAIRS: dict[str, PairConfig] = {
    "EURGBP": {
        "symbol": "EURGBP",
        "description": "Euro vs British Pound - Tight range, low spreads",
        "pip_value": 0.0001,
        "volatility_rank": 5,
        "correlation_group": "EUR_GBP",
        "data_file_5m": "CSV_FILES/MT5_5M_BT_EURGBP_Dataset.csv",
        "model_targets": ["T_5M", "T_10M", "T_15M", "T_20M", "T_30M"]
    },
    "EURJPY": {
        "symbol": "EURJPY",
        "description": "Euro vs Japanese Yen - Moderate volatility",
        "pip_value": 0.01,
        "volatility_rank": 3,
        "correlation_group": "EUR_JPY",
        "data_file_5m": "CSV_FILES/MT5_5M_BT_EURJPY_Dataset.csv",
        "model_targets": ["T_5M", "T_10M", "T_15M", "T_20M", "T_30M"]
    },
    "GBPJPY": {
        "symbol": "GBPJPY",
        "description": "British Pound vs Japanese Yen - High volatility",
        "pip_value": 0.01,
        "volatility_rank": 2,
        "correlation_group": "GBP_JPY",
        "data_file_5m": "CSV_FILES/MT5_5M_BT_GBPJPY_Dataset.csv",
        "model_targets": ["T_5M", "T_10M", "T_15M", "T_20M", "T_30M"]
    },
    "AUDNZD": {
        "symbol": "AUDNZD",
        "description": "Australian Dollar vs New Zealand Dollar - Commodity pairs",
        "pip_value": 0.0001,
        "volatility_rank": 4,
        "correlation_group": "AUD_NZD",
        "data_file_5m": "CSV_FILES/MT5_5M_BT_AUDNZD_Dataset.csv",
        "model_targets": ["T_5M", "T_10M", "T_15M", "T_20M", "T_30M"]
    },
}

# All available pairs
ALL_PAIRS: dict[str, PairConfig] = {**MAJOR_PAIRS, **CROSS_PAIRS}


class StrategyConfig(TypedDict):
    """Configuration for strategy execution"""
    strategy_type: str  # "pure_smc", "hybrid_smc_ml", "ml_only"
    pairs: list[str]
    confidence_threshold: float
    atr_sl_multiplier: float
    atr_tp_multiplier: float
    spread_pips: float
    slippage_pips: float
    position_risk_percent: float


# Recommended configurations
CONFIGS: dict[str, StrategyConfig] = {
    "conservative": {
        "strategy_type": "hybrid_smc_ml",
        "pairs": list(MAJOR_PAIRS.keys()),
        "confidence_threshold": 60,
        "atr_sl_multiplier": 2.0,
        "atr_tp_multiplier": 5.0,
        "spread_pips": 1.2,
        "slippage_pips": 0.2,
        "position_risk_percent": 1.0
    },
    "balanced": {
        "strategy_type": "hybrid_smc_ml",
        "pairs": list(MAJOR_PAIRS.keys()),
        "confidence_threshold": 55,
        "atr_sl_multiplier": 1.5,
        "atr_tp_multiplier": 4.5,
        "spread_pips": 1.2,
        "slippage_pips": 0.2,
        "position_risk_percent": 2.0
    },
    "aggressive": {
        "strategy_type": "ml_only",
        "pairs": list(MAJOR_PAIRS.keys()),
        "confidence_threshold": 50,
        "atr_sl_multiplier": 1.0,
        "atr_tp_multiplier": 3.0,
        "spread_pips": 1.2,
        "slippage_pips": 0.2,
        "position_risk_percent": 3.0
    },
    "multi_pair_diversified": {
        "strategy_type": "hybrid_smc_ml",
        "pairs": list(ALL_PAIRS.keys()),
        "confidence_threshold": 55,
        "atr_sl_multiplier": 1.5,
        "atr_tp_multiplier": 4.5,
        "spread_pips": 1.5,
        "slippage_pips": 0.3,
        "position_risk_percent": 0.5  # Lower risk per pair with more pairs
    }
}


def get_pair_config(symbol: str) -> PairConfig | None:
    """Get configuration for a specific pair"""
    return ALL_PAIRS.get(symbol)


def get_pairs_by_volatility(ascending: bool = False) -> list[str]:
    """Get pairs sorted by volatility"""
    sorted_pairs = sorted(MAJOR_PAIRS.items(), 
                         key=lambda x: x[1]["volatility_rank"],
                         reverse=not ascending)
    return [symbol for symbol, _ in sorted_pairs]


def get_pairs_by_group(group: str) -> list[str]:
    """Get all pairs in a correlation group"""
    return [symbol for symbol, config in ALL_PAIRS.items() 
            if config["correlation_group"] == group]


def validate_pair(symbol: str) -> bool:
    """Check if pair is configured"""
    return symbol in ALL_PAIRS


if __name__ == "__main__":
    print("\n" + "="*80)
    print("FOREX TRADING BOT - MULTI-PAIR CONFIGURATION")
    print("="*80)
    
    print("\n📊 MAJOR PAIRS (7 total):")
    for symbol, config in MAJOR_PAIRS.items():
        print(f"  {symbol:10} | Volatility: {config['volatility_rank']}/7 | {config['description']}")
    
    print("\n🔄 CROSS PAIRS (4 total):")
    for symbol, config in CROSS_PAIRS.items():
        print(f"  {symbol:10} | Volatility: {config['volatility_rank']}/7 | {config['description']}")
    
    print("\n⚙️  RECOMMENDED CONFIGURATIONS:")
    for config_name, config in CONFIGS.items():
        print(f"\n  {config_name.upper()}:")
        print(f"    Strategy: {config['strategy_type']}")
        print(f"    Pairs: {len(config['pairs'])} ({', '.join(config['pairs'][:3])}...)")
        print(f"    Confidence Threshold: {config['confidence_threshold']}%")
        print(f"    Risk per Trade: {config['position_risk_percent']}%")
    
    print("\n" + "="*80)
