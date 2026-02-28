"""
Multi-Pair Strategy Comparison
Compare strategy performance across all major pairs
"""

import pandas as pd
from typing import Any

from MULTI_PAIR_CONFIG import MAJOR_PAIRS, CROSS_PAIRS, ALL_PAIRS, get_pairs_by_volatility


def generate_strategy_comparison_report() -> None:
    """Generate comprehensive comparison across all pairs and strategies"""
    
    print("\n" + "="*100)
    print("MULTI-PAIR STRATEGY COMPARISON - COMPREHENSIVE REPORT")
    print("="*100)
    
    print("\n📊 MAJOR PAIRS OVERVIEW")
    print("-" * 100)
    
    major_data: dict[str, list[Any]] = {
        'Pair': [],
        'Description': [],
        'Volatility': [],
        'Pip Value': [],
        'Correlation Group': [],
    }
    
    for symbol, config in MAJOR_PAIRS.items():
        major_data['Pair'].append(symbol)
        major_data['Description'].append(config['description'])
        major_data['Volatility'].append(f"{config['volatility_rank']}/7")
        major_data['Pip Value'].append(config['pip_value'])
        major_data['Correlation Group'].append(config['correlation_group'])
    
    major_df: pd.DataFrame = pd.DataFrame(major_data)
    print("\n" + major_df.to_string(index=False))  # type: ignore[reportUnknownMemberType]
    
    print("\n\n🔄 CROSS PAIRS OVERVIEW")
    print("-" * 100)
    
    cross_data: dict[str, list[Any]] = {
        'Pair': [],
        'Description': [],
        'Volatility': [],
        'Pip Value': [],
        'Notes': [],
    }
    
    for symbol, config in CROSS_PAIRS.items():
        cross_data['Pair'].append(symbol)
        cross_data['Description'].append(config['description'])
        cross_data['Volatility'].append(f"{config['volatility_rank']}/7")
        cross_data['Pip Value'].append(config['pip_value'])
        cross_data['Notes'].append("Tight liquidity" if symbol in ["EURGBP"] else "Higher spreads")
    
    cross_df: pd.DataFrame = pd.DataFrame(cross_data)
    print("\n" + cross_df.to_string(index=False))  # type: ignore[reportUnknownMemberType]
    
    print("\n\n⚙️  STRATEGY RECOMMENDATIONS BY PAIR")
    print("-" * 100)
    
    recommendations = {
        "EURUSD": "✓ Best for beginners - Tight spreads, high liquidity. Use Hybrid SMC+ML.",
        "GBPUSD": "⚠️ High volatility - Good profit potential but wider swings. Use larger stops.",
        "USDJPY": "💡 Carry trade pair - Different pip value. Use dedicated position sizing.",
        "AUDUSD": "✓ Commodity correlated - Good for diversification. Works well with SMC.",
        "NZDUSD": "✓ Similar to AUDUSD - Lower volatility. Good complementary pair.",
        "USDCAD": "✓ Oil correlated - Low volatility. Best for conservative traders.",
        "USDHKD": "⚠️ Pegged pair - Limited movement. Use for scalping strategies.",
    }
    
    for pair, note in recommendations.items():
        print(f"  {pair:8} | {note}")
    
    print("\n\n📈 VOLATILITY RANKING (HIGH → LOW)")
    print("-" * 100)
    
    sorted_pairs = get_pairs_by_volatility(ascending=False)
    for i, symbol in enumerate(sorted_pairs, 1):
        config = ALL_PAIRS[symbol]
        print(f"  {i}. {symbol:8} | Volatility: {config['volatility_rank']}/7")
    
    print("\n\n🎯 DEPLOYMENT STRATEGIES")
    print("-" * 100)
    
    strategies: dict[str, dict[str, str | list[str]]] = {
        "Conservative (1% risk/trade)": {
            "pairs": "EURUSD, USDCAD, AUDUSD",
            "reasoning": "Low volatility pairs, tight spreads, minimal risk exposure"
        },
        "Balanced (2% risk/trade)": {
            "pairs": "EURUSD, GBPUSD, USDJPY, AUDUSD",
            "reasoning": "Mix of major pairs with varying characteristics"
        },
        "Aggressive (3% risk/trade)": {
            "pairs": "GBPUSD, EURUSD, USDJPY, AUDUSD, NZDUSD",
            "reasoning": "All major pairs with higher position sizes"
        },
        "Multi-Pair Diversified (0.5% risk/pair)": {
            "pairs": ", ".join(MAJOR_PAIRS.keys()),
            "reasoning": "Diversification across 7 pairs reduces single-pair risk"
        },
        "Aggressive Multi-Pair (1% risk/pair)": {
            "pairs": ", ".join(ALL_PAIRS.keys()),
            "reasoning": "Maximum diversification with both major and cross pairs"
        }
    }
    
    for strategy_name, strategy_info in strategies.items():
        print(f"\n  {strategy_name}")
        print(f"    Pairs: {strategy_info['pairs']}")
        print(f"    Reasoning: {strategy_info['reasoning']}")
    
    print("\n\n💡 PAIR CORRELATION & PORTFOLIO OPTIMIZATION")
    print("-" * 100)
    
    correlation_groups: dict[str, list[str]] = {
        "EUR_GROUP": ["EURUSD", "EURGBP", "EURJPY"],
        "GBP_GROUP": ["GBPUSD", "EURGBP", "GBPJPY"],
        "JPY_GROUP": ["USDJPY", "EURJPY", "GBPJPY"],
        "AUD_GROUP": ["AUDUSD", "AUDNZD"],
        "NZD_GROUP": ["NZDUSD", "AUDNZD"],
        "CAD_GROUP": ["USDCAD"],
        "HKD_GROUP": ["USDHKD"]
    }
    
    print("\n  Correlation Groups (Pairs moving together):")
    for group, pairs in correlation_groups.items():
        print(f"    {group:15} → {', '.join(pairs)}")
    
    print("\n  Portfolio Allocation Tips:")
    print("    • Avoid trading all pairs in same correlation group")
    print("    • Mix different currency groups for diversification")
    print("    • Monitor USD strength (affects all USD pairs)")
    print("    • JPY pairs respond to Japan economic data")
    print("    • Commodity pairs (AUD, NZD, CAD) follow commodity prices")
    
    print("\n\n📊 PAIR CHARACTERISTICS SUMMARY")
    print("-" * 100)
    
    characteristics_data: dict[str, list[Any]] = {
        'Pair': [],
        'Volatility': [],
        'Spreads': [],
        'Liquidity': [],
        'Best For': [],
    }
    
    pair_characteristics = {
        "EURUSD": ("Medium", "Very Tight", "Excellent", "All traders"),
        "GBPUSD": ("High", "Wide", "Good", "Experienced traders"),
        "USDJPY": ("Medium", "Tight", "Excellent", "Carry traders"),
        "AUDUSD": ("Low", "Tight", "Good", "Conservative traders"),
        "NZDUSD": ("Low", "Tight", "Good", "Conservative traders"),
        "USDCAD": ("Low", "Very Tight", "Good", "Conservative traders"),
        "USDHKD": ("Very Low", "Wide", "Fair", "Scalpers"),
    }
    
    for pair, (vol, spreads, liq, best) in pair_characteristics.items():
        characteristics_data['Pair'].append(pair)
        characteristics_data['Volatility'].append(vol)
        characteristics_data['Spreads'].append(spreads)
        characteristics_data['Liquidity'].append(liq)
        characteristics_data['Best For'].append(best)
    
    char_df: pd.DataFrame = pd.DataFrame(characteristics_data)
    print("\n" + char_df.to_string(index=False))  # type: ignore[reportUnknownMemberType]
    
    print("\n" + "="*100)
    print("END OF REPORT")
    print("="*100 + "\n")


if __name__ == "__main__":
    generate_strategy_comparison_report()
