#!/usr/bin/env python3
"""
Clean and preprocess all forex data for model training
"""

import pandas as pd
from pathlib import Path

def clean_forex_data(input_file: str, output_file: str) -> bool:
    """
    Clean forex data from yfinance format to standard OHLCV
    """
    try:
        print(f"  Cleaning {Path(input_file).name}...")
        
        # Read data
        df = pd.read_csv(input_file)
        
        # Handle yfinance format with multiple header rows
        if 'Price' in df.columns and 'Ticker' in df['Price'].values:
            # Skip the extra header rows
            df = pd.read_csv(input_file, skiprows=2)
        
        # Drop non-OHLCV columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Rename columns if needed
        df.columns = [col.strip() for col in df.columns]
        
        # Keep only OHLCV columns
        available_cols = [col for col in required_cols if col in df.columns]
        if len(available_cols) < 4:
            print(f"    ✗ Missing required columns. Found: {available_cols}")
            return False
        
        df = df[available_cols].copy()
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN
        df = df.dropna()
        
        # Sort by index (oldest first)
        df = df.iloc[::-1].reset_index(drop=True)
        
        if len(df) < 1000:
            print(f"    ✗ Insufficient data: {len(df)} rows")
            return False
        
        # Save cleaned data
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        
        print(f"    ✓ Cleaned {len(df)} records → {output_file}")
        return True
        
    except Exception as e:
        print(f"    ✗ Error: {str(e)}")
        return False


def clean_all_data():
    """Clean all pair data"""
    
    pairs = {
        "EURUSD": "MT5_5M_BT_EURUSD_Dataset.csv",
        "GBPUSD": "MT5_5M_BT_GBPUSD_Dataset.csv",
        "USDJPY": "MT5_5M_BT_USDJPY_Dataset.csv",
        "AUDUSD": "MT5_5M_BT_AUDUSD_Dataset.csv",
        "NZDUSD": "MT5_5M_BT_NZDUSD_Dataset.csv",
        "USDCAD": "MT5_5M_BT_USDCAD_Dataset.csv",
        "USDHKD": "MT5_5M_BT_USDHKD_Dataset.csv",
        "EURGBP": "MT5_5M_BT_EURGBP_Dataset.csv",
        "EURJPY": "MT5_5M_BT_EURJPY_Dataset.csv",
        "GBPJPY": "MT5_5M_BT_GBPJPY_Dataset.csv",
        "AUDNZD": "MT5_5M_BT_AUDNZD_Dataset.csv"
    }
    
    print("\n" + "="*80)
    print("CLEANING FOREX DATA")
    print("="*80 + "\n")
    
    csv_dir = Path("CSV_FILES")
    successful = 0
    failed = 0
    
    for symbol, filename in pairs.items():
        input_file = csv_dir / filename
        
        if not input_file.exists():
            print(f"  ✗ {symbol}: File not found")
            failed += 1
            continue
        
        # Clean in-place
        if clean_forex_data(str(input_file), str(input_file)):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"✓ Cleaned: {successful}/{len(pairs)}")
    print(f"✗ Failed: {failed}/{len(pairs)}")
    print("="*80 + "\n")
    
    return successful == len(pairs)


if __name__ == "__main__":
    if clean_all_data():
        print("✓ All data cleaned successfully!")
        print("Ready for model training.")
    else:
        print("⚠️  Some data cleaning failed")
