#!/usr/bin/env python3
"""
Fix forex data structure and prepare for training
"""

import pandas as pd

def fix_forex_data(symbol: str):
    """Fix the structure of downloaded forex data"""
    
    try:
        print(f"  Fixing {symbol}...")
        
        filename = f"CSV_FILES/MT5_5M_BT_{symbol}_Dataset.csv"
        
        # Read with proper header handling
        df = pd.read_csv(filename)
        
        # Remove metadata rows (Price, Ticker, Datetime)
        # Keep only numeric data
        df = df[df['Price'] != 'Price']
        df = df[df['Price'] != 'Ticker']
        df = df[df['Price'] != 'Datetime']
        
        # Rename columns properly
        df = df.rename(columns={'Price': 'Datetime'})
        
        # Set datetime index
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        
        # Remove rows with invalid datetimes
        df = df.dropna(subset=['Datetime'])
        
        # Convert columns to numeric
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN
        df = df.dropna(subset=numeric_cols)
        
        # Set datetime as index
        df = df.set_index('Datetime')
        df = df.sort_index()
        
        # Keep only OHLCV columns
        df = df[numeric_cols]
        
        if len(df) < 1000:
            print(f"    ✗ Insufficient data: {len(df)} rows")
            return False
        
        # Save
        df.to_csv(filename)
        print(f"    ✓ Fixed: {len(df)} rows, {df.isnull().sum().sum()} missing values")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Error: {str(e)}")
        return False


def fix_all_data():
    """Fix all pair data"""
    
    pairs = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD",
        "USDCAD", "USDHKD", "EURGBP", "EURJPY", "GBPJPY", "AUDNZD"
    ]
    
    print("\n" + "="*80)
    print("FIXING FOREX DATA STRUCTURE")
    print("="*80 + "\n")
    
    successful = 0
    for symbol in pairs:
        if fix_forex_data(symbol):
            successful += 1
    
    print(f"\n{'='*80}")
    print(f"✓ Fixed: {successful}/{len(pairs)}")
    print("="*80 + "\n")
    
    return successful == len(pairs)


if __name__ == "__main__":
    if fix_all_data():
        print("✓ All data fixed successfully!")
    else:
        print("⚠️  Some data fixing failed")
