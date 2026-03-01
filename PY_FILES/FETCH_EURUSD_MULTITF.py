#!/usr/bin/env python3
"""
Fetch EURUSD weekly and monthly real data from yfinance and other sources
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

try:
    import yfinance as yf
    yfinance_available = True
except ImportError:
    yfinance_available = False
    print("Installing yfinance...")
    import os
    os.system("pip install yfinance -q")
    import yfinance as yf
    yfinance_available = True

try:
    from pandas_datareader import data as pdr
    pdr_available = True
except ImportError:
    pdr_available = False


def fetch_eurusd_from_yfinance(interval: str = '1wk', period: str = '10y') -> pd.DataFrame:
    """
    Fetch EURUSD data from Yahoo Finance
    Intervals: '1m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo'
    """
    print(f"  Fetching EURUSD {interval} from yfinance ({period})...", end=" ", flush=True)
    
    try:
        df = yf.download('EURUSD=X', interval=interval, period=period, progress=False)
        
        if df is None or len(df) == 0:
            print("✗ No data")
            return None
        
        # Normalize columns
        df = df.rename(columns={
            'Open': 'Open', 'High': 'High', 'Low': 'Low',
            'Close': 'Close', 'Volume': 'Volume'
        })
        
        # Reset index to make Date a column
        df = df.reset_index()
        df = df.rename(columns={'Date': 'Date', 'Datetime': 'Date'})
        
        print(f"✓ Loaded {len(df):,} bars")
        return df
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def fetch_eurusd_monthly_alternative() -> pd.DataFrame:
    """
    Fetch monthly EURUSD from alternative source
    Uses Yahoo Finance 1-month interval
    """
    print(f"  Fetching EURUSD monthly from yfinance (15y)...", end=" ", flush=True)
    
    try:
        # Download with monthly interval
        df = yf.download('EURUSD=X', interval='1mo', period='15y', progress=False)
        
        if df is None or len(df) == 0:
            print("✗ No data")
            return None
        
        df = df.reset_index()
        df = df.rename(columns={'Date': 'Date'})
        
        print(f"✓ Loaded {len(df):,} bars")
        return df
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def aggregate_5m_to_weekly(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5M data to weekly OHLCV"""
    print("  Aggregating 5M to weekly...", end=" ", flush=True)
    
    try:
        df = df_5m.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Group by week
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Year'] = df['Date'].dt.isocalendar().year
        df['YearWeek'] = df['Year'].astype(str) + '-W' + df['Week'].astype(str).str.zfill(2)
        
        # Aggregate
        agg_df = df.groupby('YearWeek', sort=False).agg({
            'Date': 'first',
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).reset_index(drop=True)
        
        print(f"✓ Generated {len(agg_df):,} weekly bars")
        return agg_df
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def aggregate_5m_to_monthly(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5M data to monthly OHLCV"""
    print("  Aggregating 5M to monthly...", end=" ", flush=True)
    
    try:
        df = df_5m.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Group by month
        df['YearMonth'] = df['Date'].dt.to_period('M')
        
        # Aggregate
        agg_df = df.groupby('YearMonth', sort=False).agg({
            'Date': 'first',
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).reset_index(drop=True)
        
        print(f"✓ Generated {len(agg_df):,} monthly bars")
        return agg_df
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def save_csv(df: pd.DataFrame, filename: str, directory: str = "CSV_FILES") -> bool:
    """Save dataframe to CSV"""
    try:
        filepath = Path(directory) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure proper column order
        cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[[c for c in cols if c in df.columns]]
        
        df.to_csv(filepath, index=False)
        print(f"    Saved: {filename}")
        return True
    except Exception as e:
        print(f"    ✗ Failed to save {filename}: {e}")
        return False


def main():
    print("\n" + "="*80)
    print("EURUSD DATA FETCHING - WEEKLY & MONTHLY")
    print("="*80 + "\n")
    
    # Load existing 5M data
    print("Loading existing 5M EURUSD data...")
    df_5m_path = Path("CSV_FILES/MT5_5M_EURUSD_Exchange_Rate_Dataset.csv")
    if df_5m_path.exists():
        df_5m = pd.read_csv(df_5m_path, header=None)
        if len(df_5m.columns) >= 5:
            df_5m.columns = ['Date', 'Open', 'High', 'Low', 'Close'] + list(df_5m.columns[5:])
        print(f"✓ Loaded {len(df_5m):,} bars from 5M data\n")
    else:
        print(f"✗ 5M data not found at {df_5m_path}\n")
        df_5m = None
    
    # Try fetching weekly from yfinance
    print("Fetching Weekly Data:")
    df_weekly_yf = fetch_eurusd_from_yfinance(interval='1wk', period='10y')
    
    # Try aggregating from 5M
    if df_5m is not None:
        df_weekly_agg = aggregate_5m_to_weekly(df_5m)
    else:
        df_weekly_agg = None
    
    # Use whichever has more data
    if df_weekly_yf is not None and df_weekly_agg is not None:
        print(f"  Combining: yfinance ({len(df_weekly_yf)} bars) + aggregated ({len(df_weekly_agg)} bars)")
        df_weekly = pd.concat([df_weekly_agg, df_weekly_yf], ignore_index=True)
        df_weekly = df_weekly.drop_duplicates(subset=['Date'], keep='last')
        df_weekly = df_weekly.sort_values('Date')
        print(f"  Final: {len(df_weekly):,} unique bars\n")
    elif df_weekly_yf is not None:
        df_weekly = df_weekly_yf
        print()
    elif df_weekly_agg is not None:
        df_weekly = df_weekly_agg
        print()
    else:
        print("  ✗ No weekly data available\n")
        df_weekly = None
    
    # Save weekly
    if df_weekly is not None and len(df_weekly) > 0:
        save_csv(df_weekly, "EURUSD_Weekly.csv")
    
    # Try fetching monthly from yfinance
    print("\nFetching Monthly Data:")
    df_monthly_yf = fetch_eurusd_monthly_alternative()
    
    # Try aggregating from 5M
    if df_5m is not None:
        df_monthly_agg = aggregate_5m_to_monthly(df_5m)
    else:
        df_monthly_agg = None
    
    # Use whichever has more data
    if df_monthly_yf is not None and df_monthly_agg is not None:
        print(f"  Combining: yfinance ({len(df_monthly_yf)} bars) + aggregated ({len(df_monthly_agg)} bars)")
        df_monthly = pd.concat([df_monthly_agg, df_monthly_yf], ignore_index=True)
        df_monthly = df_monthly.drop_duplicates(subset=['Date'], keep='last')
        df_monthly = df_monthly.sort_values('Date')
        print(f"  Final: {len(df_monthly):,} unique bars\n")
    elif df_monthly_yf is not None:
        df_monthly = df_monthly_yf
        print()
    elif df_monthly_agg is not None:
        df_monthly = df_monthly_agg
        print()
    else:
        print("  ✗ No monthly data available\n")
        df_monthly = None
    
    # Save monthly
    if df_monthly is not None and len(df_monthly) > 0:
        save_csv(df_monthly, "EURUSD_Monthly.csv")
    
    print("\n" + "="*80)
    print("✓ Data fetching complete!")
    print("="*80)


if __name__ == "__main__":
    main()
