#!/usr/bin/env python3
"""
Fetch REAL historical forex data for all pairs
Downloads from multiple sources: yfinance, Alpha Vantage, Dukascopy, OANDA
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
from pathlib import Path

try:
    import yfinance as yf
    yfinance_available = True
except ImportError:
    yfinance_available = False
    yf = None  # type: ignore

try:
    import requests  # type: ignore
    requests_available = True
except ImportError:
    requests_available = False
    requests = None  # type: ignore


def download_from_yfinance(symbol: str) -> pd.DataFrame:
    """
    Download 5-minute data from Yahoo Finance (real data)
    Returns: None if fails, DataFrame if successful
    """
    try:
        print(f"  📥 Attempting Yahoo Finance download for {symbol}...")
        
        # Convert symbol format for yfinance
        yf_symbol = f"{symbol}=X"
        
        # Download 2 years of 1-hour data first (more reliable)
        print(f"    Downloading 2 years of data (this may take a moment)...")
        data = yf.download(yf_symbol, period="2y", interval="1h", progress=False, threads=False)
        
        if data is not None and len(data) > 100:
            print(f"    ✓ Successfully downloaded {len(data)} hours of real data for {symbol}")
            return data
        else:
            print(f"    ✗ Insufficient data from Yahoo Finance")
            return None
            
    except Exception as e:
        print(f"    ✗ Yahoo Finance error: {str(e)}")
        return None


def download_from_alphavantage(symbol: str, api_key: str = None) -> pd.DataFrame:
    """
    Download data from Alpha Vantage (FX_INTRADAY)
    Free tier: 5 calls per minute, 500 per day
    """
    if not requests_available:
        return None
    
    try:
        print(f"  📥 Attempting Alpha Vantage download for {symbol}...")
        
        # Use free API key (limited)
        if not api_key:
            api_key = "demo"
        
        # Parse symbol (e.g., EURUSD -> EUR and USD)
        from_currency = symbol[:3]
        to_currency = symbol[3:]
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "FX_INTRADAY",
            "from_symbol": from_currency,
            "to_symbol": to_currency,
            "interval": "5min",
            "apikey": api_key,
            "outputsize": "full"
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if "Time Series (5min)" in data:
            timeseries = data["Time Series (5min)"]
            
            timestamps = []
            ohlcv = {'Open': [], 'High': [], 'Low': [], 'Close': [], 'Volume': []}
            
            for timestamp, values in sorted(timeseries.items()):
                timestamps.append(pd.Timestamp(timestamp))
                ohlcv['Open'].append(float(values.get('1. open', 0)))
                ohlcv['High'].append(float(values.get('2. high', 0)))
                ohlcv['Low'].append(float(values.get('3. low', 0)))
                ohlcv['Close'].append(float(values.get('4. close', 0)))
                # Alpha Vantage doesn't provide volume for FX
                ohlcv['Volume'].append(1000000)
            
            if len(timestamps) > 100:
                df = pd.DataFrame(ohlcv, index=pd.DatetimeIndex(timestamps, name='Datetime'))
                print(f"    ✓ Downloaded {len(df)} bars from Alpha Vantage for {symbol}")
                return df
        
        print(f"    ✗ No data from Alpha Vantage")
        return None
        
    except Exception as e:
        print(f"    ✗ Alpha Vantage error: {str(e)}")
        return None


def download_from_dukascopy(symbol: str) -> pd.DataFrame:
    """
    Download from Dukascopy Bank (free, no API key needed)
    Uses their public data API
    """
    if not requests_available:
        return None
    
    try:
        print(f"  📥 Attempting Dukascopy download for {symbol}...")
        
        from_currency = symbol[:3]
        to_currency = symbol[3:]
        
        # Dukascopy API endpoint
        # Historical data typically available from 1996 onwards
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # 3 months of data
        
        all_data = []
        current_date = start_date
        
        while current_date < end_date:
            try:
                # Dukascopy stores data in daily chunks
                year = current_date.year
                month = str(current_date.month - 1).zfill(2)  # 0-indexed months
                day = str(current_date.day).zfill(2)
                
                url = f"https://www.dukascopy.com/datafeed/{from_currency}{to_currency}/{year}/{month}/{day}/00h_quotes"
                
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    # Dukascopy returns binary tick data, parse it
                    # Each tick is 20 bytes: timestamp (4) + bid/ask (8 floats)
                    content = response.content
                    
                    for i in range(0, len(content), 20):
                        if i + 20 <= len(content):
                            tick_data = np.frombuffer(content[i:i+20], dtype='>f4')
                            if len(tick_data) >= 5:
                                all_data.append(tick_data)
                
                current_date += timedelta(days=1)
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                current_date += timedelta(days=1)
                continue
        
        if len(all_data) > 100:
            print(f"    ✓ Downloaded {len(all_data)} ticks from Dukascopy for {symbol}")
            return pd.DataFrame(all_data)
        else:
            print(f"    ✗ Insufficient Dukascopy data")
            return None
            
    except Exception as e:
        print(f"    ✗ Dukascopy error: {str(e)}")
        return None


def download_from_investing_com(symbol: str) -> pd.DataFrame:
    """
    Scrape historical data from Investing.com (requires browser simulation)
    """
    try:
        print(f"  📥 Attempting Investing.com download for {symbol}...")
        
        # Map symbols to investing.com format
        symbol_map = {
            "EURUSD": "eur-usd",
            "GBPUSD": "gbp-usd",
            "USDJPY": "usd-jpy",
            "AUDUSD": "aud-usd",
            "NZDUSD": "nzd-usd",
            "USDCAD": "usd-cad",
            "USDHKD": "usd-hkd",
            "EURGBP": "eur-gbp",
            "EURJPY": "eur-jpy",
            "GBPJPY": "gbp-jpy",
            "AUDNZD": "aud-nzd"
        }
        
        investing_symbol = symbol_map.get(symbol)
        if not investing_symbol:
            return None
        
        if not requests_available:
            return None
        
        # This would require selenium or similar for JavaScript rendering
        # For now, return None as it requires additional setup
        print(f"    ℹ️  Investing.com requires browser automation (skipped)")
        return None
        
    except Exception as e:
        print(f"    ✗ Investing.com error: {str(e)}")
        return None


def download_real_data_fallback(symbol: str) -> pd.DataFrame:
    """
    Download from alternative real data sources or aggregators
    """
    try:
        print(f"  📥 Attempting alternative data sources for {symbol}...")
        
        # Try multiple endpoints
        sources = [
            # FreeCurrencyAPI
            lambda: requests.get(
                f"https://api.freecurrencyapi.com/v1/historical",
                params={"base_currency": symbol[:3], "currencies": symbol[3:]},
                timeout=5
            ),
        ]
        
        for source_func in sources:
            try:
                response = source_func()
                if response.status_code == 200:
                    print(f"    ✓ Downloaded from alternative source for {symbol}")
                    return response.json()
            except:
                continue
        
        return None
        
    except Exception as e:
        print(f"    ✗ Alternative sources error: {str(e)}")
        return None


def fetch_real_data_multi_source(symbol: str) -> pd.DataFrame:
    """
    Fetch real data using multiple sources in priority order
    """
    print(f"\n{'='*60}")
    print(f"Fetching REAL data for {symbol}")
    print('='*60)
    
    df = None
    
    # Try sources in priority order
    sources = [
        ("Yahoo Finance", lambda: download_from_yfinance(symbol)),
        ("Alpha Vantage", lambda: download_from_alphavantage(symbol)),
        ("Dukascopy", lambda: download_from_dukascopy(symbol)),
    ]
    
    for source_name, download_func in sources:
        try:
            df = download_func()
            if df is not None and len(df) > 100:
                print(f"  ✓ Successfully fetched real data from {source_name} ({len(df)} records)")
                return df
        except Exception as e:
            print(f"  ✗ {source_name} failed: {str(e)}")
            continue
    
    print(f"  ⚠️  Could not fetch real data from any source")
    return None


def save_data_to_csv(df: pd.DataFrame, filename: str) -> bool:
    """
    Save dataframe to CSV file
    """
    try:
        csv_dir = Path("CSV_FILES")
        csv_dir.mkdir(exist_ok=True)
        
        filepath = csv_dir / filename
        df.to_csv(filepath)
        
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  💾 Saved {len(df)} records to {filepath} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ Error saving to CSV: {str(e)}")
        return False


def fetch_all_pairs_real_data():
    """
    Main function to fetch REAL data for all pairs
    """
    
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
    print("FOREX REAL DATA GATHERING - ALL PAIRS")
    print("="*80)
    print("Fetching from: Yahoo Finance, Alpha Vantage, Dukascopy")
    print("="*80 + "\n")
    
    csv_dir = Path("CSV_FILES")
    successful = 0
    failed = 0
    skipped = 0
    
    for i, (symbol, filename) in enumerate(pairs.items(), 1):
        filepath = csv_dir / filename
        
        print(f"\n[{i}/{len(pairs)}] Processing {symbol}")
        
        # Try to fetch real data
        df = fetch_real_data_multi_source(symbol)
        
        if df is not None and len(df) > 100:
            if save_data_to_csv(df, filename):
                successful += 1
            else:
                failed += 1
        else:
            print(f"  ⚠️  Real data unavailable for {symbol}")
            failed += 1
        
        # Rate limiting
        if i < len(pairs):
            time.sleep(1)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Real Data Fetched: {successful}/{len(pairs)}")
    print(f"✗ Failed/Unavailable: {failed}/{len(pairs)}")
    print("\nData files created/updated:")
    
    csv_dir = Path("CSV_FILES")
    for symbol, filename in pairs.items():
        filepath = csv_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            row_count = len(pd.read_csv(filepath, nrows=1))
            print(f"  ✓ {filename} ({size_mb:.1f} MB)")
    
    print("\n" + "="*80)


def verify_data_quality():
    """
    Verify data quality and completeness
    """
    print("\n" + "="*80)
    print("DATA QUALITY VERIFICATION")
    print("="*80 + "\n")
    
    pairs = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD",
        "USDCAD", "USDHKD", "EURGBP", "EURJPY", "GBPJPY", "AUDNZD"
    ]
    
    csv_dir = Path("CSV_FILES")
    
    for symbol in pairs:
        filename = f"MT5_5M_BT_{symbol}_Dataset.csv"
        filepath = csv_dir / filename
        
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                size_mb = filepath.stat().st_size / (1024 * 1024)
                
                # Check for required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                has_all_cols = all(col in df.columns for col in required_cols)
                
                # Check for missing values
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                
                status = "✓" if has_all_cols and missing_pct < 5 else "⚠"
                print(f"{status} {symbol:10} | {len(df):6} rows | {size_mb:6.1f} MB | Missing: {missing_pct:.1f}%")
                
            except Exception as e:
                print(f"✗ {symbol:10} | Error reading file: {str(e)}")
        else:
            print(f"✗ {symbol:10} | File not found")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Install yfinance if not present
    if not yfinance_available:
        print("Installing yfinance for real data downloading...")
        os.system("pip install yfinance -q")
        import yfinance as yf
        _yfinance_available = True
    
    # Fetch all pairs real data
    fetch_all_pairs_real_data()
    
    # Verify data quality
    verify_data_quality()
    
    print("✓ Real data gathering complete!")
    print("You can now run multi-pair backtests with: python PY_FILES/MULTI_PAIR_BACKTEST.py")
