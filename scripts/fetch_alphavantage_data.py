"""
Fetch Real SPY Minute Data from Alpha Vantage
==============================================

Uses Alpha Vantage API to download actual historical minute-level data.
Free tier: 25 API calls per day
Each call can get 1 month of intraday data (extended history requires premium)
"""

import pandas as pd
import requests
import time
from pathlib import Path
from datetime import datetime

API_KEY = "LVAOHLHO9J7WHI33"
BASE_URL = "https://www.alphavantage.co/query"

OUTPUT_DIR = Path(__file__).parent / "results" / "real_historical_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_intraday_data(symbol="SPY", interval="1min", month=None, outputsize="full"):
    """
    Fetch intraday data from Alpha Vantage.
    
    Parameters:
    - symbol: Stock ticker
    - interval: 1min, 5min, 15min, 30min, 60min
    - month: 'YYYY-MM' format for historical month (premium feature)
    - outputsize: 'compact' (latest 100 points) or 'full' (full history available)
    """
    
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "apikey": API_KEY,
        "outputsize": outputsize,
        "datatype": "json"
    }
    
    if month:
        params["month"] = month
        print(f"Fetching {symbol} {interval} data for {month}...")
    else:
        print(f"Fetching {symbol} {interval} data (recent)...")
    
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        # Check for errors
        if "Error Message" in data:
            print(f"  ✗ API Error: {data['Error Message']}")
            return None
        
        if "Note" in data:
            print(f"  ✗ Rate Limit: {data['Note']}")
            return None
        
        if "Information" in data:
            print(f"  ✗ Info: {data['Information']}")
            return None
        
        # Extract time series data
        time_series_key = f"Time Series ({interval})"
        if time_series_key not in data:
            print(f"  ✗ Unexpected response format")
            print(f"  Keys: {list(data.keys())}")
            return None
        
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Rename columns
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        print(f"  ✓ Retrieved {len(df):,} bars")
        print(f"    Date range: {df.index[0]} to {df.index[-1]}")
        print(f"    Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return None


def fetch_daily_data(symbol="SPY", outputsize="full"):
    """Fetch daily data (more reliable, longer history)."""
    
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "apikey": API_KEY,
        "outputsize": outputsize,
        "datatype": "json"
    }
    
    print(f"Fetching {symbol} daily data...")
    
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if "Error Message" in data:
            print(f"  ✗ API Error: {data['Error Message']}")
            return None
        
        if "Note" in data:
            print(f"  ✗ Rate Limit: {data['Note']}")
            return None
        
        time_series = data.get("Time Series (Daily)", {})
        
        if not time_series:
            print(f"  ✗ No data returned")
            return None
        
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Rename columns
        df.columns = ['Open', 'High', 'Low', 'Close', 'Adjusted_Close', 'Volume', 'Dividend', 'Split']
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        print(f"  ✓ Retrieved {len(df):,} daily bars")
        print(f"    Date range: {df.index[0]} to {df.index[-1]}")
        print(f"    Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return None


def main():
    """Fetch real historical data from Alpha Vantage."""
    
    print("\n" + "="*80)
    print("ALPHA VANTAGE DATA FETCHER - SPY")
    print("="*80)
    print(f"API Key: {API_KEY[:4]}...{API_KEY[-4:]}")
    print(f"Free tier: 25 calls/day, 5 calls/minute")
    print("="*80 + "\n")
    
    # 1. Fetch recent minute data (last ~30 days)
    print("1. RECENT MINUTE DATA (1min interval)")
    print("-" * 80)
    minute_df = fetch_intraday_data(symbol="SPY", interval="1min", outputsize="full")
    
    if minute_df is not None:
        output_file = OUTPUT_DIR / "SPY_minute_recent_alphavantage.csv"
        minute_df.to_csv(output_file)
        print(f"  Saved to: {output_file}\n")
    
    time.sleep(12)  # Rate limit: 5 calls/minute = wait 12 seconds
    
    # 2. Fetch 5-minute data (longer history usually available)
    print("2. RECENT 5-MINUTE DATA")
    print("-" * 80)
    min5_df = fetch_intraday_data(symbol="SPY", interval="5min", outputsize="full")
    
    if min5_df is not None:
        output_file = OUTPUT_DIR / "SPY_5min_recent_alphavantage.csv"
        min5_df.to_csv(output_file)
        print(f"  Saved to: {output_file}\n")
    
    time.sleep(12)  # Rate limit
    
    # 3. Fetch daily data (20+ years available)
    print("3. DAILY DATA (Full history)")
    print("-" * 80)
    daily_df = fetch_daily_data(symbol="SPY", outputsize="full")
    
    if daily_df is not None:
        output_file = OUTPUT_DIR / "SPY_daily_full_alphavantage.csv"
        daily_df.to_csv(output_file)
        print(f"  Saved to: {output_file}\n")
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    if minute_df is not None:
        print(f"✓ 1-minute data: {len(minute_df):,} bars ({minute_df.index[0].date()} to {minute_df.index[-1].date()})")
    else:
        print("✗ 1-minute data: Failed")
    
    if min5_df is not None:
        print(f"✓ 5-minute data: {len(min5_df):,} bars ({min5_df.index[0].date()} to {min5_df.index[-1].date()})")
    else:
        print("✗ 5-minute data: Failed")
    
    if daily_df is not None:
        print(f"✓ Daily data: {len(daily_df):,} bars ({daily_df.index[0].date()} to {daily_df.index[-1].date()})")
    else:
        print("✗ Daily data: Failed")
    
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print("\n" + "="*80)
    print("NOTES:")
    print("- Free tier minute data is limited to recent ~30 days")
    print("- For Oct-Dec 2024 minute data, you need premium API access")
    print("- Daily data gives you 20+ years of history (useful for training)")
    print("- Consider using 5-minute or 15-minute bars as compromise")
    print("="*80)


if __name__ == "__main__":
    main()
