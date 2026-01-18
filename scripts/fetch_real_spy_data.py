"""
Fetch Real Historical SPY Minute Data
=====================================

Downloads actual historical minute-level data for SPY using yfinance.
For periods further back than 30 days, yfinance limits minute data availability.
Will try multiple approaches to get the most data possible.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import time

OUTPUT_DIR = Path(__file__).parent / "results" / "real_historical_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_recent_minute_data(ticker="SPY", days_back=30):
    """
    Fetch the most recent minute data (yfinance limits to ~30 days for 1m interval).
    """
    print(f"\n{'='*80}")
    print(f"FETCHING RECENT MINUTE DATA: Last {days_back} days")
    print(f"{'='*80}\n")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"Ticker: {ticker}")
    print(f"Start: {start_date.strftime('%Y-%m-%d')}")
    print(f"End: {end_date.strftime('%Y-%m-%d')}")
    print(f"Interval: 1 minute\n")
    
    try:
        print("Downloading from yfinance...")
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval='1m',
            progress=True,
            prepost=False,
            auto_adjust=True
        )
        
        if df is not None and len(df) > 0:
            print(f"\n✓ Successfully downloaded {len(df):,} minute bars")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            print(f"  Columns: {list(df.columns)}")
            
            # Save to CSV
            output_file = OUTPUT_DIR / f"{ticker}_minute_recent_{days_back}days.csv"
            df.to_csv(output_file)
            print(f"  Saved to: {output_file}")
            
            # Show sample
            print(f"\nFirst 5 rows:")
            print(df.head())
            
            # Statistics
            print(f"\nData Statistics:")
            print(f"  Trading days: {len(df.index.date.unique())}")
            print(f"  Avg bars per day: {len(df) / len(df.index.date.unique()):.0f}")
            print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
            print(f"  Avg volume: {df['Volume'].mean():,.0f}")
            
            return df
        else:
            print("✗ No data returned")
            return None
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def fetch_longer_history(ticker="SPY", months_back=6):
    """
    Fetch longer history using hourly data (more reliable for older dates).
    """
    print(f"\n{'='*80}")
    print(f"FETCHING LONGER HISTORY: Last {months_back} months (hourly data)")
    print(f"{'='*80}\n")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back*30)
    
    print(f"Ticker: {ticker}")
    print(f"Start: {start_date.strftime('%Y-%m-%d')}")
    print(f"End: {end_date.strftime('%Y-%m-%d')}")
    print(f"Interval: 1 hour (more reliable for longer periods)\n")
    
    try:
        print("Downloading from yfinance...")
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval='1h',
            progress=True,
            prepost=False,
            auto_adjust=True
        )
        
        if df is not None and len(df) > 0:
            print(f"\n✓ Successfully downloaded {len(df):,} hourly bars")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            
            # Save to CSV
            output_file = OUTPUT_DIR / f"{ticker}_hourly_{months_back}months.csv"
            df.to_csv(output_file)
            print(f"  Saved to: {output_file}")
            
            print(f"\nFirst 5 rows:")
            print(df.head())
            
            return df
        else:
            print("✗ No data returned")
            return None
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def fetch_daily_history(ticker="SPY", years_back=2):
    """
    Fetch daily data for even longer history (always reliable).
    """
    print(f"\n{'='*80}")
    print(f"FETCHING DAILY HISTORY: Last {years_back} years")
    print(f"{'='*80}\n")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back*365)
    
    print(f"Ticker: {ticker}")
    print(f"Start: {start_date.strftime('%Y-%m-%d')}")
    print(f"End: {end_date.strftime('%Y-%m-%d')}")
    print(f"Interval: 1 day\n")
    
    try:
        print("Downloading from yfinance...")
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval='1d',
            progress=True,
            prepost=False,
            auto_adjust=True
        )
        
        if df is not None and len(df) > 0:
            print(f"\n✓ Successfully downloaded {len(df):,} daily bars")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            
            # Save to CSV
            output_file = OUTPUT_DIR / f"{ticker}_daily_{years_back}years.csv"
            df.to_csv(output_file)
            print(f"  Saved to: {output_file}")
            
            print(f"\nFirst 5 rows:")
            print(df.head())
            
            return df
        else:
            print("✗ No data returned")
            return None
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def main():
    """Fetch all available real historical data."""
    
    print("\n" + "="*80)
    print("REAL HISTORICAL DATA FETCHER - SPY")
    print("="*80)
    
    # 1. Try to get recent minute data (last 7 days is most reliable for 1m)
    minute_df = fetch_recent_minute_data(ticker="SPY", days_back=7)
    
    time.sleep(1)  # Rate limit
    
    # 2. Get hourly data for longer training periods
    hourly_df = fetch_longer_history(ticker="SPY", months_back=3)
    
    time.sleep(1)  # Rate limit
    
    # 3. Get daily data for very long history
    daily_df = fetch_daily_history(ticker="SPY", years_back=2)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if minute_df is not None:
        print(f"✓ Minute data: {len(minute_df):,} bars saved")
    else:
        print("✗ Minute data: Failed")
    
    if hourly_df is not None:
        print(f"✓ Hourly data: {len(hourly_df):,} bars saved")
    else:
        print("✗ Hourly data: Failed")
    
    if daily_df is not None:
        print(f"✓ Daily data: {len(daily_df):,} bars saved")
    else:
        print("✗ Daily data: Failed")
    
    print(f"\nAll files saved to: {OUTPUT_DIR}")
    print("\n" + "="*80)
    print("\nNOTE: yfinance limits minute data to ~30 days history.")
    print("For minute-level training, you may need:")
    print("  1. A paid data provider (Polygon, Alpha Vantage, IEX)")
    print("  2. To use hourly data instead")
    print("  3. To collect minute data going forward over time")
    print("="*80)


if __name__ == "__main__":
    main()
