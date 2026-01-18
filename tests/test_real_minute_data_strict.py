"""
STRICT Real Minute Data Test - NO SYNTHETIC FALLBACK
======================================================

This test REQUIRES real minute-level data. If API fails, test fails.
This ensures we're testing the model on actual market data, not synthetic patterns.

Data Sources (in order of preference):
1. Polygon.io (requires API key)
2. Alpaca Markets (requires account)
3. Local CSV file (pre-downloaded data)

Usage:
    # With Polygon.io API key:
    POLYGON_API_KEY=your_key python tests/test_real_minute_data_strict.py
    
    # With local CSV file:
    python tests/test_real_minute_data_strict.py --csv path/to/minute_data.csv
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lightgbm_model import train_lightgbm

# ============================================================================
# CONFIGURATION
# ============================================================================

TICKER = "SPY"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "real_minute_strict"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Alpaca API Credentials (Paper Trading - no funds attached)
ALPACA_API_KEY = "PKQCAK3OCWGUZ5JHL6QFBXNSLV"
ALPACA_SECRET_KEY = "51B23zrTEMd9sBbXqWyqaewmmum5XK6oUsSn4RThcTRU"

# Default test periods
# Maximum historical data - Alpaca has minute data back to ~2016
DEFAULT_TRAINING_START = '2020-01-01'  # 4 years of training data
DEFAULT_TRAINING_END = '2024-06-30'
DEFAULT_TESTING_START = '2024-07-01'   # 6 months of testing data
DEFAULT_TESTING_END = '2024-12-31'


# ============================================================================
# DATA FETCHING - REAL DATA ONLY (NO SYNTHETIC FALLBACK)
# ============================================================================

def fetch_from_polygon(ticker, start_date, end_date, api_key):
    """
    Fetch minute data from Polygon.io API.
    Free tier: 5 API calls/minute, 2 years history
    Paid ($29/mo): Unlimited calls, full history
    """
    try:
        from polygon import RESTClient
    except ImportError:
        print("ERROR: polygon-api-client not installed. Run: pip install polygon-api-client")
        return None
    
    print(f"Fetching from Polygon.io: {ticker} {start_date} to {end_date}")
    
    try:
        client = RESTClient(api_key)
        
        all_bars = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        days_fetched = 0
        while current_date <= end:
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
            
            try:
                aggs = client.get_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan="minute",
                    from_=current_date.strftime('%Y-%m-%d'),
                    to=(current_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                    limit=50000
                )
                
                day_bars = 0
                for bar in aggs:
                    all_bars.append({
                        'Datetime': pd.to_datetime(bar.timestamp, unit='ms'),
                        'Open': bar.open,
                        'High': bar.high,
                        'Low': bar.low,
                        'Close': bar.close,
                        'Volume': bar.volume
                    })
                    day_bars += 1
                
                if day_bars > 0:
                    days_fetched += 1
                    if days_fetched % 10 == 0:
                        print(f"  Fetched {days_fetched} trading days, {len(all_bars):,} bars...")
                        
            except Exception as e:
                print(f"  Warning: Error on {current_date.date()}: {e}")
            
            current_date += timedelta(days=1)
        
        if len(all_bars) == 0:
            print("ERROR: No data retrieved from Polygon")
            return None
        
        df = pd.DataFrame(all_bars)
        df = df.sort_values('Datetime').reset_index(drop=True)
        
        print(f"✓ Retrieved {len(df):,} minute bars from Polygon")
        print(f"  Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
        
        return df
        
    except Exception as e:
        print(f"ERROR: Polygon API failed: {e}")
        return None


def fetch_from_alpaca(ticker, start_date, end_date, api_key, secret_key):
    """
    Fetch minute data from Alpaca Markets.
    Requires free Alpaca account.
    """
    try:
        from alpaca.data import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError:
        print("ERROR: alpaca-py not installed. Run: pip install alpaca-py")
        return None
    
    print(f"Fetching from Alpaca: {ticker} {start_date} to {end_date}")
    
    try:
        client = StockHistoricalDataClient(api_key, secret_key)
        
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute,
            start=datetime.strptime(start_date, '%Y-%m-%d'),
            end=datetime.strptime(end_date, '%Y-%m-%d')
        )
        
        bars = client.get_stock_bars(request)
        df = bars.df.reset_index()
        
        # Standardize columns
        df = df.rename(columns={
            'timestamp': 'Datetime',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        if 'symbol' in df.columns:
            df = df[df['symbol'] == ticker]
        
        df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.sort_values('Datetime').reset_index(drop=True)
        
        print(f"✓ Retrieved {len(df):,} minute bars from Alpaca")
        print(f"  Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
        
        return df
        
    except Exception as e:
        print(f"ERROR: Alpaca API failed: {e}")
        return None


def load_from_csv(csv_path):
    """
    Load minute data from a local CSV file.
    Expected columns: Datetime (or Date/Time), Open, High, Low, Close, Volume
    """
    print(f"Loading from CSV: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        # Try to identify datetime column
        datetime_cols = ['Datetime', 'datetime', 'Date', 'date', 'Time', 'time', 'timestamp', 'Timestamp']
        dt_col = None
        for col in datetime_cols:
            if col in df.columns:
                dt_col = col
                break
        
        if dt_col is None:
            # Check if first column is datetime-like
            if df.columns[0] not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                dt_col = df.columns[0]
        
        if dt_col:
            df['Datetime'] = pd.to_datetime(df[dt_col])
            if dt_col != 'Datetime':
                df = df.drop(columns=[dt_col])
        else:
            print("ERROR: Could not identify datetime column in CSV")
            return None
        
        # Standardize column names
        col_map = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 
            'close': 'Close', 'volume': 'Volume'
        }
        df = df.rename(columns=col_map)
        
        required_cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                print(f"ERROR: Missing required column: {col}")
                return None
        
        df = df[required_cols].sort_values('Datetime').reset_index(drop=True)
        
        print(f"✓ Loaded {len(df):,} minute bars from CSV")
        print(f"  Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
        
        return df
        
    except Exception as e:
        print(f"ERROR: Failed to load CSV: {e}")
        return None


def fetch_real_minute_data(ticker, start_date, end_date, csv_path=None):
    """
    Attempt to fetch real minute data from available sources.
    NO SYNTHETIC FALLBACK - fails if no real data available.
    """
    
    # Option 1: Local CSV (highest priority if provided)
    if csv_path:
        df = load_from_csv(csv_path)
        if df is not None:
            # Filter to requested date range
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            mask = (df['Datetime'] >= start_date) & (df['Datetime'] <= end_date + ' 23:59:59')
            df = df[mask].reset_index(drop=True)
            if len(df) > 0:
                return df
            print(f"WARNING: CSV has no data in range {start_date} to {end_date}")
    
    # Option 2: Polygon.io API
    polygon_key = os.environ.get('POLYGON_API_KEY')
    if polygon_key:
        df = fetch_from_polygon(ticker, start_date, end_date, polygon_key)
        if df is not None and len(df) > 0:
            return df
    else:
        print("INFO: POLYGON_API_KEY not set. Set it to use Polygon.io data.")
    
    # Option 3: Alpaca Markets API
    alpaca_key = os.environ.get('ALPACA_API_KEY') or ALPACA_API_KEY
    alpaca_secret = os.environ.get('ALPACA_SECRET_KEY') or ALPACA_SECRET_KEY
    if alpaca_key and alpaca_secret:
        df = fetch_from_alpaca(ticker, start_date, end_date, alpaca_key, alpaca_secret)
        if df is not None and len(df) > 0:
            return df
    else:
        print("INFO: ALPACA_API_KEY/ALPACA_SECRET_KEY not set. Set them to use Alpaca data.")
    
    # NO FALLBACK - fail explicitly
    return None


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def add_technical_indicators(df):
    """Calculate technical indicators for minute-level data."""
    df = df.copy()
    
    print("Calculating technical indicators...")
    
    # Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['SMA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['SMA_60'] = df['Close'].rolling(window=60, min_periods=1).mean()
    
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Momentum
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    
    # Rate of Change
    df['ROC_5'] = (df['Close'] - df['Close'].shift(5)) / (df['Close'].shift(5) + 1e-10)
    df['ROC_10'] = (df['Close'] - df['Close'].shift(10)) / (df['Close'].shift(10) + 1e-10)
    
    # Bollinger Bands
    bb_middle = df['Close'].rolling(window=20, min_periods=1).mean()
    bb_std = df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_Upper'] = bb_middle + (bb_std * 2)
    df['BB_Lower'] = bb_middle - (bb_std * 2)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14, min_periods=1).mean()
    
    # Volatility
    df['Returns'] = df['Close'].pct_change()
    df['Volatility_10'] = df['Returns'].rolling(window=10, min_periods=1).std()
    
    # Price ranges
    df['HL_Range'] = df['High'] - df['Low']
    df['HL_Range_Pct'] = df['HL_Range'] / df['Close']
    df['CO_Range_Pct'] = np.abs(df['Close'] - df['Open']) / df['Close']
    
    print(f"✓ Added {len(df.columns) - 6} technical indicators")
    
    return df


def get_feature_columns():
    """Feature columns for the model."""
    return [
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_60',
        'EMA_5', 'EMA_12', 'EMA_26',
        'MACD', 'Signal_Line', 'MACD_Histogram',
        'RSI_14', 'Momentum_5', 'Momentum_10',
        'ROC_5', 'ROC_10', 'BB_Upper', 'BB_Lower', 'BB_Position',
        'ATR_14', 'Volatility_10',
        'HL_Range', 'HL_Range_Pct', 'CO_Range_Pct'
    ]


# ============================================================================
# MAIN TEST
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test model on REAL minute data only')
    parser.add_argument('--csv', type=str, help='Path to local CSV file with minute data')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--train-start', type=str, default=DEFAULT_TRAINING_START, help='Training start date (YYYY-MM-DD)')
    parser.add_argument('--train-end', type=str, default=DEFAULT_TRAINING_END, help='Training end date (YYYY-MM-DD)')
    parser.add_argument('--test-start', type=str, default=DEFAULT_TESTING_START, help='Testing start date (YYYY-MM-DD)')
    parser.add_argument('--test-end', type=str, default=DEFAULT_TESTING_END, help='Testing end date (YYYY-MM-DD)')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("STRICT REAL MINUTE DATA TEST - NO SYNTHETIC FALLBACK")
    print("="*80)
    print(f"Ticker:   {args.ticker}")
    print(f"Training: {args.train_start} to {args.train_end}")
    print(f"Testing:  {args.test_start} to {args.test_end}")
    print("="*80 + "\n")
    
    # Fetch training data
    print("--- FETCHING TRAINING DATA (REAL ONLY) ---")
    df_train_raw = fetch_real_minute_data(
        args.ticker, args.train_start, args.train_end, args.csv
    )
    
    if df_train_raw is None:
        print("\n" + "="*80)
        print("FATAL ERROR: Could not fetch real training data!")
        print("="*80)
        print("\nOptions to fix this:")
        print("1. Set POLYGON_API_KEY environment variable (get free key at polygon.io)")
        print("2. Set ALPACA_API_KEY + ALPACA_SECRET_KEY (free account at alpaca.markets)")
        print("3. Provide --csv path to local minute data file")
        print("4. Download data from firstratedata.com (~$20 for full SPY history)")
        print("\nThis test REFUSES to use synthetic data.")
        print("="*80 + "\n")
        sys.exit(1)
    
    # Fetch testing data
    print("\n--- FETCHING TESTING DATA (REAL ONLY) ---")
    df_test_raw = fetch_real_minute_data(
        args.ticker, args.test_start, args.test_end, args.csv
    )
    
    if df_test_raw is None:
        print("\n" + "="*80)
        print("FATAL ERROR: Could not fetch real testing data!")
        print("="*80)
        sys.exit(1)
    
    print(f"\n✓ Real training data: {len(df_train_raw):,} minute bars")
    print(f"✓ Real testing data:  {len(df_test_raw):,} minute bars")
    
    # Verify data is real by checking for market microstructure
    # Real data should have gaps (nights, weekends) and irregular patterns
    time_diffs = df_train_raw['Datetime'].diff().dt.total_seconds().dropna()
    has_gaps = (time_diffs > 120).sum() > 0  # More than 2 min gaps
    
    if not has_gaps:
        print("\n⚠️  WARNING: Data appears to have no time gaps!")
        print("   This might indicate synthetic or fabricated data.")
        print("   Real market data should have overnight/weekend gaps.")
    
    # Add indicators
    print("\n--- ADDING TECHNICAL INDICATORS ---")
    df_train = add_technical_indicators(df_train_raw)
    df_test = add_technical_indicators(df_test_raw)
    
    # Prepare data
    features = get_feature_columns()
    
    # Create target: 20-bar forward price change
    df_train['Price_Change'] = df_train['Close'].shift(-20) - df_train['Close']
    df_test['Price_Change'] = df_test['Close'].shift(-20) - df_test['Close']
    
    # Remove last 20 rows
    df_train = df_train.iloc[:-20].copy()
    df_test = df_test.iloc[:-20].copy()
    
    # Cleanup
    df_train = df_train[features + ['Price_Change', 'Close', 'Datetime']].dropna()
    df_test = df_test[features + ['Price_Change', 'Close', 'Datetime']].dropna()
    
    print(f"\nFinal training samples: {len(df_train):,}")
    print(f"Final testing samples:  {len(df_test):,}")
    
    if len(df_train) < 1000:
        print(f"\n⚠️  WARNING: Only {len(df_train)} training samples.")
        print("   For reliable results, aim for 10,000+ samples.")
    
    if len(df_test) < 500:
        print(f"\n⚠️  WARNING: Only {len(df_test)} testing samples.")
        print("   Results may not be statistically significant.")
    
    # Save data
    train_csv = RESULTS_DIR / f"{args.ticker}_real_minute_training.csv"
    test_csv = RESULTS_DIR / f"{args.ticker}_real_minute_testing.csv"
    df_train.to_csv(train_csv, index=False)
    df_test.to_csv(test_csv, index=False)
    print(f"\n✓ Saved training data: {train_csv}")
    print(f"✓ Saved testing data:  {test_csv}")
    
    # Train model
    print("\n--- TRAINING MODEL ---")
    
    # Rename for compatibility
    df_train = df_train.rename(columns={'Datetime': 'Date'})
    df_test = df_test.rename(columns={'Datetime': 'Date'})
    
    try:
        results = train_lightgbm(df_train, df_test, results_dir=str(RESULTS_DIR))
        
        if results and 'results' in results:
            results_df = results['results']
            
            # Calculate metrics
            accuracy = results_df['Direction_Correct'].mean()
            edge_pct = (accuracy - 0.5) * 100
            
            # High confidence analysis
            high_conf_mask = results_df['Confidence'] > 0.7
            high_conf_count = high_conf_mask.sum()
            high_conf_acc = results_df.loc[high_conf_mask, 'Direction_Correct'].mean() if high_conf_count > 0 else 0
            
            print(f"\n{'='*80}")
            print(f"RESULTS - REAL MINUTE DATA TEST")
            print(f"{'='*80}")
            print(f"Data Source:          REAL MARKET DATA")
            print(f"Training samples:     {len(df_train):,}")
            print(f"Testing samples:      {len(df_test):,}")
            print(f"Features:             {len(features)}")
            print(f"\n--- OVERALL PERFORMANCE ---")
            print(f"Accuracy:             {accuracy:.2%}")
            print(f"Edge vs random:       {edge_pct:+.2f}%")
            print(f"Total predictions:    {len(results_df):,}")
            
            if high_conf_count > 0:
                print(f"\n--- HIGH CONFIDENCE (>0.7) ---")
                print(f"High-conf signals:    {high_conf_count:,} ({high_conf_count/len(results_df)*100:.1f}%)")
                print(f"High-conf accuracy:   {high_conf_acc:.2%}")
                print(f"High-conf edge:       {(high_conf_acc - 0.5) * 100:+.2f}%")
            
            # Statistical significance
            from scipy import stats
            n = len(results_df)
            p_hat = accuracy
            se = np.sqrt(p_hat * (1 - p_hat) / n)
            z_score = (p_hat - 0.5) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            print(f"\n--- STATISTICAL SIGNIFICANCE ---")
            print(f"Z-score:              {z_score:.2f}")
            print(f"P-value:              {p_value:.4f}")
            print(f"Significant at 5%?    {'YES ✓' if p_value < 0.05 else 'NO ✗'}")
            print(f"Significant at 1%?    {'YES ✓' if p_value < 0.01 else 'NO ✗'}")
            
            # Verdict
            print(f"\n{'='*80}")
            if edge_pct > 3.0 and p_value < 0.05:
                print(f"✅ SIGNIFICANT EDGE ON REAL DATA: {edge_pct:.2f}%")
                print(f"   This is a real, statistically significant advantage.")
            elif edge_pct > 1.0 and p_value < 0.05:
                print(f"⚠️  MARGINAL EDGE ON REAL DATA: {edge_pct:.2f}%")
                print(f"   Statistically significant but may not cover transaction costs.")
            elif edge_pct > 0:
                print(f"~ MINIMAL/NO EDGE: {edge_pct:.2f}%")
                print(f"   Results are not statistically significant (p={p_value:.3f})")
            else:
                print(f"✗ NO EDGE ON REAL DATA: {edge_pct:.2f}%")
                print(f"   Model performs at or below random on real market data.")
            
            # Compare to synthetic results if available
            if edge_pct < 3.0:
                print(f"\n📊 COMPARISON NOTE:")
                print(f"   If this model showed >5% edge on synthetic data,")
                print(f"   but only {edge_pct:.2f}% on real data, the synthetic")
                print(f"   data had learnable patterns not present in real markets.")
            
            print(f"{'='*80}\n")
            
            # Save summary
            summary = {
                'ticker': args.ticker,
                'data_source': 'REAL_MARKET_DATA',
                'training_period': f"{args.train_start} to {args.train_end}",
                'testing_period': f"{args.test_start} to {args.test_end}",
                'training_samples': int(len(df_train)),
                'testing_samples': int(len(df_test)),
                'accuracy': float(accuracy),
                'edge_percent': float(edge_pct),
                'high_conf_accuracy': float(high_conf_acc) if high_conf_count > 0 else None,
                'z_score': float(z_score),
                'p_value': float(p_value),
                'statistically_significant': bool(p_value < 0.05),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(RESULTS_DIR / 'summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"✓ Results saved to {RESULTS_DIR}")
            
    except Exception as e:
        print(f"ERROR: Model training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
