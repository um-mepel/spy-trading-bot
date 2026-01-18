"""
Stock data fetcher for machine learning model training and testing.
Fetches daily OHLCV data and technical indicators from yfinance or generates demo data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')


def calculate_technical_indicators(df, is_training=False, training_tail=None):
    """
    Calculate technical indicators on a SINGLE dataset (train OR test, never mixed).
    
    IMPORTANT: This function operates on one dataset at a time.
    For proper out-of-sample testing:
    - Calculate indicators on TRAINING data only
    - For testing data, we need to "warm up" using training data's tail
    
    Args:
        df: DataFrame with OHLCV data for ONE period (training OR testing)
        is_training: If True, this is training data. If False, this is test data.
        training_tail: If test data, provide last N rows of training data for warming up
                      This allows rolling windows to initialize without lookahead
    
    Returns:
        DataFrame with added technical indicators (properly calculated without lookahead)
    """
    df = df.copy()
    
    # For test data with warm-up, concatenate training tail + test data
    # Then calculate indicators, then extract only test data portion
    if not is_training and training_tail is not None:
        # Concatenate training tail + test data
        # This allows rolling windows to initialize properly
        combined = pd.concat([training_tail, df], ignore_index=True)
        
        # Calculate all indicators on combined data
        combined = _add_indicators_to_df(combined)
        
        # Extract ONLY test data portion (skip the warm-up rows)
        df = combined.iloc[len(training_tail):].reset_index(drop=True)
    else:
        # Training data (or test data without warm-up)
        df = _add_indicators_to_df(df)
    
    return df


def _add_indicators_to_df(df):
    """
    Internal function to calculate technical indicators on a dataframe.
    This should only be called on properly isolated data (no mixing).
    """
    df = df.copy()
    
    # Simple Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # Momentum
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    
    # Rate of Change
    df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # Volatility (Standard Deviation)
    df['Volatility_20'] = df['Close'].rolling(window=20).std()
    
    # Average True Range (ATR)
    df['High_Low'] = df['High'] - df['Low']
    df['High_Close'] = abs(df['High'] - df['Close'].shift())
    df['Low_Close'] = abs(df['Low'] - df['Close'].shift())
    df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    df = df.drop(['High_Low', 'High_Close', 'Low_Close', 'TR'], axis=1)
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle_20'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper_20'] = df['BB_Middle_20'] + (bb_std * 2)
    df['BB_Lower_20'] = df['BB_Middle_20'] - (bb_std * 2)
    
    # Bollinger Band Width
    df['BB_Width'] = df['BB_Upper_20'] - df['BB_Lower_20']
    
    # Price Position in Bollinger Bands (0 to 1)
    df['BB_Position'] = (df['Close'] - df['BB_Lower_20']) / (df['BB_Upper_20'] - df['BB_Lower_20'])
    df['BB_Position'] = df['BB_Position'].clip(0, 1)  # Clip to [0, 1]
    
    # Volume indicators
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    
    # Log Returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Daily Return Percentage
    df['Daily_Return_Pct'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
    
    # High-Low Range Percentage
    df['HL_Range_Pct'] = ((df['High'] - df['Low']) / df['Close']) * 100
    
    # Close-Open Range Percentage
    df['CO_Range_Pct'] = ((df['Close'] - df['Open']) / df['Open']) * 100
    
    # Distance from moving averages
    df['Price_SMA20_Distance'] = ((df['Close'] - df['SMA_20']) / df['SMA_20']) * 100
    df['Price_SMA50_Distance'] = ((df['Close'] - df['SMA_50']) / df['SMA_50']) * 100
    
    return df


def generate_demo_data(start_date, end_date, base_price=400):
    """
    Generate realistic demo stock data for testing purposes.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        base_price: Starting stock price
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Generating demo data from {start_date} to {end_date}...")
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate business dates
    dates = pd.bdate_range(start=start, end=end)
    
    np.random.seed(42)
    price = base_price
    prices = [price]
    
    # Generate realistic price movement
    for _ in range(len(dates) - 1):
        daily_return = np.random.normal(0.0005, 0.015)  # Mean 0.05%, std 1.5%
        price = price * (1 + daily_return)
        prices.append(price)
    
    # Create OHLCV data
    data = []
    for date, close in zip(dates, prices):
        open_price = close + np.random.normal(0, close * 0.003)
        high = max(open_price, close) + np.random.uniform(0, close * 0.01)
        low = min(open_price, close) - np.random.uniform(0, close * 0.01)
        volume = int(np.random.uniform(50_000_000, 100_000_000))
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume,
            'Adj Close': close
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    return df


def fetch_stock_data(ticker, start_date, end_date, max_retries=5, use_demo=False):
    """
    Fetch stock data from yfinance with retry logic and delays.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_retries: Number of retries for failed requests
        use_demo: If True, generates demo data instead of fetching live data
        
    Returns:
        DataFrame with OHLCV data
    """
    if use_demo:
        return generate_demo_data(start_date, end_date)
    
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    
    for attempt in range(max_retries):
        try:
            # Add delay before request to avoid rate limiting
            if attempt > 0:
                delay = min(2 ** attempt, 30)  # Exponential backoff, max 30 seconds
                print(f"  Waiting {delay} seconds before retry...")
                time.sleep(delay)
            
            print(f"  Attempt {attempt + 1}/{max_retries}...")
            
            # Convert dates to strings and use yfinance download
            df = yf.download(
                ticker, 
                start=str(start_date), 
                end=str(end_date), 
                progress=False,
                timeout=30,
                repair=True  # Repair data for missing timestamps
            )
            
            if df is not None and len(df) > 0:
                # Check if we got a multi-ticker result
                if isinstance(df.columns, pd.MultiIndex):
                    # Multi-ticker result, extract single ticker
                    if ticker in df.columns.get_level_values(0):
                        df = df[ticker]
                
                # Ensure we have the required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in required_cols):
                    print(f"  Successfully fetched {len(df)} rows")
                    return df
                else:
                    print(f"  Missing required columns. Got: {list(df.columns)}")
            else:
                print(f"  No data returned")
            
        except Exception as e:
            error_msg = str(e)
            print(f"  Error on attempt {attempt + 1}: {error_msg[:100]}")
    
    print(f"  Failed to fetch data for {ticker} after {max_retries} attempts")
    print(f"  Using demo data instead...")
    return generate_demo_data(start_date, end_date)


def prepare_data(ticker, training_start='2022-01-01', training_end='2024-12-31', 
                 testing_start='2025-01-01', testing_end='2025-12-31', use_demo=False):
    """
    Fetch and prepare stock data for training and testing.
    
    Args:
        ticker: Stock ticker symbol
        training_start: Training data start date
        training_end: Training data end date
        testing_start: Testing data start date
        testing_end: Testing data end date
        use_demo: If True, uses demo data instead of live yfinance data
        
    Returns:
        Tuple of (training_df, testing_df)
    """
    # Fetch training data
    training_data = fetch_stock_data(ticker, training_start, training_end, use_demo=use_demo)
    print(f"Training data fetched: {len(training_data)} trading days")
    
    # Fetch testing data
    testing_data = fetch_stock_data(ticker, testing_start, testing_end, use_demo=use_demo)
    print(f"Testing data fetched: {len(testing_data)} trading days")
    
    # Filter testing data to only include 2025 data
    if 'Date' not in testing_data.columns:
        testing_data.reset_index(inplace=True)
    testing_data['Date'] = pd.to_datetime(testing_data['Date'])
    testing_data = testing_data[(testing_data['Date'].dt.year == 2025)]
    print(f"Filtered to 2025 data: {len(testing_data)} trading days")
    
    # Handle MultiIndex columns from yfinance (when multiple tickers are in result)
    if isinstance(training_data.columns, pd.MultiIndex):
        # Flatten MultiIndex: take the first level (price type) since we have single ticker
        training_data.columns = training_data.columns.get_level_values(0)
    if isinstance(testing_data.columns, pd.MultiIndex):
        testing_data.columns = testing_data.columns.get_level_values(0)
    
    # Remove unwanted columns that yfinance might add
    cols_to_drop = [col for col in training_data.columns if col in ['Repaired?', 'Capital Gains', 'Stock Splits']]
    if cols_to_drop:
        training_data = training_data.drop(columns=cols_to_drop)
        testing_data = testing_data.drop(columns=cols_to_drop)
    
    # Calculate indicators
    print("Calculating technical indicators...")
    training_data = calculate_technical_indicators(training_data)
    testing_data = calculate_technical_indicators(testing_data)
    
    # Reset index to make Date a column
    training_data.reset_index(inplace=True)
    testing_data.reset_index(inplace=True)
    
    # Rename columns to be consistent
    training_data.columns.name = None
    testing_data.columns.name = None
    
    return training_data, testing_data


def save_to_csv(df, filename):
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        filename: Output filename
    """
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")


def main():
    """
    Main function to fetch and save stock data.
    """
    # Configuration
    ticker = "SPY"  # Change this to your desired ticker
    output_dir = "./"
    use_demo_data = False  # Set to False to use live yfinance data
    
    print(f"{'='*60}")
    print(f"Stock Data Fetcher")
    print(f"{'='*60}")
    print(f"Ticker: {ticker}")
    print(f"Output directory: {output_dir}")
    print(f"Using demo data: {use_demo_data}")
    print(f"{'='*60}\n")
    
    # Fetch and prepare data
    training_data, testing_data = prepare_data(
        ticker,
        training_start='2022-01-01',
        training_end='2024-12-31',
        testing_start='2025-01-01',
        testing_end='2025-12-31',
        use_demo=use_demo_data
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    training_file = os.path.join(output_dir, f"{ticker}_training_2022_2024.csv")
    testing_file = os.path.join(output_dir, f"{ticker}_testing_2025.csv")
    
    save_to_csv(training_data, training_file)
    save_to_csv(testing_data, testing_file)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("Data Summary")
    print(f"{'='*60}")
    print(f"\nTraining Data (2022-2024):")
    print(f"  Shape: {training_data.shape}")
    print(f"  Date range: {training_data['Date'].min()} to {training_data['Date'].max()}")
    
    print(f"\nTesting Data (2025):")
    print(f"  Shape: {testing_data.shape}")
    print(f"  Date range: {testing_data['Date'].min()} to {testing_data['Date'].max()}")
    
    print(f"\nFeatures included:")
    print(f"  Price data: Open, High, Low, Close, Volume, Adj Close")
    print(f"  Moving averages: SMA (5, 10, 20, 50, 200), EMA (12, 26)")
    print(f"  Momentum: MACD, Signal Line, MACD Histogram, Momentum, ROC")
    print(f"  Volatility: Volatility (20-day), ATR (14-day)")
    print(f"  Oscillators: RSI (14-day)")
    print(f"  Bands: Bollinger Bands, BB Width, BB Position")
    print(f"  Volume: Volume SMA, Volume Ratio")
    print(f"  Returns: Log Return, Daily Return %, HL Range %, CO Range %")
    print(f"  Distance: Price-SMA20 Distance, Price-SMA50 Distance")
    print(f"{'='*60}\n")
    
    # Print sample data
    print("Sample Training Data (first 5 rows):")
    print(training_data.head())
    print(f"\nAll columns: {list(training_data.columns)}")


if __name__ == "__main__":
    main()
