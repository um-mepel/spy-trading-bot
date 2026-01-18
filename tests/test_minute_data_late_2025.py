#!/usr/bin/env python3
"""
Extended Minute-by-Minute Trading Strategy Test (Late 2025)
Tests the model training and trading strategy on high-frequency minute-level data
for 2 full weeks in late 2025, with real market data from yfinance.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lightgbm_model import main as train_lightgbm_model
from models.ensemble_model import main as train_ensemble_model
from models.signal_generation import main as generate_trading_signals
from models.portfolio_management import main as run_portfolio_backtest
from scripts.fetch_stock_data import calculate_technical_indicators


# Configuration
TICKER = "AAPL"  # Can be changed to any ticker
TRAIN_START = "2025-12-08"  # Monday of week 1
TRAIN_END = "2025-12-12"    # Friday of week 1
TEST_START = "2025-12-15"   # Monday of week 2
TEST_END = "2025-12-19"     # Friday of week 2
MINUTE_INTERVAL = "1m"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "minute_data_late_2025"

# Global variables
TRAINING_DATA = None
TESTING_DATA = None


def fetch_minute_data():
    """
    Fetch minute-by-minute data for 2 weeks in late 2025.
    Week 1 (Dec 8-12): Training
    Week 2 (Dec 15-19): Testing
    """
    global TRAINING_DATA, TESTING_DATA
    
    print(f"\n{'='*70}")
    print(f"EXTENDED MINUTE-BY-MINUTE DATA TEST (LATE 2025)")
    print(f"{'='*70}")
    print(f"\nTicker: {TICKER}")
    print(f"Training: {TRAIN_START} to {TRAIN_END} (1 trading week)")
    print(f"Testing: {TEST_START} to {TEST_END} (1 trading week)")
    print(f"Interval: {MINUTE_INTERVAL}")
    print(f"Expected samples: ~1,950 training + ~1,950 testing = 3,900 total")
    
    try:
        print(f"\n1. Fetching TRAINING data ({TRAIN_START} to {TRAIN_END})...")
        training_ticker = yf.Ticker(TICKER)
        TRAINING_DATA = training_ticker.history(
            start=TRAIN_START,
            end=TRAIN_END,
            interval=MINUTE_INTERVAL
        )
        
        if TRAINING_DATA.empty:
            raise ValueError("No training data fetched. Market may be closed.")
        
        TRAINING_DATA = TRAINING_DATA.reset_index()
        if 'Datetime' in TRAINING_DATA.columns:
            TRAINING_DATA = TRAINING_DATA.rename(columns={'Datetime': 'Date'})
        elif 'Date' not in TRAINING_DATA.columns:
            TRAINING_DATA.columns = ['Date'] + list(TRAINING_DATA.columns[1:])
        
        TRAINING_DATA = TRAINING_DATA.dropna()
        print(f"  ✓ Loaded {len(TRAINING_DATA)} minute bars (training)")
        print(f"    Date range: {TRAINING_DATA['Date'].min()} to {TRAINING_DATA['Date'].max()}")
        
        print(f"\n2. Fetching TESTING data ({TEST_START} to {TEST_END})...")
        testing_ticker = yf.Ticker(TICKER)
        TESTING_DATA = testing_ticker.history(
            start=TEST_START,
            end=TEST_END,
            interval=MINUTE_INTERVAL
        )
        
        if TESTING_DATA.empty:
            raise ValueError("No testing data fetched. Market may be closed.")
        
        TESTING_DATA = TESTING_DATA.reset_index()
        if 'Datetime' in TESTING_DATA.columns:
            TESTING_DATA = TESTING_DATA.rename(columns={'Datetime': 'Date'})
        elif 'Date' not in TESTING_DATA.columns:
            TESTING_DATA.columns = ['Date'] + list(TESTING_DATA.columns[1:])
        
        TESTING_DATA = TESTING_DATA.dropna()
        print(f"  ✓ Loaded {len(TESTING_DATA)} minute bars (testing)")
        print(f"    Date range: {TESTING_DATA['Date'].min()} to {TESTING_DATA['Date'].max()}")
        
    except Exception as e:
        print(f"\n⚠️  Error fetching live data: {e}")
        print(f"Generating synthetic minute data instead...")
        generate_synthetic_minute_data()


def generate_synthetic_minute_data():
    """
    Generate realistic synthetic minute-by-minute OHLCV data if live data unavailable.
    Simulates 2 full trading weeks with realistic minute-level volatility.
    """
    global TRAINING_DATA, TESTING_DATA
    
    # Trading hours: 9:30 AM - 4:00 PM = 6.5 hours = 390 minutes per day
    minutes_per_day = 390
    trading_days_train = 5  # Mon-Fri week 1
    trading_days_test = 5   # Mon-Fri week 2
    
    base_price = 240.0  # AAPL price in late 2025
    
    def generate_minute_prices(num_days, start_date_str, base_price, seed_offset=0):
        """Generate realistic minute-level price data"""
        dates = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        np.random.seed(42 + seed_offset)  # Different seed for test data variation
        current_date = pd.to_datetime(start_date_str)
        current_price = base_price
        
        for day_offset in range(num_days):
            current_date = pd.to_datetime(start_date_str) + timedelta(days=day_offset)
            
            # Skip weekends
            if current_date.weekday() >= 5:
                continue
            
            # Start of day price (with slight gap)
            day_open = current_price * (1 + np.random.normal(0, 0.002))
            current_price = day_open
            
            # Generate 390 minutes of data per trading day
            for minute in range(minutes_per_day):
                # Time stamp
                market_open = current_date.replace(hour=9, minute=30, second=0)
                timestamp = market_open + timedelta(minutes=minute)
                
                # Minute-level returns (very small, tight)
                minute_return = np.random.normal(0, 0.0002)  # 0.02% volatility
                
                # Intraday mean reversion
                mean_reversion = 0.00001 * (day_open - current_price) / current_price
                
                # Slight uptrend (AAPL tends to go up)
                trend = 0.000005
                
                new_price = current_price * (1 + minute_return + mean_reversion + trend)
                
                # OHLC for this minute
                open_price = current_price
                high_price = max(open_price, new_price) * (1 + np.random.uniform(0, 0.0005))
                low_price = min(open_price, new_price) * (1 - np.random.uniform(0, 0.0005))
                close_price = new_price
                volume = int(np.random.uniform(10000, 50000))
                
                dates.append(timestamp)
                opens.append(open_price)
                highs.append(high_price)
                lows.append(low_price)
                closes.append(close_price)
                volumes.append(volume)
                
                current_price = new_price
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes,
            'Adj Close': closes
        })
        
        return df, current_price
    
    print(f"\nGenerating synthetic minute data for {TICKER}...")
    TRAINING_DATA, price_after_train = generate_minute_prices(trading_days_train, TRAIN_START, base_price, seed_offset=0)
    TESTING_DATA, _ = generate_minute_prices(trading_days_test, TEST_START, price_after_train, seed_offset=100)
    
    print(f"  ✓ Generated {len(TRAINING_DATA)} minute bars (training)")
    print(f"  ✓ Generated {len(TESTING_DATA)} minute bars (testing)")


def add_technical_indicators(df, is_training=False, training_tail=None):
    """
    Add technical indicators calculated for minute-level data.
    Adjusted windows for minute bars instead of daily bars.
    """
    df = df.copy()
    
    # Use training data tail for warm-up on test data
    if not is_training and training_tail is not None:
        combined = pd.concat([training_tail, df], ignore_index=True)
        combined = _add_minute_indicators(combined)
        df = combined.iloc[len(training_tail):].reset_index(drop=True)
    else:
        df = _add_minute_indicators(df)
    
    return df


def _add_minute_indicators(df):
    """
    Calculate technical indicators for minute-level data.
    Uses shorter windows appropriate for high-frequency data.
    """
    df = df.copy()
    
    # Short-term Moving Averages (minute-level)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()      # ~5 minutes
    df['SMA_10'] = df['Close'].rolling(window=10).mean()    # ~10 minutes
    df['SMA_20'] = df['Close'].rolling(window=20).mean()    # ~20 minutes
    df['SMA_60'] = df['Close'].rolling(window=60).mean()    # ~1 hour
    df['SMA_260'] = df['Close'].rolling(window=260).mean()  # ~1 trading day
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # Momentum (short-term)
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    
    # Rate of Change
    df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # Volatility (Standard Deviation) - minute level
    df['Volatility_20'] = df['Close'].rolling(window=20).std()
    df['Volatility_60'] = df['Close'].rolling(window=60).std()
    
    # Average True Range (ATR)
    df['High_Low'] = df['High'] - df['Low']
    df['High_Close'] = abs(df['High'] - df['Close'].shift())
    df['Low_Close'] = abs(df['Low'] - df['Close'].shift())
    df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    df = df.drop(['High_Low', 'High_Close', 'Low_Close', 'TR'], axis=1)
    
    # Relative Strength Index (RSI) - minute level
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    
    # Price position in Bollinger Band
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
    df['BB_Position'] = df['BB_Position'].fillna(0)
    
    # Returns
    df['Return_1'] = df['Close'].pct_change(1)
    df['Return_5'] = df['Close'].pct_change(5)
    df['Return_10'] = df['Close'].pct_change(10)
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df


def prepare_training_data():
    """
    Prepare training data: add indicators and create target variable.
    Target: 1 if price goes up in next 5 minutes, 0 otherwise.
    """
    global TRAINING_DATA
    
    print(f"\n3. Preparing training data...")
    
    # Add technical indicators to training data
    TRAINING_DATA = add_technical_indicators(TRAINING_DATA, is_training=True)
    
    # Create target variable
    # Look ahead 5 minutes to determine if price will go up
    TRAINING_DATA['Target'] = (TRAINING_DATA['Close'].shift(-5) > TRAINING_DATA['Close']).astype(int)
    
    # Remove rows with NaN
    TRAINING_DATA = TRAINING_DATA.dropna()
    
    # Ensure Date column is datetime
    TRAINING_DATA['Date'] = pd.to_datetime(TRAINING_DATA['Date'])
    
    print(f"  ✓ Training data ready: {len(TRAINING_DATA)} samples")
    print(f"    Target distribution: {TRAINING_DATA['Target'].value_counts().to_dict()}")
    
    return TRAINING_DATA


def prepare_testing_data():
    """
    Prepare testing data: add indicators with training data warm-up.
    """
    global TESTING_DATA, TRAINING_DATA
    
    print(f"\n4. Preparing testing data...")
    
    # Add indicators with training data tail for warm-up
    training_tail = TRAINING_DATA.iloc[-260:].copy()  # Use last ~1 hour of training for warm-up
    TESTING_DATA = add_technical_indicators(TESTING_DATA, is_training=False, training_tail=training_tail)
    
    # Create target variable for evaluation
    TESTING_DATA['Target'] = (TESTING_DATA['Close'].shift(-5) > TESTING_DATA['Close']).astype(int)
    
    # Remove rows with NaN
    TESTING_DATA = TESTING_DATA.dropna()
    
    # Ensure Date column is datetime
    TESTING_DATA['Date'] = pd.to_datetime(TESTING_DATA['Date'])
    
    print(f"  ✓ Testing data ready: {len(TESTING_DATA)} samples")
    print(f"    Target distribution: {TESTING_DATA['Target'].value_counts().to_dict()}")
    
    return TESTING_DATA


def train_models():
    """
    Train LightGBM and Ensemble models on minute-level data.
    """
    global TRAINING_DATA, TESTING_DATA
    
    print(f"\n5. Training models on minute-level data...")
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Prepare features for training
    feature_cols = [col for col in TRAINING_DATA.columns if col not in ['Date', 'Target', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    
    X_train = TRAINING_DATA[feature_cols]
    y_train = TRAINING_DATA['Target']
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Training samples: {len(X_train)}")
    
    # Train LightGBM
    print(f"\n  Training LightGBM model...")
    results = train_lightgbm_model(TRAINING_DATA, TESTING_DATA, results_dir=RESULTS_DIR)
    lgb_model = results['model']
    
    print(f"  ✓ LightGBM training complete")
    
    return lgb_model, results


def evaluate_strategy(model, model_results):
    """
    Use model predictions from training results.
    """
    print(f"\n6. Model predictions available:")
    
    if model_results and 'results' in model_results:
        predictions = model_results['results']
        print(f"  ✓ Predictions generated: {len(predictions)} samples")
        print(f"    Average confidence: {predictions['Confidence'].mean():.2f}")
        print(f"    High-confidence predictions: {(predictions['Confidence'] > 0.7).sum()}")
        
        return predictions
    
    return None


def save_results(model, predictions):
    """
    Save all results to CSV and JSON files.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save training data with predictions
    train_output = TRAINING_DATA.copy()
    train_output.to_csv(RESULTS_DIR / f"{TICKER}_minute_training_{TRAIN_START}_to_{TRAIN_END}.csv", index=False)
    
    # Save testing data with predictions
    test_output = TESTING_DATA.copy()
    test_output.to_csv(RESULTS_DIR / f"{TICKER}_minute_testing_{TEST_START}_to_{TEST_END}.csv", index=False)
    
    # Save predictions
    if predictions is not None:
        predictions.to_csv(RESULTS_DIR / f"{TICKER}_minute_predictions_{TEST_START}_to_{TEST_END}.csv", index=False)
    
    # Save summary
    summary = {
        "ticker": TICKER,
        "training_period": f"{TRAIN_START} to {TRAIN_END}",
        "testing_period": f"{TEST_START} to {TEST_END}",
        "interval": MINUTE_INTERVAL,
        "training_samples": len(TRAINING_DATA),
        "testing_samples": len(TESTING_DATA),
        "predictions": len(predictions) if predictions is not None else 0,
        "results_dir": str(RESULTS_DIR)
    }
    
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to: {RESULTS_DIR}")


def main():
    """
    Main test orchestrator: fetch data, train models, generate signals, backtest.
    """
    try:
        # Step 1: Fetch data
        fetch_minute_data()
        
        # Step 2: Prepare training data
        prepare_training_data()
        
        # Step 3: Prepare testing data
        prepare_testing_data()
        
        # Step 4: Train models
        model, model_results = train_models()
        
        # Step 5: Evaluate strategy
        predictions = evaluate_strategy(model, model_results)
        
        # Step 6: Save results
        save_results(model, predictions)
        
        print(f"\n{'='*70}")
        print(f"✓ EXTENDED MINUTE-BY-MINUTE TEST COMPLETE")
        print(f"{'='*70}")
        print(f"Results saved to: {RESULTS_DIR}")
        
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
