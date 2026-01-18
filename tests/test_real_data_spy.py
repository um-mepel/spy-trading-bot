"""
Real Data SPY Test: 2+ Months Training, 1 Month Testing
======================================================

Using actual SPY daily OHLCV data to validate the trading model edge.
- Training: 2 months of data
- Testing: 1 month of data  
- Target: >3% edge over testing period is excellent
- Strategy: Daily price direction prediction with confidence scoring

Real data removes any concerns about synthetic data artifacts.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lightgbm_model import train_lightgbm_model


# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "real_data_spy"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Ticker
TICKER = "SPY"

# Test parameters - using real historical data
# Training: October 2023 - November 2023 (60 days)
# Testing: December 2023 (21 days)
TRAINING_START = '2023-10-01'
TRAINING_END = '2023-11-30'
TESTING_START = '2023-12-01'
TESTING_END = '2023-12-29'


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_real_spy_data():
    """Load SPY data from CSV file."""
    csv_path = DATA_DIR / "SPY_training_2022_2024.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"SPY data file not found: {csv_path}")
    
    print(f"Loading SPY data from {csv_path}")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"✓ Loaded {len(df)} rows, date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    return df


def prepare_training_data(df, start_date, end_date):
    """
    Prepare training data with technical indicators and target variable.
    Target: 1 if next day's close > today's close, 0 otherwise
    """
    df = df.copy()
    
    # Filter to date range
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df = df[mask].reset_index(drop=True)
    
    print(f"Training data: {len(df)} rows ({start_date} to {end_date})")
    
    if len(df) < 30:
        raise ValueError(f"Not enough training data: {len(df)} rows")
    
    # Create target: 1 if price goes up next day, 0 otherwise
    df['Next_Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Next_Close'] > df['Close']).astype(int)
    
    # Drop last row (no next day target) and rows with NaN in features
    df = df.dropna(subset=['Target', 'Close', 'SMA_20', 'RSI_14', 'MACD'])
    
    print(f"After dropping NaN: {len(df)} rows")
    print(f"Target distribution: {df['Target'].value_counts().to_dict()}")
    
    return df


def prepare_testing_data(df_train, df_full, start_date, end_date):
    """
    Prepare testing data with warm-up using training data tail.
    Warm-up: Use last 20 rows of training data to initialize rolling indicators
    """
    df = df_full.copy()
    
    # Filter to date range
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df = df[mask].reset_index(drop=True)
    
    print(f"Testing data: {len(df)} rows ({start_date} to {end_date})")
    
    if len(df) < 10:
        raise ValueError(f"Not enough testing data: {len(df)} rows")
    
    # Create target for evaluation
    df['Next_Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Next_Close'] > df['Close']).astype(int)
    
    # Drop last row (no next day target) and rows with NaN in features
    df = df.dropna(subset=['Target', 'Close', 'SMA_20', 'RSI_14', 'MACD'])
    
    print(f"After dropping NaN: {len(df)} rows")
    print(f"Target distribution: {df['Target'].value_counts().to_dict()}")
    
    return df


def get_feature_columns():
    """Return list of feature columns to use for training."""
    return [
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
        'EMA_12', 'EMA_26', 'MACD', 'Signal_Line', 'MACD_Histogram',
        'Momentum_10', 'ROC_10', 'Volatility_20', 'ATR_14',
        'RSI_14', 'BB_Middle_20', 'BB_Upper_20', 'BB_Lower_20',
        'BB_Width', 'BB_Position', 'Volume_SMA_20', 'Volume_Ratio',
        'Log_Return', 'Daily_Return_Pct', 'HL_Range_Pct',
        'CO_Range_Pct', 'Price_SMA20_Distance', 'Price_SMA50_Distance'
    ]


# ============================================================================
# MAIN TEST
# ============================================================================

def main():
    """Run real data test with 2+ months training and 1 month testing."""
    
    print("\n" + "="*80)
    print("REAL DATA SPY TEST: 2+ Months Training vs 1 Month Testing")
    print("="*80)
    
    # Load all data
    df_full = load_real_spy_data()
    
    # Prepare training data
    print("\n--- TRAINING DATA ---")
    df_train = prepare_training_data(
        df_full,
        pd.to_datetime(TRAINING_START),
        pd.to_datetime(TRAINING_END)
    )
    
    # Prepare testing data
    print("\n--- TESTING DATA ---")
    df_test = prepare_testing_data(
        df_train,
        df_full,
        pd.to_datetime(TESTING_START),
        pd.to_datetime(TESTING_END)
    )
    
    # Show date ranges
    print("\n--- DATE RANGES ---")
    print(f"Training: {df_train['Date'].min().date()} to {df_train['Date'].max().date()}")
    print(f"Testing:  {df_test['Date'].min().date()} to {df_test['Date'].max().date()}")
    print(f"Training samples: {len(df_train)}")
    print(f"Testing samples:  {len(df_test)}")
    
    # Select features
    features = get_feature_columns()
    print(f"\n--- FEATURES ({len(features)}) ---")
    print(f"Using {len(features)} technical indicators")
    
    # Verify features exist
    missing_features = [f for f in features if f not in df_train.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        features = [f for f in features if f in df_train.columns]
        print(f"Using {len(features)} available features")
    
    # Drop NaN in features
    df_train = df_train[features + ['Target', 'Date', 'Close']].dropna()
    df_test = df_test[features + ['Target', 'Date', 'Close']].dropna()
    
    print(f"After feature cleanup:")
    print(f"  Training: {len(df_train)} rows")
    print(f"  Testing:  {len(df_test)} rows")
    
    if len(df_train) < 20 or len(df_test) < 10:
        print("ERROR: Insufficient data after cleanup")
        return
    
    # Save training and testing data
    print("\n--- SAVING DATA ---")
    train_csv = RESULTS_DIR / f"{TICKER}_real_training_{TRAINING_START}_to_{TRAINING_END}.csv"
    test_csv = RESULTS_DIR / f"{TICKER}_real_testing_{TESTING_START}_to_{TESTING_END}.csv"
    
    df_train.to_csv(train_csv, index=False)
    df_test.to_csv(test_csv, index=False)
    
    print(f"✓ Saved training data: {train_csv}")
    print(f"✓ Saved testing data: {test_csv}")
    
    # Train model
    print("\n--- TRAINING LIGHTGBM MODEL ---")
    try:
        results = train_lightgbm_model(
            df_train,
            df_test,
            results_dir=str(RESULTS_DIR)
        )
        
        # Calculate edge
        if results and hasattr(results, 'accuracy'):
            accuracy = results.accuracy
            edge = (accuracy - 0.5) * 100
            
            print(f"\n{'='*80}")
            print(f"RESULTS")
            print(f"{'='*80}")
            print(f"Accuracy:     {accuracy:.1%}")
            print(f"Edge:         {edge:+.1f}% (vs 50% random)")
            print(f"Predictions:  {len(results.predictions)}")
            print(f"Avg Confidence: {results.confidence.mean():.2f}")
            
            # Save summary
            summary = {
                'ticker': TICKER,
                'training_start': TRAINING_START,
                'training_end': TRAINING_END,
                'testing_start': TESTING_START,
                'testing_end': TESTING_END,
                'training_samples': len(df_train),
                'testing_samples': len(df_test),
                'accuracy': float(accuracy),
                'edge_percent': float(edge),
                'total_predictions': len(results.predictions),
                'average_confidence': float(results.confidence.mean()),
                'high_confidence_accuracy': float((results.confidence > 0.7).mean()),
                'timestamp': datetime.now().isoformat()
            }
            
            summary_path = RESULTS_DIR / "summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n✓ Summary saved to {summary_path}")
            
            print(f"\n{'='*80}")
            if edge > 3.0:
                print(f"✅ EXCELLENT EDGE: {edge:.1f}% (>3% target achieved!)")
            elif edge > 2.0:
                print(f"✓ GOOD EDGE: {edge:.1f}% (acceptable)")
            elif edge > 0.0:
                print(f"~ MARGINAL EDGE: {edge:.1f}% (need more testing)")
            else:
                print(f"✗ NO EDGE: {edge:.1f}% (model not profitable)")
            print(f"{'='*80}")
            
        else:
            print("✗ Model training failed - no results returned")
    
    except Exception as e:
        print(f"✗ Error during model training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
