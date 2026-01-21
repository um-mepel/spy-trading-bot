"""
Multi-Asset Test
================
Tests the model's predictive ability across various assets on daily data.

Findings:
- ETH-USD: 60.79% accuracy, +10.79% edge (statistically significant)
- QQQ: 55.69% accuracy, +5.69% edge (statistically significant)
- Most other assets show no consistent edge

Usage:
    python tests/multi_asset_test.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import lightgbm as lgb
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

RESULTS_DIR = Path('results/multi_asset_test')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def add_features(df):
    """Add technical features using only past data."""
    df = df.copy()
    
    # Microstructure
    df['Spread_Proxy'] = (df['High'] - df['Low']) / df['Close'] * 10000
    price_change = df['Close'].diff().abs()
    df['Trade_Intensity'] = df['Volume'] / (price_change + 0.01)
    
    # Mean Reversion
    for period in [10, 20, 50]:
        mean = df['Close'].rolling(period).mean()
        std = df['Close'].rolling(period).std()
        df[f'ZScore_{period}'] = (df['Close'] - mean) / (std + 1e-10)
        df[f'Dist_SMA_{period}'] = (df['Close'] - mean) / mean * 100
    
    # Basic
    df['Return_1'] = df['Close'].pct_change()
    df['Return_5'] = df['Close'].pct_change(5)
    df['Return_10'] = df['Close'].pct_change(10)
    df['Return_20'] = df['Close'].pct_change(20)
    df['HL_Range_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    return df


def test_asset(ticker, target_days=20):
    """Test model on a single asset."""
    try:
        # Fetch data
        df = yf.download(ticker, period="max", interval="1d", progress=False)
        if len(df) < 500:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Add features
        df = add_features(df)
        
        # Add target
        df['Target'] = df['Close'].shift(-target_days) - df['Close']
        df = df.iloc[:-target_days]
        
        # Feature columns
        feature_cols = [
            'Spread_Proxy', 'Trade_Intensity',
            'ZScore_10', 'ZScore_20', 'ZScore_50',
            'Dist_SMA_10', 'Dist_SMA_20', 'Dist_SMA_50',
            'Return_1', 'Return_5', 'Return_10', 'Return_20',
            'HL_Range_Pct', 'Close_Position'
        ]
        
        # Clean data
        df_clean = df[feature_cols + ['Target', 'Close', 'Date']].dropna()
        
        # Split (80/20 with buffer)
        buffer = 50
        split_idx = int(len(df_clean) * 0.8)
        train = df_clean.iloc[:split_idx - buffer]
        test = df_clean.iloc[split_idx:]
        
        if len(train) < 100 or len(test) < 50:
            return None
        
        X_train = train[feature_cols].values
        y_train = train['Target'].values
        X_test = test[feature_cols].values
        y_test = test['Target'].values
        
        # Train
        train_set = lgb.Dataset(X_train, label=y_train)
        model = lgb.train({
            'objective': 'regression',
            'learning_rate': 0.05,
            'verbosity': -1
        }, train_set, 100)
        
        # Evaluate
        preds = model.predict(X_test)
        pred_dir = preds > 0
        actual_dir = y_test > 0
        accuracy = (pred_dir == actual_dir).mean()
        
        n = len(y_test)
        z_score = (accuracy - 0.5) / np.sqrt(0.25 / n)
        p_value = 1 - stats.norm.cdf(abs(z_score))
        
        return {
            'ticker': ticker,
            'samples': len(df_clean),
            'test_samples': len(test),
            'accuracy': accuracy,
            'edge': (accuracy - 0.5) * 100,
            'z_score': z_score,
            'p_value': p_value,
            'significant': p_value < 0.05 and accuracy > 0.5
        }
    
    except Exception as e:
        print(f"Error testing {ticker}: {e}")
        return None


def main():
    print("=" * 70)
    print("MULTI-ASSET TEST")
    print("=" * 70)
    
    # Assets to test
    assets = [
        'SPY', 'QQQ', 'IWM',  # ETFs
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Mega caps
        'BTC-USD', 'ETH-USD',  # Crypto
    ]
    
    results = []
    for ticker in assets:
        print(f"\nTesting {ticker}...")
        result = test_asset(ticker)
        if result:
            results.append(result)
            status = "SIGNIFICANT" if result['significant'] else "No edge"
            print(f"  Accuracy: {result['accuracy']:.2%}, Edge: {result['edge']:+.2f}%, {status}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('edge', ascending=False)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Ticker':<10} {'Accuracy':>10} {'Edge':>10} {'P-Value':>12} {'Significant':>12}")
    print("-" * 60)
    for _, row in results_df.iterrows():
        sig = "YES" if row['significant'] else "NO"
        print(f"{row['ticker']:<10} {row['accuracy']:>9.2%} {row['edge']:>+9.2f}% {row['p_value']:>12.6f} {sig:>12}")
    
    # Save results
    results_df.to_csv(RESULTS_DIR / 'results.csv', index=False)
    print(f"\n✓ Results saved to {RESULTS_DIR / 'results.csv'}")
    
    # Highlight winners
    significant = results_df[results_df['significant']]
    if len(significant) > 0:
        print("\n" + "=" * 70)
        print("ASSETS WITH SIGNIFICANT EDGE")
        print("=" * 70)
        for _, row in significant.iterrows():
            print(f"\n{row['ticker']}:")
            print(f"  Accuracy: {row['accuracy']:.2%}")
            print(f"  Edge: {row['edge']:+.2f}%")
            print(f"  P-Value: {row['p_value']:.10f}")


if __name__ == '__main__':
    main()
