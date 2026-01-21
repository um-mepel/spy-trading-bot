#!/usr/bin/env python3
"""
COMPREHENSIVE MINUTE-LEVEL FEATURE TESTING

Tests many different feature combinations on YEARS of real minute data.
NO DATA LEAKAGE - strict train/test split.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import json

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

import lightgbm as lgb
from scipy import stats

# Alpaca API
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

API_KEY = "PKQCAK3OCWGUZ5JHL6QFBXNSLV"
SECRET_KEY = "51B23zrTEMd9sBbXqWyqaewmmum5XK6oUsSn4RThcTRU"

RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'comprehensive_feature_test'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def fetch_minute_data(start_date: str, end_date: str, ticker: str = "SPY"):
    """Fetch minute data from Alpaca."""
    print(f"Fetching {ticker} minute data: {start_date} to {end_date}")
    
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Minute,
        start=datetime.strptime(start_date, "%Y-%m-%d"),
        end=datetime.strptime(end_date, "%Y-%m-%d")
    )
    
    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()
    
    df = df.rename(columns={
        'timestamp': 'Datetime', 'open': 'Open', 'high': 'High',
        'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })
    
    if 'symbol' in df.columns:
        df = df[df['symbol'] == ticker]
    
    df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.sort_values('Datetime').reset_index(drop=True)
    
    print(f"  Fetched {len(df):,} bars")
    return df


# =============================================================================
# COMPREHENSIVE FEATURE ENGINEERING
# =============================================================================

def add_basic_features(df):
    """Basic price and volume features."""
    df = df.copy()
    
    # Returns
    df['Return_1'] = df['Close'].pct_change()
    df['Return_5'] = df['Close'].pct_change(5)
    df['Return_10'] = df['Close'].pct_change(10)
    df['Return_20'] = df['Close'].pct_change(20)
    
    # Log returns
    df['LogReturn_1'] = np.log(df['Close'] / df['Close'].shift(1))
    df['LogReturn_5'] = np.log(df['Close'] / df['Close'].shift(5))
    
    # Price position
    df['HL_Range'] = df['High'] - df['Low']
    df['HL_Range_Pct'] = df['HL_Range'] / df['Close'] * 100
    df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Open_Close_Range'] = (df['Close'] - df['Open']) / df['Open'] * 100
    
    # Gap
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
    
    return df


def add_volume_features(df):
    """Volume-based features."""
    df = df.copy()
    
    # Volume moving averages
    for period in [5, 10, 20, 50]:
        df[f'Volume_SMA_{period}'] = df['Volume'].rolling(period).mean()
        df[f'Volume_Ratio_{period}'] = df['Volume'] / df[f'Volume_SMA_{period}']
    
    # Volume momentum
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_Change_5'] = df['Volume'].pct_change(5)
    
    # On-Balance Volume
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    df['OBV_Change'] = df['OBV'].pct_change(5)
    
    # Volume-weighted price
    df['VWAP_5'] = (df['Close'] * df['Volume']).rolling(5).sum() / df['Volume'].rolling(5).sum()
    df['VWAP_20'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
    df['VWAP_Ratio'] = df['Close'] / df['VWAP_20']
    
    # Money Flow
    df['Money_Flow'] = df['Close'] * df['Volume']
    df['Money_Flow_Ratio'] = df['Money_Flow'].rolling(5).mean() / df['Money_Flow'].rolling(20).mean()
    
    return df


def add_momentum_features(df):
    """Momentum indicators."""
    df = df.copy()
    
    # Simple momentum
    for period in [3, 5, 10, 20, 60]:
        df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
        df[f'ROC_{period}'] = (df['Close'] / df['Close'].shift(period) - 1) * 100
    
    # RSI with multiple periods
    for period in [5, 9, 14, 21]:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
    # Stochastic
    for period in [5, 14]:
        low_min = df['Low'].rolling(period).min()
        high_max = df['High'].rolling(period).max()
        df[f'Stoch_K_{period}'] = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-10)
        df[f'Stoch_D_{period}'] = df[f'Stoch_K_{period}'].rolling(3).mean()
    
    # Williams %R
    for period in [10, 20]:
        high_max = df['High'].rolling(period).max()
        low_min = df['Low'].rolling(period).min()
        df[f'Williams_R_{period}'] = -100 * (high_max - df['Close']) / (high_max - low_min + 1e-10)
    
    # CCI (Commodity Channel Index)
    for period in [14, 20]:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        df[f'CCI_{period}'] = (typical_price - sma) / (0.015 * mad + 1e-10)
    
    return df


def add_trend_features(df):
    """Trend indicators."""
    df = df.copy()
    
    # Moving averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
        df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
    
    # Price relative to MAs
    for period in [5, 10, 20, 50]:
        df[f'Close_SMA_{period}_Ratio'] = df['Close'] / df[f'SMA_{period}']
        df[f'Close_EMA_{period}_Ratio'] = df['Close'] / df[f'EMA_{period}']
    
    # MA crossovers
    df['SMA_5_20_Diff'] = df['SMA_5'] - df['SMA_20']
    df['SMA_10_50_Diff'] = df['SMA_10'] - df['SMA_50']
    df['EMA_5_20_Diff'] = df['EMA_5'] - df['EMA_20']
    
    # MACD
    df['MACD'] = df['EMA_12'] if 'EMA_12' in df else df['Close'].ewm(span=12).mean()
    df['MACD'] = df['MACD'] - (df['EMA_26'] if 'EMA_26' in df else df['Close'].ewm(span=26).mean())
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # ADX (simplified)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].shift(1) - df['Low']
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = pd.concat([df['High'] - df['Low'], 
                    abs(df['High'] - df['Close'].shift()), 
                    abs(df['Low'] - df['Close'].shift())], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr_14 + 1e-10)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr_14 + 1e-10)
    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di
    df['ADX'] = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100).rolling(14).mean()
    
    return df


def add_volatility_features(df):
    """Volatility indicators."""
    df = df.copy()
    
    # Standard deviation
    for period in [5, 10, 20, 50]:
        df[f'Volatility_{period}'] = df['Return_1'].rolling(period).std() if 'Return_1' in df else df['Close'].pct_change().rolling(period).std()
        df[f'Volatility_{period}'] = df[f'Volatility_{period}'] * 100  # As percentage
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    for period in [5, 14, 20]:
        df[f'ATR_{period}'] = tr.rolling(period).mean()
        df[f'ATR_{period}_Pct'] = df[f'ATR_{period}'] / df['Close'] * 100
    
    # Bollinger Bands
    for period in [10, 20]:
        sma = df['Close'].rolling(period).mean()
        std = df['Close'].rolling(period).std()
        df[f'BB_Upper_{period}'] = sma + 2 * std
        df[f'BB_Lower_{period}'] = sma - 2 * std
        df[f'BB_Position_{period}'] = (df['Close'] - df[f'BB_Lower_{period}']) / (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}'] + 1e-10)
        df[f'BB_Width_{period}'] = (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']) / sma * 100
    
    # Keltner Channels
    ema20 = df['Close'].ewm(span=20).mean()
    atr10 = df['ATR_14'] if 'ATR_14' in df else tr.rolling(14).mean()
    df['Keltner_Upper'] = ema20 + 2 * atr10
    df['Keltner_Lower'] = ema20 - 2 * atr10
    df['Keltner_Position'] = (df['Close'] - df['Keltner_Lower']) / (df['Keltner_Upper'] - df['Keltner_Lower'] + 1e-10)
    
    # Volatility ratio
    df['Volatility_Ratio'] = df['Volatility_5'] / (df['Volatility_20'] + 1e-10) if 'Volatility_5' in df else 1
    
    return df


def add_time_features(df):
    """Time-based features (for minute data)."""
    df = df.copy()
    
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Time of day
        df['Hour'] = df['Datetime'].dt.hour
        df['Minute'] = df['Datetime'].dt.minute
        df['Time_Decimal'] = df['Hour'] + df['Minute'] / 60
        
        # Session indicators
        df['Is_Open'] = ((df['Hour'] == 9) & (df['Minute'] <= 30)).astype(int)
        df['Is_Close'] = ((df['Hour'] == 15) & (df['Minute'] >= 30)).astype(int)
        df['Is_Lunch'] = ((df['Hour'] >= 12) & (df['Hour'] < 13)).astype(int)
        df['Is_Morning'] = (df['Hour'] < 12).astype(int)
        df['Is_Afternoon'] = (df['Hour'] >= 12).astype(int)
        
        # Minutes since open (9:30)
        df['Minutes_Since_Open'] = (df['Hour'] - 9) * 60 + df['Minute'] - 30
        df['Minutes_Since_Open'] = df['Minutes_Since_Open'].clip(lower=0)
        
        # Day of week
        df['DayOfWeek'] = df['Datetime'].dt.dayofweek
        df['Is_Monday'] = (df['DayOfWeek'] == 0).astype(int)
        df['Is_Friday'] = (df['DayOfWeek'] == 4).astype(int)
        
    return df


def add_mean_reversion_features(df):
    """Mean reversion indicators."""
    df = df.copy()
    
    # Z-scores
    for period in [10, 20, 50]:
        mean = df['Close'].rolling(period).mean()
        std = df['Close'].rolling(period).std()
        df[f'ZScore_{period}'] = (df['Close'] - mean) / (std + 1e-10)
    
    # Distance from moving average
    for period in [10, 20, 50]:
        if f'SMA_{period}' in df.columns:
            df[f'Dist_SMA_{period}'] = (df['Close'] - df[f'SMA_{period}']) / df[f'SMA_{period}'] * 100
    
    # Percentile rank
    for period in [20, 50, 100]:
        df[f'Percentile_{period}'] = df['Close'].rolling(period).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) == period else 50
        )
    
    return df


def add_pattern_features(df):
    """Candlestick and pattern features."""
    df = df.copy()
    
    # Body size
    df['Body'] = df['Close'] - df['Open']
    df['Body_Pct'] = df['Body'] / df['Open'] * 100
    df['Body_Ratio'] = abs(df['Body']) / (df['HL_Range'] + 1e-10)
    
    # Upper/Lower shadows
    df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Upper_Shadow_Ratio'] = df['Upper_Shadow'] / (df['HL_Range'] + 1e-10)
    df['Lower_Shadow_Ratio'] = df['Lower_Shadow'] / (df['HL_Range'] + 1e-10)
    
    # Candle type (simplified)
    df['Is_Bullish'] = (df['Close'] > df['Open']).astype(int)
    df['Is_Doji'] = (abs(df['Body_Pct']) < 0.1).astype(int)
    
    # Consecutive up/down
    direction = np.sign(df['Close'].diff())
    df['Consecutive_Direction'] = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1
    df['Consecutive_Direction'] = df['Consecutive_Direction'] * direction
    
    # Higher highs, lower lows
    df['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)
    df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
    df['Higher_Close'] = (df['Close'] > df['Close'].shift(1)).astype(int)
    
    return df


def add_microstructure_features(df):
    """Market microstructure features."""
    df = df.copy()
    
    # Spread proxy (using high-low as approximation)
    df['Spread_Proxy'] = (df['High'] - df['Low']) / df['Close'] * 10000  # In basis points
    
    # Trade intensity (volume per price change)
    df['Trade_Intensity'] = df['Volume'] / (abs(df['Close'].diff()) + 0.01)
    
    # Amihud illiquidity ratio (simplified)
    df['Illiquidity'] = abs(df['Return_1']) / (df['Volume'] / 1e6 + 1e-10) if 'Return_1' in df else 0
    
    # Realized variance
    for period in [5, 10, 20]:
        df[f'RealizedVar_{period}'] = (df['LogReturn_1'] ** 2).rolling(period).sum() if 'LogReturn_1' in df else 0
    
    return df


def add_all_features(df):
    """Add all feature categories."""
    print("  Adding basic features...")
    df = add_basic_features(df)
    print("  Adding volume features...")
    df = add_volume_features(df)
    print("  Adding momentum features...")
    df = add_momentum_features(df)
    print("  Adding trend features...")
    df = add_trend_features(df)
    print("  Adding volatility features...")
    df = add_volatility_features(df)
    print("  Adding time features...")
    df = add_time_features(df)
    print("  Adding mean reversion features...")
    df = add_mean_reversion_features(df)
    print("  Adding pattern features...")
    df = add_pattern_features(df)
    print("  Adding microstructure features...")
    df = add_microstructure_features(df)
    
    return df


def get_feature_columns(df):
    """Get all valid feature columns (exclude targets and metadata)."""
    exclude = {
        'Datetime', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Target', 'Price_Change', 'Price_Change_20', 'Adj Close',
        'symbol', 'index'
    }
    
    features = []
    for col in df.columns:
        if col not in exclude:
            # Check if column is numeric and has reasonable values
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                if df[col].isna().sum() < len(df) * 0.5:  # Less than 50% NaN
                    features.append(col)
    
    return features


def train_and_evaluate(X_train, y_train, X_test, y_test, feature_names, name="Model"):
    """Train LightGBM and evaluate."""
    
    # Train
    train_set = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbosity': -1,
        'n_jobs': -1
    }
    
    model = lgb.train(params, train_set, num_boost_round=200)
    
    # Predict
    preds = model.predict(X_test)
    
    # Direction accuracy
    pred_direction = preds > 0
    actual_direction = y_test > 0
    accuracy = (pred_direction == actual_direction).mean()
    
    # Edge
    edge = (accuracy - 0.5) * 100
    
    # Z-score and p-value
    n = len(y_test)
    z_score = (accuracy - 0.5) / np.sqrt(0.25 / n)
    p_value = 1 - stats.norm.cdf(abs(z_score))
    
    # Feature importance
    importance = dict(zip(feature_names, model.feature_importance()))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        'name': name,
        'accuracy': accuracy,
        'edge': edge,
        'z_score': z_score,
        'p_value': p_value,
        'n_samples': n,
        'predictions': preds,
        'top_features': top_features,
        'model': model
    }


def run_feature_experiment(df_train, df_test, feature_sets, target_periods=[1, 5, 10, 20]):
    """Run experiments with different feature sets and target periods."""
    
    results = []
    
    for target_period in target_periods:
        print(f"\n{'='*60}")
        print(f"TARGET: {target_period}-minute ahead prediction")
        print(f"{'='*60}")
        
        # Create target
        df_train_copy = df_train.copy()
        df_test_copy = df_test.copy()
        
        df_train_copy['Target'] = df_train_copy['Close'].shift(-target_period) - df_train_copy['Close']
        df_test_copy['Target'] = df_test_copy['Close'].shift(-target_period) - df_test_copy['Close']
        
        # Remove last rows where target is NaN
        df_train_copy = df_train_copy.iloc[:-target_period].copy()
        df_test_copy = df_test_copy.iloc[:-target_period].copy()
        
        for set_name, feature_list in feature_sets.items():
            print(f"\n  Testing: {set_name}")
            
            # Get available features
            available = [f for f in feature_list if f in df_train_copy.columns]
            
            if len(available) < 5:
                print(f"    Skipping - only {len(available)} features available")
                continue
            
            # Prepare data
            train_clean = df_train_copy[available + ['Target']].dropna()
            test_clean = df_test_copy[available + ['Target']].dropna()
            
            if len(train_clean) < 10000 or len(test_clean) < 1000:
                print(f"    Skipping - insufficient samples ({len(train_clean)}/{len(test_clean)})")
                continue
            
            X_train = train_clean[available].values
            y_train = train_clean['Target'].values
            X_test = test_clean[available].values
            y_test = test_clean['Target'].values
            
            # Train and evaluate
            result = train_and_evaluate(
                X_train, y_train, X_test, y_test, 
                available, 
                name=f"{set_name} (T+{target_period})"
            )
            result['target_period'] = target_period
            result['feature_set'] = set_name
            result['n_features'] = len(available)
            
            print(f"    Accuracy: {result['accuracy']:.2%}")
            print(f"    Edge: {result['edge']:+.2f}%")
            print(f"    P-value: {result['p_value']:.6f}")
            print(f"    Significant: {'YES' if result['p_value'] < 0.05 and result['edge'] > 0 else 'NO'}")
            
            results.append(result)
    
    return results


def main():
    print("="*70)
    print("COMPREHENSIVE MINUTE-LEVEL FEATURE TEST")
    print("="*70)
    
    # Fetch years of real data
    # Training: 2020-01-01 to 2024-06-30 (4.5 years)
    # Testing: 2024-07-01 to 2024-12-31 (6 months)
    
    print("\n--- FETCHING TRAINING DATA (2020-2024) ---")
    df_train = fetch_minute_data("2020-01-01", "2024-06-30")
    
    print("\n--- FETCHING TESTING DATA (2024 H2) ---")
    df_test = fetch_minute_data("2024-07-01", "2024-12-31")
    
    print(f"\nTotal training samples: {len(df_train):,}")
    print(f"Total testing samples: {len(df_test):,}")
    
    # Add ALL features
    print("\n--- ENGINEERING FEATURES (TRAINING) ---")
    df_train = add_all_features(df_train)
    
    print("\n--- ENGINEERING FEATURES (TESTING) ---")  
    df_test = add_all_features(df_test)
    
    # Get all feature columns
    all_features = get_feature_columns(df_train)
    print(f"\nTotal features generated: {len(all_features)}")
    
    # Define feature sets to test
    feature_sets = {
        # Individual categories
        'Basic': [f for f in all_features if any(x in f for x in ['Return', 'LogReturn', 'HL_Range', 'Gap', 'Position'])],
        
        'Volume': [f for f in all_features if any(x in f for x in ['Volume', 'OBV', 'VWAP', 'Money_Flow'])],
        
        'Momentum': [f for f in all_features if any(x in f for x in ['Momentum', 'ROC', 'RSI', 'Stoch', 'Williams', 'CCI'])],
        
        'Trend': [f for f in all_features if any(x in f for x in ['SMA', 'EMA', 'MACD', 'ADX', 'DI'])],
        
        'Volatility': [f for f in all_features if any(x in f for x in ['Volatility', 'ATR', 'BB_', 'Keltner'])],
        
        'Time': [f for f in all_features if any(x in f for x in ['Hour', 'Minute', 'Time', 'Is_', 'Day', 'Minutes_Since'])],
        
        'Mean_Reversion': [f for f in all_features if any(x in f for x in ['ZScore', 'Dist_', 'Percentile'])],
        
        'Pattern': [f for f in all_features if any(x in f for x in ['Body', 'Shadow', 'Doji', 'Bullish', 'Consecutive', 'Higher', 'Lower'])],
        
        'Microstructure': [f for f in all_features if any(x in f for x in ['Spread', 'Intensity', 'Illiquidity', 'RealizedVar'])],
        
        # Combinations
        'Momentum_Volume': [f for f in all_features if any(x in f for x in ['Momentum', 'ROC', 'RSI', 'Volume', 'OBV'])],
        
        'Trend_Volatility': [f for f in all_features if any(x in f for x in ['SMA', 'EMA', 'MACD', 'ATR', 'BB_', 'Volatility'])],
        
        'Full_Technical': [f for f in all_features if any(x in f for x in 
            ['SMA', 'EMA', 'RSI', 'MACD', 'ATR', 'BB_', 'ROC', 'Momentum'])],
        
        'Time_Enhanced': [f for f in all_features if any(x in f for x in 
            ['Hour', 'Minute', 'Is_', 'RSI', 'MACD', 'ATR', 'Volume_Ratio'])],
        
        'All_Features': all_features,
    }
    
    # Print feature set sizes
    print("\nFeature sets to test:")
    for name, features in feature_sets.items():
        print(f"  {name}: {len(features)} features")
    
    # Run experiments
    print("\n" + "="*70)
    print("RUNNING EXPERIMENTS")
    print("="*70)
    
    results = run_feature_experiment(
        df_train, df_test, 
        feature_sets,
        target_periods=[1, 5, 10, 20, 60]  # 1min, 5min, 10min, 20min, 1hour
    )
    
    # Summarize results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Sort by edge
    results_sorted = sorted(results, key=lambda x: x['edge'], reverse=True)
    
    print(f"\n{'Feature Set':<30} {'Target':<8} {'Accuracy':<10} {'Edge':<10} {'P-value':<12} {'Significant'}")
    print("-" * 90)
    
    for r in results_sorted:
        sig = 'YES' if r['p_value'] < 0.05 and r['edge'] > 0 else 'NO'
        print(f"{r['feature_set']:<30} T+{r['target_period']:<5} {r['accuracy']:.2%}     {r['edge']:+.2f}%      {r['p_value']:.6f}    {sig}")
    
    # Best results
    print("\n" + "="*70)
    print("BEST RESULTS (Positive Edge with p < 0.05)")
    print("="*70)
    
    significant_positive = [r for r in results_sorted if r['edge'] > 0 and r['p_value'] < 0.05]
    
    if significant_positive:
        for r in significant_positive[:5]:
            print(f"\n{r['name']}")
            print(f"  Accuracy: {r['accuracy']:.2%}")
            print(f"  Edge: {r['edge']:+.2f}%")
            print(f"  P-value: {r['p_value']:.6f}")
            print(f"  Features: {r['n_features']}")
            print(f"  Top features: {[f[0] for f in r['top_features'][:5]]}")
    else:
        print("\nNO FEATURE COMBINATIONS SHOWED SIGNIFICANT POSITIVE EDGE")
        print("\nThis suggests that minute-level price movements are essentially")
        print("unpredictable using technical analysis features alone.")
    
    # Save results
    results_df = pd.DataFrame([{
        'feature_set': r['feature_set'],
        'target_period': r['target_period'],
        'accuracy': r['accuracy'],
        'edge': r['edge'],
        'p_value': r['p_value'],
        'n_features': r['n_features'],
        'n_samples': r['n_samples']
    } for r in results])
    
    results_df.to_csv(RESULTS_DIR / 'all_results.csv', index=False)
    print(f"\n✓ Results saved to {RESULTS_DIR / 'all_results.csv'}")
    
    # Save summary
    summary = {
        'total_experiments': len(results),
        'significant_positive': len(significant_positive),
        'best_accuracy': max(r['accuracy'] for r in results) if results else 0,
        'best_edge': max(r['edge'] for r in results) if results else 0,
        'training_samples': len(df_train),
        'testing_samples': len(df_test),
        'total_features': len(all_features),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(RESULTS_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = main()
