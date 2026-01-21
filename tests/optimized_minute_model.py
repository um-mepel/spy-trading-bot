#!/usr/bin/env python3
"""
OPTIMIZED MINUTE-LEVEL MODEL

Uses the best features identified from comprehensive testing:
- Microstructure (53.06% accuracy, +3.06% edge)
- Mean Reversion (52.69% accuracy, +2.69% edge)  
- Basic (52.51% accuracy, +2.51% edge)

Target: 60-minute ahead prediction (best performing)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import json
import pickle

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

RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'optimized_minute_model'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path(__file__).parent.parent / 'live_trading' / 'saved_models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Target: 60 minutes ahead (best performing in tests)
TARGET_MINUTES = 60


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
# BEST FEATURES (from comprehensive testing)
# =============================================================================

def add_microstructure_features(df):
    """Microstructure features - BEST at T+60 (53.06% accuracy)."""
    df = df.copy()
    
    # Spread proxy (using high-low as approximation)
    df['Spread_Proxy'] = (df['High'] - df['Low']) / df['Close'] * 10000  # In basis points
    
    # Trade intensity (volume per price change)
    price_change = df['Close'].diff().abs()
    df['Trade_Intensity'] = df['Volume'] / (price_change + 0.01)
    
    # Amihud illiquidity ratio
    returns = df['Close'].pct_change().abs()
    df['Illiquidity'] = returns / (df['Volume'] / 1e6 + 1e-10)
    
    # Realized variance
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    for period in [5, 10, 20]:
        df[f'RealizedVar_{period}'] = (log_returns ** 2).rolling(period).sum()
    
    return df


def add_mean_reversion_features(df):
    """Mean reversion features - 2nd BEST at T+60 (52.69% accuracy)."""
    df = df.copy()
    
    # Z-scores
    for period in [10, 20, 50]:
        mean = df['Close'].rolling(period).mean()
        std = df['Close'].rolling(period).std()
        df[f'ZScore_{period}'] = (df['Close'] - mean) / (std + 1e-10)
    
    # Distance from moving average
    for period in [10, 20, 50]:
        sma = df['Close'].rolling(period).mean()
        df[f'Dist_SMA_{period}'] = (df['Close'] - sma) / sma * 100
    
    # Percentile rank
    for period in [20, 50, 100]:
        df[f'Percentile_{period}'] = df['Close'].rolling(period).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) == period else 50
        )
    
    return df


def add_basic_features(df):
    """Basic features - 3rd BEST at T+60 (52.51% accuracy)."""
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
    
    # Keltner position (identified as important)
    ema20 = df['Close'].ewm(span=20).mean()
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    keltner_upper = ema20 + 2 * atr14
    keltner_lower = ema20 - 2 * atr14
    df['Keltner_Position'] = (df['Close'] - keltner_lower) / (keltner_upper - keltner_lower + 1e-10)
    
    return df


def add_optimized_features(df):
    """Add only the best-performing features."""
    print("  Adding microstructure features...")
    df = add_microstructure_features(df)
    print("  Adding mean reversion features...")
    df = add_mean_reversion_features(df)
    print("  Adding basic features...")
    df = add_basic_features(df)
    return df


def get_feature_columns():
    """Return the optimized feature list."""
    return [
        # Microstructure (6 features)
        'Spread_Proxy', 'Trade_Intensity', 'Illiquidity',
        'RealizedVar_5', 'RealizedVar_10', 'RealizedVar_20',
        
        # Mean Reversion (9 features)
        'ZScore_10', 'ZScore_20', 'ZScore_50',
        'Dist_SMA_10', 'Dist_SMA_20', 'Dist_SMA_50',
        'Percentile_20', 'Percentile_50', 'Percentile_100',
        
        # Basic (13 features)
        'Return_1', 'Return_5', 'Return_10', 'Return_20',
        'LogReturn_1', 'LogReturn_5',
        'HL_Range', 'HL_Range_Pct', 'Close_Position',
        'Open_Close_Range', 'Gap', 'Keltner_Position'
    ]


def calculate_confidence(prediction, std_residual):
    """Calculate prediction confidence based on magnitude vs residual std."""
    abs_pred = abs(prediction)
    
    if abs_pred > std_residual * 2:
        return 0.9
    elif abs_pred > std_residual:
        return 0.7
    elif abs_pred > std_residual * 0.5:
        return 0.5
    else:
        return 0.3


def run_portfolio_simulation(df, predictions, confidences, initial_capital=10000):
    """
    Simulate trading with realistic constraints.
    
    Rules:
    - Only trade on high confidence signals (>=0.7)
    - Position size based on confidence
    - Hold for TARGET_MINUTES (60 mins)
    - Include trading costs
    """
    print("\n" + "="*60)
    print("PORTFOLIO SIMULATION")
    print("="*60)
    
    capital = initial_capital
    position = 0  # SPY shares
    entry_price = 0
    entry_idx = 0
    
    trades = []
    equity_curve = [capital]
    
    # Trading costs
    COMMISSION = 0.0  # Most brokers free now
    SLIPPAGE = 0.01  # $0.01 per share slippage
    
    for i in range(len(df) - TARGET_MINUTES):
        current_price = df['Close'].iloc[i]
        prediction = predictions[i]
        confidence = confidences[i]
        
        # Close existing position after TARGET_MINUTES
        if position != 0 and i >= entry_idx + TARGET_MINUTES:
            exit_price = current_price
            
            if position > 0:  # Long
                pnl = (exit_price - entry_price - SLIPPAGE * 2) * position
            else:  # Short
                pnl = (entry_price - exit_price - SLIPPAGE * 2) * abs(position)
            
            capital += pnl
            
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': i,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': position,
                'pnl': pnl,
                'return_pct': pnl / (abs(position) * entry_price) * 100,
                'prediction': predictions[entry_idx],
                'confidence': confidences[entry_idx],
                'correct': (prediction > 0 and exit_price > entry_price) or (prediction < 0 and exit_price < entry_price)
            })
            
            position = 0
        
        # Open new position if no current position and high confidence
        if position == 0 and confidence >= 0.7:
            # Position sizing based on confidence
            if confidence >= 0.9:
                position_pct = 0.5  # 50% of capital
            elif confidence >= 0.7:
                position_pct = 0.3  # 30% of capital
            else:
                position_pct = 0.1  # 10% of capital
            
            position_value = capital * position_pct
            shares = int(position_value / current_price)
            
            if shares > 0:
                if prediction > 0:  # Long
                    position = shares
                else:  # Short
                    position = -shares
                
                entry_price = current_price
                entry_idx = i
        
        # Track equity
        if position > 0:
            equity = capital + position * (current_price - entry_price)
        elif position < 0:
            equity = capital + abs(position) * (entry_price - current_price)
        else:
            equity = capital
        
        equity_curve.append(equity)
    
    # Close final position
    if position != 0:
        exit_price = df['Close'].iloc[-1]
        if position > 0:
            pnl = (exit_price - entry_price - SLIPPAGE * 2) * position
        else:
            pnl = (entry_price - exit_price - SLIPPAGE * 2) * abs(position)
        capital += pnl
    
    # Calculate metrics
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) > 0:
        total_return = (capital - initial_capital) / initial_capital * 100
        win_rate = trades_df['correct'].mean() * 100
        avg_return = trades_df['return_pct'].mean()
        
        # Sharpe ratio (annualized)
        returns = trades_df['return_pct'].values
        if len(returns) > 1:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 * 6.5)  # Trading hours per year
        else:
            sharpe = 0
        
        # Max drawdown
        equity_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        print(f"\nInitial Capital: ${initial_capital:,.2f}")
        print(f"Final Capital:   ${capital:,.2f}")
        print(f"Total Return:    {total_return:+.2f}%")
        print(f"\n--- TRADE STATISTICS ---")
        print(f"Total Trades:    {len(trades_df)}")
        print(f"Win Rate:        {win_rate:.1f}%")
        print(f"Avg Return/Trade:{avg_return:+.2f}%")
        print(f"Sharpe Ratio:    {sharpe:.2f}")
        print(f"Max Drawdown:    {max_drawdown:.2f}%")
        
        # By confidence
        print(f"\n--- BY CONFIDENCE ---")
        for conf in [0.7, 0.9]:
            mask = trades_df['confidence'] >= conf
            if mask.sum() > 0:
                conf_trades = trades_df[mask]
                print(f"  Conf >= {conf}: {len(conf_trades)} trades, {conf_trades['correct'].mean()*100:.1f}% win rate, {conf_trades['return_pct'].mean():+.2f}% avg return")
        
        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'trades': trades_df,
            'equity_curve': equity_curve
        }
    else:
        print("No trades executed!")
        return None


def main():
    print("="*70)
    print("OPTIMIZED MINUTE-LEVEL MODEL")
    print("Using best features: Microstructure + Mean Reversion + Basic")
    print(f"Target: {TARGET_MINUTES}-minute ahead prediction")
    print("="*70)
    
    # Fetch data
    print("\n--- FETCHING TRAINING DATA (2020-2024) ---")
    df_train = fetch_minute_data("2020-01-01", "2024-06-30")
    
    print("\n--- FETCHING TESTING DATA (2024 H2) ---")
    df_test = fetch_minute_data("2024-07-01", "2024-12-31")
    
    # Add features
    print("\n--- ENGINEERING FEATURES ---")
    df_train = add_optimized_features(df_train)
    df_test = add_optimized_features(df_test)
    
    # Get feature columns
    feature_cols = get_feature_columns()
    available = [f for f in feature_cols if f in df_train.columns]
    print(f"Using {len(available)} features")
    
    # Create target
    df_train['Target'] = df_train['Close'].shift(-TARGET_MINUTES) - df_train['Close']
    df_test['Target'] = df_test['Close'].shift(-TARGET_MINUTES) - df_test['Close']
    
    # Remove last rows
    df_train = df_train.iloc[:-TARGET_MINUTES].copy()
    df_test = df_test.iloc[:-TARGET_MINUTES].copy()
    
    # Clean data
    train_clean = df_train[available + ['Target', 'Close', 'Datetime']].dropna()
    test_clean = df_test[available + ['Target', 'Close', 'Datetime']].dropna()
    
    print(f"\nTraining samples: {len(train_clean):,}")
    print(f"Testing samples:  {len(test_clean):,}")
    
    # Prepare training data
    X_train = train_clean[available].values
    y_train = train_clean['Target'].values
    X_test = test_clean[available].values
    y_test = test_clean['Target'].values
    
    # Train model
    print("\n--- TRAINING MODEL ---")
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
    print("\n--- GENERATING PREDICTIONS ---")
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate residual std for confidence
    residuals = y_train - train_preds
    std_residual = np.std(residuals[-10000:])  # Recent residuals
    print(f"Residual std: ${std_residual:.4f}")
    
    # Calculate confidences
    confidences = np.array([calculate_confidence(p, std_residual) for p in test_preds])
    
    # Accuracy
    pred_direction = test_preds > 0
    actual_direction = y_test > 0
    accuracy = (pred_direction == actual_direction).mean()
    edge = (accuracy - 0.5) * 100
    
    print(f"\n--- MODEL PERFORMANCE ---")
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"Edge vs Random:   {edge:+.2f}%")
    
    # Confidence breakdown
    print(f"\n--- BY CONFIDENCE ---")
    for conf in [0.3, 0.5, 0.7, 0.9]:
        mask = confidences == conf
        if mask.sum() > 0:
            acc = (pred_direction[mask] == actual_direction[mask]).mean()
            print(f"  {conf}: {mask.sum():,} signals, {acc:.2%} accuracy")
    
    # High confidence
    high_conf_mask = confidences >= 0.7
    if high_conf_mask.sum() > 0:
        high_conf_acc = (pred_direction[high_conf_mask] == actual_direction[high_conf_mask]).mean()
        print(f"\nHigh Confidence (>=0.7): {high_conf_mask.sum():,} signals, {high_conf_acc:.2%} accuracy")
    
    # Statistical significance
    n = len(y_test)
    z_score = (accuracy - 0.5) / np.sqrt(0.25 / n)
    p_value = 1 - stats.norm.cdf(abs(z_score))
    print(f"\nZ-score: {z_score:.2f}")
    print(f"P-value: {p_value:.10f}")
    print(f"Statistically Significant: {'YES' if p_value < 0.05 and edge > 0 else 'NO'}")
    
    # Run portfolio simulation
    simulation = run_portfolio_simulation(
        test_clean.reset_index(drop=True),
        test_preds,
        confidences,
        initial_capital=10000
    )
    
    # Save model
    print("\n--- SAVING MODEL ---")
    with open(MODEL_DIR / 'optimized_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    model_stats = {
        'feature_cols': available,
        'target_minutes': TARGET_MINUTES,
        'std_residual': float(std_residual),
        'accuracy': float(accuracy),
        'edge': float(edge),
        'trained_at': datetime.now().isoformat(),
        'training_samples': len(X_train)
    }
    
    with open(MODEL_DIR / 'optimized_model_stats.pkl', 'wb') as f:
        pickle.dump(model_stats, f)
    
    print(f"✓ Model saved to {MODEL_DIR / 'optimized_model.pkl'}")
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'edge': float(edge),
        'z_score': float(z_score),
        'p_value': float(p_value),
        'target_minutes': TARGET_MINUTES,
        'n_features': len(available),
        'training_samples': len(X_train),
        'testing_samples': len(X_test),
        'simulation': {
            'total_return': simulation['total_return'] if simulation else 0,
            'win_rate': simulation['win_rate'] if simulation else 0,
            'total_trades': simulation['total_trades'] if simulation else 0,
            'sharpe': simulation['sharpe'] if simulation else 0
        } if simulation else None,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    predictions_df = test_clean[['Datetime', 'Close']].copy()
    predictions_df['Target'] = y_test
    predictions_df['Predicted'] = test_preds
    predictions_df['Confidence'] = confidences
    predictions_df['Direction_Correct'] = (pred_direction == actual_direction).astype(int)
    predictions_df.to_csv(RESULTS_DIR / 'predictions.csv', index=False)
    
    if simulation:
        simulation['trades'].to_csv(RESULTS_DIR / 'trades.csv', index=False)
    
    print(f"\n✓ Results saved to {RESULTS_DIR}")
    
    return model, results, simulation


if __name__ == "__main__":
    model, results, simulation = main()
