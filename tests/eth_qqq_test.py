"""
ETH & QQQ Detailed Test
=======================
Detailed analysis and portfolio simulation for ETH and QQQ,
which showed significant edges in multi-asset testing.

Results:
- ETH: +20.09% return, 79.31% accuracy on top 10% signals
- QQQ: +7.51% edge but underperformed buy & hold in bull market

Usage:
    python tests/eth_qqq_test.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import lightgbm as lgb
import warnings
import json
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

RESULTS_DIR = Path('results/eth_qqq_test')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def add_features(df):
    """Add technical features using only past data."""
    df = df.copy()
    
    # Microstructure
    df['Spread_Proxy'] = (df['High'] - df['Low']) / df['Close'] * 10000
    price_change = df['Close'].diff().abs()
    df['Trade_Intensity'] = df['Volume'] / (price_change + 0.01)
    returns = df['Close'].pct_change().abs()
    df['Illiquidity'] = returns / (df['Volume'] / 1e6 + 1e-10)
    
    # Mean Reversion
    for period in [10, 20, 50]:
        mean = df['Close'].rolling(period).mean()
        std = df['Close'].rolling(period).std()
        df[f'ZScore_{period}'] = (df['Close'] - mean) / (std + 1e-10)
        df[f'Dist_SMA_{period}'] = (df['Close'] - mean) / mean * 100
    
    for period in [20, 50]:
        df[f'Percentile_{period}'] = df['Close'].rolling(period).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) == period else 50
        )
    
    # Basic
    df['Return_1'] = df['Close'].pct_change()
    df['Return_5'] = df['Close'].pct_change(5)
    df['Return_10'] = df['Close'].pct_change(10)
    df['Return_20'] = df['Close'].pct_change(20)
    df['LogReturn_1'] = np.log(df['Close'] / df['Close'].shift(1))
    df['HL_Range_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Open_Close_Range'] = (df['Close'] - df['Open']) / df['Open'] * 100
    
    return df


def test_and_simulate(ticker, target_days=20):
    """Test and simulate trading for an asset."""
    print(f"\n{'='*60}")
    print(f"TESTING {ticker}")
    print(f"{'='*60}")
    
    # Fetch data
    df = yf.download(ticker, period="max", interval="1d", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"Total data: {len(df)} days")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Add features
    df = add_features(df)
    
    # Add target
    df['Target'] = df['Close'].shift(-target_days) - df['Close']
    df = df.iloc[:-target_days]
    
    feature_cols = [
        'Spread_Proxy', 'Trade_Intensity', 'Illiquidity',
        'ZScore_10', 'ZScore_20', 'ZScore_50',
        'Dist_SMA_10', 'Dist_SMA_20', 'Dist_SMA_50',
        'Percentile_20', 'Percentile_50',
        'Return_1', 'Return_5', 'Return_10', 'Return_20',
        'LogReturn_1', 'HL_Range_Pct', 'Close_Position', 'Open_Close_Range'
    ]
    
    df_clean = df[feature_cols + ['Target', 'Close', 'Date']].dropna()
    
    # Split with buffer
    buffer = 50
    split_idx = int(len(df_clean) * 0.8)
    train = df_clean.iloc[:split_idx - buffer].copy()
    test = df_clean.iloc[split_idx:].copy()
    
    print(f"\nTraining: {len(train)} samples")
    print(f"Testing: {len(test)} samples")
    
    X_train = train[feature_cols].values
    y_train = train['Target'].values
    X_test = test[feature_cols].values
    y_test = test['Target'].values
    
    # Train
    train_set = lgb.Dataset(X_train, label=y_train)
    model = lgb.train({
        'objective': 'regression',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbosity': -1
    }, train_set, 200)
    
    # Evaluate
    preds = model.predict(X_test)
    abs_preds = np.abs(preds)
    
    pred_dir = preds > 0
    actual_dir = y_test > 0
    accuracy = (pred_dir == actual_dir).mean()
    edge = (accuracy - 0.5) * 100
    
    n = len(y_test)
    z_score = (accuracy - 0.5) / np.sqrt(0.25 / n)
    p_value = 1 - stats.norm.cdf(abs(z_score))
    
    print(f"\n--- MODEL PERFORMANCE ---")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Edge: {edge:+.2f}%")
    print(f"P-Value: {p_value:.10f}")
    print(f"Significant: {'YES' if p_value < 0.05 else 'NO'}")
    
    # By percentile
    print(f"\n--- BY SIGNAL STRENGTH ---")
    for pct in [50, 70, 90]:
        threshold = np.percentile(abs_preds, pct)
        mask = abs_preds >= threshold
        if mask.sum() > 0:
            acc = (pred_dir[mask] == actual_dir[mask]).mean()
            print(f"  Top {100-pct}%: {mask.sum()} signals, {acc:.2%} accuracy")
    
    # Portfolio simulation
    print(f"\n--- PORTFOLIO SIMULATION ---")
    initial_capital = 10000
    signal_threshold = np.percentile(abs_preds, 70)
    
    capital = initial_capital
    position = 0
    entry_price = 0
    entry_idx = 0
    trades = []
    
    test = test.reset_index(drop=True)
    
    for i in range(len(test) - target_days):
        current_price = test['Close'].iloc[i]
        pred = preds[i]
        
        # Close position after hold period
        if position != 0 and i >= entry_idx + target_days:
            exit_price = current_price
            if position > 0:
                pnl = position * (exit_price - entry_price) * 0.998
                ret = (exit_price - entry_price) / entry_price * 100
            else:
                pnl = abs(position) * (entry_price - exit_price) * 0.998
                ret = (entry_price - exit_price) / entry_price * 100
            
            capital += pnl
            trades.append({
                'entry_date': str(test['Date'].iloc[entry_idx].date()),
                'exit_date': str(test['Date'].iloc[i].date()),
                'direction': 'Long' if position > 0 else 'Short',
                'return_pct': ret,
                'pnl': pnl,
                'won': ret > 0
            })
            position = 0
        
        # Open new position
        if position == 0 and abs(pred) >= signal_threshold:
            pos_size = 0.3
            units = (capital * pos_size) / current_price
            position = units if pred > 0 else -units
            entry_price = current_price
            entry_idx = i
    
    # Close final position
    if position != 0:
        exit_price = test['Close'].iloc[-1]
        if position > 0:
            pnl = position * (exit_price - entry_price) * 0.998
        else:
            pnl = abs(position) * (entry_price - exit_price) * 0.998
        capital += pnl
    
    trades_df = pd.DataFrame(trades)
    
    # Buy and hold
    bh_start = test['Close'].iloc[0]
    bh_end = test['Close'].iloc[-1]
    bh_return = (bh_end - bh_start) / bh_start * 100
    
    print(f"\nStarting Capital: ${initial_capital:,.2f}")
    print(f"Final Capital: ${capital:,.2f}")
    print(f"Strategy Return: {(capital - initial_capital) / initial_capital * 100:+.2f}%")
    print(f"Buy & Hold: {bh_return:+.2f}%")
    
    if len(trades_df) > 0:
        print(f"\nTrades: {len(trades_df)}")
        print(f"Win Rate: {trades_df['won'].mean() * 100:.1f}%")
        
        # Save trades
        trades_df.to_csv(RESULTS_DIR / f'{ticker.lower().replace("-", "_")}_trades.csv', index=False)
    
    return {
        'ticker': ticker,
        'accuracy': float(accuracy),
        'edge': float(edge),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
        'strategy_return': float((capital - initial_capital) / initial_capital * 100),
        'buy_hold_return': float(bh_return),
        'trades': len(trades_df),
        'win_rate': float(trades_df['won'].mean() * 100) if len(trades_df) > 0 else 0
    }


def main():
    print("=" * 70)
    print("ETH & QQQ DETAILED TEST")
    print("=" * 70)
    
    results = []
    for ticker in ['ETH-USD', 'QQQ']:
        result = test_and_simulate(ticker)
        results.append(result)
    
    # Save summary
    with open(RESULTS_DIR / 'summary.json', 'w') as f:
        json.dump({
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Ticker':<10} {'Accuracy':>10} {'Edge':>10} {'Strategy':>12} {'B&H':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['ticker']:<10} {r['accuracy']:>9.2%} {r['edge']:>+9.2f}% {r['strategy_return']:>+11.2f}% {r['buy_hold_return']:>+9.2f}%")
    
    print(f"\n✓ Results saved to {RESULTS_DIR}")


if __name__ == '__main__':
    main()
