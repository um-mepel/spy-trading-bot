"""
ETH Strict Leak-Free Test
==========================
Rigorous test of ETH daily data prediction with explicit leak prevention.

Results: 63.54% accuracy, +13.54% edge (statistically significant)
However, portfolio simulation underperforms buy & hold due to:
- Too few trades (18 in 480 days)
- Shorting in bull market
- High volatility per trade

Usage:
    python tests/eth_strict_leak_free_test.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yfinance as yf
from scipy import stats
import lightgbm as lgb
import warnings
import json
warnings.filterwarnings('ignore')

RESULTS_DIR = Path('results/eth_strict_test')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def add_features_no_leak(df):
    """
    Add features using ONLY past data (no forward-looking).
    All rolling calculations look backward only.
    """
    df = df.copy()
    
    # === MICROSTRUCTURE (all backward-looking) ===
    df['Spread_Proxy'] = (df['High'] - df['Low']) / df['Close'] * 10000
    price_change = df['Close'].diff().abs()
    df['Trade_Intensity'] = df['Volume'] / (price_change + 0.01)
    returns = df['Close'].pct_change().abs()
    df['Illiquidity'] = returns / (df['Volume'] / 1e6 + 1e-10)
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    for period in [5, 10, 20]:
        df[f'RealizedVar_{period}'] = (log_returns ** 2).rolling(period).sum()
    
    # === MEAN REVERSION (all backward-looking) ===
    for period in [10, 20, 50]:
        mean = df['Close'].rolling(period).mean()
        std = df['Close'].rolling(period).std()
        df[f'ZScore_{period}'] = (df['Close'] - mean) / (std + 1e-10)
        df[f'Dist_SMA_{period}'] = (df['Close'] - mean) / mean * 100
    
    for period in [20, 50, 100]:
        df[f'Percentile_{period}'] = df['Close'].rolling(period).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) == period else 50
        )
    
    # === BASIC PRICE FEATURES (all backward-looking) ===
    df['Return_1'] = df['Close'].pct_change()
    df['Return_5'] = df['Close'].pct_change(5)
    df['Return_10'] = df['Close'].pct_change(10)
    df['Return_20'] = df['Close'].pct_change(20)
    df['LogReturn_1'] = np.log(df['Close'] / df['Close'].shift(1))
    df['LogReturn_5'] = np.log(df['Close'] / df['Close'].shift(5))
    df['HL_Range'] = df['High'] - df['Low']
    df['HL_Range_Pct'] = df['HL_Range'] / df['Close'] * 100
    df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Open_Close_Range'] = (df['Close'] - df['Open']) / df['Open'] * 100
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
    
    # Keltner position
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


def simulate_portfolio(preds, test_clean, y_test, long_only=False, threshold_pct=70, 
                       position_pct=0.3, hold_days=20):
    """Simulate trading portfolio."""
    threshold = np.percentile(np.abs(preds), threshold_pct)
    capital = 10000
    position = 0
    entry_price = 0
    entry_idx = 0
    trades = []
    daily_equity = []
    
    for i in range(len(test_clean) - hold_days):
        current_price = test_clean['Close'].iloc[i]
        current_date = test_clean['Date'].iloc[i]
        pred = preds[i]
        
        # Calculate current equity
        if position > 0:
            equity = capital + position * (current_price - entry_price)
        elif position < 0:
            equity = capital + abs(position) * (entry_price - current_price)
        else:
            equity = capital
        
        daily_equity.append({
            'date': current_date,
            'equity': equity,
            'position': 'Long' if position > 0 else ('Short' if position < 0 else 'Cash')
        })
        
        # Close position after hold period
        if position != 0 and i >= entry_idx + hold_days:
            exit_price = current_price
            fee_pct = 0.001
            
            if position > 0:
                pnl = position * (exit_price - entry_price) * (1 - fee_pct * 2)
                ret_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl = abs(position) * (entry_price - exit_price) * (1 - fee_pct * 2)
                ret_pct = (entry_price - exit_price) / entry_price * 100
            
            capital += pnl
            
            trades.append({
                'entry_date': test_clean['Date'].iloc[entry_idx],
                'exit_date': current_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': 'Long' if position > 0 else 'Short',
                'pnl': pnl,
                'return_pct': ret_pct,
                'won': ret_pct > 0
            })
            position = 0
        
        # Open new position
        if position == 0 and abs(pred) >= threshold:
            if long_only and pred < 0:
                continue
            
            units = (capital * position_pct) / current_price
            position = units if pred > 0 else -units
            entry_price = current_price
            entry_idx = i
    
    # Close final position
    if position != 0:
        exit_price = test_clean['Close'].iloc[-1]
        fee_pct = 0.001
        if position > 0:
            pnl = position * (exit_price - entry_price) * (1 - fee_pct * 2)
        else:
            pnl = abs(position) * (entry_price - exit_price) * (1 - fee_pct * 2)
        capital += pnl
    
    return capital, pd.DataFrame(trades), pd.DataFrame(daily_equity)


def main():
    print("=" * 70)
    print("ETH STRICT LEAK-FREE TEST")
    print("=" * 70)
    
    print("""
LEAK PREVENTION MEASURES:
1. Train/test split with NO overlap
2. Features calculated SEPARATELY for train and test
3. Target uses ONLY future data within each set
4. NO information from test set used in training
5. Indicators use ONLY past data (no forward-looking)
""")
    
    # ==========================================================================
    # STEP 1: FETCH ALL REAL DATA
    # ==========================================================================
    print("\n" + "=" * 50)
    print("STEP 1: FETCHING REAL DATA")
    print("=" * 50)
    
    df_raw = yf.download('ETH-USD', period="max", interval="1d", progress=False)
    
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    
    df_raw = df_raw.reset_index()
    df_raw = df_raw.rename(columns={'Date': 'Date'})
    df_raw = df_raw[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    df_raw = df_raw.sort_values('Date').reset_index(drop=True)
    
    print(f"\nTotal data: {len(df_raw)} days")
    print(f"Date range: {df_raw['Date'].min().date()} to {df_raw['Date'].max().date()}")
    print(f"Price range: ${df_raw['Close'].min():.2f} - ${df_raw['Close'].max():.2f}")
    
    # ==========================================================================
    # STEP 2: STRICT TRAIN/TEST SPLIT
    # ==========================================================================
    print("\n" + "=" * 50)
    print("STEP 2: STRICT TRAIN/TEST SPLIT")
    print("=" * 50)
    
    BUFFER_DAYS = 100
    TARGET_DAYS = 20
    
    split_idx = int(len(df_raw) * 0.8)
    df_train_raw = df_raw.iloc[:split_idx - BUFFER_DAYS].copy()
    df_test_raw = df_raw.iloc[split_idx:].copy()
    
    print(f"\nTraining data: {len(df_train_raw)} days")
    print(f"  From: {df_train_raw['Date'].min().date()}")
    print(f"  To:   {df_train_raw['Date'].max().date()}")
    print(f"\nBuffer: {BUFFER_DAYS} days (discarded)")
    print(f"\nTesting data: {len(df_test_raw)} days")
    print(f"  From: {df_test_raw['Date'].min().date()}")
    print(f"  To:   {df_test_raw['Date'].max().date()}")
    print(f"\n✓ NO OVERLAP between train and test")
    
    # ==========================================================================
    # STEP 3: ADD FEATURES SEPARATELY
    # ==========================================================================
    print("\n" + "=" * 50)
    print("STEP 3: CALCULATING FEATURES (LEAK-FREE)")
    print("=" * 50)
    
    print("\nCalculating features for TRAINING data...")
    df_train = add_features_no_leak(df_train_raw)
    
    print("Calculating features for TESTING data...")
    df_test = add_features_no_leak(df_test_raw)
    
    print("✓ Features calculated SEPARATELY (no leakage)")
    
    # ==========================================================================
    # STEP 4: CREATE TARGET
    # ==========================================================================
    print("\n" + "=" * 50)
    print("STEP 4: CREATING TARGET")
    print("=" * 50)
    
    df_train['Target'] = df_train['Close'].shift(-TARGET_DAYS) - df_train['Close']
    df_test['Target'] = df_test['Close'].shift(-TARGET_DAYS) - df_test['Close']
    
    df_train = df_train.iloc[:-TARGET_DAYS].copy()
    df_test = df_test.iloc[:-TARGET_DAYS].copy()
    
    print(f"Target: {TARGET_DAYS}-day forward price change")
    print(f"Training samples after target: {len(df_train)}")
    print(f"Testing samples after target: {len(df_test)}")
    
    # ==========================================================================
    # STEP 5: PREPARE CLEAN DATA
    # ==========================================================================
    print("\n" + "=" * 50)
    print("STEP 5: PREPARING CLEAN DATA")
    print("=" * 50)
    
    feature_cols = [
        'Spread_Proxy', 'Trade_Intensity', 'Illiquidity',
        'RealizedVar_5', 'RealizedVar_10', 'RealizedVar_20',
        'ZScore_10', 'ZScore_20', 'ZScore_50',
        'Dist_SMA_10', 'Dist_SMA_20', 'Dist_SMA_50',
        'Percentile_20', 'Percentile_50', 'Percentile_100',
        'Return_1', 'Return_5', 'Return_10', 'Return_20',
        'LogReturn_1', 'LogReturn_5',
        'HL_Range', 'HL_Range_Pct', 'Close_Position',
        'Open_Close_Range', 'Gap', 'Keltner_Position'
    ]
    
    train_clean = df_train[feature_cols + ['Target', 'Close', 'Date']].dropna().reset_index(drop=True)
    test_clean = df_test[feature_cols + ['Target', 'Close', 'Date']].dropna().reset_index(drop=True)
    
    print(f"\nClean training samples: {len(train_clean)}")
    print(f"Clean testing samples: {len(test_clean)}")
    
    # ==========================================================================
    # STEP 6: TRAIN MODEL
    # ==========================================================================
    print("\n" + "=" * 50)
    print("STEP 6: TRAINING MODEL")
    print("=" * 50)
    
    X_train = train_clean[feature_cols].values
    y_train = train_clean['Target'].values
    X_test = test_clean[feature_cols].values
    y_test = test_clean['Target'].values
    
    train_set = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbosity': -1
    }
    
    model = lgb.train(params, train_set, num_boost_round=200)
    print("\n✓ Model trained on TRAINING data only")
    
    # ==========================================================================
    # STEP 7: EVALUATE
    # ==========================================================================
    print("\n" + "=" * 50)
    print("STEP 7: EVALUATION (NO LEAKAGE)")
    print("=" * 50)
    
    preds = model.predict(X_test)
    abs_preds = np.abs(preds)
    
    pred_dir = preds > 0
    actual_dir = y_test > 0
    accuracy = (pred_dir == actual_dir).mean()
    edge = (accuracy - 0.5) * 100
    
    n = len(y_test)
    z_score = (accuracy - 0.5) / np.sqrt(0.25 / n)
    p_value = 1 - stats.norm.cdf(abs(z_score))
    
    print(f"\n{'=' * 40}")
    print("MODEL PERFORMANCE (LEAK-FREE)")
    print(f"{'=' * 40}")
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Edge vs Random:   {edge:+.2f}%")
    print(f"Z-score:          {z_score:.2f}")
    print(f"P-value:          {p_value:.10f}")
    print(f"Significant:      {'YES' if p_value < 0.05 and edge > 0 else 'NO'}")
    
    print(f"\n--- BY PREDICTION STRENGTH ---")
    for pct in [50, 60, 70, 80, 90, 95]:
        threshold = np.percentile(abs_preds, pct)
        mask = abs_preds >= threshold
        if mask.sum() > 0:
            acc = (pred_dir[mask] == actual_dir[mask]).mean()
            print(f"  Top {100 - pct:>2}%: {mask.sum():>3} signals, {acc:.2%} accuracy")
    
    # ==========================================================================
    # STEP 8: PORTFOLIO SIMULATION
    # ==========================================================================
    print("\n" + "=" * 50)
    print("STEP 8: PORTFOLIO SIMULATION")
    print("=" * 50)
    
    capital, trades_df, equity_df = simulate_portfolio(
        preds, test_clean, y_test, 
        long_only=False, threshold_pct=70, position_pct=0.3
    )
    
    bh_start = test_clean['Close'].iloc[0]
    bh_end = test_clean['Close'].iloc[-1]
    bh_return = (bh_end - bh_start) / bh_start * 100
    
    print(f"\nStarting Capital:  $10,000.00")
    print(f"Final Capital:     ${capital:,.2f}")
    print(f"Total Return:      {(capital - 10000) / 10000 * 100:+.2f}%")
    
    if len(trades_df) > 0:
        print(f"\nTotal Trades:      {len(trades_df)}")
        print(f"Win Rate:          {trades_df['won'].mean() * 100:.1f}%")
        print(f"Avg Return/Trade:  {trades_df['return_pct'].mean():+.2f}%")
    
    print(f"\n--- VS BUY & HOLD ---")
    print(f"Strategy:          {(capital - 10000) / 10000 * 100:+.2f}%")
    print(f"ETH Buy & Hold:    {bh_return:+.2f}%")
    
    # ==========================================================================
    # STEP 9: SAVE RESULTS
    # ==========================================================================
    print("\n" + "=" * 50)
    print("STEP 9: SAVING RESULTS")
    print("=" * 50)
    
    trades_df.to_csv(RESULTS_DIR / 'trades.csv', index=False)
    equity_df.to_csv(RESULTS_DIR / 'equity_curve.csv', index=False)
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    importance.to_csv(RESULTS_DIR / 'feature_importance.csv', index=False)
    
    summary = {
        'asset': 'ETH-USD',
        'data_source': 'Yahoo Finance (REAL)',
        'leak_prevention': {
            'train_test_buffer_days': BUFFER_DAYS,
            'features_calculated_separately': True,
            'no_forward_looking_features': True
        },
        'training': {
            'samples': len(train_clean),
            'start': str(train_clean['Date'].min().date()),
            'end': str(train_clean['Date'].max().date())
        },
        'testing': {
            'samples': len(test_clean),
            'start': str(test_clean['Date'].min().date()),
            'end': str(test_clean['Date'].max().date())
        },
        'model_performance': {
            'accuracy': float(accuracy),
            'edge': float(edge),
            'z_score': float(z_score),
            'p_value': float(p_value),
            'statistically_significant': bool(p_value < 0.05 and edge > 0)
        },
        'portfolio': {
            'initial_capital': 10000,
            'final_capital': float(capital),
            'total_return': float((capital - 10000) / 10000 * 100),
            'buy_hold_return': float(bh_return),
            'total_trades': len(trades_df),
            'win_rate': float(trades_df['won'].mean() * 100) if len(trades_df) > 0 else 0
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(RESULTS_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ All results saved to {RESULTS_DIR}")
    print("\n" + "=" * 70)
    print("LEAK-FREE TEST COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
