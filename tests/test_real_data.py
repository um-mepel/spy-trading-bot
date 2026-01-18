"""
Test Trading Model on Real Alpha Vantage Daily Data
====================================================

This script tests your minute-level trading model on REAL market data
from Alpha Vantage (converted to minute bars with realistic intraday patterns).

WARNING: The intraday bars are simulated from daily data, so minute-level
patterns may not reflect true intraday behavior. But daily price movements
are 100% real market data.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from lightgbm import LGBMRegressor

def add_technical_indicators(df):
    """Calculate technical indicators at minute granularity."""
    df = df.copy()
    
    print("Calculating technical indicators...")
    
    # Simple Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['SMA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['SMA_60'] = df['Close'].rolling(window=60, min_periods=1).mean()
    
    # Exponential Moving Averages
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
    df['ROC_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
    df['ROC_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)
    
    # Bollinger Bands
    bb_middle = df['Close'].rolling(window=20, min_periods=1).mean()
    bb_std = df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_Upper'] = bb_middle + (bb_std * 2)
    df['BB_Lower'] = bb_middle - (bb_std * 2)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)
    
    # Average True Range
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14, min_periods=1).mean()
    
    # Volatility
    df['Returns'] = df['Close'].pct_change()
    df['Volatility_10'] = df['Returns'].rolling(window=10, min_periods=1).std()
    
    # Additional features
    df['HL_Range'] = df['High'] - df['Low']
    df['HL_Range_Pct'] = df['HL_Range'] / df['Close']
    df['CO_Range_Pct'] = np.abs(df['Close'] - df['Open']) / df['Close']
    
    return df


def run_optimized_backtest(data, initial_cash, position_size_pct, confidence_threshold,
                          profit_target, stop_loss, max_holding_minutes, commission):
    """Simple backtest implementation."""
    cash = initial_cash
    position = None
    trades = []
    equity_curve = [{'timestamp': data.index[0], 'equity': cash}]
    
    for i in range(len(data)):
        row = data.iloc[i]
        
        # Exit existing position
        if position is not None:
            minutes_held = i - position['entry_idx']
            current_price = row['Close']
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            
            if position['direction'] == 'SHORT':
                pnl_pct = -pnl_pct
            
            # Check exit conditions
            should_exit = False
            exit_reason = None
            
            if pnl_pct >= profit_target:
                should_exit = True
                exit_reason = 'PROFIT_TARGET'
            elif pnl_pct <= -stop_loss:
                should_exit = True
                exit_reason = 'STOP_LOSS'
            elif minutes_held >= max_holding_minutes:
                should_exit = True
                exit_reason = 'TIME_LIMIT'
            
            if should_exit:
                shares = position['shares']
                if position['direction'] == 'LONG':
                    pnl = (current_price - position['entry_price']) * shares
                else:  # SHORT
                    pnl = (position['entry_price'] - current_price) * shares
                
                pnl_after_commission = pnl - (commission * current_price * shares * 2)
                cash += pnl_after_commission
                
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': row.name,
                    'direction': position['direction'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'shares': shares,
                    'pnl': pnl_after_commission,
                    'minutes_held': minutes_held,
                    'exit_reason': exit_reason
                })
                
                position = None
                equity_curve.append({'timestamp': row.name, 'equity': cash})
        
        # Enter new position
        if position is None and i < len(data) - max_holding_minutes:
            prediction = row['prediction']
            confidence = row['confidence']
            
            if confidence >= confidence_threshold:
                direction = 'LONG' if prediction > 0 else 'SHORT'
                shares = int((cash * position_size_pct) / row['Close'])
                
                if shares > 0:
                    position = {
                        'entry_time': row.name,
                        'entry_idx': i,
                        'entry_price': row['Close'],
                        'direction': direction,
                        'shares': shares
                    }
    
    return {'trades': trades, 'equity_curve': equity_curve}


def main():
    """Run the trading model on real data."""
    
    print("\n" + "="*80)
    print("TRADING MODEL TEST - REAL ALPHA VANTAGE DATA")
    print("="*80)
    print("Data Source: Alpha Vantage SPY daily (Aug 2025 - Jan 2026)")
    print("Minute bars: Simulated from real daily OHLC")
    print("="*80 + "\n")
    
    # Load real data
    train_file = Path("data/SPY_training_real_data.csv")
    test_file = Path("data/SPY_testing_real_data.csv")
    
    print("Loading training data...")
    df_train = pd.read_csv(train_file, index_col=0, parse_dates=True)
    print(f"  Training: {len(df_train):,} bars ({df_train.index[0].date()} to {df_train.index[-1].date()})")
    
    print("Loading testing data...")
    df_test = pd.read_csv(test_file, index_col=0, parse_dates=True)
    print(f"  Testing:  {len(df_test):,} bars ({df_test.index[0].date()} to {df_test.index[-1].date()})")
    
    # Calculate technical indicators
    print("\nCalculating technical indicators...")
    df_train_features = add_technical_indicators(df_train)
    df_test_features = add_technical_indicators(df_test)
    
    print(f"  Training features: {len(df_train_features.columns)} columns")
    print(f"  Testing features:  {len(df_test_features.columns)} columns")
    
    # Drop rows with NaN from indicator calculation
    df_train_features = df_train_features.dropna()
    df_test_features = df_test_features.dropna()
    
    print(f"  After dropna - Training: {len(df_train_features):,} bars")
    print(f"  After dropna - Testing:  {len(df_test_features):,} bars")
    
    # Train model
    print("\nTraining LightGBM model...")
    
    # Define features and target
    feature_cols = [col for col in df_train_features.columns 
                   if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'future_return_20', 'Returns']]
    
    # Create target: 20-period forward return
    df_train_features['future_return_20'] = df_train_features['Close'].shift(-20) / df_train_features['Close'] - 1
    df_train_features = df_train_features.dropna()
    
    X_train = df_train_features[feature_cols]
    y_train = df_train_features['future_return_20']
    
    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    print(f"  ✓ Model trained on {len(X_train):,} samples")
    
    # Make predictions on test set
    print("\nGenerating predictions...")
    X_test = df_test_features[feature_cols]
    predictions = model.predict(X_test)
    
    df_test_features['prediction'] = predictions
    df_test_features['confidence'] = np.abs(predictions)
    
    print(f"  Predictions: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")
    
    # Run backtest with optimized strategy
    print("\nRunning backtest with optimized strategy...")
    
    results = run_optimized_backtest(
        data=df_test_features,
        initial_cash=100000,
        position_size_pct=0.1,  # 10% of capital per trade
        confidence_threshold=0.005,  # 0.5% minimum prediction
        profit_target=0.005,  # 0.5% profit target
        stop_loss=0.005,  # 0.5% stop loss
        max_holding_minutes=20,
        commission=0.001  # 0.1% commission
    )
    
    # Display results
    print("\n" + "="*80)
    print("BACKTEST RESULTS - REAL DATA")
    print("="*80)
    
    trades = results['trades']
    equity_curve = results['equity_curve']
    
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        
        # Calculate statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                           trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else float('inf')
        
        final_equity = equity_curve[-1]['equity']
        total_return = ((final_equity - 100000) / 100000) * 100
        
        print(f"Total Trades:     {total_trades}")
        print(f"Winning Trades:   {winning_trades} ({win_rate:.1f}%)")
        print(f"Losing Trades:    {losing_trades} ({100-win_rate:.1f}%)")
        print(f"Win Rate:         {win_rate:.1f}%")
        print(f"\nProfit Factor:    {profit_factor:.2f}x")
        print(f"Average Win:      ${avg_win:.2f}")
        print(f"Average Loss:     ${avg_loss:.2f}")
        print(f"\nTotal P&L:        ${total_pnl:,.2f}")
        print(f"Total Return:     {total_return:.2f}%")
        print(f"Final Equity:     ${final_equity:,.2f}")
        
        # Save results
        output_dir = Path("results/real_data_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        trades_df.to_csv(output_dir / "trades_real_data.csv", index=False)
        
        equity_df = pd.DataFrame(equity_curve)
        equity_df.to_csv(output_dir / "equity_curve_real_data.csv", index=False)
        
        # Save summary
        summary = {
            "data_source": "Alpha Vantage SPY Daily (Aug 2025 - Jan 2026)",
            "test_period": f"{df_test.index[0].date()} to {df_test.index[-1].date()}",
            "total_bars": len(df_test),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor if profit_factor != float('inf') else None,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_pnl": total_pnl,
            "total_return_pct": total_return,
            "final_equity": final_equity
        }
        
        with open(output_dir / "summary_real_data.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_dir}")
        
    else:
        print("✗ No trades generated")
        print("  Model may not be generating signals on this data")
        print("  Try lowering confidence_threshold or adjusting parameters")
    
    print("\n" + "="*80)
    print("IMPORTANT NOTES:")
    print("="*80)
    print("1. Daily prices are REAL market data from Alpha Vantage")
    print("2. Minute bars are SIMULATED from daily OHLC (not true intraday data)")
    print("3. Results show how model performs on real daily price movements")
    print("4. For true validation, need real minute-level data (requires premium API)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
