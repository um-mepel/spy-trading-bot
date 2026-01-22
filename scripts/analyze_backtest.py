#!/usr/bin/env python3
"""Analyze the backtest results and fix return calculation."""

import pandas as pd
import numpy as np

# Load trades
trades = pd.read_csv('/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/results/high_confidence_leak_free/trades.csv')
trades['Entry_Date'] = pd.to_datetime(trades['Entry_Date'])
trades['Exit_Date'] = pd.to_datetime(trades['Exit_Date'])

print("Sample trades:")
print(trades.head(10).to_string())
print()

print(f"Total trades: {len(trades)}")
print(f"Avg return per trade: {trades['Return_Pct'].mean():.3f}%")
print(f"Win rate: {trades['Win'].mean()*100:.2f}%")

# The issue: we're compounding daily returns but trades overlap
# Each day we enter 5 new trades that exit 5 days later
# We need to properly account for capital allocation

# Method 1: Simple average of all trade returns (no compounding)
simple_avg = trades['Return_Pct'].mean()
total_entries = trades['Entry_Date'].nunique()
print(f"\nNumber of entry days: {total_entries}")

# Method 2: Proper portfolio simulation
# On each day, we allocate 20% to each of 5 positions (5-day hold)
# So at any time, we have ~25 positions (5 days x 5 per day)
# Each position is 1/25 = 4% of capital

# Group by entry date
daily_returns = trades.groupby('Entry_Date')['Return_Pct'].mean()
print(f"Avg daily return (5 positions): {daily_returns.mean():.3f}%")

# Since we hold for 5 days, and enter 5 per day, we have overlapping positions
# The return on exit day for a position is divided by 5 (hold period) for daily contribution
# This is approximate but more realistic

# Method 3: Simulate actual portfolio
dates = sorted(trades['Entry_Date'].unique())
portfolio_value = 100.0
position_size = 20.0  # 20% per position

positions = []  # Track open positions

for date in pd.date_range(dates[0], dates[-1], freq='D'):
    # Close positions that exit today
    closed_today = [p for p in positions if p['exit_date'].date() == date.date()]
    for p in closed_today:
        # Add P&L to portfolio
        pnl = position_size * (p['return_pct'] / 100)
        portfolio_value += pnl
    
    # Remove closed positions
    positions = [p for p in positions if p['exit_date'].date() != date.date()]
    
    # Add new positions
    day_trades = trades[trades['Entry_Date'].dt.date == date.date()]
    for _, trade in day_trades.iterrows():
        positions.append({
            'exit_date': trade['Exit_Date'],
            'return_pct': trade['Return_Pct']
        })

# Close any remaining positions
for p in positions:
    pnl = position_size * (p['return_pct'] / 100)
    portfolio_value += pnl

total_return = (portfolio_value / 100 - 1) * 100
print(f"\nPortfolio simulation (fixed $20 per position):")
print(f"Final value: ${portfolio_value:.2f}")
print(f"Total return: {total_return:.2f}%")

# Time period
period_days = (trades['Exit_Date'].max() - trades['Entry_Date'].min()).days
years = period_days / 365
print(f"\nPeriod: {period_days} days ({years:.1f} years)")
print(f"Annualized return: {(((portfolio_value/100) ** (1/years)) - 1) * 100:.2f}%")

# Compare to SPY
import yfinance as yf
spy = yf.download('SPY', start=trades['Entry_Date'].min(), end=trades['Exit_Date'].max(), progress=False)
spy_close = spy['Close'].squeeze()
spy_return = (spy_close.iloc[-1] / spy_close.iloc[0] - 1) * 100
print(f"\nSPY return over same period: {spy_return:.2f}%")
print(f"Alpha: {total_return - spy_return:.2f}%")

# Accuracy by confidence
print("\n\nAccuracy by confidence decile:")
trades['Confidence_Decile'] = pd.qcut(trades['Confidence'], q=10, labels=['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10'])
acc_by_decile = trades.groupby('Confidence_Decile').agg({
    'Win': 'mean',
    'Return_Pct': 'mean',
    'Confidence': ['min', 'max', 'count']
}).round(3)
print(acc_by_decile.to_string())
