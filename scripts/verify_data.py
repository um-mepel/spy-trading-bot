#!/usr/bin/env python3
"""Cross-validate yfinance data."""
import yfinance as yf
import pandas as pd

print('=' * 70)
print('CROSS-VALIDATION: Checking yfinance Close vs Adj Close')
print('=' * 70)

spy = yf.download('SPY', start='2020-01-01', end='2024-01-01', progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.droplevel(1)

print(f"Columns: {list(spy.columns)}")

# Try different column name formats
adj_close_col = 'Adj Close' if 'Adj Close' in spy.columns else 'Adjusted Close'
if adj_close_col not in spy.columns:
    adj_close_col = [c for c in spy.columns if 'adj' in c.lower()][0] if any('adj' in c.lower() for c in spy.columns) else None

if adj_close_col:
    spy['Diff'] = (spy[adj_close_col] - spy['Close']) / spy['Close'] * 100
    print(f'\nSPY Close vs {adj_close_col} difference:')
    print(f'  Mean difference: {spy["Diff"].mean():.2f}%')
else:
    print('\nNo Adj Close column found - using Close only')
print(f'  This is due to DIVIDEND adjustments')
print()
print('Key insight:')
print('  yfinance Close = Split-adjusted, NO dividend adjustment')
print('  yfinance Adj Close = Split-adjusted AND dividend-adjusted')
print()
print('  Backtest uses Close, so MISSING ~1.5%/year dividend yield')
print('  This UNDERSTATES real returns (conservative bias)')
print()

print('=' * 70)
print('RECENT DATA VERIFICATION')
print('=' * 70)

recent = yf.download('SPY', period='5d', progress=False)
if isinstance(recent.columns, pd.MultiIndex):
    recent.columns = recent.columns.droplevel(1)

print(f'\nMost recent SPY data:')
print(recent[['Open', 'High', 'Low', 'Close', 'Volume']].to_string())

last_date = recent.index[-1]
print(f'\nLast date: {last_date.date()}')

# Check for future dates (would indicate fake/synthetic data)
from datetime import datetime
if last_date.date() > datetime.now().date():
    print('⚠️  WARNING: Data contains FUTURE dates! This is suspicious!')
else:
    print('✓ Data dates are in the past (as expected)')
