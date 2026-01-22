#!/usr/bin/env python3
"""Verify yfinance data is real, not synthetic."""
import yfinance as yf
import pandas as pd
from datetime import datetime

print('='*70)
print('CRITICAL CHECK: Is yfinance returning real or synthetic data?')
print('='*70)

spy = yf.download('SPY', start='2025-01-01', end='2026-01-22', progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.droplevel(1)

print(f'\n2025-2026 SPY data points: {len(spy)}')
print(f'Date range: {spy.index[0].date()} to {spy.index[-1].date()}')

returns = spy['Close'].pct_change().dropna()
print(f'\nDaily return stats:')
print(f'  Mean: {returns.mean()*100:.3f}%')
print(f'  Std: {returns.std()*100:.3f}%')
print(f'  Min: {returns.min()*100:.2f}%')
print(f'  Max: {returns.max()*100:.2f}%')

if returns.std() < 0.005:
    print('\n⚠️  WARNING: Volatility too low - may be synthetic!')
elif returns.std() > 0.03:
    print('\n⚠️  WARNING: Volatility too high - may be synthetic!')
else:
    print('\n✓ Volatility looks realistic for SPY (~1% daily)')

big_moves = abs(returns) > 0.02
print(f'\nDays with >2% moves: {big_moves.sum()}')
if big_moves.sum() == 0 and len(returns) > 100:
    print('⚠️  WARNING: No big moves is suspicious!')
else:
    print('✓ Has realistic big move days')

print('\nRecent closing prices:')
print(spy['Close'].tail(10).to_string())

print('\n' + '='*70)
print('IMPORTANT NOTE ABOUT DATA DATES')
print('='*70)
print(f"""
Current date according to system: {datetime.now().date()}
Last data point: {spy.index[-1].date()}

If you're seeing dates in the "future" (like 2025-2026),
this could mean:
1. Your system clock is wrong
2. yfinance is returning synthetic/test data
3. The dates shown are actually correct and we're in 2026

Please verify manually that today's date is correct!
""")
