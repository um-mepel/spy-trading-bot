#!/usr/bin/env python3
"""Analyze portfolio performance over time."""
import pandas as pd
import numpy as np

df = pd.read_csv('/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/results/extended_middle_deciles/all_trades.csv')
df['Exit_Date'] = pd.to_datetime(df['Exit_Date'])
df['Return_Contribution'] = 0.04 * df['Return_Pct'] / 100

# Daily portfolio tracking
daily = df.groupby('Exit_Date')['Return_Contribution'].sum().reset_index()
daily.columns = ['Date', 'Daily_Return']
daily = daily.set_index('Date').sort_index()

# Calculate cumulative
portfolio = 100 * (1 + daily['Daily_Return']).cumprod()

print('Portfolio value over time:')
print('='*50)

# Show quarterly snapshots
for year in range(2020, 2027):
    for q, month in [(1, 3), (2, 6), (3, 9), (4, 12)]:
        try:
            date = pd.Timestamp(year=year, month=month, day=28)
            closest = portfolio.index[portfolio.index <= date]
            if len(closest) > 0:
                val = portfolio.loc[closest[-1]]
                print(f'{year} Q{q}: ${val:.2f} ({val-100:+.1f}%)')
        except:
            pass

print('\n' + '='*50)
print('Peak and Drawdown Analysis:')
print('='*50)
peak_val = portfolio.max()
peak_date = portfolio.idxmax()
final_val = portfolio.iloc[-1]
drawdown_from_peak = (final_val - peak_val) / peak_val * 100

print(f'Peak: ${peak_val:.2f} on {peak_date.date()}')
print(f'Final: ${final_val:.2f}')
print(f'Drawdown from peak: {drawdown_from_peak:.1f}%')

# Show when drawdown started
print(f'\nPortfolio values around peak:')
peak_idx = portfolio.index.get_loc(peak_date)
for i in range(-5, 15):
    try:
        idx = peak_idx + i * 20  # ~monthly
        if 0 <= idx < len(portfolio):
            d = portfolio.index[idx]
            v = portfolio.iloc[idx]
            marker = ' <-- PEAK' if d == peak_date else ''
            print(f'  {d.date()}: ${v:.2f}{marker}')
    except:
        pass
