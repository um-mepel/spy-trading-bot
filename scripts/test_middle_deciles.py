#!/usr/bin/env python3
"""
Test using only middle confidence deciles (D4-D7) for trades.
These showed better accuracy than the extreme confidence levels.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json

OUTPUT_DIR = '/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/results/high_confidence_leak_free'

# Load trades
trades = pd.read_csv(f'{OUTPUT_DIR}/trades.csv')
trades['Entry_Date'] = pd.to_datetime(trades['Entry_Date'])
trades['Exit_Date'] = pd.to_datetime(trades['Exit_Date'])

# Assign deciles
trades['Confidence_Decile'] = pd.qcut(trades['Confidence'], q=10, labels=[f'D{i}' for i in range(1, 11)])

# Show accuracy by decile
print("Accuracy by Confidence Decile (all trades):")
acc_by_decile = trades.groupby('Confidence_Decile', observed=False).agg({
    'Win': ['mean', 'count'],
    'Return_Pct': 'mean'
}).round(3)
acc_by_decile.columns = ['Win_Rate', 'Count', 'Avg_Return']
acc_by_decile['Win_Rate'] = (acc_by_decile['Win_Rate'] * 100).round(1)
print(acc_by_decile.to_string())
print()

# Filter to only D4-D7 (middle confidence)
middle_deciles = ['D4', 'D5', 'D6', 'D7']
filtered_trades = trades[trades['Confidence_Decile'].isin(middle_deciles)].copy()

print(f"Filtered to deciles {middle_deciles}")
print(f"Original trades: {len(trades)}")
print(f"Filtered trades: {len(filtered_trades)}")
print(f"Filtered win rate: {filtered_trades['Win'].mean()*100:.2f}%")
print(f"Filtered avg return: {filtered_trades['Return_Pct'].mean():.3f}%")
print()

# Count trades per day after filtering
trades_per_day = filtered_trades.groupby('Entry_Date').size()
avg_trades_per_day = trades_per_day.mean()
max_trades_per_day = trades_per_day.max()
print(f"Avg trades per day (filtered): {avg_trades_per_day:.1f}")
print(f"Max trades per day (filtered): {max_trades_per_day}")

# Position sizing: with ~2 trades/day and 5-day hold, we have ~10 concurrent positions
# So each position is 10% of portfolio
avg_concurrent = avg_trades_per_day * 5
POSITION_SIZE_PCT = 100 / avg_concurrent
print(f"Estimated concurrent positions: {avg_concurrent:.0f}")
print(f"Position size: {POSITION_SIZE_PCT:.1f}%")
print()

# Simulate portfolio
dates = pd.date_range(filtered_trades['Entry_Date'].min(), filtered_trades['Exit_Date'].max(), freq='D')
portfolio_value = 100.0
portfolio_history = []
positions = []

for date in dates:
    # Close positions that exit today
    for p in [pos for pos in positions if pos['exit_date'].date() == date.date()]:
        pnl = p['initial_value'] * (p['return_pct'] / 100)
        portfolio_value += pnl
    
    positions = [p for p in positions if p['exit_date'].date() != date.date()]
    
    # Add new positions
    day_trades = filtered_trades[filtered_trades['Entry_Date'].dt.date == date.date()]
    for _, trade in day_trades.iterrows():
        position_value = portfolio_value * (POSITION_SIZE_PCT / 100)
        positions.append({
            'exit_date': trade['Exit_Date'],
            'return_pct': trade['Return_Pct'],
            'initial_value': position_value
        })
    
    portfolio_history.append({'Date': date, 'Value': portfolio_value})

# Close remaining
for p in positions:
    pnl = p['initial_value'] * (p['return_pct'] / 100)
    portfolio_value += pnl

portfolio_df = pd.DataFrame(portfolio_history)
portfolio_df.set_index('Date', inplace=True)

# Get SPY
spy = yf.download('SPY', start=filtered_trades['Entry_Date'].min(), end=filtered_trades['Exit_Date'].max(), progress=False)
spy_close = spy['Close'].squeeze()
spy_return = (spy_close.iloc[-1] / spy_close.iloc[0] - 1) * 100
spy_normalized = spy_close / spy_close.iloc[0] * 100

# Metrics
total_return = (portfolio_value / 100 - 1) * 100
period_years = (filtered_trades['Exit_Date'].max() - filtered_trades['Entry_Date'].min()).days / 365
annualized = ((portfolio_value / 100) ** (1 / period_years) - 1) * 100
spy_annualized = ((1 + spy_return/100) ** (1 / period_years) - 1) * 100

# Drawdown
rolling_max = portfolio_df['Value'].cummax()
drawdown = (portfolio_df['Value'] - rolling_max) / rolling_max * 100
max_dd = drawdown.min()

# Sharpe
daily_returns = portfolio_df['Value'].pct_change().dropna()
sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

print("=" * 60)
print("RESULTS: MIDDLE DECILES ONLY (D4-D7)")
print("=" * 60)
print(f"Period: {filtered_trades['Entry_Date'].min().date()} to {filtered_trades['Exit_Date'].max().date()} ({period_years:.1f} years)")
print()
print(f"Strategy Total Return: {total_return:.2f}%")
print(f"Strategy Annualized: {annualized:.2f}%")
print()
print(f"SPY Total Return: {spy_return:.2f}%")
print(f"SPY Annualized: {spy_annualized:.2f}%")
print()
print(f"Alpha (Total): {total_return - spy_return:.2f}%")
print(f"Alpha (Annualized): {annualized - spy_annualized:.2f}%")
print()
print(f"Sharpe Ratio: {sharpe:.3f}")
print(f"Max Drawdown: {max_dd:.2f}%")
print()
print(f"Total Trades: {len(filtered_trades)}")
print(f"Win Rate: {filtered_trades['Win'].mean()*100:.2f}%")

verdict = "STRATEGY BEATS SPY" if total_return > spy_return else "SPY OUTPERFORMS"
print(f"\nVERDICT: {verdict}")

# Also test with ALL trades for comparison
print("\n" + "=" * 60)
print("COMPARISON: ALL DECILES vs MIDDLE DECILES")
print("=" * 60)

# Run all trades simulation
all_portfolio = 100.0
all_positions = []
all_history = []
ALL_POSITION_SIZE = 4.0  # 25 concurrent positions

for date in dates:
    for p in [pos for pos in all_positions if pos['exit_date'].date() == date.date()]:
        pnl = p['initial_value'] * (p['return_pct'] / 100)
        all_portfolio += pnl
    all_positions = [p for p in all_positions if p['exit_date'].date() != date.date()]
    
    day_trades = trades[trades['Entry_Date'].dt.date == date.date()]
    for _, trade in day_trades.iterrows():
        position_value = all_portfolio * (ALL_POSITION_SIZE / 100)
        all_positions.append({
            'exit_date': trade['Exit_Date'],
            'return_pct': trade['Return_Pct'],
            'initial_value': position_value
        })
    all_history.append({'Date': date, 'Value': all_portfolio})

for p in all_positions:
    pnl = p['initial_value'] * (p['return_pct'] / 100)
    all_portfolio += pnl

all_return = (all_portfolio / 100 - 1) * 100
all_df = pd.DataFrame(all_history).set_index('Date')

print(f"{'Metric':<25} {'All Deciles':<15} {'D4-D7 Only':<15} {'SPY':<15}")
print("-" * 70)
print(f"{'Total Return':<25} {all_return:>13.2f}% {total_return:>13.2f}% {spy_return:>13.2f}%")
print(f"{'Win Rate':<25} {trades['Win'].mean()*100:>13.2f}% {filtered_trades['Win'].mean()*100:>13.2f}% {'N/A':>15}")
print(f"{'Avg Return/Trade':<25} {trades['Return_Pct'].mean():>13.3f}% {filtered_trades['Return_Pct'].mean():>13.3f}% {'N/A':>15}")
print(f"{'Total Trades':<25} {len(trades):>14} {len(filtered_trades):>14} {'N/A':>15}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Cumulative returns comparison
ax1 = axes[0, 0]
ax1.plot(portfolio_df.index, portfolio_df['Value'], 'b-', linewidth=2, label=f'D4-D7 Only ({total_return:.1f}%)')
ax1.plot(all_df.index, all_df['Value'], 'g--', linewidth=2, alpha=0.7, label=f'All Deciles ({all_return:.1f}%)')
ax1.plot(spy_normalized.index, spy_normalized.values, 'gray', linewidth=2, alpha=0.5, label=f'SPY ({spy_return:.1f}%)')
ax1.axhline(y=100, color='black', linestyle='--', alpha=0.3)
ax1.set_title('Middle Confidence (D4-D7) vs All Deciles vs SPY', fontsize=12, fontweight='bold')
ax1.set_ylabel('Portfolio Value ($)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# 2. Accuracy by decile with D4-D7 highlighted
ax2 = axes[0, 1]
all_acc = trades.groupby('Confidence_Decile', observed=False)['Win'].mean() * 100
colors = ['lightgray'] * 3 + ['green'] * 4 + ['lightgray'] * 3  # Highlight D4-D7
bars = ax2.bar(range(10), all_acc.values, color=colors, edgecolor='black')
ax2.axhline(y=50, color='gray', linestyle='--', label='Random')
ax2.axhline(y=filtered_trades['Win'].mean()*100, color='blue', linestyle='-', linewidth=2, label=f'D4-D7 Avg ({filtered_trades["Win"].mean()*100:.1f}%)')
ax2.set_title('Win Rate by Decile (D4-D7 Selected)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Win Rate (%)')
ax2.set_xlabel('Confidence Decile')
ax2.set_xticks(range(10))
ax2.set_xticklabels([f'D{i}' for i in range(1, 11)])
ax2.legend()
ax2.set_ylim(45, 60)
for i, (bar, acc) in enumerate(zip(bars, all_acc.values)):
    ax2.annotate(f'{acc:.0f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=9, fontweight='bold' if 3 <= i <= 6 else 'normal')

# 3. Drawdown comparison
ax3 = axes[1, 0]
all_rolling_max = all_df['Value'].cummax()
all_drawdown = (all_df['Value'] - all_rolling_max) / all_rolling_max * 100
ax3.fill_between(drawdown.index, drawdown.values, 0, color='blue', alpha=0.3, label=f'D4-D7 (Max: {max_dd:.1f}%)')
ax3.fill_between(all_drawdown.index, all_drawdown.values, 0, color='green', alpha=0.2, label=f'All (Max: {all_drawdown.min():.1f}%)')
ax3.set_title('Drawdown Comparison', fontsize=12, fontweight='bold')
ax3.set_ylabel('Drawdown (%)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

# 4. Summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
MIDDLE CONFIDENCE STRATEGY (D4-D7 Only)
{'='*50}

HYPOTHESIS: Middle confidence predictions may be
more reliable than extreme high/low confidence.

RESULTS:
                   D4-D7 Only    All Deciles    SPY
  Total Return:    {total_return:+7.2f}%      {all_return:+7.2f}%    {spy_return:+7.2f}%
  Win Rate:        {filtered_trades['Win'].mean()*100:7.2f}%      {trades['Win'].mean()*100:7.2f}%       N/A
  Trades:          {len(filtered_trades):7}        {len(trades):7}       N/A

POSITION SIZING:
  D4-D7: ~{avg_trades_per_day:.0f} trades/day x 5 days = ~{avg_concurrent:.0f} positions
         Position size: {POSITION_SIZE_PCT:.1f}%
  
  All:   ~5 trades/day x 5 days = ~25 positions
         Position size: 4%

VERDICT: {verdict}
Alpha vs SPY: {total_return - spy_return:+.2f}%
"""

ax4.text(0.05, 0.5, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen' if total_return > spy_return else 'lightyellow', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/middle_deciles_results.png', dpi=150, bbox_inches='tight')
print(f"\nSaved to {OUTPUT_DIR}/middle_deciles_results.png")
plt.show()

# Save summary
summary = {
    'strategy': 'Middle Confidence (D4-D7)',
    'deciles_used': middle_deciles,
    'position_size_pct': round(POSITION_SIZE_PCT, 1),
    'strategy_return': round(total_return, 2),
    'all_deciles_return': round(all_return, 2),
    'spy_return': round(spy_return, 2),
    'alpha': round(total_return - spy_return, 2),
    'sharpe': round(sharpe, 3),
    'max_drawdown': round(max_dd, 2),
    'total_trades': len(filtered_trades),
    'win_rate': round(filtered_trades['Win'].mean() * 100, 2),
    'avg_return_per_trade': round(filtered_trades['Return_Pct'].mean(), 3),
    'verdict': verdict
}

with open(f'{OUTPUT_DIR}/middle_deciles_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
