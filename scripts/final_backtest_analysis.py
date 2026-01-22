#!/usr/bin/env python3
"""Recalculate with proper position sizing (no implicit leverage)."""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json

OUTPUT_DIR = '/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/results/high_confidence_leak_free'

trades = pd.read_csv(f'{OUTPUT_DIR}/trades.csv')
trades['Entry_Date'] = pd.to_datetime(trades['Entry_Date'])
trades['Exit_Date'] = pd.to_datetime(trades['Exit_Date'])

# Check position sizing
positions_per_day = trades.groupby('Entry_Date').size()
print(f"Avg positions entered per day: {positions_per_day.mean():.1f}")
print(f"Max positions entered per day: {positions_per_day.max()}")
print()

# With 5-day hold and 5 entries/day, we have ~25 concurrent positions
# Each position should be 4% of capital (100/25 = 4%)
POSITION_SIZE_PCT = 4.0  # 4% per position for no leverage

# Simulate portfolio properly
dates = pd.date_range(trades['Entry_Date'].min(), trades['Exit_Date'].max(), freq='D')
portfolio_value = 100.0
portfolio_history = []
positions = []

for date in dates:
    # Close positions that exit today
    for p in [pos for pos in positions if pos['exit_date'].date() == date.date()]:
        # Position value at close = initial_value * (1 + return)
        pnl = p['initial_value'] * (p['return_pct'] / 100)
        portfolio_value += pnl
    
    # Remove closed positions
    positions = [p for p in positions if p['exit_date'].date() != date.date()]
    
    # Add new positions (each is 4% of CURRENT portfolio value)
    day_trades = trades[trades['Entry_Date'].dt.date == date.date()]
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
spy = yf.download('SPY', start=trades['Entry_Date'].min(), end=trades['Exit_Date'].max(), progress=False)
spy_close = spy['Close'].squeeze()
spy_return = (spy_close.iloc[-1] / spy_close.iloc[0] - 1) * 100
spy_normalized = spy_close / spy_close.iloc[0] * 100

# Calculate metrics
total_return = (portfolio_value / 100 - 1) * 100
period_years = (trades['Exit_Date'].max() - trades['Entry_Date'].min()).days / 365
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
print("CORRECTED RESULTS (4% position sizing, no leverage)")
print("=" * 60)
print(f"Period: {trades['Entry_Date'].min().date()} to {trades['Exit_Date'].max().date()} ({period_years:.1f} years)")
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
print(f"Total Trades: {len(trades)}")
print(f"Win Rate: {trades['Win'].mean()*100:.2f}%")

# Determine winner
if total_return > spy_return:
    verdict = "STRATEGY BEATS SPY"
else:
    verdict = "SPY OUTPERFORMS"
print()
print(f"VERDICT: {verdict}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Cumulative Returns
ax1 = axes[0, 0]
ax1.plot(portfolio_df.index, portfolio_df['Value'], 'b-', linewidth=2, label=f'Strategy ({total_return:.1f}%)')
ax1.plot(spy_normalized.index, spy_normalized.values, 'gray', linewidth=2, alpha=0.7, label=f'SPY ({spy_return:.1f}%)')
ax1.axhline(y=100, color='black', linestyle='--', alpha=0.3)
ax1.set_title('High Confidence Stock Picker vs SPY (Leak-Free, No Leverage)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Portfolio Value ($)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# 2. Drawdown  
ax2 = axes[0, 1]
ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
ax2.plot(drawdown.index, drawdown.values, 'r-', linewidth=1)
ax2.axhline(y=max_dd, color='darkred', linestyle='--', label=f'Max DD: {max_dd:.1f}%')
ax2.set_title('Strategy Drawdown', fontsize=12, fontweight='bold')
ax2.set_ylabel('Drawdown (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

# 3. Win Rate by Confidence  
ax3 = axes[1, 0]
trades['Confidence_Decile'] = pd.qcut(trades['Confidence'], q=10, labels=[f'D{i}' for i in range(1,11)])
acc = trades.groupby('Confidence_Decile', observed=False)['Win'].mean() * 100
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 10))
ax3.bar(range(10), acc.values, color=colors, edgecolor='black')
ax3.axhline(y=50, color='gray', linestyle='--', label='Random')
ax3.axhline(y=trades['Win'].mean()*100, color='blue', linestyle='-', label=f'Overall ({trades["Win"].mean()*100:.1f}%)')
ax3.set_title('Win Rate by Confidence Decile', fontsize=12, fontweight='bold')
ax3.set_ylabel('Win Rate (%)')
ax3.set_xlabel('Confidence (D10 = Highest)')
ax3.set_xticks(range(10))
ax3.set_xticklabels([f'D{i}' for i in range(1, 11)])
ax3.legend()
ax3.set_ylim(40, 65)

# 4. Summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
HIGH CONFIDENCE STOCK PICKER
Leak-Free 3-Year Backtest (No Leverage)
{'='*45}

Period: {trades['Entry_Date'].min().date()} to {trades['Exit_Date'].max().date()}

RETURNS:
  Strategy Total:      {total_return:+.2f}%
  Strategy Annualized: {annualized:+.2f}%
  SPY Total:           {spy_return:+.2f}%
  SPY Annualized:      {spy_annualized:+.2f}%
  Alpha:               {total_return - spy_return:+.2f}%

RISK:
  Sharpe Ratio:  {sharpe:.3f}
  Max Drawdown:  {max_dd:.2f}%

TRADES:
  Total:           {len(trades)}
  Win Rate:        {trades['Win'].mean()*100:.2f}%
  Avg Return:      {trades['Return_Pct'].mean():.3f}%

METHODOLOGY:
  - 5 picks/day, 5-day hold
  - 4% position size (no leverage)
  - Monthly retraining
  - NO DATA LEAKAGE

VERDICT: {verdict}
"""

ax4.text(0.05, 0.5, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow' if total_return > spy_return else 'lightcoral', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/final_results.png', dpi=150, bbox_inches='tight')
print(f"\nSaved to {OUTPUT_DIR}/final_results.png")
plt.show()

# Save summary
summary = {
    'period': f"{trades['Entry_Date'].min().date()} to {trades['Exit_Date'].max().date()}",
    'years': round(period_years, 1),
    'position_size_pct': POSITION_SIZE_PCT,
    'strategy_return': round(total_return, 2),
    'strategy_annualized': round(annualized, 2),
    'spy_return': round(spy_return, 2),
    'spy_annualized': round(spy_annualized, 2),
    'alpha': round(total_return - spy_return, 2),
    'sharpe': round(sharpe, 3),
    'max_drawdown': round(max_dd, 2),
    'total_trades': len(trades),
    'win_rate': round(trades['Win'].mean() * 100, 2),
    'avg_return_per_trade': round(trades['Return_Pct'].mean(), 3),
    'verdict': verdict
}

with open(f'{OUTPUT_DIR}/final_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
