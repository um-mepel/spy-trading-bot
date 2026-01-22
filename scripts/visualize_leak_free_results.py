#!/usr/bin/env python3
"""
Proper visualization of the leak-free backtest results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf

OUTPUT_DIR = '/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/results/high_confidence_leak_free'

# Load trades
trades = pd.read_csv(f'{OUTPUT_DIR}/trades.csv')
trades['Entry_Date'] = pd.to_datetime(trades['Entry_Date'])
trades['Exit_Date'] = pd.to_datetime(trades['Exit_Date'])

print("Calculating portfolio simulation...")

# Proper portfolio simulation with fixed position sizing
dates = pd.date_range(trades['Entry_Date'].min(), trades['Exit_Date'].max(), freq='D')
portfolio_history = []
portfolio_value = 100.0
position_size = 20.0  # $20 per position (20% of starting capital per trade)

positions = []

for date in dates:
    # Close positions that exit today
    closed_today = [p for p in positions if p['exit_date'].date() == date.date()]
    for p in closed_today:
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
    
    portfolio_history.append({
        'Date': date,
        'Value': portfolio_value,
        'Open_Positions': len(positions)
    })

# Close remaining positions
for p in positions:
    pnl = position_size * (p['return_pct'] / 100)
    portfolio_value += pnl

portfolio_df = pd.DataFrame(portfolio_history)
portfolio_df.set_index('Date', inplace=True)

# Get SPY for comparison
print("Downloading SPY data...")
spy = yf.download('SPY', start=trades['Entry_Date'].min(), end=trades['Exit_Date'].max(), progress=False)
spy_close = spy['Close'].squeeze()
spy_normalized = spy_close / spy_close.iloc[0] * 100

# Calculate metrics
total_return = (portfolio_value / 100 - 1) * 100
spy_return = (spy_close.iloc[-1] / spy_close.iloc[0] - 1) * 100
period_years = (trades['Exit_Date'].max() - trades['Entry_Date'].min()).days / 365
annualized_return = ((portfolio_value / 100) ** (1 / period_years) - 1) * 100
spy_annualized = ((1 + spy_return/100) ** (1 / period_years) - 1) * 100

# Calculate drawdown
rolling_max = portfolio_df['Value'].cummax()
drawdown = (portfolio_df['Value'] - rolling_max) / rolling_max * 100
max_drawdown = drawdown.min()

# Calculate Sharpe
daily_returns = portfolio_df['Value'].pct_change().dropna()
sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

print(f"\n{'='*60}")
print("CORRECTED RESULTS - HIGH CONFIDENCE STOCK PICKER")
print(f"{'='*60}")
print(f"Period: {trades['Entry_Date'].min().date()} to {trades['Exit_Date'].max().date()} ({period_years:.1f} years)")
print(f"\nStrategy Total Return: {total_return:.2f}%")
print(f"Strategy Annualized Return: {annualized_return:.2f}%")
print(f"\nSPY Total Return: {spy_return:.2f}%")
print(f"SPY Annualized Return: {spy_annualized:.2f}%")
print(f"\nAlpha (Total): {total_return - spy_return:.2f}%")
print(f"Alpha (Annualized): {annualized_return - spy_annualized:.2f}%")
print(f"\nSharpe Ratio: {sharpe:.3f}")
print(f"Max Drawdown: {max_drawdown:.2f}%")
print(f"\nTotal Trades: {len(trades)}")
print(f"Win Rate: {trades['Win'].mean()*100:.2f}%")
print(f"Avg Return/Trade: {trades['Return_Pct'].mean():.3f}%")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Cumulative Returns
ax1 = axes[0, 0]
ax1.plot(portfolio_df.index, portfolio_df['Value'], 'b-', linewidth=2, label=f'Strategy ({total_return:.1f}%)')
ax1.plot(spy_normalized.index, spy_normalized.values, 'gray', linewidth=2, alpha=0.7, label=f'SPY ({spy_return:.1f}%)')
ax1.axhline(y=100, color='black', linestyle='--', alpha=0.3)
ax1.set_title('Cumulative Returns: Strategy vs SPY (3 Years, Leak-Free)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Portfolio Value ($)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# 2. Drawdown
ax2 = axes[0, 1]
ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
ax2.plot(drawdown.index, drawdown.values, 'r-', linewidth=1)
ax2.axhline(y=max_drawdown, color='darkred', linestyle='--', label=f'Max DD: {max_drawdown:.1f}%')
ax2.set_title('Strategy Drawdown', fontsize=12, fontweight='bold')
ax2.set_ylabel('Drawdown (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

# 3. Win Rate by Confidence Decile
ax3 = axes[1, 0]
trades['Confidence_Decile'] = pd.qcut(trades['Confidence'], q=10, labels=[f'D{i}' for i in range(1, 11)])
accuracy_by_decile = trades.groupby('Confidence_Decile', observed=False)['Win'].mean() * 100
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 10))
bars = ax3.bar(range(10), accuracy_by_decile.values, color=colors, edgecolor='black')
ax3.axhline(y=50, color='gray', linestyle='--', label='Random (50%)')
ax3.axhline(y=trades['Win'].mean()*100, color='blue', linestyle='-', label=f'Overall ({trades["Win"].mean()*100:.1f}%)')
ax3.set_title('Win Rate by Confidence Decile', fontsize=12, fontweight='bold')
ax3.set_ylabel('Win Rate (%)')
ax3.set_xlabel('Confidence Decile (D10 = Highest)')
ax3.set_xticks(range(10))
ax3.set_xticklabels([f'D{i}' for i in range(1, 11)])
ax3.legend()
ax3.set_ylim(40, 65)
for bar, acc in zip(bars, accuracy_by_decile.values):
    ax3.annotate(f'{acc:.0f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=8)

# 4. Summary Stats
ax4 = axes[1, 1]
ax4.axis('off')

# Determine verdict
if total_return > spy_return:
    verdict = "✅ STRATEGY BEATS SPY"
    verdict_color = "green"
else:
    verdict = "❌ SPY OUTPERFORMS"
    verdict_color = "red"

summary_text = f"""
    HIGH CONFIDENCE STOCK PICKER - LEAK-FREE BACKTEST
    {'='*50}
    
    Period: {trades['Entry_Date'].min().date()} to {trades['Exit_Date'].max().date()}
    Duration: {period_years:.1f} years
    
    RETURNS:
    ├─ Strategy Total: {total_return:+.2f}%
    ├─ Strategy Annualized: {annualized_return:+.2f}%
    ├─ SPY Total: {spy_return:+.2f}%
    ├─ SPY Annualized: {spy_annualized:+.2f}%
    └─ Alpha: {total_return - spy_return:+.2f}%
    
    RISK METRICS:
    ├─ Sharpe Ratio: {sharpe:.3f}
    └─ Max Drawdown: {max_drawdown:.2f}%
    
    TRADE STATISTICS:
    ├─ Total Trades: {len(trades)}
    ├─ Win Rate: {trades['Win'].mean()*100:.2f}%
    └─ Avg Return/Trade: {trades['Return_Pct'].mean():.3f}%
    
    METHODOLOGY:
    ├─ Top 5 picks per day
    ├─ 5-day hold period  
    ├─ Monthly model retraining
    └─ STRICT NO LEAKAGE
    
    {verdict}
"""

ax4.text(0.05, 0.5, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/corrected_results.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to {OUTPUT_DIR}/corrected_results.png")
plt.show()

# Update summary.json
import json
summary = {
    'period': f"{trades['Entry_Date'].min().date()} to {trades['Exit_Date'].max().date()}",
    'period_years': round(period_years, 1),
    'strategy_return': round(total_return, 2),
    'strategy_annualized': round(annualized_return, 2),
    'spy_return': round(spy_return, 2),
    'spy_annualized': round(spy_annualized, 2),
    'alpha': round(total_return - spy_return, 2),
    'alpha_annualized': round(annualized_return - spy_annualized, 2),
    'sharpe_ratio': round(sharpe, 3),
    'max_drawdown': round(max_drawdown, 2),
    'total_trades': len(trades),
    'win_rate': round(trades['Win'].mean() * 100, 2),
    'avg_return_per_trade': round(trades['Return_Pct'].mean(), 3),
    'methodology': {
        'top_n_per_day': 5,
        'hold_days': 5,
        'retrain_frequency': 'monthly',
        'data_leakage': 'NONE'
    }
}

with open(f'{OUTPUT_DIR}/summary_corrected.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved to {OUTPUT_DIR}/summary_corrected.json")
