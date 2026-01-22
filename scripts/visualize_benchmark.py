#!/usr/bin/env python3
"""Visualize the cherry-picked benchmark issue with the high confidence stock picker."""

import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Download data
print("Downloading SPY data...")
spy = yf.download('SPY', start='2025-01-01', end='2026-01-21', progress=False)
close = spy['Close'].squeeze()

# Key dates
test_start = '2025-04-10'
low_date = close.idxmin()

# Calculate returns
full_year_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
from_apr_return = (close.iloc[-1] / close.loc[test_start] - 1) * 100
strategy_return = 14.18

print(f"Full year return: {full_year_return:.2f}%")
print(f"From April return: {from_apr_return:.2f}%")
print(f"Strategy return: {strategy_return:.2f}%")

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: SPY Price with test period highlighted
ax1.plot(close.index, close.values, 'b-', linewidth=1.5, label='SPY')
ax1.axvline(pd.Timestamp(test_start), color='red', linestyle='--', linewidth=2, label='Test Start (Apr 10)')
ax1.axvline(low_date, color='orange', linestyle=':', linewidth=2, label='Yearly Low (Apr 8)')

# Shade the test period
ax1.axvspan(pd.Timestamp(test_start), close.index[-1], alpha=0.2, color='red', label='Test Period')

# Mark key points
ax1.scatter([close.index[0]], [close.iloc[0]], color='green', s=100, zorder=5)
ax1.scatter([low_date], [close.min()], color='orange', s=100, zorder=5)
ax1.scatter([pd.Timestamp(test_start)], [close.loc[test_start]], color='red', s=100, zorder=5)
ax1.scatter([close.index[-1]], [close.iloc[-1]], color='blue', s=100, zorder=5)

# Annotations
ax1.annotate(f'Jan 2: ${close.iloc[0]:.0f}', (close.index[0], close.iloc[0]), 
             textcoords="offset points", xytext=(10, 10), fontsize=10, fontweight='bold')
ax1.annotate(f'Low: ${close.min():.0f}', (low_date, close.min()), 
             textcoords="offset points", xytext=(10, -20), fontsize=10, fontweight='bold', color='orange')
ax1.annotate(f'Test Start: ${close.loc[test_start]:.0f}', (pd.Timestamp(test_start), close.loc[test_start]), 
             textcoords="offset points", xytext=(10, 10), fontsize=10, fontweight='bold', color='red')
ax1.annotate(f'Jan 2026: ${close.iloc[-1]:.0f}', (close.index[-1], close.iloc[-1]), 
             textcoords="offset points", xytext=(-80, -20), fontsize=10, fontweight='bold')

ax1.set_title('SPY 2025: Test Period Started Near Yearly Low', fontsize=14, fontweight='bold')
ax1.set_ylabel('Price ($)', fontsize=12)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

# Plot 2: Return comparison bar chart
returns = [full_year_return, from_apr_return, strategy_return]
labels = ['SPY Full Year\n(Jan → Jan)', 'SPY from Apr Low\n(Cherry-picked)', 'Strategy\nReturn']
colors = ['#2ecc71', '#e74c3c', '#3498db']

bars = ax2.bar(labels, returns, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, val in zip(bars, returns):
    ax2.annotate(f'+{val:.1f}%', 
                 xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add alpha annotations
ax2.annotate('Misleading Alpha: -19.5%', xy=(1.5, 25), fontsize=11, color='red', 
             ha='center', style='italic')
ax2.annotate('Actual Alpha: ~-3%', xy=(1.5, 22), fontsize=11, color='green', 
             ha='center', fontweight='bold')

ax2.set_title('Return Comparison: Cherry-Picked vs Fair Benchmark', fontsize=14, fontweight='bold')
ax2.set_ylabel('Return (%)', fontsize=12)
ax2.set_ylim(0, 40)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = '/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/results/high_confidence_stock_picker/benchmark_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'Saved to {output_path}')
plt.show()
