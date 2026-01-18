"""Create standalone price chart for worst day"""
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Paths
TRADES_FILE = Path("/Users/mihirepel/Personal_Projects/Trading/results/trading_model/detailed_trades.json")
TEST_DATA_FILE = Path("/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/results/real_minute_large_sample/SPY_minute_testing_2024-12-01_to_2024-12-31.csv")
OUTPUT_DIR = Path("/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/results/trading_model")

WORST_DAY = pd.Timestamp('2024-12-09')

# Load data
with open(TRADES_FILE, 'r') as f:
    trades_data = json.load(f)

trades_df = pd.DataFrame(trades_data)
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

minute_df = pd.read_csv(TEST_DATA_FILE)
minute_df['Datetime'] = pd.to_datetime(minute_df['Datetime'])

# Filter to worst day
day_trades = trades_df[trades_df['exit_time'].dt.date == WORST_DAY.date()].copy()
day_minute = minute_df[minute_df['Datetime'].dt.date == WORST_DAY.date()].copy()

# Create standalone chart
fig, ax = plt.subplots(figsize=(16, 9))

# Plot price
ax.plot(day_minute['Datetime'], day_minute['Close'], linewidth=3, color='black', label='SPY Price', zorder=2)

# Plot trades
for idx, trade in day_trades.iterrows():
    direction = trade['direction']
    
    if direction == 'DOWN':  # SHORT
        entry_color = '#FF6B6B'
        exit_color = '#8B0000'
        marker_entry = 'v'
    else:  # LONG
        entry_color = '#4169E1'
        exit_color = '#00008B'
        marker_entry = '^'
    
    # Entry
    ax.scatter(trade['entry_time'], trade['entry_price'], color=entry_color, s=250, 
              marker=marker_entry, zorder=4, edgecolor='black', linewidth=2.5)
    
    # Exit
    ax.scatter(trade['exit_time'], trade['exit_price'], color=exit_color, s=250, 
              marker='s', zorder=4, edgecolor='black', linewidth=2.5)
    
    # Trade line
    ax.plot([trade['entry_time'], trade['exit_time']], [trade['entry_price'], trade['exit_price']], 
            color=exit_color, linestyle='--', alpha=0.5, linewidth=2)

ax.set_title(f'Dec 9, 2024 - Worst Day Trading (45 trades, $627 profit)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('SPY Price ($)', fontsize=14)
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='black', label='SPY Price'),
    Patch(facecolor='#FF6B6B', edgecolor='black', label='SHORT Entry (▼)'),
    Patch(facecolor='#8B0000', edgecolor='black', label='SHORT Exit (■)'),
    Patch(facecolor='#4169E1', edgecolor='black', label='LONG Entry (▲)'),
    Patch(facecolor='#00008B', edgecolor='black', label='LONG Exit (■)'),
]
ax.legend(handles=legend_elements, fontsize=12, loc='upper left', framealpha=0.9)

plt.tight_layout()

output_file = OUTPUT_DIR / "worst_day_price_chart.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Standalone chart saved to {output_file}")
