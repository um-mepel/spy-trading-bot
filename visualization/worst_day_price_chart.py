"""
Worst Trading Day - Price Chart Only
====================================

Clean, focused visualization of the worst day's price action and all trades.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Paths
TRADES_FILE = Path("/Users/mihirepel/Personal_Projects/Trading/results/trading_model/detailed_trades.json")
TEST_DATA_FILE = Path("/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/results/real_minute_large_sample/SPY_minute_testing_2024-12-01_to_2024-12-31.csv")
OUTPUT_DIR = Path("/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/results/trading_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Worst day from analysis
WORST_DAY = pd.Timestamp('2024-12-09')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def load_data():
    """Load trades and minute test data."""
    
    # Load trades
    with open(TRADES_FILE, 'r') as f:
        trades_data = json.load(f)
    
    trades_df = pd.DataFrame(trades_data)
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    
    # Load minute data
    minute_df = pd.read_csv(TEST_DATA_FILE)
    minute_df['Datetime'] = pd.to_datetime(minute_df['Datetime'])
    
    return trades_df, minute_df


def create_price_chart_only(day_trades, day_minute):
    """Create clean price chart with all trades."""
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Plot price
    ax.plot(day_minute['Datetime'], day_minute['Close'], linewidth=3, color='black', label='SPY Price', zorder=2)
    ax.fill_between(day_minute['Datetime'], day_minute['Close'].min() - 1, day_minute['Close'], 
                    alpha=0.05, color='gray', zorder=1)
    
    # Plot all trades with clear SHORT vs LONG distinction
    short_entries = []
    short_exits = []
    long_entries = []
    long_exits = []
    
    if len(day_trades) > 0:
        for idx, trade in day_trades.iterrows():
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            entry_time = trade['entry_time']
            exit_time = trade['exit_time']
            direction = trade['direction']
            
            if direction == 'DOWN':  # SHORT
                short_entries.append((entry_time, entry_price))
                short_exits.append((exit_time, exit_price))
            else:  # UP (LONG)
                long_entries.append((entry_time, entry_price))
                long_exits.append((exit_time, exit_price))
            
            # Trade line (thin, subtle)
            if direction == 'DOWN':
                line_color = '#FF6B6B'  # Light red for SHORT
            else:
                line_color = '#4169E1'  # Blue for LONG
            
            ax.plot([entry_time, exit_time], [entry_price, exit_price], 
                   color=line_color, linestyle='--', alpha=0.4, linewidth=1.5, zorder=3)
    
    # Plot SHORT entries (red down triangles)
    if short_entries:
        short_e_times = [t[0] for t in short_entries]
        short_e_prices = [t[1] for t in short_entries]
        ax.scatter(short_e_times, short_e_prices, color='#FF4444', s=300, marker='v', 
                  zorder=5, edgecolor='darkred', linewidth=2, label='SHORT Entry', alpha=0.9)
    
    # Plot SHORT exits (dark red squares)
    if short_exits:
        short_x_times = [t[0] for t in short_exits]
        short_x_prices = [t[1] for t in short_exits]
        ax.scatter(short_x_times, short_x_prices, color='#8B0000', s=300, marker='s', 
                  zorder=5, edgecolor='black', linewidth=2, label='SHORT Exit', alpha=0.9)
    
    # Plot LONG entries (blue up triangles)
    if long_entries:
        long_e_times = [t[0] for t in long_entries]
        long_e_prices = [t[1] for t in long_entries]
        ax.scatter(long_e_times, long_e_prices, color='#4169E1', s=300, marker='^', 
                  zorder=5, edgecolor='darkblue', linewidth=2, label='LONG Entry', alpha=0.9)
    
    # Plot LONG exits (dark blue squares)
    if long_exits:
        long_x_times = [t[0] for t in long_exits]
        long_x_prices = [t[1] for t in long_exits]
        ax.scatter(long_x_times, long_x_prices, color='#00008B', s=300, marker='s', 
                  zorder=5, edgecolor='black', linewidth=2, label='LONG Exit', alpha=0.9)
    
    # Formatting
    ax.set_title(f'SPY Minute Trading - {WORST_DAY.date()} (The "Worst" Day: $627 Profit, 45 Trades, 100% Win Rate)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Time', fontsize=13, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=13, fontweight='bold')
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, fontsize=12, loc='upper left', framealpha=0.95)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=11)
    
    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / "worst_day_price_chart_only.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Price chart saved to {output_file}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Generate price chart only."""
    
    print("\n" + "="*100)
    print("WORST DAY PRICE CHART")
    print("="*100)
    
    # Load data
    trades_df, minute_df = load_data()
    
    # Filter to worst day
    day_trades = trades_df[trades_df['exit_time'].dt.date == WORST_DAY.date()].copy()
    day_minute = minute_df[minute_df['Datetime'].dt.date == WORST_DAY.date()].copy()
    
    print(f"\nLoaded {len(day_trades)} trades for {WORST_DAY.date()}")
    print(f"Loaded {len(day_minute)} minute bars")
    
    # Create chart
    create_price_chart_only(day_trades, day_minute)
    
    print("\n" + "="*100)
    print("✅ PRICE CHART COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
