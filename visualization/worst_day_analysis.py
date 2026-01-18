"""
Worst Trading Day Analysis - 2024-12-09
========================================

Detailed minute-by-minute breakdown of the worst trading day from the large sample test.
Analyzes what went "wrong" and provides insights into trading dynamics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 9

# Paths
TRADES_FILE = Path("/Users/mihirepel/Personal_Projects/Trading/results/trading_model/detailed_trades.json")
TEST_DATA_FILE = Path("/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/results/real_minute_large_sample/SPY_minute_testing_2024-12-01_to_2024-12-31.csv")
OUTPUT_DIR = Path("/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/results/trading_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Worst day from analysis
WORST_DAY = pd.Timestamp('2024-12-09')


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


def analyze_worst_day(trades_df, minute_df):
    """Analyze the worst trading day in detail."""
    
    # Filter to worst day trades
    day_trades = trades_df[trades_df['exit_time'].dt.date == WORST_DAY.date()].copy()
    
    # Filter to worst day minute data
    day_minute = minute_df[minute_df['Datetime'].dt.date == WORST_DAY.date()].copy()
    
    print("\n" + "="*150)
    print(f"WORST TRADING DAY ANALYSIS - {WORST_DAY.date()}")
    print("="*150)
    
    print(f"\nMinute Data Summary:")
    print(f"  Total minutes: {len(day_minute)}")
    print(f"  Price range: ${day_minute['Close'].min():.2f} - ${day_minute['Close'].max():.2f}")
    print(f"  Daily move: {(day_minute['Close'].iloc[-1] / day_minute['Close'].iloc[0] - 1) * 100:.2f}%")
    print(f"  Average volatility: {day_minute['Volatility_10'].mean():.3f}")
    
    print(f"\nTrade Summary:")
    print(f"  Total trades: {len(day_trades)}")
    
    if len(day_trades) > 0:
        winners = (day_trades['net_pnl'] > 0).sum()
        losers = (day_trades['net_pnl'] <= 0).sum()
        win_rate = winners / len(day_trades) * 100
        total_pnl = day_trades['net_pnl'].sum()
        avg_pnl = day_trades['net_pnl'].mean()
        
        print(f"  Winners: {winners} ({win_rate:.1f}%)")
        print(f"  Losers: {losers}")
        print(f"  Total P&L: ${total_pnl:.2f}")
        print(f"  Avg P&L per trade: ${avg_pnl:.2f}")
        print(f"  Largest win: ${day_trades['net_pnl'].max():.2f}")
        print(f"  Largest loss: ${day_trades['net_pnl'].min():.2f}")
        print(f"  Avg confidence: {day_trades['confidence'].mean():.3f}")
    
    return day_trades, day_minute


def create_worst_day_visualizations(day_trades, day_minute):
    """Create detailed visualizations for the worst day."""
    
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
    
    # 1. Full price chart with all trades overlaid (larger)
    ax1 = fig.add_subplot(gs[0:2, :3])
    ax1.plot(day_minute['Datetime'], day_minute['Close'], linewidth=2.5, color='black', label='SPY Price', zorder=2)
    
    # Plot all trades with clear SHORT vs LONG distinction
    if len(day_trades) > 0:
        for idx, trade in day_trades.iterrows():
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            entry_time = trade['entry_time']
            exit_time = trade['exit_time']
            direction = trade['direction']
            
            # Color code by direction: RED for SHORT, BLUE for LONG
            if direction == 'DOWN':  # SHORT
                entry_color = '#FF6B6B'  # Light red
                exit_color = '#8B0000'   # Dark red
                marker_entry = 'v'       # Down triangle for SHORT entry
            else:  # UP (LONG)
                entry_color = '#4169E1'  # Royal blue
                exit_color = '#00008B'   # Dark blue
                marker_entry = '^'       # Up triangle for LONG entry
            
            # Entry point
            ax1.scatter(entry_time, entry_price, color=entry_color, s=180, marker=marker_entry, 
                       zorder=4, edgecolor='black', linewidth=2, label=direction if idx == 0 else '')
            
            # Exit point (always square)
            ax1.scatter(exit_time, exit_price, color=exit_color, s=180, marker='s', 
                       zorder=4, edgecolor='black', linewidth=2)
            
            # Trade line
            ax1.plot([entry_time, exit_time], [entry_price, exit_price], 
                    color=exit_color, linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax1.set_title(f'SPY Price Action & Trades - {WORST_DAY.date()} (Worst Day)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add legend for trade types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', edgecolor='black', label='SHORT Entry (▼)'),
        Patch(facecolor='#8B0000', edgecolor='black', label='SHORT Exit (■)'),
        Patch(facecolor='#4169E1', edgecolor='black', label='LONG Entry (▲)'),
        Patch(facecolor='#00008B', edgecolor='black', label='LONG Exit (■)'),
    ]
    ax1.legend(handles=legend_elements, fontsize=10, loc='upper left')
    
    # 2. Trade P&L distribution (moved to right side, smaller)
    ax2 = fig.add_subplot(gs[0, 3])
    if len(day_trades) > 0:
        colors = ['#8B0000' if d == 'DOWN' else '#00008B' for d in day_trades['direction']]
        ax2.barh(range(min(10, len(day_trades))), day_trades['net_pnl'].head(10), color=colors, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax2.set_title('Top 10 Trade P&L', fontsize=11, fontweight='bold')
        ax2.set_xlabel('P&L ($)', fontsize=9)
        ax2.set_ylabel('Trade #', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. SHORT vs LONG breakdown
    ax3 = fig.add_subplot(gs[1, 3])
    if len(day_trades) > 0:
        short_count = (day_trades['direction'] == 'DOWN').sum()
        long_count = (day_trades['direction'] == 'UP').sum()
        colors_pie = ['#8B0000', '#00008B']
        ax3.pie([short_count, long_count], labels=[f'SHORT\n({short_count})', f'LONG\n({long_count})'],
               colors=colors_pie, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10, 'weight': 'bold'})
        ax3.set_title('SHORT vs LONG Trades', fontsize=11, fontweight='bold')
    
    # 4. Hourly statistics
    ax4 = fig.add_subplot(gs[2, :2])
    day_minute_copy = day_minute.copy()
    day_minute_copy['hour'] = day_minute_copy['Datetime'].dt.hour
    hourly_stats = day_minute_copy.groupby('hour').agg({
        'Close': ['first', 'last', lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100]
    }).round(2)
    hourly_stats.columns = ['Open', 'Close', 'Hour_Return_%']
    
    hours = hourly_stats.index
    returns = hourly_stats['Hour_Return_%']
    colors = ['green' if x > 0 else 'red' for x in returns]
    ax4.bar(hours, returns, color=colors, alpha=0.7, edgecolor='black')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_title('Hourly Returns', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Hour', fontsize=10)
    ax4.set_ylabel('Return (%)', fontsize=10)
    ax4.set_xticks(hours)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Win vs Loss trades
    ax5 = fig.add_subplot(gs[2, 2])
    if len(day_trades) > 0:
        wins = (day_trades['net_pnl'] > 0).sum()
        losses = (day_trades['net_pnl'] <= 0).sum()
        colors_pie = ['#90EE90', '#FFB6C6']
        ax5.pie([wins, losses], labels=[f'Wins\n({wins})', f'Losses\n({losses})'],
               colors=colors_pie, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10, 'weight': 'bold'})
        ax5.set_title('Win vs Loss Trades', fontsize=11, fontweight='bold')
    
    # 6. Confidence vs P&L by direction
    ax6 = fig.add_subplot(gs[2, 3])
    if len(day_trades) > 0:
        short_trades = day_trades[day_trades['direction'] == 'DOWN']
        long_trades = day_trades[day_trades['direction'] == 'UP']
        
        if len(short_trades) > 0:
            ax6.scatter(short_trades['confidence'], short_trades['net_pnl'], 
                       s=120, alpha=0.7, c='#8B0000', edgecolor='black', linewidth=1, label='SHORT')
        if len(long_trades) > 0:
            ax6.scatter(long_trades['confidence'], long_trades['net_pnl'], 
                       s=120, alpha=0.7, c='#00008B', edgecolor='black', linewidth=1, label='LONG')
        
        ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax6.set_title('Confidence vs P&L', fontsize=11, fontweight='bold')
        ax6.set_xlabel('Confidence', fontsize=9)
        ax6.set_ylabel('P&L ($)', fontsize=9)
        ax6.grid(True, alpha=0.3)
        ax6.legend(fontsize=9)
    
    # 7. Trade details table
    if len(day_trades) > 0:
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        # Create table
        table_data = [['Trade', 'Time', 'Type', 'Entry $', 'Exit $', 'Shares', 'Gross P&L', 'Net P&L', 'Conf']]
        
        for idx, trade in day_trades.iterrows():
            time_str = trade['entry_time'].strftime('%H:%M') + '-' + trade['exit_time'].strftime('%H:%M')
            trade_type = 'SHORT' if trade['direction'] == 'DOWN' else 'LONG'
            
            table_data.append([
                f"#{idx+1}",
                time_str,
                trade_type,
                f"${trade['entry_price']:.2f}",
                f"${trade['exit_price']:.2f}",
                f"{int(trade['shares'])}",
                f"${trade['gross_pnl']:.2f}",
                f"${trade['net_pnl']:.2f}",
                f"{trade['confidence']:.2f}"
            ])
        
        table = ax7.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.08, 0.12, 0.10, 0.10, 0.10, 0.08, 0.12, 0.12, 0.08])
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)
        
        # Color header
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows by direction
        for i in range(1, len(table_data)):
            trade_type = table_data[i][2]
            if trade_type == 'SHORT':
                color = '#FFB6C6'  # Light red for SHORT
            else:
                color = '#B6D7FF'  # Light blue for LONG
            
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor(color)
        
        ax7.set_title('Trade-by-Trade Breakdown (SHORT = Red, LONG = Blue)', fontsize=12, fontweight='bold', pad=10)
    
    plt.suptitle(f'Worst Trading Day Analysis - {WORST_DAY.date()} (SPY)', 
                fontsize=14, fontweight='bold', y=0.995)
    
    output_file = OUTPUT_DIR / "worst_day_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Worst day analysis saved to {output_file}")


def print_insights(day_trades, day_minute):
    """Print trading insights for the worst day."""
    
    print("\n" + "="*150)
    print("WORST DAY INSIGHTS")
    print("="*150)
    
    if len(day_trades) > 0:
        # Trade duration analysis
        day_trades['duration_minutes'] = (day_trades['exit_time'] - day_trades['entry_time']).dt.total_seconds() / 60
        
        print(f"\n📊 TRADE TIMING:")
        print(f"  Average trade duration: {day_trades['duration_minutes'].mean():.1f} minutes")
        print(f"  Shortest trade: {day_trades['duration_minutes'].min():.0f} minutes")
        print(f"  Longest trade: {day_trades['duration_minutes'].max():.0f} minutes")
        
        # Win/loss analysis
        winning_trades = day_trades[day_trades['net_pnl'] > 0]
        losing_trades = day_trades[day_trades['net_pnl'] <= 0]
        
        print(f"\n🎯 WIN/LOSS ANALYSIS:")
        print(f"  Winning trades: {len(winning_trades)} trades, avg ${winning_trades['net_pnl'].mean():.2f}")
        if len(winning_trades) > 0:
            print(f"    - Best winner: ${winning_trades['net_pnl'].max():.2f}")
            print(f"    - Avg confidence: {winning_trades['confidence'].mean():.3f}")
        
        print(f"  Losing trades: {len(losing_trades)} trades, avg ${losing_trades['net_pnl'].mean():.2f}")
        if len(losing_trades) > 0:
            print(f"    - Worst loser: ${losing_trades['net_pnl'].min():.2f}")
            print(f"    - Avg confidence: {losing_trades['confidence'].mean():.3f}")
        
        # Confidence analysis
        print(f"\n💪 CONFIDENCE ANALYSIS:")
        high_conf_trades = day_trades[day_trades['confidence'] >= 0.7]
        low_conf_trades = day_trades[day_trades['confidence'] < 0.7]
        
        if len(high_conf_trades) > 0:
            print(f"  High-confidence trades (≥0.7): {len(high_conf_trades)}")
            print(f"    - Win rate: {(high_conf_trades['net_pnl'] > 0).sum() / len(high_conf_trades) * 100:.1f}%")
            print(f"    - Avg P&L: ${high_conf_trades['net_pnl'].mean():.2f}")
        
        if len(low_conf_trades) > 0:
            print(f"  Low-confidence trades (<0.7): {len(low_conf_trades)}")
            print(f"    - Win rate: {(low_conf_trades['net_pnl'] > 0).sum() / len(low_conf_trades) * 100:.1f}%")
            print(f"    - Avg P&L: ${low_conf_trades['net_pnl'].mean():.2f}")
        
        # Market conditions
        print(f"\n📈 MARKET CONDITIONS:")
        daily_return = (day_minute['Close'].iloc[-1] / day_minute['Close'].iloc[0] - 1) * 100
        daily_high = day_minute['Close'].max()
        daily_low = day_minute['Close'].min()
        daily_range = ((daily_high - daily_low) / daily_low) * 100
        
        print(f"  Daily return: {daily_return:.2f}%")
        print(f"  Daily range: {daily_range:.2f}%")
        print(f"  Price volatility: {day_minute['Close'].pct_change().std() * 100:.2f}%")
        
        # Why was this the worst day?
        total_pnl = day_trades['net_pnl'].sum()
        trade_count = len(day_trades)
        
        print(f"\n🔍 WHY WAS THIS THE WORST DAY?")
        print(f"  Total P&L: ${total_pnl:.2f}")
        print(f"  Trade count: {trade_count} (vs avg 90)")
        print(f"  Avg per trade: ${total_pnl/trade_count:.2f} (vs avg $14.35)")
        
        if len(losing_trades) > 0 and len(winning_trades) > 0:
            loss_ratio = abs(losing_trades['net_pnl'].mean()) / winning_trades['net_pnl'].mean()
            if loss_ratio > 1:
                print(f"  ⚠️  Losses were {loss_ratio:.2f}x larger than wins")
        
        if (day_trades['confidence'] >= 0.7).mean() < 0.5:
            print(f"  ⚠️  Only {(day_trades['confidence'] >= 0.7).mean()*100:.1f}% high-confidence trades (vs usual ~50%)")
        
        if daily_return < -0.1:
            print(f"  ⚠️  Market moved against us: {daily_return:.2f}% down")
        
        if day_trades['duration_minutes'].mean() > 20:
            print(f"  ⚠️  Trades held longer than usual: {day_trades['duration_minutes'].mean():.1f} minutes avg")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Analyze the worst trading day."""
    
    print("\n" + "="*150)
    print("WORST TRADING DAY ANALYSIS - LARGE SAMPLE TEST")
    print("="*150)
    
    # Load data
    trades_df, minute_df = load_data()
    print(f"\nLoaded {len(trades_df)} total trades")
    print(f"Loaded {len(minute_df)} minute bars")
    
    # Analyze worst day
    day_trades, day_minute = analyze_worst_day(trades_df, minute_df)
    
    # Create visualizations
    create_worst_day_visualizations(day_trades, day_minute)
    
    # Print insights
    print_insights(day_trades, day_minute)
    
    print("\n" + "="*150)
    print("✅ WORST DAY ANALYSIS COMPLETE")
    print("="*150)


if __name__ == "__main__":
    main()
