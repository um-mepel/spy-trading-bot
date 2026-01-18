"""
Day-by-Day Trading Performance Charts
====================================

Create detailed visualizations showing:
1. Daily P&L breakdown
2. Cumulative equity curve by day
3. Trade count and win rate per day
4. Average trade size and confidence by day
5. Daily statistics grid
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
plt.rcParams['font.size'] = 8

# Paths
TRADES_FILE = Path("/Users/mihirepel/Personal_Projects/Trading/results/trading_model/detailed_trades.json")
OUTPUT_DIR = Path(__file__).parent / "results" / "trading_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_trades():
    """Load trades from JSON file."""
    with open(TRADES_FILE, 'r') as f:
        trades_data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(trades_data)
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['date'] = df['exit_time'].dt.date
    
    return df


def create_daily_dashboard(trades_df):
    """Create comprehensive daily dashboard."""
    
    # Daily statistics
    daily_stats = trades_df.groupby('date').agg({
        'net_pnl': ['sum', 'count', 'mean'],
        'pnl_pct': ['mean'],
        'shares': 'mean',
        'confidence': 'mean'
    }).round(2)
    
    daily_stats.columns = ['Daily_PnL', 'Trade_Count', 'Avg_Trade_PnL', 'Avg_PnL_Pct', 'Avg_Shares', 'Avg_Confidence']
    daily_stats['Win_Rate'] = (trades_df.groupby('date')['net_pnl'].apply(lambda x: (x > 0).sum() / len(x))).round(3)
    daily_stats = daily_stats.reset_index()
    
    print("\nDaily Statistics:")
    print("="*100)
    print(daily_stats.to_string(index=False))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Daily P&L bar chart
    ax1 = fig.add_subplot(gs[0, :2])
    colors = ['green' if x > 0 else 'red' for x in daily_stats['Daily_PnL']]
    ax1.bar(range(len(daily_stats)), daily_stats['Daily_PnL'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_title('Daily P&L', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily P&L ($)')
    ax1.set_xticks(range(len(daily_stats)))
    ax1.set_xticklabels([str(d) for d in daily_stats['date']], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(daily_stats['Daily_PnL']):
        ax1.text(i, v + (50 if v > 0 else -50), f'${v:.0f}', ha='center', fontsize=8, fontweight='bold')
    
    # 2. Cumulative equity by day
    ax2 = fig.add_subplot(gs[0, 2])
    cumulative = daily_stats['Daily_PnL'].cumsum()
    ax2.plot(range(len(cumulative)), cumulative, marker='o', linewidth=2.5, markersize=6, color='darkgreen')
    ax2.fill_between(range(len(cumulative)), 0, cumulative, alpha=0.3, color='green')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_title('Cumulative Equity by Day', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative P&L ($)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Trade count per day
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(range(len(daily_stats)), daily_stats['Trade_Count'], color='steelblue', alpha=0.7, edgecolor='black')
    ax3.set_title('Trades Per Day', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('# of Trades')
    ax3.set_xticks(range(len(daily_stats)))
    ax3.set_xticklabels([str(d) for d in daily_stats['date']], rotation=45, ha='right', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(daily_stats['Trade_Count']):
        ax3.text(i, v + 1, str(int(v)), ha='center', fontsize=8, fontweight='bold')
    
    # 4. Daily win rate
    ax4 = fig.add_subplot(gs[1, 1])
    colors_wr = ['green' if x >= 0.95 else 'orange' if x >= 0.90 else 'red' for x in daily_stats['Win_Rate']]
    ax4.bar(range(len(daily_stats)), daily_stats['Win_Rate'] * 100, color=colors_wr, alpha=0.7, edgecolor='black')
    ax4.axhline(y=50, color='red', linestyle='--', linewidth=1, label='Random')
    ax4.axhline(y=100, color='green', linestyle='--', linewidth=1, label='Perfect')
    ax4.set_title('Daily Win Rate', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Win Rate (%)')
    ax4.set_xticks(range(len(daily_stats)))
    ax4.set_xticklabels([str(d) for d in daily_stats['date']], rotation=45, ha='right', fontsize=8)
    ax4.set_ylim([0, 105])
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(fontsize=8)
    
    # 5. Average trade size per day
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(range(len(daily_stats)), daily_stats['Avg_Shares'], marker='s', linewidth=2, markersize=6, color='purple')
    ax5.fill_between(range(len(daily_stats)), 0, daily_stats['Avg_Shares'], alpha=0.3, color='purple')
    ax5.set_title('Avg Position Size / Day', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Avg Shares')
    ax5.set_xticks(range(len(daily_stats)))
    ax5.set_xticklabels([str(d) for d in daily_stats['date']], rotation=45, ha='right', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Average confidence per day
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.bar(range(len(daily_stats)), daily_stats['Avg_Confidence'], color='teal', alpha=0.7, edgecolor='black')
    ax6.axhline(y=0.7, color='orange', linestyle='--', linewidth=1, label='High-Conf Threshold')
    ax6.set_title('Avg Confidence / Day', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Confidence')
    ax6.set_xticks(range(len(daily_stats)))
    ax6.set_xticklabels([str(d) for d in daily_stats['date']], rotation=45, ha='right', fontsize=8)
    ax6.set_ylim([0.5, 1.0])
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.legend(fontsize=8)
    
    # 7. Average P&L per trade by day
    ax7 = fig.add_subplot(gs[2, 1])
    colors_apnl = ['green' if x > 0 else 'red' for x in daily_stats['Avg_Trade_PnL']]
    ax7.bar(range(len(daily_stats)), daily_stats['Avg_Trade_PnL'], color=colors_apnl, alpha=0.7, edgecolor='black')
    ax7.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax7.set_title('Avg Trade P&L / Day', fontsize=11, fontweight='bold')
    ax7.set_xlabel('Date')
    ax7.set_ylabel('Avg P&L per Trade ($)')
    ax7.set_xticks(range(len(daily_stats)))
    ax7.set_xticklabels([str(d) for d in daily_stats['date']], rotation=45, ha='right', fontsize=8)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Summary statistics text box
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    total_trades = int(daily_stats['Trade_Count'].sum())
    total_pnl = daily_stats['Daily_PnL'].sum()
    avg_daily_pnl = daily_stats['Daily_PnL'].mean()
    best_day = daily_stats['Daily_PnL'].max()
    worst_day = daily_stats['Daily_PnL'].min()
    winning_days = (daily_stats['Daily_PnL'] > 0).sum()
    total_days = len(daily_stats)
    
    summary_text = f"""
DAILY TRADING SUMMARY

Total Trading Days: {total_days}
Total Trades: {total_trades}
Avg Trades/Day: {total_trades/total_days:.1f}

Total P&L: ${total_pnl:.2f}
Avg Daily P&L: ${avg_daily_pnl:.2f}
Best Day: ${best_day:.2f}
Worst Day: ${worst_day:.2f}

Profitable Days: {winning_days}/{total_days}
Day Win Rate: {winning_days/total_days*100:.1f}%

Avg Daily Confidence: {daily_stats['Avg_Confidence'].mean():.3f}
Avg Win Rate: {daily_stats['Win_Rate'].mean()*100:.1f}%
    """
    
    ax8.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Day-by-Day Trading Performance Analysis', fontsize=14, fontweight='bold', y=0.995)
    
    output_file = OUTPUT_DIR / "daily_trading_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Daily dashboard saved to {output_file}")
    
    return daily_stats


def create_daily_details_table(trades_df, daily_stats):
    """Create detailed daily breakdown figure."""
    
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    table_data.append(['Date', 'Trades', 'Wins', 'Win%', 'Daily P&L', 'Avg Trade', 'Avg Conf', 'Min PnL', 'Max PnL', 'Avg Shares'])
    
    for idx, row in daily_stats.iterrows():
        date_str = str(row['date'])
        trades = int(row['Trade_Count'])
        wins = int((trades_df[trades_df['date'] == row['date']]['net_pnl'] > 0).sum())
        win_pct = f"{row['Win_Rate']*100:.1f}%"
        daily_pnl = f"${row['Daily_PnL']:.2f}"
        avg_trade = f"${row['Avg_Trade_PnL']:.2f}"
        avg_conf = f"{row['Avg_Confidence']:.3f}"
        
        # Get min/max for the day
        day_trades = trades_df[trades_df['date'] == row['date']]
        min_pnl = f"${day_trades['net_pnl'].min():.2f}"
        max_pnl = f"${day_trades['net_pnl'].max():.2f}"
        avg_shares = f"{row['Avg_Shares']:.0f}"
        
        table_data.append([date_str, trades, wins, win_pct, daily_pnl, avg_trade, avg_conf, min_pnl, max_pnl, avg_shares])
    
    # Add totals row
    total_trades = int(daily_stats['Trade_Count'].sum())
    total_wins = (trades_df['net_pnl'] > 0).sum()
    total_win_pct = f"{total_wins/total_trades*100:.1f}%"
    total_pnl = f"${daily_stats['Daily_PnL'].sum():.2f}"
    avg_trade_all = f"${daily_stats['Avg_Trade_PnL'].mean():.2f}"
    avg_conf_all = f"{daily_stats['Avg_Confidence'].mean():.3f}"
    min_pnl_all = f"${trades_df['net_pnl'].min():.2f}"
    max_pnl_all = f"${trades_df['net_pnl'].max():.2f}"
    avg_shares_all = f"{daily_stats['Avg_Shares'].mean():.0f}"
    
    table_data.append(['TOTAL', total_trades, total_wins, total_win_pct, total_pnl, avg_trade_all, avg_conf_all, min_pnl_all, max_pnl_all, avg_shares_all])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.12, 0.08, 0.08, 0.08, 0.11, 0.11, 0.10, 0.10, 0.10, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Color header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color total row
    for i in range(len(table_data[0])):
        table[(len(table_data)-1, i)].set_facecolor('#ffeb99')
        table[(len(table_data)-1, i)].set_text_props(weight='bold')
    
    # Alternate row colors
    for i in range(1, len(table_data)-1):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor(color)
    
    # Color P&L cells based on value
    for i in range(1, len(table_data)-1):
        pnl_str = table_data[i][4]  # Daily P&L
        pnl_val = float(pnl_str.replace('$', '').replace(',', ''))
        if pnl_val > 0:
            table[(i, 4)].set_facecolor('#90EE90')  # Light green
        else:
            table[(i, 4)].set_facecolor('#FFB6C6')  # Light red
    
    plt.title('Daily Trading Performance - Detailed Breakdown', fontsize=14, fontweight='bold', pad=20)
    
    output_file = OUTPUT_DIR / "daily_trading_details.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Daily details table saved to {output_file}")


def create_heatmap(trades_df, daily_stats):
    """Create heatmap showing correlation between metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Trade count vs Win Rate
    ax = axes[0, 0]
    ax.scatter(daily_stats['Trade_Count'], daily_stats['Win_Rate']*100, s=200, alpha=0.6, color='steelblue', edgecolor='black')
    for i, date in enumerate(daily_stats['date']):
        ax.annotate(str(date), (daily_stats['Trade_Count'].iloc[i], daily_stats['Win_Rate'].iloc[i]*100),
                   fontsize=7, ha='right')
    ax.set_title('Trade Count vs Win Rate', fontweight='bold')
    ax.set_xlabel('# of Trades')
    ax.set_ylabel('Win Rate (%)')
    ax.grid(True, alpha=0.3)
    
    # 2. Confidence vs Daily P&L
    ax = axes[0, 1]
    colors_scatter = ['green' if x > 0 else 'red' for x in daily_stats['Daily_PnL']]
    ax.scatter(daily_stats['Avg_Confidence'], daily_stats['Daily_PnL'], s=200, alpha=0.6, c=colors_scatter, edgecolor='black')
    for i, date in enumerate(daily_stats['date']):
        ax.annotate(str(date), (daily_stats['Avg_Confidence'].iloc[i], daily_stats['Daily_PnL'].iloc[i]),
                   fontsize=7, ha='right')
    ax.set_title('Confidence vs Daily P&L', fontweight='bold')
    ax.set_xlabel('Avg Confidence')
    ax.set_ylabel('Daily P&L ($)')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3)
    
    # 3. Avg Trade P&L vs Trade Count
    ax = axes[1, 0]
    colors_scatter = ['green' if x > 0 else 'red' for x in daily_stats['Avg_Trade_PnL']]
    ax.scatter(daily_stats['Trade_Count'], daily_stats['Avg_Trade_PnL'], s=200, alpha=0.6, c=colors_scatter, edgecolor='black')
    for i, date in enumerate(daily_stats['date']):
        ax.annotate(str(date), (daily_stats['Trade_Count'].iloc[i], daily_stats['Avg_Trade_PnL'].iloc[i]),
                   fontsize=7, ha='right')
    ax.set_title('Avg Trade P&L vs Trade Count', fontweight='bold')
    ax.set_xlabel('# of Trades')
    ax.set_ylabel('Avg Trade P&L ($)')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3)
    
    # 4. Correlation matrix heatmap
    ax = axes[1, 1]
    metrics = daily_stats[['Trade_Count', 'Win_Rate', 'Daily_PnL', 'Avg_Confidence', 'Avg_Trade_PnL']].copy()
    metrics.columns = ['Trades', 'Win%', 'Daily PnL', 'Conf', 'Avg PnL']
    corr = metrics.corr()
    
    im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)
    
    # Add correlation values
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    ax.set_title('Metric Correlation Matrix', fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('Daily Trading Metrics Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / "daily_metrics_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics heatmap saved to {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Generate all day-by-day charts."""
    
    print("\n" + "="*100)
    print("GENERATING DAY-BY-DAY TRADING CHARTS")
    print("="*100)
    
    # Load trades
    trades_df = load_trades()
    print(f"\nLoaded {len(trades_df)} trades from {len(trades_df['date'].unique())} trading days")
    
    # Create visualizations
    daily_stats = create_daily_dashboard(trades_df)
    create_daily_details_table(trades_df, daily_stats)
    create_heatmap(trades_df, daily_stats)
    
    print("\n" + "="*100)
    print("✅ ALL DAY-BY-DAY CHARTS GENERATED")
    print("="*100)
    print(f"\nFiles created in {OUTPUT_DIR}:")
    print("  - daily_trading_analysis.png (comprehensive dashboard)")
    print("  - daily_trading_details.png (detailed statistics table)")
    print("  - daily_metrics_heatmap.png (metric correlations)")


if __name__ == "__main__":
    main()
