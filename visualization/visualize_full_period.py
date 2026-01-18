#!/usr/bin/env python3
"""
Visualize the full 6-month testing period with detailed analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Set dark theme
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor'] = '#161b22'
plt.rcParams['axes.edgecolor'] = '#30363d'
plt.rcParams['axes.labelcolor'] = '#c9d1d9'
plt.rcParams['text.color'] = '#c9d1d9'
plt.rcParams['xtick.color'] = '#8b949e'
plt.rcParams['ytick.color'] = '#8b949e'
plt.rcParams['grid.color'] = '#21262d'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['font.family'] = 'monospace'

RESULTS_DIR = Path("results/real_minute_strict")
VIZ_DIR = RESULTS_DIR / "visualizations"

def main():
    print("\n" + "="*60)
    print("FULL 6-MONTH PERIOD VISUALIZATION")
    print("="*60 + "\n")
    
    # Load predictions
    pred_file = RESULTS_DIR / "lightgbm_predictions.csv"
    df = pd.read_csv(pred_file, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"Loaded {len(df):,} predictions")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Add date components
    df['Day'] = df['Date'].dt.date
    df['Month'] = df['Date'].dt.to_period('M')
    df['Week'] = df['Date'].dt.to_period('W')
    df['Hour'] = df['Date'].dt.hour
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    
    # Title
    fig.suptitle('FULL 6-MONTH PERFORMANCE ANALYSIS\nSPY Jul-Dec 2024 (Real Alpaca Minute Data)', 
                 fontsize=20, fontweight='bold', color='#58a6ff', y=0.98)
    
    # =========================================================================
    # 1. FULL PRICE CHART WITH DAILY CANDLESTICKS
    # =========================================================================
    ax1 = fig.add_subplot(5, 2, (1, 2))
    
    # Aggregate to daily OHLC
    daily = df.groupby('Day').agg({
        'Actual_Price': ['first', 'max', 'min', 'last'],
        'Direction_Correct': 'mean'
    }).reset_index()
    daily.columns = ['Day', 'Open', 'High', 'Low', 'Close', 'Accuracy']
    daily['Day'] = pd.to_datetime(daily['Day'])
    
    # Plot candlesticks (simplified as line with range)
    for i, row in daily.iterrows():
        color = '#3fb950' if row['Close'] >= row['Open'] else '#f85149'
        ax1.plot([row['Day'], row['Day']], [row['Low'], row['High']], color=color, linewidth=0.8, alpha=0.7)
        ax1.plot([row['Day']], [row['Close']], 'o', color=color, markersize=2)
    
    ax1.plot(daily['Day'], daily['Close'], color='#58a6ff', linewidth=1.5, alpha=0.8, label='Close Price')
    
    # Add monthly labels
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.set_title('SPY Daily Price Action (Jul-Dec 2024)', fontsize=14, color='#58a6ff')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # =========================================================================
    # 2. MONTHLY ACCURACY BREAKDOWN
    # =========================================================================
    ax2 = fig.add_subplot(5, 2, 3)
    
    monthly_acc = df.groupby('Month').agg({
        'Direction_Correct': ['mean', 'count'],
        'Confidence': 'mean'
    }).reset_index()
    monthly_acc.columns = ['Month', 'Accuracy', 'Count', 'Avg_Conf']
    monthly_acc['Month_Str'] = monthly_acc['Month'].astype(str)
    
    bars = ax2.bar(monthly_acc['Month_Str'], monthly_acc['Accuracy'] * 100, 
                   color='#3fb950', edgecolor='#238636', alpha=0.8)
    ax2.axhline(y=50, color='#f85149', linestyle='--', linewidth=2, label='Random (50%)')
    ax2.axhline(y=monthly_acc['Accuracy'].mean() * 100, color='#58a6ff', linestyle='--', 
                linewidth=2, label=f"Average ({monthly_acc['Accuracy'].mean()*100:.1f}%)")
    
    # Add count labels
    for bar, count in zip(bars, monthly_acc['Count']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'n={count:,}', ha='center', va='bottom', fontsize=9, color='#8b949e')
    
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Monthly Accuracy Breakdown', fontsize=14, color='#58a6ff')
    ax2.legend(loc='lower right')
    ax2.set_ylim(45, 65)
    ax2.grid(True, alpha=0.3)
    
    # =========================================================================
    # 3. WEEKLY ROLLING ACCURACY
    # =========================================================================
    ax3 = fig.add_subplot(5, 2, 4)
    
    weekly_acc = df.groupby('Week').agg({
        'Direction_Correct': 'mean',
        'Date': 'first'
    }).reset_index()
    weekly_acc.columns = ['Week', 'Accuracy', 'Date']
    
    ax3.fill_between(weekly_acc['Date'], 50, weekly_acc['Accuracy'] * 100, 
                     where=weekly_acc['Accuracy'] * 100 >= 50,
                     color='#3fb950', alpha=0.3, label='Above random')
    ax3.fill_between(weekly_acc['Date'], 50, weekly_acc['Accuracy'] * 100,
                     where=weekly_acc['Accuracy'] * 100 < 50,
                     color='#f85149', alpha=0.3, label='Below random')
    ax3.plot(weekly_acc['Date'], weekly_acc['Accuracy'] * 100, color='#58a6ff', linewidth=2)
    ax3.axhline(y=50, color='#f85149', linestyle='--', linewidth=1.5)
    
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Weekly Rolling Accuracy', fontsize=14, color='#58a6ff')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # =========================================================================
    # 4. CUMULATIVE P&L SIMULATION
    # =========================================================================
    ax4 = fig.add_subplot(5, 2, 5)
    
    # Simulate P&L: +1 for correct direction, -1 for wrong
    df['PnL'] = df['Direction_Correct'].apply(lambda x: 1 if x else -1)
    df['Cumulative_PnL'] = df['PnL'].cumsum()
    
    # High confidence only
    high_conf = df[df['Confidence'] > 0.7].copy()
    high_conf['Cumulative_PnL'] = high_conf['PnL'].cumsum()
    
    ax4.fill_between(range(len(df)), 0, df['Cumulative_PnL'], alpha=0.3, color='#8b949e')
    ax4.plot(range(len(df)), df['Cumulative_PnL'], color='#8b949e', linewidth=1, label='All trades')
    ax4.plot(range(len(high_conf)), high_conf['Cumulative_PnL'].values, 
             color='#3fb950', linewidth=2, label='High confidence (>0.7)')
    ax4.axhline(y=0, color='#f85149', linestyle='--', linewidth=1)
    
    ax4.set_xlabel('Trade Number')
    ax4.set_ylabel('Cumulative P&L (units)')
    ax4.set_title(f'Cumulative P&L Over 6 Months ({len(df):,} trades)', fontsize=14, color='#58a6ff')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # =========================================================================
    # 5. DRAWDOWN ANALYSIS
    # =========================================================================
    ax5 = fig.add_subplot(5, 2, 6)
    
    # Calculate drawdown
    cummax = df['Cumulative_PnL'].cummax()
    drawdown = df['Cumulative_PnL'] - cummax
    
    ax5.fill_between(range(len(df)), 0, drawdown, color='#f85149', alpha=0.5)
    ax5.plot(range(len(df)), drawdown, color='#f85149', linewidth=1)
    
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    ax5.scatter([max_dd_idx], [max_dd], color='#ffa657', s=100, zorder=5, marker='v')
    ax5.annotate(f'Max DD: {max_dd:.0f}', (max_dd_idx, max_dd), 
                 textcoords='offset points', xytext=(10, -10), color='#ffa657', fontsize=10)
    
    ax5.set_xlabel('Trade Number')
    ax5.set_ylabel('Drawdown (units)')
    ax5.set_title('Drawdown Analysis', fontsize=14, color='#58a6ff')
    ax5.grid(True, alpha=0.3)
    
    # =========================================================================
    # 6. ACCURACY BY HOUR (ACROSS ALL DAYS)
    # =========================================================================
    ax6 = fig.add_subplot(5, 2, 7)
    
    hourly = df.groupby('Hour').agg({
        'Direction_Correct': ['mean', 'count']
    }).reset_index()
    hourly.columns = ['Hour', 'Accuracy', 'Count']
    
    colors = ['#3fb950' if acc > 0.5 else '#f85149' for acc in hourly['Accuracy']]
    bars = ax6.bar(hourly['Hour'], hourly['Accuracy'] * 100, color=colors, alpha=0.8)
    ax6.axhline(y=50, color='#f85149', linestyle='--', linewidth=2)
    ax6.axhline(y=hourly['Accuracy'].mean() * 100, color='#58a6ff', linestyle='--', linewidth=2)
    
    # Mark market sessions
    ax6.axvspan(4, 9.5, alpha=0.1, color='#ffa657', label='Pre-market')
    ax6.axvspan(9.5, 16, alpha=0.1, color='#3fb950', label='Regular hours')
    ax6.axvspan(16, 20, alpha=0.1, color='#8b5cf6', label='After-hours')
    
    ax6.set_xlabel('Hour (UTC)')
    ax6.set_ylabel('Accuracy (%)')
    ax6.set_title('Accuracy by Hour of Day', fontsize=14, color='#58a6ff')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.set_xlim(-0.5, 23.5)
    ax6.grid(True, alpha=0.3)
    
    # =========================================================================
    # 7. CONFIDENCE DISTRIBUTION
    # =========================================================================
    ax7 = fig.add_subplot(5, 2, 8)
    
    ax7.hist(df['Confidence'], bins=50, color='#58a6ff', alpha=0.7, edgecolor='#1f6feb')
    ax7.axvline(x=0.7, color='#3fb950', linestyle='--', linewidth=2, label='High conf threshold')
    ax7.axvline(x=df['Confidence'].mean(), color='#ffa657', linestyle='--', linewidth=2, 
                label=f"Mean: {df['Confidence'].mean():.2f}")
    
    pct_high = (df['Confidence'] > 0.7).mean() * 100
    ax7.text(0.75, ax7.get_ylim()[1] * 0.9, f'{pct_high:.1f}% high conf', 
             color='#3fb950', fontsize=12, fontweight='bold')
    
    ax7.set_xlabel('Confidence Score')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Prediction Confidence Distribution', fontsize=14, color='#58a6ff')
    ax7.legend(loc='upper left')
    ax7.grid(True, alpha=0.3)
    
    # =========================================================================
    # 8. DAILY WIN RATE HEATMAP-STYLE
    # =========================================================================
    ax8 = fig.add_subplot(5, 2, 9)
    
    daily_acc = df.groupby('Day')['Direction_Correct'].mean().reset_index()
    daily_acc.columns = ['Day', 'Accuracy']
    daily_acc['Day'] = pd.to_datetime(daily_acc['Day'])
    
    # Color by accuracy
    colors = ['#3fb950' if acc > 0.55 else '#ffa657' if acc > 0.5 else '#f85149' 
              for acc in daily_acc['Accuracy']]
    
    ax8.bar(range(len(daily_acc)), daily_acc['Accuracy'] * 100, color=colors, alpha=0.8, width=1)
    ax8.axhline(y=50, color='#f85149', linestyle='--', linewidth=1.5)
    ax8.axhline(y=55, color='#3fb950', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add month labels
    month_starts = daily_acc.groupby(daily_acc['Day'].dt.to_period('M')).first().index
    
    ax8.set_xlabel('Trading Day')
    ax8.set_ylabel('Daily Accuracy (%)')
    ax8.set_title(f'Daily Accuracy ({len(daily_acc)} trading days)', fontsize=14, color='#58a6ff')
    ax8.grid(True, alpha=0.3)
    
    # =========================================================================
    # 9. SUMMARY STATISTICS BOX
    # =========================================================================
    ax9 = fig.add_subplot(5, 2, 10)
    ax9.axis('off')
    
    # Calculate stats
    total_trades = len(df)
    high_conf_trades = len(df[df['Confidence'] > 0.7])
    overall_acc = df['Direction_Correct'].mean()
    high_conf_acc = df[df['Confidence'] > 0.7]['Direction_Correct'].mean()
    final_pnl = df['Cumulative_PnL'].iloc[-1]
    high_conf_pnl = high_conf['Cumulative_PnL'].iloc[-1]
    max_drawdown = drawdown.min()
    win_days = (daily_acc['Accuracy'] > 0.5).sum()
    total_days = len(daily_acc)
    best_day = daily_acc.loc[daily_acc['Accuracy'].idxmax()]
    worst_day = daily_acc.loc[daily_acc['Accuracy'].idxmin()]
    
    stats_text = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    6-MONTH PERFORMANCE SUMMARY                ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  PERIOD              Jul 1, 2024 - Dec 31, 2024              ║
    ║  TRAINING DATA       928,494 bars (Jan 2020 - Jun 2024)      ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                         TRADE STATISTICS                      ║
    ║  Total Predictions:  {total_trades:>10,}                              ║
    ║  High Conf (>0.7):   {high_conf_trades:>10,}  ({high_conf_trades/total_trades*100:>5.1f}%)                  ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                         ACCURACY                              ║
    ║  Overall Accuracy:   {overall_acc*100:>10.2f}%                             ║
    ║  High Conf Accuracy: {high_conf_acc*100:>10.2f}%                             ║
    ║  Edge vs Random:     {(overall_acc-0.5)*100:>10.2f}%                             ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                         P&L SIMULATION                        ║
    ║  Final P&L (all):    {final_pnl:>+10,.0f} units                          ║
    ║  Final P&L (hi-conf):{high_conf_pnl:>+10,.0f} units                          ║
    ║  Max Drawdown:       {max_drawdown:>10,.0f} units                          ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                         DAILY STATS                           ║
    ║  Trading Days:       {total_days:>10}                              ║
    ║  Winning Days:       {win_days:>10}  ({win_days/total_days*100:>5.1f}%)                  ║
    ║  Best Day:           {str(best_day['Day'])[:10]}  ({best_day['Accuracy']*100:.1f}%)               ║
    ║  Worst Day:          {str(worst_day['Day'])[:10]}  ({worst_day['Accuracy']*100:.1f}%)               ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    ax9.text(0.5, 0.5, stats_text, transform=ax9.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', color='#58a6ff',
             bbox=dict(boxstyle='round', facecolor='#161b22', edgecolor='#30363d'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_file = VIZ_DIR / "05_full_6month_analysis.png"
    plt.savefig(output_file, dpi=150, facecolor='#0d1117', edgecolor='none')
    plt.close()
    
    print(f"✓ Saved: {output_file}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
