"""
Single Day Visualization - Minute-by-Minute Trading View
=========================================================
Zoom in to see exactly what the model is doing each minute of a trading day.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

# Set style
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor'] = '#161b22'
plt.rcParams['axes.edgecolor'] = '#30363d'
plt.rcParams['axes.labelcolor'] = '#c9d1d9'
plt.rcParams['text.color'] = '#c9d1d9'
plt.rcParams['xtick.color'] = '#8b949e'
plt.rcParams['ytick.color'] = '#8b949e'
plt.rcParams['grid.color'] = '#21262d'

# Paths
RESULTS_DIR = Path(__file__).parent / "results" / "real_minute_strict"
OUTPUT_DIR = RESULTS_DIR / "visualizations"

# Colors
COLORS = {
    'green': '#3fb950',
    'red': '#f85149',
    'blue': '#58a6ff',
    'purple': '#bc8cff',
    'orange': '#d29922',
    'cyan': '#39c5cf',
    'gray': '#8b949e',
    'white': '#c9d1d9'
}


def load_data():
    """Load prediction results."""
    predictions_file = RESULTS_DIR / "lightgbm_predictions.csv"
    df = pd.read_csv(predictions_file)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def plot_single_day(df, target_date=None):
    """
    Create detailed minute-by-minute visualization for a single trading day.
    """
    # Get available dates
    df['DateOnly'] = df['Date'].dt.date
    available_dates = sorted(df['DateOnly'].unique())
    
    # Pick a date with good activity (default to a mid-month date)
    if target_date is None:
        # Pick Dec 10, 2024 or closest available
        target_date = pd.to_datetime('2024-12-10').date()
        if target_date not in available_dates:
            # Find closest date
            target_date = available_dates[len(available_dates) // 2]
    
    # Filter to single day
    day_df = df[df['DateOnly'] == target_date].copy()
    
    if len(day_df) == 0:
        print(f"No data for {target_date}")
        print(f"Available dates: {available_dates[:5]} ... {available_dates[-5:]}")
        return None
    
    print(f"Visualizing {target_date}: {len(day_df)} minute bars")
    
    # Create figure
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'SINGLE DAY TRADING VIEW: {target_date}\nSPY Minute-by-Minute (Real Alpaca Data)', 
                 fontsize=18, fontweight='bold', color=COLORS['white'], y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25,
                          left=0.06, right=0.94, top=0.92, bottom=0.05)
    
    # Extract time for x-axis
    day_df['Time'] = day_df['Date'].dt.strftime('%H:%M')
    day_df['MinuteIndex'] = range(len(day_df))
    
    # 1. Price chart with predictions (TOP - full width)
    ax1 = fig.add_subplot(gs[0, :])
    
    ax1.plot(day_df['MinuteIndex'], day_df['Actual_Price'], 
             color=COLORS['blue'], linewidth=1.5, label='Actual Price', zorder=3)
    ax1.plot(day_df['MinuteIndex'], day_df['Predicted_Price'], 
             color=COLORS['orange'], linewidth=1, alpha=0.7, label='Predicted Price', zorder=2)
    
    # Mark correct/incorrect predictions with background colors
    for i, row in day_df.iterrows():
        idx = row['MinuteIndex']
        if row['Direction_Correct'] == 1:
            ax1.axvspan(idx - 0.5, idx + 0.5, alpha=0.1, color=COLORS['green'], zorder=1)
        else:
            ax1.axvspan(idx - 0.5, idx + 0.5, alpha=0.1, color=COLORS['red'], zorder=1)
    
    # Add high confidence markers
    high_conf = day_df[day_df['Confidence'] > 0.7]
    ax1.scatter(high_conf['MinuteIndex'], high_conf['Actual_Price'], 
                color=COLORS['cyan'], s=20, alpha=0.5, zorder=4, label='High Confidence')
    
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Price & Predictions (Green=Correct, Red=Incorrect, Cyan dots=High Confidence)', 
                  fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', framealpha=0.8)
    ax1.grid(True, alpha=0.3)
    
    # Custom x-axis labels (every 30 minutes)
    tick_positions = day_df['MinuteIndex'][::30].values
    tick_labels = day_df['Time'].iloc[::30].values
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45)
    
    # 2. Prediction confidence over time
    ax2 = fig.add_subplot(gs[1, 0])
    
    colors = [COLORS['green'] if c > 0.7 else COLORS['orange'] if c > 0.5 else COLORS['red'] 
              for c in day_df['Confidence']]
    ax2.bar(day_df['MinuteIndex'], day_df['Confidence'], color=colors, alpha=0.7, width=1)
    ax2.axhline(y=0.7, color=COLORS['cyan'], linestyle='--', linewidth=1.5, label='High Conf Threshold')
    ax2.axhline(y=0.5, color=COLORS['orange'], linestyle='--', linewidth=1, label='Medium Threshold')
    
    ax2.set_ylabel('Confidence')
    ax2.set_title('Prediction Confidence Throughout Day', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', framealpha=0.8)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    tick_positions = day_df['MinuteIndex'][::60].values
    tick_labels = day_df['Time'].iloc[::60].values
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45)
    
    # 3. Predicted vs Actual change scatter
    ax3 = fig.add_subplot(gs[1, 1])
    
    colors = [COLORS['green'] if d else COLORS['red'] for d in day_df['Direction_Correct']]
    ax3.scatter(day_df['Actual_Change'], day_df['Predicted_Change'], 
                c=colors, alpha=0.5, s=30, edgecolors='white', linewidths=0.3)
    
    # Add perfect prediction line
    lim = max(abs(day_df['Actual_Change']).max(), abs(day_df['Predicted_Change']).max())
    ax3.plot([-lim, lim], [-lim, lim], color=COLORS['gray'], linestyle='--', linewidth=1, label='Perfect Prediction')
    ax3.axhline(y=0, color=COLORS['gray'], linestyle='-', linewidth=0.5, alpha=0.5)
    ax3.axvline(x=0, color=COLORS['gray'], linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax3.set_xlabel('Actual Change ($)')
    ax3.set_ylabel('Predicted Change ($)')
    ax3.set_title('Predicted vs Actual Price Change (Green=Correct Direction)', 
                  fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='upper left', framealpha=0.8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative PnL simulation for the day
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Simulate trading: +1 for correct direction, -1 for wrong
    day_df['Trade_Result'] = np.where(day_df['Direction_Correct'] == 1, 1, -1)
    
    # Only trade high confidence
    day_df['High_Conf_Trade'] = np.where(day_df['Confidence'] > 0.7, day_df['Trade_Result'], 0)
    
    day_df['Cumulative_All'] = day_df['Trade_Result'].cumsum()
    day_df['Cumulative_HighConf'] = day_df['High_Conf_Trade'].cumsum()
    
    ax4.plot(day_df['MinuteIndex'], day_df['Cumulative_All'], 
             color=COLORS['blue'], linewidth=1.5, label='All Trades', alpha=0.7)
    ax4.plot(day_df['MinuteIndex'], day_df['Cumulative_HighConf'], 
             color=COLORS['green'], linewidth=2, label='High Confidence Only')
    ax4.axhline(y=0, color=COLORS['gray'], linestyle='--', linewidth=1)
    
    ax4.fill_between(day_df['MinuteIndex'], 0, day_df['Cumulative_HighConf'],
                     where=day_df['Cumulative_HighConf'] >= 0,
                     color=COLORS['green'], alpha=0.2)
    ax4.fill_between(day_df['MinuteIndex'], 0, day_df['Cumulative_HighConf'],
                     where=day_df['Cumulative_HighConf'] < 0,
                     color=COLORS['red'], alpha=0.2)
    
    ax4.set_ylabel('Cumulative PnL (units)')
    ax4.set_title('Cumulative P&L Throughout Day', fontsize=12, fontweight='bold', pad=10)
    ax4.legend(loc='upper left', framealpha=0.8)
    ax4.grid(True, alpha=0.3)
    
    tick_positions = day_df['MinuteIndex'][::60].values
    tick_labels = day_df['Time'].iloc[::60].values
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels(tick_labels, rotation=45)
    
    # 5. Rolling accuracy (30-minute window)
    ax5 = fig.add_subplot(gs[2, 1])
    
    day_df['Rolling_Accuracy'] = day_df['Direction_Correct'].rolling(window=30, min_periods=10).mean() * 100
    
    ax5.plot(day_df['MinuteIndex'], day_df['Rolling_Accuracy'], 
             color=COLORS['purple'], linewidth=2)
    ax5.axhline(y=50, color=COLORS['red'], linestyle='--', linewidth=1.5, label='Random (50%)')
    ax5.axhline(y=57, color=COLORS['green'], linestyle='--', linewidth=1, label='Overall Avg (57%)')
    
    ax5.fill_between(day_df['MinuteIndex'], 50, day_df['Rolling_Accuracy'],
                     where=day_df['Rolling_Accuracy'] >= 50,
                     color=COLORS['green'], alpha=0.3)
    ax5.fill_between(day_df['MinuteIndex'], 50, day_df['Rolling_Accuracy'],
                     where=day_df['Rolling_Accuracy'] < 50,
                     color=COLORS['red'], alpha=0.3)
    
    ax5.set_ylabel('Accuracy (%)')
    ax5.set_title('30-Minute Rolling Accuracy', fontsize=12, fontweight='bold', pad=10)
    ax5.legend(loc='upper right', framealpha=0.8)
    ax5.set_ylim(30, 80)
    ax5.grid(True, alpha=0.3)
    
    tick_positions = day_df['MinuteIndex'][::60].values
    tick_labels = day_df['Time'].iloc[::60].values
    ax5.set_xticks(tick_positions)
    ax5.set_xticklabels(tick_labels, rotation=45)
    
    # 6. Day summary stats
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    # Calculate stats
    total_bars = len(day_df)
    accuracy = day_df['Direction_Correct'].mean() * 100
    high_conf_count = len(day_df[day_df['Confidence'] > 0.7])
    high_conf_acc = day_df[day_df['Confidence'] > 0.7]['Direction_Correct'].mean() * 100 if high_conf_count > 0 else 0
    
    price_start = day_df['Actual_Price'].iloc[0]
    price_end = day_df['Actual_Price'].iloc[-1]
    price_change = price_end - price_start
    price_change_pct = (price_change / price_start) * 100
    
    final_pnl_all = day_df['Cumulative_All'].iloc[-1]
    final_pnl_hc = day_df['Cumulative_HighConf'].iloc[-1]
    
    max_price = day_df['Actual_Price'].max()
    min_price = day_df['Actual_Price'].min()
    
    # Create summary text
    summary = f"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                           DAY SUMMARY: {target_date}                                                        ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                                           ║
    ║   MARKET DATA                           MODEL PERFORMANCE                         TRADING SIMULATION                      ║
    ║   ─────────────                         ─────────────────                         ──────────────────                      ║
    ║   Total Bars:      {total_bars:>6}               Accuracy:         {accuracy:>6.1f}%               All Trades PnL:    {final_pnl_all:>+6}              ║
    ║   Open Price:     ${price_start:>7.2f}              High Conf Count:  {high_conf_count:>6}               High Conf PnL:     {final_pnl_hc:>+6}              ║
    ║   Close Price:    ${price_end:>7.2f}              High Conf Acc:    {high_conf_acc:>6.1f}%                                                   ║
    ║   Day High:       ${max_price:>7.2f}                                                                                                 ║
    ║   Day Low:        ${min_price:>7.2f}              Edge vs Random:   {accuracy - 50:>+6.1f}%                                                   ║
    ║   Price Change:   ${price_change:>+7.2f} ({price_change_pct:>+.2f}%)                                                                                       ║
    ║                                                                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax6.text(0.5, 0.5, summary, transform=ax6.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', color=COLORS['cyan'],
             bbox=dict(boxstyle='round', facecolor='#0d1117', edgecolor=COLORS['cyan'], linewidth=2))
    
    # Save
    output_file = OUTPUT_DIR / f"04_single_day_{target_date}.png"
    plt.savefig(output_file, dpi=150, facecolor='#0d1117', edgecolor='none')
    print(f"✓ Saved: {output_file}")
    
    plt.close()
    return output_file


def main():
    """Generate single day visualization."""
    print("\n" + "="*60)
    print("GENERATING SINGLE DAY VISUALIZATION")
    print("="*60 + "\n")
    
    df = load_data()
    print(f"Loaded {len(df):,} predictions")
    
    # Get unique dates
    df['DateOnly'] = df['Date'].dt.date
    dates = sorted(df['DateOnly'].unique())
    print(f"Available dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    
    # Pick a day with good trading activity
    # Let's find the day with the most minute bars
    bars_per_day = df.groupby('DateOnly').size()
    best_day = bars_per_day.idxmax()
    print(f"Day with most data: {best_day} ({bars_per_day.max()} bars)")
    
    # Create visualization for that day
    plot_single_day(df, best_day)
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
