"""
Single Day Trading Analysis
===========================

Detailed breakdown of trading performance for a single day (2026-01-15)
Shows: minute-by-minute price action, signals, trades, P&L
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 9

# Path to the AAPL minute predictions file
DATA_FILE = Path("/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/results/minute_data_2026/AAPL_minute_predictions_2026-01-12_to_2026-01-16.csv")
OUTPUT_DIR = Path(__file__).parent / "results" / "trading_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_filter_data():
    """Load minute data and filter to single day."""
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter to 2026-01-15
    target_date = pd.Timestamp('2026-01-15')
    day_df = df[df['Date'].dt.date == target_date.date()].copy()
    
    return day_df


def simulate_trades(df, confidence_threshold=0.7, target=0.005, stop=0.005):
    """Simulate trading for the day."""
    
    trades = []
    open_position = None
    
    for idx, row in df.iterrows():
        current_time = row['Date']
        current_price = row['Actual_Price']
        confidence = row['Confidence']
        
        # Check if we should enter a trade
        if open_position is None and confidence >= confidence_threshold:
            # Only trade on direction confidence
            if row['Direction_Correct'] > 0:  # High-confidence directional signal
                # Determine direction
                if row['Predicted_Change'] > 0.001:  # Upward signal
                    direction = 'LONG'
                else:  # Downward signal
                    direction = 'SHORT'
                
                open_position = {
                    'entry_time': current_time,
                    'entry_price': current_price,
                    'direction': direction,
                    'confidence': confidence,
                    'predicted_change': row['Predicted_Change'],
                    'target': current_price * (1 + target) if direction == 'LONG' else current_price * (1 - target),
                    'stop': current_price * (1 - stop) if direction == 'LONG' else current_price * (1 + stop),
                }
        
        # Check if we should exit a trade
        if open_position is not None:
            # Check profit target or stop loss
            if open_position['direction'] == 'LONG':
                if current_price >= open_position['target']:
                    trades.append({
                        'entry_time': open_position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': open_position['entry_price'],
                        'exit_price': current_price,
                        'direction': 'LONG',
                        'pnl': current_price - open_position['entry_price'],
                        'pnl_pct': (current_price / open_position['entry_price'] - 1) * 100,
                        'exit_reason': 'PROFIT_TARGET',
                        'confidence': open_position['confidence']
                    })
                    open_position = None
                elif current_price <= open_position['stop']:
                    trades.append({
                        'entry_time': open_position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': open_position['entry_price'],
                        'exit_price': current_price,
                        'direction': 'LONG',
                        'pnl': current_price - open_position['entry_price'],
                        'pnl_pct': (current_price / open_position['entry_price'] - 1) * 100,
                        'exit_reason': 'STOP_LOSS',
                        'confidence': open_position['confidence']
                    })
                    open_position = None
            else:  # SHORT
                if current_price <= open_position['target']:
                    trades.append({
                        'entry_time': open_position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': open_position['entry_price'],
                        'exit_price': current_price,
                        'direction': 'SHORT',
                        'pnl': open_position['entry_price'] - current_price,
                        'pnl_pct': (open_position['entry_price'] / current_price - 1) * 100,
                        'exit_reason': 'PROFIT_TARGET',
                        'confidence': open_position['confidence']
                    })
                    open_position = None
                elif current_price >= open_position['stop']:
                    trades.append({
                        'entry_time': open_position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': open_position['entry_price'],
                        'exit_price': current_price,
                        'direction': 'SHORT',
                        'pnl': open_position['entry_price'] - current_price,
                        'pnl_pct': (open_position['entry_price'] / current_price - 1) * 100,
                        'exit_reason': 'STOP_LOSS',
                        'confidence': open_position['confidence']
                    })
                    open_position = None
    
    trades_df = pd.DataFrame(trades)
    return trades_df


def create_single_day_dashboard(day_df, trades_df):
    """Create comprehensive single-day dashboard."""
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    
    # 1. Price chart with trades
    ax1 = fig.add_subplot(gs[0:2, :2])
    ax1.plot(day_df['Date'], day_df['Actual_Price'], linewidth=2, color='black', label='AAPL Price', zorder=2)
    
    # Plot high-confidence signals
    high_conf = day_df[day_df['Is_High_Confidence'] == 1]
    ax1.scatter(high_conf['Date'], high_conf['Actual_Price'], color='green', s=100, marker='^', 
               label='High-Confidence Signal', zorder=3, alpha=0.7)
    
    # Plot trades
    if len(trades_df) > 0:
        for idx, trade in trades_df.iterrows():
            # Entry
            ax1.scatter(trade['entry_time'], trade['entry_price'], color='blue', s=200, marker='o', zorder=4)
            # Exit
            color = 'green' if trade['pnl'] > 0 else 'red'
            ax1.scatter(trade['exit_time'], trade['exit_price'], color=color, s=200, marker='s', zorder=4)
            # Connect entry to exit
            ax1.plot([trade['entry_time'], trade['exit_time']], 
                    [trade['entry_price'], trade['exit_price']], 
                    color=color, linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax1.set_title('2026-01-15 - Price Action with Trades', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price ($)')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Confidence over time
    ax2 = fig.add_subplot(gs[0, 2])
    colors = ['green' if x >= 0.7 else 'orange' if x >= 0.5 else 'red' for x in day_df['Confidence']]
    ax2.scatter(range(len(day_df)), day_df['Confidence'], c=colors, s=50, alpha=0.6)
    ax2.axhline(y=0.7, color='green', linestyle='--', linewidth=2, label='High-Conf Threshold')
    ax2.set_title('Confidence Levels', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Minute')
    ax2.set_ylabel('Confidence')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    
    # 3. Predicted vs Actual Change
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.scatter(day_df['Predicted_Change'], day_df['Actual_Change'], s=50, alpha=0.6, color='steelblue')
    ax3.plot([day_df['Predicted_Change'].min(), day_df['Predicted_Change'].max()],
            [day_df['Predicted_Change'].min(), day_df['Predicted_Change'].max()],
            'r--', linewidth=2, label='Perfect Prediction')
    ax3.set_title('Predicted vs Actual Change', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Predicted Change')
    ax3.set_ylabel('Actual Change')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    
    # 4. Trade statistics
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    
    if len(trades_df) > 0:
        total_trades = len(trades_df)
        winning_trades = (trades_df['pnl'] > 0).sum()
        losing_trades = (trades_df['pnl'] <= 0).sum()
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        best_trade = trades_df['pnl'].max()
        worst_trade = trades_df['pnl'].min()
        
        stats_text = f"""
TRADE STATISTICS
===============

Total Trades: {total_trades}
Winning Trades: {winning_trades}
Losing Trades: {losing_trades}
Win Rate: {win_rate:.1f}%

Total P&L: ${total_pnl:.2f}
Avg Trade P&L: ${avg_pnl:.2f}
Best Trade: ${best_trade:.2f}
Worst Trade: ${worst_trade:.2f}

Avg Trade Size: {trades_df['pnl_pct'].mean():.2f}%
Profit Factor: {abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / (trades_df[trades_df['pnl'] <= 0]['pnl'].sum() + 0.01)):.2f}x
        """
    else:
        stats_text = "NO TRADES EXECUTED"
    
    ax4.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 5. Signal quality breakdown
    ax5 = fig.add_subplot(gs[2, 1])
    high_conf_count = (day_df['Is_High_Confidence'] == 1).sum()
    low_conf_count = (day_df['Is_High_Confidence'] == 0).sum()
    
    ax5.pie([high_conf_count, low_conf_count], labels=['High Confidence', 'Low Confidence'],
           colors=['#90EE90', '#FFB6C6'], autopct='%1.1f%%', startangle=90)
    ax5.set_title('Signal Quality Distribution', fontsize=11, fontweight='bold')
    
    # 6. Direction accuracy
    ax6 = fig.add_subplot(gs[2, 2])
    correct_dir = (day_df['Direction_Correct'] == 1).sum()
    incorrect_dir = (day_df['Direction_Correct'] == 0).sum()
    
    ax6.bar(['Correct', 'Incorrect'], [correct_dir, incorrect_dir], color=['green', 'red'], alpha=0.7)
    ax6.set_title('Direction Prediction Accuracy', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Count')
    accuracy_pct = correct_dir / len(day_df) * 100
    ax6.text(0.5, max(correct_dir, incorrect_dir) * 0.5, f'{accuracy_pct:.1f}%', 
            ha='center', fontsize=12, fontweight='bold')
    
    # 7. Trade results (if any trades)
    if len(trades_df) > 0:
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        # Create table
        table_data = [['Entry Time', 'Exit Time', 'Direction', 'Entry Price', 'Exit Price', 'P&L', 'P&L %', 'Reason', 'Confidence']]
        
        for idx, trade in trades_df.iterrows():
            entry_str = trade['entry_time'].strftime('%H:%M')
            exit_str = trade['exit_time'].strftime('%H:%M')
            pnl_str = f"${trade['pnl']:.2f}"
            pnl_pct_str = f"{trade['pnl_pct']:.2f}%"
            
            table_data.append([
                entry_str,
                exit_str,
                trade['direction'],
                f"${trade['entry_price']:.2f}",
                f"${trade['exit_price']:.2f}",
                pnl_str,
                pnl_pct_str,
                trade['exit_reason'],
                f"{trade['confidence']:.2f}"
            ])
        
        table = ax7.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.11, 0.11, 0.08, 0.11, 0.11, 0.10, 0.10, 0.14, 0.10])
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)
        
        # Color header
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color trade rows
        for i in range(1, len(table_data)):
            pnl_val = float(table_data[i][5].replace('$', ''))
            color = '#90EE90' if pnl_val > 0 else '#FFB6C6'
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor(color)
        
        ax7.set_title('Trade Details', fontsize=12, fontweight='bold', pad=10)
    
    plt.suptitle('Single Day Trading Analysis - 2026-01-15 (AAPL)', fontsize=14, fontweight='bold', y=0.995)
    
    output_file = OUTPUT_DIR / "single_day_trading_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Single day analysis saved to {output_file}")
    
    return trades_df


def print_minute_breakdown(day_df):
    """Print detailed minute-by-minute breakdown."""
    
    print("\n" + "="*140)
    print("MINUTE-BY-MINUTE BREAKDOWN - 2026-01-15")
    print("="*140)
    
    # Sample every 5th minute for readability
    sample_df = day_df.iloc[::5].copy()
    
    print(f"\n{'Time':<12} {'Price':<10} {'Pred Chg':<10} {'Act Chg':<10} {'Conf':<8} {'Dir':<6} {'Correct':<8} {'High Conf':<10}")
    print("-" * 140)
    
    for idx, row in sample_df.iterrows():
        time_str = row['Date'].strftime('%H:%M:%S')
        price_str = f"${row['Actual_Price']:.2f}"
        pred_chg = f"{row['Predicted_Change']*100:.2f}%"
        act_chg = f"{row['Actual_Change']*100:.2f}%"
        conf = f"{row['Confidence']:.2f}"
        direction = "UP" if row['Predicted_Change'] > 0 else "DOWN"
        correct = "✓" if row['Direction_Correct'] == 1 else "✗"
        high_conf = "YES" if row['Is_High_Confidence'] == 1 else "NO"
        
        print(f"{time_str:<12} {price_str:<10} {pred_chg:<10} {act_chg:<10} {conf:<8} {direction:<6} {correct:<8} {high_conf:<10}")
    
    # Summary statistics
    print("\n" + "="*140)
    print("DAILY SUMMARY STATISTICS")
    print("="*140)
    
    total_signals = len(day_df)
    high_conf_signals = (day_df['Is_High_Confidence'] == 1).sum()
    direction_accuracy = (day_df['Direction_Correct'] == 1).sum() / len(day_df) * 100
    high_conf_accuracy = day_df[day_df['Is_High_Confidence'] == 1]['Direction_Correct'].mean() * 100
    
    print(f"\nTotal Signals: {total_signals}")
    print(f"High-Confidence Signals: {high_conf_signals} ({high_conf_signals/total_signals*100:.1f}%)")
    print(f"Direction Accuracy: {direction_accuracy:.1f}%")
    print(f"High-Confidence Accuracy: {high_conf_accuracy:.1f}%")
    print(f"Average Confidence: {day_df['Confidence'].mean():.3f}")
    print(f"Price Range: ${day_df['Actual_Price'].min():.2f} - ${day_df['Actual_Price'].max():.2f}")
    print(f"Daily Move: {(day_df['Actual_Price'].iloc[-1] / day_df['Actual_Price'].iloc[0] - 1) * 100:.2f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Generate single day analysis."""
    
    print("\n" + "="*140)
    print("SINGLE DAY TRADING ANALYSIS")
    print("="*140)
    
    # Load data
    day_df = load_and_filter_data()
    print(f"\nLoaded {len(day_df)} minute bars for 2026-01-15")
    
    # Simulate trades
    trades_df = simulate_trades(day_df)
    print(f"Simulated {len(trades_df)} trades")
    
    # Create visualizations
    create_single_day_dashboard(day_df, trades_df)
    
    # Print breakdown
    print_minute_breakdown(day_df)
    
    # Print trade details
    if len(trades_df) > 0:
        print("\n" + "="*140)
        print("TRADE DETAILS")
        print("="*140)
        for idx, trade in trades_df.iterrows():
            print(f"\nTrade #{idx+1}:")
            print(f"  Entry: {trade['entry_time'].strftime('%H:%M:%S')} @ ${trade['entry_price']:.2f}")
            print(f"  Exit:  {trade['exit_time'].strftime('%H:%M:%S')} @ ${trade['exit_price']:.2f}")
            print(f"  P&L:   ${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)")
            print(f"  Reason: {trade['exit_reason']}")
            print(f"  Confidence: {trade['confidence']:.2f}")
    
    print("\n" + "="*140)
    print("✅ SINGLE DAY ANALYSIS COMPLETE")
    print("="*140)


if __name__ == "__main__":
    main()
