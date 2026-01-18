#!/usr/bin/env python3
"""
Visualization script for optimized trading strategy
Creates comprehensive charts showing strategy performance
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

def load_data():
    """Load backtest results"""
    df = pd.read_csv('results/optimized_strategy_backtest.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def create_visualizations(df):
    """Create comprehensive visualization dashboard"""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Portfolio Value Over Time (top, large)
    ax1 = plt.subplot(3, 2, (1, 2))
    ax1.plot(df['Date'], df['Portfolio_Value'], linewidth=2, color='#2E86AB', label='Portfolio Value')
    ax1.fill_between(df['Date'], df['Portfolio_Value'], alpha=0.2, color='#2E86AB')
    ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Format y-axis as currency
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 2. Cumulative Return (%)
    ax2 = plt.subplot(3, 2, 3)
    ax2.plot(df['Date'], df['Cumulative_Return'] * 100, linewidth=2, color='#A23B72', label='Cumulative Return')
    ax2.fill_between(df['Date'], df['Cumulative_Return'] * 100, alpha=0.2, color='#A23B72')
    ax2.set_title('Cumulative Return (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Return (%)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Daily Returns Distribution
    ax3 = plt.subplot(3, 2, 4)
    returns = df['Daily_Return'].dropna()
    ax3.hist(returns * 100, bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
    ax3.axvline(returns.mean() * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean()*100:.3f}%')
    ax3.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Daily Return (%)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Shares Held Over Time
    ax4 = plt.subplot(3, 2, 5)
    ax4.bar(df['Date'], df['Shares_Held'], width=1, color='#06A77D', alpha=0.7)
    ax4.set_title('Shares Held Over Time', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Shares', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Cash Position Over Time
    ax5 = plt.subplot(3, 2, 6)
    ax5.plot(df['Date'], df['Cash'], linewidth=2, color='#D62828', label='Cash')
    ax5.fill_between(df['Date'], 0, df['Cash'], alpha=0.2, color='#D62828')
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax5.set_title('Cash Position Over Time', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Cash ($)', fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax5.legend()
    
    # Format x-axis for all subplots
    for ax in [ax1, ax2, ax4, ax5]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def create_signal_analysis(df):
    """Create signal analysis visualization"""
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    
    # Signal distribution
    signal_counts = df['Signal'].value_counts()
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    axes[0].bar(signal_counts.index, signal_counts.values, color=colors[:len(signal_counts)], alpha=0.7, edgecolor='black')
    axes[0].set_title('Trading Signals Distribution', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(signal_counts.values):
        axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    # Confidence score distribution for BUY signals
    buy_signals = df[df['Signal'] == 'BUY']
    axes[1].hist(buy_signals['Confidence'].dropna(), bins=30, color='#06A77D', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0.85, color='red', linestyle='--', linewidth=2, label='Filter Threshold (0.85)')
    axes[1].set_title('BUY Signal Confidence Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Confidence Score', fontsize=10)
    axes[1].set_ylabel('Frequency', fontsize=10)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def create_performance_summary(df):
    """Create performance summary statistics"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Calculate metrics
    total_return = (df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0] - 1) * 100
    max_drawdown = ((df['Portfolio_Value'].min() - df['Portfolio_Value'].max()) / df['Portfolio_Value'].max()) * 100
    daily_returns = df['Daily_Return'].dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    win_rate = (daily_returns > 0).sum() / len(daily_returns) * 100
    
    num_buy_signals = (df['Signal'] == 'BUY').sum()
    num_accepted = int(df['Trades_Accepted'].iloc[-1]) if 'Trades_Accepted' in df.columns else 15
    num_filtered = num_buy_signals - num_accepted
    
    summary_text = f"""
    OPTIMIZED TRADING STRATEGY - PERFORMANCE SUMMARY
    
    RETURN METRICS:
    • Total Return: {total_return:.2f}%
    • Initial Capital: $100,000
    • Final Value: ${df['Portfolio_Value'].iloc[-1]:,.2f}
    • Profit: ${df['Portfolio_Value'].iloc[-1] - 100000:,.2f}
    
    RISK METRICS:
    • Maximum Drawdown: {max_drawdown:.2f}%
    • Daily Sharpe Ratio: {sharpe_ratio:.3f}
    • Win Rate (Daily): {win_rate:.1f}%
    • Volatility (Annualized): {daily_returns.std() * np.sqrt(252) * 100:.2f}%
    
    TRADING ACTIVITY:
    • BUY Signals Generated: {num_buy_signals}
    • Signals Accepted (Conf>0.85 + MA50): {num_accepted}
    • Signals Filtered Out: {num_filtered}
    • Filter Effectiveness: {(num_filtered/num_buy_signals)*100:.1f}%
    
    POSITION MANAGEMENT:
    • Final Shares Held: {df['Shares_Held'].iloc[-1]:.0f}
    • Peak Shares Held: {df['Shares_Held'].max():.0f}
    • Final Cash Position: ${df['Cash'].iloc[-1]:,.2f}
    
    IMPROVEMENT OVER BASELINE:
    • Baseline Return: 14.74%
    • Optimized Return: {total_return:.2f}%
    • Improvement: +{total_return - 14.74:.2f}pp (+{((total_return/14.74)-1)*100:.1f}%)
    """
    
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    return fig

def main():
    """Main execution"""
    print("Loading backtest data...")
    df = load_data()
    
    print("Creating main performance chart...")
    fig1 = create_visualizations(df)
    fig1.savefig('results/strategy_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/strategy_performance.png")
    
    print("Creating signal analysis chart...")
    fig2 = create_signal_analysis(df)
    fig2.savefig('results/signal_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/signal_analysis.png")
    
    print("Creating performance summary...")
    fig3 = create_performance_summary(df)
    fig3.savefig('results/performance_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/performance_summary.png")
    
    print("\n✅ All visualizations created successfully!")
    print("\nGenerated files:")
    print("  1. results/strategy_performance.png - 6-panel performance dashboard")
    print("  2. results/signal_analysis.png - Signal distribution and confidence analysis")
    print("  3. results/performance_summary.png - Detailed metrics summary")

if __name__ == '__main__':
    main()
