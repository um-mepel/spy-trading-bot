"""
Portfolio Performance Visualization Module
Creates plots to analyze portfolio backtest results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def plot_portfolio_value(backtest_df, initial_capital, output_dir=None):
    """
    Plot portfolio value over time with buy/sell signals marked.
    
    Args:
        backtest_df: DataFrame with backtest results
        initial_capital: Starting capital (for reference line)
        output_dir: Directory to save plot (optional)
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Convert Date to datetime if needed
    backtest_df = backtest_df.copy()
    backtest_df['Date'] = pd.to_datetime(backtest_df['Date'])
    
    # Plot portfolio value
    ax.plot(backtest_df['Date'], backtest_df['Portfolio_Value'], 
           color='#1f77b4', linewidth=2.5, label='Portfolio Value', zorder=3)
    
    # Add initial capital reference line
    ax.axhline(y=initial_capital, color='gray', linestyle='--', 
              linewidth=2, label=f'Initial Capital: ${initial_capital:,.0f}', zorder=2)
    
    # Mark BUY signals (Signal = 1)
    buy_signals = backtest_df[backtest_df['Signal'] == 1]
    ax.scatter(buy_signals['Date'], buy_signals['Portfolio_Value'], 
              color='green', s=100, marker='^', label='BUY Signal', zorder=4, edgecolors='darkgreen', linewidth=1.5)
    
    # Mark SELL signals (Signal = -1)
    sell_signals = backtest_df[backtest_df['Signal'] == -1]
    ax.scatter(sell_signals['Date'], sell_signals['Portfolio_Value'], 
              color='red', s=100, marker='v', label='SELL Signal', zorder=4, edgecolors='darkred', linewidth=1.5)
    
    ax.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    
    if output_dir:
        output_file = output_dir / "portfolio_value.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file.name}")
    
    plt.close()


def plot_cumulative_return(backtest_df, output_dir=None):
    """
    Plot cumulative return percentage over time.
    
    Args:
        backtest_df: DataFrame with backtest results
        output_dir: Directory to save plot (optional)
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Convert Date to datetime if needed
    backtest_df = backtest_df.copy()
    backtest_df['Date'] = pd.to_datetime(backtest_df['Date'])
    
    # Plot cumulative return
    ax.plot(backtest_df['Date'], backtest_df['Cumulative_Return'], 
           color='#2ca02c', linewidth=2.5, label='Cumulative Return')
    
    # Fill area above/below zero
    ax.fill_between(backtest_df['Date'], 0, backtest_df['Cumulative_Return'], 
                   where=(backtest_df['Cumulative_Return'] >= 0), 
                   color='green', alpha=0.2, label='Gains')
    ax.fill_between(backtest_df['Date'], 0, backtest_df['Cumulative_Return'], 
                   where=(backtest_df['Cumulative_Return'] < 0), 
                   color='red', alpha=0.2, label='Losses')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, zorder=2)
    
    ax.set_title('Cumulative Return Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Cumulative Return (%)', fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    plt.tight_layout()
    
    if output_dir:
        output_file = output_dir / "cumulative_return.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file.name}")
    
    plt.close()


def plot_drawdown(backtest_df, output_dir=None):
    """
    Plot portfolio drawdown over time.
    
    Args:
        backtest_df: DataFrame with backtest results
        output_dir: Directory to save plot (optional)
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Convert Date to datetime if needed
    backtest_df = backtest_df.copy()
    backtest_df['Date'] = pd.to_datetime(backtest_df['Date'])
    
    # Calculate drawdown
    cummax = backtest_df['Portfolio_Value'].cummax()
    drawdown = (backtest_df['Portfolio_Value'] - cummax) / cummax * 100
    
    # Plot drawdown
    ax.fill_between(backtest_df['Date'], 0, drawdown, 
                   color='#d62728', alpha=0.6, label='Drawdown')
    ax.plot(backtest_df['Date'], drawdown, color='#d62728', linewidth=2)
    
    # Mark maximum drawdown
    max_dd_idx = drawdown.idxmin()
    max_dd_value = drawdown.min()
    ax.scatter(backtest_df.loc[max_dd_idx, 'Date'], max_dd_value, 
              color='darkred', s=150, marker='o', zorder=4, edgecolors='black', linewidth=1.5,
              label=f'Max Drawdown: {max_dd_value:.2f}%')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, zorder=2)
    
    ax.set_title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Drawdown (%)', fontsize=11)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    plt.tight_layout()
    
    if output_dir:
        output_file = output_dir / "drawdown.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file.name}")
    
    plt.close()


def plot_cash_and_shares(backtest_df, output_dir=None):
    """
    Plot cash and shares held over time.
    
    Args:
        backtest_df: DataFrame with backtest results
        output_dir: Directory to save plot (optional)
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    
    # Convert Date to datetime if needed
    backtest_df = backtest_df.copy()
    backtest_df['Date'] = pd.to_datetime(backtest_df['Date'])
    
    # Plot cash
    ax1.fill_between(backtest_df['Date'], 0, backtest_df['Cash'], 
                    color='#1f77b4', alpha=0.6)
    ax1.plot(backtest_df['Date'], backtest_df['Cash'], color='#1f77b4', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_title('Cash Available Over Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cash ($)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot shares held
    ax2.plot(backtest_df['Date'], backtest_df['Shares_Held'], 
            color='#ff7f0e', linewidth=2.5, marker='o', markersize=3, label='Shares Held')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.fill_between(backtest_df['Date'], 0, backtest_df['Shares_Held'], 
                    color='#ff7f0e', alpha=0.2)
    ax2.set_title('Shares Held Over Time', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Shares', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if output_dir:
        output_file = output_dir / "cash_and_shares.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file.name}")
    
    plt.close()


def generate_all_portfolio_plots(backtest_df, initial_capital, output_dir=None):
    """
    Generate all portfolio visualization plots.
    
    Args:
        backtest_df: DataFrame with backtest results
        initial_capital: Starting capital (for reference)
        output_dir: Directory to save plots (optional)
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating Portfolio Visualizations")
    print("="*60 + "\n")
    
    print("Generating portfolio value plot...")
    plot_portfolio_value(backtest_df, initial_capital, output_dir)
    
    print("Generating cumulative return plot...")
    plot_cumulative_return(backtest_df, output_dir)
    
    print("Generating drawdown plot...")
    plot_drawdown(backtest_df, output_dir)
    
    print("Generating cash and shares plot...")
    plot_cash_and_shares(backtest_df, output_dir)
    
    print("\n" + "="*60)
    print("Portfolio Visualizations Complete!")
    print("="*60)


if __name__ == "__main__":
    print("Portfolio Visualization Module")
    print("This module is designed to be imported by main.py")
