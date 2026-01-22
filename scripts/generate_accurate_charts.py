#!/usr/bin/env python3
"""
Generate accurate charts from real backtest data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Set dark theme
plt.style.use('dark_background')

# Custom color palette
COLORS = {
    'bg': '#0a0a0f',
    'card': '#1a1a24',
    'grid': '#2a2a3a',
    'cyan': '#00d4ff',
    'green': '#00ff88',
    'red': '#ff4444',
    'purple': '#8b5cf6',
    'orange': '#ff6b35',
    'yellow': '#ffd700',
    'text': '#ffffff',
    'text_secondary': '#a0a0b0',
}

def setup_dark_style():
    """Configure matplotlib for dark mode."""
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': COLORS['card'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'axes.titlecolor': COLORS['text'],
        'xtick.color': COLORS['text_secondary'],
        'ytick.color': COLORS['text_secondary'],
        'text.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.3,
        'legend.facecolor': COLORS['card'],
        'legend.edgecolor': COLORS['grid'],
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })

setup_dark_style()

OUTPUT_DIR = '/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/docs/assets'
DATA_DIR = '/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/results/extended_middle_deciles'

def load_and_process_trades():
    """Load trade data and calculate daily portfolio values."""
    # Load all trades
    df = pd.read_csv(f'{DATA_DIR}/all_trades.csv')
    df['Entry_Date'] = pd.to_datetime(df['Entry_Date'])
    df['Exit_Date'] = pd.to_datetime(df['Exit_Date'])
    
    print(f"Loaded {len(df)} trades from {df['Entry_Date'].min()} to {df['Exit_Date'].max()}")
    
    # Calculate cumulative returns
    # Each trade uses 4% of portfolio, so return contribution is 0.04 * return_pct
    df['Return_Contribution'] = 0.04 * df['Return_Pct'] / 100
    
    # Group by exit date and sum daily returns
    daily_returns = df.groupby('Exit_Date')['Return_Contribution'].sum().reset_index()
    daily_returns.columns = ['Date', 'Daily_Return']
    daily_returns = daily_returns.set_index('Date').sort_index()
    
    # Fill missing days with 0 return
    date_range = pd.date_range(daily_returns.index.min(), daily_returns.index.max(), freq='B')
    daily_returns = daily_returns.reindex(date_range, fill_value=0)
    
    # Calculate cumulative portfolio value
    portfolio_value = 100 * (1 + daily_returns['Daily_Return']).cumprod()
    
    return df, daily_returns, portfolio_value

def get_spy_returns(start_date, end_date):
    """Download SPY data for comparison."""
    import yfinance as yf
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    spy_returns = spy['Close'].pct_change().fillna(0)
    spy_value = 100 * (1 + spy_returns).cumprod()
    return spy_value

def create_accurate_performance_chart(portfolio_value, spy_value, trades_df):
    """Create the main performance chart with real data."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(COLORS['bg'])
    
    # Plot 1: Equity Curves
    ax1 = axes[0, 0]
    ax1.fill_between(portfolio_value.index, 100, portfolio_value.values, alpha=0.3, color=COLORS['cyan'])
    ax1.plot(portfolio_value.index, portfolio_value.values, color=COLORS['cyan'], linewidth=2, label=f'Strategy ({portfolio_value.iloc[-1]:.1f})')
    ax1.fill_between(spy_value.index, 100, spy_value.values, alpha=0.2, color=COLORS['orange'])
    ax1.plot(spy_value.index, spy_value.values, color=COLORS['orange'], linewidth=2, label=f'SPY ({spy_value.iloc[-1]:.1f})')
    ax1.axhline(y=100, color=COLORS['grid'], linestyle='--', alpha=0.5)
    ax1.set_title('Portfolio Growth (2020-2026) - REAL DATA', fontweight='bold', pad=10)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Monthly Returns
    ax2 = axes[0, 1]
    trades_df['Month'] = trades_df['Exit_Date'].dt.to_period('M')
    monthly = trades_df.groupby('Month')['Return_Pct'].mean()
    monthly_by_month = monthly.groupby(monthly.index.month).mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_vals = [monthly_by_month.get(i, 0) for i in range(1, 13)]
    colors = [COLORS['green'] if r > 0 else COLORS['red'] for r in monthly_vals]
    ax2.bar(months, monthly_vals, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax2.axhline(y=0, color=COLORS['text_secondary'], linestyle='-', linewidth=1)
    ax2.set_title('Average Return by Month (%)', fontweight='bold', pad=10)
    ax2.set_ylabel('Avg Trade Return (%)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Win Rate by Year
    ax3 = axes[1, 0]
    trades_df['Year'] = trades_df['Exit_Date'].dt.year
    yearly_stats = trades_df.groupby('Year').agg({
        'Win': 'mean',
        'Return_Pct': 'mean'
    })
    years = yearly_stats.index.astype(str)
    win_rates = yearly_stats['Win'] * 100
    colors = [COLORS['cyan'] if w > 50 else COLORS['purple'] for w in win_rates]
    bars = ax3.bar(years, win_rates, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax3.axhline(y=50, color=COLORS['text_secondary'], linestyle='--', linewidth=1, label='Random (50%)')
    ax3.set_title('Win Rate by Year', fontweight='bold', pad=10)
    ax3.set_ylabel('Win Rate (%)')
    ax3.set_ylim(45, 60)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, rate in zip(bars, win_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9, color=COLORS['text'])
    
    # Plot 4: Drawdown
    ax4 = axes[1, 1]
    rolling_max = portfolio_value.cummax()
    drawdown = (portfolio_value - rolling_max) / rolling_max * 100
    ax4.fill_between(drawdown.index, 0, drawdown.values, alpha=0.5, color=COLORS['red'])
    ax4.plot(drawdown.index, drawdown.values, color=COLORS['red'], linewidth=1)
    ax4.set_title('Strategy Drawdown', fontweight='bold', pad=10)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Drawdown (%)')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # Add max drawdown annotation
    min_dd = drawdown.min()
    min_dd_idx = drawdown.idxmin()
    ax4.annotate(f'Max DD: {min_dd:.1f}%', xy=(min_dd_idx, min_dd), 
                 xytext=(min_dd_idx + pd.Timedelta(days=100), min_dd + 10),
                 fontsize=10, color=COLORS['red'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/stock_picker_performance.png', dpi=150, 
                facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
    plt.close()
    print("✓ Created stock_picker_performance.png (from REAL data)")

def create_yearly_breakdown(trades_df):
    """Create a yearly breakdown chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    
    trades_df['Year'] = trades_df['Exit_Date'].dt.year
    yearly = trades_df.groupby('Year').agg({
        'Return_Pct': ['mean', 'sum', 'count'],
        'Win': 'mean'
    })
    yearly.columns = ['Avg_Return', 'Total_Return', 'Trades', 'Win_Rate']
    
    # Calculate yearly portfolio return (approximation)
    # Sum of (0.04 * return_pct) for each trade
    yearly_contrib = trades_df.groupby('Year').apply(
        lambda x: ((1 + 0.04 * x['Return_Pct'] / 100).prod() - 1) * 100
    )
    
    years = yearly.index.astype(str)
    colors = [COLORS['green'] if r > 0 else COLORS['red'] for r in yearly_contrib]
    
    bars = ax.bar(years, yearly_contrib, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    # Add value labels
    for bar, ret in zip(bars, yearly_contrib):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (2 if ret > 0 else -5),
                f'{ret:.1f}%', ha='center', va='bottom' if ret > 0 else 'top', 
                fontsize=12, fontweight='bold', color=COLORS['text'])
    
    ax.axhline(y=0, color=COLORS['text_secondary'], linestyle='-', linewidth=1)
    ax.set_ylabel('Yearly Return (%)')
    ax.set_title('Strategy Performance by Year', fontweight='bold', fontsize=14, pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/yearly_breakdown.png', dpi=150,
                facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
    plt.close()
    print("✓ Created yearly_breakdown.png")
    
    return yearly_contrib

def analyze_recent_performance(trades_df, portfolio_value):
    """Analyze why recent performance is poor."""
    print("\n" + "="*60)
    print("ANALYZING RECENT PERFORMANCE")
    print("="*60)
    
    trades_df['Year'] = trades_df['Exit_Date'].dt.year
    
    for year in sorted(trades_df['Year'].unique()):
        year_trades = trades_df[trades_df['Year'] == year]
        avg_ret = year_trades['Return_Pct'].mean()
        win_rate = year_trades['Win'].mean() * 100
        total_trades = len(year_trades)
        
        print(f"\n{year}:")
        print(f"  Trades: {total_trades}")
        print(f"  Avg Return per Trade: {avg_ret:.2f}%")
        print(f"  Win Rate: {win_rate:.1f}%")
    
    # Look at last 6 months specifically
    recent_cutoff = trades_df['Exit_Date'].max() - pd.Timedelta(days=180)
    recent = trades_df[trades_df['Exit_Date'] > recent_cutoff]
    
    print(f"\n\nLAST 6 MONTHS ({recent['Exit_Date'].min().date()} to {recent['Exit_Date'].max().date()}):")
    print(f"  Trades: {len(recent)}")
    print(f"  Avg Return per Trade: {recent['Return_Pct'].mean():.2f}%")
    print(f"  Win Rate: {recent['Win'].mean() * 100:.1f}%")
    
    # Check if model is picking losers
    print(f"\n  Top Confidence Trades (should be best):")
    high_conf = recent[recent['Decile'] == 'D10']
    print(f"    D10 trades: {len(high_conf)}, Win Rate: {high_conf['Win'].mean()*100:.1f}%, Avg Return: {high_conf['Return_Pct'].mean():.2f}%")

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Generating ACCURATE Charts from Real Backtest Data")
    print("=" * 60 + "\n")
    
    # Load real data
    trades_df, daily_returns, portfolio_value = load_and_process_trades()
    
    # Get SPY for comparison
    print("\nDownloading SPY data for comparison...")
    spy_value = get_spy_returns(portfolio_value.index.min(), portfolio_value.index.max())
    
    # Align dates
    common_dates = portfolio_value.index.intersection(spy_value.index)
    portfolio_value = portfolio_value.loc[common_dates]
    spy_value = spy_value.loc[common_dates]
    
    # Convert to Series if needed
    if hasattr(spy_value, 'values'):
        spy_value = pd.Series(spy_value.values.flatten(), index=spy_value.index)
    
    print(f"\nFinal Values:")
    print(f"  Strategy: ${portfolio_value.iloc[-1]:.2f} ({portfolio_value.iloc[-1] - 100:.2f}% return)")
    print(f"  SPY: ${float(spy_value.iloc[-1]):.2f} ({float(spy_value.iloc[-1]) - 100:.2f}% return)")
    
    # Create charts
    create_accurate_performance_chart(portfolio_value, spy_value, trades_df)
    yearly_returns = create_yearly_breakdown(trades_df)
    
    # Analyze recent poor performance
    analyze_recent_performance(trades_df, portfolio_value)
    
    print("\n✓ All charts regenerated with REAL data!")
