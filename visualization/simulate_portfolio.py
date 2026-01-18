#!/usr/bin/env python3
"""
Realistic Portfolio Simulation based on minute-level trading signals.
Simulates actual trading with position sizing, transaction costs, and slippage.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

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

# ============================================================================
# PORTFOLIO CONFIGURATION
# ============================================================================
INITIAL_CAPITAL = 100_000  # Starting with $100,000
POSITION_SIZE_PCT = 0.10   # Risk 10% of portfolio per trade
MAX_POSITION_PCT = 0.50    # Never more than 50% in single position
COMMISSION_PER_SHARE = 0.00  # Most brokers are commission-free now
SLIPPAGE_PCT = 0.01        # 0.01% slippage per trade (1 cent per $100)
MIN_CONFIDENCE = 0.70      # Only trade on high-confidence signals
MIN_PREDICTED_MOVE = 0.10  # Minimum predicted move of $0.10 (10 cents)

RESULTS_DIR = Path("results/real_minute_strict")
VIZ_DIR = RESULTS_DIR / "visualizations"


def simulate_portfolio(df):
    """
    Simulate a realistic portfolio trading on high-confidence signals.
    """
    print("\n" + "="*70)
    print("PORTFOLIO SIMULATION")
    print("="*70)
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Position Size: {POSITION_SIZE_PCT*100:.0f}% of portfolio")
    print(f"Min Confidence: {MIN_CONFIDENCE}")
    print(f"Slippage: {SLIPPAGE_PCT}%")
    print("="*70 + "\n")
    
    # Initialize portfolio state
    cash = INITIAL_CAPITAL
    shares = 0
    position_entry_price = 0
    position_entry_time = None
    
    # Track history
    portfolio_history = []
    trades = []
    daily_returns = {}
    
    # Filter to high-confidence predictions with meaningful moves
    df = df.copy()
    df['Predicted_Move'] = abs(df['Predicted_Change'])
    signals = df[(df['Confidence'] >= MIN_CONFIDENCE) & 
                 (df['Predicted_Move'] >= MIN_PREDICTED_MOVE)].copy()
    
    print(f"Total predictions: {len(df):,}")
    print(f"Tradeable signals: {len(signals):,}")
    
    prev_date = None
    
    for idx, row in signals.iterrows():
        current_price = row['Actual_Price']
        predicted_change = row['Predicted_Change']
        confidence = row['Confidence']
        timestamp = row['Date']
        current_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
        
        # Track daily
        if prev_date != current_date:
            prev_date = current_date
        
        # Calculate portfolio value
        portfolio_value = cash + (shares * current_price)
        
        # Determine signal: BUY if predicting up, SELL if predicting down
        signal = 'BUY' if predicted_change > 0 else 'SELL'
        
        # Execute trades
        if signal == 'BUY' and shares == 0:
            # Open long position
            position_size = portfolio_value * POSITION_SIZE_PCT
            max_shares = int(position_size / current_price)
            
            if max_shares > 0 and cash >= max_shares * current_price:
                # Apply slippage (buy at slightly higher price)
                execution_price = current_price * (1 + SLIPPAGE_PCT/100)
                cost = max_shares * execution_price
                
                if cost <= cash:
                    shares = max_shares
                    cash -= cost
                    position_entry_price = execution_price
                    position_entry_time = timestamp
                    
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'BUY',
                        'shares': shares,
                        'price': execution_price,
                        'confidence': confidence,
                        'predicted_change': predicted_change,
                        'portfolio_value': portfolio_value
                    })
        
        elif signal == 'SELL' and shares > 0:
            # Close long position
            # Apply slippage (sell at slightly lower price)
            execution_price = current_price * (1 - SLIPPAGE_PCT/100)
            proceeds = shares * execution_price
            
            # Calculate trade P&L
            trade_pnl = proceeds - (shares * position_entry_price)
            trade_return = (execution_price - position_entry_price) / position_entry_price
            
            trades.append({
                'timestamp': timestamp,
                'action': 'SELL',
                'shares': shares,
                'price': execution_price,
                'confidence': confidence,
                'predicted_change': predicted_change,
                'portfolio_value': portfolio_value,
                'trade_pnl': trade_pnl,
                'trade_return': trade_return,
                'hold_time': (timestamp - position_entry_time).total_seconds() / 60 if position_entry_time else 0
            })
            
            cash += proceeds
            shares = 0
            position_entry_price = 0
            position_entry_time = None
        
        # Record portfolio state
        portfolio_value = cash + (shares * current_price)
        portfolio_history.append({
            'timestamp': timestamp,
            'cash': cash,
            'shares': shares,
            'price': current_price,
            'portfolio_value': portfolio_value,
            'return_pct': (portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        })
    
    # Close any remaining position at end
    if shares > 0:
        final_price = signals.iloc[-1]['Actual_Price']
        execution_price = final_price * (1 - SLIPPAGE_PCT/100)
        proceeds = shares * execution_price
        trade_pnl = proceeds - (shares * position_entry_price)
        
        trades.append({
            'timestamp': signals.iloc[-1]['Date'],
            'action': 'SELL (Close)',
            'shares': shares,
            'price': execution_price,
            'trade_pnl': trade_pnl
        })
        
        cash += proceeds
        shares = 0
    
    portfolio_df = pd.DataFrame(portfolio_history)
    trades_df = pd.DataFrame(trades)
    
    return portfolio_df, trades_df


def calculate_metrics(portfolio_df, trades_df):
    """Calculate comprehensive portfolio metrics."""
    
    initial = INITIAL_CAPITAL
    final = portfolio_df['portfolio_value'].iloc[-1] if len(portfolio_df) > 0 else initial
    
    # Filter to sell trades only for P&L analysis
    sells = trades_df[trades_df['action'].isin(['SELL', 'SELL (Close)'])].copy()
    
    metrics = {
        'initial_capital': initial,
        'final_value': final,
        'total_return': (final - initial) / initial * 100,
        'total_pnl': final - initial,
        'num_trades': len(sells),
        'winning_trades': len(sells[sells['trade_pnl'] > 0]) if 'trade_pnl' in sells.columns else 0,
        'losing_trades': len(sells[sells['trade_pnl'] <= 0]) if 'trade_pnl' in sells.columns else 0,
    }
    
    if len(sells) > 0 and 'trade_pnl' in sells.columns:
        metrics['win_rate'] = metrics['winning_trades'] / metrics['num_trades'] * 100
        metrics['avg_win'] = sells[sells['trade_pnl'] > 0]['trade_pnl'].mean() if metrics['winning_trades'] > 0 else 0
        metrics['avg_loss'] = sells[sells['trade_pnl'] <= 0]['trade_pnl'].mean() if metrics['losing_trades'] > 0 else 0
        metrics['largest_win'] = sells['trade_pnl'].max()
        metrics['largest_loss'] = sells['trade_pnl'].min()
        metrics['avg_hold_time'] = sells['hold_time'].mean() if 'hold_time' in sells.columns else 0
        
        # Profit factor
        gross_profit = sells[sells['trade_pnl'] > 0]['trade_pnl'].sum()
        gross_loss = abs(sells[sells['trade_pnl'] <= 0]['trade_pnl'].sum())
        metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Max drawdown
    if len(portfolio_df) > 0:
        cummax = portfolio_df['portfolio_value'].cummax()
        drawdown = (portfolio_df['portfolio_value'] - cummax) / cummax * 100
        metrics['max_drawdown'] = drawdown.min()
        metrics['max_drawdown_dollars'] = (portfolio_df['portfolio_value'] - cummax).min()
    
    # Sharpe ratio approximation (assuming 252 trading days, ~390 min/day)
    if len(portfolio_df) > 1:
        returns = portfolio_df['portfolio_value'].pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            # Annualize: ~98,280 minute bars per year
            annualized_return = returns.mean() * 98280
            annualized_vol = returns.std() * np.sqrt(98280)
            metrics['sharpe_ratio'] = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    return metrics


def visualize_portfolio(portfolio_df, trades_df, metrics):
    """Create comprehensive portfolio visualization."""
    
    fig = plt.figure(figsize=(20, 28))
    
    fig.suptitle('PORTFOLIO SIMULATION RESULTS\n$100,000 Starting Capital | SPY Jul-Dec 2024', 
                 fontsize=22, fontweight='bold', color='#58a6ff', y=0.98)
    
    # =========================================================================
    # 1. PORTFOLIO VALUE OVER TIME
    # =========================================================================
    ax1 = fig.add_subplot(6, 2, (1, 2))
    
    ax1.fill_between(range(len(portfolio_df)), INITIAL_CAPITAL, 
                     portfolio_df['portfolio_value'], 
                     where=portfolio_df['portfolio_value'] >= INITIAL_CAPITAL,
                     color='#3fb950', alpha=0.3)
    ax1.fill_between(range(len(portfolio_df)), INITIAL_CAPITAL,
                     portfolio_df['portfolio_value'],
                     where=portfolio_df['portfolio_value'] < INITIAL_CAPITAL,
                     color='#f85149', alpha=0.3)
    ax1.plot(portfolio_df['portfolio_value'], color='#58a6ff', linewidth=2)
    ax1.axhline(y=INITIAL_CAPITAL, color='#8b949e', linestyle='--', linewidth=1.5, 
                label=f'Initial: ${INITIAL_CAPITAL:,.0f}')
    
    final_val = portfolio_df['portfolio_value'].iloc[-1]
    ax1.axhline(y=final_val, color='#3fb950' if final_val > INITIAL_CAPITAL else '#f85149', 
                linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'Final: ${final_val:,.0f}')
    
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.set_title('Portfolio Value Over Time', fontsize=16, color='#58a6ff')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # =========================================================================
    # 2. DAILY RETURNS DISTRIBUTION
    # =========================================================================
    ax2 = fig.add_subplot(6, 2, 3)
    
    # Calculate daily returns
    portfolio_df['date'] = pd.to_datetime(portfolio_df['timestamp']).dt.date
    daily = portfolio_df.groupby('date')['portfolio_value'].last().pct_change().dropna() * 100
    
    colors = ['#3fb950' if r > 0 else '#f85149' for r in daily]
    ax2.bar(range(len(daily)), daily.values, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='#8b949e', linestyle='-', linewidth=1)
    ax2.axhline(y=daily.mean(), color='#58a6ff', linestyle='--', linewidth=2,
                label=f'Mean: {daily.mean():.3f}%')
    
    ax2.set_xlabel('Trading Day')
    ax2.set_ylabel('Daily Return (%)')
    ax2.set_title('Daily Returns', fontsize=14, color='#58a6ff')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # =========================================================================
    # 3. CUMULATIVE RETURN %
    # =========================================================================
    ax3 = fig.add_subplot(6, 2, 4)
    
    ax3.fill_between(range(len(portfolio_df)), 0, portfolio_df['return_pct'],
                     where=portfolio_df['return_pct'] >= 0, color='#3fb950', alpha=0.3)
    ax3.fill_between(range(len(portfolio_df)), 0, portfolio_df['return_pct'],
                     where=portfolio_df['return_pct'] < 0, color='#f85149', alpha=0.3)
    ax3.plot(portfolio_df['return_pct'], color='#58a6ff', linewidth=1.5)
    ax3.axhline(y=0, color='#8b949e', linestyle='--', linewidth=1)
    
    ax3.set_ylabel('Cumulative Return (%)')
    ax3.set_title('Cumulative Return Over Time', fontsize=14, color='#58a6ff')
    ax3.grid(True, alpha=0.3)
    
    # =========================================================================
    # 4. DRAWDOWN
    # =========================================================================
    ax4 = fig.add_subplot(6, 2, 5)
    
    cummax = portfolio_df['portfolio_value'].cummax()
    drawdown_pct = (portfolio_df['portfolio_value'] - cummax) / cummax * 100
    
    ax4.fill_between(range(len(portfolio_df)), 0, drawdown_pct, color='#f85149', alpha=0.5)
    ax4.plot(drawdown_pct, color='#f85149', linewidth=1)
    
    max_dd_idx = drawdown_pct.idxmin()
    max_dd = drawdown_pct.min()
    ax4.scatter([portfolio_df.index.get_loc(max_dd_idx)], [max_dd], 
                color='#ffa657', s=100, zorder=5, marker='v')
    ax4.annotate(f'Max: {max_dd:.2f}%', 
                 (portfolio_df.index.get_loc(max_dd_idx), max_dd),
                 textcoords='offset points', xytext=(10, -15), 
                 color='#ffa657', fontsize=11, fontweight='bold')
    
    ax4.set_ylabel('Drawdown (%)')
    ax4.set_title('Portfolio Drawdown', fontsize=14, color='#58a6ff')
    ax4.grid(True, alpha=0.3)
    
    # =========================================================================
    # 5. TRADE P&L DISTRIBUTION
    # =========================================================================
    ax5 = fig.add_subplot(6, 2, 6)
    
    sells = trades_df[trades_df['action'].isin(['SELL', 'SELL (Close)'])].copy()
    if 'trade_pnl' in sells.columns and len(sells) > 0:
        pnls = sells['trade_pnl']
        colors = ['#3fb950' if p > 0 else '#f85149' for p in pnls]
        ax5.bar(range(len(pnls)), pnls.values, color=colors, alpha=0.8)
        ax5.axhline(y=0, color='#8b949e', linestyle='-', linewidth=1)
        ax5.axhline(y=pnls.mean(), color='#58a6ff', linestyle='--', linewidth=2,
                    label=f'Avg: ${pnls.mean():,.2f}')
        ax5.legend(loc='upper right')
    
    ax5.set_xlabel('Trade Number')
    ax5.set_ylabel('P&L ($)')
    ax5.set_title('Individual Trade P&L', fontsize=14, color='#58a6ff')
    ax5.grid(True, alpha=0.3)
    
    # =========================================================================
    # 6. WIN/LOSS DISTRIBUTION
    # =========================================================================
    ax6 = fig.add_subplot(6, 2, 7)
    
    if len(sells) > 0 and 'trade_pnl' in sells.columns:
        wins = sells[sells['trade_pnl'] > 0]['trade_pnl']
        losses = sells[sells['trade_pnl'] <= 0]['trade_pnl']
        
        ax6.hist(wins, bins=30, color='#3fb950', alpha=0.7, label=f'Wins (n={len(wins)})')
        ax6.hist(losses, bins=30, color='#f85149', alpha=0.7, label=f'Losses (n={len(losses)})')
        ax6.axvline(x=0, color='#8b949e', linestyle='--', linewidth=2)
        ax6.legend(loc='upper right')
    
    ax6.set_xlabel('Trade P&L ($)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('P&L Distribution', fontsize=14, color='#58a6ff')
    ax6.grid(True, alpha=0.3)
    
    # =========================================================================
    # 7. EQUITY CURVE VS BUY-AND-HOLD
    # =========================================================================
    ax7 = fig.add_subplot(6, 2, 8)
    
    # Calculate buy and hold
    first_price = portfolio_df['price'].iloc[0]
    shares_bh = INITIAL_CAPITAL / first_price
    portfolio_df['buy_hold_value'] = shares_bh * portfolio_df['price']
    
    ax7.plot(portfolio_df['portfolio_value'], color='#58a6ff', linewidth=2, label='Strategy')
    ax7.plot(portfolio_df['buy_hold_value'], color='#ffa657', linewidth=2, alpha=0.7, label='Buy & Hold')
    ax7.axhline(y=INITIAL_CAPITAL, color='#8b949e', linestyle='--', linewidth=1)
    
    ax7.set_ylabel('Portfolio Value ($)')
    ax7.set_title('Strategy vs Buy-and-Hold', fontsize=14, color='#58a6ff')
    ax7.legend(loc='upper left')
    ax7.grid(True, alpha=0.3)
    ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # =========================================================================
    # 8. MONTHLY RETURNS
    # =========================================================================
    ax8 = fig.add_subplot(6, 2, 9)
    
    portfolio_df['month'] = pd.to_datetime(portfolio_df['timestamp']).dt.to_period('M')
    monthly = portfolio_df.groupby('month')['portfolio_value'].last()
    monthly_returns = monthly.pct_change().dropna() * 100
    
    colors = ['#3fb950' if r > 0 else '#f85149' for r in monthly_returns]
    bars = ax8.bar([str(m) for m in monthly_returns.index], monthly_returns.values, 
                   color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    ax8.axhline(y=0, color='#8b949e', linestyle='-', linewidth=1)
    
    for bar, val in zip(bars, monthly_returns.values):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10, 
                color='#3fb950' if val > 0 else '#f85149')
    
    ax8.set_ylabel('Monthly Return (%)')
    ax8.set_title('Monthly Returns', fontsize=14, color='#58a6ff')
    ax8.grid(True, alpha=0.3)
    
    # =========================================================================
    # 9. TRADE HOLDING TIME
    # =========================================================================
    ax9 = fig.add_subplot(6, 2, 10)
    
    if 'hold_time' in sells.columns and len(sells) > 0:
        hold_times = sells['hold_time']
        ax9.hist(hold_times, bins=50, color='#8b5cf6', alpha=0.7, edgecolor='#7c3aed')
        ax9.axvline(x=hold_times.mean(), color='#ffa657', linestyle='--', linewidth=2,
                    label=f'Mean: {hold_times.mean():.1f} min')
        ax9.axvline(x=hold_times.median(), color='#3fb950', linestyle='--', linewidth=2,
                    label=f'Median: {hold_times.median():.1f} min')
        ax9.legend(loc='upper right')
    
    ax9.set_xlabel('Hold Time (minutes)')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Trade Holding Time Distribution', fontsize=14, color='#58a6ff')
    ax9.grid(True, alpha=0.3)
    
    # =========================================================================
    # 10. CUMULATIVE TRADES P&L
    # =========================================================================
    ax10 = fig.add_subplot(6, 2, 11)
    
    if 'trade_pnl' in sells.columns and len(sells) > 0:
        cum_pnl = sells['trade_pnl'].cumsum()
        ax10.fill_between(range(len(cum_pnl)), 0, cum_pnl, 
                          where=cum_pnl >= 0, color='#3fb950', alpha=0.3)
        ax10.fill_between(range(len(cum_pnl)), 0, cum_pnl,
                          where=cum_pnl < 0, color='#f85149', alpha=0.3)
        ax10.plot(cum_pnl.values, color='#58a6ff', linewidth=2)
        ax10.axhline(y=0, color='#8b949e', linestyle='--', linewidth=1)
    
    ax10.set_xlabel('Trade Number')
    ax10.set_ylabel('Cumulative P&L ($)')
    ax10.set_title('Cumulative Trading P&L', fontsize=14, color='#58a6ff')
    ax10.grid(True, alpha=0.3)
    ax10.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # =========================================================================
    # 11. SUMMARY STATISTICS
    # =========================================================================
    ax11 = fig.add_subplot(6, 2, 12)
    ax11.axis('off')
    
    # Calculate additional stats
    bh_final = portfolio_df['buy_hold_value'].iloc[-1]
    bh_return = (bh_final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    stats_text = f"""
╔════════════════════════════════════════════════════════════════════╗
║                      PORTFOLIO PERFORMANCE SUMMARY                  ║
╠════════════════════════════════════════════════════════════════════╣
║  CAPITAL                                                            ║
║    Initial Capital:      ${INITIAL_CAPITAL:>12,.2f}                          ║
║    Final Value:          ${metrics['final_value']:>12,.2f}                          ║
║    Total P&L:            ${metrics['total_pnl']:>+12,.2f}                          ║
║    Total Return:         {metrics['total_return']:>+12.2f}%                          ║
╠════════════════════════════════════════════════════════════════════╣
║  TRADE STATISTICS                                                   ║
║    Total Trades:         {metrics['num_trades']:>12,}                              ║
║    Winning Trades:       {metrics['winning_trades']:>12,}  ({metrics.get('win_rate', 0):>5.1f}%)                 ║
║    Losing Trades:        {metrics['losing_trades']:>12,}                              ║
║    Avg Win:              ${metrics.get('avg_win', 0):>+11,.2f}                          ║
║    Avg Loss:             ${metrics.get('avg_loss', 0):>+11,.2f}                          ║
║    Largest Win:          ${metrics.get('largest_win', 0):>+11,.2f}                          ║
║    Largest Loss:         ${metrics.get('largest_loss', 0):>+11,.2f}                          ║
╠════════════════════════════════════════════════════════════════════╣
║  RISK METRICS                                                       ║
║    Max Drawdown:         {metrics.get('max_drawdown', 0):>12.2f}%                          ║
║    Max DD ($):           ${metrics.get('max_drawdown_dollars', 0):>11,.2f}                          ║
║    Profit Factor:        {metrics.get('profit_factor', 0):>12.2f}                              ║
║    Sharpe Ratio:         {metrics.get('sharpe_ratio', 0):>12.2f}                              ║
║    Avg Hold Time:        {metrics.get('avg_hold_time', 0):>10.1f} min                          ║
╠════════════════════════════════════════════════════════════════════╣
║  VS BUY-AND-HOLD                                                    ║
║    B&H Final Value:      ${bh_final:>12,.2f}                          ║
║    B&H Return:           {bh_return:>+12.2f}%                          ║
║    Alpha:                {metrics['total_return'] - bh_return:>+12.2f}%                          ║
╚════════════════════════════════════════════════════════════════════╝
"""
    
    ax11.text(0.5, 0.5, stats_text, transform=ax11.transAxes, fontsize=11,
              verticalalignment='center', horizontalalignment='center',
              fontfamily='monospace', color='#58a6ff',
              bbox=dict(boxstyle='round', facecolor='#161b22', edgecolor='#30363d'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_file = VIZ_DIR / "06_portfolio_simulation.png"
    plt.savefig(output_file, dpi=150, facecolor='#0d1117', edgecolor='none')
    plt.close()
    
    print(f"\n✓ Saved: {output_file}")
    
    return output_file


def main():
    print("\n" + "="*70)
    print("REALISTIC PORTFOLIO SIMULATION")
    print("="*70 + "\n")
    
    # Load predictions
    pred_file = RESULTS_DIR / "lightgbm_predictions.csv"
    df = pd.read_csv(pred_file, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"Loaded {len(df):,} predictions")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Run simulation
    portfolio_df, trades_df = simulate_portfolio(df)
    
    # Calculate metrics
    metrics = calculate_metrics(portfolio_df, trades_df)
    
    # Print summary
    print("\n" + "="*70)
    print("SIMULATION RESULTS")
    print("="*70)
    print(f"Initial Capital:  ${metrics['initial_capital']:>12,.2f}")
    print(f"Final Value:      ${metrics['final_value']:>12,.2f}")
    print(f"Total P&L:        ${metrics['total_pnl']:>+12,.2f}")
    print(f"Total Return:     {metrics['total_return']:>+12.2f}%")
    print(f"Number of Trades: {metrics['num_trades']:>12,}")
    print(f"Win Rate:         {metrics.get('win_rate', 0):>12.1f}%")
    print(f"Profit Factor:    {metrics.get('profit_factor', 0):>12.2f}")
    print(f"Max Drawdown:     {metrics.get('max_drawdown', 0):>12.2f}%")
    print(f"Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):>12.2f}")
    print("="*70)
    
    # Visualize
    output_file = visualize_portfolio(portfolio_df, trades_df, metrics)
    
    # Save trades to CSV
    trades_file = RESULTS_DIR / "portfolio_trades.csv"
    trades_df.to_csv(trades_file, index=False)
    print(f"✓ Trades saved: {trades_file}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
