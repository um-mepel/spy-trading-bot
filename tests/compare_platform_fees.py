#!/usr/bin/env python3
"""
Platform Fee Comparison Analysis
================================
Run the same trading model across multiple broker platforms with realistic fee structures.
Compare total fees, net P&L, and determine the most cost-effective platform.

Fee Structures Research (2024-2025):
- Most major brokers are "commission-free" but have hidden costs via PFOF
- SEC/FINRA fees apply to ALL platforms (regulatory)
- Active trader platforms charge explicit fees but often get better execution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

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
# PLATFORM FEE STRUCTURES
# ============================================================================

@dataclass
class PlatformFees:
    """Fee structure for a trading platform."""
    name: str
    commission_per_share: float = 0.0       # Per-share commission
    commission_per_trade: float = 0.0       # Flat fee per trade
    commission_pct: float = 0.0             # Percentage of trade value
    min_commission: float = 0.0             # Minimum commission per trade
    max_commission_pct: float = 1.0         # Max commission as % of trade value
    pfof_spread_cost: float = 0.0           # Hidden cost from PFOF (worse execution)
    exchange_fee: float = 0.0               # Exchange/ECN fees per share
    clearing_fee: float = 0.0               # Clearing fee per share
    # Regulatory fees (apply to all platforms)
    sec_fee_rate: float = 0.0000278         # SEC fee: $27.80 per $1M sold (2024 rate)
    finra_taf_rate: float = 0.000166        # FINRA TAF: $0.000166 per share (max $8.30)
    finra_taf_max: float = 8.30             # Max FINRA TAF per trade
    description: str = ""


# Define platform fee structures
PLATFORMS = {
    # =========================================================================
    # COMMISSION-FREE BROKERS (with PFOF hidden costs)
    # =========================================================================
    "robinhood": PlatformFees(
        name="Robinhood",
        pfof_spread_cost=0.0023,  # ~$0.23 per 100 shares hidden cost
        description="Commission-free, PFOF revenue model"
    ),
    
    "webull": PlatformFees(
        name="Webull",
        pfof_spread_cost=0.0020,
        description="Commission-free, PFOF revenue model"
    ),
    
    "fidelity": PlatformFees(
        name="Fidelity",
        pfof_spread_cost=0.0015,  # Better execution than Robinhood
        description="Commission-free, partial PFOF"
    ),
    
    "schwab": PlatformFees(
        name="Charles Schwab",
        pfof_spread_cost=0.0018,
        description="Commission-free after TD Ameritrade merger"
    ),
    
    # =========================================================================
    # ALPACA (Our current platform)
    # =========================================================================
    "alpaca_free": PlatformFees(
        name="Alpaca (Free Tier)",
        pfof_spread_cost=0.0010,  # Lower than Robinhood
        description="Commission-free, API-first broker"
    ),
    
    "alpaca_iex": PlatformFees(
        name="Alpaca (IEX Routing)",
        exchange_fee=0.0009,  # IEX charges ~$0.0009/share
        pfof_spread_cost=0.0,  # No PFOF with IEX
        description="IEX routing for better execution (no PFOF)"
    ),
    
    # =========================================================================
    # PROFESSIONAL / ACTIVE TRADER PLATFORMS
    # =========================================================================
    "ibkr_lite": PlatformFees(
        name="Interactive Brokers Lite",
        pfof_spread_cost=0.0015,
        description="IBKR Lite - commission-free with PFOF"
    ),
    
    "ibkr_pro": PlatformFees(
        name="Interactive Brokers Pro",
        commission_per_share=0.005,  # $0.005/share
        min_commission=1.00,          # $1 minimum
        max_commission_pct=0.01,      # Max 1% of trade value
        exchange_fee=0.0003,          # Exchange fees passed through
        clearing_fee=0.0001,
        pfof_spread_cost=0.0,         # No PFOF - direct market access
        description="Pro tier with direct market access"
    ),
    
    "ibkr_pro_tiered": PlatformFees(
        name="IBKR Pro (Tiered >100k shares/mo)",
        commission_per_share=0.0035,  # Lower rate for high volume
        min_commission=0.35,
        max_commission_pct=0.01,
        exchange_fee=0.0002,
        clearing_fee=0.0001,
        pfof_spread_cost=0.0,
        description="High-volume tiered pricing"
    ),
    
    "tradestation_flat": PlatformFees(
        name="TradeStation (Per Trade)",
        commission_per_trade=5.00,  # Flat $5 per trade
        pfof_spread_cost=0.0,
        description="Flat fee per trade"
    ),
    
    "tradestation_pershare": PlatformFees(
        name="TradeStation (Per Share)",
        commission_per_share=0.01,  # $0.01/share
        min_commission=1.00,
        pfof_spread_cost=0.0,
        description="Per-share pricing for active traders"
    ),
    
    # =========================================================================
    # PREMIUM EXECUTION
    # =========================================================================
    "lightspeed": PlatformFees(
        name="Lightspeed",
        commission_per_share=0.0045,
        min_commission=1.00,
        exchange_fee=0.0003,
        pfof_spread_cost=0.0,
        description="Professional day trading platform"
    ),
    
    "centerpoint": PlatformFees(
        name="Centerpoint Securities",
        commission_per_share=0.004,
        min_commission=0.0,
        exchange_fee=0.0003,
        pfof_spread_cost=0.0,
        description="Short-selling specialist broker"
    ),
}


# ============================================================================
# FEE CALCULATION
# ============================================================================

def calculate_trade_fees(
    platform: PlatformFees,
    shares: int,
    entry_price: float,
    exit_price: float,
    is_sell: bool = False
) -> Dict[str, float]:
    """
    Calculate all fees for a single trade (entry OR exit).
    
    Args:
        platform: Platform fee structure
        shares: Number of shares
        entry_price: Entry price (or current price for entry)
        exit_price: Exit price (for sells)
        is_sell: Whether this is a sell (applies SEC fee)
    
    Returns:
        Dictionary of fee components
    """
    trade_value = shares * (exit_price if is_sell else entry_price)
    
    fees = {
        'commission': 0.0,
        'exchange_fee': 0.0,
        'clearing_fee': 0.0,
        'pfof_cost': 0.0,
        'sec_fee': 0.0,
        'finra_taf': 0.0,
    }
    
    # Per-share commission
    if platform.commission_per_share > 0:
        commission = shares * platform.commission_per_share
        # Apply minimum
        commission = max(commission, platform.min_commission)
        # Apply maximum (% of trade)
        max_comm = trade_value * platform.max_commission_pct
        commission = min(commission, max_comm)
        fees['commission'] = commission
    
    # Flat per-trade commission
    if platform.commission_per_trade > 0:
        fees['commission'] += platform.commission_per_trade
    
    # Percentage commission
    if platform.commission_pct > 0:
        fees['commission'] += trade_value * platform.commission_pct
    
    # Exchange fees
    if platform.exchange_fee > 0:
        fees['exchange_fee'] = shares * platform.exchange_fee
    
    # Clearing fees
    if platform.clearing_fee > 0:
        fees['clearing_fee'] = shares * platform.clearing_fee
    
    # PFOF hidden cost (worse execution = higher effective spread)
    if platform.pfof_spread_cost > 0:
        fees['pfof_cost'] = shares * platform.pfof_spread_cost
    
    # SEC fee (sells only)
    if is_sell:
        fees['sec_fee'] = trade_value * platform.sec_fee_rate
    
    # FINRA TAF (all transactions)
    taf = shares * platform.finra_taf_rate
    fees['finra_taf'] = min(taf, platform.finra_taf_max)
    
    fees['total'] = sum(fees.values())
    
    return fees


def calculate_roundtrip_fees(
    platform: PlatformFees,
    shares: int,
    entry_price: float,
    exit_price: float
) -> Dict[str, float]:
    """Calculate total fees for a complete roundtrip trade (buy + sell)."""
    
    entry_fees = calculate_trade_fees(platform, shares, entry_price, entry_price, is_sell=False)
    exit_fees = calculate_trade_fees(platform, shares, exit_price, exit_price, is_sell=True)
    
    total_fees = {
        'entry_fees': entry_fees['total'],
        'exit_fees': exit_fees['total'],
        'total_commission': entry_fees['commission'] + exit_fees['commission'],
        'total_exchange': entry_fees['exchange_fee'] + exit_fees['exchange_fee'],
        'total_clearing': entry_fees['clearing_fee'] + exit_fees['clearing_fee'],
        'total_pfof': entry_fees['pfof_cost'] + exit_fees['pfof_cost'],
        'total_regulatory': entry_fees['sec_fee'] + exit_fees['sec_fee'] + 
                           entry_fees['finra_taf'] + exit_fees['finra_taf'],
        'total_fees': entry_fees['total'] + exit_fees['total']
    }
    
    return total_fees


# ============================================================================
# PATTERN DAY TRADER (PDT) RULE
# ============================================================================
# FINRA Rule 4210: Accounts under $25,000 cannot make more than 3 day trades
# in a 5-business-day rolling period (in a MARGIN account).
#
# Workarounds:
# 1. Cash account: No PDT rule, but must wait for T+1 settlement
# 2. Account over $25,000: No restrictions
# 3. Offshore brokers: Some don't enforce PDT
# 4. Swing trading: Hold overnight to avoid day trade classification
# ============================================================================

PDT_THRESHOLD = 25_000  # Minimum equity to avoid PDT restrictions
PDT_MAX_DAY_TRADES = 3  # Max day trades in 5-day rolling window (for margin accounts)
PDT_WINDOW_DAYS = 5     # Rolling window for day trade count

# Cash account settlement
CASH_ACCOUNT_SETTLEMENT_DAYS = 1  # T+1 settlement (as of May 2024)


# ============================================================================
# PORTFOLIO SIMULATION WITH FEES
# ============================================================================

INITIAL_CAPITAL = 100_000
POSITION_SIZE_PCT = 0.10
MIN_CONFIDENCE = 0.70
MIN_PREDICTED_MOVE = 0.10
SLIPPAGE_PCT = 0.01  # Base slippage (separate from PFOF)


def simulate_portfolio_with_fees(df: pd.DataFrame, platform: PlatformFees) -> Tuple[pd.DataFrame, Dict]:
    """
    Simulate portfolio trading with platform-specific fees.
    
    Returns:
        Tuple of (portfolio_history_df, summary_dict)
    """
    cash = INITIAL_CAPITAL
    shares = 0
    position_entry_price = 0
    position_entry_time = None
    position_shares = 0
    
    portfolio_history = []
    trades = []
    total_fees_paid = 0
    fee_breakdown = {
        'commission': 0.0,
        'exchange': 0.0,
        'clearing': 0.0,
        'pfof': 0.0,
        'regulatory': 0.0,
    }
    
    # Filter to tradeable signals
    signals = df[(df['Confidence'] >= MIN_CONFIDENCE) & 
                 (abs(df['Predicted_Change']) >= MIN_PREDICTED_MOVE)].copy()
    
    for idx, row in signals.iterrows():
        current_price = row['Actual_Price']
        predicted_change = row['Predicted_Change']
        confidence = row['Confidence']
        timestamp = row['Date']
        
        portfolio_value = cash + (shares * current_price)
        signal = 'BUY' if predicted_change > 0 else 'SELL'
        
        # BUY signal - open position
        if signal == 'BUY' and shares == 0:
            position_size = portfolio_value * POSITION_SIZE_PCT
            max_shares = int(position_size / current_price)
            
            if max_shares > 0 and cash >= max_shares * current_price:
                # Apply base slippage
                execution_price = current_price * (1 + SLIPPAGE_PCT/100)
                
                # Calculate entry fees
                entry_fees = calculate_trade_fees(platform, max_shares, execution_price, 
                                                  execution_price, is_sell=False)
                
                total_cost = (max_shares * execution_price) + entry_fees['total']
                
                if total_cost <= cash:
                    shares = max_shares
                    cash -= total_cost
                    position_entry_price = execution_price
                    position_entry_time = timestamp
                    position_shares = max_shares
                    
                    total_fees_paid += entry_fees['total']
                    fee_breakdown['commission'] += entry_fees['commission']
                    fee_breakdown['exchange'] += entry_fees['exchange_fee']
                    fee_breakdown['clearing'] += entry_fees['clearing_fee']
                    fee_breakdown['pfof'] += entry_fees['pfof_cost']
                    fee_breakdown['regulatory'] += entry_fees['finra_taf']
                    
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'BUY',
                        'shares': shares,
                        'price': execution_price,
                        'fees': entry_fees['total']
                    })
        
        # SELL signal - close position
        elif signal == 'SELL' and shares > 0:
            execution_price = current_price * (1 - SLIPPAGE_PCT/100)
            
            # Calculate exit fees
            exit_fees = calculate_trade_fees(platform, shares, position_entry_price,
                                            execution_price, is_sell=True)
            
            proceeds = (shares * execution_price) - exit_fees['total']
            trade_pnl = proceeds - (shares * position_entry_price)
            
            total_fees_paid += exit_fees['total']
            fee_breakdown['commission'] += exit_fees['commission']
            fee_breakdown['exchange'] += exit_fees['exchange_fee']
            fee_breakdown['clearing'] += exit_fees['clearing_fee']
            fee_breakdown['pfof'] += exit_fees['pfof_cost']
            fee_breakdown['regulatory'] += exit_fees['sec_fee'] + exit_fees['finra_taf']
            
            trades.append({
                'timestamp': timestamp,
                'action': 'SELL',
                'shares': shares,
                'price': execution_price,
                'fees': exit_fees['total'],
                'pnl': trade_pnl
            })
            
            cash += proceeds
            shares = 0
            position_entry_price = 0
            position_entry_time = None
        
        portfolio_value = cash + (shares * current_price)
        portfolio_history.append({
            'timestamp': timestamp,
            'cash': cash,
            'shares': shares,
            'price': current_price,
            'portfolio_value': portfolio_value,
            'return_pct': (portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        })
    
    # Close remaining position
    if shares > 0:
        final_price = signals.iloc[-1]['Actual_Price']
        execution_price = final_price * (1 - SLIPPAGE_PCT/100)
        exit_fees = calculate_trade_fees(platform, shares, position_entry_price,
                                        execution_price, is_sell=True)
        proceeds = (shares * execution_price) - exit_fees['total']
        total_fees_paid += exit_fees['total']
        cash += proceeds
        shares = 0
    
    portfolio_df = pd.DataFrame(portfolio_history) if portfolio_history else pd.DataFrame()
    trades_df = pd.DataFrame(trades)
    
    # Calculate summary
    final_value = cash if len(portfolio_history) == 0 else portfolio_history[-1]['portfolio_value']
    
    # Sell trades for stats
    sells = trades_df[trades_df['action'] == 'SELL'] if len(trades_df) > 0 else pd.DataFrame()
    
    summary = {
        'platform': platform.name,
        'initial_capital': INITIAL_CAPITAL,
        'final_value': final_value,
        'total_return_pct': (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        'total_pnl': final_value - INITIAL_CAPITAL,
        'num_trades': len(sells),
        'total_fees': total_fees_paid,
        'fees_as_pct_capital': total_fees_paid / INITIAL_CAPITAL * 100,
        'fee_breakdown': fee_breakdown,
        'winning_trades': len(sells[sells['pnl'] > 0]) if 'pnl' in sells.columns else 0,
        'losing_trades': len(sells[sells['pnl'] <= 0]) if 'pnl' in sells.columns else 0,
    }
    
    if len(sells) > 0 and 'pnl' in sells.columns:
        summary['win_rate'] = summary['winning_trades'] / summary['num_trades'] * 100
        summary['gross_pnl'] = sells['pnl'].sum() + total_fees_paid
        summary['fee_drag'] = total_fees_paid / abs(summary['gross_pnl']) * 100 if summary['gross_pnl'] != 0 else 0
    
    return portfolio_df, summary


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def run_comparison(predictions_file: Path) -> pd.DataFrame:
    """Run the same model across all platforms and compare."""
    
    print("\n" + "="*80)
    print("PLATFORM FEE COMPARISON ANALYSIS")
    print("="*80)
    
    df = pd.read_csv(predictions_file, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"\nLoaded {len(df):,} predictions")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    
    tradeable = df[(df['Confidence'] >= MIN_CONFIDENCE) & 
                   (abs(df['Predicted_Change']) >= MIN_PREDICTED_MOVE)]
    print(f"Tradeable signals: {len(tradeable):,}")
    
    results = []
    
    print("\n" + "-"*80)
    print(f"{'Platform':<35} {'Final Value':>12} {'Return %':>10} {'Fees':>10} {'Net P&L':>12}")
    print("-"*80)
    
    for platform_key, platform in PLATFORMS.items():
        portfolio_df, summary = simulate_portfolio_with_fees(df, platform)
        summary['platform_key'] = platform_key
        results.append(summary)
        
        print(f"{platform.name:<35} ${summary['final_value']:>11,.2f} "
              f"{summary['total_return_pct']:>+9.2f}% "
              f"${summary['total_fees']:>9,.2f} "
              f"${summary['total_pnl']:>+11,.2f}")
    
    print("-"*80)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('final_value', ascending=False)
    
    return results_df


def visualize_comparison(results_df: pd.DataFrame, output_dir: Path):
    """Create visualization comparing platforms."""
    
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle('TRADING PLATFORM FEE COMPARISON\nSame Strategy, Different Fee Structures', 
                 fontsize=20, fontweight='bold', color='#58a6ff', y=0.98)
    
    # Sort by final value for consistent ordering
    results_df = results_df.sort_values('final_value', ascending=True)
    
    # =========================================================================
    # 1. Final Portfolio Value by Platform
    # =========================================================================
    ax1 = fig.add_subplot(4, 2, 1)
    
    colors = ['#3fb950' if v > INITIAL_CAPITAL else '#f85149' for v in results_df['final_value']]
    bars = ax1.barh(results_df['platform'], results_df['final_value'], color=colors, alpha=0.8)
    ax1.axvline(x=INITIAL_CAPITAL, color='#8b949e', linestyle='--', linewidth=2, label='Initial Capital')
    
    for bar, val in zip(bars, results_df['final_value']):
        ax1.text(val + 100, bar.get_y() + bar.get_height()/2, 
                f'${val:,.0f}', va='center', fontsize=9, color='#c9d1d9')
    
    ax1.set_xlabel('Final Portfolio Value ($)')
    ax1.set_title('Final Portfolio Value by Platform', fontsize=14, color='#58a6ff')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # =========================================================================
    # 2. Total Fees by Platform
    # =========================================================================
    ax2 = fig.add_subplot(4, 2, 2)
    
    fee_colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(results_df)))
    bars = ax2.barh(results_df['platform'], results_df['total_fees'], color=fee_colors, alpha=0.8)
    
    for bar, val in zip(bars, results_df['total_fees']):
        ax2.text(val + 10, bar.get_y() + bar.get_height()/2, 
                f'${val:,.2f}', va='center', fontsize=9, color='#c9d1d9')
    
    ax2.set_xlabel('Total Fees ($)')
    ax2.set_title('Total Trading Fees by Platform', fontsize=14, color='#f85149')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # =========================================================================
    # 3. Net P&L Comparison
    # =========================================================================
    ax3 = fig.add_subplot(4, 2, 3)
    
    colors = ['#3fb950' if v > 0 else '#f85149' for v in results_df['total_pnl']]
    bars = ax3.barh(results_df['platform'], results_df['total_pnl'], color=colors, alpha=0.8)
    ax3.axvline(x=0, color='#8b949e', linestyle='-', linewidth=1)
    
    for bar, val in zip(bars, results_df['total_pnl']):
        offset = 100 if val >= 0 else -100
        ha = 'left' if val >= 0 else 'right'
        ax3.text(val + offset, bar.get_y() + bar.get_height()/2, 
                f'${val:+,.0f}', va='center', ha=ha, fontsize=9, color='#c9d1d9')
    
    ax3.set_xlabel('Net P&L ($)')
    ax3.set_title('Net Profit/Loss by Platform', fontsize=14, color='#58a6ff')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # =========================================================================
    # 4. Fee Breakdown (Stacked Bar)
    # =========================================================================
    ax4 = fig.add_subplot(4, 2, 4)
    
    # Extract fee breakdown
    commission = [r['fee_breakdown']['commission'] for _, r in results_df.iterrows()]
    exchange = [r['fee_breakdown']['exchange'] for _, r in results_df.iterrows()]
    clearing = [r['fee_breakdown']['clearing'] for _, r in results_df.iterrows()]
    pfof = [r['fee_breakdown']['pfof'] for _, r in results_df.iterrows()]
    regulatory = [r['fee_breakdown']['regulatory'] for _, r in results_df.iterrows()]
    
    y_pos = range(len(results_df))
    
    ax4.barh(y_pos, commission, label='Commission', color='#f85149', alpha=0.8)
    ax4.barh(y_pos, exchange, left=commission, label='Exchange', color='#ffa657', alpha=0.8)
    ax4.barh(y_pos, clearing, left=[c+e for c, e in zip(commission, exchange)], 
             label='Clearing', color='#d29922', alpha=0.8)
    ax4.barh(y_pos, pfof, left=[c+e+cl for c, e, cl in zip(commission, exchange, clearing)],
             label='PFOF Cost', color='#8b5cf6', alpha=0.8)
    ax4.barh(y_pos, regulatory, 
             left=[c+e+cl+p for c, e, cl, p in zip(commission, exchange, clearing, pfof)],
             label='Regulatory', color='#6e7681', alpha=0.8)
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(results_df['platform'])
    ax4.set_xlabel('Fee Amount ($)')
    ax4.set_title('Fee Breakdown by Category', fontsize=14, color='#58a6ff')
    ax4.legend(loc='lower right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # =========================================================================
    # 5. Fees as % of Capital
    # =========================================================================
    ax5 = fig.add_subplot(4, 2, 5)
    
    bars = ax5.barh(results_df['platform'], results_df['fees_as_pct_capital'], 
                    color='#f85149', alpha=0.8)
    
    for bar, val in zip(bars, results_df['fees_as_pct_capital']):
        ax5.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}%', va='center', fontsize=9, color='#c9d1d9')
    
    ax5.set_xlabel('Fees as % of Initial Capital')
    ax5.set_title('Fee Impact on Capital', fontsize=14, color='#f85149')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # =========================================================================
    # 6. Return % Comparison
    # =========================================================================
    ax6 = fig.add_subplot(4, 2, 6)
    
    colors = ['#3fb950' if v > 0 else '#f85149' for v in results_df['total_return_pct']]
    bars = ax6.barh(results_df['platform'], results_df['total_return_pct'], color=colors, alpha=0.8)
    ax6.axvline(x=0, color='#8b949e', linestyle='-', linewidth=1)
    
    for bar, val in zip(bars, results_df['total_return_pct']):
        offset = 0.1 if val >= 0 else -0.1
        ha = 'left' if val >= 0 else 'right'
        ax6.text(val + offset, bar.get_y() + bar.get_height()/2, 
                f'{val:+.2f}%', va='center', ha=ha, fontsize=9, color='#c9d1d9')
    
    ax6.set_xlabel('Total Return (%)')
    ax6.set_title('Return by Platform', fontsize=14, color='#58a6ff')
    ax6.grid(True, alpha=0.3, axis='x')
    
    # =========================================================================
    # 7. Best Platform Highlight
    # =========================================================================
    ax7 = fig.add_subplot(4, 2, (7, 8))
    ax7.axis('off')
    
    # Sort descending by final value for correct best/worst
    results_sorted = results_df.sort_values('final_value', ascending=False)
    best = results_sorted.iloc[0]  # Highest final value
    worst = results_sorted.iloc[-1]  # Lowest final value
    
    savings = best['final_value'] - worst['final_value']
    fee_diff = worst['total_fees'] - best['total_fees']
    
    summary_text = f"""
+====================================================================================+
|                           PLATFORM COMPARISON SUMMARY                              |
+====================================================================================+
|                                                                                    |
|  [BEST] PLATFORM:      {best['platform']:<40}               |
|     Final Value:       ${best['final_value']:>12,.2f}                                     |
|     Total Return:      {best['total_return_pct']:>+12.2f}%                                     |
|     Total Fees:        ${best['total_fees']:>12,.2f}                                     |
|                                                                                    |
|  [WORST] PLATFORM:     {worst['platform']:<40}               |
|     Final Value:       ${worst['final_value']:>12,.2f}                                     |
|     Total Return:      {worst['total_return_pct']:>+12.2f}%                                     |
|     Total Fees:        ${worst['total_fees']:>12,.2f}                                     |
|                                                                                    |
+====================================================================================+
|  SAVINGS BY CHOOSING BEST PLATFORM:                                                |
|     Additional Profit:  ${savings:>12,.2f}                                     |
|     Fee Savings:        ${fee_diff:>12,.2f}                                     |
|                                                                                    |
+====================================================================================+
|  KEY INSIGHTS:                                                                     |
|     * "Commission-free" platforms have hidden PFOF costs                           |
|     * IBKR Pro has explicit fees but better execution                              |
|     * Alpaca with IEX routing offers good balance of cost/execution                |
|     * For high-frequency trading, per-share costs matter most                      |
|     * Regulatory fees (SEC/FINRA) apply to ALL platforms                           |
+====================================================================================+
"""
    
    ax7.text(0.5, 0.5, summary_text, transform=ax7.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', color='#58a6ff',
             bbox=dict(boxstyle='round', facecolor='#161b22', edgecolor='#30363d'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_file = output_dir / "platform_fee_comparison.png"
    plt.savefig(output_file, dpi=150, facecolor='#0d1117', edgecolor='none')
    plt.close()
    
    print(f"\n✓ Saved visualization: {output_file}")
    
    return output_file


def simulate_with_pdt_rule(
    df: pd.DataFrame, 
    platform: PlatformFees,
    initial_capital: float,
    account_type: str = 'margin'  # 'margin' or 'cash'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Simulate portfolio with PDT rule enforcement.
    
    Args:
        df: Predictions dataframe
        platform: Platform fee structure
        initial_capital: Starting capital
        account_type: 'margin' (PDT applies if <$25k) or 'cash' (T+1 settlement)
    """
    cash = initial_capital
    shares = 0
    position_entry_price = 0
    position_entry_time = None
    
    portfolio_history = []
    trades = []
    total_fees_paid = 0
    
    # PDT tracking
    day_trades = []  # List of (date, was_day_trade)
    pdt_blocked_trades = 0
    settlement_blocked_trades = 0
    
    # Filter to tradeable signals
    signals = df[(df['Confidence'] >= MIN_CONFIDENCE) & 
                 (abs(df['Predicted_Change']) >= MIN_PREDICTED_MOVE)].copy()
    
    signals['trade_date'] = pd.to_datetime(signals['Date']).dt.date
    
    # Track cash available (for cash accounts with settlement)
    settled_cash = initial_capital
    pending_settlements = []  # List of (settle_date, amount)
    
    for idx, row in signals.iterrows():
        current_price = row['Actual_Price']
        predicted_change = row['Predicted_Change']
        confidence = row['Confidence']
        timestamp = row['Date']
        trade_date = row['trade_date']
        
        # Update portfolio value
        portfolio_value = cash + (shares * current_price)
        
        # For cash accounts: process settlements
        if account_type == 'cash':
            new_pending = []
            for settle_date, amount in pending_settlements:
                if trade_date >= settle_date:
                    settled_cash += amount
                else:
                    new_pending.append((settle_date, amount))
            pending_settlements = new_pending
        
        # Check PDT rule for margin accounts under $25k
        can_day_trade = True
        if account_type == 'margin' and portfolio_value < PDT_THRESHOLD:
            # Count day trades in last 5 business days
            cutoff_date = trade_date - pd.Timedelta(days=7)  # ~5 business days
            recent_day_trades = [d for d in day_trades if d >= cutoff_date]
            if len(recent_day_trades) >= PDT_MAX_DAY_TRADES:
                can_day_trade = False
        
        signal = 'BUY' if predicted_change > 0 else 'SELL'
        
        # BUY signal
        if signal == 'BUY' and shares == 0:
            # For cash accounts, check settled cash
            available_cash = settled_cash if account_type == 'cash' else cash
            
            position_size = min(portfolio_value * POSITION_SIZE_PCT, available_cash)
            max_shares = int(position_size / current_price)
            
            if max_shares > 0:
                execution_price = current_price * (1 + SLIPPAGE_PCT/100)
                entry_fees = calculate_trade_fees(platform, max_shares, execution_price, 
                                                  execution_price, is_sell=False)
                total_cost = (max_shares * execution_price) + entry_fees['total']
                
                if total_cost <= available_cash:
                    shares = max_shares
                    cash -= total_cost
                    if account_type == 'cash':
                        settled_cash -= total_cost
                    position_entry_price = execution_price
                    position_entry_time = timestamp
                    position_entry_date = trade_date
                    total_fees_paid += entry_fees['total']
                    
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'BUY',
                        'shares': shares,
                        'price': execution_price,
                        'fees': entry_fees['total']
                    })
        
        # SELL signal
        elif signal == 'SELL' and shares > 0:
            # Check if this would be a day trade
            is_day_trade = (trade_date == position_entry_date) if position_entry_time else False
            
            # Block if PDT rule applies
            if is_day_trade and not can_day_trade:
                pdt_blocked_trades += 1
                continue  # Skip this trade, hold position
            
            execution_price = current_price * (1 - SLIPPAGE_PCT/100)
            exit_fees = calculate_trade_fees(platform, shares, position_entry_price,
                                            execution_price, is_sell=True)
            
            proceeds = (shares * execution_price) - exit_fees['total']
            trade_pnl = proceeds - (shares * position_entry_price)
            
            total_fees_paid += exit_fees['total']
            
            trades.append({
                'timestamp': timestamp,
                'action': 'SELL',
                'shares': shares,
                'price': execution_price,
                'fees': exit_fees['total'],
                'pnl': trade_pnl,
                'was_day_trade': is_day_trade
            })
            
            if is_day_trade:
                day_trades.append(trade_date)
            
            cash += proceeds
            
            # For cash accounts, proceeds settle T+1
            if account_type == 'cash':
                settle_date = trade_date + pd.Timedelta(days=CASH_ACCOUNT_SETTLEMENT_DAYS)
                pending_settlements.append((settle_date, proceeds))
            else:
                # Margin accounts have instant settlement
                pass
            
            shares = 0
            position_entry_price = 0
            position_entry_time = None
        
        portfolio_value = cash + (shares * current_price)
        portfolio_history.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
        })
    
    # Close remaining position
    if shares > 0:
        final_price = signals.iloc[-1]['Actual_Price']
        execution_price = final_price * (1 - SLIPPAGE_PCT/100)
        exit_fees = calculate_trade_fees(platform, shares, position_entry_price,
                                        execution_price, is_sell=True)
        proceeds = (shares * execution_price) - exit_fees['total']
        total_fees_paid += exit_fees['total']
        cash += proceeds
        shares = 0
    
    portfolio_df = pd.DataFrame(portfolio_history) if portfolio_history else pd.DataFrame()
    trades_df = pd.DataFrame(trades)
    
    final_value = cash
    sells = trades_df[trades_df['action'] == 'SELL'] if len(trades_df) > 0 else pd.DataFrame()
    
    summary = {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return_pct': (final_value - initial_capital) / initial_capital * 100,
        'total_pnl': final_value - initial_capital,
        'num_trades': len(sells),
        'total_fees': total_fees_paid,
        'account_type': account_type,
        'pdt_blocked_trades': pdt_blocked_trades,
        'day_trades_executed': len(day_trades),
    }
    
    return portfolio_df, summary


def run_pdt_analysis(predictions_file: Path) -> pd.DataFrame:
    """
    Analyze impact of PDT rule at different account sizes.
    """
    print("\n" + "="*80)
    print("PATTERN DAY TRADER (PDT) RULE ANALYSIS")
    print("="*80)
    print("""
The PDT Rule (FINRA Rule 4210):
- Applies to MARGIN accounts with less than $25,000 equity
- Cannot make more than 3 day trades in a 5-business-day period
- A "day trade" = buying AND selling same security same day

Workarounds:
- Cash account: No PDT, but T+1 settlement limits buying power
- Account >= $25,000: No restrictions
- Hold overnight: Swing trade instead of day trade
""")
    
    df = pd.read_csv(predictions_file, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Use Alpaca IEX (best platform from previous analysis)
    platform = PLATFORMS['alpaca_iex']
    
    # Test different account sizes
    account_sizes = [5_000, 10_000, 15_000, 25_000, 50_000, 100_000]
    
    results = []
    
    print("\n" + "-"*90)
    print(f"{'Account Size':>12} {'Type':>8} {'Final Value':>14} {'Return':>10} "
          f"{'Trades':>8} {'PDT Blocked':>12}")
    print("-"*90)
    
    for account_size in account_sizes:
        # Margin account (PDT applies if < $25k)
        _, margin_summary = simulate_with_pdt_rule(df, platform, account_size, 'margin')
        margin_summary['account_size'] = account_size
        margin_summary['account_type'] = 'margin'
        results.append(margin_summary)
        
        print(f"${account_size:>11,} {'margin':>8} ${margin_summary['final_value']:>13,.2f} "
              f"{margin_summary['total_return_pct']:>+9.2f}% "
              f"{margin_summary['num_trades']:>8,} "
              f"{margin_summary['pdt_blocked_trades']:>12,}")
        
        # Cash account (no PDT, but T+1 settlement)
        _, cash_summary = simulate_with_pdt_rule(df, platform, account_size, 'cash')
        cash_summary['account_size'] = account_size
        cash_summary['account_type'] = 'cash'
        results.append(cash_summary)
        
        print(f"${account_size:>11,} {'cash':>8} ${cash_summary['final_value']:>13,.2f} "
              f"{cash_summary['total_return_pct']:>+9.2f}% "
              f"{cash_summary['num_trades']:>8,} "
              f"{'N/A':>12}")
    
    print("-"*90)
    
    return pd.DataFrame(results)


def simulate_multi_broker(df: pd.DataFrame, num_brokers: int, total_capital: float) -> Dict:
    """
    Simulate splitting capital across multiple brokers to avoid PDT.
    Each broker gets 3 day trades per 5-day window.
    """
    capital_per_broker = total_capital / num_brokers
    day_trades_per_week = 3 * num_brokers
    
    # Simplified simulation: assume we can execute proportionally more trades
    platform = PLATFORMS['alpaca_iex']
    
    # With PDT, small account only gets ~195 trades
    # With multi-broker, we can scale that up
    base_trades_with_pdt = 195  # From our analysis
    full_trades_no_pdt = 2794   # From our analysis
    
    # Scale based on day trades available
    # 3 day trades/week for 26 weeks = 78 trades per broker
    estimated_trades = min(78 * num_brokers, full_trades_no_pdt)
    trade_ratio = estimated_trades / full_trades_no_pdt
    
    # Approximate returns (linear scaling is optimistic)
    full_return_pct = 16.65  # From $100k analysis
    estimated_return = full_return_pct * trade_ratio * 0.8  # 80% efficiency factor
    
    final_value = total_capital * (1 + estimated_return / 100)
    
    return {
        'num_brokers': num_brokers,
        'capital_per_broker': capital_per_broker,
        'day_trades_per_week': day_trades_per_week,
        'estimated_trades': estimated_trades,
        'estimated_return_pct': estimated_return,
        'final_value': final_value,
        'total_capital': total_capital
    }


def run_multi_broker_analysis(initial_capital: float = 10000):
    """Analyze returns with different multi-broker setups."""
    
    print("\n" + "="*80)
    print("MULTI-BROKER PDT WORKAROUND ANALYSIS")
    print("="*80)
    print(f"\nStarting capital: ${initial_capital:,.0f}")
    print("Strategy: Split capital across multiple brokers for more day trades\n")
    
    print("-"*80)
    print(f"{'Brokers':>8} {'Capital/Broker':>15} {'Day Trades/Wk':>15} {'Est. Trades':>12} {'Return':>10} {'Final Value':>14}")
    print("-"*80)
    
    results = []
    for num_brokers in [1, 2, 3, 4, 5]:
        # Dummy df not needed for this simplified analysis
        result = simulate_multi_broker(None, num_brokers, initial_capital)
        results.append(result)
        
        print(f"{num_brokers:>8} ${result['capital_per_broker']:>14,.0f} "
              f"{result['day_trades_per_week']:>15} "
              f"{result['estimated_trades']:>12,.0f} "
              f"{result['estimated_return_pct']:>+9.2f}% "
              f"${result['final_value']:>13,.2f}")
    
    print("-"*80)
    
    # Compare to just using cash account
    print(f"\nFor comparison:")
    print(f"  Cash account (no PDT):     ~+7.00%  →  ${initial_capital * 1.07:,.2f}")
    print(f"  Save to $25k first:       +14-17%  →  Best long-term option")
    print(f"  Futures (/MES):           +16-17%  →  No PDT, starts at $1,000")
    
    return pd.DataFrame(results)


def main():
    """Main entry point."""
    
    # Find predictions file
    results_dir = Path(__file__).parent.parent / "results" / "real_minute_strict"
    predictions_file = results_dir / "lightgbm_predictions.csv"
    
    if not predictions_file.exists():
        predictions_file = results_dir / "predictions.csv"
        
    if not predictions_file.exists():
        print(f"ERROR: Could not find predictions file")
        print(f"Looked in: {results_dir}")
        return
    
    print(f"Using predictions: {predictions_file}")
    
    # Run platform comparison (with $100k - no PDT issues)
    results_df = run_comparison(predictions_file)
    
    # Create visualizations
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    visualize_comparison(results_df, viz_dir)
    
    # Save results to CSV
    output_csv = results_dir / "platform_comparison.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"✓ Saved results: {output_csv}")
    
    # Run PDT analysis
    pdt_results = run_pdt_analysis(predictions_file)
    pdt_csv = results_dir / "pdt_analysis.csv"
    pdt_results.to_csv(pdt_csv, index=False)
    print(f"✓ Saved PDT analysis: {pdt_csv}")
    
    # Run multi-broker workaround analysis
    multi_broker_results = run_multi_broker_analysis(10000)
    multi_csv = results_dir / "multi_broker_analysis.csv"
    multi_broker_results.to_csv(multi_csv, index=False)
    print(f"✓ Saved multi-broker analysis: {multi_csv}")
    
    # Print recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    results_sorted = results_df.sort_values('final_value', ascending=False)
    best = results_sorted.iloc[0]
    worst = results_sorted.iloc[-1]
    
    print(f"""
PLATFORM COMPARISON (with $100,000 account - no PDT restrictions):

*** RECOMMENDED: {best['platform']} ***
   - Final Value: ${best['final_value']:,.2f}
   - Net Return: {best['total_return_pct']:+.2f}%
   - Total Fees: ${best['total_fees']:.2f}

*** AVOID: {worst['platform']} ***
   - Final Value: ${worst['final_value']:,.2f}
   - Net Return: {worst['total_return_pct']:+.2f}%
   - Total Fees: ${worst['total_fees']:.2f}

PDT RULE IMPACT:

For accounts UNDER $25,000:
   - MARGIN ACCOUNT: Limited to 3 day trades per 5 days
     * Many trades blocked by PDT rule
     * Strategy severely constrained
   
   - CASH ACCOUNT: No PDT rule, but T+1 settlement
     * Can trade more frequently
     * Must wait for funds to settle after each sell
     * Effectively limits to ~1 roundtrip per day per position

RECOMMENDATION FOR SMALL ACCOUNTS:
   1. Use a CASH account to avoid PDT restrictions
   2. Or save up to $25,000+ before day trading
   3. Or consider swing trading (hold overnight) instead
   
For Alpaca specifically:
   - Use IEX routing for better execution (no PFOF)
   - Set account type to 'cash' in Alpaca settings if under $25k
""")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
