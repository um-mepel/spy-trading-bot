#!/usr/bin/env python3
"""
REALISTIC Backtest: No compounding leverage, proper position sizing
"""
import pandas as pd
import numpy as np

print("="*70)
print("REALISTIC BACKTEST: Proper Position Sizing (No Hidden Leverage)")
print("="*70)

df = pd.read_csv('/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/results/extended_middle_deciles/all_trades.csv')
df['Entry_Date'] = pd.to_datetime(df['Entry_Date'])
df['Exit_Date'] = pd.to_datetime(df['Exit_Date'])

# Constants
POSITION_SIZE = 0.04  # 4% per position
MAX_POSITIONS = 25     # 5 per day * 5 days hold = 25 max
MAX_CAPITAL = 1.0      # 100% of portfolio

print(f"\nConfiguration:")
print(f"  Position size: {POSITION_SIZE*100:.0f}%")
print(f"  Max positions: {MAX_POSITIONS}")
print(f"  Max capital deployed: {MAX_CAPITAL*100:.0f}%")

# Method 1: ORIGINAL (potentially uses leverage via compounding)
print("\n" + "="*70)
print("METHOD 1: Original Calculation (what was reported)")
print("="*70)

daily_contrib = df.groupby('Exit_Date')['Return_Pct'].apply(lambda x: (x * POSITION_SIZE).sum() / 100)
portfolio_original = 100 * (1 + daily_contrib).cumprod()
print(f"Final value: ${portfolio_original.iloc[-1]:.2f} ({portfolio_original.iloc[-1]-100:.1f}% return)")

# Method 2: REALISTIC (cap position value at time of entry)
print("\n" + "="*70)
print("METHOD 2: Realistic (Position value based on portfolio at ENTRY)")
print("="*70)

# Proper portfolio simulation
dates = pd.date_range(df['Entry_Date'].min(), df['Exit_Date'].max(), freq='B')
portfolio_value = 100.0
open_positions = []  # Each position tracks: entry_value, return_pct, exit_date

portfolio_history = []

for date in dates:
    # Close positions exiting today
    positions_to_close = [p for p in open_positions if p['exit_date'] == date]
    for p in positions_to_close:
        # PnL is based on the position size AT ENTRY, not current portfolio
        pnl = p['entry_value'] * (p['return_pct'] / 100)
        portfolio_value += pnl
    
    # Remove closed positions
    open_positions = [p for p in open_positions if p['exit_date'] != date]
    
    # Add new positions entering today
    new_trades = df[df['Entry_Date'] == date]
    
    # Calculate how much capital we can deploy
    currently_deployed = sum(p['entry_value'] for p in open_positions)
    available_capital = portfolio_value - currently_deployed
    
    for _, trade in new_trades.iterrows():
        # Position size is 4% of CURRENT portfolio (what we'd realistically do)
        position_value = portfolio_value * POSITION_SIZE
        
        # BUT cap it if we're running out of capital
        if currently_deployed + position_value > portfolio_value * MAX_CAPITAL:
            position_value = max(0, portfolio_value * MAX_CAPITAL - currently_deployed)
        
        if position_value > 0:
            open_positions.append({
                'entry_value': position_value,
                'return_pct': trade['Return_Pct'],
                'exit_date': trade['Exit_Date']
            })
            currently_deployed += position_value
    
    portfolio_history.append({'date': date, 'value': portfolio_value})

print(f"Final value: ${portfolio_value:.2f} ({portfolio_value-100:.1f}% return)")

# Method 3: ULTRA-CONSERVATIVE (Fixed position size, cash drag)
print("\n" + "="*70)
print("METHOD 3: Ultra-Conservative (Account for cash drag)")
print("="*70)

# Start with 100, use fixed 4% positions, track cash separately
portfolio_value = 100.0
cash = 100.0
invested = 0.0
open_positions = []

for date in dates:
    # Close positions
    positions_to_close = [p for p in open_positions if p['exit_date'] == date]
    for p in positions_to_close:
        pnl = p['entry_value'] * (p['return_pct'] / 100)
        cash += p['entry_value'] + pnl  # Return principal + profit/loss
        invested -= p['entry_value']
    
    open_positions = [p for p in open_positions if p['exit_date'] != date]
    
    # Add new positions
    new_trades = df[df['Entry_Date'] == date]
    
    for _, trade in new_trades.iterrows():
        position_value = (cash + invested) * POSITION_SIZE  # 4% of total portfolio
        
        if cash >= position_value:
            cash -= position_value
            invested += position_value
            open_positions.append({
                'entry_value': position_value,
                'return_pct': trade['Return_Pct'],
                'exit_date': trade['Exit_Date']
            })

# Final: close all remaining
for p in open_positions:
    pnl = p['entry_value'] * (p['return_pct'] / 100)
    cash += p['entry_value'] + pnl

portfolio_value = cash
print(f"Final value: ${portfolio_value:.2f} ({portfolio_value-100:.1f}% return)")

# Method 4: WHAT IF WIN RATE WAS RANDOM?
print("\n" + "="*70)
print("METHOD 4: Null Hypothesis (Random 50% Win Rate)")
print("="*70)

# If the model had NO predictive power, what would returns be?
avg_winner = df[df['Win'] == True]['Return_Pct'].mean()
avg_loser = df[df['Win'] == False]['Return_Pct'].mean()
actual_win_rate = df['Win'].mean()

print(f"Actual win rate: {actual_win_rate*100:.1f}%")
print(f"Avg winner: +{avg_winner:.2f}%")
print(f"Avg loser: {avg_loser:.2f}%")

# Expected return per trade with random guessing (50% win rate)
expected_random = 0.5 * avg_winner + 0.5 * avg_loser
print(f"\nExpected return per trade (random 50%): {expected_random:.2f}%")

# Expected return with actual win rate
expected_actual = actual_win_rate * avg_winner + (1-actual_win_rate) * avg_loser
print(f"Expected return per trade (actual {actual_win_rate*100:.1f}%): {expected_actual:.2f}%")

# The "edge" is tiny
edge = expected_actual - expected_random
print(f"\nEdge per trade: {edge:.3f}%")
print(f"Edge over 7,569 trades: {edge * 7569:.1f}% (before compounding)")

# Calculate realistic return with this edge
cycles = 7569 / 5  # ~5 trades per exit day
cycle_return = 5 * POSITION_SIZE * expected_actual / 100
six_year_compound = 100 * (1 + cycle_return) ** cycles
print(f"\n6-year compounded return (realistic): ${six_year_compound:.2f} ({six_year_compound-100:.1f}%)")

print("\n" + "="*70)
print("SUMMARY: Why the 321% is misleading")
print("="*70)
print(f"""
Original reported return:     321% (${420:.0f})
Method 2 (proper sizing):     {portfolio_history[-1]['value']-100:.0f}% 
Method 3 (cash drag):         {cash-100:.0f}%
Method 4 (edge analysis):     {six_year_compound-100:.0f}%

The key issues:
1. The original 321% comes from compounding 5 positions/day for 6 years
2. Average return per trade is only 0.50% with 51.5% win rate  
3. The EDGE over random is only ~{edge:.3f}% per trade
4. This tiny edge, compounded properly, gives ~{six_year_compound-100:.0f}%

Compare to SPY: ~132% over same period
Realistic alpha: {six_year_compound-100-132:.0f}% to {portfolio_history[-1]['value']-100-132:.0f}% (not 189%!)
""")
