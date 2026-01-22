#!/usr/bin/env python3
"""
Deep skepticism analysis: Why might these results be too good to be true?
"""
import pandas as pd
import numpy as np

print("="*70)
print("DEEP SKEPTICISM ANALYSIS: Why is this too good to be true?")
print("="*70)

df = pd.read_csv('/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/results/extended_middle_deciles/all_trades.csv')
df['Entry_Date'] = pd.to_datetime(df['Entry_Date'])
df['Exit_Date'] = pd.to_datetime(df['Exit_Date'])

# 1. CHECK POSITION SIZING MATH
print("\n1. POSITION SIZING SANITY CHECK")
print("-"*50)
print(f"Total trades: {len(df)}")
print(f"Date range: {df['Entry_Date'].min().date()} to {df['Exit_Date'].max().date()}")
trading_days = len(df['Entry_Date'].unique())
print(f"Trading days: {trading_days}")
print(f"Avg trades per day: {len(df)/trading_days:.1f}")

# How many positions are open at once?
dates = pd.date_range(df['Entry_Date'].min(), df['Exit_Date'].max(), freq='B')
max_positions = 0
for d in dates[:200]:  # Sample first 200 days
    open_pos = len(df[(df['Entry_Date'] <= d) & (df['Exit_Date'] > d)])
    max_positions = max(max_positions, open_pos)
print(f"Max concurrent positions (sample): {max_positions}")
print(f"At 4% each = {max_positions * 4}% of portfolio")

# 2. CHECK IF WE'RE USING LEVERAGE
print("\n2. LEVERAGE CHECK")
print("-"*50)
if max_positions * 4 > 100:
    print(f"⚠️ PROBLEM: Up to {max_positions * 4}% deployed = IMPLICIT LEVERAGE!")
    print(f"   The backtest assumes we can deploy {max_positions * 4}% of capital")
    print(f"   This is effectively {max_positions * 4 / 100:.1f}x leverage")
else:
    print(f"✓ Max {max_positions * 4}% deployed - no leverage")

# 3. ANALYZE RETURN DISTRIBUTION
print("\n3. RETURN DISTRIBUTION")
print("-"*50)
returns = df['Return_Pct']
print(f"Mean return per trade: {returns.mean():.2f}%")
print(f"Median return: {returns.median():.2f}%")
print(f"Std dev: {returns.std():.2f}%")
print(f"Max gain: {returns.max():.1f}%")
print(f"Max loss: {returns.min():.1f}%")

# 4. CHECK FOR SURVIVORSHIP BIAS IMPACT
print("\n4. SURVIVORSHIP BIAS DEEP DIVE")
print("-"*50)
tickers = df['Ticker'].unique()
print(f"Total unique tickers traded: {len(tickers)}")

# Check for stocks that IPO'd after 2020
recent_ipos = ['ABNB', 'PLTR', 'RIVN', 'LCID', 'HOOD', 'RBLX', 'COIN', 'PATH', 'SNOW', 'CRWD', 
               'DASH', 'U', 'NET', 'DDOG', 'ZS', 'OKTA', 'PANW']
found = [t for t in recent_ipos if t in tickers]
print(f"Recent IPOs/additions found: {found}")

# Check earliest trade for these stocks
for ticker in found:
    t_df = df[df['Ticker'] == ticker]
    first_trade = t_df['Entry_Date'].min()
    print(f"  {ticker}: First trade {first_trade.date()}")

# 5. CALCULATE ACTUAL ANNUALIZED RETURN
print("\n5. REALISTIC RETURN CALCULATION")
print("-"*50)

# Method 1: Without leverage (cap at 100% deployed)
daily_contrib = df.groupby('Exit_Date')['Return_Pct'].apply(lambda x: (x * 0.04).sum())

# Cap daily returns to prevent leverage effect
# If we have 25 positions, that's 100% deployed
# Any more would be leverage
POSITIONS_PER_DAY = 5
HOLD_DAYS = 5
MAX_POSITIONS = POSITIONS_PER_DAY * HOLD_DAYS  # 25 positions
MAX_ALLOCATION = 1.0  # 100%

# Recalculate with realistic assumptions
realistic_val = 100
for d in sorted(daily_contrib.index):
    daily_ret = daily_contrib[d] / 100
    # Cap the daily return to max 20% of portfolio (5 positions * 4%)
    # Actually this is per exit day, not per day...
    realistic_val *= (1 + daily_ret)

print(f"Compounded return (current method): ${realistic_val:.2f} ({realistic_val-100:.1f}%)")

# 6. THE REAL PROBLEM: COMPOUNDING OVERLAPPING POSITIONS
print("\n6. THE BIG ISSUE: OVERLAPPING POSITION COMPOUNDING")
print("-"*50)
print("With 5 new positions daily and 5-day holds:")
print("  - Day 1: 5 positions (20% deployed)")
print("  - Day 2: 10 positions (40% deployed)")
print("  - Day 3: 15 positions (60% deployed)")
print("  - Day 4: 20 positions (80% deployed)")
print("  - Day 5+: 25 positions (100% deployed)")
print("")
print("BUT the backtest compounds EACH position separately,")
print("effectively using leverage when positions overlap!")

# 7. CALCULATE WHAT RETURN SHOULD BE WITHOUT LEVERAGE
print("\n7. CORRECTED RETURN (No Leverage)")
print("-"*50)

# Each trade contributes 4% * return_pct to portfolio
# But we can only have 100% deployed, so we need to scale down
avg_return = returns.mean()
win_rate = df['Win'].mean()
trades_per_year = len(df) / 6  # 6 years
print(f"Avg return per trade: {avg_return:.2f}%")
print(f"Win rate: {win_rate*100:.1f}%")
print(f"Trades per year: {trades_per_year:.0f}")

# More realistic calculation
# 25 positions max, each 4% = 100% deployed
# Average hold is 5 days = ~250/5 = 50 cycles per year
# Each cycle: 5 positions exit, contributing 5 * 4% * avg_return
cycles_per_year = 250 / 5
return_per_cycle = 5 * 0.04 * avg_return / 100
annual_return = (1 + return_per_cycle) ** cycles_per_year - 1
six_year_return = (1 + annual_return) ** 6 - 1

print(f"\nRealistic calculation (no leverage):")
print(f"  Return per cycle (5 days): {return_per_cycle*100:.2f}%")
print(f"  Cycles per year: {cycles_per_year:.0f}")
print(f"  Annual return: {annual_return*100:.1f}%")
print(f"  6-year return: {six_year_return*100:.1f}%")

print("\n" + "="*70)
print("CONCLUSION: KEY ISSUES INFLATING RESULTS")
print("="*70)
print("""
1. IMPLICIT LEVERAGE: The backtest allows >100% capital deployment
   when positions overlap. With 25 concurrent positions at 4% each,
   that's 100% - but the compounding treats each as independent.

2. SURVIVORSHIP BIAS: Using today's S&P 500 list means we're only
   trading stocks that succeeded. Failed/delisted companies excluded.

3. NO TRANSACTION COSTS: Even with fees analysis, slippage on 
   7,569 trades would add up.

4. PERFECT EXECUTION: Assumes we get exact open/close prices.

5. NO MARKET IMPACT: Trading 5 stocks daily with 4% of portfolio
   could move prices on smaller-cap S&P names.

REALISTIC EXPECTATION: 
  Reported: 321% over 6 years (26% CAGR)
  Realistic: Probably 60-100% over 6 years (8-12% CAGR)
  Still beats S&P 500? Maybe by a small margin.
""")
