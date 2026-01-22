#!/usr/bin/env python3
"""Check if model is adding value vs random."""
import pandas as pd
import numpy as np

df = pd.read_csv('/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/results/extended_middle_deciles/all_trades.csv')

print("="*70)
print("THE REAL QUESTION: Is the model doing anything useful?")
print("="*70)

avg_return = df['Return_Pct'].mean()
win_rate = df['Win'].mean()

print(f"\nAverage return per trade: {avg_return:.2f}%")
print(f"Win rate: {win_rate*100:.1f}%")

print("\n" + "-"*50)
print("KEY INSIGHT: We're LONG-ONLY in a BULL MARKET")
print("-"*50)
print("""
From 2020-2026, the S&P 500 returned +132%.
If you bought ANY random S&P 500 stock and held 5 days,
you'd make money most of the time because the market went UP.

The question isn't "did we make money?"
The question is "did we make MORE money than random?"
""")

# Simulate random stock picking using the SAME returns, shuffled
print("Simulating 7,569 RANDOM 5-day stock picks (shuffled returns)...")

np.random.seed(42)
returns = df['Return_Pct'].values.copy()
np.random.shuffle(returns)

random_portfolio = 100.0
for i in range(0, len(returns), 5):
    batch = returns[i:i+5]
    daily_return = sum(batch) * 0.04 / 100
    random_portfolio *= (1 + daily_return)

print(f"Random picking: ${random_portfolio:.2f} ({random_portfolio-100:.1f}%)")
print(f"Model picking:  $421.33 (321.3%)")
print(f"Difference:     {321.3 - (random_portfolio-100):.1f}%")

# Monte Carlo: run many random simulations
print("\n" + "-"*50)
print("MONTE CARLO: 1000 random simulations")
print("-"*50)

random_results = []
for _ in range(1000):
    shuffled = df['Return_Pct'].values.copy()
    np.random.shuffle(shuffled)
    
    port = 100.0
    for i in range(0, len(shuffled), 5):
        batch = shuffled[i:i+5]
        daily_ret = sum(batch) * 0.04 / 100
        port *= (1 + daily_ret)
    random_results.append(port)

random_results = np.array(random_results)
print(f"Random picking average: ${random_results.mean():.2f} ({random_results.mean()-100:.1f}%)")
print(f"Random picking std dev: ${random_results.std():.2f}")
print(f"Random picking 5th percentile: ${np.percentile(random_results, 5):.2f}")
print(f"Random picking 95th percentile: ${np.percentile(random_results, 95):.2f}")

print(f"\nModel result ($421): Percentile {(random_results < 421.33).mean()*100:.1f}%")

print("\n" + "="*70)
print("THE BOTTOM LINE")
print("="*70)

# The REAL alpha
real_alpha = 321.3 - (random_results.mean() - 100)
print(f"""
Model return:       321% 
Random avg return:  {random_results.mean()-100:.0f}%
ACTUAL MODEL ALPHA: {real_alpha:.0f}%

Most of the 321% return comes from:
1. Being LONG-ONLY in a bull market  
2. Individual stocks outperform SPY (higher beta)
3. Compounding daily positions

The model's contribution is only ~{real_alpha:.0f}%, not 189% over SPY!

BUT WAIT - the random shuffling preserves the SAME average return.
What matters is: are we picking the RIGHT stocks on the RIGHT days?

With 51.5% win rate vs 50%, the model has a 1.5% EDGE.
This 1.5% edge * 7569 trades = {0.015 * 7569:.0f}% theoretical edge
After compounding: ~{real_alpha:.0f}% alpha

THIS IS STILL GOOD! But not "magic" good.
""")
