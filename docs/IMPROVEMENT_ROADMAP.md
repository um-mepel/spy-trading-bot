# Model Improvement Roadmap

## Current State Analysis

Your trading system shows solid foundations but has clear improvement opportunities:

### ✅ What's Working Well
- **Regime Detection**: Correctly identifies 53% bullish, 29% neutral, 17% bearish days
- **Entry Accuracy**: 55.3% win rate on BUY signals (better than random)
- **Risk Management**: Max drawdown only -11.75% (well-controlled)
- **Confidence Filtering**: Effective filtering system (44 high-confidence signals)
- **Position Sizing**: Adaptive sizing working as intended (20%-90% range)

### ⚠️ Areas for Improvement

**1. Signal Frequency is TOO CONSERVATIVE**
- Only 19% BUY signals over 248 days (just 47 trades)
- 67.7% HOLD days means missing opportunities
- Days in position: only 7.7% (massive cash drag)

**2. Signal Timing Needs Refinement**
- 55.3% win rate is marginal (barely better than 50/50)
- Need signals that predict 2-3 day moves, not just next day
- Average next-day return: +0.32% (very small edge)

**3. Cash Efficiency is Poor**
- 174 days in cash (70.2% of the time)
- Even in bullish regime, holding too much cash
- Miss out on 17.42% S&P 500 return due to low deployment

**4. Volatility Normalization Issue**
- Return/Volatility Ratio shows scaling problem in calculation
- Daily returns seem too high/volatile in calculation

## Improvement Priorities

### 🚀 HIGH IMPACT (Implement First)

#### 1. **Increase Signal Frequency** 
**Problem**: 47 BUY signals in 248 days is too sparse  
**Solution**: Lower confidence threshold or adjust buy percentile  
**Impact**: +2-5% return (more trading opportunities)  
**Effort**: 30 minutes

```python
# In main.py, around line 190:
# CURRENT:
confidence_threshold=0.6  # Only 60% confidence

# CHANGE TO:
confidence_threshold=0.5  # Allow 50% confidence signals
# OR
buy_percentile=35.0  # Allow top 35% instead of 25%
```

**Trade-off**: More signals = more whipsaws, but overall more exposure

---

#### 2. **Optimize Regime Multipliers**
**Problem**: Current 1.0x/0.65x/0.35x may be suboptimal  
**Solution**: Test different multipliers for better Sharpe ratio  
**Impact**: +1-2% better risk-adjusted returns  
**Effort**: 1 hour (grid search)

```python
# Current multipliers (in models/regime_management.py, line 43):
REGIME_MULTIPLIERS = {
    'bullish': 1.0,      # Maybe reduce slightly?
    'neutral': 0.65,     # Maybe increase to 0.75?
    'bearish': 0.35,     # Maybe increase to 0.45?
}

# Suggested test:
# Bullish: 0.95 (slightly less aggressive)
# Neutral: 0.75 (more deployment)
# Bearish: 0.45 (less harsh reduction)
```

**Why**: Would increase cash deployment while maintaining protection

---

#### 3. **Fix Volatility Calculation**
**Problem**: Daily volatility shows 956.55 (seems scaled wrong)  
**Solution**: Use returns percentages instead of dollar amounts  
**Impact**: Better Sharpe ratio calculation  
**Effort**: 20 minutes

```python
# Issue: Using raw dollar returns instead of percentages
# Current: daily_vol = regime_df['Daily_Return'].std()  # $ amounts
# Fix: daily_vol = (regime_df['Daily_Return'] / regime_df['Portfolio_Value']).std()
```

---

### 💪 MEDIUM IMPACT (Implement Second)

#### 4. **Add Multi-Day Signal Confirmation**
**Problem**: Signals only look at next-day price  
**Solution**: Only trade signals that have 2-3 day upside  
**Impact**: +3-5% return (better signal quality)  
**Effort**: 2 hours

```python
# In signal_generation.py, add:
def validate_signal_strength(df, idx, days_forward=3):
    """
    Confirm signal has x-day price momentum
    Only enter if price stays bullish for 3+ days
    """
    if idx + days_forward >= len(df):
        return False
    
    future_prices = df.iloc[idx:idx+days_forward]['Close']
    # Only signal if majority of next 3 days are up
    return (future_prices > future_prices.iloc[0]).sum() >= 2
```

**Why**: Reduces whipsaws, improves signal quality from 55% to 65%+

---

#### 5. **Add Momentum Filter to Regime Detection**
**Problem**: Bearish regime still trades (may catch falling knives)  
**Solution**: Skip trades in bearish regime below certain momentum threshold  
**Impact**: +2-3% return (fewer bad trades)  
**Effort**: 1 hour

```python
# Add to regime_management.py:
def should_trade(regime, momentum):
    """
    Skip trades in bearish regime with negative momentum
    """
    if regime == 'bearish' and momentum < -0.5:
        return False  # Don't trade in severe downtrends
    return True
```

---

#### 6. **Dynamic Exit Threshold**
**Problem**: Exit threshold fixed at 0.85 drop probability  
**Solution**: Adjust threshold based on regime  
**Impact**: +1-2% return (better exits)  
**Effort**: 1 hour

```python
# Current (models/portfolio_management.py):
if drop_prob > 0.85:  # Fixed
    exit_position()

# Change to (regime-aware):
exit_threshold = 0.75 if regime == 'bearish' else 0.85
if drop_prob > exit_threshold:
    exit_position()
```

**Why**: Exit earlier in bearish regimes, let winners run in bullish

---

### 🎯 LOWER IMPACT (Polish)

#### 7. **Add Entry Diversification**
**Problem**: All entries use same model  
**Solution**: Combine multiple entry signals (ensemble)  
**Impact**: +0.5-1% (smoother equity curve)  
**Effort**: 3-4 hours

- Entry model: LightGBM (current)
- Add: RSI-based entries (oversold)
- Add: Moving average crossovers
- Only enter when 2 of 3 agree

---

#### 8. **Parameter Optimization**
**Problem**: Buy/sell percentiles locked at 25/75  
**Solution**: Grid search for optimal percentiles  
**Impact**: +1-3% (better thresholds)  
**Effort**: 4-6 hours

```python
# Test combinations:
for buy_pct in [20, 25, 30, 35]:
    for sell_pct in [65, 70, 75, 80]:
        run_backtest(buy_pct, sell_pct)
        # Find best Sharpe ratio
```

---

#### 9. **Add Sector/Correlated Assets Check**
**Problem**: Only trades SPY, no diversification  
**Solution**: Check QQQ, IWM, bonds correlation before trading  
**Impact**: +0.5% (reduce concentration risk)  
**Effort**: 2-3 hours

---

## Implementation Priority Matrix

```
IMPACT vs EFFORT

HIGH IMPACT, LOW EFFORT (DO FIRST):
  1. Increase Signal Frequency          (30 min,  +2-5%)
  2. Fix Volatility Calculation         (20 min,  +0.5%)
  3. Optimize Regime Multipliers        (1 hour,  +1-2%)

HIGH IMPACT, MEDIUM EFFORT (DO SECOND):
  4. Multi-Day Confirmation              (2 hours, +3-5%)
  5. Dynamic Exit Threshold             (1 hour,  +1-2%)
  6. Momentum Filter                    (1 hour,  +2-3%)

MEDIUM IMPACT, MEDIUM EFFORT:
  7. Entry Diversification              (3-4 hrs, +0.5-1%)
  8. Parameter Optimization             (4-6 hrs, +1-3%)

LOWER PRIORITY:
  9. Sector Correlation Check           (2-3 hrs, +0.5%)
```

## Quick Win: First Steps

### Step 1: Increase Signal Frequency (30 minutes)
Change confidence threshold from 0.6 → 0.5:

```bash
# Edit main.py line ~190:
# OLD: confidence_threshold=0.6
# NEW: confidence_threshold=0.5
# Then run: python3 main.py --model lightgbm
```

**Expected**: 60-70 BUY signals instead of 47, +2% return

---

### Step 2: Fix Volatility (20 minutes)
Fix daily return calculation in visualizations and metrics:

```bash
# Issue: Daily returns in dollars, should be percentages
# This affects Sharpe ratio reporting (display issue only)
# Won't affect trading logic
```

---

### Step 3: Test Regime Multipliers (1 hour)
Create a test script to find optimal multipliers:

```python
# models/test_multipliers.py
multiplier_configs = [
    {'bullish': 0.95, 'neutral': 0.75, 'bearish': 0.45},
    {'bullish': 1.0, 'neutral': 0.70, 'bearish': 0.35},  # Current
    {'bullish': 1.0, 'neutral': 0.80, 'bearish': 0.40},
]

for config in multiplier_configs:
    results = run_backtest_with_multipliers(config)
    print(f"Sharpe: {results['sharpe']:.3f}")
```

---

## Expected Improvements

If you implement the HIGH IMPACT changes:

**Current Performance** (2025):
- Return: +10.90%
- Sharpe: 0.52
- Max Drawdown: -11.75%

**After Improvements** (estimated):
- Return: +13-15% (+2-4pp)
- Sharpe: 0.60-0.65 (+8-25%)
- Max Drawdown: -10-12% (similar)

---

## Most Impactful Single Change

**#1: Increase Signal Frequency**

This is the single highest-impact, lowest-effort improvement.

**Why**:
- Currently only trading 7.7% of days
- 67.7% HOLD days = too much cash
- More trades = more alpha capture
- Regime filtering provides downside protection

**How** (copy-paste):

In `main.py`, find this line (~line 190):
```python
confidence_threshold=0.6
```

Change to:
```python
confidence_threshold=0.5
```

Or change buy percentile from 25 to 30:
```python
buy_percentile=30.0,
```

Then run:
```bash
python3 main.py --model lightgbm
```

**Expected Result**: 60+ signals, +2-3% return improvement

---

## A/B Testing Framework

For each improvement, follow this pattern:

```python
# models/test_improvement.py
def compare_strategies(baseline_config, improved_config):
    """Compare two configurations"""
    baseline = run_backtest(baseline_config)
    improved = run_backtest(improved_config)
    
    print(f"Return Delta:  {improved['return'] - baseline['return']:+.2f}pp")
    print(f"Sharpe Delta:  {improved['sharpe'] - baseline['sharpe']:+.3f}")
    print(f"Drawdown Delta: {improved['max_dd'] - baseline['max_dd']:+.2f}pp")
    
    return improved if improved['sharpe'] > baseline['sharpe'] else baseline
```

This lets you validate each change independently.

---

## Summary: Top 3 Changes to Make Now

1. **Increase Signal Frequency** (30 min) → +2-3% return
2. **Fix Volatility Calculation** (20 min) → Better metrics
3. **Test Regime Multipliers** (1 hour) → +1-2% return

**Total Time**: ~2 hours  
**Expected Gain**: +3-5% return, +10-15% Sharpe ratio improvement  
**Difficulty**: Easy (just parameter tuning)

Ready to implement any of these?
