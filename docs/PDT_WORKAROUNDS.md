# Pattern Day Trader (PDT) Rule Workarounds

## The Problem

The PDT rule (FINRA Rule 4210) restricts accounts under $25,000 from making more than 3 day trades in a 5-business-day period. This severely limits minute-level trading strategies.

**Impact on this strategy:**
- With $10k margin account: Only 195 trades executed, +0.89% return
- With $10k cash account: 2,200 trades, +6.85% return  
- With $25k+ margin account: 2,794 trades, +16.65% return

## Workarounds

### 1. Multiple Broker Accounts (Easiest)

Split your capital across multiple brokers to multiply your day trade allowance:

| Setup | Day Trades/Week | Capital Split |
|-------|-----------------|---------------|
| 1 broker | 3 | All in one |
| 2 brokers | 6 | 50/50 |
| 3 brokers | 9 | 33/33/33 |
| 4 brokers | 12 | 25/25/25/25 |

**Recommended combination:**
- Alpaca (API trading, IEX routing)
- Webull (good charts)
- Fidelity (reliable execution)
- Robinhood (easy mobile)

### 2. Offshore Brokers (No PDT)

These brokers operate outside US jurisdiction and don't enforce PDT:

| Broker | Min Deposit | Leverage | Based In |
|--------|-------------|----------|----------|
| CMEG | $500 | 6:1 | Trinidad |
| TradeZero International | $500 | 4:1 | Bahamas |
| Ustocktrade | $1,000 | 4:1 | Barbados |

**Pros:** No PDT, high leverage
**Cons:** Less regulatory protection, wire transfer fees, tax complexity

### 3. Prop Trading Firms (Trade Their Capital)

Pass an evaluation and trade firm capital. No PDT because it's not your money.

| Firm | Eval Cost | Buying Power | Profit Split |
|------|-----------|--------------|--------------|
| Topstep | $165/mo | $50k-$150k | 90% |
| Apex Trader Funding | $147/mo | $25k-$300k | 90% |
| The Trading Pit | $99 | $10k-$100k | 70-80% |
| Trade The Pool | $97 one-time | $20k-$260k | 70-80% |

### 4. Futures Trading (Best Alternative)

**PDT does NOT apply to futures.** Trade the same underlying (S&P 500) via futures:

| Contract | Symbol | Notional Value | Day Margin | Point Value |
|----------|--------|----------------|------------|-------------|
| E-mini S&P | /ES | ~$250k | ~$500 | $50/point |
| **Micro E-mini S&P** | **/MES** | ~$25k | ~$50 | $5/point |
| Micro Nasdaq | /MNQ | ~$35k | ~$100 | $2/point |

**With $5,000 you can trade:**
- 10 /MES contracts (equivalent to ~$250k SPY exposure)
- Day trade unlimited times
- 23-hour market (6pm Sun - 5pm Fri)

**Brokers for futures:**
- Tradovate (low fees, good platform)
- NinjaTrader (free platform, good for automation)
- AMP Futures (lowest margins)
- Interactive Brokers (if you already have account)

### 5. Swing Trading Modification

Modify the strategy to avoid same-day round trips:

**Current (Day Trade):**
```
BUY  9:35 AM  → SELL 10:15 AM  (same day = day trade)
```

**Modified (Swing Trade):**
```
BUY  3:55 PM  → SELL 9:35 AM next day  (different days = NOT a day trade)
```

This eliminates PDT entirely but changes the strategy dynamics.

### 6. Cash Account (Already Tested)

Use a cash account instead of margin:
- No PDT restrictions
- But T+1 settlement limits trading frequency
- Tested result: ~7% return (vs 17% with no restrictions)

## Comparison

| Method | PDT Free? | Min Capital | Complexity | Expected Return |
|--------|-----------|-------------|------------|-----------------|
| Save to $25k | Yes | $25,000 | Low | 17% |
| Multiple brokers | Partial | $2,500/ea | Medium | 10-12% |
| Offshore broker | Yes | $500 | High | 17% |
| Prop firm | Yes | $100-200/mo | Medium | 70-90% of 17% |
| **Futures (/MES)** | **Yes** | **$1,000** | **Medium** | **17%+** |
| Cash account | Yes | Any | Low | 7% |
| Swing trading | Yes | Any | Low | Varies |

## Recommendation

**For accounts under $25,000:**

1. **Best option:** Trade **/MES futures** instead of SPY
   - No PDT restrictions
   - Can start with $1,000
   - Same underlying (S&P 500)
   - Extended hours (nearly 24/5)
   - Adapt the model to futures data

2. **Second best:** Use a **prop trading firm**
   - Pass evaluation (~$100-200)
   - Trade $50k-$100k of their capital
   - Keep 70-90% of profits
   - No PDT since it's not your account

3. **Simplest:** Use **multiple broker accounts**
   - Split capital across 3-4 brokers
   - 9-12 day trades per week
   - Same strategy, just distributed

## Adapting This Strategy to Futures

The minute-level model can be adapted to /MES futures:

```python
# Instead of:
symbol = 'SPY'
api.submit_order(symbol='SPY', qty=100, ...)

# Use:
symbol = '/MES'  # or 'MES' depending on broker
api.submit_order(symbol='MES', qty=1, ...)  # 1 /MES ≈ 100 shares SPY exposure
```

**Key differences:**
- 1 /MES contract = $5 per 0.25 point move = ~100 SPY shares equivalent
- Trading hours: 6pm ET Sunday - 5pm ET Friday (nearly 24/5)
- Tick size: 0.25 points
- Margin: ~$50-100 per contract (day trading)

The prediction model would need minor adjustments for the different price scale.
