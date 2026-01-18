# Dual-Model Trading System - Flow Diagram

## Data Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     Historical Stock Data                    │
│          (SPY 2022-2024 Training + 2025 Testing)            │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
   ┌───────────────┐            ┌──────────────────┐
   │ Training Data │            │  Testing Data    │
   │  (2022-2024)  │            │     (2025)       │
   └───────────────┘            └──────────────────┘
         │                               │
         │                               │
         ├───► Calculate Indicators      ├───► Calculate Indicators
         │     (SMA, RSI, MACD, BB, etc) │     (with 200-row warm-up)
         │                               │
         ▼                               ▼
   ┌───────────────────────────┐  ┌──────────────────────────┐
   │  Training Data + Indicators│  │ Testing Data + Indicators │
   └───────────────┬───────────┘  └──────────┬───────────────┘
                   │                         │
         ┌─────────┴──────────┐              │
         │                    │              │
         ▼                    ▼              ▼
      ┌─────────────┐   ┌──────────┐   ┌──────────────┐
      │ Entry Model │   │ Exit     │   │ Make          │
      │(LightGBM)   │   │ Model    │   │ Predictions   │
      │             │   │(LightGBM)│   │              │
      │ Predicts:   │   │          │   │ Returns:     │
      │ • Price     │   │ Predicts:│   │ • Entry      │
      │   movements │   │ • Drop   │   │   Signals    │
      │ • +/- price │   │   prob   │   │ • Confidence │
      │ • Confidence│   │ (>1% in  │   │   scores     │
      │   score     │   │  1-2 days)   │ • Exit model │
      └─────────────┘   └──────────┘   │   drop probs │
                                        └──────────────┘
```

## Trading Signal Generation

```
┌──────────────────────────────────────┐
│  Entry Model Predictions + Confidence │
└──────────────┬───────────────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │ Filter by Percentiles:   │
    │ • Top 25% = BUY signal   │
    │ • Bottom 75% = SELL      │
    │   (IGNORED in backtest)  │
    │ • Medium = HOLD          │
    └──────────────┬───────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │ Filter by Confidence:        │
    │ • Only keep signals          │
    │   where confidence > 60%     │
    │ • Remove low-quality signals │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │ Add Mean Reversion Overlay:  │
    │ • Buy if -5% below 20-day MA │
    │ • Sell if +5% above 20-day MA│
    │   (IGNORED in backtest)      │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Trading Signals DataFrame    │
    │  - Date                      │
    │  - Signal (BUY/HOLD/SELL)   │
    │  - Confidence score          │
    │  - Price                     │
    └──────────────────────────────┘
```

## Portfolio Backtesting Engine

```
┌──────────────────────────────────────┐
│  Entry Signals + Exit Model Probs     │
└──────────────┬───────────────────────┘
               │
     ┌─────────┴──────────┐
     │                    │
     ▼                    ▼
┌──────────────┐  ┌──────────────────────┐
│ Entry Signals│  │ Exit Model (Drop%)   │
│              │  │                      │
│ - Date       │  │ - Date               │
│ - Signal     │  │ - Drop_Probability   │
│ - Confidence │  │   (0.0 to 1.0)       │
│ - Price      │  │                      │
└──────────────┘  └──────────────────────┘
     │                    │
     └─────────┬──────────┘
               │
               ▼
    ┌────────────────────────────────┐
    │     Daily Portfolio Update      │
    │                                │
    │ For each day:                  │
    │                                │
    │ 1. Check EXIT MODEL:           │
    │    IF Drop_Prob > 0.7:         │
    │       → EXIT all positions     │
    │       → Lock in profits        │
    │                                │
    │ 2. Check ENTRY SIGNAL:         │
    │    IF BUY signal:              │
    │       → Size by confidence:    │
    │       - >0.8: 75% of cash     │
    │       - 0.65-0.8: 50%         │
    │       - 0.5-0.65: 30%         │
    │       - <0.5: 10%             │
    │       → Buy shares            │
    │                                │
    │ 3. Ignore SELL signals         │
    │    (exit model handles it)    │
    │                                │
    │ 4. Earn SHV on dead cash:      │
    │    cash += cash * 0.015%       │
    │                                │
    │ 5. Calculate daily return      │
    └────────────────────────────────┘
               │
               ▼
    ┌────────────────────────────────┐
    │   Backtest Results DataFrame    │
    │                                │
    │ For each day:                  │
    │ - Date                         │
    │ - Signal                       │
    │ - Cash                         │
    │ - Shares_Held                  │
    │ - Portfolio_Value              │
    │ - Daily_Return                 │
    │ - Cumulative_Return            │
    │ - Trade_Size_%                 │
    │ - SHV_Earnings                 │
    │ - Drop_Probability             │
    └────────────────────────────────┘
```

## Performance Analysis

```
┌─────────────────────────────────┐
│   Backtest Results               │
│   - Portfolio values             │
│   - Trade history                │
│   - Daily returns                │
└──────────────┬──────────────────┘
               │
     ┌─────────┴──────────┐
     │                    │
     ▼                    ▼
┌──────────────┐  ┌──────────────────┐
│ Strategy     │  │ Buy & Hold (S&P) │
│ Performance  │  │ Baseline         │
│              │  │                  │
│ • Final value│  │ • Final value    │
│ • Return %   │  │ • Return %       │
│ • Win rate   │  │ • Comparison     │
│ • Drawdown   │  │                  │
│ • Sharpe     │  │                  │
└──────────────┘  └──────────────────┘
     │                    │
     └─────────┬──────────┘
               │
               ▼
    ┌────────────────────────────────┐
    │   Performance Metrics           │
    │                                │
    │ • Outperformance vs benchmark  │
    │ • Risk-adjusted returns        │
    │ • Win/loss ratios              │
    │ • Maximum drawdown             │
    │ • Volatility                   │
    │ • Sharpe ratio                 │
    └────────────────────────────────┘
```

## Key Decision Points

### 1. Exit Model Check (HIGHEST PRIORITY)
```
IF Drop_Probability > 0.7:
    EXIT ALL POSITIONS
    (Price likely to drop >1% in next 1-2 days)
```
**Timing**: Exits BEFORE price drops (proactive)

### 2. Entry Signal Check
```
IF Signal == 'BUY' AND Confidence > 0.6:
    Position_Size = Confidence_Based
    BUY shares = (Cash * Position_Size) / Price
```
**Sizing**: More confident = larger position

### 3. Ignore SELL Signals
```
IF Signal == 'SELL':
    IGNORE (don't exit)
    Only exit when Exit Model predicts drop
```
**Reason**: Exit model has better timing

### 4. Cash Management
```
Unused_Cash earns 0.015% daily (5.5% annualized)
Equivalent to holding SHV (short-duration Treasury ETF)
```
**Benefit**: Dead cash isn't wasted

## Model Strengths

| Model | Input | Output | Strength |
|-------|-------|--------|----------|
| **Entry Model** | OHLCV + Indicators | Price direction + Confidence | Identifies trading opportunities |
| **Exit Model** | OHLCV + Indicators | Drop probability | Predicts short-term reversals |
| **Combined** | Both models | Entry + Exit timing | Complete trade lifecycle |

## Risk Management

✓ **Position Sizing**: Based on prediction confidence
✓ **Exit Timing**: Separate model avoids false exits
✓ **Cash Reserve**: Always maintains some cash
✓ **SHV Earnings**: Passive income on dead cash
✓ **Maximum Drawdown**: Tracked and reported
✓ **Win Rate**: Monitored for strategy health

## Expected Outcome

Strategy outperforms buy-and-hold when:
1. Entry model accurately identifies price movements
2. Exit model correctly predicts short-term drops
3. Confidence-based sizing puts more money on best signals
4. Exit timing avoids losses better than SELL signals alone

**Target**: 15-25% annual return with lower drawdown than S&P 500
