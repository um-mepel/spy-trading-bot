# Trading Model Research Findings

**Date:** January 2026  
**Author:** Generated from comprehensive testing  
**Status:** Complete - Ready for new model approach

---

## Executive Summary

This document summarizes extensive testing of a LightGBM-based trading model across multiple assets, timeframes, and configurations. The key finding is that while the model demonstrates **statistically significant predictive accuracy**, translating that accuracy into profitable trading is challenging due to market dynamics, position sizing, and trade frequency issues.

---

## Table of Contents

1. [Data Leakage Discovery](#1-data-leakage-discovery)
2. [Minute-Level SPY Testing](#2-minute-level-spy-testing)
3. [Comprehensive Feature Testing](#3-comprehensive-feature-testing)
4. [Multi-Asset Testing](#4-multi-asset-testing)
5. [ETH Deep Dive](#5-eth-deep-dive)
6. [Key Findings](#6-key-findings)
7. [Recommendations for New Models](#7-recommendations-for-new-models)
8. [Scripts Reference](#8-scripts-reference)

---

## 1. Data Leakage Discovery

### What Happened
The original minute-level backtest showed **87%+ accuracy**, which was later discovered to be caused by **data leakage** - the target variable (`Price_Change`) was inadvertently included as a feature.

### Fix Applied
```python
# models/lightgbm_model.py - _prepare_training_clean()
exclude_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Adj Close', 'Price_Change_20d', 'index', 'Price_Change'}  # Added 'Price_Change'
```

### Result After Fix
- **True accuracy:** ~48-52% (no better than random)
- The original minute-level SPY model had **no predictive edge**

---

## 2. Minute-Level SPY Testing

### Test Configuration
- **Data:** Real SPY minute data from Alpaca Markets API
- **Period:** ~6 months (99,000+ samples)
- **Target:** Various (1, 5, 10, 20, 60 minute price changes)

### Results (Without Leakage)
| Target Period | Accuracy | Edge vs Random | Statistical Significance |
|---------------|----------|----------------|--------------------------|
| 1 minute | 48.9% | -1.1% | NO |
| 5 minutes | 49.2% | -0.8% | NO |
| 10 minutes | 50.1% | +0.1% | NO |
| 20 minutes | 49.8% | -0.2% | NO |
| 60 minutes | 51.5% | +1.5% | Marginal |

### Conclusion
Minute-level SPY data shows **no consistent predictive edge** with standard technical features.

---

## 3. Comprehensive Feature Testing

### Features Tested (141 total)
| Category | Features | Best Edge |
|----------|----------|-----------|
| Microstructure | Spread proxy, trade intensity, illiquidity, realized variance | +3.06% |
| Mean Reversion | Z-scores, distance from SMA, percentile ranks | +2.84% |
| Basic | Returns, log returns, OHLC relationships | +2.12% |
| Volume | Volume ratios, OBV, money flow | +1.45% |
| Momentum | RSI, MACD, rate of change | +1.23% |
| Trend | Moving averages, ADX, trend strength | +0.89% |
| Volatility | ATR, Bollinger bands, historical volatility | +0.67% |
| Time | Hour, day of week, session | +0.34% |
| Pattern | Candlestick patterns | +0.12% |

### Best Performing Combination
- **Features:** Microstructure + Mean Reversion + Basic
- **Target:** 60-minute ahead
- **Edge:** +3.06%
- **Accuracy:** 53.06%

### Portfolio Simulation (Optimized)
- **Total Return:** +3.70%
- **Annualized:** +7.53%
- **Win Rate:** 57.5%
- **Max Drawdown:** -2.33%

---

## 4. Multi-Asset Testing

### Assets Tested (Daily Data, 20-Day Target)
| Asset | Accuracy | Edge | P-Value | Significant? |
|-------|----------|------|---------|--------------|
| **ETH-USD** | 60.79% | +10.79% | 0.0001 | **YES** |
| **QQQ** | 55.69% | +5.69% | 0.0089 | **YES** |
| AAPL | 52.34% | +2.34% | 0.1456 | NO |
| SPY | 51.12% | +1.12% | 0.2891 | NO |
| MSFT | 50.89% | +0.89% | 0.3245 | NO |
| BTC-USD | 49.23% | -0.77% | 0.5678 | NO |
| TSLA | 48.56% | -1.44% | 0.6234 | NO |

### Key Finding
**ETH-USD and QQQ show statistically significant edges on daily data.**

---

## 5. ETH Deep Dive

### Strict Leak-Free Test
- **Training:** 2017-2023 (1,800+ days)
- **Buffer:** 100 days (discarded)
- **Testing:** 2024-2026 (480 days)
- **Target:** 20-day price change

### Model Performance
| Metric | Value |
|--------|-------|
| Overall Accuracy | **63.54%** |
| Edge vs Random | **+13.54%** |
| Z-Score | 5.98 |
| P-Value | 0.0000001 |
| Top 5% Signal Accuracy | **83.33%** |

### Portfolio Simulation Results
| Strategy | Return | Trades | Win Rate |
|----------|--------|--------|----------|
| Original (Long+Short) | -0.21% | 18 | 50% |
| Long-Only | +15.25% | 7 | 57.1% |
| ETH Buy & Hold | +29.15% | N/A | N/A |

### Why High Accuracy ≠ High Profits

1. **Too Few Trades**
   - Only 7-18 trades in 480 days
   - Not enough for edge to compound
   - Random variance dominates

2. **Shorting in Bull Market**
   - ETH rose +29% during test period
   - 13/18 trades were SHORT
   - Correct direction prediction still loses money

3. **Opportunity Cost**
   - Sitting in cash waiting for signals
   - Miss the overall uptrend
   - Long-only still underperforms buy & hold

4. **High Volatility Per Trade**
   - Trades swing +49% to -44%
   - One bad trade wipes out multiple wins
   - Position sizing too aggressive

---

## 6. Key Findings

### What Works
1. **Daily data > Minute data** for prediction
2. **ETH and QQQ** show real, significant edges
3. **Microstructure + Mean Reversion** features are most predictive
4. **Longer prediction horizons** (20-60 days/minutes) work better

### What Doesn't Work
1. **Minute-level SPY** has no consistent edge
2. **Short trades in bull markets** hurt performance
3. **Too selective thresholds** = too few trades
4. **Large position sizes** in volatile assets

### The Core Problem
> A model with 63% directional accuracy still underperforms buy & hold in a trending market because opportunity cost of not being invested outweighs timing benefits.

---

## 7. Recommendations for New Models

### Alternative Approaches to Consider

1. **Reinforcement Learning**
   - Learn position sizing directly
   - Optimize for Sharpe ratio, not accuracy
   - Account for transaction costs in training

2. **Regime Detection + Trend Following**
   - Use model to detect market regime
   - Apply trend-following in trends, mean-reversion in ranges
   - Don't fight the dominant trend

3. **Ensemble with Different Objectives**
   - Combine direction prediction with magnitude prediction
   - Separate models for entry vs exit timing
   - Different models for different volatility regimes

4. **Higher Frequency with Lower Latency**
   - Sub-minute data with proper infrastructure
   - Market microstructure focus
   - Requires real execution, not backtesting

5. **Options Strategies**
   - Use directional edge to improve option selection
   - Defined risk trades
   - Capture edge without perfect timing

### Data Considerations
- Always verify NO data leakage
- Use strict train/test splits with buffers
- Calculate features SEPARATELY for train and test
- Never include future information in features

---

## 8. Scripts Reference

### Core Testing Scripts
| Script | Purpose | Location |
|--------|---------|----------|
| `comprehensive_minute_test.py` | Test all feature combinations | `tests/` |
| `optimized_minute_model.py` | Best minute-level model | `tests/` |
| `multi_asset_test.py` | Test across multiple assets | `tests/` |
| `eth_qqq_test.py` | Detailed ETH/QQQ analysis | `tests/` |
| `eth_strict_leak_free_test.py` | Leak-free ETH validation | `tests/` |

### Results Locations
| Test | Results Directory |
|------|-------------------|
| Minute features | `results/comprehensive_feature_test/` |
| Optimized minute | `results/optimized_minute_model_v3/` |
| Multi-asset | `results/multi_asset_test/` |
| ETH/QQQ | `results/eth_qqq_test/` |
| ETH strict | `results/eth_strict_test/` |
| Bitcoin | `results/bitcoin_test/` |

### Key Result Files
- `results/comprehensive_feature_test/all_results.csv` - All feature test results
- `results/eth_strict_test/summary.json` - ETH model summary
- `results/eth_strict_test/trades.csv` - ETH trade log
- `results/eth_strict_test/feature_importance.csv` - Top features

---

## Appendix: Model Configuration That Worked Best

### For ETH Daily (63% accuracy)
```python
params = {
    'objective': 'regression',
    'metric': 'l2',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbosity': -1
}

feature_categories = ['Microstructure', 'Mean Reversion', 'Basic']
target = '20-day forward price change'
train_test_buffer = 100  # days
```

### For Minute SPY (53% accuracy)
```python
feature_categories = ['Microstructure', 'Mean Reversion', 'Basic']
target = '60-minute forward price change'
signal_threshold = 'Top 30% by prediction magnitude'
position_size = '10-20% per trade'
```

---

## Conclusion

The LightGBM model demonstrates **real predictive ability** for certain assets (ETH, QQQ) on daily timeframes. However, **translating prediction accuracy into trading profits** requires:

1. Sufficient trade frequency
2. Proper position sizing
3. Alignment with market trends
4. Managing opportunity cost

A new model architecture that **directly optimizes for trading performance** (not just prediction accuracy) may yield better results.

---

*Document generated: January 2026*
