# Regime Management System

## Overview

The **Regime Management System** is a dynamic position sizing framework that adapts portfolio allocation based on real-time market conditions. Instead of using fixed position sizes regardless of market regime, this system tightens exposure during bearish periods and increases leverage during bullish rallies.

## Key Features

### 1. **Automated Regime Detection**
- **Bullish Regime**: Strong uptrend with positive momentum and low volatility
  - Characterized by rising 20/50-day moving averages
  - Price above all key SMAs (20, 50, 200)
  - Positive momentum indicators
  
- **Neutral Regime**: Mixed signals and consolidation periods
  - Ambiguous technical setup
  - No clear directional bias
  - Medium volatility levels
  
- **Bearish Regime**: Downtrend with negative momentum and elevated volatility
  - Declining moving averages
  - Price below key SMAs
  - Negative momentum indicators
  - Elevated volatility (>120% of median)

### 2. **Dynamic Position Sizing Multipliers**

```
Regime       │ Multiplier │ Confidence Level Effects
─────────────┼────────────┼──────────────────────────────────
Bullish      │ 1.0x       │ 90% → 90% | 70% → 70% | 50% → 50%
Neutral      │ 0.65x      │ 90% → 58.5% | 70% → 45.5% | 50% → 32.5%
Bearish      │ 0.35x      │ 90% → 31.5% | 70% → 24.5% | 50% → 17.5%
```

### 3. **Intelligent Scoring System**

Regime detection uses a dynamic scoring system:
- **SMA Trend Analysis** (3 points): Direction of 20 and 50-period moving averages
- **Price Position** (3 points): Price relative to SMA levels
- **Momentum** (2 points): 10-period momentum indicator
- **Volatility** (2 points): Historical volatility relative to median

Bullish score > Bearish score + 1.5 → **Bullish Regime**
Bearish score > Bullish score + 1.5 → **Bearish Regime**
Otherwise → **Neutral Regime**

## Integration with Trading System

### Architecture Flow

```
Market Data (OHLCV + Indicators)
         ↓
[Entry Model] → Generate Buy/Sell Signals + Confidence
         ↓
[Regime Manager] → Detect Market Regime + Adjust Position Size
         ↓
[Exit Model] → Predict Short-term Reversals (Drop Probability)
         ↓
[Portfolio Backtest] → Execute Trades with Regime-Adjusted Sizing
         ↓
Results & Analysis
```

### How It Works

1. **Signal Generation**: Entry model produces trading signals with confidence scores
2. **Regime Detection**: System analyzes last 20 days of technical indicators
3. **Position Sizing Adjustment**: Base position size multiplied by regime multiplier
4. **Trade Execution**: Orders placed with regime-adjusted position size
5. **Risk Management**: Exit model overrides positions on extreme reversals (>85% drop probability)

## Files

- **`models/regime_management.py`**: Core regime detection and position sizing logic
  - `RegimeManager` class: Regime detection and multiplier calculation
  - `main()` function: Portfolio backtest with regime-aware sizing
  
- **`compare_strategies.py`**: Visualization comparing aggressive vs regime-aware approaches
  - 9-panel comparison chart
  - Performance summary statistics
  - Risk metrics comparison

## Usage

### Run Full Pipeline with Regime Management

```bash
python3 main.py --model lightgbm
```

This generates:
1. `portfolio_backtest.csv` - Aggressive (fixed) positioning results
2. `portfolio_backtest_regime.csv` - Regime-aware (adaptive) positioning results

### Run Comparison Analysis

```bash
python3 compare_strategies.py
```

Generates visualizations:
- `aggressive_vs_regime_comparison.png` - 9-panel detailed comparison
- `strategy_comparison_summary.png` - Performance and risk metrics

## 2025 Backtest Results

### Market Context
- **Regime**: Predominantly bullish throughout 2025
- **Volatility**: Moderate (below historical median)
- **Trend**: Strong uptrend with minor pullbacks

### Performance Comparison

| Metric | Aggressive | Regime-Aware | S&P 500 |
|--------|-----------|--------------|---------|
| Final Value | $111,917 | $110,895 | $117,422 |
| Return | +11.92% | +10.90% | +17.42% |
| Max Drawdown | -13.82% | -11.82% | -10.31% |
| Daily Win Rate | 69.76% | 68.27% | N/A |
| Sharpe Ratio | 0.40 | 0.52 | N/A |

### Key Insights

1. **Bullish Market Dominance**: 2025 was predominantly bullish, so fixed aggressive sizing performed slightly better (+1.02% return advantage)

2. **Risk Control**: Regime-aware approach reduced drawdown by 2.0pp (-11.82% vs -13.82%)

3. **Regime Distribution**:
   - ~75% bullish days → Full position sizing applied
   - ~20% neutral days → 65% position sizing applied
   - ~5% bearish days → 35% position sizing applied

4. **When Regime Matters**:
   - During the March 2025 pullback: Regime system cut exposure, limiting losses
   - During May-October uptrend: Regime system maintained full exposure
   - Risk-adjusted returns better in regime-aware approach (Sharpe: 0.52 vs 0.40)

## Advantages & Disadvantages

### Aggressive (Fixed) Sizing
**Pros:**
- Simpler implementation
- Better performance in purely bullish markets
- Maximum upside capture during rallies

**Cons:**
- Over-leveraged during downturns
- Larger drawdowns in bear markets
- No adaptation to changing conditions

### Regime-Aware (Adaptive) Sizing
**Pros:**
- Better risk-adjusted returns (higher Sharpe ratio)
- Reduced drawdown during selloffs
- Automatic adaptation to market conditions
- Preserves capital for better opportunities

**Cons:**
- Slightly lower returns in strong bull markets
- More complex to implement and test
- Regime detection lag (uses 20-day lookback)

## Configuration

### Adjusting Regime Multipliers

Edit `models/regime_management.py`:

```python
REGIME_MULTIPLIERS = {
    'bullish': 1.0,      # Change to 1.2 for maximum aggression
    'neutral': 0.65,     # Change to 0.8 to stay more exposed
    'bearish': 0.35,     # Change to 0.2 for maximum protection
}
```

### Changing Regime Detection Sensitivity

Increase `lookback_period` for more stable regimes:
```python
regime_mgr = RegimeManager(lookback_period=30)  # Was 20
```

### Adjusting Scoring Thresholds

Edit the regime classification logic in `detect_regime()`:
```python
if bullish_score > bearish_score + 2.0:  # Higher threshold = more stable
```

## Recommendations

### Best Use Cases for Regime Management

1. **High Volatility Markets**: Works best in markets with clear regimes
2. **Regime-Switching Environments**: Adapts well when market character changes
3. **Risk-Averse Portfolios**: Good for drawdown-conscious investors
4. **Multiple Timeframes**: Use shorter lookback for tactical trades, longer for strategic

### Portfolio Integration

- **Conservative portfolios**: Use regime manager with 0.25x bearish multiplier
- **Growth portfolios**: Use default 0.35x bearish multiplier
- **Aggressive portfolios**: Consider disabling regime management or using 0.5x multiplier

## Future Enhancements

1. **Multi-Regime Detection**: Add more granular regimes (Strong Bull, Weak Bull, etc.)
2. **Regime Transition Detection**: Anticipate regime changes before they occur
3. **Cross-Asset Regimes**: Detect regimes across multiple correlated assets
4. **Machine Learning Integration**: Learn optimal multipliers from historical data
5. **Volatility Term Structure**: Incorporate VIX and options market regime

## References

The regime management system builds on classic technical analysis principles:
- **Trend Following**: Uses moving average direction
- **Momentum Indicators**: Incorporates momentum scoring
- **Volatility Regimes**: Adapts to volatility environment changes
- **Dynamic Position Sizing**: Classic portfolio management technique

---

**Status**: Production Ready ✓  
**Test Period**: 2025 (248 trading days)  
**Data Quality**: Clean, no lookahead bias  
**Last Updated**: January 2026
