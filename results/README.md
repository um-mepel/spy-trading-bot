# Trading Results Directory Structure

Organized results from model training, trading signal generation, portfolio backtesting, and performance visualization.

## Directory Organization

### `/model_predictions/`
**Raw model outputs from the Random Forest predictor**
- `random_forest_predictions.csv` — Daily predictions for all 249 trading days in 2025
  - Columns: Date, Predicted_Price, Actual_Price, Predicted_Change, Actual_Change, Price_Error, etc.
  - Shows the model's price change predictions and their accuracy

### `/trading_analysis/`
**Trading signals and portfolio backtest results**
- `trading_signals.csv` — Generated trading signals based on model predictions
  - Columns: Date, Predicted_Change_Pct, Signal (1=BUY, 0=HOLD, -1=SELL), Signal_Strength
  - 248 trading signals with win rates: BUY 44%, SELL 44%, HOLD 35%
  
- `portfolio_backtest.csv` — Full portfolio simulation results
  - Columns: Date, Signal, Shares_Held, Cash, Portfolio_Value, Daily_Return, Cumulative_Return
  - Tracks portfolio state each trading day with percentage-based position sizing

### `/visualizations/model_performance/`
**Model prediction quality and direction accuracy charts**
- `error_analysis.png` — Two subplots showing:
  - Absolute price error over time with mean reference line
  - Actual vs predicted price overlay
  
- `direction_accuracy.png` — 20-day rolling average of direction prediction accuracy
  - Shows how often the model correctly predicted up/down movements
  
- `error_distribution.png` — Histogram of prediction errors
  - Shows mean and median error with frequency distribution

### `/visualizations/portfolio_performance/`
**Trading strategy profitability and risk metrics**
- `portfolio_value.png` — Portfolio equity curve with buy/sell signal markers
  - Shows $100K → $103.5K (+3.47%) growth over 2025
  - Green triangles mark BUY signals, red triangles mark SELL signals
  
- `cumulative_return.png` — Percentage return growth with gain/loss shading
  - Green shading shows profitable periods, red shows losses
  
- `drawdown.png` — Peak-to-trough portfolio declines over time
  - Max drawdown: -10.43%, showing strategy resilience
  
- `cash_and_shares.png` — Two panels showing:
  - Available cash over time (shows when positions are taken)
  - Number of shares held in inventory

## Key Performance Metrics

| Metric | Value |
|--------|-------|
| Initial Capital | $100,000 |
| Final Portfolio Value | $103,473.81 |
| Total Return | +3.47% |
| Max Drawdown | -10.43% |
| Win Rate | 47.98% |
| Trading Days | 248 |
| BUY Signals | 9 (44% accuracy) |
| SELL Signals | 158 (44% accuracy) |
| HOLD Signals | 81 (35% accuracy) |

## Model Details

- **Algorithm:** Random Forest Regressor (100 trees, max_depth=20)
- **Training Data:** 2022-2024 (752 trading days, 552 after NaN removal)
- **Test Period:** 2025-01-02 through 2025-12-30 (248 trading days)
- **Features:** 28 technical indicators (no OHLCV leakage)
- **Prediction Target:** Daily price change (dollars)
- **Direction Accuracy:** 44.76% (vs 50% random)

## Signal Generation

- **Threshold:** 0.5% predicted price change
- **BUY:** Predicted change > 0.5%
- **SELL:** Predicted change < -0.5%
- **HOLD:** Predicted change between -0.5% and 0.5%

## Portfolio Strategy

- **Initial Capital:** $100,000
- **Position Sizing:** Percentage-based (20% allocation on BUY, 50% exit on SELL)
- **Risk Management:** No margin/short selling, capital-constrained
- **Transaction Costs:** Not modeled (in production, add slippage/commissions)

## Files Updated

Run `python3 main.py` from the project root to regenerate all results with the latest model.

All visualizations use 300 DPI for publication quality.
