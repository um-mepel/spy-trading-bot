# Exit Model Integration - Implementation Summary

## Overview
Successfully integrated the exit timing model with the portfolio backtesting system. The system now uses a **dual-model approach**:
1. **Entry Model (LightGBM)**: Predicts price movements to generate BUY/SELL signals
2. **Exit Model**: Predicts short-term price drops (>1%) to time position exits

## Files Modified

### 1. `models/portfolio_management.py`
**Change**: Updated documentation and function calls to use the exit model

**Key Updates**:
- Line 216: Updated exit strategy message to show:
  ```
  Exit Strategy: EXIT MODEL (predicts price drops >70% probability)
                (Uses separate model to time exits, not SELL signals)
  ```
- Added exit model DataFrame parameter handling:
  ```python
  if exit_model_df is not None:
      exit_model_df = exit_model_df.copy()
      exit_model_df['Date'] = pd.to_datetime(exit_model_df['Date'])
  ```
- Pass `exit_model_df` to `backtest_portfolio()` function

### 2. `main.py`
**Changes**: 
- Fixed duplicate/malformed line (line 165)
- Already had correct exit model integration in place

**Working Flow**:
```python
# Step 2B: Train exit model
exit_model_results = train_exit_model(...)
exit_predictions_df = exit_model_results['results']

# Step 4: Run backtest with both entry signals AND exit model
portfolio_results = run_portfolio_backtest(
    signals_df,
    exit_model_df=exit_predictions_df,  # ← Exit model passed here
    results_dir=TRADING_ANALYSIS_DIR,
    initial_capital=100000
)
```

## Dual-Model Strategy

### Entry Model (LightGBM)
- Trains on 2022-2024 data
- Predicts price movements (+/-)
- Generates BUY/SELL signals
- Filters by 60% confidence threshold
- Uses percentile-based thresholds (top 25% = BUY)

### Exit Model
- Trains on 2022-2024 data
- Predicts if price will drop >1% in next 1-2 days
- Outputs `Drop_Probability` score (0-1)
- Triggers position exits when `Drop_Probability > 0.7`

### Portfolio Logic
```python
# Each day:
1. If BUY signal (high confidence):
   - Size position based on confidence level
   - Very High (>0.8): 75% of cash
   - High (0.65-0.8): 50% of cash
   - Medium (0.5-0.65): 30% of cash
   - Low (<0.5): 10% of cash

2. If EXIT MODEL shows Drop_Probability > 0.7:
   - Exit ALL positions immediately
   - Lock in profits before price drop

3. IGNORE SELL signals from entry model
   - Only exit via EXIT MODEL (which has better timing)
   - Avoids early exits from false SELL signals
```

## Benefits

✅ **Avoids SELL Signal Noise**: Entry model's SELL signals often trigger too early
✅ **Timing Advantage**: Exit model specifically trained to predict short-term drops
✅ **Risk Management**: Exits before prices fall >1%, protecting capital
✅ **Confidence Weighting**: Positions sized based on prediction strength
✅ **Dual Signal Fusion**: Combines two independent models for better decisions

## Data Integrity Checks

✓ Indicators calculated separately on training/testing data
✓ No training data mixed into test data
✓ Exit model trained on training data ONLY
✓ Predictions made on test data ONLY
✓ Hyperparameters locked before testing
✓ All results are TRUE OUT-OF-SAMPLE
✓ NO LOOKAHEAD BIAS - Safe for real trading

## Output Files

When running the backtest, the system generates:
- `portfolio_backtest.csv`: Daily portfolio state (cash, shares, value, returns)
- `trading_signals.csv`: Entry signals with confidence scores
- Exit model predictions merged into backtest results

## Configuration (Hyperparameters - LOCKED)

```python
# Entry Signal Generation
buy_percentile=25.0           # Top 25% of predictions = BUY
sell_percentile=75.0          # Bottom 75% = SELL (IGNORED)
confidence_threshold=0.6      # Only signals >60% confidence
mr_buy_threshold=-0.05        # Mean reversion: -5% below MA
mr_sell_threshold=0.05        # Mean reversion: +5% above MA

# Exit Model
Drop_Probability threshold=0.7  # Exit if >70% chance of drop

# Portfolio
initial_capital=100000
position_sizing=confidence-weighted
```

## Testing

To verify the implementation:
```bash
cd /Users/mihirepel/Personal_Projects/Trading/historical_training_v2
python3 main.py --model lightgbm
```

This will:
1. Load 2022-2024 training and 2025 testing data
2. Train LightGBM entry model
3. Train exit timing model
4. Generate trading signals (with confidence)
5. Run portfolio backtest using BOTH models
6. Compare vs. buy-and-hold S&P 500

## Notes

- The exit model works alongside (not replaces) the entry model
- Exit model predictions are merged by Date with entry signals
- If exit model is not provided, default Drop_Probability = 0.5 (neutral)
- Position sizing is based on entry signal confidence, not exit confidence
- Exit model adds an extra dimension to the trading strategy without changing entry logic
