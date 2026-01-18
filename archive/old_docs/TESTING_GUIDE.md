# Quick Reference - Testing & Validation

## Running the System

### Basic Execution
```bash
cd /Users/mihirepel/Personal_Projects/Trading/historical_training_v2
python3 main.py --model lightgbm
```

### With Explicit Model Selection
```bash
python3 main.py --model lightgbm
```

## Expected Output

### Phase 1: Data Loading
```
Loading training data from SPY_training_2022_2024.csv...
  Loaded 505 rows (2022-2024), X columns

Loading testing data from SPY_testing_2025.csv...
  Loaded 15 rows, X columns
```

### Phase 2: Indicator Calculation
```
Calculating indicators on training data...
  ✓ Training data with indicators: 495 rows

Calculating indicators on testing data (with 200-row warm-up)...
  ✓ Testing data with indicators: 10 rows
```

### Phase 3: Model Training
```
============================================================
Step 2: Train Model (ON TRAINING DATA ONLY)
============================================================
Training model: lightgbm...
LightGBM Training:
  Classes: [0 1]
  Train accuracy: X%
  Test accuracy: Y%
  Feature importance: [list of top 10 features]
```

### Phase 4: Exit Model Training (NEW)
```
============================================================
Step 2B: Train EXIT Model (Predict Short-Term Price Drops)
============================================================
Exit Model Training:
  Classes: [0 1]
  Train accuracy: X%
  Test accuracy: Y%
  Exit signals generated: N rows
```

### Phase 5: Signal Generation
```
============================================================
Step 3: Generate Trading Signals (LOCKED HYPERPARAMETERS + CONFIDENCE)
============================================================
Signal Generation Results:
  Total signals: N
  BUY signals: X
  SELL signals: Y
  HOLD signals: Z
  Average confidence: 0.XX
```

### Phase 6: Portfolio Backtest (WITH EXIT MODEL)
```
============================================================
Step 4: Run Portfolio Backtest (DUAL-MODEL: Entry + Exit)
============================================================

Initial Capital: $100,000.00
Strategy: AGGRESSIVE BUYING + SMART HOLDING
Position Sizing: CONFIDENCE-WEIGHTED
Exit Strategy: EXIT MODEL (predicts price drops >70% probability)

Performance Metrics:
  Final Portfolio Value: $XXX,XXX.XX
  Total Return: $XX,XXX.XX (XX.XX%)
  Win Rate: XX.XX%
  Max Drawdown: -XX.XX%
  Daily Volatility: X.XX%
  Sharpe Ratio: X.XX
  Max Portfolio Value: $XXX,XXX.XX
  Min Portfolio Value: $XX,XXX.XX
```

### Phase 7: Comparison Results
```
============================================================
LEAK-FREE BACKTEST RESULTS
============================================================

Strategy Performance:
  Final Value:          $XXX,XXX.XX
  Return:               +XX.XX%
  Max Drawdown:         -XX.XX%

Buy-and-Hold (S&P 500):
  Final Value:          $XXX,XXX.XX
  Return:               +XX.XX%
  Max Drawdown:         -XX.XX%

Comparison:
  Absolute Gain:        $XX,XXX.XX
  Outperformance:       +XX.XXpp
  Strategy Wins?        ✓ YES (if positive)
```

### Phase 8: Data Integrity Verification
```
============================================================
DATA INTEGRITY VERIFICATION
============================================================

✓ Indicators calculated separately on training and testing data
✓ No training data mixed into test data
✓ Model trained on training data ONLY
✓ Predictions made on test data ONLY
✓ Hyperparameters locked before testing (not tuned on test set)
✓ All results are TRUE OUT-OF-SAMPLE
✓ NO LOOKAHEAD BIAS - Safe to use for real trading
```

## Key Differences to Look For

### New in This Version
1. **Exit Model Training**: "Step 2B: Train EXIT Model" section
   - Shows exit model accuracy
   - Indicates how many drop predictions were made

2. **Exit Strategy Message**: 
   - Old: "Exit Strategy: NO EXIT SIGNALS - HOLD ALL POSITIONS"
   - New: "Exit Strategy: EXIT MODEL (predicts price drops >70% probability)"

3. **Exit Model Columns** in backtest output:
   - `Drop_Probability`: 0.0-1.0 probability of drop
   - `Exit_Signal_Strength`: Confidence in drop prediction
   - See these when opening `portfolio_backtest.csv`

4. **Potential Portfolio Changes**:
   - Exits may happen mid-holding period
   - `Trade_Size_%` = -100.0 indicates exit
   - `Shares_Held` drops to 0 when exiting
   - `Cash` increases when exiting

## Validation Checklist

### Code Execution
- [ ] No Python syntax errors
- [ ] All imports successful
- [ ] No module not found errors
- [ ] No missing CSV files

### Data Integrity
- [ ] Training data: 505 rows
- [ ] Testing data: 15 rows
- [ ] After indicator calc: ~495 train, ~10 test
- [ ] No NaN values in final datasets

### Model Training
- [ ] Entry model trains successfully
- [ ] Exit model trains successfully (NEW)
- [ ] Both models produce predictions
- [ ] Predictions are 0-1 range

### Signal Generation
- [ ] BUY and SELL signals generated
- [ ] Confidence scores 0-1 range
- [ ] Confidence threshold filtering works (>0.6)
- [ ] Mean reversion overlay applied

### Portfolio Backtest
- [ ] Backtest completes without error
- [ ] Exit model used for exits (NEW)
- [ ] SELL signals ignored (no early exits)
- [ ] Position sizing by confidence
- [ ] Drop_Probability column present

### Output Files
- [ ] `trading_signals.csv` created
- [ ] `portfolio_backtest.csv` created
- [ ] Both files have expected columns
- [ ] All rows have valid data

## Debugging Tips

### If Exit Model Doesn't Seem Active

Check the `portfolio_backtest.csv`:
```python
import pandas as pd

df = pd.read_csv('results/trading_analysis/portfolio_backtest.csv')

# Should see Drop_Probability column
print(df['Drop_Probability'].describe())
# Mean should be around 0.5 (average probability)

# Should see some -100.0 in Trade_Size_%
print(df[df['Trade_Size_%'] == -100.0].shape)
# Should have some rows (exits triggered)

# Shares_Held should have zeros (from exits)
print((df['Shares_Held'] == 0).sum())
# Should be > 0
```

### If Still No Exits

**Issue 1**: Exit model not trained
- Check console output for "Step 2B" section
- Verify `exit_model.py` runs without error

**Issue 2**: Drop probabilities all < 0.7
- Check: `print(df['Drop_Probability'].max())`
- If max < 0.7, no exits will trigger
- This is normal if market conditions don't show drops

**Issue 3**: exit_model_df not passed correctly
- Verify main.py line 205: `exit_model_df=exit_predictions_df`
- Verify portfolio_management.py line 225: receives `exit_model_df=exit_model_df`

### If Getting Errors

**ImportError: No module named 'exit_model'**
- Check: `ls -la models/exit_model.py` exists
- Verify: `python3 -c "from models.exit_model import main"`

**KeyError: 'Drop_Probability'**
- Check: exit_model returns DataFrame with this column
- Verify: `print(exit_predictions_df.columns)`

**Date merge issues**
- Check: Both dataframes have 'Date' column
- Verify: Dates are datetime format
- Check: Date ranges overlap

## Performance Interpretation

### Good Signs ✓
- Win Rate > 50% (better than random)
- Sharpe Ratio > 0.5 (decent risk-adjusted)
- Max Drawdown < 20% (controlled risk)
- Avg Win > Avg Loss (favorable ratio)

### Excellent Signs ✓✓
- Win Rate > 55% (entry model working)
- Sharpe Ratio > 1.0 (very good returns)
- Max Drawdown < 10% (excellent risk control)
- Outperforms S&P 500 (beating benchmark)

### Red Flags ⚠
- Win Rate < 40% (worse than random)
- Many exits triggered (>50% of days) → adjust threshold
- Exit model never triggered (Drop_Prob always < 0.7) → model issue
- Strategy underperforms S&P 500 significantly → rethink strategy

## Adjusting Parameters

### To Get More Exits
```python
# Line 108 in portfolio_management.py
if drop_prob > 0.7:  # ← Lower this threshold
if drop_prob > 0.6:  # More conservative exits
```

### To Get Fewer Exits  
```python
if drop_prob > 0.8:  # More aggressive holds
```

### To Change Entry Sizing
```python
# Lines 81-95 in portfolio_management.py
if confidence > 0.8:  position_size = 0.75  # ← Adjust these
elif confidence > 0.65: position_size = 0.5
```

### To Change Entry Threshold
```python
# In main.py line 196
confidence_threshold=0.6  # ← Change this value
```

## Files to Check

After running, examine these:
```bash
results/
├── trading_analysis/
│   ├── trading_signals.csv          # Entry signals
│   ├── portfolio_backtest.csv       # Backtest results with exits
│   └── portfolio_metrics.json       # Performance metrics
├── model_predictions/
│   ├── model_predictions.csv        # Entry model output
│   └── exit_model_predictions.csv   # Exit model output (NEW)
└── visualizations/
    ├── model_performance/           # Prediction charts
    └── portfolio_performance/       # Equity curves
```

## Quick Validation Script

```python
#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Load results
backtest = pd.read_csv('results/trading_analysis/portfolio_backtest.csv')
signals = pd.read_csv('results/trading_analysis/trading_signals.csv')
exit_preds = pd.read_csv('results/model_predictions/exit_model_predictions.csv')

print("=== DUAL-MODEL VALIDATION ===\n")

# Check entry signals
print(f"Entry signals: {len(signals)} rows")
print(f"  BUY: {(signals['Signal'] == 'BUY').sum()}")
print(f"  SELL: {(signals['Signal'] == 'SELL').sum()}")
print(f"  HOLD: {(signals['Signal'] == 'HOLD').sum()}")
print(f"  Avg confidence: {signals['Confidence'].mean():.3f}\n")

# Check exit model
print(f"Exit model predictions: {len(exit_preds)} rows")
print(f"  Drop probability range: {exit_preds['Drop_Probability'].min():.3f} - {exit_preds['Drop_Probability'].max():.3f}")
print(f"  Mean: {exit_preds['Drop_Probability'].mean():.3f}\n")

# Check backtest
print(f"Backtest results: {len(backtest)} rows")
print(f"  Exits triggered: {(backtest['Trade_Size_%'] == -100.0).sum()}")
print(f"  Entries made: {(backtest['Trade_Size_%'] > 0).sum()}")
print(f"  Final portfolio: ${backtest['Portfolio_Value'].iloc[-1]:,.2f}")
print(f"  Return: {backtest['Cumulative_Return'].iloc[-1]:.2f}%\n")

# Check data alignment
print(f"Date alignment: {(backtest['Date'] == signals['Date']).sum()} / {len(backtest)} match")
if 'Drop_Probability' in backtest.columns:
    print("✓ Exit model successfully integrated")
else:
    print("✗ Exit model not found in backtest")
```

## Success Criteria

**Minimal Success**: 
✓ Code runs without errors
✓ All sections execute (Steps 1-4)
✓ Exit model is trained and used
✓ Portfolio backtest includes Drop_Probability column

**Strong Success**:
✓ Win rate > 50%
✓ Sharpe ratio > 0.5
✓ Some exits triggered (Drop_Probability > 0.7)
✓ Returns > 0% (positive performance)

**Excellent Success**:
✓ Win rate > 55%
✓ Sharpe ratio > 1.0
✓ Outperforms S&P 500
✓ Max drawdown < 15%
✓ Logical exit patterns (exits before drops)
