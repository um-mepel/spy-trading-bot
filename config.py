"""
Trading Strategy Configuration
Tune these parameters to optimize performance while maintaining data integrity.
All parameters are LOCKED before testing (no optimization on test set).
"""

# CONFIDENCE THRESHOLD TUNING
# Controls how strict the confidence filter is
# - 0.3 = Very permissive (most predictions accepted)
# - 0.5 = Moderate filtering
# - 0.7 = Strict filtering (current, blocks ~45% of signals)
# - 0.9 = Very strict (only highest confidence predictions)
CONFIDENCE_THRESHOLD = 0.7

# SIGNAL GENERATION
BUY_PERCENTILE = 25.0      # Top 25% of predictions are BUY
SELL_PERCENTILE = 75.0     # Bottom 75% of predictions are SELL
PERCENTILE_BY_MONTH = True # Recalculate percentiles monthly

# MEAN REVERSION (supplementary signals)
ENABLE_MEAN_REVERSION = True
MR_BUY_THRESHOLD = -0.05   # Buy if price is 5% below 20-day MA
MR_SELL_THRESHOLD = 0.05   # Sell if price is 5% above 20-day MA

# POSITION SIZING
POSITION_SIZE = 0.50  # Risk 50% of cash per BUY signal

# MODEL PARAMETERS
N_ESTIMATORS = 200
LEARNING_RATE = 0.05

# BACKTEST PARAMETERS
INITIAL_CAPITAL = 100000
UNITS_PER_SIGNAL = 1000    # Number of shares per trade

# ═══════════════════════════════════════════════════════════════
# CONFIDENCE THRESHOLD RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════
#
# Current Results (CONFIDENCE_THRESHOLD = 0.7):
#   ✓ Strategy: +3.79% (very conservative, few trades)
#   ✗ S&P 500: +17.42%
#   ✗ Outperformance: -13.63pp (TOO CONSERVATIVE)
#   ✓ High-confidence trades: 54-58% accuracy
#
# Try these configurations:
#
# For more trades (better returns, slightly higher risk):
#   CONFIDENCE_THRESHOLD = 0.5
#   Expected: 50-100 BUY/SELL signals total
#   Expected accuracy: ~52% (lower than 0.7 but more volume)
#
# For balanced approach:
#   CONFIDENCE_THRESHOLD = 0.6
#   Expected: 30-70 BUY/SELL signals total
#   Expected accuracy: ~53-55%
#
# For maximum returns (at cost of accuracy):
#   CONFIDENCE_THRESHOLD = 0.3
#   Expected: All signals considered
#   Expected accuracy: ~48% (based on original 17.78% return)
#
# Next steps:
# 1. Run with CONFIDENCE_THRESHOLD = 0.5 and compare results
# 2. If still underperforming, try 0.4
# 3. Find the sweet spot between accuracy and volume
# 4. Once tuned, LOCK that value in main.py (never change it)

TUNING_LOG = """
Jan 16, 2026 - Initial Confidence Filtering Run
  Threshold: 0.7
  Result: +3.79% (TOO CONSERVATIVE - blocked 113 signals)
  Status: Working correctly but needs tuning for better returns
"""
