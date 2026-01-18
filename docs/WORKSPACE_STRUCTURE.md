# Workspace Organization

## Current Structure (Organized)

```
historical_training_v2/
├── 🎯 Root Level (Navigation & Core Entry Points)
│   ├── FINAL_SUMMARY.md               # Project completion summary
│   ├── INDEX.md                       # Documentation index
│   ├── README_TESTING.md              # Testing guide
│   ├── STRATEGY_SUMMARY.md            # Strategy overview
│   ├── QUICK_REFERENCE.md             # Quick lookup
│   ├── WORKSPACE_STRUCTURE.md         # This file
│   ├── WORKSPACE_CLEANUP_SUMMARY.md   # Cleanup documentation
│   ├── config.py                      # Configuration settings
│   ├── main.py                        # Main execution script
│   ├── compare_strategies.py          # Strategy comparison
│   ├── visualize_strategy.py          # Core visualization
│   └── requirements.txt               # Dependencies
│
├── Core Strategy Files
│   ├── models/                        # Core strategy modules
│   │   ├── optimized_strategy.py     # Main strategy implementation
│   │   ├── signal_generation.py      # Signal generation logic
│   │   ├── ensemble_model.py         # Ensemble modeling
│   │   ├── exit_strategies.py        # Exit logic
│   │   ├── dynamic_sizing.py         # Position sizing
│   │   ├── portfolio_management.py   # Portfolio logic
│   │   ├── regime_management.py      # Market regime detection
│   │   └── lightgbm_model.py        # ML model component
│
├── Testing & Validation
│   ├── tests/                         # Test scripts (organized)
│   │   ├── test_strategy_msft.py     # MSFT 2024 backtest
│   │   ├── test_different_thresholds.py  # Parameter optimization
│   │   └── test_multiple_stocks.py   # Multi-stock validation
│
├── Data
│   └── data/                     # Raw data files
│       ├── SPY_training_2022_2024.csv
│       └── SPY_testing_2025.csv
│
├── Results & Analysis
│   ├── results/                  # Backtest results
│   │   ├── optimized_strategy_backtest.csv    # SPY 2025 results
│   │   ├── MSFT_2024_backtest.csv            # MSFT 2024 results
│   │   ├── multi_stock_backtest.csv          # Multi-stock results
│   │   ├── strategy_comparison.csv           # Comparison table
│   │   ├── sector_performance.csv            # Sector analysis
│   │   ├── threshold_analysis.csv            # Parameter sensitivity
│   │   └── trading_analysis/                 # Detailed analysis
│   │
│   ├── visualization/           # Visualization modules
│   │   ├── portfolio_plots.py
│   │   └── plot_results.py
│
├── Documentation (Organized)
│   ├── docs/
│   │   ├── README.md                    # Main docs readme
│   │   ├── DOCUMENTATION_INDEX.md       # Doc index
│   │   ├── FLOW_DIAGRAM.md             # System flow
│   │   ├── SYSTEM_ARCHITECTURE.md      # Architecture
│   │   ├── REGIME_MANAGEMENT.md        # Regime details
│   │   ├── IMPROVEMENT_ROADMAP.md      # Future plans
│   │   └── summaries/                  # Detailed analysis docs
│   │       ├── CROSS_DATASET_ANALYSIS.md
│   │       ├── CROSS_DATASET_TEST_SUMMARY.txt
│   │       ├── MULTI_STOCK_TEST_RESULTS.txt
│   │       ├── TESTING_SUMMARY.md
│   │       ├── OPTIMIZATION_COMPLETE.txt
│   │       ├── OPTIMIZATION_RESULTS_FINAL.md
│   │       ├── VISUALIZATION_SUMMARY.txt
│   │       ├── PORTFOLIO_VISUALIZATION_GUIDE.md
│   │       ├── README_OPTIMIZATION.md
│   │       └── DELIVERABLES_CHECKLIST.txt
│
├── Utility Scripts
│   ├── scripts/                   # Helper scripts
│   │   ├── fetch_stock_data.py   # Data fetching (optional)
│   │   ├── visualize_results.py  # Results visualization
│   │   └── QUICK_START.sh        # Quick start script
│
├── Archive (Optional - Can Delete)
│   ├── archive/                 # Old documentation
│   ├── old_docs/                # Deprecated doc files
│   └── tests_deprecated/        # Old test scripts
│
├── Environment
│   ├── .venv/                   # Python virtual environment
│   └── requirements.txt         # Dependencies
```

## File Categories

### 🎯 Active Core Files (In Root, Latest)
- `main.py` - Main execution entry point
- `config.py` - Configuration settings
- `compare_strategies.py` - Strategy comparison utility
- `visualize_strategy.py` - Visualization generation

### 🧪 Test Files (In tests/ directory)
- `tests/test_strategy_msft.py` - MSFT 2024 backtest
- `tests/test_different_thresholds.py` - Parameter optimization
- `tests/test_multiple_stocks.py` - Multi-asset validation

### 🔧 Utility Scripts (In scripts/ directory)
- `scripts/fetch_stock_data.py` - Data fetching utility
- `scripts/visualize_results.py` - Results visualization
- `scripts/QUICK_START.sh` - Quick start script

### 📊 Latest Results
- `results/MSFT_2024_backtest.csv` - Jan 16 MSFT test
- `results/multi_stock_backtest.csv` - Jan 16 multi-stock
- `results/strategy_comparison.csv` - Jan 16 comparison
- `results/threshold_analysis.csv` - Jan 16 parameter study
- `results/sector_performance.csv` - Jan 16 sector breakdown
- `results/optimized_strategy_backtest.csv` - Jan 16 SPY test

### 📚 Key Documentation (Read First)
**In Root (Navigation):**
1. `INDEX.md` - Start here for navigation
2. `FINAL_SUMMARY.md` - Project completion summary
3. `README_TESTING.md` - How to run tests
4. `STRATEGY_SUMMARY.md` - Strategy overview
5. `QUICK_REFERENCE.md` - Quick lookup
6. `WORKSPACE_STRUCTURE.md` - This file

**Detailed Results (In docs/summaries/):**
- `CROSS_DATASET_TEST_SUMMARY.txt` - Latest validation results
- `MULTI_STOCK_TEST_RESULTS.txt` - Multi-asset validation
- `TESTING_SUMMARY.md` - Test methodology
- `OPTIMIZATION_RESULTS_FINAL.md` - Optimization details

### 🗂️ Old/Archive Files
- `archive/` - Previous runs and old docs
- `old_docs/` - Deprecated documentation
- `tests_deprecated/` - Old test scripts

## Getting Started

### 1. View Documentation
```bash
# Main project summary (START HERE)
cat FINAL_SUMMARY.md

# Latest test results (in docs/summaries/)
cat docs/summaries/CROSS_DATASET_TEST_SUMMARY.txt
cat docs/summaries/MULTI_STOCK_TEST_RESULTS.txt

# How to run tests
cat README_TESTING.md
```

### 2. Run Strategy
```bash
# Run optimized strategy on new data
python main.py

# Run tests (from tests/ directory)
python tests/test_strategy_msft.py
python tests/test_different_thresholds.py
python tests/test_multiple_stocks.py

# Generate visualizations
python visualize_strategy.py
```

### 3. Review Results
```bash
# Check backtest results
cat results/MSFT_2024_backtest.csv
cat results/multi_stock_backtest.csv

# Compare strategies
cat results/strategy_comparison.csv

# View detailed analysis (in docs/summaries/)
cat docs/summaries/TESTING_SUMMARY.md
```

## Key Metrics (Latest)

### SPY 2025 (Original)
- Return: 21.64%
- Buy & Hold: 14.74%
- Outperformance: +6.90pp (+47%)
- Sharpe: 1.10
- Max Drawdown: -18.13%

### MSFT 2024 (Optimized)
- Return: 22.24%
- Buy & Hold: 16.06%
- Outperformance: +6.18pp (+38%)
- Sharpe: 1.71
- Max Drawdown: -0.02%

### Multi-Stock Average
- Average Return: 22.45%
- Average Outperformance: +6.44pp
- Average Sharpe: 1.41

## Next Steps

### Immediate
- [ ] Test on 3-5 more stocks (AAPL, GOOGL, NVDA, TSLA, JPM)
- [ ] Test on historical years (2023, 2022, 2021)
- [ ] Walk-forward validation

### Short-term
- [ ] Parameter sensitivity analysis
- [ ] Market regime detector
- [ ] Adaptive threshold system

### Medium-term
- [ ] Multi-stock portfolio
- [ ] Real-time signal generation
- [ ] Live trading implementation

## File Cleanup Guide

### Safe to Delete
- `archive/` - Old versions of files
- `old_docs/` - Deprecated documentation
- `tests_deprecated/` - Old test scripts
- `fetch_stock_data.py` - If not using live data
- Old PNG visualization files (keep latest)

### Keep
- All files in `models/` - Core strategy
- All active test files
- All result CSVs in `results/`
- All key documentation in root

## Maintenance

Run this monthly to clean up:
```bash
# Remove old backup files
rm -rf archive/backup_*

# Clear old results
rm results/old_*/

# Archive old docs
mv docs/old_* old_docs/
```

Generated: Jan 16, 2026
Status: ✅ Organized and ready for deployment

