#!/bin/bash
# Quick start commands for trading strategy visualization

cd /Users/mihirepel/Personal_Projects/Trading/historical_training_v2

echo "🎯 Trading Strategy Optimization - Quick Start"
echo "=============================================="
echo ""

# View visualizations
echo "📊 Opening visualizations..."
echo ""
echo "Command to open all charts:"
echo "  open results/*.png"
echo ""
echo "Or open individually:"
echo "  open -a Preview results/strategy_performance.png"
echo "  open -a Preview results/signal_analysis.png"
echo "  open -a Preview results/performance_summary.png"
echo ""

# Key metrics
echo "📈 Key Performance Metrics:"
echo "  Return: 21.64% (vs 14.74% baseline = +6.9pp)"
echo "  Drawdown: -18.13% (better than -28.6%)"
echo "  Trades: 15 accepted out of 48 signals"
echo "  Sharpe Ratio: ~1.10"
echo ""

# Files to review
echo "📖 Documentation to read:"
echo "  1. INDEX.md - Navigation guide"
echo "  2. README_OPTIMIZATION.md - Quick reference"
echo "  3. STRATEGY_SUMMARY.md - Why it works"
echo ""

# Run pipeline
echo "🚀 To re-run the pipeline:"
echo "  source .venv/bin/activate"
echo "  python models/optimized_strategy.py"
echo "  python visualize_strategy.py"
echo ""

# View data
echo "📊 To analyze the backtest data:"
echo "  python -c \"import pandas as pd; df = pd.read_csv('results/optimized_strategy_backtest.csv'); print(df.describe())\""
echo ""

echo "✅ Everything ready!"
