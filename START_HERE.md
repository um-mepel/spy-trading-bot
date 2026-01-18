# SPY Trading Bot - Quick Start Guide

## 🚀 Quick Commands

```bash
# Run main backtesting pipeline (daily data)
python3 main.py

# Run minute-level test on real Alpaca data
python3 tests/test_real_minute_data_strict.py

# Generate visualizations
python3 visualization/visualize_full_period.py
python3 visualization/simulate_portfolio.py

# Start live trading bot (paper trading)
python3 live_trading/run_bot.py --simulate

# Deploy to Google Cloud
./live_trading/deploy/deploy.sh trading-bot-vm us-central1-a
```

---

## 📁 Project Structure

```
historical_training_v2/
│
├── main.py                    # Main backtesting pipeline (daily data)
├── config.py                  # Global configuration parameters
├── trading_model.py           # Core trading logic (Kelly, risk mgmt)
├── requirements.txt           # Python dependencies
│
├── models/                    # ML models and trading logic
│   ├── lightgbm_model.py      # LightGBM prediction model
│   ├── ensemble_model.py      # Ensemble (LGB + XGB + RF)
│   ├── signal_generation.py   # Trading signal generation
│   ├── portfolio_management.py # Backtesting engine
│   ├── regime_management.py   # Market regime detection
│   └── exit_model.py          # Exit signal model
│
├── live_trading/              # Live trading bot (Alpaca)
│   ├── run_bot.py             # Entry point
│   ├── trading_bot.py         # Bot logic
│   ├── model_trainer.py       # Model training
│   ├── config.py              # Bot configuration
│   └── deploy/                # Google Cloud deployment
│       ├── deploy.sh          # Deployment script
│       ├── setup_vm.sh        # VM setup script
│       └── GCLOUD_SETUP.md    # Deployment guide
│
├── tests/                     # Test scripts
│   ├── test_real_minute_data_strict.py  # Main minute-level test
│   └── ...
│
├── visualization/             # Visualization scripts
│   ├── visualize_full_period.py    # 6-month analysis
│   ├── simulate_portfolio.py       # Portfolio simulation
│   ├── visualize_single_day.py     # Single day view
│   └── ...
│
├── scripts/                   # Utility scripts
│   ├── fetch_stock_data.py    # Data fetching (yfinance)
│   ├── fetch_alphavantage_data.py  # Alpha Vantage data
│   └── ...
│
├── data/                      # Training/testing data
│   ├── SPY_training_2022_2024.csv
│   └── SPY_testing_2025.csv
│
├── results/                   # Output results
│   ├── real_minute_strict/    # Real minute data results
│   │   ├── visualizations/    # Charts and graphs
│   │   └── *.csv              # Predictions, trades
│   └── ...
│
├── docs/                      # Documentation
│   ├── README.md              # Project overview
│   ├── QUICK_REFERENCE.md     # Command reference
│   └── ...
│
└── archive/                   # Old/deprecated files
```

---

## 📊 Latest Results (Real Minute Data)

**Data Period:** Jan 2020 - Dec 2024 (4.5 years)
- Training: 928,524 minute bars
- Testing: 99,325 minute bars

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 56.74% |
| **High Confidence Accuracy** | 58.23% |
| **Edge vs Random** | +6.74% |
| **Z-score** | 42.89 |
| **Statistically Significant** | ✅ YES |

### Portfolio Simulation ($100k starting)

| Metric | Value |
|--------|-------|
| **Final Value** | $117,685 |
| **Total Return** | +17.68% |
| **Win Rate** | 65.0% |
| **Profit Factor** | 6.92 |
| **Max Drawdown** | -0.08% |
| **vs Buy-Hold Alpha** | +9.89% |

---

## 🔧 Configuration

Edit `config.py` for main pipeline settings:
- `CONFIDENCE_THRESHOLD` - Min confidence for trades
- `POSITION_SIZE` - Default position size %
- `BUY_PERCENTILE` / `SELL_PERCENTILE` - Signal thresholds

Edit `live_trading/config.py` for live bot:
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`
- `PAPER_TRADING` - True for paper, False for real
- `SYMBOL` - Ticker to trade
- `DAILY_LOSS_LIMIT_PCT` - Risk management

---

## 🌐 Live Trading Bot

The bot is currently deployed on Google Cloud:
- **VM:** trading-bot-vm
- **Mode:** Paper Trading
- **Symbol:** SPY
- **Sessions:** Pre-market, Regular, After-hours

```bash
# SSH to VM
gcloud compute ssh trading-bot-vm --zone=us-central1-a

# View logs
sudo journalctl -u trading-bot -f

# Check status
sudo systemctl status trading-bot
```

---

## 📈 Key Visualizations

All saved to `results/real_minute_strict/visualizations/`:

1. `01_real_data_overview.png` - Model performance overview
2. `02_trading_simulation.png` - Trade simulation results
3. `05_full_6month_analysis.png` - 6-month comprehensive view
4. `06_portfolio_simulation.png` - Portfolio performance

---

## 🔑 API Keys

Alpaca keys are configured in `live_trading/config.py`:
- Paper trading account (no real money)
- Extended hours trading enabled
