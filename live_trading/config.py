"""
Live Trading Configuration
==========================
API keys and trading parameters for the Alpaca trading bot.
"""

# ============================================================================
# ALPACA API CREDENTIALS
# ============================================================================
# Paper trading account (no real money)
ALPACA_API_KEY = "PKQCAK3OCWGUZ5JHL6QFBXNSLV"
ALPACA_SECRET_KEY = "51B23zrTEMd9sBbXqWyqaewmmum5XK6oUsSn4RThcTRU"

# Set to False for live trading (BE CAREFUL!)
PAPER_TRADING = True

# Base URL for Alpaca API
ALPACA_BASE_URL = "https://paper-api.alpaca.markets" if PAPER_TRADING else "https://api.alpaca.markets"


# ============================================================================
# TRADING PARAMETERS
# ============================================================================

# Symbol to trade
SYMBOL = "SPY"

# Position sizing
MAX_POSITION_SIZE = 0.10  # Max 10% of portfolio per trade
MIN_TRADE_VALUE = 100     # Minimum trade value in dollars

# Risk management
MAX_DAILY_LOSS = 0.02     # Stop trading if down 2% for the day
MAX_DRAWDOWN = 0.05       # Stop trading if down 5% from peak

# Confidence thresholds (from model)
HIGH_CONFIDENCE_THRESHOLD = 0.7
MIN_CONFIDENCE_THRESHOLD = 0.5

# Session-based position sizing multipliers
SESSION_MULTIPLIERS = {
    'premarket': 0.5,     # Half size in pre-market (wider spreads)
    'regular': 1.0,       # Full size during regular hours
    'afterhours': 0.5,    # Half size in after-hours (wider spreads)
}

# Prediction horizon (from model training)
PREDICTION_HORIZON_MINUTES = 20


# ============================================================================
# TRADING SCHEDULE (Eastern Time)
# ============================================================================

TRADING_HOURS = {
    'premarket_start': '04:00',
    'premarket_end': '09:30',
    'regular_start': '09:30',
    'regular_end': '16:00',
    'afterhours_start': '16:00',
    'afterhours_end': '20:00',
}


# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Retrain model every N days
RETRAIN_INTERVAL_DAYS = 7

# Minimum training samples required
MIN_TRAINING_SAMPLES = 10000

# Feature columns (must match model training)
FEATURE_COLUMNS = [
    'SMA_5', 'SMA_10', 'SMA_20', 'SMA_60',
    'EMA_5', 'EMA_12', 'EMA_26',
    'MACD', 'Signal_Line', 'MACD_Histogram',
    'RSI_14', 'Momentum_5', 'Momentum_10',
    'ROC_5', 'ROC_10', 'BB_Upper', 'BB_Lower', 'BB_Position',
    'ATR_14', 'Volatility_10',
    'HL_Range', 'HL_Range_Pct', 'CO_Range_Pct'
]


# ============================================================================
# LOGGING & MONITORING
# ============================================================================

# Log level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL = "INFO"

# Save trade history to CSV
SAVE_TRADE_HISTORY = True

# Slack/Discord webhook for alerts (optional)
ALERT_WEBHOOK_URL = None
