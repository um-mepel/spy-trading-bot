"""
Stock Picker Live Trading Bot for Alpaca
=========================================
Daily S&P 500 stock picker based on the backtested strategy:
- Trains LightGBM model on S&P 500 universe
- Picks top 5 stocks daily based on confidence
- 5-day holding period
- Monthly model retraining

Based on extended backtest showing 230% return vs SPY 132% (2020-2026)

PAPER TRADING BY DEFAULT - Set PAPER_TRADING=False for live trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging
import json
import pickle
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: LightGBM not installed. Install with: pip install lightgbm")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import pytz

from .config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER_TRADING
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('stock_picker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Strategy parameters (from backtest)
TOP_N_PER_DAY = 5          # Number of stocks to pick daily
HOLD_DAYS = 5              # Days to hold each position
RETRAIN_DAYS = 21          # Retrain model every ~month
INITIAL_TRAIN_YEARS = 2    # Years of training data

# Position sizing
POSITION_SIZE_PCT = 4.0    # 4% per position (allows ~25 concurrent)

# S&P 500 Universe (comprehensive list)
SP500_TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'BRK-B', 'UNH',
    'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP',
    'KO', 'COST', 'AVGO', 'LLY', 'WMT', 'MCD', 'CSCO', 'ACN', 'TMO', 'ABT',
    'DHR', 'NEE', 'VZ', 'ADBE', 'NKE', 'PM', 'TXN', 'CRM', 'BMY', 'QCOM',
    'RTX', 'HON', 'UPS', 'MS', 'UNP', 'LOW', 'INTC', 'IBM', 'SPGI', 'GS',
    'CAT', 'DE', 'INTU', 'AXP', 'BLK', 'ELV', 'AMD', 'ISRG', 'GILD', 'SYK',
    'BKNG', 'ADI', 'MDLZ', 'PLD', 'VRTX', 'ADP', 'REGN', 'TMUS', 'CI', 'MMC',
    'CB', 'LMT', 'CME', 'ZTS', 'SCHW', 'MO', 'DUK', 'SO', 'EQIX', 'PNC',
    'SLB', 'CL', 'BDX', 'ITW', 'ETN', 'AON', 'USB', 'GE', 'FISV', 'WM',
    'HUM', 'CSX', 'TGT', 'PGR', 'NOC', 'TJX', 'MCO', 'ICE', 'BSX', 'ORLY',
    'NSC', 'CCI', 'FCX', 'GD', 'EMR', 'APD', 'HCA', 'AZO', 'SNPS',
    'CDNS', 'MAR', 'SHW', 'MCK', 'NXPI', 'F', 'GM', 'KLAC', 'ANET', 'MNST',
    'ROP', 'CMG', 'FTNT', 'KMB', 'ECL', 'PSA', 'PCAR', 'MPC', 'PSX', 'VLO',
    'EW', 'LRCX', 'AIG', 'ROST', 'TRV', 'AFL', 'OXY', 'CTVA', 'AEP',
    'SRE', 'D', 'WELL', 'O', 'KMI', 'OKE', 'FANG', 'DVN', 'HAL', 'WMB',
    'STZ', 'YUM', 'KHC', 'GIS', 'SYY', 'HSY', 'K', 'ADM', 'TAP', 'CAG',
    'BIIB', 'MRNA', 'ZBH', 'DXCM', 'IQV', 'MTD', 'A', 'IDXX', 'BAX',
    'LH', 'DGX', 'VTRS', 'PKI', 'TECH', 'BIO', 'HOLX', 'WAT',
    'AME', 'ROK', 'PH', 'IR', 'OTIS', 'CARR', 'XYL', 'IEX', 'GNRC', 'DOV',
    'FAST', 'RMD', 'TDY', 'KEYS', 'ZBRA', 'TRMB', 'FTV', 'GRMN', 'TER', 'ANSS',
    'PTC', 'CDW', 'EPAM', 'IT', 'CTSH', 'DXC', 'LDOS', 'SAIC', 'CSGP', 'PAYC',
    'PAYX', 'CEQP', 'WDC', 'STX', 'NTAP', 'SWKS', 'MCHP', 'ON', 'MPWR', 'ENPH',
    'SEDG', 'FSLR', 'CEG', 'ES', 'EXC', 'ED', 'XEL', 'WEC', 'DTE', 'AEE',
    'CMS', 'CNP', 'NI', 'EVRG', 'PNW', 'ATO', 'NRG', 'PPL', 'FE', 'LNT',
    'SPG', 'AMT', 'SBAC', 'DLR', 'EQR', 'AVB', 'UDR', 'ESS', 'MAA', 'CPT',
    'VTR', 'PEAK', 'HST', 'REG', 'KIM', 'BXP', 'ARE', 'SLG', 'VNO',
    'AOS', 'MAS', 'LII', 'JCI', 'LEN', 'PHM', 'NVR', 'DHI', 'TOL', 'MTH',
    'KBH', 'MDC', 'MHO', 'GRBK', 'CCS', 'LGIH', 'TPH', 'TMHC', 'SKY',
    'ALLE', 'HWM', 'TXT', 'LHX', 'HII', 'AXON', 'LDOS', 'CW', 'MOG-A', 'WWD',
    'TDG', 'HEI', 'FTAI', 'RBC', 'AER', 'AL', 'FLS', 'WTS', 'AAON', 'TREX',
    'SCI', 'POOL', 'WSO', 'SITE', 'FOUR', 'DDOG', 'CRWD', 'NET', 'ZS', 'SNOW',
    'MDB', 'OKTA', 'TEAM', 'HUBS', 'NOW', 'WDAY', 'VEEV', 'ZM', 'DOCU', 'ROKU',
    'SQ', 'PYPL', 'COIN', 'HOOD', 'SOFI', 'AFRM', 'UPST', 'LC', 'OPEN', 'RDFN',
    'ABNB', 'BKNG', 'EXPE', 'TRIP', 'MAR', 'HLT', 'H', 'WH', 'IHG', 'CHH',
    'CCL', 'RCL', 'NCLH', 'LUV', 'DAL', 'UAL', 'AAL', 'ALK', 'JBLU', 'SAVE',
    'FDX', 'UPS', 'XPO', 'ODFL', 'SAIA', 'JBHT', 'CHRW', 'EXPD', 'LSTR', 'HUBG',
    'URI', 'PCAR', 'PACW', 'CMC', 'NUE', 'STLD', 'RS', 'ATI', 'CLF', 'X',
    'AA', 'CENX', 'ARNC', 'KALU', 'ZEUS', 'HAYN', 'IIIN', 'USAP', 'HY', 'MLI',
    'DIS', 'CMCSA', 'NFLX', 'WBD', 'PARA', 'FOX', 'FOXA', 'NWS', 'NWSA', 'NYT',
    'OMC', 'IPG', 'PUB', 'WPP', 'MGIC', 'JKHY', 'FIS', 'FISV', 'GPN', 'AXP',
    'COF', 'DFS', 'SYF', 'ALLY', 'CACC', 'OMF', 'SLM', 'NAVI', 'ECPG',
    'BAC', 'WFC', 'C', 'USB', 'PNC', 'TFC', 'MTB', 'FITB', 'HBAN', 'RF',
    'CFG', 'KEY', 'ZION', 'CMA', 'ALLY', 'FCNCA', 'FHN', 'FRC', 'SIVB', 'WAL',
    'SBNY', 'PACW', 'EWBC', 'BOH', 'OZK', 'FNB', 'UMBF', 'PNFP', 'HWC', 'ABCB',
    'MET', 'PRU', 'AFL', 'LNC', 'UNM', 'GL', 'PFG', 'VOYA', 'EQH', 'ATH',
    'RGA', 'CNO', 'FAF', 'FNF', 'ESNT', 'RDN', 'MTG', 'NMIH', 'ACGL', 'RNR',
    'ALL', 'PGR', 'CB', 'TRV', 'CNA', 'WRB', 'HIG', 'L', 'AIZ', 'AFG',
    'XOM', 'CVX', 'COP', 'EOG', 'PXD', 'DVN', 'FANG', 'OXY', 'MRO', 'APA',
    'BKR', 'HAL', 'SLB', 'NOV', 'CHX', 'HP', 'RIG', 'DO', 'VAL', 'NE',
    'SPY', 'QQQ', 'IWM', 'DIA', 'MDY', 'IWF', 'IWD', 'VTV', 'VUG', 'SCHG',
]


class StockPickerBot:
    """
    Daily stock picker bot for Alpaca paper trading.
    Based on the backtested LightGBM strategy.
    """
    
    def __init__(self):
        """Initialize the stock picker bot."""
        logger.info("=" * 60)
        logger.info("INITIALIZING STOCK PICKER BOT")
        logger.info("=" * 60)
        
        # Alpaca client
        self.trading_client = TradingClient(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY,
            paper=PAPER_TRADING
        )
        
        # Model storage
        self.model_dir = Path(__file__).parent / "saved_models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "stock_picker_model.pkl"
        self.state_path = self.model_dir / "stock_picker_state.json"
        
        # Model and data
        self.model = None
        self.stock_data = {}
        self.feature_cols = None
        
        # Trading state
        self.active_positions = {}  # ticker -> {entry_date, entry_price, qty}
        self.last_train_date = None
        self.last_pick_date = None
        
        # Timezone
        self.et_tz = pytz.timezone('US/Eastern')
        
        # Load saved state
        self._load_state()
        
        logger.info(f"Paper Trading: {PAPER_TRADING}")
        logger.info(f"Top N per day: {TOP_N_PER_DAY}")
        logger.info(f"Hold days: {HOLD_DAYS}")
        logger.info(f"Position size: {POSITION_SIZE_PCT}%")
        
    def _load_state(self):
        """Load saved state from disk."""
        if self.state_path.exists():
            with open(self.state_path, 'r') as f:
                state = json.load(f)
                self.active_positions = state.get('active_positions', {})
                self.last_train_date = state.get('last_train_date')
                self.last_pick_date = state.get('last_pick_date')
                logger.info(f"Loaded state: {len(self.active_positions)} active positions")
        
        if self.model_path.exists():
            with open(self.model_path, 'rb') as f:
                saved = pickle.load(f)
                self.model = saved.get('model')
                self.feature_cols = saved.get('feature_cols')
                logger.info("Loaded saved model")
    
    def _save_state(self):
        """Save current state to disk."""
        state = {
            'active_positions': self.active_positions,
            'last_train_date': self.last_train_date,
            'last_pick_date': self.last_pick_date
        }
        with open(self.state_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def _save_model(self):
        """Save model to disk."""
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_cols': self.feature_cols,
                'train_date': self.last_train_date
            }, f)
    
    def download_stock_data(self, years_back=3):
        """Download historical data for all stocks."""
        logger.info(f"Downloading {len(SP500_TICKERS)} stocks for {years_back} years...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)
        
        self.stock_data = {}
        failed = []
        
        for i, ticker in enumerate(SP500_TICKERS):
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                # Handle multi-level columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                if len(df) > 100:
                    df.index = pd.to_datetime(df.index)
                    self.stock_data[ticker] = df
            except Exception as e:
                failed.append(ticker)
            
            if (i + 1) % 50 == 0:
                logger.info(f"  Downloaded {i+1}/{len(SP500_TICKERS)}...")
        
        logger.info(f"Successfully downloaded {len(self.stock_data)} stocks, {len(failed)} failed")
        return len(self.stock_data) > 100
    
    def create_features(self, df):
        """Create technical features for prediction."""
        features = pd.DataFrame(index=df.index)
        
        # Returns
        features['Return_1d'] = df['Close'].pct_change()
        features['Return_5d'] = df['Close'].pct_change(5)
        features['Return_20d'] = df['Close'].pct_change(20)
        
        # Volatility
        features['Volatility_5d'] = df['Close'].pct_change().rolling(5).std()
        features['Volatility_20d'] = df['Close'].pct_change().rolling(20).std()
        
        # Moving averages
        features['SMA_5'] = df['Close'].rolling(5).mean()
        features['SMA_20'] = df['Close'].rolling(20).mean()
        features['SMA_50'] = df['Close'].rolling(50).mean()
        
        # Price relative to MAs
        features['Price_to_SMA5'] = df['Close'] / features['SMA_5'] - 1
        features['Price_to_SMA20'] = df['Close'] / features['SMA_20'] - 1
        features['Price_to_SMA50'] = df['Close'] / features['SMA_50'] - 1
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        features['RSI_14'] = 100 - (100 / (1 + rs))
        
        delta5 = df['Close'].diff()
        gain5 = (delta5.where(delta5 > 0, 0)).rolling(window=5).mean()
        loss5 = (-delta5.where(delta5 < 0, 0)).rolling(window=5).mean()
        rs5 = gain5 / (loss5 + 1e-10)
        features['RSI_5'] = 100 - (100 / (1 + rs5))
        
        # Volume features
        if 'Volume' in df.columns:
            features['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            features['Volume_Change'] = df['Volume'].pct_change()
        
        # Range features
        if 'High' in df.columns and 'Low' in df.columns:
            features['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
            
            # ATR
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            features['ATR_14'] = ranges.max(axis=1).rolling(14).mean()
        
        # Mean reversion
        features['Distance_from_20d_high'] = df['Close'] / df['Close'].rolling(20).max() - 1
        features['Distance_from_20d_low'] = df['Close'] / df['Close'].rolling(20).min() - 1
        
        # Trend strength
        features['Trend_5d'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
        features['Trend_20d'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)
        
        return features
    
    def prepare_training_data(self, end_date, min_days=252):
        """Prepare training data from stock data up to end_date."""
        X_list = []
        y_list = []
        
        for ticker, df in self.stock_data.items():
            train_df = df[df.index < end_date].copy()
            
            if len(train_df) < min_days:
                continue
            
            features = self.create_features(train_df)
            # Target: 5-day forward return (direction)
            target = train_df['Close'].shift(-HOLD_DAYS) / train_df['Close'] - 1
            
            combined = features.copy()
            combined['Target'] = target
            combined = combined.dropna()
            
            if len(combined) > HOLD_DAYS:
                combined = combined.iloc[:-HOLD_DAYS]
            
            if len(combined) > 50:
                X_list.append(combined.drop('Target', axis=1))
                y_list.append(combined['Target'])
        
        if not X_list:
            return None, None
        
        X = pd.concat(X_list)
        y = pd.concat(y_list)
        
        return X, y
    
    def train_model(self, end_date=None):
        """Train the LightGBM model."""
        if end_date is None:
            end_date = datetime.now()
        
        logger.info(f"Training model with data up to {end_date.date()}...")
        
        X, y = self.prepare_training_data(end_date)
        
        if X is None or len(X) < 1000:
            logger.error("Not enough training data")
            return False
        
        # Store feature columns
        self.feature_cols = list(X.columns)
        
        # Binary classification: predict if return > 0
        y_binary = (y > 0).astype(int)
        
        logger.info(f"Training on {len(X):,} samples, {len(self.feature_cols)} features")
        
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        self.model.fit(X, y_binary)
        
        self.last_train_date = str(end_date.date())
        self._save_model()
        self._save_state()
        
        logger.info("Model training complete")
        return True
    
    def make_predictions(self, prediction_date=None):
        """Make predictions for all stocks on a given date."""
        if prediction_date is None:
            prediction_date = datetime.now()
        
        if self.model is None:
            logger.error("No model loaded")
            return []
        
        predictions = []
        
        for ticker, df in self.stock_data.items():
            try:
                available = df[df.index <= prediction_date]
                
                if len(available) < 60:
                    continue
                
                features = self.create_features(available)
                
                if features.empty:
                    continue
                
                latest = features.iloc[-1:][self.feature_cols]
                
                if latest.isnull().any().any():
                    continue
                
                proba = self.model.predict_proba(latest)[0]
                confidence = max(proba)
                predicted_direction = 1 if proba[1] > 0.5 else 0
                
                predictions.append({
                    'Ticker': ticker,
                    'Confidence': confidence,
                    'Predicted_Up': predicted_direction,
                    'Current_Price': float(available['Close'].iloc[-1])
                })
                
            except Exception as e:
                continue
        
        return predictions
    
    def get_top_picks(self, n=TOP_N_PER_DAY):
        """Get top N stock picks for today."""
        predictions = self.make_predictions()
        
        if not predictions:
            logger.warning("No predictions generated")
            return []
        
        pred_df = pd.DataFrame(predictions)
        
        # Filter for bullish predictions only
        bullish = pred_df[pred_df['Predicted_Up'] == 1]
        
        if len(bullish) == 0:
            logger.warning("No bullish predictions")
            return []
        
        # Sort by confidence and take top N
        top_picks = bullish.nlargest(n, 'Confidence')
        
        logger.info(f"Top {len(top_picks)} picks:")
        for _, row in top_picks.iterrows():
            logger.info(f"  {row['Ticker']}: {row['Confidence']:.1%} confidence, ${row['Current_Price']:.2f}")
        
        return top_picks.to_dict('records')
    
    def get_account_info(self):
        """Get current account information."""
        account = self.trading_client.get_account()
        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
        }
    
    def get_current_positions(self):
        """Get all current positions from Alpaca."""
        positions = self.trading_client.get_all_positions()
        return {
            p.symbol: {
                'qty': float(p.qty),
                'market_value': float(p.market_value),
                'avg_entry_price': float(p.avg_entry_price),
                'unrealized_pnl': float(p.unrealized_pl),
            }
            for p in positions
        }
    
    def place_buy_order(self, ticker, dollar_amount):
        """Place a buy order for a given dollar amount."""
        try:
            # Get current price
            df = yf.download(ticker, period='1d', progress=False)
            if df.empty:
                logger.error(f"Could not get price for {ticker}")
                return None
            
            current_price = float(df['Close'].iloc[-1])
            shares = int(dollar_amount / current_price)
            
            if shares < 1:
                logger.warning(f"Not enough to buy 1 share of {ticker}")
                return None
            
            order = MarketOrderRequest(
                symbol=ticker,
                qty=shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            
            result = self.trading_client.submit_order(order)
            logger.info(f"BUY ORDER: {shares} shares of {ticker} @ ~${current_price:.2f}")
            
            return {
                'ticker': ticker,
                'shares': shares,
                'price': current_price,
                'order_id': result.id
            }
            
        except Exception as e:
            logger.error(f"Error placing buy order for {ticker}: {e}")
            return None
    
    def place_sell_order(self, ticker, shares):
        """Place a sell order for all shares of a ticker."""
        try:
            order = MarketOrderRequest(
                symbol=ticker,
                qty=shares,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            result = self.trading_client.submit_order(order)
            logger.info(f"SELL ORDER: {shares} shares of {ticker}")
            
            return {
                'ticker': ticker,
                'shares': shares,
                'order_id': result.id
            }
            
        except Exception as e:
            logger.error(f"Error placing sell order for {ticker}: {e}")
            return None
    
    def check_and_close_expired_positions(self):
        """Close positions that have reached their hold period."""
        today = datetime.now(self.et_tz).date()
        current_positions = self.get_current_positions()
        
        positions_to_close = []
        
        for ticker, info in self.active_positions.items():
            entry_date = datetime.strptime(info['entry_date'], '%Y-%m-%d').date()
            days_held = (today - entry_date).days
            
            if days_held >= HOLD_DAYS:
                positions_to_close.append(ticker)
        
        for ticker in positions_to_close:
            if ticker in current_positions:
                shares = int(current_positions[ticker]['qty'])
                if shares > 0:
                    self.place_sell_order(ticker, shares)
                    logger.info(f"Closed position in {ticker} after {HOLD_DAYS} days")
            
            del self.active_positions[ticker]
        
        self._save_state()
        return positions_to_close
    
    def execute_daily_picks(self):
        """Execute the daily stock picking routine."""
        today = datetime.now(self.et_tz).date()
        today_str = str(today)
        
        # Check if we already picked today
        if self.last_pick_date == today_str:
            logger.info("Already made picks today")
            return
        
        logger.info("=" * 60)
        logger.info(f"DAILY STOCK PICKING - {today}")
        logger.info("=" * 60)
        
        # Close expired positions first
        closed = self.check_and_close_expired_positions()
        if closed:
            logger.info(f"Closed {len(closed)} expired positions")
        
        # Get account info
        account = self.get_account_info()
        equity = account['equity']
        cash = account['cash']
        
        logger.info(f"Account: ${equity:,.2f} equity, ${cash:,.2f} cash")
        
        # Calculate position size
        position_value = equity * (POSITION_SIZE_PCT / 100)
        
        # Check if we need to retrain
        if self.model is None or self._should_retrain():
            if not self.stock_data:
                self.download_stock_data(years_back=INITIAL_TRAIN_YEARS + 1)
            self.train_model()
        
        # Get top picks
        picks = self.get_top_picks()
        
        if not picks:
            logger.warning("No picks generated today")
            self.last_pick_date = today_str
            self._save_state()
            return
        
        # Execute buys for each pick
        for pick in picks:
            ticker = pick['Ticker']
            
            # Skip if we already have a position
            if ticker in self.active_positions:
                logger.info(f"Already holding {ticker}, skipping")
                continue
            
            # Skip if not enough cash
            if cash < position_value * 0.5:
                logger.warning(f"Low cash (${cash:.2f}), skipping {ticker}")
                continue
            
            # Place buy order
            result = self.place_buy_order(ticker, position_value)
            
            if result:
                self.active_positions[ticker] = {
                    'entry_date': today_str,
                    'entry_price': result['price'],
                    'qty': result['shares'],
                    'confidence': pick['Confidence']
                }
                cash -= result['shares'] * result['price']
        
        self.last_pick_date = today_str
        self._save_state()
        
        logger.info(f"Active positions: {len(self.active_positions)}")
        for ticker, info in self.active_positions.items():
            logger.info(f"  {ticker}: {info['qty']} shares @ ${info['entry_price']:.2f}")
    
    def _should_retrain(self):
        """Check if model should be retrained."""
        if self.last_train_date is None:
            return True
        
        last_train = datetime.strptime(self.last_train_date, '%Y-%m-%d')
        days_since = (datetime.now() - last_train).days
        
        return days_since >= RETRAIN_DAYS
    
    def run_daily_check(self):
        """Run the daily check routine - call this once per day."""
        now = datetime.now(self.et_tz)
        
        # Only run during market hours (or shortly before)
        if now.weekday() >= 5:
            logger.info("Weekend - market closed")
            return
        
        # Refresh stock data periodically
        if not self.stock_data or self._should_retrain():
            self.download_stock_data()
        
        self.execute_daily_picks()
    
    def run_continuous(self, check_interval_minutes=60):
        """Run the bot continuously, checking every interval."""
        logger.info("Starting continuous mode...")
        logger.info(f"Check interval: {check_interval_minutes} minutes")
        
        while True:
            try:
                now = datetime.now(self.et_tz)
                
                # Run between 9:30 AM and 4:00 PM ET on weekdays
                if now.weekday() < 5:
                    market_open = now.replace(hour=9, minute=30, second=0)
                    market_close = now.replace(hour=16, minute=0, second=0)
                    
                    if market_open <= now <= market_close:
                        self.run_daily_check()
                    else:
                        logger.info(f"Market closed - next check in {check_interval_minutes} min")
                else:
                    logger.info("Weekend - market closed")
                
                time.sleep(check_interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Picker Trading Bot')
    parser.add_argument('--run-once', action='store_true', help='Run once and exit')
    parser.add_argument('--train', action='store_true', help='Train model only')
    parser.add_argument('--picks', action='store_true', help='Show today\'s picks without trading')
    parser.add_argument('--status', action='store_true', help='Show current status')
    args = parser.parse_args()
    
    bot = StockPickerBot()
    
    if args.train:
        bot.download_stock_data()
        bot.train_model()
    elif args.picks:
        if not bot.stock_data:
            bot.download_stock_data()
        if bot.model is None:
            bot.train_model()
        picks = bot.get_top_picks()
        print("\nToday's Top Picks:")
        for i, pick in enumerate(picks, 1):
            print(f"  {i}. {pick['Ticker']}: {pick['Confidence']:.1%} confidence")
    elif args.status:
        account = bot.get_account_info()
        print(f"\nAccount Status:")
        print(f"  Equity: ${account['equity']:,.2f}")
        print(f"  Cash: ${account['cash']:,.2f}")
        print(f"\nActive Positions: {len(bot.active_positions)}")
        for ticker, info in bot.active_positions.items():
            print(f"  {ticker}: {info['qty']} shares @ ${info['entry_price']:.2f}")
    elif args.run_once:
        bot.run_daily_check()
    else:
        bot.run_continuous()


if __name__ == '__main__':
    main()
