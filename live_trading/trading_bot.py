"""
Live Trading Bot for Alpaca
============================
Executes trades based on model predictions across all 3 sessions:
- Pre-market: 4:00 AM - 9:30 AM ET
- Regular hours: 9:30 AM - 4:00 PM ET  
- After-hours: 4:00 PM - 8:00 PM ET

PAPER TRADING BY DEFAULT - Set PAPER_TRADING=False in config.py for live trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging
import json

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame

import pytz

from .config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER_TRADING, SYMBOL,
    MAX_POSITION_SIZE, MIN_TRADE_VALUE,
    MAX_DAILY_LOSS, MAX_DRAWDOWN,
    HIGH_CONFIDENCE_THRESHOLD, MIN_CONFIDENCE_THRESHOLD,
    SESSION_MULTIPLIERS, PREDICTION_HORIZON_MINUTES,
    FEATURE_COLUMNS, LOG_LEVEL, SAVE_TRADE_HISTORY
)
from .model_trainer import LiveModelTrainer


# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class AlpacaTradingBot:
    """
    Live trading bot that uses the trained model to execute trades on Alpaca.
    Supports pre-market, regular hours, and after-hours trading.
    """
    
    def __init__(self):
        """Initialize the trading bot."""
        # Trading client
        self.trading_client = TradingClient(
            ALPACA_API_KEY, 
            ALPACA_SECRET_KEY, 
            paper=PAPER_TRADING
        )
        
        # Data client
        self.data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        
        # Model trainer (for predictions)
        self.model_trainer = LiveModelTrainer()
        
        # Load or train model
        if not self.model_trainer.load_model():
            logger.info("No saved model found, training new model...")
            self.model_trainer.run_full_training()
        
        # Trading state
        self.daily_pnl = 0.0
        self.peak_equity = None
        self.current_position = None
        self.trade_history = []
        self.last_prediction_time = None
        
        # Recent data buffer for indicators
        self.data_buffer = pd.DataFrame()
        self.buffer_size = 100  # Keep last 100 bars for indicator calculation
        
        # Timezone
        self.et_tz = pytz.timezone('US/Eastern')
        
        # Trade history file
        self.history_dir = Path(__file__).parent / "trade_history"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trading bot initialized (Paper: {PAPER_TRADING})")
    
    def get_current_session(self):
        """
        Determine current trading session.
        
        Returns:
            'premarket', 'regular', 'afterhours', or 'closed'
        """
        now_et = datetime.now(self.et_tz)
        current_time = now_et.time()
        
        # Check if weekend
        if now_et.weekday() >= 5:
            return 'closed'
        
        from datetime import time as dt_time
        
        premarket_start = dt_time(4, 0)
        regular_start = dt_time(9, 30)
        regular_end = dt_time(16, 0)
        afterhours_end = dt_time(20, 0)
        
        if premarket_start <= current_time < regular_start:
            return 'premarket'
        elif regular_start <= current_time < regular_end:
            return 'regular'
        elif regular_end <= current_time < afterhours_end:
            return 'afterhours'
        else:
            return 'closed'
    
    def get_account_info(self):
        """Get current account information."""
        account = self.trading_client.get_account()
        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
        }
    
    def get_current_position(self):
        """Get current position in the symbol."""
        try:
            position = self.trading_client.get_open_position(SYMBOL)
            return {
                'qty': float(position.qty),
                'market_value': float(position.market_value),
                'avg_entry_price': float(position.avg_entry_price),
                'unrealized_pnl': float(position.unrealized_pl),
                'side': 'long' if float(position.qty) > 0 else 'short'
            }
        except Exception:
            return None
    
    def get_latest_bar(self):
        """Get the most recent minute bar."""
        try:
            request = StockLatestBarRequest(symbol_or_symbols=SYMBOL)
            bars = self.data_client.get_stock_latest_bar(request)
            bar = bars[SYMBOL]
            return {
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            }
        except Exception as e:
            logger.error(f"Error fetching latest bar: {e}")
            return None
    
    def get_recent_bars(self, n_bars=100):
        """Get recent minute bars for indicator calculation."""
        try:
            end = datetime.now()
            start = end - timedelta(minutes=n_bars * 2)  # Fetch extra in case of gaps
            
            request = StockBarsRequest(
                symbol_or_symbols=SYMBOL,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end
            )
            
            bars = self.data_client.get_stock_bars(request)
            df = bars.df.reset_index()
            
            df = df.rename(columns={
                'timestamp': 'Datetime',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            if 'symbol' in df.columns:
                df = df[df['symbol'] == SYMBOL]
            
            df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df = df.sort_values('Datetime').tail(n_bars).reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching recent bars: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators on recent data."""
        return self.model_trainer.add_technical_indicators(df)
    
    def get_prediction(self):
        """
        Get model prediction for current market conditions.
        
        Returns:
            (predicted_change, confidence, direction) or None if error
        """
        # Get recent bars
        df = self.get_recent_bars(n_bars=100)
        if df is None or len(df) < 60:
            logger.warning("Insufficient data for prediction")
            return None
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Get latest row with all indicators
        df = df.dropna(subset=FEATURE_COLUMNS)
        if len(df) == 0:
            logger.warning("No valid data after indicator calculation")
            return None
        
        latest = df.iloc[-1]
        
        # Make prediction
        try:
            predicted_change, confidence = self.model_trainer.predict(latest)
            direction = 'UP' if predicted_change > 0 else 'DOWN'
            
            return {
                'predicted_change': predicted_change,
                'confidence': confidence,
                'direction': direction,
                'current_price': latest['Close'],
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def calculate_position_size(self, confidence, session):
        """
        Calculate position size based on confidence and session.
        
        Args:
            confidence: Model confidence (0-1)
            session: Current trading session
            
        Returns:
            Position size as fraction of portfolio
        """
        # Base size from confidence
        if confidence >= 0.9:
            base_size = 0.10  # 10% for very high confidence
        elif confidence >= 0.7:
            base_size = 0.07  # 7% for high confidence
        elif confidence >= 0.5:
            base_size = 0.05  # 5% for medium confidence
        else:
            base_size = 0.02  # 2% for low confidence
        
        # Apply session multiplier
        session_mult = SESSION_MULTIPLIERS.get(session, 0.5)
        adjusted_size = base_size * session_mult
        
        # Cap at max position size
        return min(adjusted_size, MAX_POSITION_SIZE)
    
    def check_risk_limits(self):
        """
        Check if we've hit any risk limits.
        
        Returns:
            True if safe to trade, False if limits hit
        """
        account = self.get_account_info()
        equity = account['equity']
        
        # Initialize peak equity
        if self.peak_equity is None:
            self.peak_equity = equity
        
        # Update peak
        self.peak_equity = max(self.peak_equity, equity)
        
        # Check daily loss limit
        if self.daily_pnl < -MAX_DAILY_LOSS * self.peak_equity:
            logger.warning(f"Daily loss limit hit: {self.daily_pnl:.2f}")
            return False
        
        # Check max drawdown
        drawdown = (self.peak_equity - equity) / self.peak_equity
        if drawdown > MAX_DRAWDOWN:
            logger.warning(f"Max drawdown hit: {drawdown:.2%}")
            return False
        
        return True
    
    def place_order(self, side, qty, extended_hours=True):
        """
        Place an order on Alpaca.
        
        Args:
            side: 'buy' or 'sell'
            qty: Number of shares
            extended_hours: Allow extended hours trading
            
        Returns:
            Order object or None if failed
        """
        try:
            order_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL
            
            # Use limit order for extended hours (required by most brokers)
            session = self.get_current_session()
            
            if session in ['premarket', 'afterhours']:
                # Get current price for limit order
                latest = self.get_latest_bar()
                if latest is None:
                    return None
                
                # Set limit price slightly better than current
                if side == 'buy':
                    limit_price = round(latest['close'] * 1.001, 2)  # 0.1% above
                else:
                    limit_price = round(latest['close'] * 0.999, 2)  # 0.1% below
                
                order_request = LimitOrderRequest(
                    symbol=SYMBOL,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                    extended_hours=True
                )
            else:
                # Market order for regular hours
                order_request = MarketOrderRequest(
                    symbol=SYMBOL,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY
                )
            
            order = self.trading_client.submit_order(order_request)
            logger.info(f"Order placed: {side.upper()} {qty} {SYMBOL} @ {order.type}")
            
            return order
            
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return None
    
    def execute_trade(self, prediction, session):
        """
        Execute a trade based on prediction.
        
        Args:
            prediction: Prediction dict from get_prediction()
            session: Current trading session
        """
        # Check confidence threshold
        if prediction['confidence'] < MIN_CONFIDENCE_THRESHOLD:
            logger.debug(f"Confidence too low: {prediction['confidence']:.2f}")
            return
        
        # Check risk limits
        if not self.check_risk_limits():
            logger.warning("Risk limits exceeded, skipping trade")
            return
        
        # Get account and position info
        account = self.get_account_info()
        position = self.get_current_position()
        
        # Calculate position size
        size_pct = self.calculate_position_size(prediction['confidence'], session)
        trade_value = account['equity'] * size_pct
        
        if trade_value < MIN_TRADE_VALUE:
            logger.debug(f"Trade value too small: ${trade_value:.2f}")
            return
        
        # Calculate shares
        current_price = prediction['current_price']
        shares = int(trade_value / current_price)
        
        if shares < 1:
            return
        
        # Determine action
        if prediction['direction'] == 'UP':
            # Want to be long
            if position is None:
                # Open long position
                order = self.place_order('buy', shares)
                if order:
                    self.log_trade('BUY', shares, current_price, prediction)
            elif position['side'] == 'short':
                # Close short and go long
                close_shares = int(abs(position['qty']))
                order = self.place_order('buy', close_shares + shares)
                if order:
                    self.log_trade('COVER_AND_BUY', close_shares + shares, current_price, prediction)
        
        else:  # DOWN prediction
            # Want to be short or flat
            if position is not None and position['side'] == 'long':
                # Close long position
                close_shares = int(position['qty'])
                order = self.place_order('sell', close_shares)
                if order:
                    self.log_trade('SELL', close_shares, current_price, prediction)
    
    def log_trade(self, action, shares, price, prediction):
        """Log trade to history."""
        trade = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'symbol': SYMBOL,
            'shares': shares,
            'price': price,
            'predicted_change': prediction['predicted_change'],
            'confidence': prediction['confidence'],
            'direction': prediction['direction'],
            'session': self.get_current_session()
        }
        
        self.trade_history.append(trade)
        logger.info(f"TRADE: {action} {shares} {SYMBOL} @ ${price:.2f} (conf: {prediction['confidence']:.2f})")
        
        if SAVE_TRADE_HISTORY:
            self.save_trade_history()
    
    def save_trade_history(self):
        """Save trade history to file."""
        today = datetime.now().strftime('%Y-%m-%d')
        history_file = self.history_dir / f"trades_{today}.json"
        
        with open(history_file, 'w') as f:
            json.dump(self.trade_history, f, indent=2)
    
    def run_once(self):
        """
        Run one iteration of the trading loop.
        Checks market conditions and executes trade if appropriate.
        """
        # Check session
        session = self.get_current_session()
        
        if session == 'closed':
            logger.debug("Market closed")
            return False
        
        logger.info(f"Session: {session.upper()}")
        
        # Get prediction
        prediction = self.get_prediction()
        
        if prediction is None:
            logger.warning("Could not get prediction")
            return False
        
        logger.info(
            f"Prediction: {prediction['direction']} "
            f"(change: ${prediction['predicted_change']:.4f}, "
            f"conf: {prediction['confidence']:.2f})"
        )
        
        # Execute trade if high confidence
        if prediction['confidence'] >= HIGH_CONFIDENCE_THRESHOLD:
            self.execute_trade(prediction, session)
        else:
            logger.info(f"Confidence below threshold ({HIGH_CONFIDENCE_THRESHOLD}), holding")
        
        return True
    
    def run(self, interval_seconds=60):
        """
        Main trading loop.
        
        Args:
            interval_seconds: Seconds between prediction checks
        """
        logger.info("="*60)
        logger.info("STARTING LIVE TRADING BOT")
        logger.info(f"Symbol: {SYMBOL}")
        logger.info(f"Paper Trading: {PAPER_TRADING}")
        logger.info(f"Prediction Interval: {interval_seconds}s")
        logger.info("="*60)
        
        # Print account info
        account = self.get_account_info()
        logger.info(f"Account Equity: ${account['equity']:,.2f}")
        logger.info(f"Buying Power: ${account['buying_power']:,.2f}")
        
        try:
            while True:
                session = self.get_current_session()
                
                if session != 'closed':
                    self.run_once()
                else:
                    logger.debug("Market closed, waiting...")
                
                # Wait for next interval
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("\nBot stopped by user")
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down...")
        
        # Save trade history
        if self.trade_history:
            self.save_trade_history()
            logger.info(f"Saved {len(self.trade_history)} trades to history")
        
        # Print summary
        position = self.get_current_position()
        if position:
            logger.info(f"Final position: {position['qty']} shares @ ${position['avg_entry_price']:.2f}")
            logger.info(f"Unrealized P&L: ${position['unrealized_pnl']:.2f}")
        
        logger.info("Shutdown complete")


def main():
    """Run the trading bot."""
    bot = AlpacaTradingBot()
    
    # Run with 1-minute intervals
    bot.run(interval_seconds=60)


if __name__ == "__main__":
    main()
