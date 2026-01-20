#!/usr/bin/env python3
"""
Prop Firm Paper Trading Bot for Google Cloud
=============================================

Simulates prop firm trading using your model's predictions.
Designed to test the strategy before committing real money to an evaluation.

Features:
- Prop firm rule enforcement (daily loss, trailing drawdown, profit target)
- Futures simulation (/MES) based on SPY signals
- Real-time model predictions using yfinance (FREE, near real-time)
- Dashboard-style logging
- Email alerts for important events
- Persistence across restarts

Data Sources:
- yfinance: Real-time price data (FREE, ~1-2 second delay)
- Alpaca: Not used for data (their free tier is 15-min delayed)

Usage:
    python -m live_trading.prop_firm_paper_bot --firm apex_50k
    python -m live_trading.prop_firm_paper_bot --firm the_trading_pit_10k
    python -m live_trading.prop_firm_paper_bot --simulate-only
"""

import argparse
import json
import sys
import time
import logging
import signal
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
import threading

import pytz
import pandas as pd

# Import from existing modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from live_trading.model_trainer import LiveModelTrainer
from live_trading.config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, SYMBOL,
    HIGH_CONFIDENCE_THRESHOLD, LOG_LEVEL
)
from live_trading.prop_firm_bot import (
    PROP_FIRMS, PropFirmRules, PropFirmAccount, 
    AccountStatus
)

# Import yfinance for real-time data (FREE!)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    
# Alpaca still available for execution if needed
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

# Futures contract specs (/MES)
MES_POINT_VALUE = 5.0        # $5 per point
MES_TICK_SIZE = 0.25         # Minimum price movement
MES_TICK_VALUE = 1.25        # $1.25 per tick
MES_COMMISSION = 0.62        # Round-turn commission
MES_SLIPPAGE_TICKS = 1       # Assumed slippage

# Trading parameters
# NOTE: Lowered from 0.70 to 0.50 because live model produces smaller predictions
# Backtest showed 57.2% accuracy at conf >= 0.5
MIN_CONFIDENCE = 0.50
MIN_PREDICTED_MOVE = 0.03    # In SPY terms (lowered from $0.10)
CHECK_INTERVAL_SECONDS = 60  # How often to check for signals

# Timezone
ET_TZ = pytz.timezone('US/Eastern')

# State persistence
STATE_FILE = Path(__file__).parent / "paper_trading_state.json"


# ============================================================================
# LOGGING
# ============================================================================

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "paper_trading.log")
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# STATE PERSISTENCE
# ============================================================================

@dataclass
class TradingState:
    """Persistent trading state."""
    firm_key: str
    balance: float
    peak_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    trading_days: int
    daily_starting_balance: float
    consecutive_losses: int
    status: str
    position_contracts: int
    position_direction: str
    position_entry_price: float
    position_entry_time: str
    last_update: str
    trade_history: List[Dict]
    
    def save(self, filepath: Path = STATE_FILE):
        """Save state to file."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.debug(f"State saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path = STATE_FILE) -> Optional['TradingState']:
        """Load state from file."""
        if not filepath.exists():
            return None
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None


# ============================================================================
# PROP FIRM PAPER TRADING BOT
# ============================================================================

class PropFirmPaperBot:
    """
    Paper trading bot that simulates prop firm futures trading.
    Uses real-time SPY data to generate signals, then simulates /MES execution.
    """
    
    def __init__(self, firm_key: str = 'the_trading_pit_10k', resume: bool = True):
        """
        Initialize the paper trading bot.
        
        Args:
            firm_key: Which prop firm to simulate
            resume: Whether to resume from saved state
        """
        if firm_key not in PROP_FIRMS:
            raise ValueError(f"Unknown firm: {firm_key}. Options: {list(PROP_FIRMS.keys())}")
        
        self.firm_key = firm_key
        self.firm_rules = PROP_FIRMS[firm_key]
        
        # Initialize or load account state
        self.state = None
        if resume:
            self.state = TradingState.load()
            if self.state and self.state.firm_key != firm_key:
                logger.warning(f"Saved state is for {self.state.firm_key}, not {firm_key}. Starting fresh.")
                self.state = None
        
        if self.state is None:
            self._init_fresh_state()
        else:
            logger.info(f"Resumed from saved state: Balance ${self.state.balance:,.2f}")
        
        # Initialize yfinance ticker for real-time data (FREE!)
        if YFINANCE_AVAILABLE:
            self.ticker = yf.Ticker(SYMBOL)
            logger.info(f"Using yfinance for real-time {SYMBOL} data (FREE, ~1-2s delay)")
        else:
            self.ticker = None
            logger.warning("yfinance not available - running in offline simulation mode")
        
        # Initialize Alpaca client for execution (if needed for live trading)
        if ALPACA_AVAILABLE:
            self.trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
            logger.info("Alpaca trading client ready for paper execution")
        else:
            self.trading_client = None
        
        # Initialize model
        self.model_trainer = LiveModelTrainer()
        if not self.model_trainer.load_model():
            logger.info("No saved model found, training new model...")
            self.model_trainer.run_full_training()
        
        # Runtime state
        self.running = False
        self.last_check_time = None
        self.current_date = None
        
        # Print startup info
        self._print_startup_info()
    
    def _init_fresh_state(self):
        """Initialize fresh trading state."""
        self.state = TradingState(
            firm_key=self.firm_key,
            balance=self.firm_rules.account_size,
            peak_balance=self.firm_rules.account_size,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            trading_days=0,
            daily_starting_balance=self.firm_rules.account_size,
            consecutive_losses=0,
            status=AccountStatus.EVALUATION.value,
            position_contracts=0,
            position_direction='',
            position_entry_price=0.0,
            position_entry_time='',
            last_update=datetime.now().isoformat(),
            trade_history=[]
        )
        logger.info(f"Initialized fresh state for {self.firm_rules.name}")
    
    def _print_startup_info(self):
        """Print startup information."""
        print("\n" + "="*70)
        print("     PROP FIRM PAPER TRADING BOT")
        print("="*70)
        print(f"  Data Source:       yfinance (FREE, real-time ~1-2s delay)")
        print(f"  Symbol:            {SYMBOL}")
        print("-"*70)
        print(f"  Firm:              {self.firm_rules.name}")
        print(f"  Account Size:      ${self.firm_rules.account_size:,.0f}")
        print(f"  Current Balance:   ${self.state.balance:,.2f}")
        print(f"  Profit Target:     ${self.firm_rules.profit_target:,.0f} ({self.firm_rules.profit_target_pct}%)")
        print(f"  Daily Loss Limit:  ${self.firm_rules.daily_loss_limit:,.0f} ({self.firm_rules.daily_loss_limit_pct}%)")
        print(f"  Trailing Drawdown: ${self.firm_rules.trailing_drawdown:,.0f} ({self.firm_rules.trailing_drawdown_pct}%)")
        print(f"  Max Contracts:     {self.firm_rules.max_contracts}")
        print("-"*70)
        print(f"  Status:            {self.state.status.upper()}")
        print(f"  Total Trades:      {self.state.total_trades}")
        print(f"  Trading Days:      {self.state.trading_days}")
        if self.state.total_trades > 0:
            win_rate = self.state.winning_trades / self.state.total_trades * 100
            print(f"  Win Rate:          {win_rate:.1f}%")
        print("="*70 + "\n")
    
    def get_current_session(self) -> str:
        """Determine current trading session."""
        now_et = datetime.now(ET_TZ)
        current_time = now_et.time()
        
        # Weekend
        if now_et.weekday() >= 5:
            return 'closed'
        
        # Futures trade 23 hours - only closed 5pm-6pm ET
        close_start = dt_time(17, 0)
        close_end = dt_time(18, 0)
        
        if close_start <= current_time < close_end:
            return 'closed'
        
        # Regular hours (for reference)
        if dt_time(9, 30) <= current_time < dt_time(16, 0):
            return 'regular'
        elif dt_time(4, 0) <= current_time < dt_time(9, 30):
            return 'premarket'
        elif dt_time(16, 0) <= current_time < dt_time(17, 0) or \
             dt_time(18, 0) <= current_time < dt_time(20, 0):
            return 'afterhours'
        else:
            return 'overnight'  # Futures trade overnight too
    
    def get_current_spy_price(self) -> Optional[float]:
        """Get current SPY price from yfinance (real-time, FREE!)."""
        if not self.ticker:
            return None
        
        try:
            # Get real-time quote from yfinance
            info = self.ticker.fast_info
            price = info.get('lastPrice') or info.get('last_price')
            if price:
                return float(price)
            
            # Fallback: get latest from history
            hist = self.ticker.history(period='1d', interval='1m')
            if len(hist) > 0:
                return float(hist['Close'].iloc[-1])
            
            return None
        except Exception as e:
            logger.error(f"Error fetching SPY price from yfinance: {e}")
            return None
    
    def spy_to_futures_price(self, spy_price: float) -> float:
        """Convert SPY price to S&P 500 futures points."""
        return spy_price * 10  # SPY ≈ S&P 500 / 10
    
    def get_prediction(self) -> Optional[Dict]:
        """Get model prediction for current market conditions using yfinance."""
        if not self.ticker:
            # Offline mode - return None
            return None
        
        try:
            # Get recent minute bars from yfinance (FREE real-time data!)
            # yfinance provides data with ~1-2 second delay
            df = self.ticker.history(period='1d', interval='1m')
            
            if df is None or len(df) == 0:
                logger.warning("No data from yfinance")
                return None
            
            # Reset index and rename columns
            df = df.reset_index()
            df = df.rename(columns={
                'Datetime': 'Datetime',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            # Ensure we have the Datetime column
            if 'Datetime' not in df.columns and 'index' in df.columns:
                df = df.rename(columns={'index': 'Datetime'})
            
            # Select and order columns
            df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df = df.tail(100).reset_index(drop=True)
            
            if len(df) < 60:
                logger.warning(f"Insufficient data for prediction: {len(df)} bars")
                return None
            
            # Add indicators
            df = self.model_trainer.add_technical_indicators(df)
            df = df.dropna()
            
            if len(df) == 0:
                logger.warning("No valid data after adding indicators")
                return None
            
            latest = df.iloc[-1]
            
            # Get prediction
            predicted_change, confidence = self.model_trainer.predict(latest)
            
            return {
                'predicted_change': predicted_change,
                'confidence': confidence,
                'direction': 'UP' if predicted_change > 0 else 'DOWN',
                'spy_price': float(latest['Close']),
                'futures_price': self.spy_to_futures_price(latest['Close']),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def can_trade(self) -> Tuple[bool, str]:
        """Check if we can place a new trade based on prop firm rules."""
        
        # Check status
        if self.state.status == AccountStatus.BREACHED.value:
            return False, "Account breached - evaluation failed"
        
        if self.state.status == AccountStatus.PASSED.value:
            return False, "Profit target reached - evaluation passed!"
        
        # Check daily loss
        daily_pnl = self.state.balance - self.state.daily_starting_balance
        if daily_pnl <= -self.firm_rules.daily_loss_limit:
            return False, f"Daily loss limit hit (${self.firm_rules.daily_loss_limit:,.0f})"
        
        # Check trailing drawdown
        trailing_dd = self.state.peak_balance - self.state.balance
        if trailing_dd >= self.firm_rules.trailing_drawdown:
            self.state.status = AccountStatus.BREACHED.value
            self.state.save()
            return False, f"Trailing drawdown breached (${self.firm_rules.trailing_drawdown:,.0f})"
        
        # Check consecutive losses
        if self.state.consecutive_losses >= 3:
            return False, "3 consecutive losses - take a break"
        
        # Check if too close to limits
        daily_remaining = self.firm_rules.daily_loss_limit - abs(min(0, daily_pnl))
        trailing_remaining = self.firm_rules.trailing_drawdown - trailing_dd
        
        if daily_remaining < self.firm_rules.daily_loss_limit * 0.3:
            return False, "Too close to daily limit - stopping for today"
        
        if trailing_remaining < self.firm_rules.trailing_drawdown * 0.3:
            return False, "Too close to trailing drawdown - reducing risk"
        
        return True, "OK"
    
    def calculate_position_size(self) -> int:
        """Calculate safe position size based on current risk levels."""
        # Calculate remaining risk room
        daily_pnl = self.state.balance - self.state.daily_starting_balance
        daily_remaining = self.firm_rules.daily_loss_limit - abs(min(0, daily_pnl))
        trailing_remaining = self.firm_rules.trailing_drawdown - (self.state.peak_balance - self.state.balance)
        
        # Use 25% of remaining room for next trade
        max_risk = min(daily_remaining, trailing_remaining) * 0.25
        
        # Assume 5-point stop loss
        stop_distance = 5.0
        risk_per_contract = stop_distance * MES_POINT_VALUE
        
        contracts = int(max_risk / risk_per_contract)
        contracts = min(contracts, self.firm_rules.max_contracts)
        contracts = max(1, contracts)  # At least 1
        
        return contracts
    
    def open_position(self, direction: str, contracts: int, futures_price: float):
        """Open a simulated futures position."""
        self.state.position_contracts = contracts
        self.state.position_direction = direction
        self.state.position_entry_price = futures_price
        self.state.position_entry_time = datetime.now().isoformat()
        
        # Apply entry slippage
        slippage_cost = MES_SLIPPAGE_TICKS * MES_TICK_VALUE * contracts
        self.state.balance -= slippage_cost
        
        logger.info(f"OPENED {direction} {contracts} contracts @ {futures_price:.2f}")
        self.state.save()
    
    def close_position(self, futures_price: float, reason: str = "signal") -> float:
        """Close current position and return P&L."""
        if self.state.position_contracts == 0:
            return 0.0
        
        contracts = self.state.position_contracts
        entry_price = self.state.position_entry_price
        direction = self.state.position_direction
        
        # Calculate P&L
        if direction == 'LONG':
            points = futures_price - entry_price
        else:
            points = entry_price - futures_price
        
        gross_pnl = points * MES_POINT_VALUE * contracts
        
        # Subtract costs
        commission = MES_COMMISSION * contracts
        slippage = MES_SLIPPAGE_TICKS * MES_TICK_VALUE * contracts
        net_pnl = gross_pnl - commission - slippage
        
        # Update balance
        self.state.balance += net_pnl
        self.state.total_trades += 1
        
        is_win = net_pnl > 0
        if is_win:
            self.state.winning_trades += 1
            self.state.consecutive_losses = 0
        else:
            self.state.losing_trades += 1
            self.state.consecutive_losses += 1
        
        # Update peak
        if self.state.balance > self.state.peak_balance:
            self.state.peak_balance = self.state.balance
        
        # Log trade
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'direction': direction,
            'contracts': contracts,
            'entry_price': entry_price,
            'exit_price': futures_price,
            'pnl': net_pnl,
            'reason': reason
        }
        self.state.trade_history.append(trade_record)
        
        result_str = "WIN" if is_win else "LOSS"
        logger.info(f"CLOSED {direction} @ {futures_price:.2f} | {result_str}: ${net_pnl:+,.2f}")
        
        # Check profit target
        total_pnl = self.state.balance - self.firm_rules.account_size
        if total_pnl >= self.firm_rules.profit_target:
            self.state.status = AccountStatus.PASSED.value
            logger.info("🎉 PROFIT TARGET REACHED! EVALUATION PASSED!")
        
        # Check if breached
        if self.state.peak_balance - self.state.balance >= self.firm_rules.trailing_drawdown:
            self.state.status = AccountStatus.BREACHED.value
            logger.warning("❌ TRAILING DRAWDOWN BREACHED - EVALUATION FAILED")
        
        # Reset position
        self.state.position_contracts = 0
        self.state.position_direction = ''
        self.state.position_entry_price = 0.0
        self.state.position_entry_time = ''
        
        self.state.save()
        return net_pnl
    
    def process_signal(self, prediction: Dict) -> Optional[float]:
        """Process a trading signal and execute if appropriate."""
        
        # Check signal quality
        if prediction['confidence'] < MIN_CONFIDENCE:
            return None
        
        if abs(prediction['predicted_change']) < MIN_PREDICTED_MOVE:
            return None
        
        # Check if we can trade
        can_trade, reason = self.can_trade()
        if not can_trade:
            logger.info(f"Cannot trade: {reason}")
            # Close position if we have one and hit limits
            if self.state.position_contracts > 0 and "limit" in reason.lower():
                return self.close_position(prediction['futures_price'], "risk_limit")
            return None
        
        desired_direction = 'LONG' if prediction['direction'] == 'UP' else 'SHORT'
        
        # If we have a position
        if self.state.position_contracts > 0:
            current_direction = self.state.position_direction
            if current_direction != desired_direction:
                # Close and reverse
                pnl = self.close_position(prediction['futures_price'], "signal_reverse")
                # Open new position
                contracts = self.calculate_position_size()
                self.open_position(desired_direction, contracts, prediction['futures_price'])
                return pnl
        else:
            # Open new position
            contracts = self.calculate_position_size()
            self.open_position(desired_direction, contracts, prediction['futures_price'])
        
        return None
    
    def check_new_day(self):
        """Check if it's a new trading day and reset daily limits."""
        now_et = datetime.now(ET_TZ)
        today = now_et.date()
        
        if self.current_date != today:
            if self.current_date is not None:
                # End of previous day
                self.state.trading_days += 1
                logger.info(f"New trading day #{self.state.trading_days}")
            
            self.current_date = today
            self.state.daily_starting_balance = self.state.balance
            self.state.consecutive_losses = 0  # Reset at start of day
            self.state.save()
    
    def print_status(self):
        """Print current status dashboard."""
        total_pnl = self.state.balance - self.firm_rules.account_size
        daily_pnl = self.state.balance - self.state.daily_starting_balance
        trailing_dd = self.state.peak_balance - self.state.balance
        
        print("\n" + "-"*60)
        print(f"  Balance: ${self.state.balance:,.2f} | P&L: ${total_pnl:+,.2f} ({total_pnl/self.firm_rules.account_size*100:+.2f}%)")
        print(f"  Daily P&L: ${daily_pnl:+,.2f} | Trailing DD: ${trailing_dd:,.2f}")
        print(f"  Progress: {total_pnl/self.firm_rules.profit_target*100:.1f}% to target")
        if self.state.position_contracts > 0:
            print(f"  Position: {self.state.position_direction} {self.state.position_contracts} @ {self.state.position_entry_price:.2f}")
        print("-"*60)
    
    def run_once(self):
        """Run one iteration of the trading loop."""
        session = self.get_current_session()
        
        if session == 'closed':
            logger.debug("Market closed")
            return False
        
        # Check for new day
        self.check_new_day()
        
        # Check if evaluation is complete
        if self.state.status in [AccountStatus.PASSED.value, AccountStatus.BREACHED.value]:
            logger.info(f"Evaluation complete: {self.state.status}")
            return False
        
        logger.info(f"Session: {session.upper()}")
        
        # Get prediction
        prediction = self.get_prediction()
        
        if prediction is None:
            logger.debug("No prediction available")
            return True
        
        logger.info(
            f"Prediction: {prediction['direction']} "
            f"(change: ${prediction['predicted_change']:.4f}, "
            f"conf: {prediction['confidence']:.2f}, "
            f"futures: {prediction['futures_price']:.2f})"
        )
        
        # Process signal
        if prediction['confidence'] >= MIN_CONFIDENCE:
            pnl = self.process_signal(prediction)
            if pnl is not None:
                logger.info(f"Trade closed: ${pnl:+,.2f}")
        else:
            logger.debug(f"Confidence below threshold ({MIN_CONFIDENCE})")
        
        # Print status
        self.print_status()
        
        return True
    
    def run(self, interval_seconds: int = 60):
        """Main trading loop."""
        self.running = True
        
        logger.info("="*60)
        logger.info("STARTING PROP FIRM PAPER TRADING BOT")
        logger.info(f"Firm: {self.firm_rules.name}")
        logger.info(f"Check interval: {interval_seconds}s")
        logger.info("="*60)
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info("\nShutdown signal received...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while self.running:
                try:
                    active = self.run_once()
                    
                    if not active:
                        # Check if evaluation is done
                        if self.state.status in [AccountStatus.PASSED.value, AccountStatus.BREACHED.value]:
                            self._print_final_report()
                            break
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    time.sleep(30)
        
        finally:
            self.shutdown()
    
    def _print_final_report(self):
        """Print final evaluation report."""
        total_pnl = self.state.balance - self.firm_rules.account_size
        
        print("\n" + "="*70)
        print("     PROP FIRM EVALUATION COMPLETE")
        print("="*70)
        
        if self.state.status == AccountStatus.PASSED.value:
            print(f"""
    ✅ EVALUATION PASSED!
    
    Firm:               {self.firm_rules.name}
    Final Balance:      ${self.state.balance:,.2f}
    Total P&L:          ${total_pnl:+,.2f}
    Your Share (90%):   ${total_pnl * self.firm_rules.profit_split:+,.2f}
    
    Total Trades:       {self.state.total_trades}
    Win Rate:           {self.state.winning_trades/self.state.total_trades*100:.1f}%
    Trading Days:       {self.state.trading_days}
    
    NEXT STEP: Sign up for real evaluation at the prop firm!
    """)
        else:
            print(f"""
    ❌ EVALUATION FAILED
    
    Firm:               {self.firm_rules.name}
    Final Balance:      ${self.state.balance:,.2f}
    Total P&L:          ${total_pnl:+,.2f}
    
    Reason:             Drawdown limit breached
    
    Total Trades:       {self.state.total_trades}
    Trading Days:       {self.state.trading_days}
    
    NEXT STEPS:
    1. Review trade history
    2. Reduce position size
    3. Increase confidence threshold
    4. Try again with --reset flag
    """)
        
        print("="*70 + "\n")
    
    def shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down...")
        
        # Close any open position
        if self.state.position_contracts > 0:
            current_price = self.get_current_spy_price()
            if current_price:
                futures_price = self.spy_to_futures_price(current_price)
                self.close_position(futures_price, "shutdown")
        
        # Save state
        self.state.last_update = datetime.now().isoformat()
        self.state.save()
        
        logger.info("State saved. Shutdown complete.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Prop Firm Paper Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Firms available:
  apex_50k           - Apex Trader Funding $50K
  apex_100k          - Apex Trader Funding $100K  
  topstep_50k        - Topstep $50K
  bulenox_50k        - Bulenox $50K
  the_trading_pit_10k - The Trading Pit $10K (recommended for beginners)

Examples:
  python -m live_trading.prop_firm_paper_bot --firm the_trading_pit_10k
  python -m live_trading.prop_firm_paper_bot --firm apex_50k --interval 30
  python -m live_trading.prop_firm_paper_bot --reset  # Start fresh
        """
    )
    
    parser.add_argument(
        '--firm',
        type=str,
        default='the_trading_pit_10k',
        help='Prop firm to simulate (default: the_trading_pit_10k)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Seconds between checks (default: 60)'
    )
    
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset state and start fresh evaluation'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Just print current status and exit'
    )
    
    args = parser.parse_args()
    
    # Handle reset
    if args.reset:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
            print("State reset. Starting fresh evaluation.")
    
    # Handle status
    if args.status:
        state = TradingState.load()
        if state:
            print(f"\nCurrent State:")
            print(f"  Firm: {state.firm_key}")
            print(f"  Balance: ${state.balance:,.2f}")
            print(f"  Status: {state.status}")
            print(f"  Trades: {state.total_trades}")
            print(f"  Trading Days: {state.trading_days}")
        else:
            print("No saved state found.")
        return
    
    # Run bot
    try:
        bot = PropFirmPaperBot(firm_key=args.firm, resume=not args.reset)
        bot.run(interval_seconds=args.interval)
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
