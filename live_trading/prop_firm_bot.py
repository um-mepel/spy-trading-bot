#!/usr/bin/env python3
"""
Prop Firm Trading Bot
=====================

Trade with a prop firm's capital using your model's signals.
Designed to pass evaluations and maintain funded accounts.

Supported Prop Firms:
- Apex Trader Funding
- Topstep
- The Trading Pit
- Bulenox
- Trade The Pool

Key Rules (common across most prop firms):
1. Daily Loss Limit: Don't lose more than X% in a single day
2. Trailing Drawdown: Account can't drop below peak - X%
3. Profit Target: Hit X% profit to pass evaluation
4. No revenge trading: Stop after consecutive losses
5. Consistency: Don't make all profit in one day
"""

import os
import json
from datetime import datetime, time, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PROP FIRM CONFIGURATIONS
# ============================================================================

@dataclass
class PropFirmRules:
    """Rules and limits for a prop firm account."""
    name: str
    account_size: float
    profit_target_pct: float          # % profit needed to pass
    daily_loss_limit_pct: float       # Max % loss per day
    trailing_drawdown_pct: float      # Max drawdown from peak
    max_contracts: int                # Max position size
    min_trading_days: int             # Minimum days to trade
    profit_split: float               # Your share (0.0 - 1.0)
    overnight_allowed: bool           # Can hold overnight?
    news_trading_allowed: bool        # Can trade during news?
    scaling_plan: bool                # Does account size grow?
    evaluation_cost: float            # Monthly/one-time cost
    
    # Calculated values
    @property
    def profit_target(self) -> float:
        return self.account_size * (self.profit_target_pct / 100)
    
    @property
    def daily_loss_limit(self) -> float:
        return self.account_size * (self.daily_loss_limit_pct / 100)
    
    @property
    def trailing_drawdown(self) -> float:
        return self.account_size * (self.trailing_drawdown_pct / 100)


# Pre-configured prop firm rules
PROP_FIRMS = {
    'apex_50k': PropFirmRules(
        name="Apex Trader Funding - $50K",
        account_size=50_000,
        profit_target_pct=6.0,        # $3,000 profit target
        daily_loss_limit_pct=2.5,     # $1,250 daily max loss
        trailing_drawdown_pct=5.0,    # $2,500 trailing drawdown
        max_contracts=10,
        min_trading_days=7,
        profit_split=0.90,            # 90% to you
        overnight_allowed=True,
        news_trading_allowed=True,
        scaling_plan=True,
        evaluation_cost=147.0
    ),
    'apex_100k': PropFirmRules(
        name="Apex Trader Funding - $100K",
        account_size=100_000,
        profit_target_pct=6.0,        # $6,000 profit target
        daily_loss_limit_pct=2.5,     # $2,500 daily max loss
        trailing_drawdown_pct=5.0,    # $5,000 trailing drawdown
        max_contracts=14,
        min_trading_days=7,
        profit_split=0.90,
        overnight_allowed=True,
        news_trading_allowed=True,
        scaling_plan=True,
        evaluation_cost=207.0
    ),
    'topstep_50k': PropFirmRules(
        name="Topstep - $50K",
        account_size=50_000,
        profit_target_pct=6.0,        # $3,000
        daily_loss_limit_pct=2.0,     # $1,000
        trailing_drawdown_pct=4.0,    # $2,000
        max_contracts=5,
        min_trading_days=5,
        profit_split=0.90,
        overnight_allowed=False,      # Must be flat EOD
        news_trading_allowed=False,
        scaling_plan=True,
        evaluation_cost=165.0
    ),
    'bulenox_50k': PropFirmRules(
        name="Bulenox - $50K",
        account_size=50_000,
        profit_target_pct=6.0,
        daily_loss_limit_pct=2.5,
        trailing_drawdown_pct=5.0,
        max_contracts=10,
        min_trading_days=0,           # No minimum
        profit_split=0.90,
        overnight_allowed=True,
        news_trading_allowed=True,
        scaling_plan=True,
        evaluation_cost=115.0
    ),
    'the_trading_pit_10k': PropFirmRules(
        name="The Trading Pit - $10K",
        account_size=10_000,
        profit_target_pct=10.0,       # $1,000
        daily_loss_limit_pct=5.0,     # $500
        trailing_drawdown_pct=10.0,   # $1,000
        max_contracts=2,
        min_trading_days=3,
        profit_split=0.70,
        overnight_allowed=True,
        news_trading_allowed=True,
        scaling_plan=True,
        evaluation_cost=99.0
    ),
}


# ============================================================================
# ACCOUNT STATE TRACKING
# ============================================================================

class AccountStatus(Enum):
    EVALUATION = "evaluation"      # In evaluation phase
    FUNDED = "funded"              # Passed, trading real money
    BREACHED = "breached"          # Failed - hit drawdown limit
    PASSED = "passed"              # Hit profit target
    INACTIVE = "inactive"          # Not trading


@dataclass
class DailyStats:
    """Track daily trading statistics."""
    date: str
    starting_balance: float
    current_balance: float
    high_balance: float
    low_balance: float
    trades: int
    wins: int
    losses: int
    pnl: float
    
    @property
    def win_rate(self) -> float:
        return (self.wins / self.trades * 100) if self.trades > 0 else 0.0
    
    @property
    def drawdown(self) -> float:
        return self.high_balance - self.current_balance


@dataclass 
class PropFirmAccount:
    """Track prop firm account state."""
    firm_rules: PropFirmRules
    status: AccountStatus = AccountStatus.EVALUATION
    starting_balance: float = 0.0
    current_balance: float = 0.0
    peak_balance: float = 0.0
    daily_starting_balance: float = 0.0
    
    # Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    trading_days: int = 0
    consecutive_losses: int = 0
    
    # Daily tracking
    daily_stats: List[DailyStats] = field(default_factory=list)
    
    def __post_init__(self):
        self.starting_balance = self.firm_rules.account_size
        self.current_balance = self.firm_rules.account_size
        self.peak_balance = self.firm_rules.account_size
        self.daily_starting_balance = self.firm_rules.account_size
    
    @property
    def total_pnl(self) -> float:
        return self.current_balance - self.starting_balance
    
    @property
    def total_pnl_pct(self) -> float:
        return (self.total_pnl / self.starting_balance) * 100
    
    @property
    def daily_pnl(self) -> float:
        return self.current_balance - self.daily_starting_balance
    
    @property
    def daily_pnl_pct(self) -> float:
        return (self.daily_pnl / self.daily_starting_balance) * 100
    
    @property
    def trailing_drawdown_current(self) -> float:
        return self.peak_balance - self.current_balance
    
    @property
    def daily_loss_remaining(self) -> float:
        """How much more can we lose today before hitting daily limit."""
        return self.firm_rules.daily_loss_limit - abs(min(0, self.daily_pnl))
    
    @property
    def trailing_drawdown_remaining(self) -> float:
        """How much more can we lose before hitting trailing drawdown."""
        return self.firm_rules.trailing_drawdown - self.trailing_drawdown_current
    
    def can_trade(self) -> Tuple[bool, str]:
        """Check if we can place a new trade."""
        
        # Check account status
        if self.status == AccountStatus.BREACHED:
            return False, "Account breached - trading disabled"
        
        if self.status == AccountStatus.PASSED:
            return False, "Profit target reached - evaluation complete!"
        
        # Check daily loss limit
        if self.daily_pnl <= -self.firm_rules.daily_loss_limit:
            return False, f"Daily loss limit reached (${self.firm_rules.daily_loss_limit:,.2f})"
        
        # Check trailing drawdown
        if self.trailing_drawdown_current >= self.firm_rules.trailing_drawdown:
            self.status = AccountStatus.BREACHED
            return False, f"Trailing drawdown breached (${self.firm_rules.trailing_drawdown:,.2f})"
        
        # Check consecutive losses (self-imposed risk management)
        if self.consecutive_losses >= 3:
            return False, "3 consecutive losses - take a break"
        
        return True, "OK"
    
    def update_balance(self, pnl: float, is_win: bool):
        """Update account after a trade."""
        self.current_balance += pnl
        self.total_trades += 1
        
        if is_win:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
        
        # Update peak (for trailing drawdown)
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        # Check if profit target reached
        if self.total_pnl >= self.firm_rules.profit_target:
            self.status = AccountStatus.PASSED
            logger.info(f"🎉 PROFIT TARGET REACHED! +${self.total_pnl:,.2f}")
        
        # Check if breached
        if self.trailing_drawdown_current >= self.firm_rules.trailing_drawdown:
            self.status = AccountStatus.BREACHED
            logger.warning(f"❌ ACCOUNT BREACHED - Trailing drawdown exceeded")
    
    def new_trading_day(self):
        """Reset daily tracking for new day."""
        self.daily_starting_balance = self.current_balance
        self.trading_days += 1
        self.consecutive_losses = 0  # Reset at start of day
        
        logger.info(f"📅 Day {self.trading_days} starting. Balance: ${self.current_balance:,.2f}")
    
    def get_max_risk(self) -> float:
        """Calculate maximum dollar risk for next trade."""
        # Don't risk more than what would breach us
        max_from_daily = self.daily_loss_remaining * 0.5  # Use 50% of remaining
        max_from_trailing = self.trailing_drawdown_remaining * 0.25  # Use 25% of remaining
        
        return min(max_from_daily, max_from_trailing)
    
    def get_position_size(self, stop_distance_points: float, point_value: float = 5.0) -> int:
        """Calculate safe position size based on risk."""
        max_risk = self.get_max_risk()
        risk_per_contract = stop_distance_points * point_value
        
        if risk_per_contract <= 0:
            return 0
        
        contracts = int(max_risk / risk_per_contract)
        contracts = min(contracts, self.firm_rules.max_contracts)
        
        return max(1, contracts)  # At least 1 contract if we can trade
    
    def print_status(self):
        """Print current account status."""
        print("\n" + "="*60)
        print(f"PROP FIRM ACCOUNT STATUS: {self.firm_rules.name}")
        print("="*60)
        print(f"Status:              {self.status.value.upper()}")
        print(f"Balance:             ${self.current_balance:,.2f}")
        print(f"Total P&L:           ${self.total_pnl:+,.2f} ({self.total_pnl_pct:+.2f}%)")
        print(f"Today's P&L:         ${self.daily_pnl:+,.2f} ({self.daily_pnl_pct:+.2f}%)")
        print("-"*60)
        print(f"Profit Target:       ${self.firm_rules.profit_target:,.2f} ({self.total_pnl/self.firm_rules.profit_target*100:.1f}% complete)")
        print(f"Daily Loss Limit:    ${self.firm_rules.daily_loss_limit:,.2f} (${self.daily_loss_remaining:,.2f} remaining)")
        print(f"Trailing Drawdown:   ${self.firm_rules.trailing_drawdown:,.2f} (${self.trailing_drawdown_remaining:,.2f} remaining)")
        print("-"*60)
        print(f"Total Trades:        {self.total_trades}")
        print(f"Win Rate:            {self.winning_trades/self.total_trades*100:.1f}%" if self.total_trades > 0 else "Win Rate:            N/A")
        print(f"Trading Days:        {self.trading_days}/{self.firm_rules.min_trading_days}")
        print(f"Consecutive Losses:  {self.consecutive_losses}")
        print("="*60 + "\n")


# ============================================================================
# TRADING BOT
# ============================================================================

class PropFirmBot:
    """
    Trading bot designed for prop firm accounts.
    
    Key features:
    - Respects daily loss limits
    - Tracks trailing drawdown
    - Adaptive position sizing
    - Automatic risk reduction when approaching limits
    """
    
    def __init__(self, firm_key: str = 'apex_50k'):
        if firm_key not in PROP_FIRMS:
            raise ValueError(f"Unknown firm: {firm_key}. Available: {list(PROP_FIRMS.keys())}")
        
        self.firm_rules = PROP_FIRMS[firm_key]
        self.account = PropFirmAccount(self.firm_rules)
        
        # Trading settings
        self.min_confidence = 0.70
        self.min_predicted_move = 0.10
        self.stop_distance_points = 5.0  # Default stop loss distance
        self.point_value = 5.0  # /MES
        
        # State
        self.current_position = 0
        self.entry_price = 0.0
        self.entry_time = None
        
        logger.info(f"Initialized PropFirmBot for {self.firm_rules.name}")
        logger.info(f"Account size: ${self.firm_rules.account_size:,.0f}")
        logger.info(f"Profit target: ${self.firm_rules.profit_target:,.0f}")
        logger.info(f"Daily limit: ${self.firm_rules.daily_loss_limit:,.0f}")
    
    def should_trade(self, confidence: float, predicted_move: float) -> Tuple[bool, str]:
        """Check if we should take this signal."""
        
        # Check account status
        can_trade, reason = self.account.can_trade()
        if not can_trade:
            return False, reason
        
        # Check signal quality
        if confidence < self.min_confidence:
            return False, f"Confidence too low ({confidence:.2f} < {self.min_confidence})"
        
        if abs(predicted_move) < self.min_predicted_move:
            return False, f"Predicted move too small (${predicted_move:.2f})"
        
        # Check if we're too close to limits (reduce risk)
        if self.account.daily_loss_remaining < self.firm_rules.daily_loss_limit * 0.3:
            return False, "Too close to daily loss limit - stopping for today"
        
        if self.account.trailing_drawdown_remaining < self.firm_rules.trailing_drawdown * 0.3:
            return False, "Too close to trailing drawdown - reducing risk"
        
        return True, "Signal approved"
    
    def calculate_position(self) -> int:
        """Calculate position size based on current risk."""
        contracts = self.account.get_position_size(
            self.stop_distance_points, 
            self.point_value
        )
        
        # Scale down if approaching limits
        risk_factor = min(
            self.account.daily_loss_remaining / self.firm_rules.daily_loss_limit,
            self.account.trailing_drawdown_remaining / self.firm_rules.trailing_drawdown
        )
        
        if risk_factor < 0.5:
            contracts = max(1, contracts // 2)  # Halve position size
        
        return contracts
    
    def open_position(self, direction: int, price: float, timestamp: datetime) -> bool:
        """
        Open a new position.
        
        Args:
            direction: 1 for long, -1 for short
            price: Entry price (in futures points)
            timestamp: Entry time
        
        Returns:
            True if position opened, False otherwise
        """
        if self.current_position != 0:
            logger.warning("Already have open position")
            return False
        
        contracts = self.calculate_position()
        if contracts == 0:
            logger.warning("Position size is 0 - cannot trade")
            return False
        
        self.current_position = contracts * direction
        self.entry_price = price
        self.entry_time = timestamp
        
        direction_str = "LONG" if direction > 0 else "SHORT"
        logger.info(f"OPENED {direction_str} {contracts} contracts @ {price:.2f}")
        
        return True
    
    def close_position(self, price: float, timestamp: datetime, reason: str = "signal") -> float:
        """
        Close current position.
        
        Returns:
            Net P&L from the trade
        """
        if self.current_position == 0:
            return 0.0
        
        # Calculate P&L
        if self.current_position > 0:  # Long
            points = price - self.entry_price
        else:  # Short
            points = self.entry_price - price
        
        contracts = abs(self.current_position)
        gross_pnl = points * self.point_value * contracts
        
        # Subtract costs
        commission = 0.62 * contracts  # Round-turn commission
        slippage = 1.25 * contracts    # 1 tick slippage
        net_pnl = gross_pnl - commission - slippage
        
        # Update account
        is_win = net_pnl > 0
        self.account.update_balance(net_pnl, is_win)
        
        direction_str = "LONG" if self.current_position > 0 else "SHORT"
        result_str = "WIN" if is_win else "LOSS"
        logger.info(f"CLOSED {direction_str} @ {price:.2f} | {result_str}: ${net_pnl:+,.2f}")
        
        # Reset position
        self.current_position = 0
        self.entry_price = 0.0
        self.entry_time = None
        
        return net_pnl
    
    def process_signal(
        self, 
        price: float, 
        predicted_change: float, 
        confidence: float,
        timestamp: datetime
    ) -> Optional[float]:
        """
        Process a trading signal.
        
        Returns:
            P&L if a trade was closed, None otherwise
        """
        # Convert SPY price to futures
        futures_price = price * 10  # SPY to S&P 500 points
        
        should, reason = self.should_trade(confidence, predicted_change)
        
        if not should:
            if "limit" in reason.lower() or "drawdown" in reason.lower():
                # Close any open position if hitting limits
                if self.current_position != 0:
                    return self.close_position(futures_price, timestamp, "risk_limit")
            return None
        
        desired_direction = 1 if predicted_change > 0 else -1
        
        # If we have a position in wrong direction, close it
        if self.current_position != 0:
            current_direction = 1 if self.current_position > 0 else -1
            if current_direction != desired_direction:
                pnl = self.close_position(futures_price, timestamp, "direction_change")
                # Open new position in opposite direction
                self.open_position(desired_direction, futures_price, timestamp)
                return pnl
        else:
            # Open new position
            self.open_position(desired_direction, futures_price, timestamp)
        
        return None


# ============================================================================
# SIMULATION
# ============================================================================

def simulate_prop_firm(predictions_file: str, firm_key: str = 'apex_50k'):
    """
    Simulate trading on a prop firm account using the model's signals.
    """
    import pandas as pd
    
    firm_rules = PROP_FIRMS[firm_key]
    
    print("\n" + "="*70)
    print(f"PROP FIRM SIMULATION: {firm_rules.name}")
    print("="*70)
    print(f"""
Account Size:        ${firm_rules.account_size:,.0f}
Profit Target:       ${firm_rules.profit_target:,.0f} ({firm_rules.profit_target_pct}%)
Daily Loss Limit:    ${firm_rules.daily_loss_limit:,.0f} ({firm_rules.daily_loss_limit_pct}%)
Trailing Drawdown:   ${firm_rules.trailing_drawdown:,.0f} ({firm_rules.trailing_drawdown_pct}%)
Max Contracts:       {firm_rules.max_contracts}
Profit Split:        {firm_rules.profit_split*100:.0f}%
Evaluation Cost:     ${firm_rules.evaluation_cost:.0f}
""")
    
    # Load predictions
    df = pd.read_csv(predictions_file, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"Loaded {len(df):,} predictions")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Initialize bot
    bot = PropFirmBot(firm_key)
    
    # Track by day
    current_date = None
    daily_trades = 0
    total_pnl = 0
    days_traded = 0
    passed_day = None
    
    # Process signals
    for idx, row in df.iterrows():
        timestamp = row['Date']
        trade_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
        
        # New day handling
        if current_date != trade_date:
            if current_date is not None:
                # End of previous day
                if daily_trades > 0:
                    days_traded += 1
            
            current_date = trade_date
            daily_trades = 0
            bot.account.new_trading_day()
            
            # Close overnight position if required
            if not firm_rules.overnight_allowed and bot.current_position != 0:
                futures_price = row['Actual_Price'] * 10
                pnl = bot.close_position(futures_price, timestamp, "end_of_day")
                if pnl:
                    total_pnl += pnl
        
        # Check if we've passed or failed
        if bot.account.status in [AccountStatus.PASSED, AccountStatus.BREACHED]:
            if passed_day is None:
                passed_day = trade_date
            break
        
        # Process signal
        pnl = bot.process_signal(
            price=row['Actual_Price'],
            predicted_change=row['Predicted_Change'],
            confidence=row['Confidence'],
            timestamp=timestamp
        )
        
        if pnl is not None:
            total_pnl += pnl
            daily_trades += 1
    
    # Close any remaining position
    if bot.current_position != 0:
        final_row = df.iloc[-1]
        futures_price = final_row['Actual_Price'] * 10
        pnl = bot.close_position(futures_price, final_row['Date'], "simulation_end")
        if pnl:
            total_pnl += pnl
    
    # Print results
    bot.account.print_status()
    
    print("="*70)
    print("SIMULATION RESULTS")
    print("="*70)
    
    if bot.account.status == AccountStatus.PASSED:
        print(f"""
✅ EVALUATION PASSED!

Days to pass:        {days_traded}
Final P&L:           ${bot.account.total_pnl:+,.2f}
Your profit share:   ${bot.account.total_pnl * firm_rules.profit_split:+,.2f}
Evaluation cost:     ${firm_rules.evaluation_cost:.2f}
Net profit:          ${bot.account.total_pnl * firm_rules.profit_split - firm_rules.evaluation_cost:+,.2f}

ROI on eval cost:    {((bot.account.total_pnl * firm_rules.profit_split) / firm_rules.evaluation_cost - 1) * 100:+,.0f}%
""")
    elif bot.account.status == AccountStatus.BREACHED:
        print(f"""
❌ EVALUATION FAILED

Days traded:         {days_traded}
Final P&L:           ${bot.account.total_pnl:+,.2f}
Reason:              Trailing drawdown breached

Cost of attempt:     ${firm_rules.evaluation_cost:.2f}

Next steps:
1. Review losing trades
2. Reduce position size
3. Increase signal quality threshold
4. Try again with new evaluation
""")
    else:
        print(f"""
⏳ EVALUATION IN PROGRESS

Days traded:         {days_traded}
Current P&L:         ${bot.account.total_pnl:+,.2f}
Progress:            {bot.account.total_pnl/firm_rules.profit_target*100:.1f}% to target

Remaining:
- Profit needed:     ${firm_rules.profit_target - bot.account.total_pnl:,.2f}
- Trading days:      {max(0, firm_rules.min_trading_days - days_traded)} more required
""")
    
    # Show what-if for different prop firms
    print("\n" + "="*70)
    print("COMPARISON: DIFFERENT PROP FIRMS")
    print("="*70)
    
    for key, rules in PROP_FIRMS.items():
        target_pct = rules.profit_target_pct
        max_dd = rules.trailing_drawdown_pct
        split = rules.profit_split * 100
        cost = rules.evaluation_cost
        
        # Estimate based on our results
        if bot.account.total_pnl > 0:
            potential_profit = min(bot.account.total_pnl, rules.profit_target)
            your_share = potential_profit * rules.profit_split
            net = your_share - cost
            roi = (net / cost) * 100 if cost > 0 else 0
            status = "✅" if bot.account.total_pnl >= rules.profit_target else "⏳"
        else:
            status = "❌"
            your_share = 0
            net = -cost
            roi = -100
        
        print(f"{status} {rules.name}")
        print(f"   Target: {target_pct}% | DD: {max_dd}% | Split: {split:.0f}% | Cost: ${cost:.0f}")
        print(f"   Potential profit: ${your_share:+,.0f} | Net: ${net:+,.0f} | ROI: {roi:+,.0f}%")
        print()
    
    return bot


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run prop firm simulation."""
    
    # Find predictions file
    predictions_file = Path(__file__).parent.parent / "results" / "real_minute_strict" / "lightgbm_predictions.csv"
    
    if not predictions_file.exists():
        print(f"ERROR: Predictions file not found: {predictions_file}")
        print("Run the SPY model first to generate predictions.")
        return
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    PROP FIRM TRADING SIMULATION                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  This simulation shows how your trading model would perform on a      ║
║  prop firm evaluation account.                                        ║
║                                                                       ║
║  Key benefits of prop firms:                                          ║
║  • Trade with $50,000 - $300,000 (not your money!)                   ║
║  • NO Pattern Day Trader (PDT) restrictions                          ║
║  • Keep 70-90% of profits                                            ║
║  • Only risk the evaluation fee ($100-200)                           ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Simulate with different prop firms
    for firm_key in ['apex_50k', 'topstep_50k', 'the_trading_pit_10k']:
        bot = simulate_prop_firm(str(predictions_file), firm_key)
        print("\n" + "─"*70 + "\n")
    
    # Print recommendation
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                         RECOMMENDATION                                ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  Based on the simulations:                                            ║
║                                                                       ║
║  🥇 BEST FOR BEGINNERS: The Trading Pit $10K                         ║
║     - Lower profit target (easier to pass)                           ║
║     - Lower evaluation cost ($99)                                    ║
║     - Good starting point                                            ║
║                                                                       ║
║  🥈 BEST VALUE: Apex Trader Funding $50K                             ║
║     - 90% profit split (highest)                                     ║
║     - Reasonable rules                                               ║
║     - Good scaling plan                                              ║
║                                                                       ║
║  🥉 MOST FORGIVING: Bulenox $50K                                     ║
║     - No minimum trading days                                        ║
║     - Overnight allowed                                              ║
║     - News trading allowed                                           ║
║                                                                       ║
║  NEXT STEPS:                                                          ║
║  1. Paper trade for 1-2 weeks to verify strategy                     ║
║  2. Start with smallest account ($10K) to learn the rules            ║
║  3. Scale up to $50K-$100K once you pass consistently                ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
