#!/usr/bin/env python3
"""
Live Trading Bot Runner
=======================
Main entry point for the live trading system.

Usage:
    # Train model and start trading
    python live_trading/run_bot.py
    
    # Just train the model (no trading)
    python live_trading/run_bot.py --train-only
    
    # Run in simulation mode (predictions only, no orders)
    python live_trading/run_bot.py --simulate
    
    # Retrain model before trading
    python live_trading/run_bot.py --retrain

IMPORTANT: This uses PAPER TRADING by default.
Edit live_trading/config.py and set PAPER_TRADING=False for live money.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from live_trading.model_trainer import LiveModelTrainer
from live_trading.trading_bot import AlpacaTradingBot
from live_trading.config import PAPER_TRADING, SYMBOL


def train_model(days_back=90):
    """Train or retrain the model."""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    trainer = LiveModelTrainer()
    success = trainer.run_full_training(days_back=days_back)
    
    return success


def run_simulation():
    """Run in simulation mode - predictions only, no real orders."""
    print("\n" + "="*60)
    print("SIMULATION MODE")
    print("="*60)
    print("Making predictions but NOT placing orders")
    print("="*60 + "\n")
    
    trainer = LiveModelTrainer()
    
    # Load or train model
    if not trainer.load_model():
        print("No saved model, training...")
        trainer.run_full_training()
    
    # Get recent data and make a prediction
    print("\nFetching recent data...")
    df = trainer.fetch_minute_data(days_back=7)
    
    if df is not None and len(df) > 60:
        df = trainer.add_technical_indicators(df)
        df = df.dropna()
        
        if len(df) > 0:
            latest = df.iloc[-1]
            pred, conf = trainer.predict(latest)
            direction = "UP" if pred > 0 else "DOWN"
            
            print("\n" + "="*60)
            print("LATEST PREDICTION")
            print("="*60)
            print(f"Symbol:      {SYMBOL}")
            print(f"Price:       ${latest['Close']:.2f}")
            print(f"Prediction:  ${pred:+.4f} ({direction})")
            print(f"Confidence:  {conf:.2f}")
            print(f"Time:        {latest['Datetime']}")
            print("="*60)
            
            if conf >= 0.7:
                print(f"\n✓ HIGH CONFIDENCE - Would trade in live mode")
            else:
                print(f"\n○ Low confidence - Would skip trade")


def run_bot(interval=60):
    """Run the live trading bot."""
    print("\n" + "="*60)
    print("LIVE TRADING BOT")
    print("="*60)
    
    if PAPER_TRADING:
        print("MODE: PAPER TRADING (no real money)")
    else:
        print("⚠️  MODE: LIVE TRADING (REAL MONEY)")
        confirm = input("Are you sure you want to trade with real money? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Aborted.")
            return
    
    print("="*60 + "\n")
    
    bot = AlpacaTradingBot()
    bot.run(interval_seconds=interval)


def main():
    parser = argparse.ArgumentParser(
        description='Live Trading Bot for Alpaca',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python live_trading/run_bot.py              # Train model (if needed) and start trading
  python live_trading/run_bot.py --train-only # Just train the model
  python live_trading/run_bot.py --simulate   # Predictions only, no orders
  python live_trading/run_bot.py --retrain    # Force retrain before trading
  python live_trading/run_bot.py --interval 30 # Check every 30 seconds

NOTE: Paper trading is enabled by default. 
Edit live_trading/config.py to use real money.
        """
    )
    
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only train the model, do not start trading'
    )
    
    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Simulation mode: make predictions but do not place orders'
    )
    
    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Force retrain the model before trading'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Seconds between prediction checks (default: 60)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=90,
        help='Days of historical data for training (default: 90)'
    )
    
    args = parser.parse_args()
    
    # Header
    print("\n" + "="*60)
    print("     ALPACA TRADING BOT - SPY MINUTE-LEVEL PREDICTIONS")
    print("="*60)
    print(f"  Symbol:        {SYMBOL}")
    print(f"  Paper Trading: {PAPER_TRADING}")
    print(f"  Interval:      {args.interval}s")
    print("="*60)
    
    try:
        if args.train_only:
            # Just train
            success = train_model(days_back=args.days)
            if success:
                print("\n✓ Model trained and saved. Ready for trading.")
            else:
                print("\n✗ Training failed.")
                sys.exit(1)
        
        elif args.simulate:
            # Simulation mode
            run_simulation()
        
        else:
            # Normal trading mode
            if args.retrain:
                print("\nRetraining model as requested...")
                train_model(days_back=args.days)
            
            run_bot(interval=args.interval)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
