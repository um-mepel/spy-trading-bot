#!/usr/bin/env python3
"""
Run the Stock Picker Bot
========================
Usage:
    python run_stock_picker.py              # Run continuously
    python run_stock_picker.py --run-once   # Run once and exit
    python run_stock_picker.py --train      # Train model only
    python run_stock_picker.py --picks      # Show today's picks
    python run_stock_picker.py --status     # Show current status
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from live_trading.stock_picker_bot import main

if __name__ == '__main__':
    main()
