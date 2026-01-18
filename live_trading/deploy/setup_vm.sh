#!/bin/bash
# ============================================================================
# Google Cloud VM Setup Script for Trading Bot
# ============================================================================
# Run this on a fresh Ubuntu VM to set up the trading bot
#
# Usage:
#   chmod +x setup_vm.sh
#   ./setup_vm.sh
# ============================================================================

set -e

echo "=============================================="
echo "Setting up Trading Bot on Google Cloud VM"
echo "=============================================="

# Update system
echo ""
echo ">>> Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Python 3.11+ and pip
echo ""
echo ">>> Installing Python..."
sudo apt-get install -y python3 python3-pip python3-venv

# Install git
echo ""
echo ">>> Installing git..."
sudo apt-get install -y git

# Create app directory
echo ""
echo ">>> Creating application directory..."
sudo mkdir -p /opt/trading-bot
sudo chown $USER:$USER /opt/trading-bot
cd /opt/trading-bot

# Create virtual environment
echo ""
echo ">>> Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo ""
echo ">>> Installing Python dependencies..."
pip install --upgrade pip
pip install alpaca-py pandas numpy lightgbm scikit-learn pytz

# Create logs directory
mkdir -p logs
mkdir -p trade_history
mkdir -p saved_models

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Copy your trading bot files to /opt/trading-bot/"
echo "2. Run: sudo cp trading-bot.service /etc/systemd/system/"
echo "3. Run: sudo systemctl enable trading-bot"
echo "4. Run: sudo systemctl start trading-bot"
echo ""
echo "To check status: sudo systemctl status trading-bot"
echo "To view logs: sudo journalctl -u trading-bot -f"
echo ""
