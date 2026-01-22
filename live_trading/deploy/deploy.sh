#!/bin/bash
# ============================================================================
# Deploy Trading Bot to Google Cloud VM
# ============================================================================
# This script deploys the trading bot from your local machine to GCP.
#
# Prerequisites:
#   1. gcloud CLI installed and authenticated
#   2. A GCP project set up
#   3. VM already created (or this script will create one)
#
# Usage:
#   ./deploy.sh [OPTIONS] <VM_NAME> [ZONE]
#
# Options:
#   --prop-firm    Deploy the prop firm paper trading bot (recommended)
#   --live         Deploy the live trading bot (SPY, requires Alpaca)
#
# Examples:
#   ./deploy.sh --prop-firm trading-bot-vm us-central1-a
#   ./deploy.sh --live trading-bot-vm us-central1-a
# ============================================================================

set -e

# Parse arguments
BOT_TYPE="prop-firm"  # Default to prop firm (safer)
while [[ $# -gt 0 ]]; do
    case $1 in
        --prop-firm)
            BOT_TYPE="prop-firm"
            shift
            ;;
        --live)
            BOT_TYPE="live"
            shift
            ;;
        --stock-picker)
            BOT_TYPE="stock-picker"
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Configuration
VM_NAME=${1:-"trading-bot-vm"}
ZONE=${2:-"us-central1-a"}
MACHINE_TYPE="e2-small"  # $15-20/month, enough for a trading bot
PROJECT=$(gcloud config get-value project)

echo "=============================================="
echo "Deploying Trading Bot to Google Cloud"
echo "=============================================="
echo "Bot Type:     $BOT_TYPE"
echo "VM Name:      $VM_NAME"
echo "Zone:         $ZONE"
echo "Project:      $PROJECT"
echo "Machine Type: $MACHINE_TYPE"
echo "=============================================="

if [ "$BOT_TYPE" == "prop-firm" ]; then
    SERVICE_FILE="prop-firm-bot.service"
    SERVICE_NAME="prop-firm-bot"
    echo ""
    echo ">>> Using PROP FIRM paper trading bot"
    echo "    Simulates /MES futures trading with prop firm rules"
    echo ""
elif [ "$BOT_TYPE" == "stock-picker" ]; then
    SERVICE_FILE="stock-picker.service"
    SERVICE_NAME="stock-picker"
    echo ""
    echo ">>> Using STOCK PICKER bot (S&P 500 daily picks)"
    echo "    Picks 5 stocks daily, 5-day hold, Alpaca paper trading"
    echo ""
else
    SERVICE_FILE="trading-bot.service"
    SERVICE_NAME="trading-bot"
    echo ""
    echo ">>> Using LIVE trading bot (SPY with Alpaca)"
    echo "    WARNING: This trades real/paper money on Alpaca"
    echo ""
fi

# Check if VM exists
if gcloud compute instances describe $VM_NAME --zone=$ZONE &>/dev/null; then
    echo "VM '$VM_NAME' already exists."
else
    echo ""
    echo ">>> Creating new VM..."
    gcloud compute instances create $VM_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=20GB \
        --boot-disk-type=pd-standard \
        --tags=trading-bot \
        --metadata=startup-script='#!/bin/bash
apt-get update
apt-get install -y python3 python3-pip python3-venv git'
    
    echo "Waiting for VM to be ready..."
    sleep 30
fi

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo ""
echo ">>> Copying project files to VM..."

# Create tarball of the project (excluding large files)
cd "$PROJECT_DIR"
tar --exclude='results' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='*.csv' \
    --exclude='*.png' \
    --exclude='*.pkl' \
    -czf /tmp/trading-bot.tar.gz .

# Copy to VM
gcloud compute scp /tmp/trading-bot.tar.gz $VM_NAME:/tmp/ --zone=$ZONE

# Extract and set up on VM
echo ""
echo ">>> Setting up on VM..."
gcloud compute ssh $VM_NAME --zone=$ZONE --command="
    sudo mkdir -p /opt/trading-bot
    sudo chown \$USER:\$USER /opt/trading-bot
    cd /opt/trading-bot
    tar -xzf /tmp/trading-bot.tar.gz
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Install dependencies
    pip install --upgrade pip
    pip install alpaca-py pandas numpy lightgbm scikit-learn pytz yfinance
    
    # Create directories
    mkdir -p logs trade_history saved_models
    
    # Set up systemd service
    sudo cp live_trading/deploy/$SERVICE_FILE /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable $SERVICE_NAME
    
    echo 'Setup complete!'
"

# Clean up local tarball
rm /tmp/trading-bot.tar.gz

echo ""
echo "=============================================="
echo "Deployment Complete!"
echo "=============================================="
echo ""
echo "Bot Type: $BOT_TYPE"
echo "Service:  $SERVICE_NAME"
echo ""
echo "Your trading bot is now on the VM but NOT STARTED yet."
echo ""
echo "To SSH into the VM:"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "To start the bot:"
echo "  sudo systemctl start $SERVICE_NAME"
echo ""
echo "To check status:"
echo "  sudo systemctl status $SERVICE_NAME"
echo ""
echo "To view logs:"
echo "  sudo journalctl -u $SERVICE_NAME -f"
if [ "$BOT_TYPE" == "prop-firm" ]; then
echo "  # or"
echo "  tail -f /opt/trading-bot/logs/prop_firm_bot.log"
else
echo "  # or"
echo "  tail -f /opt/trading-bot/logs/bot.log"
fi
echo ""
echo "To stop the bot:"
echo "  sudo systemctl stop $SERVICE_NAME"
echo ""
if [ "$BOT_TYPE" == "prop-firm" ]; then
echo "To check evaluation status:"
echo "  cd /opt/trading-bot"
echo "  source venv/bin/activate"
echo "  python -m live_trading.prop_firm_paper_bot --status"
echo ""
echo "To reset evaluation (start fresh):"
echo "  sudo systemctl stop $SERVICE_NAME"
echo "  python -m live_trading.prop_firm_paper_bot --reset"
echo "  sudo systemctl start $SERVICE_NAME"
echo ""
fi
echo "=============================================="
