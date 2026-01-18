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
#   ./deploy.sh <VM_NAME> [ZONE]
#
# Example:
#   ./deploy.sh trading-bot-vm us-central1-a
# ============================================================================

set -e

# Configuration
VM_NAME=${1:-"trading-bot-vm"}
ZONE=${2:-"us-central1-a"}
MACHINE_TYPE="e2-small"  # $15-20/month, enough for a trading bot
PROJECT=$(gcloud config get-value project)

echo "=============================================="
echo "Deploying Trading Bot to Google Cloud"
echo "=============================================="
echo "VM Name:      $VM_NAME"
echo "Zone:         $ZONE"
echo "Project:      $PROJECT"
echo "Machine Type: $MACHINE_TYPE"
echo "=============================================="

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
    pip install alpaca-py pandas numpy lightgbm scikit-learn pytz
    
    # Create directories
    mkdir -p logs trade_history saved_models
    
    # Set up systemd service
    sudo cp live_trading/deploy/trading-bot.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable trading-bot
    
    echo 'Setup complete!'
"

# Clean up local tarball
rm /tmp/trading-bot.tar.gz

echo ""
echo "=============================================="
echo "Deployment Complete!"
echo "=============================================="
echo ""
echo "Your trading bot is now on the VM but NOT STARTED yet."
echo ""
echo "To start the bot:"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE"
echo "  sudo systemctl start trading-bot"
echo ""
echo "To check status:"
echo "  sudo systemctl status trading-bot"
echo ""
echo "To view logs:"
echo "  sudo journalctl -u trading-bot -f"
echo "  # or"
echo "  tail -f /opt/trading-bot/logs/bot.log"
echo ""
echo "To stop the bot:"
echo "  sudo systemctl stop trading-bot"
echo ""
echo "=============================================="
