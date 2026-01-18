# Deploying Trading Bot to Google Cloud

## Quick Start

### Option 1: Automated Deployment (Recommended)

```bash
# Make script executable
chmod +x live_trading/deploy/deploy.sh

# Deploy to GCP (creates VM if it doesn't exist)
./live_trading/deploy/deploy.sh trading-bot-vm us-central1-a
```

### Option 2: Manual Setup

#### Step 1: Create a VM

```bash
# Create a small VM (~$15/month)
gcloud compute instances create trading-bot-vm \
    --zone=us-central1-a \
    --machine-type=e2-small \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=20GB
```

#### Step 2: Connect to VM

```bash
gcloud compute ssh trading-bot-vm --zone=us-central1-a
```

#### Step 3: Set Up on VM

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python
sudo apt-get install -y python3 python3-pip python3-venv git

# Create directory
sudo mkdir -p /opt/trading-bot
sudo chown $USER:$USER /opt/trading-bot
cd /opt/trading-bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install alpaca-py pandas numpy lightgbm scikit-learn pytz
```

#### Step 4: Copy Your Code

From your local machine:
```bash
# Create tarball (from project root)
tar --exclude='results' --exclude='__pycache__' --exclude='.git' \
    -czf trading-bot.tar.gz .

# Copy to VM
gcloud compute scp trading-bot.tar.gz trading-bot-vm:/opt/trading-bot/ \
    --zone=us-central1-a

# Extract on VM
gcloud compute ssh trading-bot-vm --zone=us-central1-a \
    --command="cd /opt/trading-bot && tar -xzf trading-bot.tar.gz"
```

#### Step 5: Set Up as System Service

On the VM:
```bash
# Copy service file
sudo cp /opt/trading-bot/live_trading/deploy/trading-bot.service \
    /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable on boot
sudo systemctl enable trading-bot

# Start the bot
sudo systemctl start trading-bot
```

---

## Managing the Bot

### Check Status
```bash
sudo systemctl status trading-bot
```

### View Live Logs
```bash
# System logs
sudo journalctl -u trading-bot -f

# Application logs
tail -f /opt/trading-bot/logs/bot.log
```

### Stop the Bot
```bash
sudo systemctl stop trading-bot
```

### Restart the Bot
```bash
sudo systemctl restart trading-bot
```

### Update the Code
```bash
# Stop bot
sudo systemctl stop trading-bot

# Re-deploy from local machine
./live_trading/deploy/deploy.sh trading-bot-vm us-central1-a

# Restart
sudo systemctl start trading-bot
```

---

## Cost Estimate

| Resource | Monthly Cost |
|----------|--------------|
| e2-small VM (24/7) | ~$15 |
| Boot disk (20GB) | ~$1 |
| Network (minimal) | ~$1 |
| **Total** | **~$17/month** |

### Cost Optimization

To reduce costs:
```bash
# Use preemptible instance (~70% cheaper, but can be stopped anytime)
gcloud compute instances create trading-bot-vm \
    --zone=us-central1-a \
    --machine-type=e2-small \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --preemptible \
    --maintenance-policy=TERMINATE
```

**Note:** Preemptible VMs can be stopped by Google with 30 seconds notice. Not recommended for live trading with real money.

---

## Security Best Practices

### 1. Use Secret Manager for API Keys

Instead of hardcoding keys in `config.py`:

```bash
# Store secrets
echo -n "your-api-key" | gcloud secrets create alpaca-api-key --data-file=-
echo -n "your-secret" | gcloud secrets create alpaca-secret-key --data-file=-
```

Update `config.py` to read from environment or Secret Manager.

### 2. Restrict Firewall

The bot doesn't need incoming connections:
```bash
# Create firewall rule to block all incoming (except SSH)
gcloud compute firewall-rules create deny-all-incoming \
    --direction=INGRESS \
    --priority=1000 \
    --action=DENY \
    --rules=all \
    --target-tags=trading-bot

# Allow SSH only
gcloud compute firewall-rules create allow-ssh \
    --direction=INGRESS \
    --priority=900 \
    --action=ALLOW \
    --rules=tcp:22 \
    --target-tags=trading-bot
```

### 3. Set Up Alerts

```bash
# Create alert for VM downtime
gcloud alpha monitoring policies create \
    --display-name="Trading Bot Down" \
    --condition="..." \
    --notification-channels="..."
```

---

## Troubleshooting

### Bot Not Starting

```bash
# Check service status
sudo systemctl status trading-bot

# Check logs for errors
sudo journalctl -u trading-bot -n 50

# Test manually
cd /opt/trading-bot
source venv/bin/activate
python3 -m live_trading.run_bot --simulate
```

### Python Package Issues

```bash
cd /opt/trading-bot
source venv/bin/activate
pip install --upgrade alpaca-py pandas numpy lightgbm scikit-learn pytz
```

### Model Training Failing

```bash
# Check available disk space
df -h

# Check memory
free -m

# Increase swap if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Files on VM

```
/opt/trading-bot/
├── venv/                    # Python virtual environment
├── live_trading/
│   ├── config.py            # Your API keys & settings
│   ├── trading_bot.py       # Main bot logic
│   └── saved_models/        # Trained ML models
├── logs/
│   ├── bot.log              # Application logs
│   └── bot_error.log        # Error logs
└── trade_history/           # JSON trade logs
```

---

## Quick Commands Reference

```bash
# SSH into VM
gcloud compute ssh trading-bot-vm --zone=us-central1-a

# Start bot
sudo systemctl start trading-bot

# Stop bot
sudo systemctl stop trading-bot

# View logs
tail -f /opt/trading-bot/logs/bot.log

# Check bot status
sudo systemctl status trading-bot

# Delete VM (when done)
gcloud compute instances delete trading-bot-vm --zone=us-central1-a
```
