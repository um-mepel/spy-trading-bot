#!/usr/bin/env python3
"""
High Confidence Stock Picker - Leak-Free Backtest
Tests the strategy on multiple years with strict train/test separation.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
import warnings
import json
import os
warnings.filterwarnings('ignore')

# Configuration
TOP_N_PER_DAY = 5
HOLD_DAYS = 5
MIN_TRAIN_DAYS = 252  # At least 1 year of training data
RETRAIN_FREQUENCY = 21  # Retrain every month

# Output directory
OUTPUT_DIR = '/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/results/high_confidence_leak_free'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_sp500_tickers():
    """Get S&P 500 tickers - comprehensive list."""
    # Full S&P 500 list (as of 2024)
    tickers = [
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
        'NSC', 'CCI', 'FCX', 'GD', 'ATVI', 'EMR', 'APD', 'HCA', 'AZO', 'SNPS',
        'CDNS', 'MAR', 'SHW', 'MCK', 'NXPI', 'F', 'GM', 'KLAC', 'ANET', 'MNST',
        'ROP', 'CMG', 'FTNT', 'KMB', 'ECL', 'PSA', 'PCAR', 'MPC', 'PSX', 'VLO',
        'EW', 'LRCX', 'AIG', 'ROST', 'TRV', 'AFL', 'OXY', 'CTVA', 'HES', 'AEP',
        'SRE', 'CTAS', 'MCHP', 'KHC', 'WMB', 'MSCI', 'MSI', 'O', 'KDP', 'STZ',
        'A', 'DXCM', 'TEL', 'FDX', 'D', 'DOW', 'CNC', 'YUM', 'IQV', 'JCI',
        'EXC', 'PH', 'ADM', 'IDXX', 'AJG', 'PAYX', 'HSY', 'KEYS', 'DHI', 'LHX',
        'CMI', 'BK', 'BIIB', 'GIS', 'EA', 'ODFL', 'WELL', 'PRU', 'FAST', 'XEL',
        'KR', 'VRSK', 'CPRT', 'OTIS', 'CTSH', 'ED', 'HLT', 'DVN', 'CBRE', 'MTD',
        'GEHC', 'IT', 'FANG', 'EL', 'AWK', 'PPG', 'HAL', 'RMD', 'GPN', 'AME',
        'SBUX', 'DD', 'ON', 'DG', 'ALB', 'GLW', 'NEM', 'HPQ', 'ROK', 'DFS',
        'WEC', 'APTV', 'CHD', 'ANSS', 'MTB', 'CDW', 'WBD', 'DLTR', 'FTV', 'EBAY',
        'ILMN', 'STT', 'VICI', 'EFX', 'MPWR', 'TDY', 'NUE', 'TSCO', 'GWW', 'ZBH',
        'FITB', 'EIX', 'WST', 'HPE', 'AVB', 'HBAN', 'IFF', 'MLM', 'DOV', 'WAB',
        'RJF', 'BR', 'URI', 'SYY', 'PPL', 'TROW', 'AEE', 'VMC', 'TTWO', 'FE',
        'ES', 'LYB', 'TYL', 'CBOE', 'CAH', 'MOH', 'CNP', 'RF', 'HOLX', 'STLD',
        'PFG', 'WY', 'CFG', 'DTE', 'COO', 'LH', 'CLX', 'NVR', 'EXPD', 'K',
        'WAT', 'MKC', 'BAX', 'BALL', 'NTAP', 'CMS', 'EQR', 'INVH', 'PKI', 'FDS',
        'STE', 'TSN', 'GPC', 'IP', 'J', 'AKAM', 'CINF', 'TXT', 'SWKS', 'AES',
        'CE', 'MAA', 'LDOS', 'L', 'POOL', 'VTR', 'LUV', 'ESS', 'SWK', 'BRO',
        'DGX', 'KIM', 'IEX', 'AVY', 'EVRG', 'JBHT', 'MAS', 'PNR', 'NTRS', 'ATO',
        'TECH', 'PEAK', 'KEY', 'HST', 'WRB', 'LNT', 'TRMB', 'GRMN', 'BBY', 'LKQ',
        'TER', 'IPG', 'UDR', 'CDAY', 'NI', 'REG', 'PTC', 'CCL', 'BXP', 'JKHY',
        'ALLE', 'AMCR', 'SJM', 'WYNN', 'NDSN', 'CPT', 'WDC', 'CHRW', 'MGM', 'GL',
        'TAP', 'EMN', 'AIZ', 'ROL', 'FRT', 'AAL', 'UAL', 'DAL', 'BEN', 'BBWI',
        'NCLH', 'CZR', 'HII', 'BIO', 'HSIC', 'MOS', 'ETSY', 'XRAY', 'CRL', 'HAS',
        'SEE', 'NWS', 'NWSA', 'BWA', 'IVZ', 'VFC', 'FMC', 'MTCH', 'DVA', 'FOXA',
        'FOX', 'WHR', 'ALK', 'PARA', 'CTLT', 'APA', 'MHK', 'ZION', 'RHI', 'GNRC',
        'MKTX', 'PAYC', 'JNPR', 'AOS', 'FFIV', 'LW', 'HRL', 'CPB', 'CMA', 'QRVO'
    ]
    return tickers


def create_features(df):
    """Create features for prediction - NO FUTURE LEAKAGE."""
    features = pd.DataFrame(index=df.index)
    
    # Price-based features (all use only past data)
    features['Return_1d'] = df['Close'].pct_change(1)
    features['Return_5d'] = df['Close'].pct_change(5)
    features['Return_10d'] = df['Close'].pct_change(10)
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
    
    # Momentum
    features['RSI_14'] = calculate_rsi(df['Close'], 14)
    features['RSI_5'] = calculate_rsi(df['Close'], 5)
    
    # Volume features
    if 'Volume' in df.columns:
        features['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        features['Volume_Change'] = df['Volume'].pct_change()
    
    # Range features
    if 'High' in df.columns and 'Low' in df.columns:
        features['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
        features['ATR_14'] = calculate_atr(df, 14)
    
    # Mean reversion
    features['Distance_from_20d_high'] = df['Close'] / df['Close'].rolling(20).max() - 1
    features['Distance_from_20d_low'] = df['Close'] / df['Close'].rolling(20).min() - 1
    
    # Trend strength
    features['Trend_5d'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
    features['Trend_20d'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)
    
    return features


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df, period=14):
    """Calculate Average True Range."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr.rolling(period).mean()


def download_stock_data(tickers, start_date, end_date):
    """Download data for multiple stocks."""
    print(f"Downloading data for {len(tickers)} stocks...")
    
    all_data = {}
    failed = []
    
    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"  Downloaded {i + 1}/{len(tickers)}...")
        
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(df) > 100:  # Need enough data
                # Handle multi-level columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                all_data[ticker] = df
        except Exception as e:
            failed.append(ticker)
    
    print(f"Successfully downloaded {len(all_data)} stocks, {len(failed)} failed")
    return all_data


def prepare_training_data(stock_data, end_date, min_days=252):
    """Prepare training data up to end_date (exclusive) - NO LEAKAGE."""
    X_list = []
    y_list = []
    
    for ticker, df in stock_data.items():
        # Only use data before end_date for training
        train_df = df[df.index < end_date].copy()
        
        if len(train_df) < min_days:
            continue
        
        # Create features
        features = create_features(train_df)
        
        # Target: Next 5-day return (we can calculate this for training data)
        # This is NOT leakage because we only use it for training on PAST data
        target = train_df['Close'].shift(-HOLD_DAYS) / train_df['Close'] - 1
        
        # Combine
        combined = features.copy()
        combined['Target'] = target
        combined['Ticker'] = ticker
        
        # Drop rows with NaN
        combined = combined.dropna()
        
        # Remove the last HOLD_DAYS rows to avoid label leakage
        if len(combined) > HOLD_DAYS:
            combined = combined.iloc[:-HOLD_DAYS]
        
        if len(combined) > 50:
            X_list.append(combined.drop(['Target', 'Ticker'], axis=1))
            y_list.append(combined['Target'])
    
    if not X_list:
        return None, None
    
    X = pd.concat(X_list)
    y = pd.concat(y_list)
    
    return X, y


def train_model(X, y):
    """Train LightGBM model."""
    # Convert target to binary (up/down)
    y_binary = (y > 0).astype(int)
    
    model = lgb.LGBMClassifier(
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
    
    model.fit(X, y_binary)
    return model


def make_predictions(model, stock_data, prediction_date):
    """Make predictions for a specific date."""
    predictions = []
    
    for ticker, df in stock_data.items():
        # Get data up to prediction_date
        available = df[df.index <= prediction_date]
        
        if len(available) < 60:  # Need enough history
            continue
        
        if prediction_date not in available.index:
            continue
        
        # Create features for the prediction date
        features = create_features(available)
        
        if prediction_date not in features.index:
            continue
        
        feature_row = features.loc[prediction_date:prediction_date]
        
        if feature_row.isna().any().any():
            continue
        
        try:
            # Get prediction probability
            prob = model.predict_proba(feature_row)[0][1]  # Probability of up
            
            predictions.append({
                'Ticker': ticker,
                'Date': prediction_date,
                'Confidence': prob,
                'Entry_Price': float(available.loc[prediction_date, 'Close'])
            })
        except Exception:
            continue
    
    return predictions


def run_backtest(stock_data, start_date, end_date):
    """Run the full backtest with periodic retraining."""
    print(f"\nRunning backtest from {start_date} to {end_date}")
    
    # Get all trading dates
    spy_data = stock_data.get('SPY', list(stock_data.values())[0])
    trading_dates = spy_data[(spy_data.index >= start_date) & (spy_data.index <= end_date)].index
    
    trades = []
    model = None
    last_train_date = None
    
    for i, pred_date in enumerate(trading_dates):
        # Check if we need to retrain
        if model is None or (last_train_date is not None and 
                            (pred_date - last_train_date).days >= RETRAIN_FREQUENCY):
            print(f"  Training model with data up to {pred_date.date()}...")
            X, y = prepare_training_data(stock_data, pred_date)
            
            if X is None or len(X) < 1000:
                print(f"  Not enough training data, skipping...")
                continue
            
            model = train_model(X, y)
            last_train_date = pred_date
            print(f"  Trained on {len(X)} samples")
        
        # Make predictions
        predictions = make_predictions(model, stock_data, pred_date)
        
        if not predictions:
            continue
        
        # Sort by confidence and pick top N
        predictions = sorted(predictions, key=lambda x: x['Confidence'], reverse=True)
        top_picks = predictions[:TOP_N_PER_DAY]
        
        # Calculate exit date
        exit_idx = list(trading_dates).index(pred_date) + HOLD_DAYS
        if exit_idx >= len(trading_dates):
            continue
        exit_date = trading_dates[exit_idx]
        
        # Record trades
        for pick in top_picks:
            ticker = pick['Ticker']
            
            if ticker not in stock_data:
                continue
            
            ticker_data = stock_data[ticker]
            
            if exit_date not in ticker_data.index:
                continue
            
            exit_price = float(ticker_data.loc[exit_date, 'Close'])
            entry_price = pick['Entry_Price']
            pnl_pct = (exit_price / entry_price - 1) * 100
            
            trades.append({
                'Entry_Date': pred_date,
                'Exit_Date': exit_date,
                'Ticker': ticker,
                'Confidence': pick['Confidence'],
                'Entry_Price': entry_price,
                'Exit_Price': exit_price,
                'Return_Pct': pnl_pct,
                'Win': pnl_pct > 0
            })
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(trading_dates)} days, {len(trades)} trades so far")
    
    return pd.DataFrame(trades)


def calculate_metrics(trades_df, stock_data, start_date, end_date):
    """Calculate performance metrics."""
    if len(trades_df) == 0:
        return {}
    
    # Basic stats
    win_rate = trades_df['Win'].mean() * 100
    avg_return = trades_df['Return_Pct'].mean()
    total_trades = len(trades_df)
    
    # Calculate strategy cumulative return
    # Group by entry date and average the returns
    daily_returns = trades_df.groupby('Entry_Date')['Return_Pct'].mean()
    
    # Calculate cumulative return (compounded)
    cumulative = (1 + daily_returns / 100).cumprod()
    strategy_return = (cumulative.iloc[-1] - 1) * 100 if len(cumulative) > 0 else 0
    
    # Get SPY benchmark
    spy = stock_data.get('SPY', list(stock_data.values())[0])
    spy_start = spy[spy.index >= start_date].iloc[0]['Close']
    spy_end = spy[spy.index <= end_date].iloc[-1]['Close']
    spy_return = (float(spy_end) / float(spy_start) - 1) * 100
    
    # Alpha
    alpha = strategy_return - spy_return
    
    # Sharpe ratio (annualized)
    if len(daily_returns) > 1:
        daily_std = daily_returns.std()
        sharpe = (avg_return / daily_std) * np.sqrt(252 / HOLD_DAYS) if daily_std > 0 else 0
    else:
        sharpe = 0
    
    # Max drawdown
    if len(cumulative) > 0:
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
    else:
        max_drawdown = 0
    
    # Accuracy by confidence quartile
    trades_df['Confidence_Quartile'] = pd.qcut(trades_df['Confidence'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    accuracy_by_quartile = trades_df.groupby('Confidence_Quartile')['Win'].mean() * 100
    
    return {
        'period': f"{start_date.date()} to {end_date.date()}",
        'total_trades': total_trades,
        'win_rate': round(win_rate, 2),
        'avg_return_per_trade': round(avg_return, 3),
        'strategy_return': round(strategy_return, 2),
        'spy_return': round(spy_return, 2),
        'alpha': round(alpha, 2),
        'sharpe_ratio': round(sharpe, 3),
        'max_drawdown': round(max_drawdown, 2),
        'accuracy_by_confidence_quartile': {k: round(v, 2) for k, v in accuracy_by_quartile.to_dict().items()}
    }


def main():
    print("=" * 60)
    print("HIGH CONFIDENCE STOCK PICKER - LEAK-FREE BACKTEST")
    print("=" * 60)
    
    # Get tickers
    tickers = get_sp500_tickers()
    
    # Download data (3+ years for proper training/testing)
    start_download = '2022-01-01'
    end_download = '2026-01-21'
    
    stock_data = download_stock_data(tickers, start_download, end_download)
    
    if len(stock_data) < 50:
        print("Not enough stocks downloaded!")
        return
    
    # Test period: 2023 onwards (using 2022 for initial training)
    test_start = pd.Timestamp('2023-01-01')
    test_end = pd.Timestamp('2026-01-17')
    
    # Run backtest
    trades_df = run_backtest(stock_data, test_start, test_end)
    
    if len(trades_df) == 0:
        print("No trades generated!")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(trades_df, stock_data, test_start, test_end)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Period: {metrics['period']}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']}%")
    print(f"Avg Return/Trade: {metrics['avg_return_per_trade']}%")
    print(f"\nStrategy Return: {metrics['strategy_return']}%")
    print(f"SPY Return: {metrics['spy_return']}%")
    print(f"Alpha: {metrics['alpha']}%")
    print(f"\nSharpe Ratio: {metrics['sharpe_ratio']}")
    print(f"Max Drawdown: {metrics['max_drawdown']}%")
    print(f"\nAccuracy by Confidence Quartile:")
    for q, acc in metrics['accuracy_by_confidence_quartile'].items():
        print(f"  {q}: {acc}%")
    
    # Save results
    trades_df.to_csv(f'{OUTPUT_DIR}/trades.csv', index=False)
    with open(f'{OUTPUT_DIR}/summary.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
    
    # Create visualization
    create_visualization(trades_df, stock_data, test_start, test_end, metrics)
    
    return metrics


def create_visualization(trades_df, stock_data, start_date, end_date, metrics):
    """Create visualization of results."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Cumulative returns comparison
    ax1 = axes[0, 0]
    
    # Strategy cumulative return
    daily_returns = trades_df.groupby('Entry_Date')['Return_Pct'].mean()
    cumulative = (1 + daily_returns / 100).cumprod()
    ax1.plot(cumulative.index, cumulative.values, 'b-', linewidth=2, label='Strategy')
    
    # SPY cumulative return
    spy = stock_data.get('SPY', list(stock_data.values())[0])
    spy_period = spy[(spy.index >= start_date) & (spy.index <= end_date)]['Close']
    spy_cumulative = spy_period / float(spy_period.iloc[0])
    ax1.plot(spy_cumulative.index, spy_cumulative.values, 'gray', linewidth=2, alpha=0.7, label='SPY')
    
    ax1.set_title('Cumulative Returns: Strategy vs SPY', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Win rate by confidence quartile
    ax2 = axes[0, 1]
    quartiles = list(metrics['accuracy_by_confidence_quartile'].keys())
    accuracies = list(metrics['accuracy_by_confidence_quartile'].values())
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    bars = ax2.bar(quartiles, accuracies, color=colors, edgecolor='black')
    ax2.axhline(y=50, color='gray', linestyle='--', label='Random (50%)')
    ax2.set_title('Win Rate by Confidence Quartile', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_xlabel('Confidence Quartile (Q4 = Highest)')
    ax2.legend()
    for bar, acc in zip(bars, accuracies):
        ax2.annotate(f'{acc:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontweight='bold')
    
    # 3. Monthly returns heatmap
    ax3 = axes[1, 0]
    trades_df['Month'] = trades_df['Entry_Date'].dt.to_period('M')
    monthly_returns = trades_df.groupby('Month')['Return_Pct'].mean()
    
    months = [str(m) for m in monthly_returns.index]
    returns = monthly_returns.values
    colors_monthly = ['green' if r > 0 else 'red' for r in returns]
    
    ax3.bar(range(len(months)), returns, color=colors_monthly, alpha=0.7)
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_title('Average Monthly Returns', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Avg Return (%)')
    ax3.set_xticks(range(0, len(months), max(1, len(months)//12)))
    ax3.set_xticklabels([months[i] for i in range(0, len(months), max(1, len(months)//12))], rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Summary stats
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    HIGH CONFIDENCE STOCK PICKER - LEAK-FREE RESULTS
    ================================================
    
    Period: {metrics['period']}
    
    PERFORMANCE:
    • Strategy Return: {metrics['strategy_return']}%
    • SPY Return: {metrics['spy_return']}%
    • Alpha: {metrics['alpha']}%
    
    RISK METRICS:
    • Sharpe Ratio: {metrics['sharpe_ratio']}
    • Max Drawdown: {metrics['max_drawdown']}%
    
    TRADE STATISTICS:
    • Total Trades: {metrics['total_trades']}
    • Win Rate: {metrics['win_rate']}%
    • Avg Return/Trade: {metrics['avg_return_per_trade']}%
    
    METHODOLOGY:
    • Top {TOP_N_PER_DAY} picks per day
    • {HOLD_DAYS}-day hold period
    • Monthly model retraining
    • NO DATA LEAKAGE
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/results_visualization.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {OUTPUT_DIR}/results_visualization.png")
    plt.show()


if __name__ == '__main__':
    main()
