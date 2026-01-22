#!/usr/bin/env python3
"""
Extended Backtest: Middle Confidence (D4-D7) Strategy
- Longer time period (2018-2026)
- Pure initial training period (2 years) before test begins
- Monthly retraining during test period
- Middle deciles filtering (D4-D7)

NOTE: This is a NEW script - original scripts preserved.
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
INITIAL_TRAIN_YEARS = 2  # Pure training period before testing
RETRAIN_FREQUENCY = 21   # Retrain every month during test period
MIDDLE_DECILES = ['D4', 'D5', 'D6', 'D7']  # Only use these deciles

# Output directory - NEW directory for this extended test
OUTPUT_DIR = '/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/results/extended_middle_deciles'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_sp500_tickers():
    """Get S&P 500 tickers - comprehensive list."""
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
        'NSC', 'CCI', 'FCX', 'GD', 'EMR', 'APD', 'HCA', 'AZO', 'SNPS',
        'CDNS', 'MAR', 'SHW', 'MCK', 'NXPI', 'F', 'GM', 'KLAC', 'ANET', 'MNST',
        'ROP', 'CMG', 'FTNT', 'KMB', 'ECL', 'PSA', 'PCAR', 'MPC', 'PSX', 'VLO',
        'EW', 'LRCX', 'AIG', 'ROST', 'TRV', 'AFL', 'OXY', 'CTVA', 'AEP',
        'SRE', 'CTAS', 'MCHP', 'KHC', 'WMB', 'MSCI', 'MSI', 'O', 'KDP', 'STZ',
        'A', 'DXCM', 'TEL', 'FDX', 'D', 'DOW', 'CNC', 'YUM', 'IQV', 'JCI',
        'EXC', 'PH', 'ADM', 'IDXX', 'AJG', 'PAYX', 'HSY', 'KEYS', 'DHI', 'LHX',
        'CMI', 'BK', 'BIIB', 'GIS', 'EA', 'ODFL', 'WELL', 'PRU', 'FAST', 'XEL',
        'KR', 'VRSK', 'CPRT', 'OTIS', 'CTSH', 'ED', 'HLT', 'DVN', 'CBRE', 'MTD',
        'GEHC', 'IT', 'FANG', 'EL', 'AWK', 'PPG', 'HAL', 'RMD', 'GPN', 'AME',
        'SBUX', 'DD', 'ON', 'DG', 'ALB', 'GLW', 'NEM', 'HPQ', 'ROK',
        'WEC', 'APTV', 'CHD', 'MTB', 'CDW', 'WBD', 'DLTR', 'FTV', 'EBAY',
        'STT', 'VICI', 'EFX', 'MPWR', 'TDY', 'NUE', 'TSCO', 'GWW', 'ZBH',
        'FITB', 'EIX', 'WST', 'HPE', 'AVB', 'HBAN', 'IFF', 'MLM', 'DOV', 'WAB',
        'RJF', 'BR', 'URI', 'SYY', 'PPL', 'TROW', 'AEE', 'VMC', 'TTWO', 'FE',
        'ES', 'LYB', 'TYL', 'CBOE', 'CAH', 'MOH', 'CNP', 'RF', 'HOLX', 'STLD',
        'PFG', 'WY', 'CFG', 'DTE', 'COO', 'LH', 'CLX', 'NVR', 'EXPD', 'K',
        'WAT', 'MKC', 'BAX', 'BALL', 'NTAP', 'CMS', 'EQR', 'INVH', 'FDS',
        'STE', 'TSN', 'GPC', 'IP', 'J', 'AKAM', 'CINF', 'TXT', 'SWKS', 'AES',
        'CE', 'MAA', 'LDOS', 'L', 'POOL', 'VTR', 'LUV', 'ESS', 'SWK', 'BRO',
        'DGX', 'KIM', 'IEX', 'AVY', 'EVRG', 'JBHT', 'MAS', 'PNR', 'NTRS', 'ATO',
        'TECH', 'KEY', 'HST', 'WRB', 'LNT', 'TRMB', 'GRMN', 'BBY', 'LKQ',
        'TER', 'IPG', 'UDR', 'NI', 'REG', 'PTC', 'CCL', 'BXP', 'JKHY',
        'ALLE', 'AMCR', 'SJM', 'WYNN', 'NDSN', 'CPT', 'WDC', 'CHRW', 'MGM', 'GL',
        'TAP', 'EMN', 'AIZ', 'ROL', 'FRT', 'AAL', 'UAL', 'DAL', 'BEN', 'BBWI',
        'NCLH', 'CZR', 'HII', 'BIO', 'HSIC', 'MOS', 'ETSY', 'XRAY', 'CRL', 'HAS',
        'SEE', 'NWS', 'NWSA', 'BWA', 'IVZ', 'VFC', 'FMC', 'MTCH', 'DVA', 'FOXA',
        'FOX', 'WHR', 'ALK', 'APA', 'MHK', 'ZION', 'RHI', 'GNRC',
        'MKTX', 'PAYC', 'AOS', 'FFIV', 'LW', 'HRL', 'CPB', 'CMA', 'QRVO'
    ]
    return tickers


def create_features(df):
    """Create features for prediction - NO FUTURE LEAKAGE."""
    features = pd.DataFrame(index=df.index)
    
    # Price-based features
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
    print(f"Downloading data for {len(tickers)} stocks from {start_date} to {end_date}...")
    
    all_data = {}
    failed = []
    
    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"  Downloaded {i + 1}/{len(tickers)}...")
        
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(df) > 100:
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
        train_df = df[df.index < end_date].copy()
        
        if len(train_df) < min_days:
            continue
        
        features = create_features(train_df)
        target = train_df['Close'].shift(-HOLD_DAYS) / train_df['Close'] - 1
        
        combined = features.copy()
        combined['Target'] = target
        combined['Ticker'] = ticker
        combined = combined.dropna()
        
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
        available = df[df.index <= prediction_date]
        
        if len(available) < 60:
            continue
        
        if prediction_date not in available.index:
            continue
        
        features = create_features(available)
        
        if prediction_date not in features.index:
            continue
        
        feature_row = features.loc[prediction_date:prediction_date]
        
        if feature_row.isna().any().any():
            continue
        
        try:
            prob = model.predict_proba(feature_row)[0][1]
            
            predictions.append({
                'Ticker': ticker,
                'Date': prediction_date,
                'Confidence': prob,
                'Entry_Price': float(available.loc[prediction_date, 'Close'])
            })
        except Exception:
            continue
    
    return predictions


def assign_deciles(predictions):
    """Assign decile labels to predictions based on confidence."""
    if len(predictions) < 10:
        return predictions
    
    df = pd.DataFrame(predictions)
    df['Decile'] = pd.qcut(df['Confidence'], q=10, labels=[f'D{i}' for i in range(1, 11)])
    return df.to_dict('records')


def run_extended_backtest(stock_data, train_start, test_start, test_end):
    """Run the extended backtest with initial training period."""
    print(f"\n{'='*60}")
    print("EXTENDED BACKTEST - MIDDLE DECILES (D4-D7)")
    print(f"{'='*60}")
    print(f"Initial training period: {train_start} to {test_start}")
    print(f"Test period: {test_start} to {test_end}")
    print(f"Using deciles: {MIDDLE_DECILES}")
    
    # Get trading dates
    spy_data = stock_data.get('SPY', list(stock_data.values())[0])
    trading_dates = spy_data[(spy_data.index >= test_start) & (spy_data.index <= test_end)].index
    
    # Initial training with FULL training period
    print(f"\nInitial training with {INITIAL_TRAIN_YEARS} years of data...")
    X_init, y_init = prepare_training_data(stock_data, test_start)
    if X_init is None:
        print("Not enough initial training data!")
        return None
    
    model = train_model(X_init, y_init)
    print(f"Initial model trained on {len(X_init)} samples")
    
    all_trades = []
    filtered_trades = []
    last_train_date = test_start
    
    for i, pred_date in enumerate(trading_dates):
        # Check if we need to retrain
        if (pred_date - last_train_date).days >= RETRAIN_FREQUENCY:
            print(f"  Retraining model with data up to {pred_date.date()}...")
            X, y = prepare_training_data(stock_data, pred_date)
            
            if X is not None and len(X) > 1000:
                model = train_model(X, y)
                last_train_date = pred_date
                print(f"  Retrained on {len(X)} samples")
        
        # Make predictions
        predictions = make_predictions(model, stock_data, pred_date)
        
        if not predictions or len(predictions) < 10:
            continue
        
        # Assign deciles across ALL predictions for this day
        predictions = assign_deciles(predictions)
        
        # Calculate exit date
        exit_idx = list(trading_dates).index(pred_date) + HOLD_DAYS
        if exit_idx >= len(trading_dates):
            continue
        exit_date = trading_dates[exit_idx]
        
        # Get top N from ALL deciles for the "all trades" comparison
        sorted_all = sorted(predictions, key=lambda x: x['Confidence'], reverse=True)
        top_all = sorted_all[:TOP_N_PER_DAY]
        
        # Get top N from MIDDLE DECILES only for filtered strategy
        middle_only = [p for p in predictions if p.get('Decile') in MIDDLE_DECILES]
        sorted_middle = sorted(middle_only, key=lambda x: x['Confidence'], reverse=True)
        top_middle = sorted_middle[:TOP_N_PER_DAY]
        
        # Record ALL trades (from top 5 overall)
        for pick in top_all:
            ticker = pick['Ticker']
            
            if ticker not in stock_data:
                continue
            
            ticker_data = stock_data[ticker]
            
            if exit_date not in ticker_data.index:
                continue
            
            exit_price = float(ticker_data.loc[exit_date, 'Close'])
            entry_price = pick['Entry_Price']
            pnl_pct = (exit_price / entry_price - 1) * 100
            
            trade = {
                'Entry_Date': pred_date,
                'Exit_Date': exit_date,
                'Ticker': ticker,
                'Confidence': pick['Confidence'],
                'Decile': pick.get('Decile', 'N/A'),
                'Entry_Price': entry_price,
                'Exit_Price': exit_price,
                'Return_Pct': pnl_pct,
                'Win': pnl_pct > 0
            }
            
            all_trades.append(trade)
        
        # Record FILTERED trades (from middle deciles only)
        for pick in top_middle:
            ticker = pick['Ticker']
            
            if ticker not in stock_data:
                continue
            
            ticker_data = stock_data[ticker]
            
            if exit_date not in ticker_data.index:
                continue
            
            exit_price = float(ticker_data.loc[exit_date, 'Close'])
            entry_price = pick['Entry_Price']
            pnl_pct = (exit_price / entry_price - 1) * 100
            
            trade = {
                'Entry_Date': pred_date,
                'Exit_Date': exit_date,
                'Ticker': ticker,
                'Confidence': pick['Confidence'],
                'Decile': pick.get('Decile', 'N/A'),
                'Entry_Price': entry_price,
                'Exit_Price': exit_price,
                'Return_Pct': pnl_pct,
                'Win': pnl_pct > 0
            }
            
            filtered_trades.append(trade)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(trading_dates)} days, {len(filtered_trades)} filtered trades")
    
    return pd.DataFrame(all_trades), pd.DataFrame(filtered_trades)


def simulate_portfolio(trades_df, position_size_pct):
    """Simulate portfolio with proper position sizing."""
    if len(trades_df) == 0:
        return None, 0
    
    dates = pd.date_range(trades_df['Entry_Date'].min(), trades_df['Exit_Date'].max(), freq='D')
    portfolio_value = 100.0
    portfolio_history = []
    positions = []
    
    for date in dates:
        # Close positions
        for p in [pos for pos in positions if pos['exit_date'].date() == date.date()]:
            pnl = p['initial_value'] * (p['return_pct'] / 100)
            portfolio_value += pnl
        
        positions = [p for p in positions if p['exit_date'].date() != date.date()]
        
        # Add new positions
        day_trades = trades_df[trades_df['Entry_Date'].dt.date == date.date()]
        for _, trade in day_trades.iterrows():
            position_value = portfolio_value * (position_size_pct / 100)
            positions.append({
                'exit_date': trade['Exit_Date'],
                'return_pct': trade['Return_Pct'],
                'initial_value': position_value
            })
        
        portfolio_history.append({'Date': date, 'Value': portfolio_value})
    
    # Close remaining
    for p in positions:
        pnl = p['initial_value'] * (p['return_pct'] / 100)
        portfolio_value += pnl
    
    return pd.DataFrame(portfolio_history).set_index('Date'), portfolio_value


def main():
    print("=" * 60)
    print("EXTENDED BACKTEST: MIDDLE DECILES STRATEGY")
    print("=" * 60)
    
    # Get tickers
    tickers = get_sp500_tickers()
    
    # Extended date range: 2018 to present
    # 2018-2019 = initial training (2 years)
    # 2020-2026 = test period (6+ years, includes COVID crash and recovery)
    start_download = '2018-01-01'
    end_download = '2026-01-21'
    
    train_start = pd.Timestamp('2018-01-01')
    test_start = pd.Timestamp('2020-01-01')  # Test starts after 2 years of training
    test_end = pd.Timestamp('2026-01-17')
    
    # Download data
    stock_data = download_stock_data(tickers, start_download, end_download)
    
    if len(stock_data) < 50:
        print("Not enough stocks downloaded!")
        return
    
    # Run backtest
    all_trades_df, filtered_trades_df = run_extended_backtest(
        stock_data, train_start, test_start, test_end
    )
    
    if all_trades_df is None or len(filtered_trades_df) == 0:
        print("No trades generated!")
        return
    
    # Calculate position sizing for filtered trades
    trades_per_day = filtered_trades_df.groupby('Entry_Date').size().mean()
    avg_concurrent = trades_per_day * HOLD_DAYS
    filtered_position_size = 100 / max(avg_concurrent, 1)
    
    # Position sizing for all trades
    all_position_size = 4.0  # ~25 concurrent positions
    
    print(f"\nFiltered trades: {len(filtered_trades_df)}")
    print(f"All trades: {len(all_trades_df)}")
    print(f"Filtered position size: {filtered_position_size:.1f}%")
    
    # Simulate portfolios
    print("\nSimulating portfolios...")
    filtered_history, filtered_final = simulate_portfolio(filtered_trades_df, filtered_position_size)
    all_history, all_final = simulate_portfolio(all_trades_df, all_position_size)
    
    # Get SPY benchmark (download separately if not in stock_data)
    if 'SPY' in stock_data:
        spy = stock_data['SPY']
    else:
        print("Downloading SPY for benchmark...")
        spy = yf.download('SPY', start=test_start - timedelta(days=30), end=datetime.now(), progress=False)
    spy_period = spy[(spy.index >= test_start) & (spy.index <= test_end)]['Close']
    spy_return = (float(spy_period.iloc[-1]) / float(spy_period.iloc[0]) - 1) * 100
    
    # Calculate metrics
    filtered_return = (filtered_final / 100 - 1) * 100
    all_return = (all_final / 100 - 1) * 100
    period_years = (test_end - test_start).days / 365
    
    filtered_annualized = ((filtered_final / 100) ** (1 / period_years) - 1) * 100
    all_annualized = ((all_final / 100) ** (1 / period_years) - 1) * 100
    spy_annualized = ((1 + spy_return/100) ** (1 / period_years) - 1) * 100
    
    # Drawdowns
    filtered_rolling_max = filtered_history['Value'].cummax()
    filtered_dd = ((filtered_history['Value'] - filtered_rolling_max) / filtered_rolling_max * 100).min()
    
    all_rolling_max = all_history['Value'].cummax()
    all_dd = ((all_history['Value'] - all_rolling_max) / all_rolling_max * 100).min()
    
    # Sharpe ratios
    filtered_daily = filtered_history['Value'].pct_change().dropna()
    filtered_sharpe = filtered_daily.mean() / filtered_daily.std() * np.sqrt(252) if filtered_daily.std() > 0 else 0
    
    all_daily = all_history['Value'].pct_change().dropna()
    all_sharpe = all_daily.mean() / all_daily.std() * np.sqrt(252) if all_daily.std() > 0 else 0
    
    # Print results
    print("\n" + "=" * 70)
    print("EXTENDED BACKTEST RESULTS (2020-2026, ~6 years)")
    print("=" * 70)
    print(f"Initial Training: 2018-2019 (2 years)")
    print(f"Test Period: {test_start.date()} to {test_end.date()} ({period_years:.1f} years)")
    print()
    print(f"{'Metric':<25} {'D4-D7 Only':<15} {'All Deciles':<15} {'SPY':<15}")
    print("-" * 70)
    print(f"{'Total Return':<25} {filtered_return:>13.2f}% {all_return:>13.2f}% {spy_return:>13.2f}%")
    print(f"{'Annualized Return':<25} {filtered_annualized:>13.2f}% {all_annualized:>13.2f}% {spy_annualized:>13.2f}%")
    print(f"{'Alpha vs SPY':<25} {filtered_return - spy_return:>13.2f}% {all_return - spy_return:>13.2f}% {'--':>15}")
    print(f"{'Sharpe Ratio':<25} {filtered_sharpe:>14.3f} {all_sharpe:>14.3f} {'--':>15}")
    print(f"{'Max Drawdown':<25} {filtered_dd:>13.2f}% {all_dd:>13.2f}% {'--':>15}")
    print(f"{'Win Rate':<25} {filtered_trades_df['Win'].mean()*100:>13.2f}% {all_trades_df['Win'].mean()*100:>13.2f}% {'--':>15}")
    print(f"{'Total Trades':<25} {len(filtered_trades_df):>14} {len(all_trades_df):>14} {'--':>15}")
    
    verdict = "D4-D7 BEATS SPY" if filtered_return > spy_return else "SPY OUTPERFORMS"
    print()
    print(f"VERDICT: {verdict} (Alpha: {filtered_return - spy_return:+.2f}%)")
    
    # Save results
    filtered_trades_df.to_csv(f'{OUTPUT_DIR}/filtered_trades.csv', index=False)
    all_trades_df.to_csv(f'{OUTPUT_DIR}/all_trades.csv', index=False)
    
    # Create visualization
    create_visualization(
        filtered_history, all_history, spy_period, 
        filtered_trades_df, all_trades_df,
        filtered_return, all_return, spy_return,
        filtered_annualized, all_annualized, spy_annualized,
        filtered_sharpe, all_sharpe,
        filtered_dd, all_dd,
        test_start, test_end, period_years
    )
    
    # Save summary
    summary = {
        'test_period': f"{test_start.date()} to {test_end.date()}",
        'initial_training': f"{train_start.date()} to {test_start.date()}",
        'years': round(period_years, 1),
        'filtered_strategy': {
            'deciles': MIDDLE_DECILES,
            'total_return': round(filtered_return, 2),
            'annualized': round(filtered_annualized, 2),
            'alpha': round(filtered_return - spy_return, 2),
            'sharpe': round(filtered_sharpe, 3),
            'max_drawdown': round(filtered_dd, 2),
            'trades': len(filtered_trades_df),
            'win_rate': round(filtered_trades_df['Win'].mean() * 100, 2)
        },
        'all_deciles': {
            'total_return': round(all_return, 2),
            'annualized': round(all_annualized, 2),
            'alpha': round(all_return - spy_return, 2),
            'sharpe': round(all_sharpe, 3),
            'max_drawdown': round(all_dd, 2),
            'trades': len(all_trades_df),
            'win_rate': round(all_trades_df['Win'].mean() * 100, 2)
        },
        'spy': {
            'total_return': round(spy_return, 2),
            'annualized': round(spy_annualized, 2)
        },
        'verdict': verdict
    }
    
    with open(f'{OUTPUT_DIR}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
    
    return summary


def create_visualization(filtered_history, all_history, spy_period,
                         filtered_trades, all_trades,
                         filtered_return, all_return, spy_return,
                         filtered_ann, all_ann, spy_ann,
                         filtered_sharpe, all_sharpe,
                         filtered_dd, all_dd,
                         test_start, test_end, period_years):
    """Create comprehensive visualization."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Normalize SPY
    spy_normalized = spy_period / float(spy_period.iloc[0]) * 100
    
    # 1. Cumulative Returns
    ax1 = axes[0, 0]
    ax1.plot(filtered_history.index, filtered_history['Value'], 'b-', linewidth=2, 
             label=f'D4-D7 Only ({filtered_return:.1f}%)')
    ax1.plot(all_history.index, all_history['Value'], 'g--', linewidth=2, alpha=0.7,
             label=f'All Deciles ({all_return:.1f}%)')
    ax1.plot(spy_normalized.index, spy_normalized.values, 'gray', linewidth=2, alpha=0.5,
             label=f'SPY ({spy_return:.1f}%)')
    ax1.axhline(y=100, color='black', linestyle='--', alpha=0.3)
    ax1.set_title('Extended Backtest: 6 Years (2020-2026) with 2-Year Initial Training', 
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # 2. Drawdown comparison
    ax2 = axes[0, 1]
    filtered_rolling_max = filtered_history['Value'].cummax()
    filtered_drawdown = (filtered_history['Value'] - filtered_rolling_max) / filtered_rolling_max * 100
    all_rolling_max = all_history['Value'].cummax()
    all_drawdown = (all_history['Value'] - all_rolling_max) / all_rolling_max * 100
    
    ax2.fill_between(filtered_drawdown.index, filtered_drawdown.values, 0, 
                     color='blue', alpha=0.3, label=f'D4-D7 (Max: {filtered_dd:.1f}%)')
    ax2.fill_between(all_drawdown.index, all_drawdown.values, 0,
                     color='green', alpha=0.2, label=f'All (Max: {all_dd:.1f}%)')
    ax2.set_title('Drawdown Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # 3. Win rate by decile
    ax3 = axes[1, 0]
    all_trades['Decile_Cat'] = pd.Categorical(all_trades['Decile'], 
                                               categories=[f'D{i}' for i in range(1, 11)],
                                               ordered=True)
    acc_by_decile = all_trades.groupby('Decile_Cat', observed=False)['Win'].mean() * 100
    colors = ['lightgray'] * 3 + ['green'] * 4 + ['lightgray'] * 3
    bars = ax3.bar(range(10), acc_by_decile.values, color=colors, edgecolor='black')
    ax3.axhline(y=50, color='gray', linestyle='--', label='Random')
    ax3.axhline(y=filtered_trades['Win'].mean()*100, color='blue', linestyle='-', linewidth=2,
                label=f'D4-D7 Avg ({filtered_trades["Win"].mean()*100:.1f}%)')
    ax3.set_title('Win Rate by Decile (D4-D7 Selected)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Win Rate (%)')
    ax3.set_xlabel('Confidence Decile')
    ax3.set_xticks(range(10))
    ax3.set_xticklabels([f'D{i}' for i in range(1, 11)])
    ax3.legend()
    ax3.set_ylim(45, 60)
    
    # 4. Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    EXTENDED BACKTEST: MIDDLE DECILES (D4-D7)
    {'='*55}
    
    Initial Training: 2018-2019 (2 years, pure training)
    Test Period: {test_start.date()} to {test_end.date()} ({period_years:.1f} years)
    
    RESULTS:
                       D4-D7 Only    All Deciles    SPY
      Total Return:    {filtered_return:+8.2f}%     {all_return:+8.2f}%    {spy_return:+8.2f}%
      Annualized:      {filtered_ann:+8.2f}%     {all_ann:+8.2f}%    {spy_ann:+8.2f}%
      Alpha:           {filtered_return - spy_return:+8.2f}%     {all_return - spy_return:+8.2f}%        --
      Sharpe:          {filtered_sharpe:8.3f}      {all_sharpe:8.3f}        --
      Max Drawdown:    {filtered_dd:8.2f}%     {all_dd:8.2f}%        --
    
    TRADE STATISTICS:
      D4-D7 Trades:  {len(filtered_trades)}  (Win Rate: {filtered_trades['Win'].mean()*100:.1f}%)
      All Trades:    {len(all_trades)}  (Win Rate: {all_trades['Win'].mean()*100:.1f}%)
    
    METHODOLOGY:
      - 5 picks/day, 5-day hold
      - Monthly retraining during test
      - Position sizing: {100/max((filtered_trades.groupby('Entry_Date').size().mean()*5), 1):.0f}% per position
      - NO DATA LEAKAGE
    
    VERDICT: {'D4-D7 BEATS SPY' if filtered_return > spy_return else 'SPY OUTPERFORMS'}
    """
    
    color = 'lightgreen' if filtered_return > spy_return else 'lightyellow'
    ax4.text(0.02, 0.5, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/extended_results.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {OUTPUT_DIR}/extended_results.png")
    plt.show()


if __name__ == '__main__':
    main()
