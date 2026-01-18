#!/usr/bin/env python3
"""
Test optimized strategy on a different stock (Microsoft - MSFT)
Downloads real market data and applies the same filters
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def download_stock_data(ticker, start_date, end_date):
    """Download stock data from Yahoo Finance"""
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")
    try:
        # Add retries for network issues
        import time
        for attempt in range(3):
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if len(data) > 0:
                    print(f"✓ Downloaded {len(data)} trading days")
                    return data
                else:
                    time.sleep(2)
            except Exception as retry_error:
                if attempt < 2:
                    print(f"  Retry {attempt+1}...")
                    time.sleep(2)
                else:
                    raise retry_error
        return None
    except Exception as e:
        print(f"✗ Error downloading data: {e}")
        print(f"  Retrying with alternative method...")
        return None

def calculate_simple_signals(data):
    """Generate simple trading signals based on moving averages and momentum"""
    df = data.copy()
    
    # Calculate indicators
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = calculate_rsi(df['Close'], period=14)
    df['Volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
    
    # Generate signals (without LightGBM, just technical analysis)
    # Use confidence score based on indicator alignment
    df['Signal'] = 'HOLD'
    df['Confidence'] = 0.5
    
    for idx in range(200, len(df)):
        # BUY signal: Price above MA50 and MA50 > MA200 (uptrend) and RSI < 70
        if (df['Close'].iloc[idx] > df['MA50'].iloc[idx] and 
            df['MA50'].iloc[idx] > df['MA200'].iloc[idx] and 
            df['RSI'].iloc[idx] < 70):
            
            # Confidence based on RSI strength
            rsi = df['RSI'].iloc[idx]
            if rsi < 30:
                confidence = 0.95  # Oversold
            elif rsi < 40:
                confidence = 0.85  # Strong momentum
            elif rsi < 50:
                confidence = 0.75  # Good momentum
            else:
                confidence = 0.65  # Weak momentum
            
            df.loc[df.index[idx], 'Signal'] = 'BUY'
            df.loc[df.index[idx], 'Confidence'] = confidence
        
        # SELL signal: Price below MA50 and MA50 < MA200 (downtrend) and RSI > 30
        elif (df['Close'].iloc[idx] < df['MA50'].iloc[idx] and 
              df['MA50'].iloc[idx] < df['MA200'].iloc[idx] and 
              df['RSI'].iloc[idx] > 30):
            
            rsi = df['RSI'].iloc[idx]
            if rsi > 70:
                confidence = 0.95  # Overbought
            elif rsi > 60:
                confidence = 0.85  # Weak momentum
            elif rsi > 50:
                confidence = 0.75  # Weaker momentum
            else:
                confidence = 0.65  # Even weaker
            
            df.loc[df.index[idx], 'Signal'] = 'SELL'
            df.loc[df.index[idx], 'Confidence'] = confidence
    
    return df

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)
    
    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)
    
    return rsi

def apply_optimized_filters(signals_df):
    """Apply the same filters from the optimized strategy:
    - BUY signal
    - Price > 50-day MA (momentum confirmation)
    - Confidence > 0.85 (quality filter)
    """
    signals_df['Above_MA50'] = signals_df['Close'] > signals_df['MA50']
    signals_df['Passes_Filter'] = (
        (signals_df['Signal'] == 'BUY') & 
        (signals_df['Above_MA50']) & 
        (signals_df['Confidence'] > 0.85)
    )
    return signals_df

def run_backtest(df, initial_capital=100000):
    """Run backtest with the optimized strategy"""
    results = []
    
    cash = initial_capital
    shares = 0
    portfolio_values = []
    
    # Position sizing (base sizes used in optimized strategy)
    position_sizes = {
        'high': 0.90,      # Confidence > 0.8
        'medium': 0.70,    # Confidence > 0.65
        'low': 0.50,       # Confidence > 0.5
        'very_low': 0.20   # Confidence < 0.5
    }
    
    # Apply 1.3x leverage multiplier
    leverage_multiplier = 1.3
    position_sizes = {k: v * leverage_multiplier for k, v in position_sizes.items()}
    
    shv_daily_return = 0.00015  # 0.015% daily return on cash
    
    for idx in range(len(df)):
        date = df.index[idx]
        price = df['Close'].iloc[idx]
        signal = df['Signal'].iloc[idx]
        confidence = df['Confidence'].iloc[idx]
        passes_filter = df['Passes_Filter'].iloc[idx]
        
        # Apply cash interest
        cash *= (1 + shv_daily_return)
        
        # Execute trades
        if signal == 'BUY' and passes_filter:
            # Determine position size based on confidence
            if confidence > 0.8:
                pos_size = position_sizes['high']
            elif confidence > 0.65:
                pos_size = position_sizes['medium']
            elif confidence > 0.5:
                pos_size = position_sizes['low']
            else:
                pos_size = position_sizes['very_low']
            
            # Calculate shares to buy
            cost = cash * pos_size
            shares_to_buy = int(cost / price)
            
            if shares_to_buy > 0 and cost <= cash * 2.0:  # Allow 2x leverage
                shares += shares_to_buy
                cash -= shares_to_buy * price
        
        elif signal == 'SELL' and shares > 0:
            # Exit trades (but we ignore SELL signals based on optimization)
            pass
        
        # Calculate portfolio value
        portfolio_value = cash + (shares * price)
        portfolio_values.append(portfolio_value)
        
        results.append({
            'Date': date,
            'Close': price,
            'Signal': signal,
            'Confidence': confidence,
            'Passes_Filter': passes_filter,
            'Shares': shares,
            'Cash': cash,
            'Portfolio_Value': portfolio_value
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    total_return = (portfolio_values[-1] / initial_capital - 1) * 100
    max_drawdown = (min(portfolio_values) - max(portfolio_values)) / max(portfolio_values) * 100
    daily_returns = results_df['Portfolio_Value'].pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    
    num_buy_signals = (results_df['Signal'] == 'BUY').sum()
    num_accepted = results_df['Passes_Filter'].sum()
    num_filtered = num_buy_signals - num_accepted
    
    return results_df, {
        'total_return': total_return,
        'final_value': portfolio_values[-1],
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'num_buy_signals': num_buy_signals,
        'num_accepted': num_accepted,
        'num_filtered': num_filtered,
        'final_shares': shares,
        'final_cash': cash,
        'daily_win_rate': (daily_returns > 0).sum() / len(daily_returns) * 100
    }

def main():
    """Main execution"""
    print("="*80)
    print("TESTING OPTIMIZED STRATEGY ON DIFFERENT STOCK")
    print("="*80)
    print()
    
    # Test on Microsoft (MSFT) for 2024
    ticker = "MSFT"
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    initial_capital = 100000
    
    print(f"📊 Testing on: {ticker} ({start_date} to {end_date})")
    print(f"💰 Initial Capital: ${initial_capital:,.0f}")
    print()
    
    # Download data
    data = download_stock_data(ticker, start_date, end_date)
    if data is None:
        print("Failed to download data")
        return
    
    print(f"Generating trading signals...")
    signals_df = calculate_simple_signals(data)
    
    print(f"Applying optimized filters...")
    signals_df = apply_optimized_filters(signals_df)
    
    print(f"Running backtest...")
    results_df, metrics = run_backtest(signals_df, initial_capital)
    
    # Display results
    print()
    print("="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    print()
    print(f"Strategy: {ticker} with Optimized Filters (MA50 + Conf>0.85 + 1.3x Leverage)")
    print()
    print(f"  Initial Capital:        ${initial_capital:,.2f}")
    print(f"  Final Portfolio Value:  ${metrics['final_value']:,.2f}")
    print(f"  Total Return:           {metrics['total_return']:.2f}%")
    print(f"  Profit/Loss:            ${metrics['final_value'] - initial_capital:,.2f}")
    print()
    print(f"RISK METRICS:")
    print(f"  Maximum Drawdown:       {metrics['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio:           {metrics['sharpe_ratio']:.3f}")
    print(f"  Daily Win Rate:         {metrics['daily_win_rate']:.1f}%")
    print()
    print(f"TRADING ACTIVITY:")
    print(f"  BUY Signals Generated:  {metrics['num_buy_signals']}")
    print(f"  Signals Accepted:       {metrics['num_accepted']}")
    print(f"  Signals Filtered Out:   {metrics['num_filtered']}")
    print(f"  Filter Rate:            {(metrics['num_filtered']/metrics['num_buy_signals']*100):.1f}%")
    print()
    print(f"POSITION:")
    print(f"  Final Shares Held:      {metrics['final_shares']}")
    print(f"  Final Cash:             ${metrics['final_cash']:,.2f}")
    print()
    
    # Compare with baseline (simple buy and hold)
    print("="*80)
    print("COMPARISON WITH BUY & HOLD BASELINE")
    print("="*80)
    print()
    
    buyhold_value = initial_capital * (signals_df['Close'].iloc[-1] / signals_df['Close'].iloc[200])
    buyhold_return = (buyhold_value / initial_capital - 1) * 100
    
    print(f"Buy & Hold Return:      {buyhold_return:.2f}%")
    print(f"Optimized Return:       {metrics['total_return']:.2f}%")
    print(f"Improvement:            {metrics['total_return'] - buyhold_return:+.2f}pp")
    if metrics['total_return'] > buyhold_return:
        improvement_pct = ((metrics['total_return'] / buyhold_return) - 1) * 100
        print(f"Improvement %:          +{improvement_pct:.1f}%")
    print()
    
    # Save results
    results_df.to_csv(f'results/{ticker}_2024_backtest.csv', index=False)
    print(f"✓ Results saved to: results/{ticker}_2024_backtest.csv")
    print()
    
    # Summary comparison
    print("="*80)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("="*80)
    print()
    print(f"{'Stock':<10} {'Period':<20} {'Return':<12} {'Drawdown':<12} {'Sharpe':<10}")
    print("-" * 64)
    print(f"{'Original':<10} {'2025 (SPY)':<20} {'21.64%':<12} {'-18.13%':<12} {'1.10':<10}")
    print(f"{ticker:<10} {'2024':<20} {metrics['total_return']:>10.2f}% {metrics['max_drawdown']:>10.2f}% {metrics['sharpe_ratio']:>8.3f}")
    print()
    
    if metrics['total_return'] > 15:
        print("✅ Strategy shows strong performance on different stock/timeframe!")
    elif metrics['total_return'] > 5:
        print("⚠️  Strategy shows positive but lower performance on this stock")
    else:
        print("❌ Strategy underperforms on this stock - may need adjustment")
    
    print()

if __name__ == '__main__':
    main()
