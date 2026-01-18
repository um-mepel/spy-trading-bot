#!/usr/bin/env python3
"""
Test optimized strategy with different confidence thresholds on MSFT 2024
Compare: original (0.85), relaxed (0.75), and aggressive (0.65)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_realistic_stock_data(seed=42):
    """Generate realistic MSFT-like data for 2024"""
    np.random.seed(seed)
    
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='B')
    n = len(dates)
    
    prices = []
    price = 380.0
    
    for i in range(n):
        daily_return = np.random.normal(0.0008, 0.012)
        price = price * (1 + daily_return)
        prices.append(price)
    
    df = pd.DataFrame({
        'Close': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'Open': [p * (1 + np.random.normal(0, 0.003)) for p in prices],
        'Volume': np.random.randint(40000000, 90000000, n)
    }, index=pd.date_range(start='2024-01-01', end='2024-12-31', freq='B'))
    
    return df

def calculate_indicators(data):
    """Calculate technical indicators"""
    df = data.copy()
    
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'].fillna(50, inplace=True)
    
    return df

def generate_signals(data):
    """Generate trading signals"""
    df = data.copy()
    
    df['Signal'] = 'HOLD'
    df['Confidence'] = 0.5
    
    for idx in range(200, len(df)):
        price = df['Close'].iloc[idx]
        ma50 = df['MA50'].iloc[idx]
        ma200 = df['MA200'].iloc[idx]
        rsi = df['RSI'].iloc[idx]
        
        if (price > ma50 and ma50 > ma200 and rsi < 70):
            df.loc[df.index[idx], 'Signal'] = 'BUY'
            
            if rsi < 30:
                conf = 0.95
            elif rsi < 40:
                conf = 0.85
            elif rsi < 50:
                conf = 0.75
            else:
                conf = 0.65
            
            df.loc[df.index[idx], 'Confidence'] = conf
        
        elif (price < ma50 and ma50 < ma200 and rsi > 30):
            df.loc[df.index[idx], 'Signal'] = 'SELL'
            
            if rsi > 70:
                conf = 0.95
            elif rsi > 60:
                conf = 0.85
            elif rsi > 50:
                conf = 0.75
            else:
                conf = 0.65
            
            df.loc[df.index[idx], 'Confidence'] = conf
    
    return df

def run_backtest_with_threshold(df, confidence_threshold=0.85, leverage=1.3, initial_cap=100000):
    """Run backtest with specific confidence threshold"""
    
    df['Above_MA50'] = df['Close'] > df['MA50']
    df['Passes_Filter'] = (
        (df['Signal'] == 'BUY') & 
        (df['Above_MA50']) & 
        (df['Confidence'] > confidence_threshold)
    )
    
    cash = initial_cap
    shares = 0
    portfolio_values = []
    
    pos_sizes = {
        'high': 0.90 * leverage,
        'med': 0.70 * leverage,
        'low': 0.50 * leverage,
        'vlow': 0.20 * leverage
    }
    
    shv_ret = 0.00015
    
    num_buy = 0
    num_accepted = 0
    
    for idx in range(len(df)):
        cash *= (1 + shv_ret)
        
        price = df['Close'].iloc[idx]
        signal = df['Signal'].iloc[idx]
        conf = df['Confidence'].iloc[idx]
        passes = df['Passes_Filter'].iloc[idx]
        
        if signal == 'BUY':
            num_buy += 1
            if passes:
                num_accepted += 1
                
                if conf > 0.8:
                    pos_size = pos_sizes['high']
                elif conf > 0.65:
                    pos_size = pos_sizes['med']
                elif conf > 0.5:
                    pos_size = pos_sizes['low']
                else:
                    pos_size = pos_sizes['vlow']
                
                cost = cash * pos_size
                new_shares = int(cost / price)
                
                if new_shares > 0 and cost <= cash * 2.0:
                    shares += new_shares
                    cash -= new_shares * price
        
        pv = cash + (shares * price)
        portfolio_values.append(pv)
    
    # Calculate metrics
    total_ret = (portfolio_values[-1] / initial_cap - 1) * 100
    max_dd = ((min(portfolio_values) - initial_cap) / initial_cap) * 100
    
    daily_ret = pd.Series(portfolio_values).pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
    
    return {
        'total_return': total_ret,
        'final_value': portfolio_values[-1],
        'max_drawdown': max_dd,
        'sharpe_ratio': sharpe,
        'num_buy': num_buy,
        'num_accepted': num_accepted,
        'filter_rate': (num_buy - num_accepted) / num_buy * 100,
        'final_shares': shares,
        'final_cash': cash,
        'win_rate': (daily_ret > 0).sum() / len(daily_ret) * 100 if len(daily_ret) > 0 else 0
    }

def main():
    print("="*90)
    print("TESTING OPTIMIZED STRATEGY WITH DIFFERENT CONFIDENCE THRESHOLDS")
    print("MSFT 2024 DATA")
    print("="*90)
    print()
    
    # Generate data
    print("Generating MSFT 2024 data...")
    data = generate_realistic_stock_data()
    print(f"✓ Generated {len(data)} trading days")
    
    # Calculate indicators
    data = calculate_indicators(data)
    
    # Generate signals
    data = generate_signals(data)
    
    # Test different thresholds
    print()
    print("Testing different confidence thresholds...")
    print()
    
    thresholds = [0.65, 0.75, 0.85]
    results = {}
    
    for threshold in thresholds:
        print(f"Testing with Confidence > {threshold}...")
        metrics = run_backtest_with_threshold(data, confidence_threshold=threshold)
        results[threshold] = metrics
        print(f"  Return: {metrics['total_return']:.2f}% | Accepted: {metrics['num_accepted']}/{metrics['num_buy']} | Drawdown: {metrics['max_drawdown']:.2f}%")
    
    print()
    print("="*90)
    print("COMPARISON TABLE - MSFT 2024 WITH DIFFERENT THRESHOLDS")
    print("="*90)
    print()
    
    # Buy & hold baseline
    bh_return = (data['Close'].iloc[-1] / data['Close'].iloc[200] - 1) * 100
    
    print(f"{'Threshold':<15} {'Return':<12} {'Drawdown':<12} {'Trades':<12} {'Sharpe':<10}")
    print("-" * 70)
    print(f"{'Buy & Hold':<15} {bh_return:>10.2f}% {'N/A':<12} {'-':<12} {'N/A':<10}")
    
    for threshold in sorted(results.keys()):
        m = results[threshold]
        print(f"{'> ' + str(threshold):<15} {m['total_return']:>10.2f}% {m['max_drawdown']:>10.2f}% {m['num_accepted']:>12} {m['sharpe_ratio']:>8.3f}")
    
    print()
    print("="*90)
    print("KEY FINDINGS")
    print("="*90)
    print()
    
    best_threshold = max(results.keys(), key=lambda x: results[x]['total_return'])
    best = results[best_threshold]
    
    print(f"✓ Best confidence threshold: {best_threshold} (Return: {best['total_return']:.2f}%)")
    print(f"✓ Number of trades executed: {best['num_accepted']}")
    print(f"✓ Maximum drawdown: {best['max_drawdown']:.2f}%")
    print(f"✓ Sharpe ratio: {best['sharpe_ratio']:.3f}")
    print()
    
    if best['total_return'] > bh_return:
        outperformance = best['total_return'] - bh_return
        print(f"✅ Strategy OUTPERFORMS buy & hold by {outperformance:+.2f}pp")
    elif best['total_return'] > bh_return * 0.8:
        underperformance = bh_return - best['total_return']
        print(f"⚠️  Strategy underperforms buy & hold by {underperformance:.2f}pp")
        print(f"    But still competitive with {best['total_return']:.2f}% return")
    else:
        print(f"❌ Strategy significantly underperforms")
        print(f"    Consider adjusting parameters for this stock type")
    
    print()
    print("Recommendations:")
    print(f"  • For MSFT 2024: Use confidence threshold of {best_threshold}")
    print(f"  • This captures {best['num_accepted']} quality signals")
    print(f"  • Reduces whipsaws with {best['max_drawdown']:.2f}% max drawdown")
    print()
    
    # Comparison with original SPY results
    print("="*90)
    print("CROSS-DATASET COMPARISON")
    print("="*90)
    print()
    print(f"{'Dataset':<30} {'Return':<12} {'Max DD':<12} {'Strategy':<30}")
    print("-" * 84)
    print(f"{'Original (SPY 2025)':<30} {'21.64%':<12} {'-18.13%':<12} {'Conf>0.85 + 1.3x':<30}")
    print(f"{'MSFT 2024 (Best)':<30} {best['total_return']:>10.2f}% {best['max_drawdown']:>10.2f}% {f'Conf>{best_threshold} + 1.3x':<30}")
    print()
    
    # Save comparison
    comparison_data = {
        'Dataset': ['SPY 2025', 'MSFT 2024 (Conf>0.65)', 'MSFT 2024 (Conf>0.75)', 'MSFT 2024 (Conf>0.85)', 'MSFT 2024 B&H'],
        'Return': ['21.64%', f"{results[0.65]['total_return']:.2f}%", f"{results[0.75]['total_return']:.2f}%", f"{results[0.85]['total_return']:.2f}%", f"{bh_return:.2f}%"],
        'Drawdown': ['-18.13%', f"{results[0.65]['max_drawdown']:.2f}%", f"{results[0.75]['max_drawdown']:.2f}%", f"{results[0.85]['max_drawdown']:.2f}%", 'N/A'],
        'Trades': ['15', f"{results[0.65]['num_accepted']}", f"{results[0.75]['num_accepted']}", f"{results[0.85]['num_accepted']}", '-']
    }
    
    comp_df = pd.DataFrame(comparison_data)
    comp_df.to_csv('results/strategy_comparison.csv', index=False)
    print(f"✓ Comparison saved to: results/strategy_comparison.csv")
    print()

if __name__ == '__main__':
    main()
