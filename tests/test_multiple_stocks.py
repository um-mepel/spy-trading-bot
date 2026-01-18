#!/usr/bin/env python3
"""
Multi-Stock Strategy Testing Framework
Tests the optimized trading strategy across multiple stocks and time periods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

# Set plotting style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

@dataclass
class StockTestConfig:
    """Configuration for a stock test"""
    symbol: str
    year: int
    base_price: float
    volatility: float
    trend: float
    sector: str

# Define stocks to test across different sectors
TEST_CONFIGS = [
    # Tech
    StockTestConfig("AAPL", 2024, 182, 0.018, 0.15, "Technology"),
    StockTestConfig("GOOGL", 2024, 141, 0.016, 0.12, "Technology"),
    StockTestConfig("MSFT", 2024, 416, 0.014, 0.18, "Technology"),
    
    # Healthcare
    StockTestConfig("JNJ", 2024, 160, 0.012, 0.08, "Healthcare"),
    StockTestConfig("UNH", 2024, 425, 0.015, 0.20, "Healthcare"),
    
    # Finance
    StockTestConfig("JPM", 2024, 180, 0.017, 0.22, "Finance"),
    StockTestConfig("BAC", 2024, 38, 0.020, 0.25, "Finance"),
    
    # Consumer
    StockTestConfig("AMZN", 2024, 190, 0.019, 0.35, "Consumer"),
    StockTestConfig("WMT", 2024, 88, 0.013, 0.10, "Consumer"),
    
    # Industrial
    StockTestConfig("BA", 2024, 174, 0.021, -0.05, "Industrial"),
    StockTestConfig("CAT", 2024, 320, 0.017, 0.28, "Industrial"),
]

def generate_realistic_stock_data(config: StockTestConfig) -> pd.DataFrame:
    """Generate realistic stock price data with trend and seasonality"""
    trading_days = 252  # Standard trading days per year
    dates = pd.date_range(start=f'{config.year}-01-01', periods=trading_days, freq='B')
    
    # Create price path with trend and mean reversion
    returns = np.random.normal(config.trend / 252, config.volatility / np.sqrt(252), trading_days)
    
    # Add mean reversion component
    prices = [config.base_price]
    for i, ret in enumerate(returns[1:], 1):
        # Add seasonality (slight dip in summer, rise in Q4)
        seasonality = 0.002 * np.sin(2 * np.pi * i / 252)
        
        # Add momentum (trending component)
        momentum = 0.0001 * config.trend
        
        # Mean reversion to base price
        mean_reversion = 0.0002 * (config.base_price - prices[-1]) / config.base_price
        
        new_price = prices[-1] * (1 + ret + seasonality + momentum + mean_reversion)
        prices.append(max(new_price, prices[-1] * 0.95))  # Prevent crashes
    
    df = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Symbol': config.symbol,
        'Year': config.year,
        'Sector': config.sector
    })
    
    # Add volume and other fields
    df['Volume'] = np.random.randint(30000000, 100000000, len(df))
    df['High'] = df['Close'] * (1 + np.abs(np.random.normal(0, 0.01, len(df))))
    df['Low'] = df['Close'] * (1 - np.abs(np.random.normal(0, 0.01, len(df))))
    df['Open'] = df['Close'].shift(1).fillna(df['Close'].iloc[0])
    
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)
    
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    return df

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate BUY/SELL signals based on technical analysis"""
    df['Signal'] = 0
    
    for i in range(50, len(df)):
        rsi = df['RSI'].iloc[i]
        price = df['Close'].iloc[i]
        ma50 = df['MA50'].iloc[i]
        ma20 = df['MA20'].iloc[i]
        macd_hist = df['MACD_Hist'].iloc[i]
        
        # BUY signal conditions
        buy_conditions = [
            rsi < 70,  # Not overbought
            rsi > 30,  # Not oversold
            price > ma50,  # Above 50-day MA
            ma20 > ma50,  # Uptrend
            macd_hist > 0,  # MACD positive
        ]
        
        if sum(buy_conditions) >= 4:  # At least 4 conditions met
            df.loc[df.index[i], 'Signal'] = 1  # BUY
        elif rsi > 75:  # SELL signal
            df.loc[df.index[i], 'Signal'] = -1
    
    return df

def calculate_signal_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate confidence score for each signal"""
    df['Confidence'] = 0.5
    
    for i in range(50, len(df)):
        if df['Signal'].iloc[i] == 0:
            continue
            
        rsi = df['RSI'].iloc[i]
        price = df['Close'].iloc[i]
        ma50 = df['MA50'].iloc[i]
        ma200 = df['MA200'].iloc[i]
        
        # Calculate confidence based on multiple factors
        confidence = 0.5
        
        # RSI strength
        if 40 < rsi < 60:
            confidence += 0.15
        elif 30 < rsi < 70:
            confidence += 0.10
        
        # Price position
        if price > ma50:
            confidence += 0.15
        if ma50 > ma200:
            confidence += 0.15
        
        # Trend strength
        if ma50 > ma200 * 1.02:
            confidence += 0.1
        
        # Add randomness for realism
        confidence += np.random.normal(0, 0.05)
        confidence = np.clip(confidence, 0.4, 0.95)
        
        df.loc[df.index[i], 'Confidence'] = confidence
    
    return df

def run_backtest(df: pd.DataFrame, confidence_threshold: float = 0.70, initial_capital: float = 100000) -> Dict:
    """Run backtest with specified parameters"""
    
    cash = initial_capital
    shares = 0
    portfolio_values = [initial_capital]
    trades = []
    trades_accepted = 0
    max_drawdown = 0
    peak_value = initial_capital
    
    position_sizes = [0.90, 0.70, 0.50, 0.20]
    leverage = 1.3
    
    entry_price = None
    entry_date = None
    entry_index = None
    
    for i in range(50, len(df)):
        current_price = df['Close'].iloc[i]
        current_date = df['Date'].iloc[i]
        signal = df['Signal'].iloc[i]
        confidence = df['Confidence'].iloc[i]
        ma50 = df['MA50'].iloc[i]
        
        # Check for exit condition
        if shares > 0:
            # Exit if price drops below MA50
            if current_price < ma50:
                cash += shares * current_price
                profit = (current_price - entry_price) * shares
                profit_pct = (profit / (entry_price * shares)) * 100 if entry_price else 0
                trades.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': current_date,
                    'Entry_Price': entry_price,
                    'Exit_Price': current_price,
                    'Shares': shares,
                    'Profit': profit,
                    'Profit_Pct': profit_pct,
                    'Days_Held': i - entry_index if entry_index else 0
                })
                shares = 0
                entry_price = None
            # Or take profit at 3% gain
            elif current_price >= entry_price * 1.03 and entry_price:
                cash += shares * current_price
                profit = (current_price - entry_price) * shares
                profit_pct = 3.0
                trades.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': current_date,
                    'Entry_Price': entry_price,
                    'Exit_Price': current_price,
                    'Shares': shares,
                    'Profit': profit,
                    'Profit_Pct': profit_pct,
                    'Days_Held': i - entry_index if entry_index else 0
                })
                shares = 0
                entry_price = None
        
        # Check for entry signal
        if signal == 1 and shares == 0 and confidence > confidence_threshold and current_price > ma50:
            # Determine position size based on confidence
            if confidence > 0.85:
                position_size = position_sizes[0]
            elif confidence > 0.75:
                position_size = position_sizes[1]
            elif confidence > 0.65:
                position_size = position_sizes[2]
            else:
                position_size = position_sizes[3]
            
            # Calculate shares with leverage
            allocation = cash * position_size * leverage
            shares = allocation / current_price
            cash -= allocation
            entry_price = current_price
            entry_date = current_date
            entry_index = i
            trades_accepted += 1
        
        # Calculate portfolio value
        portfolio_value = cash + (shares * current_price if shares > 0 else 0)
        portfolio_values.append(portfolio_value)
        
        # Track maximum drawdown
        if portfolio_value > peak_value:
            peak_value = portfolio_value
        drawdown = (peak_value - portfolio_value) / peak_value
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Close any open position
    if shares > 0:
        cash += shares * df['Close'].iloc[-1]
        shares = 0
    
    final_value = cash
    portfolio_values[-1] = final_value
    
    # Calculate metrics
    total_return = (final_value - initial_capital) / initial_capital
    
    # Buy & hold baseline
    entry_price_bh = df['Close'].iloc[50]
    exit_price_bh = df['Close'].iloc[-1]
    bh_return = (exit_price_bh - entry_price_bh) / entry_price_bh
    
    # Sharpe ratio
    returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
    daily_return = np.mean(returns)
    daily_std = np.std(returns)
    sharpe = (daily_return * 252) / (daily_std * np.sqrt(252)) if daily_std > 0 else 0
    
    return {
        'Symbol': df['Symbol'].iloc[0],
        'Year': df['Year'].iloc[0],
        'Sector': df['Sector'].iloc[0],
        'Threshold': confidence_threshold,
        'Initial_Capital': initial_capital,
        'Final_Value': final_value,
        'Total_Return': total_return,
        'B&H_Return': bh_return,
        'Outperformance': total_return - bh_return,
        'Max_Drawdown': -max_drawdown,
        'Sharpe_Ratio': sharpe,
        'Trades_Executed': trades_accepted,
        'Num_Trades': len(trades),
        'Win_Rate': len([t for t in trades if t['Profit'] > 0]) / len(trades) if trades else 0,
    }

def test_stock_with_thresholds(config: StockTestConfig, thresholds: List[float]) -> List[Dict]:
    """Test a stock with multiple confidence thresholds"""
    print(f"\nTesting {config.symbol} ({config.sector}) - {config.year}...")
    
    # Generate data
    df = generate_realistic_stock_data(config)
    df = calculate_indicators(df)
    df = generate_signals(df)
    df = calculate_signal_confidence(df)
    
    # Test with different thresholds
    results = []
    for threshold in thresholds:
        result = run_backtest(df, confidence_threshold=threshold)
        results.append(result)
        print(f"  Threshold {threshold:.2f}: {result['Total_Return']:.2%} return, "
              f"{result['Trades_Executed']} trades, Sharpe: {result['Sharpe_Ratio']:.2f}")
    
    return results

def main():
    """Run comprehensive multi-stock testing"""
    print("=" * 80)
    print("MULTI-STOCK STRATEGY TESTING")
    print("=" * 80)
    
    # Test with multiple thresholds to find optimal for each stock
    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    all_results = []
    
    for config in TEST_CONFIGS:
        results = test_stock_with_thresholds(config, thresholds)
        all_results.extend(results)
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    # Save detailed results
    results_df.to_csv('results/multi_stock_backtest.csv', index=False)
    print("\n" + "=" * 80)
    print(f"Tested {len(TEST_CONFIGS)} stocks with {len(thresholds)} thresholds")
    print(f"Total configurations: {len(results_df)}")
    print("Results saved to results/multi_stock_backtest.csv")
    
    # Find optimal threshold per stock
    print("\n" + "=" * 80)
    print("OPTIMAL THRESHOLD PER STOCK")
    print("=" * 80)
    
    optimal_by_stock = results_df.loc[results_df.groupby('Symbol')['Total_Return'].idxmax()]
    optimal_by_stock = optimal_by_stock.sort_values('Total_Return', ascending=False)
    
    print("\nTop Performers:")
    for idx, row in optimal_by_stock.head(10).iterrows():
        print(f"{row['Symbol']:6} ({row['Sector']:12}) - {row['Total_Return']:6.2%} return "
              f"(threshold: {row['Threshold']:.2f}, B&H: {row['B&H_Return']:.2%}, "
              f"outperformance: {row['Outperformance']:6.2%})")
    
    # Analysis by sector
    print("\n" + "=" * 80)
    print("PERFORMANCE BY SECTOR (Average Best Return Per Stock)")
    print("=" * 80)
    
    sector_performance = optimal_by_stock.groupby('Sector').agg({
        'Total_Return': ['mean', 'min', 'max'],
        'Outperformance': 'mean',
        'Sharpe_Ratio': 'mean',
        'Max_Drawdown': 'mean'
    }).round(4)
    
    print(sector_performance)
    sector_performance.to_csv('results/sector_performance.csv')
    
    # Statistical summary
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS (All Configurations)")
    print("=" * 80)
    
    print(f"Average Return: {results_df['Total_Return'].mean():.2%}")
    print(f"Median Return: {results_df['Total_Return'].median():.2%}")
    print(f"Std Dev Return: {results_df['Total_Return'].std():.2%}")
    print(f"Min Return: {results_df['Total_Return'].min():.2%}")
    print(f"Max Return: {results_df['Total_Return'].max():.2%}")
    print(f"Avg B&H Return: {results_df['B&H_Return'].mean():.2%}")
    print(f"Avg Outperformance: {results_df['Outperformance'].mean():.2%}")
    print(f"Avg Sharpe Ratio: {results_df['Sharpe_Ratio'].mean():.2f}")
    print(f"Avg Max Drawdown: {results_df['Max_Drawdown'].mean():.2%}")
    
    # Threshold analysis
    print("\n" + "=" * 80)
    print("ANALYSIS BY CONFIDENCE THRESHOLD")
    print("=" * 80)
    
    threshold_analysis = results_df.groupby('Threshold').agg({
        'Total_Return': ['mean', 'std'],
        'Outperformance': 'mean',
        'Sharpe_Ratio': 'mean',
        'Trades_Executed': 'mean'
    }).round(4)
    
    print(threshold_analysis)
    threshold_analysis.to_csv('results/threshold_analysis.csv')
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 80)
    
    # 1. Return distribution by stock
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Returns by symbol
    ax1 = axes[0, 0]
    optimal_by_stock_sorted = optimal_by_stock.sort_values('Total_Return', ascending=False).head(15)
    colors = ['green' if x > y else 'red' for x, y in 
              zip(optimal_by_stock_sorted['Total_Return'], optimal_by_stock_sorted['B&H_Return'])]
    ax1.barh(optimal_by_stock_sorted['Symbol'], optimal_by_stock_sorted['Total_Return'], color=colors, alpha=0.7)
    ax1.axvline(optimal_by_stock_sorted['B&H_Return'].mean(), color='blue', linestyle='--', label='Avg B&H')
    ax1.set_xlabel('Return (%)')
    ax1.set_title('Top 15 Stocks - Strategy Return vs Buy & Hold')
    ax1.legend()
    
    # Outperformance by sector
    ax2 = axes[0, 1]
    sector_outperf = optimal_by_stock.groupby('Sector')['Outperformance'].mean().sort_values()
    sector_outperf.plot(kind='barh', ax=ax2, color='steelblue')
    ax2.set_xlabel('Average Outperformance (%)')
    ax2.set_title('Average Outperformance by Sector')
    ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    # Sharpe Ratio distribution
    ax3 = axes[1, 0]
    ax3.hist(results_df['Sharpe_Ratio'], bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(results_df['Sharpe_Ratio'].mean(), color='red', linestyle='--', label='Mean')
    ax3.set_xlabel('Sharpe Ratio')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Sharpe Ratios')
    ax3.legend()
    
    # Threshold performance
    ax4 = axes[1, 1]
    threshold_perf = results_df.groupby('Threshold')['Total_Return'].mean()
    ax4.plot(threshold_perf.index, threshold_perf.values, marker='o', linewidth=2, markersize=8, color='darkgreen')
    ax4.fill_between(threshold_perf.index, threshold_perf.values, alpha=0.3, color='green')
    ax4.set_xlabel('Confidence Threshold')
    ax4.set_ylabel('Average Return (%)')
    ax4.set_title('Average Return by Confidence Threshold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/multi_stock_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved multi_stock_analysis.png")
    plt.close()
    
    # 2. Heatmap of returns by stock and threshold
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_table = results_df.pivot_table(values='Total_Return', index='Symbol', columns='Threshold')
    sns.heatmap(pivot_table, annot=True, fmt='.2%', cmap='RdYlGn', center=0.15, ax=ax, cbar_kws={'label': 'Return'})
    ax.set_title('Strategy Return by Stock and Confidence Threshold')
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Stock Symbol')
    plt.tight_layout()
    plt.savefig('results/return_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved return_heatmap.png")
    plt.close()
    
    # 3. Scatter plot: Return vs Drawdown
    fig, ax = plt.subplots(figsize=(12, 8))
    sectors = optimal_by_stock['Sector'].unique()
    colors_map = plt.cm.tab10(np.linspace(0, 1, len(sectors)))
    
    for sector, color in zip(sectors, colors_map):
        sector_data = optimal_by_stock[optimal_by_stock['Sector'] == sector]
        ax.scatter(sector_data['Max_Drawdown'], sector_data['Total_Return'], 
                  s=100, alpha=0.6, label=sector, color=color)
    
    ax.set_xlabel('Maximum Drawdown (%)')
    ax.set_ylabel('Total Return (%)')
    ax.set_title('Risk-Return Profile by Sector (Optimal Threshold per Stock)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add efficient frontier line
    ax.axhline(0.15, color='red', linestyle='--', alpha=0.5, label='15% Return Target')
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('results/risk_return_profile.png', dpi=300, bbox_inches='tight')
    print("✓ Saved risk_return_profile.png")
    plt.close()
    
    print("\n" + "=" * 80)
    print("MULTI-STOCK TESTING COMPLETE")
    print("=" * 80)
    print(f"✓ Tested {len(TEST_CONFIGS)} stocks")
    print(f"✓ Generated {len(results_df)} configurations")
    print(f"✓ Created 3 visualizations")
    print(f"✓ Generated detailed CSV reports")
    
    return results_df, optimal_by_stock

if __name__ == "__main__":
    results_df, optimal_by_stock = main()
