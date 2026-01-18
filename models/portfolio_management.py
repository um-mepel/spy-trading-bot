"""
Portfolio Management and Backtesting Module
Simulates trading based on signals and tracks portfolio performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def backtest_portfolio(signals_df, initial_capital=100000, units_per_signal=1000, shv_return_daily=0.00015, confidence_weighted=True, exit_model_df=None):
    """
    Backtest a trading strategy based on signals.
    DUAL-MODEL: Entry model (BUY) + Exit model (DROP probability)
    
    Strategy:
    - Start with initial_capital
    - Each day: 
      - If BUY signal: 
        - VERY HIGH CONFIDENCE (>0.8): allocate 75% of available cash (aggressive)
        - HIGH CONFIDENCE (0.65-0.8): allocate 50% of available cash
        - MEDIUM CONFIDENCE (0.5-0.65): allocate 30% of available cash
        - LOW CONFIDENCE (<0.5): allocate 10% of available cash (minimal)
      - If EXIT MODEL shows high drop probability (>0.7): Exit positions (profit-taking)
      - Otherwise: Hold positions
    - Dead cash earns SHV (short-duration Treasury ETF) returns (~5.5% annually ≈ 0.015% daily)
    
    Args:
        signals_df: DataFrame with entry signals
        exit_model_df: DataFrame with exit model drop probabilities (optional)
        initial_capital: Starting cash (default $100,000)
        units_per_signal: NOT USED - portfolio uses percentage-based allocation
        shv_return_daily: Daily return for cash in SHV (default 0.015% ≈ 5.5% annual)
        confidence_weighted: If True, size positions based on confidence (default True)
        
    Returns:
        DataFrame with daily portfolio state and performance metrics
    """
    results_df = signals_df.copy()
    
    # Merge exit model predictions if provided
    if exit_model_df is not None:
        results_df = results_df.merge(
            exit_model_df[['Date', 'Drop_Probability', 'Exit_Signal_Strength']], 
            on='Date', 
            how='left'
        )
        results_df['Drop_Probability'] = results_df['Drop_Probability'].fillna(0.5)  # Default to neutral
    else:
        results_df['Drop_Probability'] = 0.5  # No exit signals if model not provided
    
    # Initialize portfolio state
    results_df['Cash'] = 0.0
    results_df['Shares_Held'] = 0.0
    results_df['Portfolio_Value'] = 0.0
    results_df['Daily_Return'] = 0.0
    results_df['Cumulative_Return'] = 0.0
    results_df['SHV_Earnings'] = 0.0
    results_df['Trade_Size_%'] = 0.0
    
    cash = initial_capital
    shares_held = 0.0
    prev_portfolio_value = initial_capital
    
    for idx, row in results_df.iterrows():
        # Get today's price and signal
        price = row['Actual_Price']
        signal = row['Signal']
        confidence = row.get('Confidence', 0.5)  # Default to 0.5 if not present
        
        # Cash earns SHV returns
        shv_earning = cash * shv_return_daily
        cash += shv_earning
        
        # Execute trade based on signal with confidence-weighted sizing
        if signal == 'BUY' or signal == 1:  # BUY
            # Determine position size based on confidence - AGGRESSIVE SIZING FOR RETURNS
            if confidence_weighted:
                if confidence > 0.8:  # Very high confidence
                    position_size = 0.90  # 90% of available cash (very aggressive)
                elif confidence > 0.65:  # High confidence
                    position_size = 0.70  # 70% of available cash
                elif confidence > 0.5:   # Medium confidence
                    position_size = 0.50  # 50% of available cash
                else:  # Low confidence
                    position_size = 0.20  # 20% of available cash (still take some exposure)
            else:
                position_size = 0.5  # Default 50%
            
            available_for_buying = cash * position_size
            shares_to_buy = int(available_for_buying / price)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                cash -= cost
                shares_held += shares_to_buy
                results_df.loc[idx, 'Trade_Size_%'] = position_size * 100
        
        elif signal == 'SELL' or signal == -1:  # SELL - IGNORED (don't exit on SELL signals)
            pass  # Ignore SELL signals - just hold positions
        
        # EXIT MODEL CHECK: If exit model predicts very high drop probability, exit positions
        if exit_model_df is not None and shares_held > 0:
            drop_prob = row.get('Drop_Probability', 0.5)
            
            # Exit only if model says drop probability is VERY HIGH (>0.85)
            # This prioritizes returns: let trades run, exit only on extreme reversals
            if drop_prob > 0.85:
                proceeds = shares_held * price
                cash += proceeds
                results_df.loc[idx, 'Trade_Size_%'] = -100.0
                shares_held = 0
        # else: signal == 'HOLD' or 0 - do nothing
        
        # Calculate portfolio value
        portfolio_value = cash + (shares_held * price)
        
        # Calculate daily return
        daily_return = portfolio_value - prev_portfolio_value
        
        # Store results
        results_df.loc[idx, 'Cash'] = cash
        results_df.loc[idx, 'Shares_Held'] = shares_held
        results_df.loc[idx, 'Portfolio_Value'] = portfolio_value
        results_df.loc[idx, 'Daily_Return'] = daily_return
        results_df.loc[idx, 'SHV_Earnings'] = shv_earning
        
        prev_portfolio_value = portfolio_value
    
    # Calculate cumulative returns
    results_df['Cumulative_Return'] = (results_df['Portfolio_Value'] - initial_capital) / initial_capital * 100
    
    return results_df


def calculate_performance_metrics(backtest_df, initial_capital=100000):
    """
    Calculate performance metrics for the backtest.
    
    Args:
        backtest_df: DataFrame with backtest results
        initial_capital: Starting capital
        
    Returns:
        Dictionary with performance metrics
    """
    total_return = backtest_df['Portfolio_Value'].iloc[-1] - initial_capital
    total_return_pct = (total_return / initial_capital) * 100
    
    # Winning days
    winning_days = (backtest_df['Daily_Return'] > 0).sum()
    total_days = len(backtest_df)
    win_rate = (winning_days / total_days) * 100
    
    # Drawdown
    cummax = backtest_df['Portfolio_Value'].cummax()
    drawdown = (backtest_df['Portfolio_Value'] - cummax) / cummax * 100
    max_drawdown = drawdown.min()
    
    # Volatility (calculate as % of portfolio value)
    daily_returns_pct = (backtest_df['Daily_Return'] / backtest_df['Portfolio_Value'].shift(1)) * 100
    volatility = daily_returns_pct.std()
    
    # Sharpe ratio (assuming 0% risk-free rate for simplicity)
    sharpe_ratio = (daily_returns_pct.mean() / volatility * np.sqrt(252)) if volatility > 0 else 0
    
    metrics = {
        'Total_Return_$': round(total_return, 2),
        'Total_Return_%': round(total_return_pct, 2),
        'Winning_Days': winning_days,
        'Total_Days': total_days,
        'Win_Rate_%': round(win_rate, 2),
        'Max_Drawdown_%': round(max_drawdown, 2),
        'Daily_Volatility_%': round(volatility, 2),
        'Sharpe_Ratio': round(sharpe_ratio, 2),
        'Final_Portfolio_Value': round(backtest_df['Portfolio_Value'].iloc[-1], 2),
        'Max_Portfolio_Value': round(backtest_df['Portfolio_Value'].max(), 2),
        'Min_Portfolio_Value': round(backtest_df['Portfolio_Value'].min(), 2)
    }
    
    return metrics


def main(signals_df, results_dir=None, initial_capital=100000, units_per_signal=1000, confidence_weighted=True, exit_model_df=None):
    """
    Main function to run portfolio backtest.
    DUAL-MODEL: Entry signals (BUY) + Exit model (DROP probability)
    
    Args:
        signals_df: DataFrame with trading signals from signal_generation.py
        exit_model_df: DataFrame with exit model drop probabilities (optional)
        results_dir: Directory to save backtest results CSV (optional)
        initial_capital: Starting capital (default $100,000)
        units_per_signal: Shares to trade per signal unit (default 1000)
        confidence_weighted: Size positions based on confidence (default True)
        
    Returns:
        Dictionary with:
            - 'backtest_df': Full backtest results
            - 'metrics': Performance metrics dictionary
    """
    print("\n" + "="*60)
    print("Portfolio Backtesting (OPTIMIZED - Buy & Hold Strategy)")
    print("="*60 + "\n")
    
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Strategy: RETURN-FOCUSED (Aggressive positioning, hold through volatility)")
    print(f"Position Sizing: CONFIDENCE-WEIGHTED (aggressive)")
    print(f"  Very High Confidence (>0.8): 90% of available cash (maximum exposure)")
    print(f"  High Confidence (0.65-0.8): 70% of available cash")
    print(f"  Medium Confidence (0.5-0.65): 50% of available cash")
    print(f"  Low Confidence (<0.5): 20% of available cash (still trade)")
    print(f"Exit Strategy: EXIT MODEL (only exits on extreme reversals >85% probability)")
    print(f"              (Prioritizes returns over risk control)\n")
    
    # Ensure Date column is datetime
    signals_df = signals_df.copy()
    signals_df['Date'] = pd.to_datetime(signals_df['Date'])
    
    if exit_model_df is not None:
        exit_model_df = exit_model_df.copy()
        exit_model_df['Date'] = pd.to_datetime(exit_model_df['Date'])
    
    # Run backtest with confidence weighting and exit model
    backtest_df = backtest_portfolio(signals_df, initial_capital, units_per_signal, confidence_weighted=confidence_weighted, exit_model_df=exit_model_df)
    
    # Calculate metrics
    metrics = calculate_performance_metrics(backtest_df, initial_capital)
    
    # Print summary
    print("Performance Metrics:")
    print(f"  Final Portfolio Value: ${metrics['Final_Portfolio_Value']:,.2f}")
    print(f"  Total Return: ${metrics['Total_Return_$']:,.2f} ({metrics['Total_Return_%']:.2f}%)")
    print(f"  Win Rate: {metrics['Win_Rate_%']:.2f}%")
    print(f"  Max Drawdown: {metrics['Max_Drawdown_%']:.2f}%")
    print(f"  Daily Volatility: {metrics['Daily_Volatility_%']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['Sharpe_Ratio']:.2f}")
    print(f"  Max Portfolio Value: ${metrics['Max_Portfolio_Value']:,.2f}")
    print(f"  Min Portfolio Value: ${metrics['Min_Portfolio_Value']:,.2f}")
    
    print("\n" + "="*60)
    
    # Save backtest results to CSV if directory provided
    if results_dir:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        output_file = results_dir / "portfolio_backtest.csv"
        
        # Select relevant columns for output
        output_cols = ['Date', 'Actual_Price', 'Signal', 'Shares_Held', 'Cash', 
                      'Portfolio_Value', 'Daily_Return', 'Cumulative_Return']
        available_cols = [col for col in output_cols if col in backtest_df.columns]
        backtest_df[available_cols].to_csv(output_file, index=False)
        print(f"Portfolio backtest saved to {output_file.name}\n")
    
    return {
        'backtest_df': backtest_df,
        'metrics': metrics
    }


if __name__ == "__main__":
    print("Portfolio Management Module")
    print("This module is designed to be imported by main.py")
