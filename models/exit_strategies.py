"""
Advanced Exit Strategies for Trading
Tests multiple exit approaches:
1. Profit-taking rules (exit after X% gain)
2. Trailing stops (exit on X% pullback from recent high)
3. Mean reversion exits (exit when price reverts back to entry)
4. Volatility-based exits (tighter stops in high volatility)
"""

import pandas as pd
import numpy as np
from pathlib import Path


class ExitStrategies:
    """Container for multiple exit strategy implementations"""
    
    @staticmethod
    def backtest_with_profit_taking(signals_df, profit_target_pct=5.0, max_hold_days=20):
        """
        Exit strategy: Take profits at X% gain or exit after max holding period
        
        Args:
            signals_df: DataFrame with trading signals
            profit_target_pct: Target profit percentage (e.g., 5.0 for 5%)
            max_hold_days: Maximum days to hold a position
            
        Returns:
            DataFrame with backtest results
        """
        results_df = signals_df.copy()
        results_df['Cash'] = 0.0
        results_df['Shares_Held'] = 0.0
        results_df['Entry_Price'] = 0.0
        results_df['Entry_Date_Idx'] = -1
        results_df['Portfolio_Value'] = 0.0
        results_df['Daily_Return'] = 0.0
        
        cash = 100000.0
        shares_held = 0.0
        entry_price = 0.0
        entry_date_idx = -1
        prev_portfolio_value = 100000.0
        
        for idx, row in results_df.iterrows():
            price = row['Actual_Price']
            signal = row['Signal']
            confidence = row.get('Confidence', 0.5)
            
            # Check if we should exit (profit target or max hold)
            if shares_held > 0:
                profit_pct = (price - entry_price) / entry_price * 100
                hold_days = idx - entry_date_idx
                
                # Exit on profit target or max hold days
                if profit_pct >= profit_target_pct or hold_days >= max_hold_days:
                    proceeds = shares_held * price
                    cash += proceeds
                    shares_held = 0.0
                    entry_price = 0.0
            
            # Entry on BUY signal
            if signal == 'BUY' and shares_held == 0:
                if confidence > 0.8:
                    position_size = 0.90
                elif confidence > 0.65:
                    position_size = 0.70
                elif confidence > 0.5:
                    position_size = 0.50
                else:
                    position_size = 0.20
                
                available = cash * position_size
                shares_to_buy = int(available / price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    cash -= cost
                    shares_held = shares_to_buy
                    entry_price = price
                    entry_date_idx = idx
            
            # Calculate portfolio value
            portfolio_value = cash + (shares_held * price)
            daily_return = portfolio_value - prev_portfolio_value
            
            results_df.loc[idx, 'Cash'] = cash
            results_df.loc[idx, 'Shares_Held'] = shares_held
            results_df.loc[idx, 'Entry_Price'] = entry_price
            results_df.loc[idx, 'Portfolio_Value'] = portfolio_value
            results_df.loc[idx, 'Daily_Return'] = daily_return
            
            prev_portfolio_value = portfolio_value
        
        return results_df
    
    @staticmethod
    def backtest_with_trailing_stop(signals_df, trailing_stop_pct=3.0, lookback_days=5):
        """
        Exit strategy: Use trailing stop - exit on X% pullback from recent high
        
        Args:
            signals_df: DataFrame with trading signals
            trailing_stop_pct: Stop loss percentage (e.g., 3.0 for 3%)
            lookback_days: Days to lookback for recent high
            
        Returns:
            DataFrame with backtest results
        """
        results_df = signals_df.copy()
        results_df['Cash'] = 0.0
        results_df['Shares_Held'] = 0.0
        results_df['Entry_Price'] = 0.0
        results_df['Recent_High'] = 0.0
        results_df['Portfolio_Value'] = 0.0
        results_df['Daily_Return'] = 0.0
        
        cash = 100000.0
        shares_held = 0.0
        entry_price = 0.0
        recent_high = 0.0
        prev_portfolio_value = 100000.0
        
        for idx, row in results_df.iterrows():
            price = row['Actual_Price']
            signal = row['Signal']
            confidence = row.get('Confidence', 0.5)
            
            # Update recent high for trailing stop
            if shares_held > 0:
                lookback_start = max(0, idx - lookback_days)
                recent_high = results_df.loc[lookback_start:idx, 'Actual_Price'].max()
                
                # Exit on trailing stop
                stop_level = recent_high * (1 - trailing_stop_pct / 100)
                if price <= stop_level:
                    proceeds = shares_held * price
                    cash += proceeds
                    shares_held = 0.0
                    entry_price = 0.0
                    recent_high = 0.0
            
            # Entry on BUY signal
            if signal == 'BUY' and shares_held == 0:
                if confidence > 0.8:
                    position_size = 0.90
                elif confidence > 0.65:
                    position_size = 0.70
                elif confidence > 0.5:
                    position_size = 0.50
                else:
                    position_size = 0.20
                
                available = cash * position_size
                shares_to_buy = int(available / price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    cash -= cost
                    shares_held = shares_to_buy
                    entry_price = price
                    recent_high = price
            
            # Calculate portfolio value
            portfolio_value = cash + (shares_held * price)
            daily_return = portfolio_value - prev_portfolio_value
            
            results_df.loc[idx, 'Cash'] = cash
            results_df.loc[idx, 'Shares_Held'] = shares_held
            results_df.loc[idx, 'Entry_Price'] = entry_price
            results_df.loc[idx, 'Recent_High'] = recent_high
            results_df.loc[idx, 'Portfolio_Value'] = portfolio_value
            results_df.loc[idx, 'Daily_Return'] = daily_return
            
            prev_portfolio_value = portfolio_value
        
        return results_df
    
    @staticmethod
    def backtest_with_mean_reversion_exit(signals_df, reversion_threshold=0.5):
        """
        Exit strategy: Hold until price reverts back to entry price + reversion_threshold
        
        Args:
            signals_df: DataFrame with trading signals
            reversion_threshold: % above entry price to exit (e.g., 0.5 for 0.5% gain)
            
        Returns:
            DataFrame with backtest results
        """
        results_df = signals_df.copy()
        results_df['Cash'] = 0.0
        results_df['Shares_Held'] = 0.0
        results_df['Entry_Price'] = 0.0
        results_df['Portfolio_Value'] = 0.0
        results_df['Daily_Return'] = 0.0
        
        cash = 100000.0
        shares_held = 0.0
        entry_price = 0.0
        prev_portfolio_value = 100000.0
        
        for idx, row in results_df.iterrows():
            price = row['Actual_Price']
            signal = row['Signal']
            confidence = row.get('Confidence', 0.5)
            
            # Check for reversion-based exit
            if shares_held > 0:
                # Exit when price falls back to entry price
                exit_target = entry_price * (1 + reversion_threshold / 100)
                if price <= exit_target:
                    proceeds = shares_held * price
                    cash += proceeds
                    shares_held = 0.0
                    entry_price = 0.0
            
            # Entry on BUY signal
            if signal == 'BUY' and shares_held == 0:
                if confidence > 0.8:
                    position_size = 0.90
                elif confidence > 0.65:
                    position_size = 0.70
                elif confidence > 0.5:
                    position_size = 0.50
                else:
                    position_size = 0.20
                
                available = cash * position_size
                shares_to_buy = int(available / price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    cash -= cost
                    shares_held = shares_to_buy
                    entry_price = price
            
            # Calculate portfolio value
            portfolio_value = cash + (shares_held * price)
            daily_return = portfolio_value - prev_portfolio_value
            
            results_df.loc[idx, 'Cash'] = cash
            results_df.loc[idx, 'Shares_Held'] = shares_held
            results_df.loc[idx, 'Entry_Price'] = entry_price
            results_df.loc[idx, 'Portfolio_Value'] = portfolio_value
            results_df.loc[idx, 'Daily_Return'] = daily_return
            
            prev_portfolio_value = portfolio_value
        
        return results_df
    
    @staticmethod
    def backtest_with_volatility_stop(signals_df, volatility_multiplier=2.0, lookback_days=20):
        """
        Exit strategy: Use volatility-based stops - wider stops in high volatility
        
        Args:
            signals_df: DataFrame with trading signals
            volatility_multiplier: Multiplier for volatility-based stop (e.g., 2.0 = 2x volatility)
            lookback_days: Days to calculate volatility
            
        Returns:
            DataFrame with backtest results
        """
        results_df = signals_df.copy()
        results_df['Cash'] = 0.0
        results_df['Shares_Held'] = 0.0
        results_df['Entry_Price'] = 0.0
        results_df['Stop_Level'] = 0.0
        results_df['Portfolio_Value'] = 0.0
        results_df['Daily_Return'] = 0.0
        
        # Calculate volatility if not present
        if 'Volatility_20' not in results_df.columns:
            results_df['Volatility_20'] = results_df['Actual_Price'].rolling(lookback_days).std() / results_df['Actual_Price'].rolling(lookback_days).mean()
        
        cash = 100000.0
        shares_held = 0.0
        entry_price = 0.0
        stop_level = 0.0
        prev_portfolio_value = 100000.0
        
        for idx, row in results_df.iterrows():
            price = row['Actual_Price']
            signal = row['Signal']
            confidence = row.get('Confidence', 0.5)
            volatility = row.get('Volatility_20', 0.01)
            
            # Check for volatility-based exit
            if shares_held > 0:
                stop_level = entry_price * (1 - volatility_multiplier * volatility)
                if price <= stop_level:
                    proceeds = shares_held * price
                    cash += proceeds
                    shares_held = 0.0
                    entry_price = 0.0
                    stop_level = 0.0
            
            # Entry on BUY signal
            if signal == 'BUY' and shares_held == 0:
                if confidence > 0.8:
                    position_size = 0.90
                elif confidence > 0.65:
                    position_size = 0.70
                elif confidence > 0.5:
                    position_size = 0.50
                else:
                    position_size = 0.20
                
                available = cash * position_size
                shares_to_buy = int(available / price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    cash -= cost
                    shares_held = shares_to_buy
                    entry_price = price
                    stop_level = entry_price * (1 - volatility_multiplier * volatility)
            
            # Calculate portfolio value
            portfolio_value = cash + (shares_held * price)
            daily_return = portfolio_value - prev_portfolio_value
            
            results_df.loc[idx, 'Cash'] = cash
            results_df.loc[idx, 'Shares_Held'] = shares_held
            results_df.loc[idx, 'Entry_Price'] = entry_price
            results_df.loc[idx, 'Stop_Level'] = stop_level
            results_df.loc[idx, 'Portfolio_Value'] = portfolio_value
            results_df.loc[idx, 'Daily_Return'] = daily_return
            
            prev_portfolio_value = portfolio_value
        
        return results_df


def test_all_exit_strategies(signals_df, baseline_df):
    """
    Test all exit strategies and compare to baseline
    
    Args:
        signals_df: DataFrame with trading signals
        baseline_df: Baseline portfolio backtest (current approach)
        
    Returns:
        Dictionary with all results and comparison
    """
    print("\n" + "="*80)
    print("TESTING EXIT STRATEGIES")
    print("="*80)
    
    strategies = ExitStrategies()
    results = {}
    
    # Baseline (from current approach)
    baseline_return = ((baseline_df['Portfolio_Value'].iloc[-1] - 100000) / 100000 * 100)
    baseline_sharpe = baseline_df['Daily_Return'].mean() / baseline_df['Daily_Return'].std() * np.sqrt(252)
    baseline_dd = ((baseline_df['Portfolio_Value'].min() - 100000) / 100000 * 100)
    
    print(f"\n[BASELINE - Current Approach]")
    print(f"  Return:       {baseline_return:.2f}%")
    print(f"  Sharpe:       {baseline_sharpe:.3f}")
    print(f"  Max Drawdown: {baseline_dd:.2f}%")
    
    # Strategy 1: Profit-taking
    print(f"\n[1] PROFIT-TAKING EXIT (5% target, 20-day max hold)")
    pt_df = strategies.backtest_with_profit_taking(signals_df, profit_target_pct=5.0, max_hold_days=20)
    pt_return = ((pt_df['Portfolio_Value'].iloc[-1] - 100000) / 100000 * 100)
    pt_sharpe = pt_df['Daily_Return'].mean() / pt_df['Daily_Return'].std() * np.sqrt(252) if pt_df['Daily_Return'].std() > 0 else 0
    pt_dd = ((pt_df['Portfolio_Value'].min() - 100000) / 100000 * 100)
    
    print(f"  Return:       {pt_return:.2f}% ({pt_return-baseline_return:+.2f}pp)")
    print(f"  Sharpe:       {pt_sharpe:.3f} ({pt_sharpe-baseline_sharpe:+.3f})")
    print(f"  Max Drawdown: {pt_dd:.2f}%")
    results['profit_taking'] = {'df': pt_df, 'return': pt_return, 'sharpe': pt_sharpe, 'dd': pt_dd}
    
    # Strategy 2: Trailing stop
    print(f"\n[2] TRAILING STOP EXIT (3% pullback from recent high)")
    ts_df = strategies.backtest_with_trailing_stop(signals_df, trailing_stop_pct=3.0, lookback_days=5)
    ts_return = ((ts_df['Portfolio_Value'].iloc[-1] - 100000) / 100000 * 100)
    ts_sharpe = ts_df['Daily_Return'].mean() / ts_df['Daily_Return'].std() * np.sqrt(252) if ts_df['Daily_Return'].std() > 0 else 0
    ts_dd = ((ts_df['Portfolio_Value'].min() - 100000) / 100000 * 100)
    
    print(f"  Return:       {ts_return:.2f}% ({ts_return-baseline_return:+.2f}pp)")
    print(f"  Sharpe:       {ts_sharpe:.3f} ({ts_sharpe-baseline_sharpe:+.3f})")
    print(f"  Max Drawdown: {ts_dd:.2f}%")
    results['trailing_stop'] = {'df': ts_df, 'return': ts_return, 'sharpe': ts_sharpe, 'dd': ts_dd}
    
    # Strategy 3: Mean reversion
    print(f"\n[3] MEAN REVERSION EXIT (exit at +0.5% from entry)")
    mr_df = strategies.backtest_with_mean_reversion_exit(signals_df, reversion_threshold=0.5)
    mr_return = ((mr_df['Portfolio_Value'].iloc[-1] - 100000) / 100000 * 100)
    mr_sharpe = mr_df['Daily_Return'].mean() / mr_df['Daily_Return'].std() * np.sqrt(252) if mr_df['Daily_Return'].std() > 0 else 0
    mr_dd = ((mr_df['Portfolio_Value'].min() - 100000) / 100000 * 100)
    
    print(f"  Return:       {mr_return:.2f}% ({mr_return-baseline_return:+.2f}pp)")
    print(f"  Sharpe:       {mr_sharpe:.3f} ({mr_sharpe-baseline_sharpe:+.3f})")
    print(f"  Max Drawdown: {mr_dd:.2f}%")
    results['mean_reversion'] = {'df': mr_df, 'return': mr_return, 'sharpe': mr_sharpe, 'dd': mr_dd}
    
    # Strategy 4: Volatility-based
    print(f"\n[4] VOLATILITY-BASED EXIT (2x volatility stop)")
    vb_df = strategies.backtest_with_volatility_stop(signals_df, volatility_multiplier=2.0, lookback_days=20)
    vb_return = ((vb_df['Portfolio_Value'].iloc[-1] - 100000) / 100000 * 100)
    vb_sharpe = vb_df['Daily_Return'].mean() / vb_df['Daily_Return'].std() * np.sqrt(252) if vb_df['Daily_Return'].std() > 0 else 0
    vb_dd = ((vb_df['Portfolio_Value'].min() - 100000) / 100000 * 100)
    
    print(f"  Return:       {vb_return:.2f}% ({vb_return-baseline_return:+.2f}pp)")
    print(f"  Sharpe:       {vb_sharpe:.3f} ({vb_sharpe-baseline_sharpe:+.3f})")
    print(f"  Max Drawdown: {vb_dd:.2f}%")
    results['volatility_stop'] = {'df': vb_df, 'return': vb_return, 'sharpe': vb_sharpe, 'dd': vb_dd}
    
    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY - Ranked by Return")
    print("="*80)
    
    ranked = sorted(
        [('Baseline', baseline_return, baseline_sharpe, baseline_dd)] + 
        [(name, data['return'], data['sharpe'], data['dd']) for name, data in results.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    print(f"\n{'Rank':<3} {'Strategy':<25} {'Return':<12} {'Sharpe':<12} {'Max DD':<12}")
    print("-"*64)
    for i, (name, ret, sharpe, dd) in enumerate(ranked, 1):
        print(f"{i:<3} {name:<25} {ret:>10.2f}% {sharpe:>10.3f} {dd:>10.2f}%")
    
    best_strategy = ranked[0][0]
    best_return = ranked[0][1]
    improvement = best_return - baseline_return
    
    print(f"\n[WINNER] {best_strategy}: +{best_return:.2f}% ({improvement:+.2f}pp vs baseline)")
    
    return results, best_strategy, ranked
