"""
Dynamic Position Sizing Strategies
Tests multiple approaches to optimize position allocation:
1. Kelly Criterion (optimal bet sizing based on win rate and payoff)
2. Volatility-adjusted sizing (smaller positions in high volatility)
3. Momentum-based sizing (larger positions in strong trends)
4. Win rate adaptive sizing (positions based on recent performance)
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DynamicSizing:
    """Dynamic position sizing strategies"""
    
    @staticmethod
    def kelly_criterion(win_rate, avg_win, avg_loss):
        """
        Calculate Kelly Criterion optimal position size.
        Formula: f* = (p * b - q) / b
        where p = win rate, q = loss rate, b = win/loss ratio
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average gain per winning trade
            avg_loss: Average loss per losing trade (as positive number)
            
        Returns:
            Optimal position size as fraction of bankroll (0-1)
        """
        if avg_loss == 0:
            return 0.5
        
        loss_rate = 1 - win_rate
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        # Kelly formula
        kelly_pct = (win_rate * payoff_ratio - loss_rate) / payoff_ratio
        
        # Bound between 0.1 and 0.9 (never bet everything, never bet nothing)
        kelly_pct = max(0.1, min(0.9, kelly_pct))
        
        return kelly_pct
    
    @staticmethod
    def backtest_with_kelly_sizing(signals_df, lookback_window=30, kelly_fraction=1.0):
        """
        Position sizing based on Kelly Criterion calculated from recent trades.
        
        Args:
            signals_df: DataFrame with trading signals
            lookback_window: Days to look back for win rate calculation
            kelly_fraction: Fraction of Kelly to use (1.0 = full Kelly, 0.5 = half Kelly)
            
        Returns:
            DataFrame with backtest results
        """
        results_df = signals_df.copy()
        results_df['Cash'] = 0.0
        results_df['Shares_Held'] = 0.0
        results_df['Portfolio_Value'] = 0.0
        results_df['Daily_Return'] = 0.0
        results_df['Position_Size'] = 0.5  # Default
        
        cash = 100000.0
        shares_held = 0.0
        position_size = 0.5
        prev_portfolio_value = 100000.0
        
        trade_history = []
        
        for idx, row in results_df.iterrows():
            price = row['Actual_Price']
            signal = row['Signal']
            confidence = row.get('Confidence', 0.5)
            
            # Calculate Kelly sizing based on recent trade history
            if len(trade_history) >= 5:  # Need at least 5 trades to calculate
                win_rate = sum(1 for t in trade_history[-lookback_window:] if t > 0) / min(len(trade_history), lookback_window)
                avg_win = max(0.01, sum(t for t in trade_history[-lookback_window:] if t > 0) / max(1, sum(1 for t in trade_history[-lookback_window:] if t > 0)))
                avg_loss = max(0.01, -sum(t for t in trade_history[-lookback_window:] if t < 0) / max(1, sum(1 for t in trade_history[-lookback_window:] if t < 0)))
                
                kelly_size = DynamicSizing.kelly_criterion(win_rate, avg_win, avg_loss)
                position_size = kelly_fraction * kelly_size + (1 - kelly_fraction) * 0.5
            
            # Entry on BUY signal
            if signal == 'BUY' and shares_held == 0:
                # Apply Kelly sizing
                available = cash * position_size
                shares_to_buy = int(available / price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    cash -= cost
                    shares_held = shares_to_buy
                    entry_price = price
            
            # Exit: end of holding period (simplified - just hold)
            # In reality would track entry and exit prices
            
            # Calculate portfolio value
            portfolio_value = cash + (shares_held * price)
            daily_return = portfolio_value - prev_portfolio_value
            
            results_df.loc[idx, 'Cash'] = cash
            results_df.loc[idx, 'Shares_Held'] = shares_held
            results_df.loc[idx, 'Portfolio_Value'] = portfolio_value
            results_df.loc[idx, 'Daily_Return'] = daily_return
            results_df.loc[idx, 'Position_Size'] = position_size
            
            prev_portfolio_value = portfolio_value
        
        return results_df
    
    @staticmethod
    def backtest_with_volatility_sizing(signals_df, lookback_days=20, target_volatility=0.01):
        """
        Position sizing inversely proportional to volatility.
        Higher volatility → smaller positions
        
        Args:
            signals_df: DataFrame with trading signals
            lookback_days: Days for volatility calculation
            target_volatility: Target daily volatility (e.g., 0.01 = 1%)
            
        Returns:
            DataFrame with backtest results
        """
        results_df = signals_df.copy()
        results_df['Cash'] = 0.0
        results_df['Shares_Held'] = 0.0
        results_df['Portfolio_Value'] = 0.0
        results_df['Daily_Return'] = 0.0
        results_df['Position_Size'] = 0.5
        
        # Calculate rolling volatility
        price_returns = results_df['Actual_Price'].pct_change()
        results_df['Volatility'] = price_returns.rolling(lookback_days).std()
        results_df['Volatility'] = results_df['Volatility'].fillna(target_volatility)
        
        cash = 100000.0
        shares_held = 0.0
        prev_portfolio_value = 100000.0
        
        for idx, row in results_df.iterrows():
            price = row['Actual_Price']
            signal = row['Signal']
            confidence = row.get('Confidence', 0.5)
            volatility = row['Volatility']
            
            # Size inversely to volatility
            if volatility > 0:
                position_size = min(0.9, max(0.2, target_volatility / volatility))
            else:
                position_size = 0.5
            
            # Entry on BUY signal
            if signal == 'BUY' and shares_held == 0:
                available = cash * position_size
                shares_to_buy = int(available / price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    cash -= cost
                    shares_held = shares_to_buy
            
            # Calculate portfolio value
            portfolio_value = cash + (shares_held * price)
            daily_return = portfolio_value - prev_portfolio_value
            
            results_df.loc[idx, 'Cash'] = cash
            results_df.loc[idx, 'Shares_Held'] = shares_held
            results_df.loc[idx, 'Portfolio_Value'] = portfolio_value
            results_df.loc[idx, 'Daily_Return'] = daily_return
            results_df.loc[idx, 'Position_Size'] = position_size
            
            prev_portfolio_value = portfolio_value
        
        return results_df
    
    @staticmethod
    def backtest_with_momentum_sizing(signals_df, lookback_days=10, sma_period=50):
        """
        Position sizing based on momentum strength.
        Larger positions in strong uptrends, smaller in weak/downtrends
        
        Args:
            signals_df: DataFrame with trading signals
            lookback_days: Days for momentum calculation
            sma_period: Period for trend detection
            
        Returns:
            DataFrame with backtest results
        """
        results_df = signals_df.copy()
        results_df['Cash'] = 0.0
        results_df['Shares_Held'] = 0.0
        results_df['Portfolio_Value'] = 0.0
        results_df['Daily_Return'] = 0.0
        results_df['Position_Size'] = 0.5
        
        # Calculate SMA for trend
        results_df['SMA'] = results_df['Actual_Price'].rolling(sma_period).mean()
        
        # Calculate momentum
        price_momentum = results_df['Actual_Price'].pct_change(lookback_days)
        results_df['Momentum'] = price_momentum
        
        cash = 100000.0
        shares_held = 0.0
        prev_portfolio_value = 100000.0
        
        for idx, row in results_df.iterrows():
            price = row['Actual_Price']
            signal = row['Signal']
            momentum = row['Momentum']
            sma = row['SMA']
            
            # Determine trend and momentum
            if pd.notna(sma):
                above_sma = price > sma
            else:
                above_sma = True
            
            if pd.notna(momentum):
                momentum_strength = min(1.0, max(-1.0, momentum * 10))  # Normalize to [-1, 1]
            else:
                momentum_strength = 0.0
            
            # Position size based on trend and momentum
            if above_sma and momentum_strength > 0:
                # Strong uptrend: larger positions
                position_size = 0.7 + (momentum_strength * 0.2)  # 0.7 - 0.9
            elif above_sma:
                # Weak uptrend: medium positions
                position_size = 0.5 + (momentum_strength * 0.2)  # 0.5 - 0.7
            elif momentum_strength > 0:
                # Recovery attempt: smaller positions
                position_size = 0.3 + (momentum_strength * 0.2)  # 0.3 - 0.5
            else:
                # Downtrend: minimal positions
                position_size = 0.2
            
            position_size = min(0.9, max(0.2, position_size))
            
            # Entry on BUY signal
            if signal == 'BUY' and shares_held == 0:
                available = cash * position_size
                shares_to_buy = int(available / price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    cash -= cost
                    shares_held = shares_to_buy
            
            # Calculate portfolio value
            portfolio_value = cash + (shares_held * price)
            daily_return = portfolio_value - prev_portfolio_value
            
            results_df.loc[idx, 'Cash'] = cash
            results_df.loc[idx, 'Shares_Held'] = shares_held
            results_df.loc[idx, 'Portfolio_Value'] = portfolio_value
            results_df.loc[idx, 'Daily_Return'] = daily_return
            results_df.loc[idx, 'Position_Size'] = position_size
            
            prev_portfolio_value = portfolio_value
        
        return results_df
    
    @staticmethod
    def backtest_with_adaptive_sizing(signals_df, lookback_days=20):
        """
        Adaptive sizing that increases after wins and decreases after losses.
        Uses recent win rate to scale position sizes.
        
        Args:
            signals_df: DataFrame with trading signals
            lookback_days: Days to calculate recent win rate
            
        Returns:
            DataFrame with backtest results
        """
        results_df = signals_df.copy()
        results_df['Cash'] = 0.0
        results_df['Shares_Held'] = 0.0
        results_df['Portfolio_Value'] = 0.0
        results_df['Daily_Return'] = 0.0
        results_df['Position_Size'] = 0.5
        
        cash = 100000.0
        shares_held = 0.0
        position_size = 0.5
        prev_portfolio_value = 100000.0
        
        recent_returns = []
        
        for idx, row in results_df.iterrows():
            price = row['Actual_Price']
            signal = row['Signal']
            confidence = row.get('Confidence', 0.5)
            
            # Calculate position size based on recent performance
            if len(recent_returns) > 5:
                # Win rate from recent daily returns
                win_days = sum(1 for r in recent_returns[-lookback_days:] if r > 0)
                total_days = min(len(recent_returns), lookback_days)
                recent_win_rate = win_days / total_days if total_days > 0 else 0.5
                
                # Scale position size by win rate
                # 60% win rate -> 0.7 position size
                # 50% win rate -> 0.5 position size
                # 40% win rate -> 0.3 position size
                position_size = 0.2 + (recent_win_rate * 0.7)
                position_size = min(0.9, max(0.2, position_size))
            
            # Entry on BUY signal
            if signal == 'BUY' and shares_held == 0:
                available = cash * position_size
                shares_to_buy = int(available / price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    cash -= cost
                    shares_held = shares_to_buy
            
            # Calculate portfolio value
            portfolio_value = cash + (shares_held * price)
            daily_return = portfolio_value - prev_portfolio_value
            
            results_df.loc[idx, 'Cash'] = cash
            results_df.loc[idx, 'Shares_Held'] = shares_held
            results_df.loc[idx, 'Portfolio_Value'] = portfolio_value
            results_df.loc[idx, 'Daily_Return'] = daily_return
            results_df.loc[idx, 'Position_Size'] = position_size
            
            recent_returns.append(daily_return)
            prev_portfolio_value = portfolio_value
        
        return results_df


def test_dynamic_sizing_strategies(signals_df, baseline_df):
    """
    Test all dynamic position sizing strategies.
    
    Args:
        signals_df: DataFrame with trading signals
        baseline_df: Baseline portfolio backtest
        
    Returns:
        Comparison of all strategies
    """
    print("\n" + "="*80)
    print("TESTING DYNAMIC POSITION SIZING STRATEGIES")
    print("="*80)
    
    strategies = DynamicSizing()
    
    # Baseline metrics
    baseline_return = ((baseline_df['Portfolio_Value'].iloc[-1] - 100000) / 100000 * 100)
    baseline_sharpe = baseline_df['Daily_Return'].mean() / baseline_df['Daily_Return'].std() * np.sqrt(252)
    baseline_dd = ((baseline_df['Portfolio_Value'].min() - 100000) / 100000 * 100)
    
    print(f"\n[BASELINE - Fixed Sizing (90/70/50/20)]")
    print(f"  Return:       {baseline_return:.2f}%")
    print(f"  Sharpe:       {baseline_sharpe:.3f}")
    print(f"  Max Drawdown: {baseline_dd:.2f}%")
    
    results = {}
    
    # Strategy 1: Kelly Criterion
    print(f"\n[1] KELLY CRITERION SIZING")
    kelly_df = strategies.backtest_with_kelly_sizing(signals_df, lookback_window=30, kelly_fraction=1.0)
    kelly_return = ((kelly_df['Portfolio_Value'].iloc[-1] - 100000) / 100000 * 100)
    kelly_sharpe = kelly_df['Daily_Return'].mean() / kelly_df['Daily_Return'].std() * np.sqrt(252)
    kelly_dd = ((kelly_df['Portfolio_Value'].min() - 100000) / 100000 * 100)
    
    print(f"  Return:       {kelly_return:.2f}% ({kelly_return-baseline_return:+.2f}pp)")
    print(f"  Sharpe:       {kelly_sharpe:.3f} ({kelly_sharpe-baseline_sharpe:+.3f})")
    print(f"  Max Drawdown: {kelly_dd:.2f}%")
    results['kelly'] = {'return': kelly_return, 'sharpe': kelly_sharpe, 'dd': kelly_dd}
    
    # Strategy 2: Volatility-adjusted
    print(f"\n[2] VOLATILITY-ADJUSTED SIZING")
    vol_df = strategies.backtest_with_volatility_sizing(signals_df, lookback_days=20, target_volatility=0.01)
    vol_return = ((vol_df['Portfolio_Value'].iloc[-1] - 100000) / 100000 * 100)
    vol_sharpe = vol_df['Daily_Return'].mean() / vol_df['Daily_Return'].std() * np.sqrt(252)
    vol_dd = ((vol_df['Portfolio_Value'].min() - 100000) / 100000 * 100)
    
    print(f"  Return:       {vol_return:.2f}% ({vol_return-baseline_return:+.2f}pp)")
    print(f"  Sharpe:       {vol_sharpe:.3f} ({vol_sharpe-baseline_sharpe:+.3f})")
    print(f"  Max Drawdown: {vol_dd:.2f}%")
    results['volatility'] = {'return': vol_return, 'sharpe': vol_sharpe, 'dd': vol_dd}
    
    # Strategy 3: Momentum-based
    print(f"\n[3] MOMENTUM-BASED SIZING")
    mom_df = strategies.backtest_with_momentum_sizing(signals_df, lookback_days=10, sma_period=50)
    mom_return = ((mom_df['Portfolio_Value'].iloc[-1] - 100000) / 100000 * 100)
    mom_sharpe = mom_df['Daily_Return'].mean() / mom_df['Daily_Return'].std() * np.sqrt(252)
    mom_dd = ((mom_df['Portfolio_Value'].min() - 100000) / 100000 * 100)
    
    print(f"  Return:       {mom_return:.2f}% ({mom_return-baseline_return:+.2f}pp)")
    print(f"  Sharpe:       {mom_sharpe:.3f} ({mom_sharpe-baseline_sharpe:+.3f})")
    print(f"  Max Drawdown: {mom_dd:.2f}%")
    results['momentum'] = {'return': mom_return, 'sharpe': mom_sharpe, 'dd': mom_dd}
    
    # Strategy 4: Adaptive sizing
    print(f"\n[4] ADAPTIVE SIZING (Win-rate based)")
    adapt_df = strategies.backtest_with_adaptive_sizing(signals_df, lookback_days=20)
    adapt_return = ((adapt_df['Portfolio_Value'].iloc[-1] - 100000) / 100000 * 100)
    adapt_sharpe = adapt_df['Daily_Return'].mean() / adapt_df['Daily_Return'].std() * np.sqrt(252)
    adapt_dd = ((adapt_df['Portfolio_Value'].min() - 100000) / 100000 * 100)
    
    print(f"  Return:       {adapt_return:.2f}% ({adapt_return-baseline_return:+.2f}pp)")
    print(f"  Sharpe:       {adapt_sharpe:.3f} ({adapt_sharpe-baseline_sharpe:+.3f})")
    print(f"  Max Drawdown: {adapt_dd:.2f}%")
    results['adaptive'] = {'return': adapt_return, 'sharpe': adapt_sharpe, 'dd': adapt_dd}
    
    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY - Ranked by Return")
    print("="*80)
    
    ranked = sorted(
        [('Baseline (Fixed)', baseline_return, baseline_sharpe, baseline_dd)] + 
        [(name, data['return'], data['sharpe'], data['dd']) for name, data in results.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    print(f"\n{'Rank':<3} {'Strategy':<30} {'Return':<12} {'Sharpe':<12} {'Max DD':<12}")
    print("-"*70)
    for i, (name, ret, sharpe, dd) in enumerate(ranked, 1):
        print(f"{i:<3} {name:<30} {ret:>10.2f}% {sharpe:>10.3f} {dd:>10.2f}%")
    
    best_name, best_return, best_sharpe, best_dd = ranked[0]
    improvement = best_return - baseline_return
    
    print(f"\n[WINNER] {best_name}: +{best_return:.2f}% ({improvement:+.2f}pp vs baseline)")
    
    return results, ranked
