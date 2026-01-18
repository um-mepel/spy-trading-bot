"""
Comprehensive Regime Multiplier Grid Search
Tests different multiplier combinations to find optimal values for each regime.

Strategy: Load the existing backtest CSV and test different multipliers on it.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def calculate_metrics(backtest_df, multiplier_config):
    """Calculate performance metrics for a backtest."""
    portfolio_values = backtest_df['Portfolio_Value'].values
    daily_returns = backtest_df['Daily_Return'].values
    
    # Returns
    final_value = portfolio_values[-1]
    total_return = (final_value - 100000) / 100000 * 100
    
    # Daily volatility (as % of portfolio)
    daily_returns_pct = []
    for i in range(1, len(daily_returns)):
        pct = (daily_returns[i] / portfolio_values[i-1]) * 100
        daily_returns_pct.append(pct)
    
    daily_returns_pct = np.array(daily_returns_pct)
    volatility = daily_returns_pct.std() if len(daily_returns_pct) > 0 else 0
    
    # Sharpe ratio
    mean_daily_return = daily_returns_pct.mean() if len(daily_returns_pct) > 0 else 0
    sharpe = (mean_daily_return / volatility * np.sqrt(252)) if volatility > 0 else 0
    
    # Drawdown
    cummax = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cummax) / cummax * 100
    max_drawdown = np.min(drawdown)
    
    # Win rate
    win_rate = (np.sum(daily_returns > 0) / len(daily_returns) * 100) if len(daily_returns) > 0 else 0
    
    return {
        'config': multiplier_config,
        'final_value': final_value,
        'total_return': total_return,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
    }


def apply_multipliers_to_backtest(baseline_df, bullish_mult, neutral_mult, bearish_mult):
    """
    Apply multipliers to baseline backtest by adjusting position sizes.
    This simulates what would happen with different regime multipliers.
    """
    df = baseline_df.copy()
    
    for idx in range(len(df)):
        regime = df.iloc[idx]['Regime']
        base_position = df.iloc[idx]['Adjusted_Position_Size']
        
        # Get multiplier for this regime
        if regime == 'bullish':
            mult = bullish_mult
        elif regime == 'bearish':
            mult = bearish_mult
        else:
            mult = neutral_mult
        
        # Apply multiplier to position size
        new_position = base_position * (mult / 1.0)  # Normalize by baseline bullish (1.0)
        
        df.iloc[idx, df.columns.get_loc('Adjusted_Position_Size')] = new_position
    
    # Recalculate portfolio based on new position sizes
    # This is simplified - actual backtesting would need full recalculation
    # For now, we'll scale returns proportionally
    
    return df


def main():
    """Run comprehensive regime multiplier grid search using existing backtest data."""
    print("╔" + "═" * 78 + "╗")
    print("║" + " REGIME MULTIPLIER GRID SEARCH - COMPREHENSIVE TEST ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Load the existing regime-aware backtest
    results_dir = Path(__file__).parent.parent / "results" / "trading_analysis"
    backtest_file = results_dir / "portfolio_backtest_regime.csv"
    
    print(f"\n📂 Loading backtest data...")
    if not backtest_file.exists():
        print(f"✗ Error: {backtest_file} not found")
        print(f"  Please run: python3 main.py --model lightgbm")
        return
    
    baseline_df = pd.read_csv(backtest_file)
    baseline_df['Date'] = pd.to_datetime(baseline_df['Date'])
    
    print(f"✓ Loaded {len(baseline_df)} days of backtest data")
    
    print(f"\n" + "=" * 80)
    print(f"GRID SEARCH CONFIGURATION")
    print(f"=" * 80)
    
    # Define multiplier ranges to test
    bullish_values = [0.85, 0.90, 0.95, 1.00]
    neutral_values = [0.55, 0.65, 0.75, 0.85]
    bearish_values = [0.25, 0.35, 0.45, 0.55]
    
    print(f"Bullish multipliers:  {bullish_values}")
    print(f"Neutral multipliers:  {neutral_values}")
    print(f"Bearish multipliers:  {bearish_values}")
    print(f"\nTotal combinations to test: {len(bullish_values) * len(neutral_values) * len(bearish_values)}")
    
    # Current baseline (from current regime multipliers in code)
    baseline_config = {'bullish': 1.0, 'neutral': 0.65, 'bearish': 0.35}
    
    results = []
    best_sharpe = -float('inf')
    best_return = -float('inf')
    best_drawdown = float('inf')
    
    best_config_sharpe = None
    best_config_return = None
    best_config_drawdown = None
    
    print(f"\n" + "=" * 80)
    print(f"RUNNING GRID SEARCH ({len(bullish_values) * len(neutral_values) * len(bearish_values)} combinations)")
    print(f"=" * 80)
    
    combination_count = 0
    total_combinations = len(bullish_values) * len(neutral_values) * len(bearish_values)
    
    for bull_mult in bullish_values:
        for neut_mult in neutral_values:
            for bear_mult in bearish_values:
                combination_count += 1
                config = {
                    'bullish': bull_mult,
                    'neutral': neut_mult,
                    'bearish': bear_mult
                }
                
                try:
                    # Scale the baseline multipliers
                    # Current baseline uses 1.0, 0.65, 0.35
                    # We normalize by scaling: new_mult = proposed_mult / baseline_mult * original_effect
                    
                    # Recalculate position sizes with new multipliers
                    test_df = baseline_df.copy()
                    
                    # Recalculate portfolio value with adjusted position sizes
                    # This is a simplified approach: we scale daily returns proportionally to position size changes
                    
                    for idx in range(len(test_df)):
                        regime = test_df.iloc[idx]['Regime']
                        base_adjusted_position = test_df.iloc[idx]['Adjusted_Position_Size']
                        
                        # Get current multiplier from baseline
                        if regime == 'bullish':
                            current_mult = 1.0
                            new_mult = bull_mult
                        elif regime == 'bearish':
                            current_mult = 0.35
                            new_mult = bear_mult
                        else:
                            current_mult = 0.65
                            new_mult = neut_mult
                        
                        # Calculate the position size under old vs new multipliers
                        # base_adjusted = confidence_position * current_mult
                        # new_adjusted = confidence_position * new_mult
                        # ratio = new_adjusted / base_adjusted = new_mult / current_mult
                        
                        if current_mult > 0:
                            position_ratio = new_mult / current_mult
                            test_df.loc[idx, 'Adjusted_Position_Size'] = base_adjusted_position * position_ratio
                    
                    # Recalculate portfolio values
                    cash = 100000.0
                    shares_held = 0.0
                    
                    for idx in range(len(test_df)):
                        price = test_df.loc[idx, 'Actual_Price']
                        signal = test_df.loc[idx, 'Signal']
                        position_size = test_df.loc[idx, 'Adjusted_Position_Size']
                        
                        if signal == 'BUY' and position_size > 0:
                            buy_amount = cash * position_size
                            if buy_amount > 0 and price > 0:
                                buy_shares = buy_amount / price
                                shares_held += buy_shares
                                cash -= buy_amount
                        elif signal == 'SELL' and shares_held > 0:
                            sell_amount = shares_held * price
                            shares_held = 0
                            cash += sell_amount
                        
                        portfolio_value = cash + (shares_held * price)
                        test_df.loc[idx, 'Portfolio_Value'] = portfolio_value
                        test_df.loc[idx, 'Shares_Held'] = shares_held
                        test_df.loc[idx, 'Cash'] = cash
                        
                        if idx > 0:
                            prev_portfolio = test_df.loc[idx-1, 'Portfolio_Value']
                            daily_return = portfolio_value - prev_portfolio
                            test_df.loc[idx, 'Daily_Return'] = daily_return
                    
                    # Calculate metrics
                    metrics = calculate_metrics(test_df, config)
                    results.append(metrics)
                    
                    # Track best configurations
                    if metrics['sharpe'] > best_sharpe:
                        best_sharpe = metrics['sharpe']
                        best_config_sharpe = config.copy()
                    
                    if metrics['total_return'] > best_return:
                        best_return = metrics['total_return']
                        best_config_return = config.copy()
                    
                    if metrics['max_drawdown'] > best_drawdown:
                        best_drawdown = metrics['max_drawdown']
                        best_config_drawdown = config.copy()
                    
                    # Progress indicator
                    if combination_count % 4 == 0:
                        progress = (combination_count / total_combinations) * 100
                        print(f"  Progress: {combination_count:2d}/{total_combinations} ({progress:5.1f}%) - "
                              f"Config: B:{bull_mult:.2f} N:{neut_mult:.2f} Be:{bear_mult:.2f} | "
                              f"Return: {metrics['total_return']:+6.2f}% | Sharpe: {metrics['sharpe']:6.3f}")
                
                except Exception as e:
                    print(f"  ✗ Error with B:{bull_mult:.2f} N:{neut_mult:.2f} Be:{bear_mult:.2f}: {str(e)[:50]}")
                    continue
    
    print(f"\n" + "=" * 80)
    print(f"RESULTS SUMMARY")
    print(f"=" * 80)
    
    # Sort by different metrics
    results_by_sharpe = sorted(results, key=lambda x: x['sharpe'], reverse=True)
    results_by_return = sorted(results, key=lambda x: x['total_return'], reverse=True)
    results_by_drawdown = sorted(results, key=lambda x: x['max_drawdown'], reverse=True)
    
    print(f"\n🏆 TOP 5 BY SHARPE RATIO (Risk-Adjusted Return)")
    print("─" * 80)
    print(f"{'Rank':<5} {'Bull':<8} {'Neut':<8} {'Bear':<8} {'Return':<10} {'Sharpe':<10} {'Max DD':<10}")
    print("─" * 80)
    for i, result in enumerate(results_by_sharpe[:5], 1):
        cfg = result['config']
        print(f"{i:<5} {cfg['bullish']:<8.2f} {cfg['neutral']:<8.2f} {cfg['bearish']:<8.2f} "
              f"{result['total_return']:<10.2f}% {result['sharpe']:<10.3f} {result['max_drawdown']:<10.2f}%")
    
    print(f"\n💰 TOP 5 BY TOTAL RETURN")
    print("─" * 80)
    print(f"{'Rank':<5} {'Bull':<8} {'Neut':<8} {'Bear':<8} {'Return':<10} {'Sharpe':<10} {'Max DD':<10}")
    print("─" * 80)
    for i, result in enumerate(results_by_return[:5], 1):
        cfg = result['config']
        print(f"{i:<5} {cfg['bullish']:<8.2f} {cfg['neutral']:<8.2f} {cfg['bearish']:<8.2f} "
              f"{result['total_return']:<10.2f}% {result['sharpe']:<10.3f} {result['max_drawdown']:<10.2f}%")
    
    print(f"\n🛡️  TOP 5 BY BEST DRAWDOWN CONTROL (Least Negative)")
    print("─" * 80)
    print(f"{'Rank':<5} {'Bull':<8} {'Neut':<8} {'Bear':<8} {'Return':<10} {'Sharpe':<10} {'Max DD':<10}")
    print("─" * 80)
    for i, result in enumerate(results_by_drawdown[:5], 1):
        cfg = result['config']
        print(f"{i:<5} {cfg['bullish']:<8.2f} {cfg['neutral']:<8.2f} {cfg['bearish']:<8.2f} "
              f"{result['total_return']:<10.2f}% {result['sharpe']:<10.3f} {result['max_drawdown']:<10.2f}%")
    
    print(f"\n📊 BASELINE CONFIGURATION (Current)")
    print("─" * 80)
    baseline_results = [r for r in results if r['config'] == baseline_config]
    if baseline_results:
        baseline = baseline_results[0]
        print(f"Config:     B:{baseline['config']['bullish']:.2f} N:{baseline['config']['neutral']:.2f} Be:{baseline['config']['bearish']:.2f}")
        print(f"Return:     {baseline['total_return']:.2f}%")
        print(f"Sharpe:     {baseline['sharpe']:.3f}")
        print(f"Max DD:     {baseline['max_drawdown']:.2f}%")
        print(f"Win Rate:   {baseline['win_rate']:.1f}%")
    
    print(f"\n" + "=" * 80)
    print(f"COMPARATIVE ANALYSIS")
    print(f"=" * 80)
    
    if best_config_sharpe:
        best_sharpe_result = [r for r in results if r['config'] == best_config_sharpe][0]
        sharpe_improvement = best_sharpe_result['sharpe'] - baseline['sharpe']
        return_change = best_sharpe_result['total_return'] - baseline['total_return']
        dd_change = best_sharpe_result['max_drawdown'] - baseline['max_drawdown']
        print(f"\n✅ BEST SHARPE RATIO")
        print(f"  Config:    B:{best_config_sharpe['bullish']:.2f} N:{best_config_sharpe['neutral']:.2f} Be:{best_config_sharpe['bearish']:.2f}")
        print(f"  Sharpe:    {best_sharpe_result['sharpe']:.3f} ({sharpe_improvement:+.3f} vs baseline)")
        print(f"  Return:    {best_sharpe_result['total_return']:.2f}% ({return_change:+.2f}pp vs baseline)")
        print(f"  Max DD:    {best_sharpe_result['max_drawdown']:.2f}% ({dd_change:+.2f}pp vs baseline)")
    
    if best_config_return:
        best_return_result = [r for r in results if r['config'] == best_config_return][0]
        return_improvement = best_return_result['total_return'] - baseline['total_return']
        sharpe_change = best_return_result['sharpe'] - baseline['sharpe']
        dd_change = best_return_result['max_drawdown'] - baseline['max_drawdown']
        print(f"\n💰 BEST RETURN")
        print(f"  Config:    B:{best_config_return['bullish']:.2f} N:{best_config_return['neutral']:.2f} Be:{best_config_return['bearish']:.2f}")
        print(f"  Return:    {best_return_result['total_return']:.2f}% ({return_improvement:+.2f}pp vs baseline)")
        print(f"  Sharpe:    {best_return_result['sharpe']:.3f} ({sharpe_change:+.3f} vs baseline)")
        print(f"  Max DD:    {best_return_result['max_drawdown']:.2f}% ({dd_change:+.2f}pp vs baseline)")
    
    if best_config_drawdown:
        best_dd_result = [r for r in results if r['config'] == best_config_drawdown][0]
        dd_improvement = best_dd_result['max_drawdown'] - baseline['max_drawdown']
        return_change = best_dd_result['total_return'] - baseline['total_return']
        sharpe_change = best_dd_result['sharpe'] - baseline['sharpe']
        print(f"\n🛡️  BEST DRAWDOWN PROTECTION")
        print(f"  Config:    B:{best_config_drawdown['bullish']:.2f} N:{best_config_drawdown['neutral']:.2f} Be:{best_config_drawdown['bearish']:.2f}")
        print(f"  Max DD:    {best_dd_result['max_drawdown']:.2f}% ({dd_improvement:+.2f}pp vs baseline)")
        print(f"  Return:    {best_dd_result['total_return']:.2f}% ({return_change:+.2f}pp vs baseline)")
        print(f"  Sharpe:    {best_dd_result['sharpe']:.3f} ({sharpe_change:+.3f} vs baseline)")
    
    # Save results to CSV for further analysis
    results_csv_path = Path(__file__).parent.parent / "results" / "regime_multiplier_grid_search.csv"
    results_df_export = pd.DataFrame(results)
    results_df_export.to_csv(results_csv_path, index=False)
    
    print(f"\n" + "=" * 80)
    print(f"📁 RESULTS SAVED")
    print("─" * 80)
    print(f"✓ Full results: {results_csv_path}")
    print(f"✓ Total combinations tested: {len(results)}")
    
    # Print recommendations
    print(f"\n" + "=" * 80)
    print(f"📋 RECOMMENDATIONS")
    print(f"═" * 80)
    
    if best_config_sharpe:
        print(f"\n1️⃣  FOR RISK-ADJUSTED RETURNS (Sharpe Ratio):")
        print(f"    Use: B:{best_config_sharpe['bullish']:.2f} N:{best_config_sharpe['neutral']:.2f} Be:{best_config_sharpe['bearish']:.2f}")
        print(f"    Expected improvement: +{sharpe_improvement:.3f} Sharpe")
    
    if best_config_return and best_return_result['total_return'] > baseline['total_return']:
        print(f"\n2️⃣  FOR MAXIMUM RETURN:")
        print(f"    Use: B:{best_config_return['bullish']:.2f} N:{best_config_return['neutral']:.2f} Be:{best_config_return['bearish']:.2f}")
        print(f"    Expected improvement: +{return_improvement:.2f}pp return")
    
    if best_config_drawdown and best_dd_result['max_drawdown'] > baseline['max_drawdown']:
        print(f"\n3️⃣  FOR DRAWDOWN PROTECTION:")
        print(f"    Use: B:{best_config_drawdown['bullish']:.2f} N:{best_config_drawdown['neutral']:.2f} Be:{best_config_drawdown['bearish']:.2f}")
        print(f"    Expected improvement: {dd_improvement:.2f}pp better (less negative) drawdown")
    
    print(f"\n" + "=" * 80)
    print(f"✅ GRID SEARCH COMPLETE")
    print(f"═" * 80)


if __name__ == "__main__":
    main()
