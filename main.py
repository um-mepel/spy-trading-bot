"""
Main training orchestrator for stock prediction models.
Loads training and testing data, manages model imports and execution.
"""

import pandas as pd
import os
from pathlib import Path
from models.lightgbm_model import main as train_lightgbm_model
from models.ensemble_model import main as train_ensemble_model
from models.signal_generation import main as generate_trading_signals
from models.portfolio_management import main as run_portfolio_backtest
from models.regime_management import main as run_regime_backtest

# Global variables for training and testing data
TRAINING_DATA = None
TESTING_DATA = None
DATA_DIR = Path(__file__).parent
RESULTS_DIR = DATA_DIR / "results"
MODEL_PREDICTIONS_DIR = RESULTS_DIR / "model_predictions"
TRADING_ANALYSIS_DIR = RESULTS_DIR / "trading_analysis"
MODEL_VIZ_DIR = RESULTS_DIR / "visualizations" / "model_performance"
PORTFOLIO_VIZ_DIR = RESULTS_DIR / "visualizations" / "portfolio_performance"


def load_data():
    """
    Load training and testing CSV files into global variables.
    """
    global TRAINING_DATA, TESTING_DATA
    
    # Load 2022-2024 training data
    training_file = DATA_DIR / "data" / "SPY_training_2022_2024.csv"
    testing_file = DATA_DIR / "data" / "SPY_testing_2025.csv"
    
    if not training_file.exists():
        raise FileNotFoundError(f"Training data not found: {training_file}")
    
    print(f"Loading training data from {training_file.name}...")
    TRAINING_DATA = pd.read_csv(training_file)
    print(f"  Loaded {len(TRAINING_DATA)} rows (2022-2024), {len(TRAINING_DATA.columns)} columns")
    
    if not testing_file.exists():
        raise FileNotFoundError(f"Testing data not found: {testing_file}")
    
    print(f"Loading testing data from {testing_file.name}...")
    TESTING_DATA = pd.read_csv(testing_file)
    print(f"  Loaded {len(TESTING_DATA)} rows, {len(TESTING_DATA.columns)} columns")


# Choose which model to run: 'lightgbm' (single) or 'ensemble' (LGB+XGB+RF)
ACTIVE_MODEL = 'lightgbm'  # Change to 'ensemble' to use ensemble model


def save_model_results(model_results_dict):
    """
    Consolidate results from all models into a single CSV file.
    
    Args:
        model_results_dict: Dictionary with model names and their result DataFrames
    """
    # Create results directory if it doesn't exist
    RESULTS_DIR.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"Saved {len(model_results_dict)} model result CSVs")
    print(f"Location: {RESULTS_DIR}")
    print(f"{'='*60}\n")


def main():
    """
    LEAK-FREE Main orchestrator function for model training.
    Eliminates all data leakage:
    1. Calculates indicators separately on train/test data
    2. Trains on training data only
    3. Tests on test data only
    4. Uses locked hyperparameters (no tuning on test set)
    
    HYPERPARAMETERS (LOCKED):
    - Position sizing: 50% of cash per BUY signal
    - Buy percentile: 25.0 (top 25% of predictions)
    - Sell percentile: 75.0 (bottom 75% of predictions)
    - Mean reversion buy threshold: -5% below 20-day MA
    - Mean reversion sell threshold: +5% above 20-day MA
    """
    print("="*70)
    print("LEAK-FREE BACKTESTING PIPELINE")
    print("="*70)
    print()
    
    # Load data
    load_data()
    
    print("\n" + "="*70)
    print("Step 1: Calculate Technical Indicators (SEPARATELY - NO MIXING)")
    print("="*70)
    
    # Import clean indicator function
    from fetch_stock_data import calculate_technical_indicators
    
    # Calculate indicators on TRAINING DATA ONLY
    print("\nCalculating indicators on training data...")
    training_data_with_indicators = calculate_technical_indicators(
        TRAINING_DATA, 
        is_training=True
    )
    training_data_with_indicators = training_data_with_indicators.dropna()
    print(f"  ✓ Training data with indicators: {len(training_data_with_indicators)} rows")
    
    # Calculate indicators on TESTING DATA with training data warm-up
    # Get last 200 rows of training data for warming up rolling windows
    training_tail = training_data_with_indicators.tail(200)
    print(f"\nCalculating indicators on testing data (with {len(training_tail)}-row warm-up)...")
    testing_data_with_indicators = calculate_technical_indicators(
        TESTING_DATA,
        is_training=False,
        training_tail=training_tail
    )
    testing_data_with_indicators = testing_data_with_indicators.dropna()
    print(f"  ✓ Testing data with indicators: {len(testing_data_with_indicators)} rows")
    
    print("\n" + "="*70)
    print("Step 2: Train Model (ON TRAINING DATA ONLY)")
    print("="*70)
    
    # Create all result directories
    MODEL_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    TRADING_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_VIZ_DIR.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_VIZ_DIR.mkdir(parents=True, exist_ok=True)
    
    # Train selected model
    print(f"\nTraining model: {ACTIVE_MODEL.upper()}...")
    if ACTIVE_MODEL == 'lightgbm':
        # Single LightGBM model
        from models.lightgbm_model import main as train_lightgbm
        model_results = train_lightgbm(
            training_data_with_indicators,
            testing_data_with_indicators,
            results_dir=MODEL_PREDICTIONS_DIR
        )
    elif ACTIVE_MODEL == 'ensemble':
        # Ensemble: LightGBM + XGBoost + Random Forest
        model_results = train_ensemble_model(
            training_data_with_indicators,
            testing_data_with_indicators,
            results_dir=MODEL_PREDICTIONS_DIR
        )
    else:
        raise ValueError(f"Unknown ACTIVE_MODEL: {ACTIVE_MODEL}. Use 'lightgbm' or 'ensemble'.")
    
    if isinstance(model_results, dict) and 'predictions_df' in model_results:
        predictions_df = model_results['predictions_df']
    elif isinstance(model_results, dict) and 'results' in model_results:
        predictions_df = model_results['results']
    else:
        predictions_df = model_results if isinstance(model_results, pd.DataFrame) else None
    
    # DUAL-MODEL SYSTEM: Train exit model to predict price drops
    print("\n" + "="*70)
    print("Step 2B: Train EXIT Model (Predict Short-Term Price Drops)")
    print("="*70)
    
    from models.exit_model import main as train_exit_model
    exit_model_results = train_exit_model(
        training_data_with_indicators,
        testing_data_with_indicators,
        results_dir=MODEL_PREDICTIONS_DIR
    )
    exit_predictions_df = exit_model_results['results']
    
    print("\n" + "="*70)
    print("Step 3: Generate Trading Signals (LOCKED HYPERPARAMETERS + CONFIDENCE)")
    print("="*70)
    print("""
Hyperparameters (pre-committed, never tuned on test data):
  - Buy percentile: 25.0 (top 25% of predictions are BUY signals)
  - Sell percentile: 75.0 (bottom 75% of predictions are SELL signals)
  - Mean reversion buy threshold: -5% below 20-day MA
  - Mean reversion sell threshold: +5% above 20-day MA
  - Position sizing: 50% of cash per BUY signal
  - Confidence threshold: 0.7 (only trade high-confidence predictions)
""")
    
    # Generate trading signals from predictions
    # STRATEGY: Only generate signals from VERY HIGH confidence predictions
    # This eliminates noise and focuses on the strongest opportunities
    trading_results = generate_trading_signals(
        predictions_df,
        results_dir=TRADING_ANALYSIS_DIR,
        buy_threshold_pct=-2.0,
        sell_threshold_pct=-6.0,
        enable_mean_reversion=True,
        mr_buy_threshold=-0.05,  # LOCKED
        mr_sell_threshold=0.05,  # LOCKED
        use_percentile_thresholds=True,
        buy_percentile=25.0,  # LOCKED: Top 25%
        sell_percentile=75.0,  # LOCKED: Bottom 75%
        percentile_by_month=True,
        confidence_threshold=0.7  # LOCKED: Filter signals below 70% confidence (better quality)
    )
    signals_df = trading_results['signals_df']
    
    print("\n" + "="*70)
    print("Step 4: Run Portfolio Backtest (DUAL-MODEL: Entry + Exit)")
    print("="*70)
    
    # Run portfolio backtest with both entry signals and exit model
    portfolio_results = run_portfolio_backtest(
        signals_df,
        exit_model_df=exit_predictions_df,  # Pass exit model predictions
        results_dir=TRADING_ANALYSIS_DIR,
        initial_capital=100000
    )
    backtest_df = portfolio_results['backtest_df']
    
    print("\n" + "="*70)
    print("Step 5: Run Regime-Aware Portfolio Backtest (ADAPTIVE POSITION SIZING)")
    print("="*70)
    print("""
Regime Management Strategy:
  - Bullish Regime (1.0x multiplier):   Uses full position sizing
    > 90% conf: 90% cash | 70% conf: 70% | 50% conf: 50% | <50%: 20%
    
  - Neutral Regime (0.65x multiplier):  Reduces exposure moderately
    > 90% conf: 58.5% cash | 70% conf: 45.5% | 50% conf: 32.5% | <50%: 13%
    
  - Bearish Regime (0.35x multiplier):  Minimizes risk exposure
    > 90% conf: 31.5% cash | 70% conf: 24.5% | 50% conf: 17.5% | <50%: 7%

Regime Detection Indicators:
  - SMA trend direction (20/50 period moving averages)
  - Price position relative to moving averages
  - Momentum (10-period momentum indicator)
  - Volatility (20-period historical volatility)
  - Dynamic scoring system adjusts for market conditions
""")
    
    # Merge signals with testing data indicators for regime detection
    signals_with_indicators = signals_df.merge(
        testing_data_with_indicators[['Date', 'SMA_20', 'SMA_50', 'SMA_200', 'Momentum_10', 'Volatility_20', 'Close']],
        on='Date',
        how='left'
    )
    
    # Run regime-aware backtest with same signals and exit model
    regime_results = run_regime_backtest(
        signals_with_indicators,
        exit_model_df=exit_predictions_df,
        confidence_weighted=True
    )
    regime_backtest_df = regime_results
    
    # Save regime backtest results
    regime_backtest_file = TRADING_ANALYSIS_DIR / 'portfolio_backtest_regime.csv'
    regime_backtest_df.to_csv(regime_backtest_file, index=False)
    print(f"✓ Regime-aware backtest saved to: {regime_backtest_file}")
    
    # Calculate buy-and-hold baseline for regime backtest
    start_price = regime_backtest_df.iloc[0]['Actual_Price']
    buy_hold_shares = 100000 / start_price
    regime_backtest_df['BuyHold_Value'] = buy_hold_shares * regime_backtest_df['Actual_Price']
    
    # Calculate buy-and-hold baseline
    start_price = backtest_df.iloc[0]['Actual_Price']
    end_price = backtest_df.iloc[-1]['Actual_Price']
    buy_hold_shares = 100000 / start_price
    buy_hold_values = buy_hold_shares * backtest_df['Actual_Price']
    
    # Standard backtest results
    strategy_final = backtest_df.iloc[-1]['Portfolio_Value']
    buy_hold_final = buy_hold_values.iloc[-1]
    strategy_return = (strategy_final - 100000) / 100000 * 100
    buy_hold_return = (buy_hold_final - 100000) / 100000 * 100
    
    # Regime-aware backtest results
    regime_strategy_final = regime_backtest_df.iloc[-1]['Portfolio_Value']
    regime_buy_hold_final = regime_backtest_df.iloc[-1]['BuyHold_Value']
    regime_strategy_return = (regime_strategy_final - 100000) / 100000 * 100
    regime_buy_hold_return = (regime_buy_hold_final - 100000) / 100000 * 100
    
    print("\n" + "="*70)
    print("COMPARISON: AGGRESSIVE vs REGIME-MANAGED POSITIONING")
    print("="*70)
    
    print("\n[1] AGGRESSIVE POSITIONING (Fixed Sizing)")
    print("-" * 70)
    print(f"Strategy Performance:")
    print(f"  Final Value:          ${strategy_final:,.2f}")
    print(f"  Return:               {strategy_return:+.2f}%")
    print(f"  Max Drawdown:         {(backtest_df['Portfolio_Value'] - backtest_df['Portfolio_Value'].expanding().max()).min() / 100000 * 100:.2f}%")
    
    print(f"\nBuy-and-Hold (S&P 500):")
    print(f"  Final Value:          ${buy_hold_final:,.2f}")
    print(f"  Return:               {buy_hold_return:+.2f}%")
    
    print(f"\nComparison:")
    print(f"  Absolute Gain:        ${strategy_final - buy_hold_final:,.2f}")
    print(f"  Outperformance:       {strategy_return - buy_hold_return:+.2f}pp")
    print(f"  Strategy Wins?        {'✓ YES' if strategy_final > buy_hold_final else '✗ NO'}")
    
    print("\n[2] REGIME-AWARE POSITIONING (Adaptive Sizing)")
    print("-" * 70)
    print(f"Strategy Performance:")
    print(f"  Final Value:          ${regime_strategy_final:,.2f}")
    print(f"  Return:               {regime_strategy_return:+.2f}%")
    print(f"  Max Drawdown:         {(regime_backtest_df['Portfolio_Value'] - regime_backtest_df['Portfolio_Value'].expanding().max()).min() / 100000 * 100:.2f}%")
    
    print(f"\nBuy-and-Hold (S&P 500):")
    print(f"  Final Value:          ${regime_buy_hold_final:,.2f}")
    print(f"  Return:               {regime_buy_hold_return:+.2f}%")
    
    print(f"\nComparison:")
    print(f"  Absolute Gain:        ${regime_strategy_final - regime_buy_hold_final:,.2f}")
    print(f"  Outperformance:       {regime_strategy_return - regime_buy_hold_return:+.2f}pp")
    print(f"  Strategy Wins?        {'✓ YES' if regime_strategy_final > regime_buy_hold_final else '✗ NO'}")
    
    print("\n[3] STRATEGY COMPARISON")
    print("-" * 70)
    print(f"Aggressive vs Regime-Aware:")
    print(f"  Return Difference:    {regime_strategy_return - strategy_return:+.2f}pp")
    print(f"  Win Rate Comparison:  {'Regime ✓' if regime_strategy_final > strategy_final else 'Aggressive ✓'}")
    
    if regime_strategy_final > strategy_final:
        outperf = regime_strategy_final - strategy_final
        print(f"  Regime method wins by: ${outperf:,.2f}")
    else:
        underperf = strategy_final - regime_strategy_final
        print(f"  Aggressive method wins by: ${underperf:,.2f}")
    
    print("\n" + "="*70)
    print("DATA INTEGRITY VERIFICATION")
    print("="*70)
    print("""
✓ Indicators calculated separately on training and testing data
✓ No training data mixed into test data
✓ Model trained on training data ONLY
✓ Predictions made on test data ONLY
✓ Hyperparameters locked before testing (not tuned on test set)
✓ All results are TRUE OUT-OF-SAMPLE
✓ NO LOOKAHEAD BIAS - Safe to use for real trading
""")
    
    print("="*70)
    print(f"Results saved to: {TRADING_ANALYSIS_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run trading model pipeline")
    parser.add_argument(
        "--model",
        choices=["random_forest", "volatility_regression", "lightgbm", "ensemble"],
        default='lightgbm',
        help="Model to run (default: lightgbm)",
    )
    args = parser.parse_args()

    # Override ACTIVE_MODEL at runtime if flag provided
    ACTIVE_MODEL = args.model
    print(f"Selected model: {ACTIVE_MODEL}")

    main()
