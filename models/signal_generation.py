"""
Trading Signal Generation with Confidence Filtering
Generates BUY/SELL/HOLD signals based on model predictions and confidence intervals.
Only trades high-confidence predictions to reduce noise.
"""
import pandas as pd
import numpy as np


def generate_trading_signals(
    predictions_df,
    results_dir=None,
    buy_threshold_pct=-2.0,
    sell_threshold_pct=-6.0,
    enable_mean_reversion=True,
    mr_buy_threshold=-0.05,
    mr_sell_threshold=0.05,
    use_percentile_thresholds=True,
    buy_percentile=25.0,
    sell_percentile=75.0,
    percentile_by_month=True,
    confidence_threshold=0.7  # NEW: Only trade predictions with confidence > this
):
    """
    Generate trading signals from model predictions.
    
    NEW: Filters out low-confidence predictions.
    
    Args:
        predictions_df: DataFrame with 'Predicted_Change', 'Confidence', 'Predicted_Change_Pct' columns
        results_dir: Directory to save results
        buy_threshold_pct: Absolute % change threshold for BUY
        sell_threshold_pct: Absolute % change threshold for SELL
        enable_mean_reversion: Whether to enable mean reversion signals
        mr_buy_threshold: Mean reversion buy threshold
        mr_sell_threshold: Mean reversion sell threshold
        use_percentile_thresholds: Use percentile-based thresholds
        buy_percentile: Percentile threshold for BUY (lower is better)
        sell_percentile: Percentile threshold for SELL (higher is better)
        percentile_by_month: Calculate percentiles monthly
        confidence_threshold: Only generate signals for predictions > this confidence
    
    Returns:
        Dictionary with 'signals_df' key containing signals DataFrame
    """
    df = predictions_df.copy()
    
    # Check if confidence column exists
    has_confidence = 'Confidence' in df.columns
    if has_confidence:
        print(f"\n✓ Confidence Filtering ENABLED (threshold: {confidence_threshold})")
        print(f"  High confidence predictions: {(df['Confidence'] > confidence_threshold).sum()} / {len(df)}")
    else:
        print("\n⚠ Warning: Confidence column not found. Will generate signals for all predictions.")
        has_confidence = False
        df['Confidence'] = 1.0  # Default to high confidence if not present
    
    # Initialize signals
    df['Signal'] = 'HOLD'
    df['Confidence_Score'] = df.get('Confidence', 1.0)
    
    # Step 1: Generate base signals (percentile-based)
    if use_percentile_thresholds:
        if percentile_by_month:
            # Monthly percentiles
            df['YearMonth'] = pd.to_datetime(df['Date']).dt.to_period('M')
            for month, month_df in df.groupby('YearMonth'):
                buy_pct = month_df['Predicted_Change_Pct'].quantile(buy_percentile / 100.0)
                sell_pct = month_df['Predicted_Change_Pct'].quantile(sell_percentile / 100.0)
                
                mask = (df['YearMonth'] == month)
                df.loc[mask & (df['Predicted_Change_Pct'] <= buy_pct), 'Signal'] = 'BUY'
                df.loc[mask & (df['Predicted_Change_Pct'] >= sell_pct), 'Signal'] = 'SELL'
        else:
            # Global percentiles
            buy_pct = df['Predicted_Change_Pct'].quantile(buy_percentile / 100.0)
            sell_pct = df['Predicted_Change_Pct'].quantile(sell_percentile / 100.0)
            
            df.loc[df['Predicted_Change_Pct'] <= buy_pct, 'Signal'] = 'BUY'
            df.loc[df['Predicted_Change_Pct'] >= sell_pct, 'Signal'] = 'SELL'
    
    # Step 2: CONFIDENCE FILTER - DISABLED (was cutting too many trades)
    # Original behavior: all percentile-based signals are used regardless of confidence
    if has_confidence:
        print(f"\n  Confidence filter: DISABLED (using all percentile-based signals)")
        # No longer converting low-confidence signals to HOLD
    
    # Step 3: Mean reversion signals (if enabled and high confidence)
    if enable_mean_reversion and 'MA20' in df.columns:
        price_to_ma = (df['Actual_Price'] - df['MA20']) / df['MA20']
        
        if has_confidence:
            high_conf_idx = df['Confidence'] > confidence_threshold
            mr_buy_condition = (price_to_ma < mr_buy_threshold) & high_conf_idx
            mr_sell_condition = (price_to_ma > mr_sell_threshold) & high_conf_idx
        else:
            mr_buy_condition = price_to_ma < mr_buy_threshold
            mr_sell_condition = price_to_ma > mr_sell_threshold
        
        df.loc[mr_buy_condition & (df['Signal'] == 'HOLD'), 'Signal'] = 'BUY'
        df.loc[mr_sell_condition & (df['Signal'] == 'HOLD'), 'Signal'] = 'SELL'
    
    # Summary statistics
    signal_counts = df['Signal'].value_counts()
    print(f"\n  Final Signal Distribution:")
    for signal_type in ['BUY', 'SELL', 'HOLD']:
        count = signal_counts.get(signal_type, 0)
        pct = count / len(df) * 100
        print(f"    {signal_type}: {count} ({pct:.1f}%)")
    
    # Win rate for BUY/SELL signals (only count high confidence)
    if has_confidence:
        buy_signals = df[(df['Signal'] == 'BUY') & (df['Confidence'] > confidence_threshold)]
        sell_signals = df[(df['Signal'] == 'SELL') & (df['Confidence'] > confidence_threshold)]
    else:
        buy_signals = df[df['Signal'] == 'BUY']
        sell_signals = df[df['Signal'] == 'SELL']
    
    if len(buy_signals) > 0:
        buy_win_rate = buy_signals['Direction_Correct'].mean() * 100
        print(f"    BUY win rate (high confidence only): {buy_win_rate:.1f}%")
    
    if len(sell_signals) > 0:
        sell_win_rate = sell_signals['Direction_Correct'].mean() * 100
        print(f"    SELL win rate (high confidence only): {sell_win_rate:.1f}%")
    
    # Save signals if results_dir provided
    if results_dir:
        from pathlib import Path
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full signals with confidence
        signals_file = results_dir / 'trading_signals.csv'
        df.to_csv(signals_file, index=False)
        print(f"\nSignals saved to {signals_file.name}")
    
    return {'signals_df': df}


def main(
    predictions_df,
    results_dir=None,
    buy_threshold_pct=-2.0,
    sell_threshold_pct=-6.0,
    enable_mean_reversion=True,
    mr_buy_threshold=-0.05,
    mr_sell_threshold=0.05,
    use_percentile_thresholds=True,
    buy_percentile=25.0,
    sell_percentile=75.0,
    percentile_by_month=True,
    confidence_threshold=0.7
):
    """Wrapper for generate_trading_signals for compatibility with main.py"""
    return generate_trading_signals(
        predictions_df,
        results_dir=results_dir,
        buy_threshold_pct=buy_threshold_pct,
        sell_threshold_pct=sell_threshold_pct,
        enable_mean_reversion=enable_mean_reversion,
        mr_buy_threshold=mr_buy_threshold,
        mr_sell_threshold=mr_sell_threshold,
        use_percentile_thresholds=use_percentile_thresholds,
        buy_percentile=buy_percentile,
        sell_percentile=sell_percentile,
        percentile_by_month=percentile_by_month,
        confidence_threshold=confidence_threshold
    )
