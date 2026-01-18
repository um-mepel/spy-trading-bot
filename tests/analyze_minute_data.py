#!/usr/bin/env python3
"""
Extended Analysis: Minute-Level Data (Late 2025 vs Early 2026)
Compares the performance across the two test periods
"""

import pandas as pd
import numpy as np
from pathlib import Path

results_dir_2025 = Path("results/minute_data_late_2025")
results_dir_2026 = Path("results/minute_data_2026")

def analyze_results(results_dir, period_name):
    """Analyze results from a test period"""
    try:
        predictions = pd.read_csv(results_dir / "lightgbm_predictions.csv")
        training = pd.read_csv(results_dir / "AAPL_minute_training_2025-12-08_to_2025-12-12.csv" if "2025" in results_dir.name else results_dir / "AAPL_minute_training_2026-01-12_to_2026-01-16.csv")
        testing = pd.read_csv(results_dir / "AAPL_minute_testing_2025-12-15_to_2025-12-19.csv" if "2025" in results_dir.name else results_dir / "AAPL_minute_testing_2026-01-12_to_2026-01-16.csv")
    except FileNotFoundError:
        # Try alternative naming
        import glob
        train_files = glob.glob(str(results_dir / "*training*.csv"))
        test_files = glob.glob(str(results_dir / "*testing*.csv"))
        pred_files = glob.glob(str(results_dir / "*predictions*.csv"))
        
        if not train_files or not test_files or not pred_files:
            return None
        
        training = pd.read_csv(train_files[0])
        testing = pd.read_csv(test_files[0])
        predictions = pd.read_csv(pred_files[0])
    
    # Convert dates
    predictions['Date'] = pd.to_datetime(predictions['Date'])
    training['Date'] = pd.to_datetime(training['Date'])
    testing['Date'] = pd.to_datetime(testing['Date'])
    
    # Calculate metrics
    correct_direction = predictions['Direction_Correct'].sum()
    total_preds = len(predictions)
    accuracy = correct_direction / total_preds * 100
    
    high_conf = predictions[predictions['Confidence'] > 0.7]
    low_conf = predictions[predictions['Confidence'] <= 0.7]
    
    high_conf_acc = high_conf['Direction_Correct'].mean() * 100 if len(high_conf) > 0 else 0
    low_conf_acc = low_conf['Direction_Correct'].mean() * 100 if len(low_conf) > 0 else 0
    
    error_pct = (predictions['Abs_Price_Error'] / predictions['Actual_Price'] * 100).mean()
    
    # Price movements
    predictions['Is_Price_Rise'] = predictions['Predicted_Change'] > 0
    actual_rises = (predictions['Actual_Change'] > 0).sum()
    predicted_rises = predictions['Is_Price_Rise'].sum()
    
    tp = ((predictions['Is_Price_Rise']) & (predictions['Actual_Change'] > 0)).sum()
    tn = ((~predictions['Is_Price_Rise']) & (predictions['Actual_Change'] <= 0)).sum()
    fp = ((predictions['Is_Price_Rise']) & (predictions['Actual_Change'] <= 0)).sum()
    fn = ((~predictions['Is_Price_Rise']) & (predictions['Actual_Change'] > 0)).sum()
    
    sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    
    return {
        'period': period_name,
        'train_samples': len(training),
        'test_samples': len(testing),
        'predictions': len(predictions),
        'accuracy': accuracy,
        'high_conf_count': len(high_conf),
        'high_conf_accuracy': high_conf_acc,
        'low_conf_accuracy': low_conf_acc,
        'avg_confidence': predictions['Confidence'].mean(),
        'error_pct': error_pct,
        'avg_error': predictions['Abs_Price_Error'].mean(),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'actual_rises': actual_rises / total_preds * 100,
        'predicted_rises': predicted_rises / total_preds * 100
    }


def main():
    print("\n" + "="*90)
    print("MINUTE-LEVEL MODEL: EXTENDED 2-WEEK TEST ANALYSIS")
    print("="*90)
    
    # Analyze both periods
    results_2025 = analyze_results(results_dir_2025, "Late 2025 (Dec 8-19)")
    results_2026 = analyze_results(results_dir_2026, "Early 2026 (Jan 12-16)")
    
    if results_2025 is None:
        print("⚠️  Could not load 2025 results")
        return 1
    
    if results_2026 is None:
        print("⚠️  Could not load 2026 results")
        return 1
    
    print("\n📊 COMPARISON: LATE 2025 vs EARLY 2026")
    print("-" * 90)
    
    print(f"\n{'Metric':<30} {'Late 2025':<20} {'Early 2026':<20} {'Difference':<15}")
    print("-" * 90)
    
    # Training data
    print(f"{'Training Samples':<30} {results_2025['train_samples']:<20} {results_2026['train_samples']:<20} {results_2026['train_samples'] - results_2025['train_samples']:+.0f}")
    print(f"{'Testing Samples':<30} {results_2025['test_samples']:<20} {results_2026['test_samples']:<20} {results_2026['test_samples'] - results_2025['test_samples']:+.0f}")
    
    # Predictions
    print(f"{'Total Predictions':<30} {results_2025['predictions']:<20} {results_2026['predictions']:<20} {results_2026['predictions'] - results_2025['predictions']:+.0f}")
    
    # Accuracy
    print(f"\n{'Directional Accuracy':<30} {results_2025['accuracy']:.1f}%{'':<15} {results_2026['accuracy']:.1f}%{'':<15} {results_2026['accuracy'] - results_2025['accuracy']:+.1f}%")
    
    # Confidence
    print(f"{'Avg Confidence Score':<30} {results_2025['avg_confidence']:.2f}{'':<17} {results_2026['avg_confidence']:.2f}{'':<17} {results_2026['avg_confidence'] - results_2025['avg_confidence']:+.2f}")
    print(f"{'High Confidence (>0.7)':<30} {results_2025['high_conf_count']:<20} {results_2026['high_conf_count']:<20} {results_2026['high_conf_count'] - results_2025['high_conf_count']:+.0f}")
    
    # Accuracy by confidence
    print(f"\n{'High Confidence Accuracy':<30} {results_2025['high_conf_accuracy']:.1f}%{'':<15} {results_2026['high_conf_accuracy']:.1f}%{'':<15} {results_2026['high_conf_accuracy'] - results_2025['high_conf_accuracy']:+.1f}%")
    print(f"{'Low Confidence Accuracy':<30} {results_2025['low_conf_accuracy']:.1f}%{'':<15} {results_2026['low_conf_accuracy']:.1f}%{'':<15} {results_2026['low_conf_accuracy'] - results_2025['low_conf_accuracy']:+.1f}%")
    
    # Errors
    print(f"\n{'Avg Price Error ($)':<30} ${results_2025['avg_error']:.4f}{'':<14} ${results_2026['avg_error']:.4f}{'':<14} ${results_2026['avg_error'] - results_2025['avg_error']:+.4f}")
    print(f"{'Avg Error (% of price)':<30} {results_2025['error_pct']:.3f}%{'':<14} {results_2026['error_pct']:.3f}%{'':<14} {results_2026['error_pct'] - results_2025['error_pct']:+.3f}%")
    
    # Sensitivity / Specificity
    print(f"\n{'Sensitivity (True Pos Rate)':<30} {results_2025['sensitivity']:.1f}%{'':<15} {results_2026['sensitivity']:.1f}%{'':<15} {results_2026['sensitivity'] - results_2025['sensitivity']:+.1f}%")
    print(f"{'Specificity (True Neg Rate)':<30} {results_2025['specificity']:.1f}%{'':<15} {results_2026['specificity']:.1f}%{'':<15} {results_2026['specificity'] - results_2025['specificity']:+.1f}%")
    
    # Predictions distribution
    print(f"\n{'Actual Up Moves (%)':<30} {results_2025['actual_rises']:.1f}%{'':<15} {results_2026['actual_rises']:.1f}%{'':<15} {results_2026['actual_rises'] - results_2025['actual_rises']:+.1f}%")
    print(f"{'Predicted Up Moves (%)':<30} {results_2025['predicted_rises']:.1f}%{'':<15} {results_2026['predicted_rises']:.1f}%{'':<15} {results_2026['predicted_rises'] - results_2025['predicted_rises']:+.1f}%")
    
    print("\n" + "="*90)
    print("KEY FINDINGS")
    print("="*90)
    
    findings = []
    
    # Finding 1: Accuracy consistency
    acc_diff = abs(results_2026['accuracy'] - results_2025['accuracy'])
    if acc_diff < 2:
        findings.append(f"✓ CONSISTENT EDGE: Both periods show {results_2025['accuracy']:.1f}% accuracy (±{acc_diff:.1f}%)")
    elif results_2026['accuracy'] > results_2025['accuracy']:
        findings.append(f"✓ IMPROVING: 2026 accuracy ({results_2026['accuracy']:.1f}%) better than 2025 ({results_2025['accuracy']:.1f}%)")
    else:
        findings.append(f"⚠️  DEGRADING: 2026 accuracy ({results_2026['accuracy']:.1f}%) worse than 2025 ({results_2025['accuracy']:.1f}%)")
    
    # Finding 2: Confidence reliability
    conf_diff_2025 = results_2025['high_conf_accuracy'] - results_2025['low_conf_accuracy']
    conf_diff_2026 = results_2026['high_conf_accuracy'] - results_2026['low_conf_accuracy']
    
    avg_conf_diff = (conf_diff_2025 + conf_diff_2026) / 2
    if avg_conf_diff > 5:
        findings.append(f"✓ RELIABLE CONFIDENCE: High vs Low accuracy gap averages {avg_conf_diff:.1f}% (consistent signal)")
    else:
        findings.append(f"⚠️  WEAK CONFIDENCE: High vs Low accuracy gap only {avg_conf_diff:.1f}% (unreliable)")
    
    # Finding 3: Sample size effect
    if results_2025['predictions'] > 1000 and results_2026['predictions'] < 1000:
        findings.append(f"📊 SAMPLE SIZE: 2025 test ({results_2025['predictions']} samples) provides more statistical significance")
    elif results_2025['predictions'] < 1000 and results_2026['predictions'] > 1000:
        findings.append(f"📊 SAMPLE SIZE: 2026 test ({results_2026['predictions']} samples) provides more statistical significance")
    else:
        findings.append(f"📊 SAMPLE SIZE: Both periods have similar sample sizes (~{(results_2025['predictions'] + results_2026['predictions'])/2:.0f})")
    
    # Finding 4: Model bias
    bias_2025 = results_2025['predicted_rises'] - results_2025['actual_rises']
    bias_2026 = results_2026['predicted_rises'] - results_2026['actual_rises']
    
    if abs(bias_2025) < 10 and abs(bias_2026) < 10:
        findings.append(f"✓ BALANCED MODEL: Predictions align with market (~50% up moves)")
    else:
        avg_bias = (abs(bias_2025) + abs(bias_2026)) / 2
        findings.append(f"⚠️  BIASED MODEL: Model predicts {avg_bias:.0f}% fewer up moves than market shows")
    
    # Finding 5: Stability
    error_stability = abs(results_2026['error_pct'] - results_2025['error_pct'])
    if error_stability < 0.02:
        findings.append(f"✓ STABLE PREDICTIONS: Error rates consistent across periods ({error_stability:.3f}% variance)")
    else:
        findings.append(f"⚠️  UNSTABLE: Error rates vary {error_stability:.3f}% between periods")
    
    for i, finding in enumerate(findings, 1):
        print(f"{i}. {finding}\n")
    
    print("="*90)
    print("\n📈 TRADING RECOMMENDATIONS")
    print("-"*90)
    
    # Recommendation logic
    if results_2025['accuracy'] > 52 and results_2026['accuracy'] > 52:
        print("✓ EDGE IS REAL: Both periods >52% accuracy suggests genuine predictive power")
        print("  → Recommend: Go live with tight risk management (0.5% position sizing)")
    elif results_2025['accuracy'] > 51 or results_2026['accuracy'] > 51:
        print("~ EDGE IS MARGINAL: One period shows edge, needs more testing")
        print("  → Recommend: Test on 4+ more weeks before live trading")
    else:
        print("✗ NO EDGE DETECTED: Both periods ~50% (random)")
        print("  → Recommend: Improve model with different features or longer lookback")
    
    # Confidence recommendation
    if results_2025['high_conf_accuracy'] > 55 and results_2026['high_conf_accuracy'] > 55:
        print("\n✓ HIGH-CONFIDENCE SIGNALS RELIABLE: Can be trusted for trade execution")
        print(f"  → Only trade signals with confidence >0.7 ({(results_2025['high_conf_count'] + results_2026['high_conf_count'])/2:.0f} signals/week)")
    else:
        print("\n⚠️  CONFIDENCE NOT PREDICTIVE: Filter isn't helping much")
        print("  → Consider removing confidence filter or recalibrating threshold")
    
    print("\n" + "="*90)

if __name__ == "__main__":
    exit(main())
