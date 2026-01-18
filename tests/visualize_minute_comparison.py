#!/usr/bin/env python3
"""
Comparison Visualization: Late 2025 vs Early 2026 Minute-Level Tests
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 9

results_dir_2025 = Path("results/minute_data_late_2025")
results_dir_2026 = Path("results/minute_data_2026")

def load_predictions(results_dir):
    """Load predictions data"""
    pred_files = glob.glob(str(results_dir / "*predictions*.csv"))
    if not pred_files:
        return None
    return pd.read_csv(pred_files[0])

def create_comparison_charts():
    """Create comparison visualizations"""
    
    # Load data
    pred_2025 = load_predictions(results_dir_2025)
    pred_2026 = load_predictions(results_dir_2026)
    
    if pred_2025 is None or pred_2026 is None:
        print("Error loading data")
        return
    
    # Convert dates
    pred_2025['Date'] = pd.to_datetime(pred_2025['Date'])
    pred_2026['Date'] = pd.to_datetime(pred_2026['Date'])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Accuracy comparison
    ax1 = axes[0, 0]
    periods = ['Late 2025\n(1,950 samples)', 'Early 2026\n(779 samples)']
    accuracies = [
        (pred_2025['Direction_Correct'].sum() / len(pred_2025) * 100),
        (pred_2026['Direction_Correct'].sum() / len(pred_2026) * 100)
    ]
    colors = ['#2ecc71' if acc > 52 else '#e74c3c' for acc in accuracies]
    bars = ax1.bar(periods, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.axhline(y=50, color='gray', linestyle='--', linewidth=2, label='Random Baseline')
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Directional Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylim([45, 65])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Confidence distribution
    ax2 = axes[0, 1]
    conf_bins = np.arange(0.2, 1.1, 0.1)
    ax2.hist(pred_2025['Confidence'], bins=conf_bins, alpha=0.6, label='Late 2025', color='steelblue', edgecolor='black')
    ax2.hist(pred_2026['Confidence'], bins=conf_bins, alpha=0.6, label='Early 2026', color='coral', edgecolor='black')
    ax2.axvline(x=pred_2025['Confidence'].mean(), color='steelblue', linestyle='--', linewidth=2, label=f'2025 Mean: {pred_2025["Confidence"].mean():.2f}')
    ax2.axvline(x=pred_2026['Confidence'].mean(), color='coral', linestyle='--', linewidth=2, label=f'2026 Mean: {pred_2026["Confidence"].mean():.2f}')
    ax2.set_xlabel('Confidence Score', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: High vs Low Confidence Accuracy
    ax3 = axes[0, 2]
    high_conf_2025 = pred_2025[pred_2025['Confidence'] > 0.7]['Direction_Correct'].mean() * 100
    low_conf_2025 = pred_2025[pred_2025['Confidence'] <= 0.7]['Direction_Correct'].mean() * 100
    high_conf_2026 = pred_2026[pred_2026['Confidence'] > 0.7]['Direction_Correct'].mean() * 100
    low_conf_2026 = pred_2026[pred_2026['Confidence'] <= 0.7]['Direction_Correct'].mean() * 100
    
    x = np.arange(2)
    width = 0.35
    ax3.bar(x - width/2, [high_conf_2025, high_conf_2026], width, label='High Conf (>0.7)', color='#2ecc71', alpha=0.7, edgecolor='black')
    ax3.bar(x + width/2, [low_conf_2025, low_conf_2026], width, label='Low Conf (≤0.7)', color='#e74c3c', alpha=0.7, edgecolor='black')
    ax3.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('Accuracy by Confidence Level', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Late 2025', 'Early 2026'])
    ax3.legend()
    ax3.set_ylim([40, 75])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Error magnitude
    ax4 = axes[1, 0]
    error_data = [
        pred_2025['Abs_Price_Error'],
        pred_2026['Abs_Price_Error']
    ]
    bp = ax4.boxplot(error_data, labels=['Late 2025', 'Early 2026'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['steelblue', 'coral']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_ylabel('Absolute Error ($)', fontsize=11)
    ax4.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Sensitivity vs Specificity
    ax5 = axes[1, 1]
    
    # Calculate metrics
    def calc_metrics(pred):
        pred_rise = pred['Predicted_Change'] > 0
        tp = ((pred_rise) & (pred['Actual_Change'] > 0)).sum()
        tn = ((~pred_rise) & (pred['Actual_Change'] <= 0)).sum()
        fp = ((pred_rise) & (pred['Actual_Change'] <= 0)).sum()
        fn = ((~pred_rise) & (pred['Actual_Change'] > 0)).sum()
        sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        return sensitivity, specificity
    
    sens_2025, spec_2025 = calc_metrics(pred_2025)
    sens_2026, spec_2026 = calc_metrics(pred_2026)
    
    x = np.arange(2)
    width = 0.35
    ax5.bar(x - width/2, [sens_2025, sens_2026], width, label='Sensitivity (Catch Ups)', color='#3498db', alpha=0.7, edgecolor='black')
    ax5.bar(x + width/2, [spec_2025, spec_2026], width, label='Specificity (Catch Downs)', color='#e67e22', alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Rate (%)', fontsize=11)
    ax5.set_title('Sensitivity vs Specificity', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(['Late 2025', 'Early 2026'])
    ax5.legend()
    ax5.set_ylim([0, 100])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Summary metrics table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    metrics = [
        ['Metric', 'Late 2025', 'Early 2026', 'Diff'],
        ['Training Size', '1,950', '1,170', '-780'],
        ['Test Size', '1,950', '779', '-1,170'],
        ['Accuracy', f'{accuracies[0]:.1f}%', f'{accuracies[1]:.1f}%', f'{accuracies[1]-accuracies[0]:+.1f}%'],
        ['Avg Confidence', f'{pred_2025["Confidence"].mean():.2f}', f'{pred_2026["Confidence"].mean():.2f}', f'{pred_2026["Confidence"].mean()-pred_2025["Confidence"].mean():+.2f}'],
        ['Sensitivity', f'{sens_2025:.1f}%', f'{sens_2026:.1f}%', f'{sens_2026-sens_2025:+.1f}%'],
        ['Specificity', f'{spec_2025:.1f}%', f'{spec_2026:.1f}%', f'{spec_2026-spec_2025:+.1f}%'],
    ]
    
    table = ax6.table(cellText=metrics, cellLoc='center', loc='center', 
                     colWidths=[0.3, 0.23, 0.23, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(metrics)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    ax6.set_title('Performance Metrics Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(results_dir_2025.parent / "comparison_analysis.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_analysis.png")
    plt.close()


def main():
    print("\n" + "="*70)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*70 + "\n")
    
    create_comparison_charts()
    
    print(f"\n✓ Comparison chart saved to: {results_dir_2025.parent / 'comparison_analysis.png'}")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
