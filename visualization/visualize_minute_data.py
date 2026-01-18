#!/usr/bin/env python3
"""
Minute-Level Data Visualization
Visualizes minute-by-minute trading data with predictions and technical indicators
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 9

# Results directory
RESULTS_DIR = Path("/Users/mihirepel/Personal_Projects/Trading/historical_training_v2/results/minute_data_2026")

def load_results():
    """Load all result files"""
    try:
        predictions = pd.read_csv(RESULTS_DIR / "lightgbm_predictions.csv")
        training = pd.read_csv(RESULTS_DIR / "AAPL_minute_training_2026-01-12_to_2026-01-16.csv")
        testing = pd.read_csv(RESULTS_DIR / "AAPL_minute_testing_2026-01-12_to_2026-01-16.csv")
        
        # Convert Date columns
        predictions['Date'] = pd.to_datetime(predictions['Date'])
        training['Date'] = pd.to_datetime(training['Date'])
        testing['Date'] = pd.to_datetime(testing['Date'])
        
        return predictions, training, testing
    except Exception as e:
        print(f"Error loading results: {e}")
        return None, None, None


def plot_price_and_predictions(predictions):
    """Plot actual vs predicted prices with confidence intervals"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Price with predictions
    ax1 = axes[0]
    ax1.plot(predictions['Date'], predictions['Actual_Price'], 'b-', label='Actual Price', linewidth=2)
    ax1.plot(predictions['Date'], predictions['Predicted_Price'], 'r--', label='Predicted Price', alpha=0.7, linewidth=1.5)
    
    # Confidence bands
    ax1.fill_between(
        predictions['Date'],
        predictions['Confidence_Lower_Bound'],
        predictions['Confidence_Upper_Bound'],
        alpha=0.2,
        color='green',
        label='95% Confidence Interval'
    )
    
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.set_title('Minute-Level AAPL Price Predictions (2026-01-15 to 2026-01-16)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction errors
    ax2 = axes[1]
    colors = ['green' if x > 0 else 'red' for x in predictions['Predicted_Change']]
    ax2.bar(predictions['Date'], predictions['Predicted_Change'], color=colors, alpha=0.6, label='Predicted Change')
    ax2.plot(predictions['Date'], predictions['Actual_Change'], 'b-', label='Actual Change', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_ylabel('Price Change ($)', fontsize=11)
    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_title('Price Changes: Actual vs Predicted', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "01_price_predictions.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 01_price_predictions.png")
    plt.close()


def plot_confidence_metrics(predictions):
    """Plot confidence scores and accuracy metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Confidence over time
    ax1 = axes[0, 0]
    high_conf = predictions[predictions['Confidence'] > 0.7]
    low_conf = predictions[predictions['Confidence'] <= 0.7]
    ax1.scatter(low_conf['Date'], low_conf['Confidence'], alpha=0.5, s=30, c='orange', label='Low Confidence (≤0.7)')
    ax1.scatter(high_conf['Date'], high_conf['Confidence'], alpha=0.7, s=30, c='green', label='High Confidence (>0.7)')
    ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax1.set_ylabel('Confidence Score', fontsize=11)
    ax1.set_title('Prediction Confidence Over Time', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.2, 1.0])
    
    # Plot 2: Confidence distribution
    ax2 = axes[0, 1]
    ax2.hist(predictions['Confidence'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='High Confidence Threshold')
    ax2.axvline(x=predictions['Confidence'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {predictions["Confidence"].mean():.2f}')
    ax2.set_xlabel('Confidence Score', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Prediction Confidence', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Direction accuracy
    ax3 = axes[1, 0]
    direction_correct = predictions['Direction_Correct'].value_counts()
    labels = ['Wrong', 'Correct']
    colors_pie = ['#ff6b6b', '#51cf66']
    ax3.pie(direction_correct, labels=labels, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax3.set_title(f'Direction Accuracy: {direction_correct.get(1, 0) / len(predictions) * 100:.1f}% Correct', fontsize=12, fontweight='bold')
    
    # Plot 4: Error magnitude distribution
    ax4 = axes[1, 1]
    ax4.hist(predictions['Abs_Price_Error'], bins=50, color='crimson', alpha=0.7, edgecolor='black')
    ax4.axvline(x=predictions['Abs_Price_Error'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean Error: ${predictions["Abs_Price_Error"].mean():.2f}')
    ax4.set_xlabel('Absolute Price Error ($)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "02_confidence_metrics.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 02_confidence_metrics.png")
    plt.close()


def plot_technical_indicators(training, testing):
    """Plot technical indicators from training and testing data"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Combine training and testing for continuous view
    all_data = pd.concat([training, testing], ignore_index=True)
    
    # Plot 1: Moving averages
    ax1 = axes[0, 0]
    ax1.plot(all_data['Date'], all_data['Close'], 'b-', label='Close Price', linewidth=2)
    if 'SMA_5' in all_data.columns:
        ax1.plot(all_data['Date'], all_data['SMA_5'], 'r--', alpha=0.7, label='SMA 5 min')
    if 'SMA_20' in all_data.columns:
        ax1.plot(all_data['Date'], all_data['SMA_20'], 'g--', alpha=0.7, label='SMA 20 min')
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.set_title('Moving Averages', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: RSI
    ax2 = axes[0, 1]
    if 'RSI_14' in all_data.columns:
        ax2.plot(all_data['Date'], all_data['RSI_14'], 'purple', linewidth=2)
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
        ax2.set_ylabel('RSI', fontsize=11)
        ax2.set_title('Relative Strength Index (14)', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 100])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: MACD
    ax3 = axes[1, 0]
    if 'MACD' in all_data.columns:
        ax3.plot(all_data['Date'], all_data['MACD'], 'b-', label='MACD', linewidth=1.5)
        ax3.plot(all_data['Date'], all_data['Signal_Line'], 'r-', label='Signal Line', linewidth=1.5)
        ax3.bar(all_data['Date'], all_data['MACD_Histogram'], label='MACD Histogram', alpha=0.3, color='gray')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax3.set_ylabel('MACD', fontsize=11)
        ax3.set_title('MACD', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Bollinger Bands
    ax4 = axes[1, 1]
    if 'BB_Upper' in all_data.columns:
        ax4.plot(all_data['Date'], all_data['Close'], 'b-', label='Close Price', linewidth=2)
        ax4.plot(all_data['Date'], all_data['BB_Upper'], 'r--', alpha=0.7, label='Upper Band')
        ax4.plot(all_data['Date'], all_data['BB_Lower'], 'g--', alpha=0.7, label='Lower Band')
        ax4.fill_between(all_data['Date'], all_data['BB_Upper'], all_data['BB_Lower'], alpha=0.1, color='blue')
        ax4.set_ylabel('Price ($)', fontsize=11)
        ax4.set_title('Bollinger Bands', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Volume and Volatility
    ax5 = axes[2, 0]
    ax5_twin = ax5.twinx()
    bars = ax5.bar(all_data['Date'], all_data['Volume'], alpha=0.3, color='steelblue', label='Volume')
    if 'Volatility_20' in all_data.columns:
        line = ax5_twin.plot(all_data['Date'], all_data['Volatility_20'], 'r-', linewidth=2, label='Volatility (20)')
    ax5.set_ylabel('Volume', fontsize=11, color='steelblue')
    ax5_twin.set_ylabel('Volatility', fontsize=11, color='red')
    ax5.set_title('Volume and Volatility', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Momentum
    ax6 = axes[2, 1]
    if 'Momentum_10' in all_data.columns:
        ax6.plot(all_data['Date'], all_data['Momentum_10'], 'b-', label='Momentum (10)', linewidth=2)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax6.fill_between(all_data['Date'], all_data['Momentum_10'], 0, alpha=0.3, color='steelblue')
        ax6.set_ylabel('Momentum', fontsize=11)
        ax6.set_title('Price Momentum', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "03_technical_indicators.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 03_technical_indicators.png")
    plt.close()


def plot_summary_statistics(predictions, training, testing):
    """Plot summary statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Train vs Test accuracy
    ax1 = axes[0, 0]
    train_stats = {
        'Accuracy': len(training[training['Target'] == 1]) / len(training),
        'Up Days': len(training[training['Target'] == 1]),
        'Down Days': len(training[training['Target'] == 0])
    }
    test_stats = {
        'Accuracy': len(testing[testing['Target'] == 1]) / len(testing),
        'Up Days': len(testing[testing['Target'] == 1]),
        'Down Days': len(testing[testing['Target'] == 0])
    }
    
    categories = ['Up Days', 'Down Days']
    train_counts = [train_stats['Up Days'], train_stats['Down Days']]
    test_counts = [test_stats['Up Days'], test_stats['Down Days']]
    
    x = np.arange(len(categories))
    width = 0.35
    ax1.bar(x - width/2, train_counts, width, label='Training', color='skyblue')
    ax1.bar(x + width/2, test_counts, width, label='Testing', color='lightcoral')
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Target Distribution: Training vs Testing', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Key metrics
    ax2 = axes[0, 1]
    ax2.axis('off')
    metrics_text = f"""
    MINUTE-LEVEL TEST SUMMARY (2026-01-12 to 2026-01-16)
    
    Training Data (Mon-Wed, 3 days):
      • Total samples: {len(training):,}
      • Up minutes: {len(training[training['Target'] == 1]):,}
      • Down minutes: {len(training[training['Target'] == 0]):,}
      • Avg daily bars: {len(training) / 3:,.0f}
    
    Testing Data (Thu-Fri, 2 days):
      • Total samples: {len(testing):,}
      • Up minutes: {len(testing[testing['Target'] == 1]):,}
      • Down minutes: {len(testing[testing['Target'] == 0]):,}
      • Avg daily bars: {len(testing) / 2:,.0f}
    
    Predictions (Testing Set):
      • Total predictions: {len(predictions):,}
      • Direction accuracy: {predictions['Direction_Correct'].mean() * 100:.1f}%
      • Avg confidence: {predictions['Confidence'].mean():.2f}
      • High confidence (>0.7): {(predictions['Confidence'] > 0.7).sum():,}
      • Avg prediction error: ${predictions['Abs_Price_Error'].mean():.4f}
    """
    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 3: Price statistics
    ax3 = axes[1, 0]
    all_data = pd.concat([training, testing])
    stats_data = {
        'Training': [training['Close'].min(), training['Close'].mean(), training['Close'].max()],
        'Testing': [testing['Close'].min(), testing['Close'].mean(), testing['Close'].max()]
    }
    x_pos = np.arange(len(stats_data))
    for i, (label, values) in enumerate(stats_data.items()):
        ax3.bar(i, values[1], color=['skyblue', 'lightcoral'][i], label=label, alpha=0.7)
        ax3.plot([i, i], [values[0], values[2]], 'k-', linewidth=2, alpha=0.5)
    ax3.set_ylabel('Price ($)', fontsize=11)
    ax3.set_title('Price Statistics (Min, Mean, Max)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(stats_data.keys())
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Error distribution by confidence
    ax4 = axes[1, 1]
    high_conf_errors = predictions[predictions['Confidence'] > 0.7]['Abs_Price_Error']
    low_conf_errors = predictions[predictions['Confidence'] <= 0.7]['Abs_Price_Error']
    
    bp = ax4.boxplot([high_conf_errors, low_conf_errors], labels=['High Conf (>0.7)', 'Low Conf (≤0.7)'],
                      patch_artist=True)
    colors = ['lightgreen', 'lightsalmon']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax4.set_ylabel('Absolute Error ($)', fontsize=11)
    ax4.set_title('Prediction Error by Confidence Level', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "04_summary_statistics.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 04_summary_statistics.png")
    plt.close()


def main():
    print("\n" + "="*70)
    print("MINUTE-LEVEL DATA VISUALIZATION")
    print("="*70 + "\n")
    
    # Load data
    predictions, training, testing = load_results()
    
    if predictions is None:
        print("Error: Could not load results")
        return 1
    
    print(f"Loaded {len(predictions)} predictions")
    print(f"Loaded {len(training)} training samples")
    print(f"Loaded {len(testing)} testing samples\n")
    
    # Create visualizations
    print("Creating visualizations...")
    plot_price_and_predictions(predictions)
    plot_confidence_metrics(predictions)
    plot_technical_indicators(training, testing)
    plot_summary_statistics(predictions, training, testing)
    
    print(f"\n✓ All visualizations saved to: {RESULTS_DIR}")
    print("\nGenerated plots:")
    print("  1. 01_price_predictions.png - Actual vs predicted prices")
    print("  2. 02_confidence_metrics.png - Confidence analysis")
    print("  3. 03_technical_indicators.png - Technical indicator analysis")
    print("  4. 04_summary_statistics.png - Summary statistics")
    
    return 0


if __name__ == "__main__":
    exit(main())
