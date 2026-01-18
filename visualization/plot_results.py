"""
Visualization module for model performance analysis.
Generates plots showing error metrics over time.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import numpy as np


def plot_error_over_time(results_dict, output_dir=None):
    """
    Plot prediction errors over time for all models.
    
    Args:
        results_dict: Dictionary with model names and their result DataFrames
        output_dir: Directory to save plots (optional)
    """
    if not results_dict:
        print("No results to plot")
        return
    
    # Create output directory if provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up the figure with subplots
    num_models = len(results_dict)
    fig, axes = plt.subplots(num_models, 2, figsize=(16, 5*num_models))
    
    # Handle single model case
    if num_models == 1:
        axes = axes.reshape(1, -1)
    
    # Plot for each model
    for idx, (model_name, model_output) in enumerate(results_dict.items()):
        results_df = model_output['results']
        results_df['Date'] = pd.to_datetime(results_df['Date'])
        
        # Plot 1: Absolute Price Error over time
        ax1 = axes[idx, 0]
        ax1.plot(results_df['Date'], results_df['Abs_Price_Error'], 
                color='#1f77b4', linewidth=2, label='Absolute Error', alpha=0.7)
        ax1.fill_between(results_df['Date'], results_df['Abs_Price_Error'], 
                         alpha=0.3, color='#1f77b4')
        ax1.axhline(y=results_df['Abs_Price_Error'].mean(), 
                   color='red', linestyle='--', linewidth=2, label=f"Mean: ${results_df['Abs_Price_Error'].mean():.2f}")
        ax1.set_title(f'{model_name} - Absolute Price Error Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Absolute Price Error ($)', fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 2: Actual vs Predicted Price
        ax2 = axes[idx, 1]
        ax2.plot(results_df['Date'], results_df['Actual_Price'], 
                color='green', linewidth=2, label='Actual Price', marker='o', markersize=3, alpha=0.7)
        ax2.plot(results_df['Date'], results_df['Predicted_Price'], 
                color='orange', linewidth=2, label='Predicted Price', marker='s', markersize=3, alpha=0.7)
        ax2.set_title(f'{model_name} - Actual vs Predicted Price', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylabel('Price ($)', fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    if output_dir:
        output_file = output_dir / "error_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {output_file.name}")
    
    plt.close()


def plot_direction_accuracy(results_dict, output_dir=None):
    """
    Plot direction prediction accuracy over time (rolling window).
    
    Args:
        results_dict: Dictionary with model names and their result DataFrames
        output_dir: Directory to save plots (optional)
    """
    if not results_dict:
        print("No results to plot")
        return
    
    # Create output directory if provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot for each model
    for model_name, model_output in results_dict.items():
        results_df = model_output['results']
        results_df['Date'] = pd.to_datetime(results_df['Date'])
        
        # Calculate rolling accuracy (20-day window)
        window = 20
        rolling_accuracy = results_df['Direction_Correct'].rolling(window=window).mean() * 100
        
        ax.plot(results_df['Date'], rolling_accuracy, 
               linewidth=2.5, label=f'{model_name} ({window}-day rolling)', marker='o', markersize=4)
    
    # Overall accuracy line
    if len(results_dict) == 1:
        first_results = list(results_dict.values())[0]['results']
        overall_accuracy = first_results['Direction_Correct'].mean() * 100
        ax.axhline(y=overall_accuracy, color='black', linestyle='--', 
                  linewidth=2, label=f'Overall Accuracy: {overall_accuracy:.1f}%')
    
    ax.set_title('Direction Prediction Accuracy Over Time (20-Day Rolling Average)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_ylim([0, 110])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    if output_dir:
        output_file = output_dir / "direction_accuracy.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file.name}")
    
    plt.close()


def plot_error_distribution(results_dict, output_dir=None):
    """
    Plot error distribution histograms for all models.
    
    Args:
        results_dict: Dictionary with model names and their result DataFrames
        output_dir: Directory to save plots (optional)
    """
    if not results_dict:
        print("No results to plot")
        return
    
    # Create output directory if provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    num_models = len(results_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(7*num_models, 5))
    
    # Handle single model case
    if num_models == 1:
        axes = [axes]
    
    # Plot for each model
    for idx, (model_name, model_output) in enumerate(results_dict.items()):
        results_df = model_output['results']
        
        ax = axes[idx]
        ax.hist(results_df['Price_Error'], bins=30, edgecolor='black', 
               color='#1f77b4', alpha=0.7)
        ax.axvline(x=results_df['Price_Error'].mean(), color='red', 
                  linestyle='--', linewidth=2, label=f'Mean: ${results_df["Price_Error"].mean():.2f}')
        ax.axvline(x=results_df['Price_Error'].median(), color='orange', 
                  linestyle='--', linewidth=2, label=f'Median: ${results_df["Price_Error"].median():.2f}')
        ax.set_title(f'{model_name} - Error Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Price Error ($)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save the plot
    if output_dir:
        output_file = output_dir / "error_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file.name}")
    
    plt.close()


def generate_all_plots(results_dict, output_dir=None):
    """
    Generate all visualization plots.
    
    Args:
        results_dict: Dictionary with model names and their result DataFrames
        output_dir: Directory to save plots (optional)
    """
    if not results_dict:
        print("No results to plot")
        return
    
    # Filter results_dict to only include model predictions (with 'results' key)
    # Skip trading signals and other non-model outputs
    model_results = {k: v for k, v in results_dict.items() if isinstance(v, dict) and 'results' in v}
    
    if not model_results:
        print("No model predictions to plot")
        return
    
    print("\n" + "="*60)
    print("Generating Performance Visualizations")
    print("="*60 + "\n")
    
    print("Generating error analysis plots...")
    plot_error_over_time(model_results, output_dir)
    
    print("Generating direction accuracy plots...")
    plot_direction_accuracy(model_results, output_dir)
    
    print("Generating error distribution plots...")
    plot_error_distribution(model_results, output_dir)
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("Visualization module for stock prediction models")
    print("Import and use in main.py with generate_all_plots()")
