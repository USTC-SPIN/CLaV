"""
Plotting utilities for RAE-ImLPR training visualization.
Inspired by Ultralytics YOLO plotting system.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Optional
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set default plotting style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Optional seaborn styling
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    # Fallback to matplotlib default grid style
    plt.style.use('seaborn-v0_8-whitegrid') if 'seaborn-v0_8-whitegrid' in plt.style.available else plt.style.use('default')


def plot_training_results(
    csv_file: str,
    save_dir: Optional[Path] = None,
    metrics_to_plot: Optional[List[str]] = None,
    smooth_sigma: float = 3.0
):
    """
    Plot training results from CSV file with smoothing.

    Args:
        csv_file: Path to results CSV file
        save_dir: Directory to save plots (defaults to CSV file directory)
        metrics_to_plot: List of metric names to plot. If None, plots all available metrics
        smooth_sigma: Sigma for Gaussian smoothing

    Returns:
        Path to saved plot file
    """
    import pandas as pd

    csv_path = Path(csv_file)
    save_dir = save_dir or csv_path.parent
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    if not csv_path.exists():
        print(f"Warning: CSV file not found: {csv_path}")
        return None

    data = pd.read_csv(csv_path)

    # Get columns to plot (exclude epoch and time)
    exclude_cols = {'epoch', 'time', 'epoch_time'}
    if metrics_to_plot is None:
        cols_to_plot = [col for col in data.columns if col not in exclude_cols]
    else:
        cols_to_plot = [col for col in metrics_to_plot if col in data.columns]

    if len(cols_to_plot) == 0:
        print("Warning: No metrics to plot")
        return None

    # Determine subplot layout
    n_metrics = len(cols_to_plot)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows), squeeze=False)
    axes = axes.flatten()

    # Plot each metric
    for idx, metric in enumerate(cols_to_plot):
        ax = axes[idx]
        epochs = data['epoch'].values
        values = data[metric].values

        # Plot raw values
        ax.plot(epochs, values, 'o-', label='Raw', alpha=0.6, markersize=3, linewidth=1)

        # Plot smoothed values
        if len(values) > 1:
            smoothed = gaussian_filter1d(values, sigma=smooth_sigma)
            ax.plot(epochs, smoothed, '-', label='Smoothed', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save figure
    save_path = save_dir / 'training_results.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Training results plot saved to: {save_path}")
    return save_path


def plot_losses(
    csv_file: str,
    save_dir: Optional[Path] = None,
    smooth_sigma: float = 3.0
):
    """
    Plot training and validation losses.

    Args:
        csv_file: Path to results CSV file
        save_dir: Directory to save plots
        smooth_sigma: Sigma for Gaussian smoothing

    Returns:
        Path to saved plot file
    """
    import pandas as pd

    csv_path = Path(csv_file)
    save_dir = save_dir or csv_path.parent
    save_dir = Path(save_dir)

    if not csv_path.exists():
        print(f"Warning: CSV file not found: {csv_path}")
        return None

    data = pd.read_csv(csv_path)

    # Find loss columns
    loss_cols = [col for col in data.columns if 'loss' in col.lower()]

    if len(loss_cols) == 0:
        print("Warning: No loss columns found")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = data['epoch'].values

    for loss_col in loss_cols:
        values = data[loss_col].values

        # Plot raw values
        ax.plot(epochs, values, 'o', label=f'{loss_col} (raw)',
                alpha=0.4, markersize=4)

        # Plot smoothed values
        if len(values) > 1:
            smoothed = gaussian_filter1d(values, sigma=smooth_sigma)
            ax.plot(epochs, smoothed, '-', label=f'{loss_col} (smooth)', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Losses', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    save_path = save_dir / 'losses.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Loss plot saved to: {save_path}")
    return save_path


def plot_metrics_comparison(
    csv_file: str,
    train_prefix: str = 'train/',
    val_prefix: str = 'val/',
    save_dir: Optional[Path] = None,
    smooth_sigma: float = 3.0
):
    """
    Plot training vs validation metrics comparison.

    Args:
        csv_file: Path to results CSV file
        train_prefix: Prefix for training metrics
        val_prefix: Prefix for validation metrics
        save_dir: Directory to save plots
        smooth_sigma: Sigma for Gaussian smoothing

    Returns:
        Path to saved plot file
    """
    import pandas as pd

    csv_path = Path(csv_file)
    save_dir = save_dir or csv_path.parent
    save_dir = Path(save_dir)

    if not csv_path.exists():
        print(f"Warning: CSV file not found: {csv_path}")
        return None

    data = pd.read_csv(csv_path)

    # Find train/val metric pairs
    train_cols = [col for col in data.columns if col.startswith(train_prefix)]

    if len(train_cols) == 0:
        print(f"Warning: No training metrics found with prefix '{train_prefix}'")
        return None

    # Create figure
    n_metrics = len(train_cols)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows), squeeze=False)
    axes = axes.flatten()

    epochs = data['epoch'].values

    for idx, train_col in enumerate(train_cols):
        ax = axes[idx]

        # Get metric name without prefix
        metric_name = train_col.replace(train_prefix, '')
        val_col = val_prefix + metric_name

        # Plot training metric
        train_values = data[train_col].values
        ax.plot(epochs, train_values, 'o-', label='Train (raw)',
                alpha=0.4, markersize=3, color='blue')

        if len(train_values) > 1:
            train_smooth = gaussian_filter1d(train_values, sigma=smooth_sigma)
            ax.plot(epochs, train_smooth, '-', label='Train (smooth)',
                    linewidth=2, color='blue')

        # Plot validation metric if exists
        if val_col in data.columns:
            val_values = data[val_col].values
            ax.plot(epochs, val_values, 's-', label='Val (raw)',
                    alpha=0.4, markersize=3, color='orange')

            if len(val_values) > 1:
                val_smooth = gaussian_filter1d(val_values, sigma=smooth_sigma)
                ax.plot(epochs, val_smooth, '-', label='Val (smooth)',
                        linewidth=2, color='orange')

        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save figure
    save_path = save_dir / 'metrics_comparison.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Metrics comparison plot saved to: {save_path}")
    return save_path


def plot_learning_rate(
    csv_file: str,
    save_dir: Optional[Path] = None
):
    """
    Plot learning rate schedule.

    Args:
        csv_file: Path to results CSV file
        save_dir: Directory to save plots

    Returns:
        Path to saved plot file
    """
    import pandas as pd

    csv_path = Path(csv_file)
    save_dir = save_dir or csv_path.parent
    save_dir = Path(save_dir)

    if not csv_path.exists():
        print(f"Warning: CSV file not found: {csv_path}")
        return None

    data = pd.read_csv(csv_path)

    # Find learning rate columns
    lr_cols = [col for col in data.columns if col.startswith('lr/')]

    if len(lr_cols) == 0:
        print("Warning: No learning rate columns found")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = data['epoch'].values

    for lr_col in lr_cols:
        values = data[lr_col].values
        label = lr_col.replace('lr/', 'LR ')
        ax.plot(epochs, values, 'o-', label=label, markersize=4, linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    # Save figure
    save_path = save_dir / 'learning_rate.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Learning rate plot saved to: {save_path}")
    return save_path


def create_all_plots(
    csv_file: str,
    save_dir: Optional[Path] = None,
    smooth_sigma: float = 3.0
):
    """
    Create all available plots from training results.

    Args:
        csv_file: Path to results CSV file
        save_dir: Directory to save plots
        smooth_sigma: Sigma for Gaussian smoothing

    Returns:
        Dictionary of plot types and their saved paths
    """
    csv_path = Path(csv_file)
    save_dir = save_dir or csv_path.parent
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = {}

    # Create all plots
    print("\nGenerating training visualization plots...")

    # Main results plot
    path = plot_training_results(csv_file, save_dir, smooth_sigma=smooth_sigma)
    if path:
        plot_paths['results'] = path

    # Loss plot
    path = plot_losses(csv_file, save_dir, smooth_sigma=smooth_sigma)
    if path:
        plot_paths['losses'] = path

    # Metrics comparison (if train/val split exists)
    path = plot_metrics_comparison(csv_file, save_dir=save_dir, smooth_sigma=smooth_sigma)
    if path:
        plot_paths['comparison'] = path

    # Learning rate plot
    path = plot_learning_rate(csv_file, save_dir)
    if path:
        plot_paths['learning_rate'] = path

    print(f"\nAll plots saved to: {save_dir}")
    return plot_paths


if __name__ == '__main__':
    # Test with example CSV
    import argparse

    parser = argparse.ArgumentParser(description='Plot RAE-ImLPR training results')
    parser.add_argument('--csv', type=str, required=True, help='Path to results CSV file')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save plots')
    parser.add_argument('--smooth-sigma', type=float, default=3.0, help='Gaussian smoothing sigma')

    args = parser.parse_args()

    create_all_plots(args.csv, args.save_dir, args.smooth_sigma)
