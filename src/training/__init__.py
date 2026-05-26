"""
RAE-ImLPR Enhanced Training Module

This module provides enhanced training capabilities with improved logging
and visualization, inspired by Ultralytics YOLO training system.

Main components:
- TrainingLogger: Enhanced logger with colored terminal output and CSV export
- plotting: Professional visualization with Gaussian smoothing
- train: Main training script with enhanced callbacks
- train_utils: Utility functions for training
- experiment_tracker: Experiment tracking and metrics visualization
- visualizer: Denoising effect visualization

Usage:
    python train_5090/train.py --config configs/train_config_joint.yaml
"""

from .logger import TrainingLogger, colorstr, Colors
from .plotting import (
    plot_training_results,
    plot_losses,
    plot_metrics_comparison,
    plot_learning_rate,
    create_all_plots
)
from .train_utils import (
    setup_optimizer,
    setup_scheduler,
    save_checkpoint,
    load_checkpoint,
    Logger,
    AverageMeter,
    count_parameters,
    set_seed,
    EMA
)
from .experiment_tracker import ExperimentTracker
from .visualizer import DenoisingVisualizer

__version__ = '1.1.0'
__all__ = [
    'TrainingLogger',
    'colorstr',
    'Colors',
    'plot_training_results',
    'plot_losses',
    'plot_metrics_comparison',
    'plot_learning_rate',
    'create_all_plots',
    'setup_optimizer',
    'setup_scheduler',
    'save_checkpoint',
    'load_checkpoint',
    'Logger',
    'AverageMeter',
    'count_parameters',
    'set_seed',
    'EMA',
    'ExperimentTracker',
    'DenoisingVisualizer',
]
