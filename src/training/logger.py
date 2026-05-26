"""
Enhanced logging module for RAE-ImLPR training.
Inspired by Ultralytics YOLO logging system with callbacks and structured logging.
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import csv


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def colorstr(*args):
    """
    Return colored string for terminal output.

    Args:
        *args: Color names and string to color

    Example:
        >>> colorstr('blue', 'bold', 'hello world')
        >>> colorstr('red', 'warning message')
    """
    *colors, string = args if len(args) > 1 else ('blue', 'bold', args[0])

    color_map = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'bold': '\033[1m',
        'underline': '\033[4m',
    }

    return ''.join(color_map.get(x, '') for x in colors) + str(string) + '\033[0m'


class TrainingLogger:
    """
    Enhanced training logger with CSV logging, console output, and callback support.
    Similar to Ultralytics BaseTrainer logging system.
    """

    def __init__(
        self,
        save_dir: Path,
        resume: bool = False,
        verbose: bool = True
    ):
        """
        Initialize training logger.

        Args:
            save_dir: Directory to save logs and results
            resume: Whether resuming from checkpoint
            verbose: Whether to print detailed logs
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.verbose = verbose

        # Setup CSV file
        self.csv_file = self.save_dir / 'results.csv'
        self.csv_header = None
        self.csv_written = False

        # Setup text log file
        self.log_file = self.save_dir / 'train.log'
        self._setup_file_logger()

        # Training state
        self.epoch = 0
        self.train_start_time = None
        self.epoch_start_time = None

        # Metrics history
        self.metrics_history = []

        print(colorstr('bright_blue', 'bold', f'\nLogging to: {self.save_dir}'))
        if self.csv_file.exists() and not resume:
            print(colorstr('yellow', f'Warning: Existing results file will be overwritten: {self.csv_file}'))

    def _setup_file_logger(self):
        """Setup file logger for detailed logging"""
        self.file_logger = logging.getLogger(f'RAE_{id(self)}')
        self.file_logger.setLevel(logging.INFO)

        # Remove existing handlers
        self.file_logger.handlers.clear()

        # File handler
        fh = logging.FileHandler(self.log_file, mode='a' if self.resume else 'w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        self.file_logger.addHandler(fh)

    def info(self, message: str):
        """Log info message"""
        self.file_logger.info(message)
        if self.verbose:
            print(message)

    def warning(self, message: str):
        """Log warning message"""
        self.file_logger.warning(message)
        if self.verbose:
            print(colorstr('yellow', f'WARNING: {message}'))

    def error(self, message: str):
        """Log error message"""
        self.file_logger.error(message)
        print(colorstr('red', 'bold', f'ERROR: {message}'))

    def on_train_start(self, epochs: int, model_info: Optional[Dict] = None):
        """
        Called when training starts.

        Args:
            epochs: Total number of epochs
            model_info: Optional dictionary with model information
        """
        self.train_start_time = time.time()

        msg = f'\n{colorstr("bright_blue", "bold", "="*70)}\n'
        msg += colorstr('bright_green', 'bold', 'Training Started\n')
        msg += f'{colorstr("bright_blue", "bold", "="*70)}\n'
        msg += f'{colorstr("cyan", "Total Epochs:")} {epochs}\n'
        msg += f'{colorstr("cyan", "Save Directory:")} {self.save_dir}\n'
        msg += f'{colorstr("cyan", "Start Time:")} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'

        if model_info:
            msg += f'\n{colorstr("bright_blue", "Model Information:")}\n'
            for key, value in model_info.items():
                msg += f'  {colorstr("cyan", key)}: {value}\n'

        msg += f'{colorstr("bright_blue", "bold", "="*70)}\n'

        print(msg)
        self.file_logger.info('Training started')
        if model_info:
            self.file_logger.info(f'Model info: {model_info}')

    def on_epoch_start(self, epoch: int):
        """
        Called when epoch starts.

        Args:
            epoch: Current epoch number
        """
        self.epoch = epoch
        self.epoch_start_time = time.time()

    def log_metrics(self, metrics: Dict[str, Any], phase: str = 'train'):
        """
        Log metrics for current epoch.

        Args:
            metrics: Dictionary of metric names and values
            phase: Training phase ('train' or 'val')
        """
        # Add phase prefix to metrics
        prefixed_metrics = {f'{phase}/{k}': v for k, v in metrics.items()}

        # Store in history
        self.metrics_history.append({
            'epoch': self.epoch,
            'phase': phase,
            **prefixed_metrics
        })

    def on_epoch_end(self, epoch_metrics: Dict[str, Any]):
        """
        Called when epoch ends. Saves metrics to CSV and displays summary.

        Args:
            epoch_metrics: Dictionary containing all metrics for this epoch
        """
        # Calculate epoch time
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        total_time = time.time() - self.train_start_time if self.train_start_time else 0

        # Add timing information
        epoch_metrics['epoch'] = self.epoch
        epoch_metrics['time'] = total_time
        epoch_metrics['epoch_time'] = epoch_time

        # Save to CSV
        self._save_to_csv(epoch_metrics)

        # Display epoch summary
        self._display_epoch_summary(epoch_metrics)

    def _save_to_csv(self, metrics: Dict[str, Any]):
        """Save metrics to CSV file"""
        # Initialize CSV header on first write
        if not self.csv_written:
            self.csv_header = list(metrics.keys())

            # Write header if not resuming or file doesn't exist
            if not self.resume or not self.csv_file.exists():
                with open(self.csv_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.csv_header)
                    writer.writeheader()

            self.csv_written = True

        # Append metrics
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_header)
            writer.writerow(metrics)

    def _display_epoch_summary(self, metrics: Dict[str, Any]):
        """Display epoch summary in console"""
        epoch = metrics.get('epoch', 0)
        epoch_time = metrics.get('epoch_time', 0)

        # Build summary string
        summary = f'\n{colorstr("bright_blue", "bold", f"Epoch {epoch} Summary")}\n'
        summary += colorstr('bright_blue', '-' * 50) + '\n'

        # Group metrics by category
        train_metrics = {k.replace('train/', ''): v for k, v in metrics.items()
                        if k.startswith('train/')}
        val_metrics = {k.replace('val/', ''): v for k, v in metrics.items()
                      if k.startswith('val/')}
        lr_metrics = {k: v for k, v in metrics.items() if k.startswith('lr/')}

        # Display training metrics
        if train_metrics:
            summary += colorstr('green', 'Training Metrics:\n')
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    summary += f'  {key}: {value:.6f}\n'
                else:
                    summary += f'  {key}: {value}\n'

        # Display validation metrics
        if val_metrics:
            summary += colorstr('cyan', '\nValidation Metrics:\n')
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    summary += f'  {key}: {value:.6f}\n'
                else:
                    summary += f'  {key}: {value}\n'

        # Display learning rate
        if lr_metrics:
            summary += colorstr('yellow', '\nLearning Rate:\n')
            for key, value in lr_metrics.items():
                summary += f'  {key}: {value:.8f}\n'

        # Display timing
        summary += colorstr('bright_magenta', f'\nEpoch Time: {epoch_time:.2f}s\n')
        summary += colorstr('bright_blue', '-' * 50) + '\n'

        print(summary)

    def on_train_end(self, final_stats: Optional[Dict] = None):
        """
        Called when training ends.

        Args:
            final_stats: Optional dictionary with final statistics
        """
        total_time = time.time() - self.train_start_time if self.train_start_time else 0
        hours = total_time / 3600

        msg = f'\n{colorstr("bright_blue", "bold", "="*70)}\n'
        msg += colorstr('bright_green', 'bold', 'Training Completed!\n')
        msg += f'{colorstr("bright_blue", "bold", "="*70)}\n'
        msg += f'{colorstr("cyan", "Total Time:")} {hours:.3f} hours ({total_time:.1f}s)\n'
        msg += f'{colorstr("cyan", "Results saved to:")} {self.save_dir}\n'

        if final_stats:
            msg += f'\n{colorstr("bright_blue", "Final Statistics:")}\n'
            for key, value in final_stats.items():
                msg += f'  {colorstr("cyan", key)}: {value}\n'

        msg += f'{colorstr("bright_blue", "bold", "="*70)}\n'

        print(msg)
        self.file_logger.info(f'Training completed in {hours:.3f} hours')
        if final_stats:
            self.file_logger.info(f'Final stats: {final_stats}')

    def save_config(self, config: Dict):
        """
        Save training configuration to YAML file.

        Args:
            config: Configuration dictionary
        """
        import yaml

        config_file = self.save_dir / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        self.info(f'Configuration saved to: {config_file}')

    def get_csv_path(self) -> Path:
        """Return path to CSV results file"""
        return self.csv_file

    def get_log_path(self) -> Path:
        """Return path to log file"""
        return self.log_file


class ProgressTracker:
    """Track and display training progress with progress bar"""

    def __init__(self, total_epochs: int, batches_per_epoch: int):
        """
        Initialize progress tracker.

        Args:
            total_epochs: Total number of epochs
            batches_per_epoch: Number of batches per epoch
        """
        self.total_epochs = total_epochs
        self.batches_per_epoch = batches_per_epoch
        self.current_epoch = 0
        self.current_batch = 0

    def update_epoch(self, epoch: int):
        """Update current epoch"""
        self.current_epoch = epoch

    def update_batch(self, batch: int, loss: float):
        """
        Update current batch and display progress.

        Args:
            batch: Current batch number
            loss: Current loss value
        """
        self.current_batch = batch
        progress = (batch + 1) / self.batches_per_epoch
        bar_length = 30
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)

        msg = f'\rEpoch {self.current_epoch}/{self.total_epochs} '
        msg += f'[{bar}] {progress*100:>5.1f}% '
        msg += f'Batch {batch+1}/{self.batches_per_epoch} '
        msg += f'Loss: {loss:.4f}'

        sys.stdout.write(msg)
        sys.stdout.flush()

    def finish_epoch(self):
        """Finish current epoch"""
        sys.stdout.write('\n')
        sys.stdout.flush()


if __name__ == '__main__':
    # Example usage
    save_dir = Path('runs/test_logging')
    logger = TrainingLogger(save_dir, verbose=True)

    # Simulate training
    logger.on_train_start(
        epochs=10,
        model_info={
            'name': 'RAE-ImLPR',
            'params': '1.2M',
            'device': 'cuda:0'
        }
    )

    for epoch in range(1, 11):
        logger.on_epoch_start(epoch)

        # Simulate epoch metrics
        import random
        metrics = {
            'train/loss': random.uniform(0.5, 1.0),
            'train/active_triplets': random.randint(100, 500),
            'val/loss': random.uniform(0.4, 0.9),
            'val/recall@1': random.uniform(0.7, 0.95),
            'lr/pg0': 0.001 * (0.95 ** epoch)
        }

        logger.on_epoch_end(metrics)

    logger.on_train_end(final_stats={
        'best_recall@1': 0.95,
        'best_epoch': 8
    })

    print(f'\nLogs saved to: {save_dir}')
