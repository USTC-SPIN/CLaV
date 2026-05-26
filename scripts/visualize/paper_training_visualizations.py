"""
Paper Training Visualization Generator
Creates publication-quality visualizations for BEV denoising and geo-localization training
Generated: 2025-11-13
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive mode to avoid errors

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns
from PIL import Image

# Set publication-quality defaults
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts for editing
plt.rcParams['ps.fonttype'] = 42

# Colorblind-friendly palette
COLORS = {
    'kitti': '#0173B2',      # Blue
    'nclt': '#DE8F05',       # Orange
    'boreas': '#029E73',     # Green
    'stage2': '#CC78BC',     # Purple
    'stage3': '#CA9161',     # Brown
}

class TrainingVisualizer:
    """Generate paper-quality training visualizations"""

    def __init__(self, base_results_dir: str = '/data/users/cxw/pro/clav/results'):
        self.base_dir = Path(base_results_dir)
        self.output_dir = Path('/data/users/cxw/pro/clav/Figure/paper_visualizations')
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Dataset configurations
        self.datasets = {
            'KITTI': {
                'stage2': 'kitti_stage2_flow_matching_20251108_1724',
                'stage3': 'kitti_descriptor_training_20251109_0347',
            },
            'NCLT': {
                'stage2': 'nclt_stage2_training_20251107_0025',
                'stage3': None,  # Not available
            },
            'Boreas': {
                'stage2': 'boreas_stage2_flow_matching_optimized_20251113_1242',
                'stage3': 'boreas_descriptor_training_optimized_20251112_1414',
            }
        }

    def load_metrics(self, dataset: str, stage: str) -> Dict:
        """Load training metrics from JSON file"""
        exp_name = self.datasets[dataset][stage]
        if exp_name is None:
            return None

        metrics_path = self.base_dir / exp_name / 'logs' / 'metrics.json'
        if not metrics_path.exists():
            print(f"Warning: Metrics not found for {dataset} {stage}: {metrics_path}")
            return None

        with open(metrics_path, 'r') as f:
            return json.load(f)

    def plot_training_loss_heatmap(self):
        """
        Create heatmap matrix showing loss evolution across datasets and stages
        Figure 1: Training Loss Evolution Heatmap
        """
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle('Training Loss Evolution Across Datasets and Stages',
                     fontweight='bold', fontsize=14)

        positions = {
            'KITTI': (0, 0),
            'NCLT': (0, 1),
            'Boreas': (0, 2),
        }

        max_epochs = 0

        # Stage 2 (Denoising) - Top row
        for dataset in ['KITTI', 'NCLT', 'Boreas']:
            row, col = positions[dataset]
            ax = axes[row, col]

            metrics = self.load_metrics(dataset, 'stage2')
            if metrics is None:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_title(f'{dataset} - Stage2 (Denoising)')
                continue

            epochs = np.array(metrics['epoch'])
            train_loss = np.array(metrics['train_diffusion_loss'])
            lr = np.array(metrics['learning_rate'])

            max_epochs = max(max_epochs, len(epochs))

            # Create heatmap data: rows are metrics, cols are epochs
            # Sample every N epochs for visualization if too many
            step = max(1, len(epochs) // 100)
            sampled_epochs = epochs[::step]
            sampled_loss = train_loss[::step]
            sampled_lr = lr[::step]

            # Normalize for visualization
            loss_norm = (sampled_loss - sampled_loss.min()) / (sampled_loss.max() - sampled_loss.min() + 1e-8)
            lr_norm = (sampled_lr - sampled_lr.min()) / (sampled_lr.max() - sampled_lr.min() + 1e-8)

            # Stack metrics
            heatmap_data = np.vstack([loss_norm, lr_norm])

            # Plot heatmap
            im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn_r',
                          interpolation='bilinear')

            ax.set_title(f'{dataset} - Stage2 (Denoising)\nFinal Loss: {train_loss[-1]:.4f}')
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Loss', 'LR'])
            ax.set_xlabel('Training Progress (%)')

            # Set x-axis as percentage
            n_ticks = 5
            tick_positions = np.linspace(0, len(sampled_epochs)-1, n_ticks)
            tick_labels = [f'{int(100*i/(n_ticks-1))}' for i in range(n_ticks)]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Normalized Value')

        # Stage 3 (Descriptor) - Bottom row
        stage3_data = {}
        for dataset in ['KITTI', 'Boreas']:
            metrics = self.load_metrics(dataset, 'stage3')
            if metrics is not None:
                stage3_data[dataset] = metrics

        # Plot stage3 for available datasets
        col_idx = 0
        for dataset in ['KITTI', 'Boreas', 'NCLT']:
            ax = axes[1, col_idx]
            col_idx += 1

            if dataset not in stage3_data:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                       fontsize=12, style='italic')
                ax.set_title(f'{dataset} - Stage3 (Descriptor)')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            metrics = stage3_data[dataset]
            epochs = np.array(metrics['epoch'])
            train_loss = np.array(metrics['train_loss'])
            lr = np.array(metrics['learning_rate'])

            # Sample for visualization
            step = max(1, len(epochs) // 100)
            sampled_epochs = epochs[::step]
            sampled_loss = train_loss[::step]
            sampled_lr = lr[::step]

            # Normalize
            loss_norm = (sampled_loss - sampled_loss.min()) / (sampled_loss.max() - sampled_loss.min() + 1e-8)
            lr_norm = (sampled_lr - sampled_lr.min()) / (sampled_lr.max() - sampled_lr.min() + 1e-8)

            heatmap_data = np.vstack([loss_norm, lr_norm])

            im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn_r',
                          interpolation='bilinear')

            ax.set_title(f'{dataset} - Stage3 (Descriptor)\nFinal Loss: {train_loss[-1]:.4f}')
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Loss', 'LR'])
            ax.set_xlabel('Training Progress (%)')

            n_ticks = 5
            tick_positions = np.linspace(0, len(sampled_epochs)-1, n_ticks)
            tick_labels = [f'{int(100*i/(n_ticks-1))}' for i in range(n_ticks)]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Normalized Value')

        plt.tight_layout()

        # Save
        output_path = self.output_dir / 'training_loss_heatmap'
        plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}.pdf/.png")
        plt.close()

    def plot_denoising_quality_evolution(self):
        """
        Visualize denoising quality improvement across epochs
        Figure 2: Denoising Quality Evolution with Original Images
        Shows complete denoising process: Noisy Input -> Denoised -> Clean Target
        """
        # Key epochs to visualize (select representative epochs across training)
        key_epochs = [4, 19, 34, 49, 64, 79, 94]  # Early, mid, late training

        datasets_to_viz = ['KITTI', 'NCLT', 'Boreas']

        fig, axes = plt.subplots(len(datasets_to_viz), len(key_epochs),
                                figsize=(22, 9))
        fig.suptitle('Denoising Quality Evolution: Progressive Improvement Across Training\n'
                     '(Each image shows: Noisy Input | Noisy Latent | Denoised Latent | Clean Target)',
                     fontweight='bold', fontsize=13, y=0.995)

        for dataset_idx, dataset in enumerate(datasets_to_viz):
            exp_name = self.datasets[dataset]['stage2']
            if exp_name is None:
                # Fill empty row
                for epoch_idx in range(len(key_epochs)):
                    ax = axes[dataset_idx, epoch_idx]
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                           fontsize=10, style='italic', color='gray')
                    ax.axis('off')
                continue

            denoising_dir = self.base_dir / exp_name / 'denoising_vis'

            for epoch_idx, epoch in enumerate(key_epochs):
                ax = axes[dataset_idx, epoch_idx]

                # Find denoising visualization for this epoch
                denoising_file = denoising_dir / f'denoising_epoch_{epoch:03d}.png'

                if not denoising_file.exists():
                    ax.text(0.5, 0.5, f'Epoch {epoch}\nNot Available',
                           ha='center', va='center', fontsize=8, color='gray')
                    ax.axis('off')
                    continue

                # Load and display the complete denoising visualization image
                img = Image.open(denoising_file)
                ax.imshow(img)
                ax.axis('off')

                # Add epoch label on top row
                if dataset_idx == 0:
                    ax.set_title(f'Epoch {epoch}', fontsize=11, fontweight='bold')

            # Add dataset label on left side
            axes[dataset_idx, 0].text(-0.02, 0.5, dataset,
                                     transform=axes[dataset_idx, 0].transAxes,
                                     fontsize=13, fontweight='bold',
                                     rotation=90, va='center', ha='right')

        plt.tight_layout(rect=[0.01, 0, 1, 0.98])

        output_path = self.output_dir / 'denoising_quality_evolution'
        plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}.pdf/.png")
        plt.close()

    def plot_denoising_progression_single_sample(self):
        """
        Create a focused visualization showing ONE sample's progression
        across all training epochs - demonstrates learning dynamics
        Figure 2B: Single Sample Denoising Progression
        """
        # Select one dataset for detailed analysis
        dataset = 'KITTI'  # or 'Boreas' or 'NCLT'
        exp_name = self.datasets[dataset]['stage2']

        if exp_name is None:
            print(f"Warning: No stage2 training for {dataset}")
            return

        denoising_dir = self.base_dir / exp_name / 'denoising_vis'

        # Get all available epochs
        available_files = sorted(denoising_dir.glob('denoising_epoch_*.png'))
        if not available_files:
            print(f"Warning: No denoising visualizations found in {denoising_dir}")
            return

        # Extract epochs
        all_epochs = []
        for f in available_files:
            epoch_num = int(f.stem.split('_')[-1])
            all_epochs.append(epoch_num)

        # Select evenly spaced epochs across training
        num_to_show = 12  # Show 12 key points
        indices = np.linspace(0, len(all_epochs)-1, num_to_show, dtype=int)
        selected_epochs = [all_epochs[i] for i in indices]

        # Create visualization
        rows = 3
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
        fig.suptitle(f'{dataset} Dataset: Denoising Quality Progression Throughout Training\n'
                     f'Visualizing the same sample at different training stages',
                     fontweight='bold', fontsize=14)

        for idx, epoch in enumerate(selected_epochs):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            denoising_file = denoising_dir / f'denoising_epoch_{epoch:03d}.png'

            if denoising_file.exists():
                img = Image.open(denoising_file)
                ax.imshow(img)
                ax.set_title(f'Epoch {epoch}', fontsize=11, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'Epoch {epoch}\nNot Available',
                       ha='center', va='center')

            ax.axis('off')

        plt.tight_layout()

        output_path = self.output_dir / 'denoising_progression_detailed'
        plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}.pdf/.png")
        plt.close()

    def plot_loss_curves_comparison(self):
        """
        Plot training loss curves comparing all datasets
        Figure 3: Training Loss Comparison
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Stage 2 comparison
        ax = axes[0]
        for dataset in ['KITTI', 'NCLT', 'Boreas']:
            metrics = self.load_metrics(dataset, 'stage2')
            if metrics is None:
                continue

            epochs = np.array(metrics['epoch'])
            train_loss = np.array(metrics['train_diffusion_loss'])

            color = COLORS[dataset.lower()]
            ax.plot(epochs, train_loss, label=dataset, color=color, linewidth=2, alpha=0.8)

        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Diffusion Loss', fontweight='bold')
        ax.set_title('Stage 2: Denoising Training Loss', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Stage 3 comparison
        ax = axes[1]
        for dataset in ['KITTI', 'Boreas']:
            metrics = self.load_metrics(dataset, 'stage3')
            if metrics is None:
                continue

            epochs = np.array(metrics['epoch'])
            train_loss = np.array(metrics['train_loss'])

            color = COLORS[dataset.lower()]
            ax.plot(epochs, train_loss, label=dataset, color=color, linewidth=2, alpha=0.8)

        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Total Loss', fontweight='bold')
        ax.set_title('Stage 3: Descriptor Training Loss', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        output_path = self.output_dir / 'loss_curves_comparison'
        plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}.pdf/.png")
        plt.close()

    def plot_training_dynamics_heatmap(self):
        """
        Comprehensive training dynamics heatmap showing multiple metrics
        Figure 4: Multi-Metric Training Dynamics
        """
        datasets_list = ['KITTI', 'NCLT', 'Boreas']

        fig, axes = plt.subplots(len(datasets_list), 2, figsize=(14, 10))
        fig.suptitle('Training Dynamics: Multi-Metric Heatmaps',
                     fontweight='bold', fontsize=14)

        for dataset_idx, dataset in enumerate(datasets_list):
            # Stage 2
            ax_s2 = axes[dataset_idx, 0]
            metrics = self.load_metrics(dataset, 'stage2')

            if metrics is not None:
                epochs = np.array(metrics['epoch'])

                # Collect metrics
                metric_names = []
                metric_values = []

                if 'train_diffusion_loss' in metrics:
                    metric_names.append('Diffusion Loss')
                    metric_values.append(metrics['train_diffusion_loss'])

                if 'learning_rate' in metrics:
                    metric_names.append('Learning Rate')
                    metric_values.append(metrics['learning_rate'])

                # Sample for visualization
                step = max(1, len(epochs) // 100)
                sampled_data = []
                for values in metric_values:
                    sampled = np.array(values)[::step]
                    # Normalize
                    normalized = (sampled - np.min(sampled)) / (np.max(sampled) - np.min(sampled) + 1e-8)
                    sampled_data.append(normalized)

                if sampled_data:
                    heatmap_data = np.array(sampled_data)

                    im = ax_s2.imshow(heatmap_data, aspect='auto', cmap='viridis',
                                     interpolation='bilinear')
                    ax_s2.set_yticks(range(len(metric_names)))
                    ax_s2.set_yticklabels(metric_names, fontsize=9)

                    n_ticks = 5
                    tick_positions = np.linspace(0, heatmap_data.shape[1]-1, n_ticks)
                    tick_labels = [f'{int(100*i/(n_ticks-1))}%' for i in range(n_ticks)]
                    ax_s2.set_xticks(tick_positions)
                    ax_s2.set_xticklabels(tick_labels)

                    ax_s2.set_title(f'{dataset} - Stage2', fontweight='bold')
                    ax_s2.set_xlabel('Training Progress')

                    plt.colorbar(im, ax=ax_s2, fraction=0.046, pad=0.04)
            else:
                ax_s2.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax_s2.set_title(f'{dataset} - Stage2')

            # Stage 3
            ax_s3 = axes[dataset_idx, 1]
            metrics = self.load_metrics(dataset, 'stage3')

            if metrics is not None:
                epochs = np.array(metrics['epoch'])

                metric_names = []
                metric_values = []

                if 'train_loss' in metrics:
                    metric_names.append('Total Loss')
                    metric_values.append(metrics['train_loss'])

                if 'learning_rate' in metrics:
                    metric_names.append('Learning Rate')
                    metric_values.append(metrics['learning_rate'])

                # Sample and normalize
                step = max(1, len(epochs) // 100)
                sampled_data = []
                for values in metric_values:
                    sampled = np.array(values)[::step]
                    normalized = (sampled - np.min(sampled)) / (np.max(sampled) - np.min(sampled) + 1e-8)
                    sampled_data.append(normalized)

                if sampled_data:
                    heatmap_data = np.array(sampled_data)

                    im = ax_s3.imshow(heatmap_data, aspect='auto', cmap='viridis',
                                     interpolation='bilinear')
                    ax_s3.set_yticks(range(len(metric_names)))
                    ax_s3.set_yticklabels(metric_names, fontsize=9)

                    n_ticks = 5
                    tick_positions = np.linspace(0, heatmap_data.shape[1]-1, n_ticks)
                    tick_labels = [f'{int(100*i/(n_ticks-1))}%' for i in range(n_ticks)]
                    ax_s3.set_xticks(tick_positions)
                    ax_s3.set_xticklabels(tick_labels)

                    ax_s3.set_title(f'{dataset} - Stage3', fontweight='bold')
                    ax_s3.set_xlabel('Training Progress')

                    plt.colorbar(im, ax=ax_s3, fraction=0.046, pad=0.04)
            else:
                ax_s3.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                          style='italic')
                ax_s3.set_title(f'{dataset} - Stage3')

        plt.tight_layout()

        output_path = self.output_dir / 'training_dynamics_heatmap'
        plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}.pdf/.png")
        plt.close()

    def plot_learning_rate_schedules(self):
        """
        Visualize learning rate schedules across datasets
        Supplementary Figure: Learning Rate Scheduling
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Stage 2
        ax = axes[0]
        for dataset in ['KITTI', 'NCLT', 'Boreas']:
            metrics = self.load_metrics(dataset, 'stage2')
            if metrics is None:
                continue

            epochs = np.array(metrics['epoch'])
            lr = np.array(metrics['learning_rate'])

            color = COLORS[dataset.lower()]
            ax.plot(epochs, lr, label=dataset, color=color, linewidth=2, alpha=0.8)

        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Learning Rate', fontweight='bold')
        ax.set_title('Stage 2: Learning Rate Schedule', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_yscale('log')

        # Stage 3
        ax = axes[1]
        for dataset in ['KITTI', 'Boreas']:
            metrics = self.load_metrics(dataset, 'stage3')
            if metrics is None:
                continue

            epochs = np.array(metrics['epoch'])
            lr = np.array(metrics['learning_rate'])

            color = COLORS[dataset.lower()]
            ax.plot(epochs, lr, label=dataset, color=color, linewidth=2, alpha=0.8)

        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Learning Rate', fontweight='bold')
        ax.set_title('Stage 3: Learning Rate Schedule', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_yscale('log')

        plt.tight_layout()

        output_path = self.output_dir / 'learning_rate_schedules'
        plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}.pdf/.png")
        plt.close()

    def generate_all_visualizations(self):
        """Generate all paper visualizations"""
        print("=" * 70)
        print("Generating Paper Training Visualizations with Original Images")
        print("=" * 70)

        print("\n[1/6] Creating training loss heatmap...")
        self.plot_training_loss_heatmap()

        print("\n[2/6] Creating loss curves comparison...")
        self.plot_loss_curves_comparison()

        print("\n[3/6] Creating training dynamics heatmap...")
        self.plot_training_dynamics_heatmap()

        print("\n[4/6] Creating learning rate schedules...")
        self.plot_learning_rate_schedules()

        print("\n[5/6] Creating denoising quality evolution (multi-dataset)...")
        self.plot_denoising_quality_evolution()

        print("\n[6/6] Creating detailed denoising progression (single sample)...")
        self.plot_denoising_progression_single_sample()

        print("\n" + "=" * 70)
        print(f"All visualizations saved to: {self.output_dir}")
        print("=" * 70)
        print("\nGenerated visualizations:")
        print("  1. training_loss_heatmap.pdf/png - Loss evolution heatmaps")
        print("  2. loss_curves_comparison.pdf/png - Training loss curves")
        print("  3. training_dynamics_heatmap.pdf/png - Multi-metric dynamics")
        print("  4. learning_rate_schedules.pdf/png - LR scheduling")
        print("  5. denoising_quality_evolution.pdf/png - Cross-dataset denoising")
        print("  6. denoising_progression_detailed.pdf/png - Single sample progression")
        print("\nAll figures include original images showing:")
        print("  - Noisy Input (adverse weather)")
        print("  - Noisy Latent representation")
        print("  - Denoised Latent representation")
        print("  - Clean Target (ground truth)")
        print("=" * 70)


if __name__ == '__main__':
    visualizer = TrainingVisualizer()
    visualizer.generate_all_visualizations()
