"""
Direct BEV Image-based Denoising Visualization for Paper
Shows intuitive before/after comparison using actual BEV images
Generated: 2025-11-13
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import re

plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


class BEVDenoisingVisualizer:
    """Create intuitive BEV-based denoising visualizations"""

    def __init__(self, base_results_dir: str = '/data/users/cxw/pro/clav/results'):
        self.base_dir = Path(base_results_dir)
        self.output_dir = Path('/data/users/cxw/pro/clav/Figure/paper_visualizations')
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.datasets = {
            'KITTI': 'kitti_stage2_flow_matching_20251108_1724',
            'NCLT': 'nclt_stage2_training_20251107_0025',
            'Boreas': 'boreas_stage2_flow_matching_optimized_20251113_1242',
        }

    def extract_bev_images_from_visualization(self, vis_image):
        """
        从现有的去噪可视化中提取单独的BEV图像

        布局: [Noisy BEV | Noisy Latent | Denoised Latent | Clean BEV]
        每行3个样本,每个样本4列
        """
        img_array = np.array(vis_image)
        height, width = img_array.shape[:2]

        # 估算每个子图的大小(3行4列布局)
        # 实际图像可能有边距,需要智能裁剪
        num_rows = 3
        num_cols = 4

        row_height = height // num_rows
        col_width = width // num_cols

        # 提取每个样本的noisy和clean BEV(第0列和第3列)
        samples = []
        for row in range(num_rows):
            y_start = row * row_height
            y_end = (row + 1) * row_height

            # Noisy BEV (第0列)
            noisy_x_start = 0
            noisy_x_end = col_width
            noisy_bev = img_array[y_start:y_end, noisy_x_start:noisy_x_end]

            # Clean BEV (第3列)
            clean_x_start = 3 * col_width
            clean_x_end = 4 * col_width
            clean_bev = img_array[y_start:y_end, clean_x_start:clean_x_end]

            samples.append({
                'noisy': noisy_bev,
                'clean': clean_bev
            })

        return samples

    def plot_bev_denoising_evolution(self):
        """
        Create a clean BEV-only visualization showing:
        Noisy Input | Clean Target
        Across different training epochs
        """
        key_epochs = [4, 19, 34, 49, 64, 79, 94]
        datasets = ['KITTI', 'NCLT', 'Boreas']

        # Create figure: datasets × epochs, each showing noisy + clean side by side
        fig = plt.figure(figsize=(24, 10))

        # Use GridSpec for flexible layout
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(len(datasets), len(key_epochs),
                              hspace=0.3, wspace=0.15,
                              top=0.94, bottom=0.02, left=0.03, right=0.97)

        fig.suptitle('BEV Denoising Quality Evolution: Input (Noisy) vs Target (Clean)\n'
                     'Showing first sample from each epoch',
                     fontsize=16, fontweight='bold')

        for dataset_idx, dataset in enumerate(datasets):
            exp_name = self.datasets[dataset]
            denoising_dir = self.base_dir / exp_name / 'denoising_vis'

            for epoch_idx, epoch in enumerate(key_epochs):
                # Create subplot for this cell
                ax = fig.add_subplot(gs[dataset_idx, epoch_idx])

                denoising_file = denoising_dir / f'denoising_epoch_{epoch:03d}.png'

                if not denoising_file.exists():
                    ax.text(0.5, 0.5, f'Epoch {epoch}\nN/A',
                           ha='center', va='center', fontsize=9)
                    ax.axis('off')
                    continue

                # Load full visualization
                full_vis = Image.open(denoising_file)

                # Extract BEV images
                samples = self.extract_bev_images_from_visualization(full_vis)

                # Use first sample only
                if samples:
                    sample = samples[0]
                    noisy_bev = sample['noisy']
                    clean_bev = sample['clean']

                    # Concatenate side by side
                    combined = np.concatenate([noisy_bev, clean_bev], axis=1)

                    ax.imshow(combined)
                    ax.axis('off')

                    # Add epoch label on top row
                    if dataset_idx == 0:
                        ax.set_title(f'Epoch {epoch}', fontsize=12, fontweight='bold', pad=8)

                    # Add "Noisy | Clean" label on first column
                    if epoch_idx == 0:
                        # Add dataset label
                        ax.text(-0.02, 0.5, dataset,
                               transform=ax.transAxes,
                               fontsize=14, fontweight='bold',
                               rotation=90, va='center', ha='right')
                else:
                    ax.text(0.5, 0.5, 'Extraction\nFailed',
                           ha='center', va='center', fontsize=8)
                    ax.axis('off')

        # Add legend at bottom
        legend_text = "Each cell shows: Left=Noisy Input (adverse weather) | Right=Clean Target (sunny)"
        fig.text(0.5, 0.01, legend_text, ha='center', fontsize=11, style='italic')

        output_path = self.output_dir / 'bev_denoising_evolution_clean'
        plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}.pdf/.png")
        plt.close()

    def plot_bev_three_column_comparison(self):
        """
        Alternative layout: 3 columns showing progression
        Column 1: Early training (epoch 4-19)
        Column 2: Mid training (epoch 34-49)
        Column 3: Late training (epoch 64-94)

        Each showing Noisy vs Clean for all 3 datasets
        """
        epoch_groups = {
            'Early\n(Epoch 4-19)': [4, 19],
            'Mid\n(Epoch 34-49)': [34, 49],
            'Late\n(Epoch 64-94)': [64, 94]
        }

        datasets = ['KITTI', 'NCLT', 'Boreas']

        fig, axes = plt.subplots(len(datasets), len(epoch_groups),
                                figsize=(16, 12))

        fig.suptitle('BEV Denoising Progression: Early → Mid → Late Training\n'
                     'Each panel shows Noisy Input (left) | Clean Target (right)',
                     fontsize=15, fontweight='bold')

        for group_idx, (group_name, epochs) in enumerate(epoch_groups.items()):
            for dataset_idx, dataset in enumerate(datasets):
                ax = axes[dataset_idx, group_idx]

                exp_name = self.datasets[dataset]
                denoising_dir = self.base_dir / exp_name / 'denoising_vis'

                # Use last epoch in this group
                epoch = epochs[-1]
                denoising_file = denoising_dir / f'denoising_epoch_{epoch:03d}.png'

                if denoising_file.exists():
                    full_vis = Image.open(denoising_file)
                    samples = self.extract_bev_images_from_visualization(full_vis)

                    if samples:
                        sample = samples[0]  # First sample
                        noisy_bev = sample['noisy']
                        clean_bev = sample['clean']
                        combined = np.concatenate([noisy_bev, clean_bev], axis=1)

                        ax.imshow(combined)
                        ax.axis('off')

                        # Add title on top row
                        if dataset_idx == 0:
                            ax.set_title(group_name, fontsize=13, fontweight='bold')

                        # Add dataset label on left
                        if group_idx == 0:
                            ax.set_ylabel(dataset, fontsize=13, fontweight='bold', rotation=90)
                    else:
                        ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                        ax.axis('off')
                else:
                    ax.text(0.5, 0.5, f'Epoch {epoch}\nN/A',
                           ha='center', va='center', fontsize=9)
                    ax.axis('off')

        plt.tight_layout()

        output_path = self.output_dir / 'bev_denoising_three_stage'
        plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}.pdf/.png")
        plt.close()

    def plot_single_dataset_detailed(self, dataset='KITTI'):
        """
        Detailed progression for one dataset
        Shows multiple samples across many epochs
        """
        exp_name = self.datasets[dataset]
        denoising_dir = self.base_dir / exp_name / 'denoising_vis'

        # Get all available epochs
        available_files = sorted(denoising_dir.glob('denoising_epoch_*.png'))
        all_epochs = [int(f.stem.split('_')[-1]) for f in available_files]

        # Select 12 representative epochs
        num_to_show = 12
        indices = np.linspace(0, len(all_epochs)-1, num_to_show, dtype=int)
        selected_epochs = [all_epochs[i] for i in indices]

        # Create 3×4 grid
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        fig.suptitle(f'{dataset} Dataset: Detailed BEV Denoising Progression\n'
                     f'Noisy Input (left) | Clean Target (right) at different training stages',
                     fontsize=15, fontweight='bold')

        for idx, epoch in enumerate(selected_epochs):
            row = idx // 4
            col = idx % 4
            ax = axes[row, col]

            denoising_file = denoising_dir / f'denoising_epoch_{epoch:03d}.png'

            if denoising_file.exists():
                full_vis = Image.open(denoising_file)
                samples = self.extract_bev_images_from_visualization(full_vis)

                if samples:
                    sample = samples[0]
                    noisy_bev = sample['noisy']
                    clean_bev = sample['clean']
                    combined = np.concatenate([noisy_bev, clean_bev], axis=1)

                    ax.imshow(combined)
                    ax.set_title(f'Epoch {epoch}', fontsize=11, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, f'Epoch {epoch}', ha='center', va='center')
            else:
                ax.text(0.5, 0.5, f'Epoch {epoch}\nN/A', ha='center', va='center')

            ax.axis('off')

        plt.tight_layout()

        output_path = self.output_dir / f'bev_denoising_detailed_{dataset.lower()}'
        plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}.pdf/.png")
        plt.close()

    def generate_all_bev_visualizations(self):
        """Generate all BEV-based visualizations"""
        print("=" * 70)
        print("Generating BEV Image-Based Denoising Visualizations")
        print("=" * 70)

        print("\n[1/4] Creating BEV denoising evolution (all epochs)...")
        self.plot_bev_denoising_evolution()

        print("\n[2/4] Creating 3-stage comparison (early/mid/late)...")
        self.plot_bev_three_column_comparison()

        print("\n[3/4] Creating detailed KITTI progression...")
        self.plot_single_dataset_detailed('KITTI')

        print("\n[4/4] Creating detailed Boreas progression...")
        self.plot_single_dataset_detailed('Boreas')

        print("\n" + "=" * 70)
        print("All BEV-based visualizations generated!")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - bev_denoising_evolution_clean.pdf/png")
        print("  - bev_denoising_three_stage.pdf/png")
        print("  - bev_denoising_detailed_kitti.pdf/png")
        print("  - bev_denoising_detailed_boreas.pdf/png")
        print("\nThese visualizations show ONLY the actual BEV images:")
        print("  Left side: Noisy input (adverse weather)")
        print("  Right side: Clean target (sunny/clear)")
        print("  No confusing latent heatmaps - direct visual comparison!")
        print("=" * 70)


if __name__ == '__main__':
    visualizer = BEVDenoisingVisualizer()
    visualizer.generate_all_bev_visualizations()
