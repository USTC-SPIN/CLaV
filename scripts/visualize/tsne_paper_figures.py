#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Uniform Square T-SNE Plots for Paper
从已保存的嵌入数据生成统一尺寸的方形T-SNE图，用于论文

Created: 2025-11-14 06:54
Purpose: Extract subplot (a) from saved embeddings and create uniform square figures
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def plot_single_tsne_square(
    embeddings_dict,
    dataset_name,
    save_path,
    figsize=(10, 10),
    dpi=300
):
    """
    Create single square T-SNE scatter plot

    Args:
        embeddings_dict: dict with weather types as keys, embeddings as values
        dataset_name: name of dataset for title
        save_path: output path
        figsize: figure size (width, height) - use square for consistency
        dpi: resolution
    """
    # Color scheme - consistent across all datasets
    color_map = {
        'clear': '#2E7D32',      # Green
        'snow': '#1976D2',       # Blue
        'rain': '#D32F2F',       # Red
        'fog': '#F57C00'         # Orange
    }

    label_map = {
        'clear': 'Clear (Database)',
        'snow': 'Snow (Query)',
        'rain': 'Rain (Query)',
        'fog': 'Fog (Query)'
    }

    # Create square figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each weather condition
    for weather_type in ['clear', 'snow', 'rain', 'fog']:
        if weather_type in embeddings_dict:
            embedding = embeddings_dict[weather_type]
            ax.scatter(
                embedding[:, 0], embedding[:, 1],
                c=color_map[weather_type],
                label=label_map[weather_type],
                alpha=0.6,
                s=40,  # Slightly larger points for better visibility
                edgecolors='white',
                linewidth=0.5
            )

    # Remove all axes elements
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Make it perfectly square
    ax.set_aspect('equal', adjustable='box')

    # Remove padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save with exact size (no bbox_inches='tight' to ensure consistent dimensions)
    print(f"Saving {dataset_name} figure to: {save_path}")
    plt.savefig(save_path, dpi=dpi, facecolor='white', pad_inches=0)
    plt.close()

    print(f"✓ {dataset_name} figure saved successfully!")


def main():
    """Main function to generate paper figures from saved embeddings"""

    # Paths
    data_dir = Path('/data/users/cxw/pro/clav/Figure')
    output_dir = Path('/data/users/cxw/pro/clav/Figure/paper_tsne')
    output_dir.mkdir(exist_ok=True)

    # Dataset configurations
    datasets = {
        'kitti': {
            'name': 'KITTI',
            'data_file': data_dir / 'tsne_kitti_embeddings.npz',
            'output': output_dir / 'tsne_kitti_paper.png'
        },
        'nclt': {
            'name': 'NCLT',
            'data_file': data_dir / 'tsne_nclt_embeddings.npz',
            'output': output_dir / 'tsne_nclt_paper.png'
        },
        'boreas': {
            'name': 'Boreas',
            'data_file': data_dir / 'tsne_boreas_embeddings.npz',
            'output': output_dir / 'tsne_boreas_paper.png'
        }
    }

    print("="*80)
    print("Generating Uniform Square T-SNE Figures for Paper")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Figure size: 10×10 inches (square)")
    print(f"Resolution: 300 dpi")
    print()

    # Process each dataset
    for dataset_key, config in datasets.items():
        print("\n" + "-"*80)
        print(f"Processing {config['name']} Dataset")
        print("-"*80)

        # Check if data file exists
        if not config['data_file'].exists():
            print(f"⚠ Warning: {config['data_file']} not found, skipping...")
            continue

        # Load embeddings
        print(f"Loading embeddings from: {config['data_file']}")
        data = np.load(config['data_file'], allow_pickle=True)

        # Extract embeddings
        embeddings_dict = {}

        # Database (clear weather)
        if 'clear_embedding' in data:
            embeddings_dict['clear'] = data['clear_embedding']
            print(f"  Clear (Database): {len(embeddings_dict['clear'])} samples")

        # Queries
        for weather in ['snow', 'rain', 'fog']:
            key = f'{weather}_embedding'
            if key in data:
                embeddings_dict[weather] = data[key]
                print(f"  {weather.capitalize()} (Query): {len(embeddings_dict[weather])} samples")

        # Generate figure
        plot_single_tsne_square(
            embeddings_dict=embeddings_dict,
            dataset_name=config['name'],
            save_path=str(config['output']),
            figsize=(10, 10),  # Perfect square
            dpi=300
        )

    print("\n" + "="*80)
    print("ALL FIGURES GENERATED!")
    print("="*80)
    print(f"\nOutput location: {output_dir}/")
    print("\nGenerated files:")
    for dataset_key, config in datasets.items():
        if config['output'].exists():
            size_mb = config['output'].stat().st_size / (1024 * 1024)
            print(f"  ✓ {config['output'].name} ({size_mb:.2f} MB)")

    print("\n✓ All figures are square (10×10 inches) and ready for paper submission!")


if __name__ == '__main__':
    main()
