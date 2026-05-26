#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T-SNE Visualization for All Datasets (KITTI, NCLT, Boreas)
用于论文展示的多数据集地理定位T-SNE可视化

Created: 2025-11-14 04:47
Purpose: Generate T-SNE visualizations for all three datasets
"""

import torch
import numpy as np
import sys
import os
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde

# Add parent directory to path
PARENT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PARENT_DIR))

from src.models.clav import CLaV
from evaluation.evaluate import extract_descriptors


# Dataset configurations
DATASET_CONFIGS = {
    'kitti': {
        'name': 'KITTI',
        'checkpoint': '/data/users/cxw/pro/clav/results/boreas_descriptor_training_optimized_20251113_2345/best.pt',
        'database': '/data/users/cxw/pro/clav/data/kitti_bev_snow_evaluation_database.pickle',
        'queries': {
            'snow': '/data/users/cxw/pro/clav/data/kitti_bev_snow_evaluation_query.pickle',
            'rain': '/data/users/cxw/pro/clav/data/kitti_bev_rain_evaluation_query.pickle',
            'fog': '/data/users/cxw/pro/clav/data/kitti_bev_fog_evaluation_query.pickle',
        }
    },
    'nclt': {
        'name': 'NCLT',
        'checkpoint': '/data/users/cxw/pro/clav/results/boreas_descriptor_training_optimized_20251113_2345/best.pt',
        'database': '/data/users/cxw/pro/clav/data/nclt_bev_snow_evaluation_database.pickle',
        'queries': {
            'snow': '/data/users/cxw/pro/clav/data/nclt_bev_snow_evaluation_query.pickle',
            'rain': '/data/users/cxw/pro/clav/data/nclt_bev_rain_evaluation_query.pickle',
            'fog': '/data/users/cxw/pro/clav/data/nclt_bev_fog_evaluation_query.pickle',
        }
    },
    'boreas': {
        'name': 'Boreas',
        'checkpoint': '/data/users/cxw/pro/clav/results/boreas_descriptor_training_optimized_20251113_2345/best.pt',
        'database': '/data/users/cxw/pro/clav/data/boreas_bev_snow_evaluation_database_spatial.pickle',
        'queries': {
            'snow': '/data/users/cxw/pro/clav/data/boreas_bev_snow_evaluation_query_spatial.pickle',
            'rain': '/data/users/cxw/pro/clav/data/boreas_bev_rain_evaluation_query_spatial.pickle',
        }
    }
}


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Clear GPU cache first
    if 'cuda' in device:
        torch.cuda.empty_cache()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config
    config = checkpoint.get('config', {})

    # Create model
    model = CLaV(config)

    # Load state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
    # Remove 'module.' prefix if exists
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    return model, config


def load_data_pickle(pickle_path):
    """Load evaluation pickle file"""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def parse_pickle_data(data):
    """
    Parse pickle data to extract image paths and coordinates

    Supports multiple formats:
    - Boreas: list with one dict, keys are indices
    - KITTI: dict with numeric keys
    - NCLT: list of dicts with numeric keys
    """
    image_paths = []
    coords = []

    # Format 1: List with one dict (Boreas)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        sample_dict = data[0]
        for idx in sorted(sample_dict.keys()):
            entry = sample_dict[idx]
            image_paths.append(entry['query'])
            coords.append([entry['northing'], entry['easting']])

    # Format 2: Direct dict with numeric keys (KITTI)
    elif isinstance(data, dict) and all(isinstance(k, int) for k in list(data.keys())[:5]):
        for idx in sorted(data.keys()):
            entry = data[idx]
            image_paths.append(entry['query'])
            coords.append([entry['northing'], entry['easting']])

    # Format 3: List of dicts (NCLT - each dict is a session)
    elif isinstance(data, list) and all(isinstance(d, dict) for d in data):
        for session_dict in data:
            for idx in sorted(session_dict.keys()):
                entry = session_dict[idx]
                image_paths.append(entry['query'])
                coords.append([entry['northing'], entry['easting']])

    else:
        raise ValueError(f"Unsupported data format: {type(data)}")

    coords = np.array(coords) if coords else None
    return image_paths, coords


def extract_features_from_pickle(model, pickle_path, device='cuda', batch_size=8, desc="Extracting"):
    """
    Extract features from pickle file

    Returns:
        features: (N, 8448) numpy array
        image_paths: list of image paths
        metadata: dict with additional info (coordinates, etc.)
    """
    print(f"\nLoading data from: {pickle_path}")
    data = load_data_pickle(pickle_path)

    # Parse data
    image_paths, coords = parse_pickle_data(data)
    print(f"Found {len(image_paths)} images")

    # Extract features
    features = extract_descriptors(
        model=model,
        image_paths=image_paths,
        device=device,
        batch_size=batch_size,
        show_progress=True,
        skip_denoising=False  # Use full pipeline with denoising
    )

    metadata = {
        'coords': coords,
        'pickle_path': pickle_path
    }

    print(f"Extracted features shape: {features.shape}")
    return features, image_paths, metadata


def compute_tsne(features, perplexity=30, max_iter=1000, random_state=42):
    """Compute T-SNE embedding"""
    print(f"\nComputing T-SNE with perplexity={perplexity}, max_iter={max_iter}")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
        verbose=1
    )
    embedding = tsne.fit_transform(features)
    print(f"T-SNE embedding shape: {embedding.shape}")
    return embedding


def compute_retrieval_pairs(query_features, database_features, k=1):
    """Compute top-k retrieval pairs"""
    print(f"\nComputing top-{k} retrieval pairs...")
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    nbrs.fit(database_features)
    distances, indices = nbrs.kneighbors(query_features)
    return indices, distances


def plot_tsne_visualization(
    embeddings_dict,
    labels_dict,
    dataset_name,
    retrieval_pairs=None,
    save_path='tsne_visualization.png',
    dpi=300
):
    """Create comprehensive T-SNE visualization with 4 subplots"""
    print(f"\nCreating T-SNE visualization for {dataset_name}...")

    # Color scheme
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

    # Create figure
    fig = plt.figure(figsize=(20, 16))

    # ------ Subplot 1: Simple scatter plot ------
    ax1 = plt.subplot(2, 2, 1)

    for weather_type in ['clear', 'snow', 'rain', 'fog']:
        if weather_type in embeddings_dict:
            embedding = embeddings_dict[weather_type]
            ax1.scatter(
                embedding[:, 0], embedding[:, 1],
                c=color_map[weather_type],
                label=label_map[weather_type],
                alpha=0.6,
                s=30,
                edgecolors='white',
                linewidth=0.5
            )

    ax1.set_title('(a) T-SNE Embedding of BEV Descriptors', fontsize=14, fontweight='bold')
    ax1.set_xlabel('T-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('T-SNE Dimension 2', fontsize=12)
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal')

    # ------ Subplot 2: With retrieval connections ------
    ax2 = plt.subplot(2, 2, 2)

    # Plot database
    if 'clear' in embeddings_dict:
        embedding = embeddings_dict['clear']
        ax2.scatter(
            embedding[:, 0], embedding[:, 1],
            c=color_map['clear'],
            label=label_map['clear'],
            alpha=0.4,
            s=20,
            edgecolors='white',
            linewidth=0.3
        )

    # Draw retrieval connections
    if retrieval_pairs is not None:
        query_types = ['snow', 'rain', 'fog']
        connection_colors = {'snow': '#64B5F6', 'rain': '#EF5350', 'fog': '#FFB74D'}

        for query_type in query_types:
            if query_type in retrieval_pairs and query_type in embeddings_dict:
                pairs = retrieval_pairs[query_type]
                query_emb = embeddings_dict[query_type]
                db_emb = embeddings_dict['clear']

                n_connections = min(50, len(pairs))
                for i in range(n_connections):
                    query_idx = i
                    db_idx = pairs[i][0]

                    ax2.plot(
                        [query_emb[query_idx, 0], db_emb[db_idx, 0]],
                        [query_emb[query_idx, 1], db_emb[db_idx, 1]],
                        c=connection_colors[query_type],
                        alpha=0.15,
                        linewidth=0.5,
                        zorder=1
                    )

    # Plot queries
    for weather_type in ['snow', 'rain', 'fog']:
        if weather_type in embeddings_dict:
            embedding = embeddings_dict[weather_type]
            ax2.scatter(
                embedding[:, 0], embedding[:, 1],
                c=color_map[weather_type],
                label=label_map[weather_type],
                alpha=0.8,
                s=40,
                edgecolors='white',
                linewidth=0.5,
                zorder=2
            )

    ax2.set_title('(b) Cross-Weather Retrieval Connections (Top-1)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('T-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel('T-SNE Dimension 2', fontsize=12)
    ax2.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_aspect('equal')

    # ------ Subplot 3: With density contours ------
    ax3 = plt.subplot(2, 2, 3)

    for weather_type in ['clear', 'snow', 'rain', 'fog']:
        if weather_type in embeddings_dict:
            embedding = embeddings_dict[weather_type]

            # Scatter
            ax3.scatter(
                embedding[:, 0], embedding[:, 1],
                c=color_map[weather_type],
                label=label_map[weather_type],
                alpha=0.5,
                s=25,
                edgecolors='white',
                linewidth=0.3
            )

            # KDE contours
            if len(embedding) > 10:
                try:
                    kde = gaussian_kde(embedding.T)
                    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
                    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
                    x_margin = (x_max - x_min) * 0.1
                    y_margin = (y_max - y_min) * 0.1

                    xx, yy = np.mgrid[
                        x_min-x_margin:x_max+x_margin:100j,
                        y_min-y_margin:y_max+y_margin:100j
                    ]
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    density = kde(positions).reshape(xx.shape)

                    ax3.contour(
                        xx, yy, density,
                        colors=color_map[weather_type],
                        alpha=0.4,
                        linewidths=1.5,
                        levels=5
                    )
                except:
                    pass

    ax3.set_title('(c) Cluster Boundaries with Density Estimation', fontsize=14, fontweight='bold')
    ax3.set_xlabel('T-SNE Dimension 1', fontsize=12)
    ax3.set_ylabel('T-SNE Dimension 2', fontsize=12)
    ax3.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_aspect('equal')

    # ------ Subplot 4: Distance matrix heatmap ------
    ax4 = plt.subplot(2, 2, 4)

    # Get available classes
    available_classes = []
    class_keys = []
    for key in ['clear', 'snow', 'rain', 'fog']:
        if key in embeddings_dict:
            available_classes.append(label_map[key].split()[0])
            class_keys.append(key)

    n_classes = len(available_classes)
    distance_matrix = np.zeros((n_classes, n_classes))

    # Compute distances
    for i, key_i in enumerate(class_keys):
        emb_i = embeddings_dict[key_i]
        for j, key_j in enumerate(class_keys):
            emb_j = embeddings_dict[key_j]

            n_samples = min(100, len(emb_i), len(emb_j))
            rand_idx1 = np.random.choice(len(emb_i), n_samples, replace=False)
            rand_idx2 = np.random.choice(len(emb_j), n_samples, replace=False)
            distances = np.linalg.norm(emb_i[rand_idx1] - emb_j[rand_idx2], axis=1)
            distance_matrix[i, j] = distances.mean()

    # Plot heatmap
    im = ax4.imshow(distance_matrix, cmap='RdYlGn_r', aspect='auto')
    ax4.set_xticks(np.arange(n_classes))
    ax4.set_yticks(np.arange(n_classes))
    ax4.set_xticklabels(available_classes)
    ax4.set_yticklabels(available_classes)
    plt.setp(ax4.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Mean Euclidean Distance', rotation=270, labelpad=20, fontsize=11)

    for i in range(n_classes):
        for j in range(n_classes):
            ax4.text(j, i, f'{distance_matrix[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=10)

    ax4.set_title('(d) Inter-class Distance Matrix', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Weather Condition', fontsize=12)
    ax4.set_ylabel('Weather Condition', fontsize=12)

    # Overall title
    fig.suptitle(f'T-SNE Visualization: {dataset_name} Dataset (8448-D → 2-D)',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    print(f"Saving figure to: {save_path}")
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved successfully!")
    plt.close()


def process_dataset(dataset_key, device='cuda', batch_size=16):
    """Process single dataset"""
    config = DATASET_CONFIGS[dataset_key]
    dataset_name = config['name']

    print("\n" + "="*80)
    print(f"Processing {dataset_name} Dataset")
    print("="*80)

    # Output paths
    output_dir = Path('/data/users/cxw/pro/clav/Figure')
    output_image = output_dir / f'tsne_{dataset_key}_geolocation_results.png'
    output_data = output_dir / f'tsne_{dataset_key}_embeddings.npz'

    # Load model
    model, model_config = load_checkpoint(config['checkpoint'], device=device)

    # Extract database features
    print("\n" + "-"*80)
    print("Extracting Database Features")
    print("-"*80)
    db_features, db_paths, db_meta = extract_features_from_pickle(
        model, config['database'], device=device, batch_size=batch_size
    )

    # Extract query features
    query_features_dict = {}
    query_paths_dict = {}
    query_meta_dict = {}

    for weather, query_path in config['queries'].items():
        if not Path(query_path).exists():
            print(f"Warning: {query_path} not found, skipping {weather}")
            continue

        print("\n" + "-"*80)
        print(f"Extracting {weather.upper()} Query Features")
        print("-"*80)

        features, paths, meta = extract_features_from_pickle(
            model, query_path, device=device, batch_size=batch_size
        )
        query_features_dict[weather] = features
        query_paths_dict[weather] = paths
        query_meta_dict[weather] = meta

    # Combine all features for T-SNE
    print("\n" + "="*80)
    print("T-SNE Dimensionality Reduction")
    print("="*80)

    all_features = [db_features]
    for weather in sorted(query_features_dict.keys()):
        all_features.append(query_features_dict[weather])

    all_features = np.vstack(all_features)
    print(f"Total samples: {len(all_features)}")
    print(f"Feature dimension: {all_features.shape[1]}")

    # Compute T-SNE
    all_embeddings = compute_tsne(all_features, perplexity=30, max_iter=1000)

    # Split embeddings
    embeddings_dict = {}
    offset = 0

    embeddings_dict['clear'] = all_embeddings[offset:offset+len(db_features)]
    offset += len(db_features)

    for weather in sorted(query_features_dict.keys()):
        n_samples = len(query_features_dict[weather])
        embeddings_dict[weather] = all_embeddings[offset:offset+n_samples]
        offset += n_samples

    # Compute retrieval pairs
    print("\n" + "="*80)
    print("Computing Retrieval Pairs")
    print("="*80)

    retrieval_pairs = {}
    for weather, features in query_features_dict.items():
        pairs, distances = compute_retrieval_pairs(features, db_features, k=1)
        retrieval_pairs[weather] = pairs
        print(f"\n{weather.capitalize()} queries: {len(pairs)}")
        print(f"  Mean retrieval distance: {distances.mean():.4f}")
        print(f"  Median retrieval distance: {np.median(distances):.4f}")

    # Save data
    print("\n" + "="*80)
    print("Saving Data")
    print("="*80)

    save_dict = {
        'clear_embedding': embeddings_dict['clear'],
        'clear_features': db_features,
        'clear_paths': db_paths,
    }

    for weather in sorted(query_features_dict.keys()):
        save_dict[f'{weather}_embedding'] = embeddings_dict[weather]
        save_dict[f'{weather}_features'] = query_features_dict[weather]
        save_dict[f'{weather}_paths'] = query_paths_dict[weather]
        save_dict[f'{weather}_pairs'] = retrieval_pairs[weather]

    np.savez(output_data, **save_dict)
    print(f"Saved embeddings to: {output_data}")

    # Create visualization
    print("\n" + "="*80)
    print("Creating Visualization")
    print("="*80)

    labels_dict = {
        'clear': f'Clear (DB, n={len(db_features)})'
    }
    for weather in sorted(query_features_dict.keys()):
        labels_dict[weather] = f'{weather.capitalize()} (Query, n={len(query_features_dict[weather])})'

    plot_tsne_visualization(
        embeddings_dict=embeddings_dict,
        labels_dict=labels_dict,
        dataset_name=dataset_name,
        retrieval_pairs=retrieval_pairs,
        save_path=str(output_image),
        dpi=300
    )

    print("\n" + "="*80)
    print(f"{dataset_name} COMPLETED!")
    print("="*80)
    print(f"Visualization: {output_image}")
    print(f"Data: {output_data}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate T-SNE visualizations for datasets')
    parser.add_argument('--dataset', type=str, choices=['kitti', 'nclt', 'boreas', 'all'],
                        default='all', help='Dataset to process')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')

    args = parser.parse_args()

    if args.dataset == 'all':
        datasets = ['kitti', 'nclt', 'boreas']
    else:
        datasets = [args.dataset]

    for dataset_key in datasets:
        process_dataset(dataset_key, device=args.device, batch_size=args.batch_size)
        # Clear GPU memory between datasets
        if 'cuda' in args.device:
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
