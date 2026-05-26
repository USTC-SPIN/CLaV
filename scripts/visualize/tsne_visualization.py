#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T-SNE Visualization for Geo-localization Results
用于论文展示的地理定位T-SNE可视化

Created: 2025-11-14 03:45
Purpose: Generate T-SNE visualization for BEV geo-localization paper
"""

import torch
import numpy as np
import sys
import os
import pickle
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde
import seaborn as sns

# Add parent directory to path
PARENT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PARENT_DIR))

from src.models.clav import CLaV
from evaluation.evaluate import extract_descriptors


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

    # Parse the data structure
    # Format: list with one dict, keys are indices, values are dicts with 'query', 'position', etc.
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        sample_dict = data[0]
        image_paths = []
        coords = []

        # Extract from indexed dictionary
        for idx in sorted(sample_dict.keys()):
            entry = sample_dict[idx]
            image_paths.append(entry['query'])
            coords.append([entry['northing'], entry['easting']])

        coords = np.array(coords) if coords else None
    elif isinstance(data, dict) and 'image_paths' in data:
        image_paths = data['image_paths']
        coords = data.get('coords', None)
    elif isinstance(data, list):
        image_paths = data
        coords = None
    else:
        raise ValueError(f"Unexpected data format in {pickle_path}")

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
    """
    Compute T-SNE embedding

    Args:
        features: (N, D) numpy array
        perplexity: T-SNE perplexity parameter
        max_iter: number of iterations
        random_state: random seed

    Returns:
        embedding: (N, 2) numpy array
    """
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
    """
    Compute top-k retrieval pairs using nearest neighbor search

    Args:
        query_features: (N_q, D) query features
        database_features: (N_db, D) database features
        k: number of neighbors

    Returns:
        indices: (N_q, k) indices of top-k matches in database
        distances: (N_q, k) distances to top-k matches
    """
    print(f"\nComputing top-{k} retrieval pairs...")
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    nbrs.fit(database_features)
    distances, indices = nbrs.kneighbors(query_features)
    return indices, distances


def compute_distance_matrix(features_dict, labels_dict):
    """
    Compute inter-class and intra-class distance statistics

    Args:
        features_dict: dict mapping class names to features
        labels_dict: dict mapping class names to sample count

    Returns:
        distance_matrix: (n_classes, n_classes) mean distance matrix
    """
    class_names = list(features_dict.keys())
    n_classes = len(class_names)
    distance_matrix = np.zeros((n_classes, n_classes))

    for i, class_i in enumerate(class_names):
        for j, class_j in enumerate(class_names):
            feats_i = features_dict[class_i]
            feats_j = features_dict[class_j]

            # Compute pairwise cosine distances
            # Normalize features
            feats_i_norm = feats_i / (np.linalg.norm(feats_i, axis=1, keepdims=True) + 1e-8)
            feats_j_norm = feats_j / (np.linalg.norm(feats_j, axis=1, keepdims=True) + 1e-8)

            # Cosine distance = 1 - cosine similarity
            similarity = feats_i_norm @ feats_j_norm.T
            distances = 1 - similarity

            # Mean distance
            distance_matrix[i, j] = distances.mean()

    return distance_matrix, class_names


def plot_tsne_visualization(
    embeddings_dict,
    labels_dict,
    retrieval_pairs=None,
    save_path='tsne_visualization.png',
    dpi=300
):
    """
    Create comprehensive T-SNE visualization with 4 subplots

    Args:
        embeddings_dict: dict mapping class names to (N, 2) embeddings
        labels_dict: dict mapping class names to readable labels
        retrieval_pairs: optional dict with 'query_indices' and 'db_indices' for drawing connections
        save_path: output path
        dpi: figure resolution
    """
    print(f"\nCreating T-SNE visualization...")

    # Prepare data
    all_embeddings = []
    all_labels = []
    all_colors = []

    # Color scheme for weather conditions
    color_map = {
        'clear': '#2E7D32',      # Green - Clear weather (database)
        'snow': '#1976D2',       # Blue - Snow
        'rain': '#D32F2F'        # Red - Rain
    }

    label_map = {
        'clear': 'Clear (Database)',
        'snow': 'Snow (Query)',
        'rain': 'Rain (Query)'
    }

    # Collect all data
    offset = 0
    class_offsets = {}
    for weather_type, embedding in embeddings_dict.items():
        all_embeddings.append(embedding)
        n_samples = len(embedding)
        all_labels.extend([label_map[weather_type]] * n_samples)
        all_colors.extend([color_map[weather_type]] * n_samples)
        class_offsets[weather_type] = (offset, offset + n_samples)
        offset += n_samples

    all_embeddings = np.vstack(all_embeddings)

    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(20, 16))

    # ------ Subplot 1: Simple scatter plot ------
    ax1 = plt.subplot(2, 2, 1)

    for weather_type in ['clear', 'snow', 'rain']:
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

    # Plot database first (background)
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

    # Draw retrieval connections if provided
    if retrieval_pairs is not None:
        query_types = ['snow', 'rain']
        connection_colors = {'snow': '#64B5F6', 'rain': '#EF5350'}  # Lighter colors for lines

        for query_type in query_types:
            if query_type in retrieval_pairs:
                pairs = retrieval_pairs[query_type]
                query_emb = embeddings_dict[query_type]
                db_emb = embeddings_dict['clear']

                # Draw only first 50 connections per type to avoid clutter
                n_connections = min(50, len(pairs))
                for i in range(n_connections):
                    query_idx = i
                    db_idx = pairs[i][0]  # Top-1 match

                    ax2.plot(
                        [query_emb[query_idx, 0], db_emb[db_idx, 0]],
                        [query_emb[query_idx, 1], db_emb[db_idx, 1]],
                        c=connection_colors[query_type],
                        alpha=0.15,
                        linewidth=0.5,
                        zorder=1
                    )

    # Plot queries on top
    for weather_type in ['snow', 'rain']:
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

    for weather_type in ['clear', 'snow', 'rain']:
        if weather_type in embeddings_dict:
            embedding = embeddings_dict[weather_type]

            # Scatter plot
            ax3.scatter(
                embedding[:, 0], embedding[:, 1],
                c=color_map[weather_type],
                label=label_map[weather_type],
                alpha=0.5,
                s=25,
                edgecolors='white',
                linewidth=0.3
            )

            # KDE density contours
            if len(embedding) > 10:  # Need enough points for KDE
                try:
                    kde = gaussian_kde(embedding.T)

                    # Create grid
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

                    # Plot contours
                    ax3.contour(
                        xx, yy, density,
                        colors=color_map[weather_type],
                        alpha=0.4,
                        linewidths=1.5,
                        levels=5
                    )
                except:
                    pass  # Skip if KDE fails

    ax3.set_title('(c) Cluster Boundaries with Density Estimation', fontsize=14, fontweight='bold')
    ax3.set_xlabel('T-SNE Dimension 1', fontsize=12)
    ax3.set_ylabel('T-SNE Dimension 2', fontsize=12)
    ax3.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_aspect('equal')

    # ------ Subplot 4: Distance matrix heatmap ------
    ax4 = plt.subplot(2, 2, 4)

    # Compute distance matrix
    features_dict = {}
    for weather_type in ['clear', 'snow', 'rain']:
        if weather_type in embeddings_dict:
            # Get original features (need to pass them separately)
            # For now, use embeddings as proxy
            features_dict[weather_type] = embeddings_dict[weather_type]

    # Simple distance computation on embeddings
    class_names = ['Clear', 'Snow', 'Rain']
    weather_keys = ['clear', 'snow', 'rain']
    n_classes = len([k for k in weather_keys if k in embeddings_dict])

    distance_matrix = np.zeros((n_classes, n_classes))
    actual_classes = []

    idx_i = 0
    for i, key_i in enumerate(weather_keys):
        if key_i not in embeddings_dict:
            continue
        actual_classes.append(class_names[i])
        emb_i = embeddings_dict[key_i]

        idx_j = 0
        for j, key_j in enumerate(weather_keys):
            if key_j not in embeddings_dict:
                continue
            emb_j = embeddings_dict[key_j]

            # Compute mean pairwise Euclidean distance
            if idx_i == idx_j:
                # Intra-class: random pairs
                n_samples = min(100, len(emb_i))
                rand_idx1 = np.random.choice(len(emb_i), n_samples, replace=False)
                rand_idx2 = np.random.choice(len(emb_i), n_samples, replace=False)
                distances = np.linalg.norm(emb_i[rand_idx1] - emb_i[rand_idx2], axis=1)
            else:
                # Inter-class: sample pairs
                n_samples = min(100, len(emb_i), len(emb_j))
                rand_idx1 = np.random.choice(len(emb_i), n_samples, replace=False)
                rand_idx2 = np.random.choice(len(emb_j), n_samples, replace=False)
                distances = np.linalg.norm(emb_i[rand_idx1] - emb_j[rand_idx2], axis=1)

            distance_matrix[idx_i, idx_j] = distances.mean()
            idx_j += 1
        idx_i += 1

    # Plot heatmap
    im = ax4.imshow(distance_matrix, cmap='RdYlGn_r', aspect='auto')

    # Set ticks
    ax4.set_xticks(np.arange(n_classes))
    ax4.set_yticks(np.arange(n_classes))
    ax4.set_xticklabels(actual_classes)
    ax4.set_yticklabels(actual_classes)

    # Rotate x labels
    plt.setp(ax4.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Mean Euclidean Distance', rotation=270, labelpad=20, fontsize=11)

    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax4.text(j, i, f'{distance_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)

    ax4.set_title('(d) Inter-class Distance Matrix', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Weather Condition', fontsize=12)
    ax4.set_ylabel('Weather Condition', fontsize=12)

    # Overall title
    fig.suptitle('T-SNE Visualization of BEV Geo-localization Descriptors (8448-D → 2-D)',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    print(f"Saving figure to: {save_path}")
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved successfully!")

    plt.close()


def main():
    """Main function"""
    # Configuration
    CHECKPOINT_PATH = '/data/users/cxw/pro/clav/results/boreas_descriptor_training_optimized_20251113_2345/best.pt'

    DATA_DIR = Path('/data/users/cxw/pro/clav/data')
    CLEAR_DB_PICKLE = DATA_DIR / 'boreas_bev_snow_evaluation_database_spatial.pickle'
    SNOW_QUERY_PICKLE = DATA_DIR / 'boreas_bev_snow_evaluation_query_spatial.pickle'
    RAIN_QUERY_PICKLE = DATA_DIR / 'boreas_bev_rain_evaluation_query_spatial.pickle'

    OUTPUT_DIR = Path('/data/users/cxw/pro/clav/Figure')
    OUTPUT_DIR.mkdir(exist_ok=True)

    OUTPUT_IMAGE = OUTPUT_DIR / 'tsne_geolocation_results.png'
    OUTPUT_DATA = OUTPUT_DIR / 'tsne_embeddings.npz'

    DEVICE = 'cuda'  # Will use GPU 7 via CUDA_VISIBLE_DEVICES
    BATCH_SIZE = 16
    TSNE_PERPLEXITY = 30
    TSNE_ITERATIONS = 1000

    print("="*80)
    print("T-SNE Visualization for BEV Geo-localization")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Output image: {OUTPUT_IMAGE}")
    print(f"Output data: {OUTPUT_DATA}")
    print("")

    # Load model
    model, config = load_checkpoint(CHECKPOINT_PATH, device=DEVICE)

    # Extract features
    print("\n" + "="*80)
    print("STEP 1: Feature Extraction")
    print("="*80)

    clear_features, clear_paths, clear_meta = extract_features_from_pickle(
        model, CLEAR_DB_PICKLE, device=DEVICE, batch_size=BATCH_SIZE, desc="Clear DB"
    )

    snow_features, snow_paths, snow_meta = extract_features_from_pickle(
        model, SNOW_QUERY_PICKLE, device=DEVICE, batch_size=BATCH_SIZE, desc="Snow Query"
    )

    rain_features, rain_paths, rain_meta = extract_features_from_pickle(
        model, RAIN_QUERY_PICKLE, device=DEVICE, batch_size=BATCH_SIZE, desc="Rain Query"
    )

    # Combine all features for T-SNE
    print("\n" + "="*80)
    print("STEP 2: T-SNE Dimensionality Reduction")
    print("="*80)

    all_features = np.vstack([clear_features, snow_features, rain_features])
    print(f"Total samples: {len(all_features)}")
    print(f"Feature dimension: {all_features.shape[1]}")

    # Compute T-SNE
    all_embeddings = compute_tsne(
        all_features,
        perplexity=TSNE_PERPLEXITY,
        max_iter=TSNE_ITERATIONS
    )

    # Split embeddings back
    n_clear = len(clear_features)
    n_snow = len(snow_features)
    n_rain = len(rain_features)

    clear_embedding = all_embeddings[:n_clear]
    snow_embedding = all_embeddings[n_clear:n_clear+n_snow]
    rain_embedding = all_embeddings[n_clear+n_snow:]

    embeddings_dict = {
        'clear': clear_embedding,
        'snow': snow_embedding,
        'rain': rain_embedding
    }

    # Compute retrieval pairs
    print("\n" + "="*80)
    print("STEP 3: Computing Retrieval Pairs")
    print("="*80)

    snow_pairs, snow_distances = compute_retrieval_pairs(snow_features, clear_features, k=1)
    rain_pairs, rain_distances = compute_retrieval_pairs(rain_features, clear_features, k=1)

    retrieval_pairs = {
        'snow': snow_pairs,
        'rain': rain_pairs
    }

    # Compute statistics
    print(f"\nSnow queries: {len(snow_pairs)}")
    print(f"  Mean retrieval distance: {snow_distances.mean():.4f}")
    print(f"  Median retrieval distance: {np.median(snow_distances):.4f}")

    print(f"\nRain queries: {len(rain_pairs)}")
    print(f"  Mean retrieval distance: {rain_distances.mean():.4f}")
    print(f"  Median retrieval distance: {np.median(rain_distances):.4f}")

    # Save embeddings
    print("\n" + "="*80)
    print("STEP 4: Saving Data")
    print("="*80)

    np.savez(
        OUTPUT_DATA,
        clear_embedding=clear_embedding,
        snow_embedding=snow_embedding,
        rain_embedding=rain_embedding,
        clear_features=clear_features,
        snow_features=snow_features,
        rain_features=rain_features,
        snow_pairs=snow_pairs,
        rain_pairs=rain_pairs,
        clear_paths=clear_paths,
        snow_paths=snow_paths,
        rain_paths=rain_paths
    )
    print(f"Saved embeddings and features to: {OUTPUT_DATA}")

    # Create visualization
    print("\n" + "="*80)
    print("STEP 5: Creating Visualization")
    print("="*80)

    labels_dict = {
        'clear': f'Clear (DB, n={n_clear})',
        'snow': f'Snow (Query, n={n_snow})',
        'rain': f'Rain (Query, n={n_rain})'
    }

    plot_tsne_visualization(
        embeddings_dict=embeddings_dict,
        labels_dict=labels_dict,
        retrieval_pairs=retrieval_pairs,
        save_path=str(OUTPUT_IMAGE),
        dpi=300
    )

    print("\n" + "="*80)
    print("COMPLETED!")
    print("="*80)
    print(f"Visualization saved to: {OUTPUT_IMAGE}")
    print(f"Embeddings saved to: {OUTPUT_DATA}")
    print("")


if __name__ == '__main__':
    main()
