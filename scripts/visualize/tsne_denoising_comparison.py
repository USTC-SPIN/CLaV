#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T-SNE Visualization: Denoising Comparison for NCLT Snow Retrieval
对比去噪前后在NCLT雪天检索任务中的T-SNE特征分布

Created: 2025-11-14 14:23
Purpose: Compare feature distributions with and without denoising for snow query retrieval
"""

import torch
import numpy as np
import sys
import os
import pickle
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Add parent directory to path
PARENT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PARENT_DIR))

from src.models.clav import CLaV
from evaluation.evaluate import extract_descriptors


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")

    if 'cuda' in device:
        torch.cuda.empty_cache()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})

    model = CLaV(config)

    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
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
    """Parse pickle data to extract image paths and coordinates"""
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


def extract_features_from_pickle(model, pickle_path, device='cuda', batch_size=16, skip_denoising=False, desc="Extracting"):
    """Extract features from pickle file"""
    print(f"\nLoading data from: {pickle_path}")
    data = load_data_pickle(pickle_path)

    image_paths, coords = parse_pickle_data(data)
    print(f"Found {len(image_paths)} images")
    print(f"Skip denoising: {skip_denoising}")

    features = extract_descriptors(
        model=model,
        image_paths=image_paths,
        device=device,
        batch_size=batch_size,
        show_progress=True,
        skip_denoising=skip_denoising
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


def plot_clean_tsne(embeddings_dict, save_path, figsize=(10, 10), dpi=300):
    """Create clean T-SNE scatter plot without any decorations"""

    color_map = {
        'clear': '#2E7D32',      # Green
        'snow': '#1976D2',       # Blue
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Plot clear database
    if 'clear' in embeddings_dict:
        embedding = embeddings_dict['clear']
        ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=color_map['clear'],
            alpha=0.6,
            s=40,
            edgecolors='white',
            linewidth=0.5
        )

    # Plot snow queries
    if 'snow' in embeddings_dict:
        embedding = embeddings_dict['snow']
        ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=color_map['snow'],
            alpha=0.6,
            s=40,
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

    # Save
    print(f"Saving figure to: {save_path}")
    plt.savefig(save_path, dpi=dpi, facecolor='white', pad_inches=0)
    plt.close()

    print(f"✓ Figure saved successfully!")


def main():
    """Main function"""

    # Configuration
    CHECKPOINT_PATH = '/data/users/cxw/pro/clav/results/boreas_descriptor_training_optimized_20251113_2345/best.pt'

    DATA_DIR = Path('/data/users/cxw/pro/clav/data')
    CLEAR_DB_PICKLE = DATA_DIR / 'nclt_bev_snow_evaluation_database.pickle'
    SNOW_QUERY_PICKLE = DATA_DIR / 'nclt_bev_snow_evaluation_query.pickle'

    OUTPUT_DIR = Path('/data/users/cxw/pro/clav/Figure/paper_tsne')
    OUTPUT_DIR.mkdir(exist_ok=True)

    OUTPUT_WITHOUT_DENOISING = OUTPUT_DIR / 'tsne_nclt_snow_without_denoising.png'
    OUTPUT_WITH_DENOISING = OUTPUT_DIR / 'tsne_nclt_snow_with_denoising.png'
    OUTPUT_DATA = OUTPUT_DIR / 'tsne_nclt_snow_denoising_comparison.npz'

    DEVICE = 'cuda'
    BATCH_SIZE = 16
    TSNE_PERPLEXITY = 30
    TSNE_ITERATIONS = 1000

    print("="*80)
    print("T-SNE Visualization: NCLT Snow Retrieval Denoising Comparison")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Output (without denoising): {OUTPUT_WITHOUT_DENOISING}")
    print(f"Output (with denoising): {OUTPUT_WITH_DENOISING}")
    print("")

    # Load model
    model, config = load_checkpoint(CHECKPOINT_PATH, device=DEVICE)

    # ========================================================================
    # PART 1: Extract features WITHOUT denoising
    # ========================================================================
    print("\n" + "="*80)
    print("PART 1: Feature Extraction WITHOUT Denoising")
    print("="*80)

    print("\n" + "-"*80)
    print("Extracting Clear Database Features (without denoising)")
    print("-"*80)
    clear_features_no_denoise, clear_paths, clear_meta = extract_features_from_pickle(
        model, CLEAR_DB_PICKLE, device=DEVICE, batch_size=BATCH_SIZE,
        skip_denoising=True, desc="Clear DB (no denoise)"
    )

    print("\n" + "-"*80)
    print("Extracting Snow Query Features (without denoising)")
    print("-"*80)
    snow_features_no_denoise, snow_paths, snow_meta = extract_features_from_pickle(
        model, SNOW_QUERY_PICKLE, device=DEVICE, batch_size=BATCH_SIZE,
        skip_denoising=True, desc="Snow Query (no denoise)"
    )

    # Compute T-SNE for without denoising
    print("\n" + "-"*80)
    print("Computing T-SNE (without denoising)")
    print("-"*80)
    all_features_no_denoise = np.vstack([clear_features_no_denoise, snow_features_no_denoise])
    print(f"Total samples: {len(all_features_no_denoise)}")

    all_embeddings_no_denoise = compute_tsne(
        all_features_no_denoise,
        perplexity=TSNE_PERPLEXITY,
        max_iter=TSNE_ITERATIONS
    )

    n_clear = len(clear_features_no_denoise)
    clear_embedding_no_denoise = all_embeddings_no_denoise[:n_clear]
    snow_embedding_no_denoise = all_embeddings_no_denoise[n_clear:]

    embeddings_no_denoise = {
        'clear': clear_embedding_no_denoise,
        'snow': snow_embedding_no_denoise
    }

    # ========================================================================
    # PART 2: Extract features WITH denoising
    # ========================================================================
    print("\n" + "="*80)
    print("PART 2: Feature Extraction WITH Denoising")
    print("="*80)

    print("\n" + "-"*80)
    print("Extracting Clear Database Features (with denoising)")
    print("-"*80)
    clear_features_with_denoise, _, _ = extract_features_from_pickle(
        model, CLEAR_DB_PICKLE, device=DEVICE, batch_size=BATCH_SIZE,
        skip_denoising=False, desc="Clear DB (with denoise)"
    )

    print("\n" + "-"*80)
    print("Extracting Snow Query Features (with denoising)")
    print("-"*80)
    snow_features_with_denoise, _, _ = extract_features_from_pickle(
        model, SNOW_QUERY_PICKLE, device=DEVICE, batch_size=BATCH_SIZE,
        skip_denoising=False, desc="Snow Query (with denoise)"
    )

    # Compute T-SNE for with denoising
    print("\n" + "-"*80)
    print("Computing T-SNE (with denoising)")
    print("-"*80)
    all_features_with_denoise = np.vstack([clear_features_with_denoise, snow_features_with_denoise])
    print(f"Total samples: {len(all_features_with_denoise)}")

    all_embeddings_with_denoise = compute_tsne(
        all_features_with_denoise,
        perplexity=TSNE_PERPLEXITY,
        max_iter=TSNE_ITERATIONS
    )

    clear_embedding_with_denoise = all_embeddings_with_denoise[:n_clear]
    snow_embedding_with_denoise = all_embeddings_with_denoise[n_clear:]

    embeddings_with_denoise = {
        'clear': clear_embedding_with_denoise,
        'snow': snow_embedding_with_denoise
    }

    # ========================================================================
    # PART 3: Save data
    # ========================================================================
    print("\n" + "="*80)
    print("PART 3: Saving Data")
    print("="*80)

    np.savez(
        OUTPUT_DATA,
        clear_embedding_no_denoise=clear_embedding_no_denoise,
        snow_embedding_no_denoise=snow_embedding_no_denoise,
        clear_embedding_with_denoise=clear_embedding_with_denoise,
        snow_embedding_with_denoise=snow_embedding_with_denoise,
        clear_features_no_denoise=clear_features_no_denoise,
        snow_features_no_denoise=snow_features_no_denoise,
        clear_features_with_denoise=clear_features_with_denoise,
        snow_features_with_denoise=snow_features_with_denoise,
        clear_paths=clear_paths,
        snow_paths=snow_paths
    )
    print(f"Saved embeddings and features to: {OUTPUT_DATA}")

    # ========================================================================
    # PART 4: Create visualizations
    # ========================================================================
    print("\n" + "="*80)
    print("PART 4: Creating Visualizations")
    print("="*80)

    print("\nGenerating visualization WITHOUT denoising...")
    plot_clean_tsne(
        embeddings_dict=embeddings_no_denoise,
        save_path=str(OUTPUT_WITHOUT_DENOISING),
        figsize=(10, 10),
        dpi=300
    )

    print("\nGenerating visualization WITH denoising...")
    plot_clean_tsne(
        embeddings_dict=embeddings_with_denoise,
        save_path=str(OUTPUT_WITH_DENOISING),
        figsize=(10, 10),
        dpi=300
    )

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("COMPLETED!")
    print("="*80)
    print(f"\nNCLT Snow Retrieval Statistics:")
    print(f"  Clear database: {n_clear} samples")
    print(f"  Snow queries: {len(snow_features_no_denoise)} samples")
    print(f"  Total: {n_clear + len(snow_features_no_denoise)} samples")
    print(f"\nGenerated files:")
    print(f"  Without denoising: {OUTPUT_WITHOUT_DENOISING}")
    print(f"  With denoising: {OUTPUT_WITH_DENOISING}")
    print(f"  Embeddings data: {OUTPUT_DATA}")
    print("")


if __name__ == '__main__':
    main()
