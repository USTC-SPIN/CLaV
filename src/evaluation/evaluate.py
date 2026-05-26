#!/usr/bin/env python3
"""
使用KDTree的评估脚本
参考MinkLoc3Dv2的评估方法，使用sklearn.KDTree进行最近邻搜索

CUDA_VISIBLE_DEVICES=4 python scripts/eval_with_kdtree.py \
    --checkpoint results/nclt_descriptor_training_20251107_1506/best.pt \
    --config configs/nclt_stage3_training.yaml

Created: 2025-11-07
Author: Based on MinkLoc3Dv2 pnv_evaluate.py
"""
import torch
import yaml
import numpy as np
import pickle
import os
import sys
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.neighbors import KDTree
from tqdm import tqdm
from typing import Dict, List, Tuple

# 添加项目路径
PROJECT_DIR = Path(__file__).parent.parent.parent  # src/evaluation -> src -> clav/
sys.path.insert(0, str(PROJECT_DIR))
os.chdir(PROJECT_DIR)  # 切换到项目根目录

from src.models.clav import CLaV


def load_pickle_data(pickle_path: str):
    """加载pickle文件"""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def extract_descriptors(
    model,
    image_paths: List[str],
    device: str,
    skip_denoising: bool = False,
    batch_size: int = 8,
    show_progress: bool = True
) -> np.ndarray:
    """
    提取图像描述符

    Args:
        model: CLaV模型
        image_paths: 图像路径列表
        device: 设备
        skip_denoising: 是否跳过去噪
        batch_size: 批量大小
        show_progress: 是否显示进度条

    Returns:
        descriptors: (N, D) numpy数组
    """
    model.eval()

    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    descriptors_list = []
    num_batches = (len(image_paths) + batch_size - 1) // batch_size

    iterator = range(num_batches)
    if show_progress:
        iterator = tqdm(iterator, desc='Extracting descriptors', ncols=100)

    with torch.no_grad():
        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]

            # 加载图像
            batch_images = []
            for img_path in batch_paths:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                batch_images.append(img_tensor)

            batch_tensor = torch.stack(batch_images).to(device)

            # 提取描述符
            batch_descriptors = model.extract_descriptor(batch_tensor, skip_denoising=skip_denoising)
            batch_descriptors = batch_descriptors.cpu().numpy()
            descriptors_list.append(batch_descriptors)

    # 合并所有描述符
    descriptors = np.concatenate(descriptors_list, axis=0)

    return descriptors


def build_ground_truth_from_pickle(
    query_entries: Dict,
    database_data_raw,
    database_entries: Dict
) -> List[List[int]]:
    """
    从pickle文件的预计算字段构建ground truth

    对于列表格式的数据，需要计算offset来调整索引

    Args:
        query_entries: query数据字典（已合并）
        database_data_raw: 原始database数据（用于计算offset）
        database_entries: database数据字典（已合并）

    Returns:
        ground_truth: 每个query对应的正样本database索引列表
    """
    ground_truth = []

    # 计算每个database set的offset（如果是列表格式）
    db_set_offsets = []
    if isinstance(database_data_raw, list):
        offset = 0
        for db_set in database_data_raw:
            db_set_offsets.append(offset)
            offset += len(db_set)

    for query_idx in sorted(query_entries.keys()):
        entry = query_entries[query_idx]

        # 提取预计算的ground truth（数字键）
        gt_indices = []
        for key in entry.keys():
            if isinstance(key, int):
                # 这个键指向一个database set的索引
                if isinstance(entry[key], list):
                    # 调整索引：加上对应database set的offset
                    if db_set_offsets and key < len(db_set_offsets):
                        adjusted_indices = [idx + db_set_offsets[key] for idx in entry[key]]
                        gt_indices.extend(adjusted_indices)
                    else:
                        # 如果没有offset信息，直接使用原始索引
                        gt_indices.extend(entry[key])

        ground_truth.append(gt_indices)

    return ground_truth


def build_ground_truth_from_positions(
    query_positions: np.ndarray,
    database_positions: np.ndarray,
    positive_threshold: float = 10.0
) -> List[List[int]]:
    """
    基于位置信息动态构建ground truth

    Args:
        query_positions: (N_q, 2) query位置
        database_positions: (N_db, 2) database位置
        positive_threshold: 正样本距离阈值（米）

    Returns:
        ground_truth: 每个query对应的正样本database索引列表
    """
    ground_truth = []

    for q_pos in query_positions:
        # 计算与所有database的欧氏距离
        distances = np.linalg.norm(database_positions - q_pos, axis=1)
        # 找到距离小于阈值的索引
        positive_indices = np.where(distances < positive_threshold)[0].tolist()
        ground_truth.append(positive_indices)

    return ground_truth


def compute_recall_with_kdtree(
    query_descriptors: np.ndarray,
    database_descriptors: np.ndarray,
    ground_truth: List[List[int]],
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """
    使用KDTree计算Recall@K和F1分数
    参考MinkLoc3Dv2的实现

    Args:
        query_descriptors: (N_q, D) 查询描述符
        database_descriptors: (N_db, D) 数据库描述符
        ground_truth: 每个query的正样本索引列表
        k_values: 要计算的K值列表

    Returns:
        metrics: 包含各个Recall@K和F1的字典
    """
    # 构建KDTree（使用欧氏距离）
    # 注释：当embeddings被L2归一化时，欧氏距离与余弦距离给出相同的最近邻结果
    database_kdtree = KDTree(database_descriptors)

    max_k = max(k_values)
    num_queries = len(query_descriptors)

    # 查询最近邻
    # distances: (N_q, max_k), indices: (N_q, max_k)
    distances, indices = database_kdtree.query(query_descriptors, k=max_k)

    # 计算Recall@K和F1
    recalls = {}
    f1_scores = {}

    for k in k_values:
        correct = 0
        num_evaluated = 0
        total_precision = 0.0
        total_recall = 0.0

        for query_idx in range(num_queries):
            true_neighbors = ground_truth[query_idx]

            # 跳过没有正样本的query
            if len(true_neighbors) == 0:
                continue

            num_evaluated += 1

            # 检查Top-K中是否有正样本
            top_k_indices = indices[query_idx, :k]

            # 如果Top-K中有任意一个正样本，则算正确
            if any(idx in true_neighbors for idx in top_k_indices):
                correct += 1

            # 计算precision和recall用于F1
            # Precision: Top-K中正样本的比例
            true_positives = sum(1 for idx in top_k_indices if idx in true_neighbors)
            precision = true_positives / k
            # Recall: 在所有正样本中被检索到的比例
            recall = true_positives / len(true_neighbors)

            total_precision += precision
            total_recall += recall

        # 计算Recall@K
        if num_evaluated > 0:
            recall_at_k = (correct / num_evaluated) * 100.0
            avg_precision = total_precision / num_evaluated
            avg_recall = total_recall / num_evaluated

            # 计算F1分数
            if avg_precision + avg_recall > 0:
                f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) * 100.0
            else:
                f1 = 0.0
        else:
            recall_at_k = 0.0
            f1 = 0.0

        recalls[f'recall@{k}'] = recall_at_k
        f1_scores[f'f1@{k}'] = f1

    # 计算Top-1% Recall（类似MinkLoc）
    one_percent_threshold = max(int(round(len(database_descriptors) / 100.0)), 1)
    one_percent_correct = 0
    num_evaluated = 0
    total_precision_1p = 0.0
    total_recall_1p = 0.0

    for query_idx in range(num_queries):
        true_neighbors = ground_truth[query_idx]
        if len(true_neighbors) == 0:
            continue

        num_evaluated += 1

        # 检查Top-1%中是否有正样本
        top_1percent_indices = indices[query_idx, :one_percent_threshold]
        if any(idx in true_neighbors for idx in top_1percent_indices):
            one_percent_correct += 1

        # 计算precision和recall用于F1
        true_positives = sum(1 for idx in top_1percent_indices if idx in true_neighbors)
        precision = true_positives / one_percent_threshold
        recall = true_positives / len(true_neighbors)

        total_precision_1p += precision
        total_recall_1p += recall

    if num_evaluated > 0:
        recalls['recall@1%'] = (one_percent_correct / num_evaluated) * 100.0
        avg_precision_1p = total_precision_1p / num_evaluated
        avg_recall_1p = total_recall_1p / num_evaluated

        if avg_precision_1p + avg_recall_1p > 0:
            f1_1p = 2 * (avg_precision_1p * avg_recall_1p) / (avg_precision_1p + avg_recall_1p) * 100.0
        else:
            f1_1p = 0.0
        f1_scores['f1@1%'] = f1_1p
    else:
        recalls['recall@1%'] = 0.0
        f1_scores['f1@1%'] = 0.0

    # 合并所有指标
    metrics = {**recalls, **f1_scores}
    return metrics


def evaluate_single_condition(
    model,
    database_pickle: str,
    query_pickle: str,
    dataset_folder: str,
    device: str,
    skip_denoising: bool = False,
    positive_threshold: float = 10.0,
    batch_size: int = 8,
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict:
    """
    评估单个天气条件

    Args:
        model: CLaV模型
        database_pickle: 数据库pickle路径
        query_pickle: 查询pickle路径
        dataset_folder: 数据集根目录
        device: 设备
        skip_denoising: 是否跳过去噪
        positive_threshold: 正样本距离阈值（米）
        batch_size: 批量大小
        k_values: 要计算的K值列表

    Returns:
        metrics: 评估指标字典
    """
    print(f"\nLoading data...")
    print(f"  Database: {database_pickle}")
    print(f"  Query: {query_pickle}")

    # 加载pickle数据
    database_data = load_pickle_data(database_pickle)
    query_data = load_pickle_data(query_pickle)

    # 解析数据格式
    # 格式1: 列表格式（MinkLoc/NCLT风格）- 列表中每个元素是一个session/set
    # 格式2: 字典格式（单一数据集）

    if isinstance(database_data, list):
        # 列表格式：合并所有sets
        print(f"  Database format: List of {len(database_data)} sets")
        database_entries = {}
        offset = 0
        for set_idx, dataset_set in enumerate(database_data):
            for local_idx, entry in dataset_set.items():
                database_entries[offset + local_idx] = entry
            offset += len(dataset_set)
    elif isinstance(database_data, dict):
        if 'databases' in database_data:
            database_entries = database_data['databases']
        else:
            database_entries = database_data
    else:
        raise ValueError(f"Unsupported database format: {type(database_data)}")

    if isinstance(query_data, list):
        # 列表格式：合并所有sets
        print(f"  Query format: List of {len(query_data)} sets")
        query_entries = {}
        offset = 0
        for set_idx, query_set in enumerate(query_data):
            for local_idx, entry in query_set.items():
                query_entries[offset + local_idx] = entry
            offset += len(query_set)
    elif isinstance(query_data, dict):
        if 'queries' in query_data:
            query_entries = query_data['queries']
        else:
            query_entries = query_data
    else:
        raise ValueError(f"Unsupported query format: {type(query_data)}")

    # 提取图像路径和位置
    database_paths = []
    database_positions = []
    for idx in sorted(database_entries.keys()):
        entry = database_entries[idx]
        file_key = 'query' if 'query' in entry else 'file'
        file_path = os.path.join(dataset_folder, entry[file_key])
        database_paths.append(file_path)
        if 'position' in entry:
            database_positions.append(entry['position'])

    query_paths = []
    query_positions = []
    for idx in sorted(query_entries.keys()):
        entry = query_entries[idx]
        file_key = 'query' if 'query' in entry else 'file'
        file_path = os.path.join(dataset_folder, entry[file_key])
        query_paths.append(file_path)
        if 'position' in entry:
            query_positions.append(entry['position'])

    print(f"  Database images: {len(database_paths)}")
    print(f"  Query images: {len(query_paths)}")

    # 提取描述符
    print("\nExtracting database descriptors...")
    database_descriptors = extract_descriptors(
        model, database_paths, device, skip_denoising, batch_size, show_progress=True
    )

    print("\nExtracting query descriptors...")
    query_descriptors = extract_descriptors(
        model, query_paths, device, skip_denoising, batch_size, show_progress=True
    )

    print(f"\nDescriptor shapes:")
    print(f"  Database: {database_descriptors.shape}")
    print(f"  Query: {query_descriptors.shape}")

    # 构建ground truth
    print("\nBuilding ground truth...")

    # 优先使用预计算的ground truth
    ground_truth = build_ground_truth_from_pickle(query_entries, database_data, database_entries)

    # 如果没有预计算的，使用位置信息
    if all(len(gt) == 0 for gt in ground_truth) and len(query_positions) > 0:
        print(f"  Using position-based ground truth (threshold={positive_threshold}m)")
        query_positions = np.array(query_positions)
        database_positions = np.array(database_positions)
        ground_truth = build_ground_truth_from_positions(
            query_positions, database_positions, positive_threshold
        )
    else:
        print(f"  Using precomputed ground truth from pickle")

    # 统计ground truth
    num_with_gt = sum(1 for gt in ground_truth if len(gt) > 0)
    avg_positives = np.mean([len(gt) for gt in ground_truth if len(gt) > 0]) if num_with_gt > 0 else 0
    print(f"  Queries with positives: {num_with_gt}/{len(ground_truth)}")
    print(f"  Average positives per query: {avg_positives:.1f}")

    # 计算Recall@K（使用KDTree）
    print("\nComputing metrics with KDTree...")
    metrics = compute_recall_with_kdtree(
        query_descriptors, database_descriptors, ground_truth, k_values
    )

    return metrics


def evaluate_all_conditions(
    checkpoint_path: str,
    config_path: str,
    device: str = 'cuda',
    batch_size: int = 8
) -> Dict:
    """
    评估所有天气条件

    Args:
        checkpoint_path: 模型checkpoint路径
        config_path: 配置文件路径
        device: 设备
        batch_size: 批量大小

    Returns:
        all_metrics: 所有天气条件的评估结果
    """
    print("="*80)
    print("Model Evaluation with KDTree (MinkLoc-style)")
    print("="*80)

    # 加载配置
    print(f"\nLoading config: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 加载base config
    if 'base_config' in config:
        base_config_path = Path(config_path).parent / config['base_config']
        if base_config_path.exists():
            with open(base_config_path, 'r') as f:
                base_config = yaml.safe_load(f)

            # 深度合并
            def deep_merge(base, override):
                result = base.copy()
                for key, value in override.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = deep_merge(result[key], value)
                    else:
                        result[key] = value
                return result

            config = deep_merge(base_config, config)

    # 创建模型
    print(f"\nCreating model...")
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device_obj}")

    model = CLaV(config=config)

    # 加载checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print(f"✓ Loaded state dict")
    else:
        model.load_state_dict(checkpoint)
        print(f"✓ Loaded checkpoint")

    model = model.to(device_obj)
    model.eval()

    # 获取配置参数
    dataset_folder = config['data']['dataset_folder']
    skip_denoising = config.get('model', {}).get('skip_denoising', False)
    eval_pickles = config['data'].get('eval_pickles', {})

    print(f"\nConfiguration:")
    print(f"  Dataset folder: {dataset_folder}")
    print(f"  Skip denoising: {skip_denoising}")
    print(f"  Weather conditions: {list(eval_pickles.keys())}")

    # 评估每个天气条件
    all_metrics = {}

    for weather_name, pickle_paths in eval_pickles.items():
        print("\n" + "="*80)
        print(f"Evaluating: {weather_name}")
        print("="*80)

        database_pickle = pickle_paths['database']
        query_pickle = pickle_paths['query']

        # 处理绝对路径和相对路径
        if not os.path.isabs(database_pickle):
            database_pickle = os.path.join(dataset_folder, database_pickle)
        if not os.path.isabs(query_pickle):
            query_pickle = os.path.join(dataset_folder, query_pickle)

        metrics = evaluate_single_condition(
            model=model,
            database_pickle=database_pickle,
            query_pickle=query_pickle,
            dataset_folder=dataset_folder,
            device=str(device_obj),
            skip_denoising=skip_denoising,
            batch_size=batch_size
        )

        all_metrics[weather_name] = metrics

        # 打印详细结果
        print("\n" + "-"*80)
        print(f"Results for {weather_name}:")
        print("-"*80)
        for metric_name, value in metrics.items():
            print(f"  {metric_name:20s}: {value:6.2f}%")
        print("-"*80)

    # 打印表格格式的汇总结果
    print("\n" + "="*80)
    print("Summary Table:")
    print("="*80)
    print_summary_table(all_metrics)

    return all_metrics


def print_summary_table(all_metrics: Dict[str, Dict[str, float]]):
    """
    打印表格格式的汇总结果

    Args:
        all_metrics: 所有天气条件的评估结果
    """
    if not all_metrics:
        return

    # 表头
    header = f"{'Weather Condition':<25} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@1%':>8} {'F1':>8}"
    separator = "-" * 80

    print(separator)
    print(header)
    print(separator)

    # 计算平均值用于存储
    avg_metrics = {}
    metric_keys = ['recall@1', 'recall@5', 'recall@10', 'recall@1%', 'f1@1']

    for key in metric_keys:
        values = [metrics.get(key, 0.0) for metrics in all_metrics.values()]
        avg_metrics[key] = np.mean(values) if values else 0.0

    # 打印每个天气条件的结果
    for weather_name, metrics in all_metrics.items():
        r1 = metrics.get('recall@1', 0.0)
        r5 = metrics.get('recall@5', 0.0)
        r10 = metrics.get('recall@10', 0.0)
        r1p = metrics.get('recall@1%', 0.0)
        f1 = metrics.get('f1@1', 0.0)

        row = f"{weather_name:<25} {r1:>7.2f}% {r5:>7.2f}% {r10:>7.2f}% {r1p:>7.2f}% {f1:>7.2f}%"
        print(row)

    # 打印分隔线
    print(separator)

    # 打印平均值
    avg_row = (f"{'Average':<25} {avg_metrics['recall@1']:>7.2f}% "
               f"{avg_metrics['recall@5']:>7.2f}% {avg_metrics['recall@10']:>7.2f}% "
               f"{avg_metrics['recall@1%']:>7.2f}% {avg_metrics['f1@1']:>7.2f}%")
    print(avg_row)
    print(separator)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate RAE model using KDTree')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for descriptor extraction')

    args = parser.parse_args()

    metrics = evaluate_all_conditions(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        batch_size=args.batch_size
    )

    print("\n✓ Evaluation completed!")
