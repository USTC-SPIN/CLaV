#!/bin/bash
# KITTI数据集模型评估脚本
# Created: 2025-11-08
# Purpose: 评估训练好的KITTI模型（跨天气地理定位性能）

PROJECT_ROOT="/data/users/cxw/pro/clav"
cd "$PROJECT_ROOT" || exit 1

echo "========================================"
echo "KITTI模型评估"
echo "========================================"
echo ""

# 设置Python路径
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
# 默认参数
CHECKPOINT="/data/users/cxw/pro/clav/results/kitti_descriptor_training_v2_20251109_2210/best.pt"
CONFIG="configs/kitti/kitti_stage3_training_v2.yaml"
DEVICE="cuda"
BATCH_SIZE=64
GPU_ID=5

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --help)
            echo "用法: bash $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --checkpoint PATH    模型checkpoint路径（必需）"
            echo "  --config PATH        配置文件路径（默认: configs/kitti/kitti_stage3_training.yaml）"
            echo "  --device DEVICE      设备（默认: cuda）"
            echo "  --batch_size N       批量大小（默认: 64）"
            echo "  --gpu N              GPU ID（默认: 5）"
            echo "  --help               显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  bash $0 --checkpoint results/kitti_descriptor_training_*/best.pt"
            echo "  bash $0 --checkpoint results/kitti_descriptor_training_*/best.pt --gpu 1"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 如果没有指定checkpoint，尝试自动查找最新的
if [ -z "$CHECKPOINT" ]; then
    echo "未指定checkpoint，正在查找最新的训练结果..."

    # 查找最新的descriptor training checkpoint
    LATEST_DIR=$(ls -td results/kitti_descriptor_training_* 2>/dev/null | head -1)

    if [ -n "$LATEST_DIR" ] && [ -f "$LATEST_DIR/best.pt" ]; then
        CHECKPOINT="$LATEST_DIR/best.pt"
        echo "  ✓ 找到: $CHECKPOINT"
    else
        echo ""
        echo "❌ 错误: 未找到checkpoint文件"
        echo ""
        echo "请指定checkpoint路径:"
        echo "  bash $0 --checkpoint results/kitti_descriptor_training_<timestamp>/best.pt"
        echo ""
        echo "或确保已完成Stage3训练:"
        echo "  bash scripts/train_kitti_stage3.sh"
        exit 1
    fi
fi

echo ""
echo "评估配置:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Config: $CONFIG"
echo "  Device: $DEVICE"
echo "  Batch Size: $BATCH_SIZE"
echo "  GPU ID: $GPU_ID"
echo ""

# 检查checkpoint文件
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint文件不存在: $CHECKPOINT"
    exit 1
fi

# 检查配置文件
if [ ! -f "$CONFIG" ]; then
    echo "❌ 配置文件不存在: $CONFIG"
    exit 1
fi

# 检查评估数据文件
echo "检查评估数据文件..."
EVAL_FILES_MISSING=false

# 从配置文件中读取评估pickle路径
eval_pickles=$(python3 -c "
import yaml
try:
    with open('$CONFIG') as f:
        cfg = yaml.safe_load(f)
    eval_cfg = cfg.get('data', {}).get('eval_pickles', {})
    for weather, paths in eval_cfg.items():
        db_path = paths.get('database', '')
        q_path = paths.get('query', '')
        print(f'{db_path}')
        print(f'{q_path}')
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
" 2>/dev/null)

if [ -n "$eval_pickles" ]; then
    for file in $eval_pickles; do
        if [ ! -f "$file" ]; then
            echo "  ❌ 缺失: $file"
            EVAL_FILES_MISSING=true
        else
            echo "  ✓ 存在: $file"
        fi
    done
else
    echo "  ⚠️  警告: 无法从配置文件读取评估文件路径"
fi

if [ "$EVAL_FILES_MISSING" = true ]; then
    echo ""
    echo "错误: 部分评估数据文件不存在"
    echo "请先运行数据生成脚本:"
    echo "  bash data_pre/kitti/run_all_kitti_generation.py"
    exit 1
fi

echo ""
echo "所有数据文件检查通过 ✓"
echo ""

# 显示模型信息
echo "模型信息:"
python3 -c "
import torch
import sys

checkpoint = torch.load('$CHECKPOINT', map_location='cpu')

if 'epoch' in checkpoint:
    print(f'  Epoch: {checkpoint[\"epoch\"]}')
if 'best_val_loss' in checkpoint:
    print(f'  Best Val Loss: {checkpoint.get(\"best_val_loss\", \"N/A\")}')
if 'train_loss' in checkpoint:
    print(f'  Train Loss: {checkpoint.get(\"train_loss\", \"N/A\")}')

# 显示配置信息（如果保存了）
if 'config' in checkpoint:
    cfg = checkpoint['config']
    model_cfg = cfg.get('model', {})
    print(f'  Skip Denoising: {model_cfg.get(\"skip_denoising\", \"N/A\")}')
    stage2_ckpt = model_cfg.get('stage2', {}).get('checkpoint_path', 'N/A')
    if stage2_ckpt != 'N/A':
        print(f'  Stage2 Checkpoint: .../{stage2_ckpt.split(\"/\")[-2]}/{stage2_ckpt.split(\"/\")[-1]}')
" 2>/dev/null || echo "  （无法读取模型信息）"

echo ""
echo "========================================"
echo "开始评估..."
echo "========================================"
echo ""

# 设置GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 运行评估
python src/evaluation/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================"
    echo "评估完成 ✓"
    echo "========================================"
else
    echo "========================================"
    echo "评估失败 ❌ (退出码: $EXIT_CODE)"
    echo "========================================"
fi

exit $EXIT_CODE
