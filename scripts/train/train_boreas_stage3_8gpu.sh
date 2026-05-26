#!/bin/bash
# Boreas数据集Stage3训练启动脚本（8卡并行）
# Created: 2025-11-13 17:30
# Purpose: 使用8卡并行训练Boreas数据集的Stage3描述符头（跨天气地理定位）

# 项目根目录
PROJECT_ROOT="/data/users/cxw/pro/clav"
cd "$PROJECT_ROOT" || exit 1

# 训练配置文件（默认使用优化版配置）
CONFIG="${1:-configs/boreas/boreas_stage3_training_v3.yaml}"
BASE_CONFIG="configs/boreas/base_config.yaml"

# 检查数据文件是否存在
echo "========================================"
echo "Boreas Stage3 描述符训练准备 (8卡并行)"
echo "========================================"
echo ""
echo "检查数据文件..."

DATA_FILES=(
    "data/boreas_bev_training_queries_spatial.pickle"
    "data/boreas_bev_test_queries_spatial.pickle"
)

ALL_EXIST=true
for file in "${DATA_FILES[@]}"; do
    if [ ! -f "$PROJECT_ROOT/$file" ]; then
        echo "  ❌ 缺失: $file"
        ALL_EXIST=false
    else
        # 显示文件大小和修改时间
        size=$(ls -lh "$PROJECT_ROOT/$file" | awk '{print $5}')
        time=$(ls -l "$PROJECT_ROOT/$file" | awk '{print $6, $7, $8}')
        echo "  ✓ 存在: $file (大小: $size, 时间: $time)"
    fi
done

if [ "$ALL_EXIST" = false ]; then
    echo ""
    echo "错误: 部分数据文件不存在"
    echo "请先运行数据生成脚本:"
    echo "  bash data_pre/boreas/run_all_boreas_data_generation.sh"
    exit 1
fi

echo ""
echo "所有数据文件检查通过 ✓"
echo ""

# 检查配置文件
if [ ! -f "$PROJECT_ROOT/$CONFIG" ]; then
    echo "❌ 配置文件不存在: $CONFIG"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/$BASE_CONFIG" ]; then
    echo "❌ 基础配置文件不存在: $BASE_CONFIG"
    exit 1
fi

echo "使用配置文件:"
echo "  主配置: $CONFIG"
echo "  基础配置: $BASE_CONFIG"
echo ""

# 检查Stage2 checkpoint
echo "检查Stage2模型..."
STAGE2_CHECKPOINT=$(python3 -c "
import yaml
with open('$PROJECT_ROOT/$CONFIG') as f:
    cfg = yaml.safe_load(f)
checkpoint = cfg.get('model', {}).get('stage2', {}).get('checkpoint_path', '')
print(checkpoint if checkpoint else '')
" 2>/dev/null)

if [ -n "$STAGE2_CHECKPOINT" ]; then
    if [ -f "$PROJECT_ROOT/$STAGE2_CHECKPOINT" ]; then
        echo "  ✓ Stage2 checkpoint: $STAGE2_CHECKPOINT"

        # 显示Stage2模型信息
        python3 -c "
import torch
ckpt = torch.load('$PROJECT_ROOT/$STAGE2_CHECKPOINT', map_location='cpu')
print(f'    Epoch: {ckpt.get(\"epoch\", \"N/A\")}')
print(f'    Loss: {ckpt.get(\"best_val_loss\", ckpt.get(\"train_loss\", \"N/A\"))}')
" 2>/dev/null
    else
        echo "  ❌ Stage2 checkpoint未找到: $STAGE2_CHECKPOINT"
        echo ""
        echo "请先训练Stage2模型或修改配置文件中的checkpoint路径"
        echo "训练Stage2: bash scripts/train_boreas_stage2.sh"
        exit 1
    fi
else
    echo "  ⚠️  未配置Stage2 checkpoint"
    echo "  将使用skip_denoising模式（不推荐）"
    echo ""
    read -p "是否继续？(y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "训练已取消"
        exit 1
    fi
fi
echo ""

# 显示训练信息
echo "========================================"
echo "Boreas Stage3 训练信息"
echo "========================================"
python3 -c "
import pickle
train_data = pickle.load(open('$PROJECT_ROOT/data/boreas_bev_training_queries_spatial.pickle', 'rb'))
test_data = pickle.load(open('$PROJECT_ROOT/data/boreas_bev_test_queries_spatial.pickle', 'rb'))
print(f'训练样本数: {len(train_data):,}')
print(f'测试样本数: {len(test_data):,}')
print(f'总计: {len(train_data) + len(test_data):,}')
print(f'')
print(f'任务: 跨天气地理定位（描述符学习）')
print(f'天气条件: Clear (database) ↔ Snow/Rain (query)')
print(f'数据来源: Boreas真实采集数据（已清理）')
print(f'空间分离: 测试区域 (820m×820m) vs 训练区域')
"
echo "========================================"
echo ""

# 显示训练参数
echo "训练参数:"
python3 -c "
import yaml
with open('$PROJECT_ROOT/$CONFIG') as f:
    cfg = yaml.safe_load(f)

# 读取base_config来获取device配置
with open('$PROJECT_ROOT/$BASE_CONFIG') as f:
    base_cfg = yaml.safe_load(f)

train_cfg = cfg.get('stage3_training', cfg.get('stage2_training', {}))
print(f'  Epochs: {train_cfg[\"epochs\"]}')
print(f'  Batch Size: {train_cfg[\"batch_size\"]} (per GPU)')
print(f'  Grad Accum Steps: {train_cfg.get(\"grad_accum_steps\", 1)}')
total_bs = train_cfg[\"batch_size\"] * train_cfg.get(\"grad_accum_steps\", 1) * 8
print(f'  Effective Batch Size: {total_bs} (8 GPUs)')
print(f'  Learning Rate: {train_cfg[\"optimizer\"][\"lr\"]}')
print(f'  Weight Decay: {train_cfg[\"optimizer\"].get(\"weight_decay\", 0.01)}')
print(f'  Scheduler: {train_cfg[\"scheduler\"][\"type\"]}')
print(f'  使用EMA: {train_cfg.get(\"use_ema\", False)}')
print(f'  混合精度: {base_cfg.get(\"device\", {}).get(\"mixed_precision\", True)}')
print(f'')
print(f'Freeze策略:')
freeze_cfg = train_cfg.get('freeze', {})
encoder_trainable = freeze_cfg.get('encoder_trainable_blocks', 0)
if freeze_cfg.get('freeze_encoder', True):
    print(f'  Encoder (DINOv2): 冻结')
else:
    print(f'  Encoder (DINOv2): 训练最后{encoder_trainable}层')
print(f'  Stage2 (DiT): {\"冻结\" if freeze_cfg.get(\"freeze_stage2\", True) else \"训练\"}')
print(f'  Descriptor Head: {\"冻结\" if freeze_cfg.get(\"freeze_descriptor\", False) else \"训练\"}')
"
echo ""

# 开始训练
echo "========================================"
echo "开始8卡并行训练 (GPU 0-7)..."
echo "========================================"
echo ""

# 设置Python路径
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 设置可见GPU（8卡）
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 创建日志目录
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# 生成日志文件名（带时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/boreas_stage3_8gpu_${TIMESTAMP}.log"

echo "训练配置:"
echo "  并行方式: DistributedDataParallel (DDP)"
echo "  GPU数量: 8"
echo "  Master Port: 29501"
echo "  日志文件: $LOG_FILE"
echo ""

# 使用nohup启动分布式训练，后台运行
# nproc_per_node: 每个节点的进程数（4个GPU）
echo "正在启动训练进程..."
if [ $# -gt 1 ]; then
    # 如果有额外参数，传递给训练脚本（跳过第一个参数，因为是config）
    nohup torchrun --nproc_per_node=4 --master_port=29501 \
        src/training/trainer.py \
        --config "$CONFIG" \
        --base_config "$BASE_CONFIG" \
        "${@:2}" > "$LOG_FILE" 2>&1 &
else
    nohup torchrun --nproc_per_node=4 --master_port=29501 \
        src/training/trainer.py \
        --config "$CONFIG" \
        --base_config "$BASE_CONFIG" > "$LOG_FILE" 2>&1 &
fi

# 获取进程ID
TRAIN_PID=$!

echo ""
echo "========================================"
echo "训练已在后台启动 ✓"
echo "========================================"
echo ""
echo "进程信息:"
echo "  PID: $TRAIN_PID"
echo "  日志: $LOG_FILE"
echo "  查看: tail -f $LOG_FILE"
