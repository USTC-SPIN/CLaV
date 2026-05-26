#!/bin/bash
# KITTI数据集Stage3训练启动脚本 v2
# Created: 2025-11-09
# Modified from: train_kitti_stage3.sh
# Purpose: 训练改进版KITTI Stage3配置（解决过拟合和LR问题）

# 项目根目录
PROJECT_ROOT="/data/users/cxw/pro/clav"
cd "$PROJECT_ROOT" || exit 1

# 训练配置文件（使用v2改进版）
CONFIG="configs/kitti/boreas_stage3_training_skip-de.yaml"
BASE_CONFIG="configs/base_config.yaml"

# 检查数据文件是否存在
echo "========================================"
echo "KITTI Stage3 v2 描述符训练准备"
echo "========================================"
echo ""
echo "✨ v2改进版配置："
echo "  - 修复LR调度问题（T_0=25, T_mult=1）"
echo "  - 降低初始LR（5e-5）"
echo "  - 增强数据增强（rotation=±15°）"
echo "  - 延长训练（150 epochs）"
echo "  - 目标：Fog 65-75%, Rain 40-50%"
echo ""
echo "检查数据文件..."

DATA_FILES=(
    "data/kitti_bev_training_queries.pickle"
    "data/kitti_bev_test_queries.pickle"
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
    echo "  bash data_pre/kitti/run_all_kitti_generation.py"
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
echo "  主配置: $CONFIG (v2改进版)"
echo "  基础配置: $BASE_CONFIG"
echo ""

# 检查Stage2 checkpoint
echo "检查Stage2模型..."
STAGE2_CHECKPOINT=$(python3 -c "
import yaml
import glob
with open('$PROJECT_ROOT/$CONFIG') as f:
    cfg = yaml.safe_load(f)
checkpoint_pattern = cfg.get('model', {}).get('stage2', {}).get('checkpoint_path', '')
if '*' in checkpoint_pattern:
    # 通配符模式，查找最新的checkpoint
    matches = sorted(glob.glob('$PROJECT_ROOT/' + checkpoint_pattern))
    print(matches[-1].replace('$PROJECT_ROOT/', '') if matches else '')
else:
    print(checkpoint_pattern if checkpoint_pattern else '')
" 2>/dev/null)

if [ -n "$STAGE2_CHECKPOINT" ]; then
    if [ -f "$PROJECT_ROOT/$STAGE2_CHECKPOINT" ]; then
        echo "  ✓ Stage2 checkpoint: $STAGE2_CHECKPOINT"
    else
        echo "  ❌ Stage2 checkpoint未找到: $STAGE2_CHECKPOINT"
        echo ""
        echo "请先运行Stage2训练:"
        echo "  bash scripts/train_kitti_stage2.sh"
        echo ""
        echo "或手动更新配置文件中的checkpoint路径:"
        echo "  $CONFIG"
        exit 1
    fi
else
    echo "  ⚠️  未配置Stage2 checkpoint"
    echo "  将使用未训练的Stage2模型（不推荐）"
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
echo "KITTI Stage3 v2 训练信息"
echo "========================================"
python3 -c "
import pickle
train_data = pickle.load(open('$PROJECT_ROOT/data/kitti_bev_training_queries.pickle', 'rb'))
test_data = pickle.load(open('$PROJECT_ROOT/data/kitti_bev_test_queries.pickle', 'rb'))
print(f'训练样本数: {len(train_data)}')
print(f'测试样本数: {len(test_data)}')
print(f'总计: {len(train_data) + len(test_data)}')
print(f'')
print(f'任务: 跨天气地理定位（描述符学习）')
print(f'天气条件: Orin (database) ↔ Fog/Rain/Snow (query)')
print(f'数据来源: KITTI天气仿真数据')
print(f'空间分离: 前20%轨迹 (测试) vs 后80%轨迹 (训练)')
print(f'')
print(f'v2改进：')
print(f'  - 修复学习率调度（避免epoch 59 LR归零）')
print(f'  - 增强数据增强（rotation ±15°, scale [0.85, 1.15]）')
print(f'  - 加强正则化（weight_decay 0.05）')
print(f'  - 延长训练（150 epochs）')
"
echo "========================================"
echo ""

# 显示训练参数
echo "v2配置参数:"
python3 -c "
import yaml
with open('$PROJECT_ROOT/$CONFIG') as f:
    cfg = yaml.safe_load(f)

# 读取base_config来获取device配置
with open('$PROJECT_ROOT/$BASE_CONFIG') as f:
    base_cfg = yaml.safe_load(f)

train_cfg = cfg.get('stage3_training', cfg.get('training', {}))
print(f'  Epochs: {train_cfg[\"epochs\"]} (v1: 100)')
print(f'  Batch Size: {train_cfg[\"batch_size\"]}')
print(f'  Grad Accum Steps: {train_cfg.get(\"grad_accum_steps\", 1)}')
print(f'  Learning Rate: {train_cfg[\"optimizer\"][\"lr\"]} (v1: 1.0e-4)')
print(f'  Weight Decay: {train_cfg[\"optimizer\"][\"weight_decay\"]} (v1: 0.01)')
print(f'  Scheduler: {train_cfg[\"scheduler\"][\"type\"]}')
print(f'    T_0: {train_cfg[\"scheduler\"][\"T_0\"]} (v1: 20)')
print(f'    T_mult: {train_cfg[\"scheduler\"][\"T_mult\"]} (v1: 2)')
print(f'    eta_min: {train_cfg[\"scheduler\"][\"eta_min\"]} (v1: 1e-7)')
print(f'  使用EMA: {train_cfg.get(\"use_ema\", False)}')
print(f'  混合精度: {base_cfg.get(\"device\", {}).get(\"mixed_precision\", True)}')
print(f'')
print(f'数据增强:')
aug_cfg = train_cfg.get('augmentation', {})
print(f'  Rotation: ±{aug_cfg.get(\"rotation_degrees\", 5)}° (v1: ±5°)')
print(f'  Scale: {aug_cfg.get(\"scale_range\", [0.95, 1.05])} (v1: [0.95, 1.05])')
print(f'  Aug Probability: {aug_cfg.get(\"augmentation_probability\", 0.8)} (v1: 0.8)')
print(f'')
print(f'Freeze策略:')
freeze_cfg = train_cfg.get('freeze', {})
encoder_trainable = freeze_cfg.get('encoder_trainable_blocks', 0)
print(f'  Encoder (DINOv2): {\"冻结\" if freeze_cfg.get(\"freeze_encoder\", True) else f\"训练最后{encoder_trainable}层\"}')
print(f'  Stage2 (DiT): {\"冻结\" if freeze_cfg.get(\"freeze_stage2\", True) else \"训练\"}')
print(f'  Descriptor Head: {\"冻结\" if freeze_cfg.get(\"freeze_descriptor\", False) else \"训练\"}')
print(f'')
print(f'Early Stopping:')
es_cfg = train_cfg.get('early_stopping', {})
print(f'  Patience: {es_cfg.get(\"patience\", 15)} (v1: 15)')
print(f'  Min Delta: {es_cfg.get(\"min_delta\", 0.5)} (v1: 0.5)')
"
echo ""

# 开始训练
echo "========================================"
echo "开始4卡并行训练 (GPU 0,1,2,3)..."
echo "========================================"
echo ""

# 设置Python路径
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 设置可见GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 创建日志目录
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# 生成日志文件名（带时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/kitti_stage3_train_v2_${TIMESTAMP}.log"

echo "训练日志将保存到: $LOG_FILE"
echo ""

# 使用nohup启动分布式训练，后台运行
# nproc_per_node: 每个节点的进程数（4个GPU）
# master_port: 29504（避免与Stage2的29502、原Stage3的29503冲突）
echo "正在启动训练进程..."
if [ $# -gt 0 ]; then
    nohup torchrun --nproc_per_node=4 --master_port=29504 \
        src/training/trainer.py \
        --config "$CONFIG" \
        --base_config "$BASE_CONFIG" \
        "$@" > "$LOG_FILE" 2>&1 &
else
    nohup torchrun --nproc_per_node=4 --master_port=29504 \
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
echo "  日志文件: $LOG_FILE"
echo "  配置版本: v2 (改进版)"
echo ""
echo "查看训练日志:"
echo "  tail -f $LOG_FILE"
echo ""
echo "查看进程状态:"
echo "  ps aux | grep $TRAIN_PID"
echo ""
echo "停止训练:"
echo "  kill $TRAIN_PID"
echo ""
echo "即使SSH断开，训练也会继续运行"
echo "========================================"

# 保存PID到文件，方便后续管理
PID_FILE="$LOG_DIR/kitti_stage3_train_v2.pid"
echo "$TRAIN_PID" > "$PID_FILE"
echo "PID已保存到: $PID_FILE"
echo ""

# 等待几秒，确保进程启动
sleep 3

# 检查进程是否还在运行
if ps -p $TRAIN_PID > /dev/null 2>&1; then
    echo "✓ 训练进程正在运行 (PID: $TRAIN_PID)"
    echo "  日志文件: $LOG_FILE"
    echo "  查看日志: tail -f $LOG_FILE"
else
    echo "✗ 训练进程启动失败，请检查日志: $LOG_FILE"
    exit 1
fi
