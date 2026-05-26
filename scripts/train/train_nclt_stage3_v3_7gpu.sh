#!/bin/bash
# NCLT数据集Stage3训练启动脚本（7卡并行）
# Created: 2025-11-14
# Purpose: 使用7卡并行训练NCLT数据集的Stage3描述符头（跨天气地理定位）
# Based on: Boreas v3 successful configuration

# 项目根目录
PROJECT_ROOT="/data/users/cxw/pro/clav"
cd "$PROJECT_ROOT" || exit 1

# 训练配置文件（默认使用v3优化配置）
CONFIG="${1:-configs/nclt/nclt_stage3_training_v3.yaml}"
BASE_CONFIG="configs/nclt/base_config.yaml"

# 检查数据文件是否存在
echo "========================================"
echo "NCLT Stage3 描述符训练准备 (7卡并行)"
echo "========================================"
echo ""
echo "检查数据文件..."

DATA_FILES=(
    "data/nclt_bev_training_queries.pickle"
    "data/nclt_bev_fog_evaluation_database.pickle"
    "data/nclt_bev_fog_evaluation_query.pickle"
    "data/nclt_bev_rain_evaluation_database.pickle"
    "data/nclt_bev_rain_evaluation_query.pickle"
    "data/nclt_bev_snow_evaluation_database.pickle"
    "data/nclt_bev_snow_evaluation_query.pickle"
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
    echo "  python data_pre/nclt/generate_nclt_data.py"
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
        echo "训练Stage2: bash scripts/train_nclt_stage2.sh"
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
echo "NCLT Stage3 训练信息"
echo "========================================"
python3 -c "
import pickle
train_data = pickle.load(open('$PROJECT_ROOT/data/nclt_bev_training_queries.pickle', 'rb'))
fog_db = pickle.load(open('$PROJECT_ROOT/data/nclt_bev_fog_evaluation_database.pickle', 'rb'))
fog_q = pickle.load(open('$PROJECT_ROOT/data/nclt_bev_fog_evaluation_query.pickle', 'rb'))
rain_db = pickle.load(open('$PROJECT_ROOT/data/nclt_bev_rain_evaluation_database.pickle', 'rb'))
rain_q = pickle.load(open('$PROJECT_ROOT/data/nclt_bev_rain_evaluation_query.pickle', 'rb'))
snow_db = pickle.load(open('$PROJECT_ROOT/data/nclt_bev_snow_evaluation_database.pickle', 'rb'))
snow_q = pickle.load(open('$PROJECT_ROOT/data/nclt_bev_snow_evaluation_query.pickle', 'rb'))

print(f'训练样本数: {len(train_data):,}')
print(f'评估数据库: {len(fog_db):,} 样本')
print(f'  Fog查询: {len(fog_q):,}')
print(f'  Rain查询: {len(rain_q):,}')
print(f'  Snow查询: {len(snow_q):,}')
print(f'  总查询数: {len(fog_q) + len(rain_q) + len(snow_q):,}')
print(f'')
print(f'任务: 跨天气地理定位（描述符学习）')
print(f'天气条件: Origin (database) ↔ Fog/Rain/Snow (query)')
print(f'数据来源: NCLT真实采集数据')
print(f'测试区域: 3个独立区域 (150m×150m each)')
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

train_cfg = cfg.get('stage3_training', {})
print(f'  Epochs: {train_cfg[\"epochs\"]}')
print(f'  Batch Size: {train_cfg[\"batch_size\"]} (per GPU)')
print(f'  Grad Accum Steps: {train_cfg.get(\"grad_accum_steps\", 1)}')
total_bs = train_cfg[\"batch_size\"] * train_cfg.get(\"grad_accum_steps\", 1) * 7
print(f'  Effective Batch Size: {total_bs} (7 GPUs)')
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
print(f'')
print(f'Early Stopping:')
es_cfg = train_cfg.get('early_stopping', {})
if es_cfg.get('enabled', False):
    patience = es_cfg.get('patience', 30)
    interval = train_cfg.get('eval_interval', 5)
    actual_epochs = patience * interval
    print(f'  启用: 是')
    print(f'  Patience: {patience} (验证次数)')
    print(f'  Eval Interval: {interval} (epochs)')
    print(f'  实际等待: {actual_epochs} epochs')
    print(f'  监控指标: {es_cfg.get(\"monitor\", \"eval_avg_recall@1\")}')
    print(f'  Min Delta: {es_cfg.get(\"min_delta\", 0.3)}%')
else:
    print(f'  启用: 否')
"
echo ""

# 开始训练
echo "========================================"
echo "开始7卡并行训练 (GPU 0-6)..."
echo "========================================"
echo ""

# 设置Python路径
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 设置可见GPU（7卡）
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 创建日志目录
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# 生成日志文件名（带时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/nclt_stage3_v3_7gpu_${TIMESTAMP}.log"

echo "训练配置:"
echo "  并行方式: DistributedDataParallel (DDP)"
echo "  GPU数量: 7"
echo "  Master Port: 29503"
echo "  日志文件: $LOG_FILE"
echo ""

# 使用nohup启动分布式训练，后台运行
# nproc_per_node: 每个节点的进程数（4个GPU）
echo "正在启动训练进程..."
if [ $# -gt 1 ]; then
    # 如果有额外参数，传递给训练脚本（跳过第一个参数，因为是config）
    nohup torchrun --nproc_per_node=4 --master_port=29503 \
        src/training/trainer.py \
        --config "$CONFIG" \
        --base_config "$BASE_CONFIG" \
        "${@:2}" > "$LOG_FILE" 2>&1 &
else
    nohup torchrun --nproc_per_node=4 --master_port=29503 \
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
echo "  主进程PID: $TRAIN_PID"
echo "  日志文件: $LOG_FILE"
echo "  GPU配置: 0-6 (7卡)"
echo ""
echo "查看训练日志:"
echo "  tail -f $LOG_FILE"
echo ""
echo "查看验证结果:"
echo "  tail -f $LOG_FILE | grep -E \"Validation|Early Stopping\""
echo ""
echo "查看GPU使用情况:"
echo "  nvidia-smi"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "查看进程状态:"
echo "  ps aux | grep trainer.py"
echo ""
echo "停止训练:"
echo "  pkill -f trainer.py"
echo "  # 或者"
echo "  kill $TRAIN_PID"
echo ""
echo "即使SSH断开，训练也会继续运行"
echo "========================================"

# 等待几秒，确保进程启动
sleep 5

# 检查进程是否还在运行
if ps -p $TRAIN_PID > /dev/null 2>&1; then
    echo "✓ 训练进程正在运行 (PID: $TRAIN_PID)"
    echo "  日志文件: $LOG_FILE"
    echo "  查看日志: tail -f $LOG_FILE"
else
    echo "✗ 训练进程启动失败，请检查日志: $LOG_FILE"
    exit 1
fi
