#!/bin/bash
# KITTI数据集Stage2训练启动脚本
# Created: 2025-11-08
# Purpose: 训练KITTI数据集的Stage2扩散去噪模型（Flow Matching）

# 项目根目录
PROJECT_ROOT="/data/users/cxw/pro/clav"
cd "$PROJECT_ROOT" || exit 1

# 训练配置文件
CONFIG="configs/kitti/kitti_stage2_flow_matching.yaml"
BASE_CONFIG="configs/base_config.yaml"

# 检查数据文件是否存在
echo "========================================"
echo "KITTI Stage2 Flow Matching 训练准备"
echo "========================================"
echo ""
echo "检查数据文件..."
DATA_FILES=(
    "data/kitti_denoising_tuples.pkl"
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
echo "  主配置: $CONFIG"
echo "  基础配置: $BASE_CONFIG"
echo ""

# 显示训练信息
echo "========================================"
echo "KITTI Stage2 训练信息"
echo "========================================"
python3 -c "
import pickle
data = pickle.load(open('$PROJECT_ROOT/data/kitti_denoising_tuples.pkl', 'rb'))
print(f'训练样本数: {len(data[\"train\"])}')
print(f'验证样本数: {len(data[\"val\"])}')
print(f'测试样本数: {len(data[\"test\"])}')
print(f'总计: {len(data[\"train\"]) + len(data[\"val\"]) + len(data[\"test\"])}')
print(f'')
print(f'天气条件: Fog, Rain, Snow')
print(f'清洁参考: Orin')
print(f'数据来源: KITTI天气仿真数据')
print(f'配对方法: GPS最近邻匹配')
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

train_cfg = cfg.get('stage2_training', cfg.get('training', {}))
print(f'  Epochs: {train_cfg[\"epochs\"]}')
print(f'  Batch Size: {train_cfg[\"batch_size\"]}')
print(f'  Learning Rate: {train_cfg[\"optimizer\"][\"lr\"]}')
print(f'  Scheduler: {train_cfg[\"scheduler\"][\"type\"]}')
print(f'  使用EMA: {train_cfg[\"use_ema\"]}')
print(f'  混合精度: {base_cfg.get(\"device\", {}).get(\"mixed_precision\", True)}')
print(f'')
print(f'冻结策略:')
print(f'  Encoder (DINOv2): {\"冻结\" if train_cfg[\"freeze\"][\"freeze_encoder\"] else \"训练\"}')
print(f'  Stage2 (DiT): {\"冻结\" if train_cfg[\"freeze\"][\"freeze_stage2\"] else \"训练\"}')
print(f'  Descriptor Head: {\"冻结\" if train_cfg[\"freeze\"][\"freeze_descriptor\"] else \"训练\"}')
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
LOG_FILE="$LOG_DIR/kitti_stage2_train_${TIMESTAMP}.log"

echo "训练日志将保存到: $LOG_FILE"
echo ""

# 使用nohup启动分布式训练，后台运行
# nproc_per_node: 每个节点的进程数（4个GPU）
# master_port: 29502（避免与Boreas Stage2的29500冲突）
echo "正在启动训练进程..."
if [ $# -gt 0 ]; then
    nohup torchrun --nproc_per_node=4 --master_port=29502 \
        src/training/trainer.py \
        --config "$CONFIG" \
        --base_config "$BASE_CONFIG" \
        "$@" > "$LOG_FILE" 2>&1 &
else
    nohup torchrun --nproc_per_node=4 --master_port=29502 \
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
PID_FILE="$LOG_DIR/kitti_stage2_train.pid"
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
