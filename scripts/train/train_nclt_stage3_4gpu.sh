#!/bin/bash
# NCLT联合训练启动脚本 - 4卡并行
# Stage2去噪模型 + 描述符头联合训练
# Created: 2025-11-06

set -e  # 遇到错误立即退出

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

echo -e "${BLUE}========================================"
echo "NCLT Stage3训练 - 4卡并行"
echo "描述符头训练（Descriptor Head）"
echo -e "========================================${NC}"

# 配置文件
CONFIG_FILE="configs/nclt_training.yaml"
BASE_CONFIG="configs/base_config.yaml"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi

if [ ! -f "$BASE_CONFIG" ]; then
    echo -e "${RED}错误: 基础配置文件不存在: $BASE_CONFIG${NC}"
    exit 1
fi

# 检查必要的数据文件
echo "检查数据文件..."

DATA_FILES=(
    "data/nclt_denoising_tuples.pkl"
    "data/nclt_bev_fog_evaluation_database.pickle"
    "data/nclt_bev_fog_evaluation_query.pickle"
    "data/nclt_bev_rain_evaluation_database.pickle"
    "data/nclt_bev_rain_evaluation_query.pickle"
    "data/nclt_bev_snow_evaluation_database.pickle"
    "data/nclt_bev_snow_evaluation_query.pickle"
)

ALL_EXIST=true
for file in "${DATA_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} 存在: $file"
    else
        echo -e "  ${RED}✗${NC} 缺失: $file"
        ALL_EXIST=false
    fi
done

if [ "$ALL_EXIST" = false ]; then
    echo -e "${RED}错误: 部分数据文件缺失${NC}"
    exit 1
fi

echo -e "${GREEN}所有数据文件检查通过 ✓${NC}"
echo ""

# 检查去噪配对数据的样本数
echo "检查去噪配对数据..."
python3 - <<EOF
import pickle
with open('data/nclt_denoising_tuples.pkl', 'rb') as f:
    data = pickle.load(f)
print(f"去噪配对样本数: {len(data)}")
EOF

echo ""
echo -e "${BLUE}使用配置文件: $CONFIG_FILE${NC}"
echo ""
# 根据nvidia-smi检查结果修改这里，只使用空闲GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用4张GPU（0-3号卡）
# 如果部分GPU被占用，可以改为: export CUDA_VISIBLE_DEVICES=0,1  # 只用2卡
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1  # 禁用InfiniBand（如果没有IB网络）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 减少内存碎片

# 多卡训练配置
NUM_GPUS=4  # 修改为实际使用的GPU数量（如果只用2卡改为2）
MASTER_PORT=${MASTER_PORT:-29500}

echo -e "${BLUE}========================================"
echo "开始NCLT Stage3训练 - 4卡并行"
echo "======================================${NC}"
echo -e "${YELLOW}训练模式: 描述符头训练（冻结Stage2）${NC}"
echo -e "${YELLOW}数据集: NCLT${NC}"
echo -e "${YELLOW}GPU数量: ${NUM_GPUS}${NC}"
echo -e "${YELLOW}等效Batch Size: $(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); s=c['stage2_training']; print(s['batch_size']*s.get('grad_accum_steps',1)*${NUM_GPUS})")${NC}"
echo -e "${YELLOW}评估条件: fog, rain, snow${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 运行分布式训练 - 使用torchrun
torchrun --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    src/training/trainer.py \
    --config "$CONFIG_FILE" \
    --base_config "$BASE_CONFIG"

echo ""
echo -e "${GREEN}训练完成${NC}"
