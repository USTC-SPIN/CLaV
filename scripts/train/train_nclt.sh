#!/bin/bash
# NCLT数据集训练启动脚本
# Created: 2025-11-05

# 项目根目录
PROJECT_ROOT="/data/users/cxw/pro/clav"
cd "$PROJECT_ROOT" || exit 1

# 训练配置文件
CONFIG="configs/nclt_stage2_training.yaml"
BASE_CONFIG="configs/base_config.yaml"
# 检查数据文件是否存在
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
        echo "  ✓ 存在: $file"
    fi
done

if [ "$ALL_EXIST" = false ]; then
    echo ""
    echo "错误: 部分数据文件不存在"
    echo "请先运行数据生成脚本:"
    echo "  python preprocess/generate_nclt_bev_tuples_spatial.py"
    echo "  python preprocess/generate_nclt_bev_evaluation_sets_spatial.py"
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

echo "使用配置文件: $CONFIG"
echo ""

# 开始训练
echo "========================================"
echo "开始NCLT描述符训练"
echo "========================================"
echo "训练样本: 34,896"
echo "测试样本: 11,080"
echo "评估: fog, rain, snow (各549个queries)"
echo "========================================"
echo ""

# 如果有命令行参数，传递给训练脚本
if [ $# -gt 0 ]; then
    python src/training/trainer.py --config "$CONFIG" --base_config "$BASE_CONFIG" "$@"
else
    python src/training/trainer.py --config "$CONFIG" --base_config "$BASE_CONFIG"
fi

echo ""
echo "训练完成"
