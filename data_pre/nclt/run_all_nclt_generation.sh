#!/bin/bash
# NCLT数据集生成脚本 - 批量运行所有生成脚本
# 创建时间: 2025-11-06 00:00

set -e  # 遇到错误立即退出

# 默认配置
DATASET_ROOT="${DATASET_ROOT:-/home/cxw/pro/best}"
IND_NN_R="${IND_NN_R:-10}"      # 正样本距离阈值(米)
IND_R_R="${IND_R_R:-50}"        # 负样本距离阈值(米)
VAL_RATIO="${VAL_RATIO:-0.15}"  # 验证集比例

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 打印配置信息
echo "========================================"
echo "NCLT数据集批量生成"
echo "========================================"
echo "数据集根目录: ${DATASET_ROOT}"
echo "正样本阈值: ${IND_NN_R}m"
echo "负样本阈值: ${IND_R_R}m"
echo "验证集比例: ${VAL_RATIO}"
echo "========================================"
echo ""

# 1. 生成训练tuples（空间分离）
echo "[1/3] 生成训练tuples..."
python3 "${SCRIPT_DIR}/generate_nclt_bev_tuples_spatial.py" \
    --dataset_root "${DATASET_ROOT}" \
    --ind_nn_r ${IND_NN_R} \
    --ind_r_r ${IND_R_R}

if [ $? -ne 0 ]; then
    echo "错误: 训练tuples生成失败"
    exit 1
fi

echo ""
echo "✓ 训练tuples生成完成"
echo ""

# 2. 生成去噪配对数据
echo "[2/3] 生成去噪配对数据..."
python3 "${SCRIPT_DIR}/generate_nclt_denoising_pairs_spatial.py" \
    --dataset_root "${DATASET_ROOT}" \
    --val_ratio ${VAL_RATIO}

if [ $? -ne 0 ]; then
    echo "错误: 去噪配对数据生成失败"
    exit 1
fi

echo ""
echo "✓ 去噪配对数据生成完成"
echo ""

# 3. 生成评估文件
echo "[3/3] 生成评估文件..."
python3 "${SCRIPT_DIR}/generate_nclt_bev_evaluation_sets_spatial.py" \
    --dataset_root "${DATASET_ROOT}"

if [ $? -ne 0 ]; then
    echo "错误: 评估文件生成失败"
    exit 1
fi

echo ""
echo "✓ 评估文件生成完成"
echo ""

# 完成
echo "========================================"
echo "所有数据集生成完成！"
echo "========================================"
echo "输出位置: ${DATASET_ROOT}/data/"
echo ""
echo "生成的文件:"
echo "  - nclt_bev_training_queries.pickle    (训练tuples)"
echo "  - nclt_bev_test_queries.pickle        (测试tuples)"
echo "  - nclt_denoising_tuples.pkl           (去噪配对)"
echo "  - nclt_bev_evaluation_database.pickle (评估database)"
echo "  - nclt_bev_evaluation_query.pickle    (评估query)"
echo "  - nclt_bev_{fog,rain,snow}_evaluation_*.pickle (各天气评估文件)"
echo ""
echo "使用方法:"
echo "  默认运行: ./run_all_nclt_generation.sh"
echo "  自定义路径: DATASET_ROOT=/path/to/data ./run_all_nclt_generation.sh"
echo "  自定义参数: IND_NN_R=15 IND_R_R=60 ./run_all_nclt_generation.sh"
echo "========================================"
