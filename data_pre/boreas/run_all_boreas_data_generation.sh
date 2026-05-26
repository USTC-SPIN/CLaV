#!/bin/bash
# Boreas数据集完整数据生成脚本
# Created: 2025-11-07 22:45
# Purpose: 按顺序运行所有Boreas数据预处理脚本

PROJECT_ROOT="/data/users/cxw/pro/clav"
cd "$PROJECT_ROOT" || exit 1

echo "========================================"
echo "Boreas数据集完整数据生成"
echo "========================================"
echo ""
echo "项目根目录: $PROJECT_ROOT"
echo ""

# 设置Python路径
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 默认参数
DATASET_ROOT="$PROJECT_ROOT"
DATA_DIR="$PROJECT_ROOT/data"

# 确保data目录存在
mkdir -p "$DATA_DIR"

# 检查是否需要跳过某些步骤
SKIP_TUPLES=false
SKIP_DENOISING=false
SKIP_EVALUATION=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-tuples)
            SKIP_TUPLES=true
            shift
            ;;
        --skip-denoising)
            SKIP_DENOISING=true
            shift
            ;;
        --skip-evaluation)
            SKIP_EVALUATION=true
            shift
            ;;
        --help)
            echo "用法: bash $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --skip-tuples       跳过训练元组生成"
            echo "  --skip-denoising    跳过去噪配对数据生成"
            echo "  --skip-evaluation   跳过评估数据集生成"
            echo "  --help              显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  bash $0                    # 运行所有步骤"
            echo "  bash $0 --skip-tuples      # 跳过tuples，只生成denoising和evaluation"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# ========================================
# 步骤1: 生成训练元组（用于地理定位训练）
# ========================================
if [ "$SKIP_TUPLES" = false ]; then
    echo "========================================"
    echo "步骤 1/3: 生成训练元组（TrainingTuples）"
    echo "========================================"
    echo ""
    echo "功能: 为地理定位训练准备训练数据"
    echo "输出: data/boreas_bev_training_queries_spatial.pickle"
    echo "      data/boreas_bev_test_queries_spatial.pickle"
    echo ""

    python data_pre/boreas/generate_boreas_bev_tuples_spatial.py \
        --dataset_root "$DATASET_ROOT" \
        --ind_nn_r 10 \
        --ind_r_r 50 \
        --test_ratio 0.2

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ 步骤1完成: 训练元组生成成功"
        echo ""

        # 显示生成的文件信息
        echo "生成的文件:"
        if [ -f "$DATA_DIR/boreas_bev_training_queries_spatial.pickle" ]; then
            size=$(ls -lh "$DATA_DIR/boreas_bev_training_queries_spatial.pickle" | awk '{print $5}')
            echo "  训练集: $size"
        fi
        if [ -f "$DATA_DIR/boreas_bev_test_queries_spatial.pickle" ]; then
            size=$(ls -lh "$DATA_DIR/boreas_bev_test_queries_spatial.pickle" | awk '{print $5}')
            echo "  测试集: $size"
        fi
    else
        echo ""
        echo "✗ 步骤1失败: 训练元组生成失败"
        echo "请检查错误信息并重试"
        exit 1
    fi
else
    echo "跳过步骤1: 训练元组生成"
fi

echo ""
sleep 2

# ========================================
# 步骤2: 生成去噪配对数据（用于Stage2扩散模型训练）
# ========================================
if [ "$SKIP_DENOISING" = false ]; then
    echo "========================================"
    echo "步骤 2/3: 生成去噪配对数据"
    echo "========================================"
    echo ""
    echo "功能: 为Stage2扩散模型训练准备noisy-clean配对数据"
    echo "输出: data/boreas_bev_denoising_pairs_spatial.pickle"
    echo ""

    python data_pre/boreas/generate_boreas_bev_denoising_pairs_spatial.py \
        --dataset_root "$DATASET_ROOT" \
        --output_file "$DATA_DIR/boreas_bev_denoising_pairs_spatial.pickle" \
        --val_ratio 0.15 \
        --distance_threshold 5.0

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ 步骤2完成: 去噪配对数据生成成功"
        echo ""

        # 显示生成的文件信息
        if [ -f "$DATA_DIR/boreas_bev_denoising_pairs_spatial.pickle" ]; then
            size=$(ls -lh "$DATA_DIR/boreas_bev_denoising_pairs_spatial.pickle" | awk '{print $5}')
            echo "文件大小: $size"
        fi
    else
        echo ""
        echo "✗ 步骤2失败: 去噪配对数据生成失败"
        echo "请检查错误信息并重试"
        exit 1
    fi
else
    echo "跳过步骤2: 去噪配对数据生成"
fi

echo ""
sleep 2

# ========================================
# 步骤3: 生成评估数据集
# ========================================
if [ "$SKIP_EVALUATION" = false ]; then
    echo "========================================"
    echo "步骤 3/3: 生成评估数据集"
    echo "========================================"
    echo ""
    echo "功能: 为模型评估准备database和query文件"
    echo "输出: data/kitti_bev_*_evaluation_*.pickle"
    echo ""

    python data_pre/boreas/generate_boreas_bev_evaluation_sets_spatial.py \
        --dataset_root "$DATASET_ROOT" \
        --test_ratio 0.2

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ 步骤3完成: 评估数据集生成成功"
        echo ""

        # 显示生成的文件信息
        echo "生成的评估文件:"
        ls -lh "$DATA_DIR"/kitti_bev_*_evaluation_*.pickle 2>/dev/null || echo "未找到评估文件"
    else
        echo ""
        echo "✗ 步骤3失败: 评估数据集生成失败"
        echo "请检查错误信息并重试"
        exit 1
    fi
else
    echo "跳过步骤3: 评估数据集生成"
fi

echo ""
echo "========================================"
echo "数据生成完成 ✓"
echo "========================================"
echo ""
echo "生成的文件:"
echo "----------------------------------------"
echo ""
echo "1. 训练元组 (用于地理定位训练):"
ls -lh "$DATA_DIR"/boreas_bev_*_queries_spatial.pickle 2>/dev/null || echo "  未找到"
echo ""
echo "2. 去噪配对数据 (用于Stage2训练):"
ls -lh "$DATA_DIR"/boreas_bev_denoising_pairs_spatial.pickle 2>/dev/null || echo "  未找到"
echo ""
echo "3. 评估数据集 (用于模型评估):"
ls -lh "$DATA_DIR"/kitti_bev_*_evaluation_*.pickle 2>/dev/null || echo "  未找到"
echo ""
echo "----------------------------------------"
echo ""
echo "后续步骤:"
echo "1. 训练Stage2扩散模型 (去噪):"
echo "   bash scripts/train_boreas_stage2.sh"
echo ""
echo "2. 训练Stage3描述符头 (地理定位):"
echo "   bash scripts/train_boreas_stage3.sh"
echo ""
echo "3. 运行评估:"
echo "   bash scripts/evaluate_boreas.sh"
echo ""
