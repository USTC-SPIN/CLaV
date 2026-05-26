#!/bin/bash
# KITTI数据集生成脚本 - 批量运行所有生成脚本
# 使用UTM坐标系统（单位：米）
# 创建时间: 2025-11-06

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认配置
DATASET_ROOT="${DATASET_ROOT:-/data/users/cxw/pro/clav}"
IND_NN_R="${IND_NN_R:-10}"      # 正样本距离阈值(米) - 与描述符训练保持一致
IND_R_R="${IND_R_R:-50}"        # 负样本距离阈值(米) - 与描述符训练保持一致
VAL_RATIO="${VAL_RATIO:-0.15}"  # 验证集比例
SEQUENCES="${SEQUENCES:-01-10-03-42 02-10-03-14 04-09-30-16 08-09-30-28 09-09-30-33 10-09-30-34}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 创建输出目录
OUTPUT_DIR="${DATASET_ROOT}/data"
mkdir -p "${OUTPUT_DIR}"

# 创建日志文件
LOG_FILE="${OUTPUT_DIR}/kitti_generation_$(date +%Y%m%d_%H%M%S).log"

# 日志函数
log_message() {
    echo "$1" | tee -a "${LOG_FILE}"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}✗ $1${NC}" | tee -a "${LOG_FILE}"
}

log_warning() {
    echo -e "${YELLOW}⚠ $1${NC}" | tee -a "${LOG_FILE}"
}

# 打印配置信息
log_message "========================================"
log_message "KITTI数据集批量生成（UTM坐标版本）"
log_message "========================================"
log_message "数据集根目录: ${DATASET_ROOT}"
log_message "数据集位置: ${DATASET_ROOT}/kitti-bev-skip"
log_message "输出目录: ${OUTPUT_DIR}"
log_message "正样本阈值: ${IND_NN_R}m"
log_message "负样本阈值: ${IND_R_R}m"
log_message "验证集比例: ${VAL_RATIO}"
log_message "处理序列: ${SEQUENCES}"
log_message "日志文件: ${LOG_FILE}"
log_message "========================================"
log_message ""

# 检查数据集是否存在
if [ ! -d "${DATASET_ROOT}/kitti-bev-skip" ]; then
    log_error "错误: 数据集目录不存在 - ${DATASET_ROOT}/kitti-bev-skip"
    exit 1
fi

# 记录开始时间
START_TIME=$(date +%s)

# 1. 生成训练tuples（空间分离）
log_message "[1/3] 生成训练tuples..."
log_message "命令: python3 generate_kitti_bev_tuples_spatial.py --dataset_root ${DATASET_ROOT} --ind_nn_r ${IND_NN_R} --ind_r_r ${IND_R_R}"

if python3 "${SCRIPT_DIR}/generate_kitti_bev_tuples_spatial.py" \
    --dataset_root "${DATASET_ROOT}" \
    --ind_nn_r ${IND_NN_R} \
    --ind_r_r ${IND_R_R} >> "${LOG_FILE}" 2>&1; then
    log_success "训练tuples生成完成"

    # 检查输出文件
    if [ -f "${OUTPUT_DIR}/kitti_bev_training_queries.pickle" ]; then
        SIZE=$(du -h "${OUTPUT_DIR}/kitti_bev_training_queries.pickle" | cut -f1)
        log_message "  - kitti_bev_training_queries.pickle (${SIZE})"
    fi
    if [ -f "${OUTPUT_DIR}/kitti_bev_test_queries.pickle" ]; then
        SIZE=$(du -h "${OUTPUT_DIR}/kitti_bev_test_queries.pickle" | cut -f1)
        log_message "  - kitti_bev_test_queries.pickle (${SIZE})"
    fi
else
    log_error "训练tuples生成失败"
    log_message "请检查日志文件: ${LOG_FILE}"
    exit 1
fi

log_message ""

# 2. 生成去噪配对数据
log_message "[2/3] 生成去噪配对数据..."
log_message "命令: python3 generate_kitti_denoising_pairs_spatial.py --dataset_root ${DATASET_ROOT} --val_ratio ${VAL_RATIO}"

# 将序列转换为数组
SEQUENCES_ARRAY=($SEQUENCES)

if python3 "${SCRIPT_DIR}/generate_kitti_denoising_pairs_spatial.py" \
    --dataset_root "${DATASET_ROOT}" \
    --sequences ${SEQUENCES_ARRAY[@]} \
    --val_ratio ${VAL_RATIO} >> "${LOG_FILE}" 2>&1; then
    log_success "去噪配对数据生成完成"

    # 检查输出文件
    if [ -f "${OUTPUT_DIR}/kitti_denoising_tuples.pkl" ]; then
        SIZE=$(du -h "${OUTPUT_DIR}/kitti_denoising_tuples.pkl" | cut -f1)
        log_message "  - kitti_denoising_tuples.pkl (${SIZE})"
    fi
else
    log_error "去噪配对数据生成失败"
    log_message "请检查日志文件: ${LOG_FILE}"
    exit 1
fi

log_message ""

# 3. 生成评估文件
log_message "[3/3] 生成评估文件..."
log_message "命令: python3 generate_kitti_bev_evaluation_sets_spatial.py --dataset_root ${DATASET_ROOT}"

if python3 "${SCRIPT_DIR}/generate_kitti_bev_evaluation_sets_spatial.py" \
    --dataset_root "${DATASET_ROOT}" >> "${LOG_FILE}" 2>&1; then
    log_success "评估文件生成完成"

    # 检查输出文件
    for weather in orin fog rain snow; do
        if [ "${weather}" = "orin" ]; then
            DB_FILE="${OUTPUT_DIR}/kitti_bev_evaluation_database.pickle"
            Q_FILE="${OUTPUT_DIR}/kitti_bev_evaluation_query.pickle"
        else
            DB_FILE="${OUTPUT_DIR}/kitti_bev_${weather}_evaluation_database.pickle"
            Q_FILE="${OUTPUT_DIR}/kitti_bev_${weather}_evaluation_query.pickle"
        fi

        if [ -f "${DB_FILE}" ]; then
            SIZE=$(du -h "${DB_FILE}" | cut -f1)
            log_message "  - $(basename ${DB_FILE}) (${SIZE})"
        fi
        if [ -f "${Q_FILE}" ]; then
            SIZE=$(du -h "${Q_FILE}" | cut -f1)
            log_message "  - $(basename ${Q_FILE}) (${SIZE})"
        fi
    done
else
    log_error "评估文件生成失败"
    log_message "请检查日志文件: ${LOG_FILE}"
    exit 1
fi

log_message ""

# 计算总耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

# 完成
log_message "========================================"
log_success "所有数据集生成完成！"
log_message "========================================"
log_message "总耗时: ${MINUTES}分${SECONDS}秒"
log_message "输出位置: ${OUTPUT_DIR}/"
log_message ""
log_message "生成的文件:"
log_message "  训练数据:"
log_message "    - kitti_bev_training_queries.pickle    (训练tuples)"
log_message "    - kitti_bev_test_queries.pickle        (测试tuples)"
log_message "  去噪数据:"
log_message "    - kitti_denoising_tuples.pkl           (去噪配对)"
log_message "  评估数据:"
log_message "    - kitti_bev_evaluation_database.pickle (评估database)"
log_message "    - kitti_bev_evaluation_query.pickle    (评估query)"
log_message "    - kitti_bev_{fog,rain,snow}_evaluation_*.pickle (各天气评估文件)"
log_message ""
log_message "数据特性:"
log_message "  - 坐标系统: UTM（单位：米）"
log_message "  - 空间划分: 累积距离法（前20%测试，后80%训练）"
log_message "  - 天气条件: orin(清晰), fog, rain, snow"
log_message ""
log_message "使用方法:"
log_message "  默认运行: ./run_all_kitti_generation.sh"
log_message "  自定义路径: DATASET_ROOT=/path/to/data ./run_all_kitti_generation.sh"
log_message "  自定义参数: IND_NN_R=5 IND_R_R=50 ./run_all_kitti_generation.sh"
log_message "  指定序列: SEQUENCES='01-10-03-42 02-10-03-14' ./run_all_kitti_generation.sh"
log_message "========================================"

# 可选：运行验证脚本
if [ -f "${SCRIPT_DIR}/verify_denoising_data.py" ] && [ -f "${OUTPUT_DIR}/kitti_denoising_tuples.pkl" ]; then
    log_message ""
    log_message "运行数据验证..."
    if python3 "${SCRIPT_DIR}/verify_denoising_data.py" "${OUTPUT_DIR}/kitti_denoising_tuples.pkl" >> "${LOG_FILE}" 2>&1; then
        log_success "数据验证通过"
    else
        log_warning "数据验证失败，请检查日志"
    fi
fi

log_message ""
log_message "日志文件保存在: ${LOG_FILE}"