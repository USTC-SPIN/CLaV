#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KITTI数据集一键生成脚本（Python版本）
使用UTM坐标系统（单位：米）

创建时间: 2025-11-06
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime


class Colors:
    """终端颜色定义"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def log_message(message, color=None):
    """打印日志消息"""
    if color:
        print(f"{color}{message}{Colors.NC}")
    else:
        print(message)


def log_success(message):
    """打印成功消息"""
    log_message(f"✓ {message}", Colors.GREEN)


def log_error(message):
    """打印错误消息"""
    log_message(f"✗ {message}", Colors.RED)


def log_warning(message):
    """打印警告消息"""
    log_message(f"⚠ {message}", Colors.YELLOW)


def log_info(message):
    """打印信息消息"""
    log_message(f"ℹ {message}", Colors.BLUE)


def run_script(script_path, args=None, description=""):
    """
    运行Python脚本

    参数:
    - script_path: 脚本路径
    - args: 命令行参数列表
    - description: 任务描述

    返回:
    - success: 是否成功
    - duration: 耗时（秒）
    """
    if args is None:
        args = []

    start_time = time.time()
    cmd = [sys.executable, str(script_path)] + args

    log_info(f"运行: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # 打印输出的最后几行
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines[-5:]:  # 只显示最后5行
                print(f"  {line}")

        duration = time.time() - start_time
        log_success(f"{description}完成 (耗时: {duration:.1f}秒)")
        return True, duration

    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        log_error(f"{description}失败 (耗时: {duration:.1f}秒)")
        if e.stderr:
            print(f"错误信息:\n{e.stderr}")
        return False, duration

    except FileNotFoundError:
        log_error(f"脚本不存在: {script_path}")
        return False, 0


def check_dataset(dataset_root):
    """检查数据集是否存在"""
    dataset_path = Path(dataset_root) / "kitti-bev-skip"
    if not dataset_path.exists():
        log_error(f"数据集目录不存在: {dataset_path}")
        return False

    # 检查是否有数据
    subdirs = list(dataset_path.glob("*-*-*-*-*"))
    if len(subdirs) == 0:
        log_error(f"数据集目录为空: {dataset_path}")
        return False

    log_success(f"找到 {len(subdirs)} 个数据目录")
    return True


def check_output_files(output_dir):
    """检查生成的输出文件"""
    output_path = Path(output_dir)

    expected_files = [
        # 训练数据
        "kitti_bev_training_queries.pickle",
        "kitti_bev_test_queries.pickle",
        # 去噪数据
        "kitti_denoising_tuples.pkl",
        # 评估数据
        "kitti_bev_evaluation_database.pickle",
        "kitti_bev_evaluation_query.pickle",
        "kitti_bev_fog_evaluation_database.pickle",
        "kitti_bev_fog_evaluation_query.pickle",
        "kitti_bev_rain_evaluation_database.pickle",
        "kitti_bev_rain_evaluation_query.pickle",
        "kitti_bev_snow_evaluation_database.pickle",
        "kitti_bev_snow_evaluation_query.pickle",
    ]

    log_message("\n生成的文件:")
    found_count = 0
    for filename in expected_files:
        filepath = output_path / filename
        if filepath.exists():
            size = filepath.stat().st_size / (1024 * 1024)  # MB
            log_success(f"  {filename} ({size:.1f} MB)")
            found_count += 1
        else:
            log_warning(f"  {filename} (未找到)")

    return found_count, len(expected_files)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='KITTI数据集一键生成脚本（UTM坐标版本）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--dataset_root',
        type=str,
        default='/data/users/cxw/pro/clav',
        help='数据集根目录'
    )

    parser.add_argument(
        '--ind_nn_r',
        type=float,
        default=10.0,
        help='正样本距离阈值（米）'
    )

    parser.add_argument(
        '--ind_r_r',
        type=float,
        default=50.0,
        help='负样本距离阈值（米）'
    )

    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='验证集比例'
    )

    parser.add_argument(
        '--sequences',
        nargs='+',
        default=['01-10-03-42', '02-10-03-14', '04-09-30-16',
                 '08-09-30-28', '09-09-30-33', '10-09-30-34'],
        help='要处理的序列列表'
    )

    parser.add_argument(
        '--skip_tuples',
        action='store_true',
        help='跳过训练tuples生成'
    )

    parser.add_argument(
        '--skip_denoising',
        action='store_true',
        help='跳过去噪配对数据生成'
    )

    parser.add_argument(
        '--skip_evaluation',
        action='store_true',
        help='跳过评估文件生成'
    )

    parser.add_argument(
        '--verify',
        action='store_true',
        help='运行数据验证'
    )

    args = parser.parse_args()

    # 获取脚本目录
    script_dir = Path(__file__).parent.absolute()

    # 创建输出目录
    output_dir = Path(args.dataset_root) / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打印配置
    log_message("=" * 70)
    log_message("KITTI数据集批量生成（UTM坐标版本）")
    log_message("=" * 70)
    log_message(f"数据集根目录: {args.dataset_root}")
    log_message(f"数据集位置: {args.dataset_root}/kitti-bev-skip")
    log_message(f"输出目录: {output_dir}")
    log_message(f"正样本阈值: {args.ind_nn_r}m")
    log_message(f"负样本阈值: {args.ind_r_r}m")
    log_message(f"验证集比例: {args.val_ratio}")
    log_message(f"处理序列: {args.sequences}")
    log_message("=" * 70)

    # 检查数据集
    log_message("\n检查数据集...")
    if not check_dataset(args.dataset_root):
        return 1

    total_duration = 0
    success_count = 0
    total_count = 0

    # 1. 生成训练tuples
    if not args.skip_tuples:
        log_message("\n[1/3] 生成训练tuples...")
        script_path = script_dir / "generate_kitti_bev_tuples_spatial.py"

        if script_path.exists():
            success, duration = run_script(
                script_path,
                [
                    '--dataset_root', args.dataset_root,
                    '--ind_nn_r', str(args.ind_nn_r),
                    '--ind_r_r', str(args.ind_r_r)
                ],
                "训练tuples生成"
            )
            total_duration += duration
            total_count += 1
            if success:
                success_count += 1
        else:
            log_warning(f"脚本不存在，跳过: {script_path}")

    # 2. 生成去噪配对数据
    if not args.skip_denoising:
        log_message("\n[2/3] 生成去噪配对数据...")
        script_path = script_dir / "generate_kitti_denoising_pairs_spatial.py"

        if script_path.exists():
            success, duration = run_script(
                script_path,
                [
                    '--dataset_root', args.dataset_root,
                    '--sequences'] + args.sequences + [
                    '--val_ratio', str(args.val_ratio)
                ],
                "去噪配对数据生成"
            )
            total_duration += duration
            total_count += 1
            if success:
                success_count += 1
        else:
            log_warning(f"脚本不存在，跳过: {script_path}")

    # 3. 生成评估文件
    if not args.skip_evaluation:
        log_message("\n[3/3] 生成评估文件...")
        script_path = script_dir / "generate_kitti_bev_evaluation_sets_spatial.py"

        if script_path.exists():
            success, duration = run_script(
                script_path,
                ['--dataset_root', args.dataset_root],
                "评估文件生成"
            )
            total_duration += duration
            total_count += 1
            if success:
                success_count += 1
        else:
            log_warning(f"脚本不存在，跳过: {script_path}")

    # 检查输出文件
    found, expected = check_output_files(output_dir)

    # 可选：运行验证
    if args.verify:
        log_message("\n运行数据验证...")
        verify_script = script_dir / "verify_denoising_data.py"
        denoising_file = output_dir / "kitti_denoising_tuples.pkl"

        if verify_script.exists() and denoising_file.exists():
            run_script(
                verify_script,
                [str(denoising_file)],
                "数据验证"
            )
        else:
            log_warning("验证脚本或数据文件不存在，跳过验证")

    # 总结
    log_message("\n" + "=" * 70)
    if success_count == total_count:
        log_success("所有数据集生成完成！")
    else:
        log_warning(f"部分生成任务失败 ({success_count}/{total_count})")

    log_message("=" * 70)
    log_message(f"总耗时: {total_duration:.1f}秒 ({total_duration/60:.1f}分钟)")
    log_message(f"输出文件: {found}/{expected}")
    log_message(f"输出位置: {output_dir}/")
    log_message("\n数据特性:")
    log_message("  - 坐标系统: UTM（单位：米）")
    log_message("  - 空间划分: 累积距离法（前20%测试，后80%训练）")
    log_message("  - 天气条件: orin(清晰), fog, rain, snow")

    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())