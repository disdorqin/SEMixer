#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SEMixer 实验结果收集脚本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = ROOT_DIR / "repo" / "LongTermTSF_SEMixer"
DEFAULT_OUTPUT_FILE = ROOT_DIR / "实验结果汇总.md"


def collect_results(base_dir: str | Path = DEFAULT_BASE_DIR):
    """收集所有实验结果，按验证集最佳 epoch 对应的测试集 MSE/MAE 汇总。"""

    datasets = [
        'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather',
        'electricity', 'traffic', 'exchange_rate', 'solar_AL', 'national_illness'
    ]
    pred_lens = [96, 192, 336, 720]
    seeds = [0, 1, 2, 3, 4]

    base_dir = Path(base_dir)
    results = {}

    print("=" * 60)
    print("SEMixer 实验结果收集")
    print(f"结果目录：{base_dir}")
    print("=" * 60)

    for dataset in datasets:
        results[dataset] = {}
        dataset_dir = base_dir / dataset

        if not dataset_dir.exists():
            print(f"[!] {dataset}: 未找到结果目录")
            continue

        for pred_len in pred_lens:
            key = f"pred_{pred_len}"
            results[dataset][key] = {'mse': [], 'mae': []}

            mse_list = []
            mae_list = []

            for seed in seeds:
                seed_dir = dataset_dir / f"random_seed_{seed}"
                if not seed_dir.exists():
                    continue

                pattern = f"*_SeqLen*_PredLen{pred_len}_*"
                result_dirs = list(seed_dir.glob(pattern))

                for result_dir in result_dirs:
                    test_loss_file = result_dir / "record_all_loss_test.json"
                    val_loss_file = result_dir / "record_all_loss_val.json"

                    if not test_loss_file.exists() or not val_loss_file.exists():
                        continue

                    with open(test_loss_file, 'r', encoding='utf-8') as file:
                        test_data = json.load(file)

                    with open(val_loss_file, 'r', encoding='utf-8') as file:
                        val_data = json.load(file)

                    best_mse = float('inf')
                    best_epoch = None

                    for epoch, metrics in val_data.items():
                        if isinstance(metrics, dict) and 'mse' in metrics and metrics['mse'] < best_mse:
                            best_mse = metrics['mse']
                            best_epoch = str(epoch)

                    if best_epoch is None:
                        continue

                    test_metrics = test_data.get(best_epoch)
                    if not isinstance(test_metrics, dict):
                        continue

                    test_mse = test_metrics.get('mse')
                    test_mae = test_metrics.get('mae')
                    if test_mse is None or test_mae is None:
                        continue

                    mse_list.append(test_mse)
                    mae_list.append(test_mae)
                    print(
                        f"[OK] {dataset} | pred_len={pred_len} | seed={seed} | "
                        f"MSE={test_mse:.4f} | MAE={test_mae:.4f}"
                    )

            results[dataset][key]['mse'] = mse_list
            results[dataset][key]['mae'] = mae_list

    return results


def generate_report(results, output_file: str | Path = DEFAULT_OUTPUT_FILE):
    """生成实验结果报告。"""

    report = []
    report.append("# SEMixer 实验结果汇总报告")
    report.append("")
    report.append("## 1. 各数据集结果详情")
    report.append("")

    datasets = [
        'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather',
        'electricity', 'traffic', 'exchange_rate', 'solar_AL', 'national_illness'
    ]
    pred_lens = [96, 192, 336, 720]

    for dataset in datasets:
        if dataset not in results:
            continue

        report.append(f"### {dataset}")
        report.append("")
        report.append("| 预测长度 | 种子 | MSE | MAE |")
        report.append("|----------|------|-----|-----|")

        for pred_len in pred_lens:
            key = f"pred_{pred_len}"
            if key in results[dataset]:
                mse_list = results[dataset][key]['mse']
                mae_list = results[dataset][key]['mae']

                for index, (mse, mae) in enumerate(zip(mse_list, mae_list)):
                    report.append(f"| {pred_len} | {index} | {mse:.4f} | {mae:.4f} |")

        report.append("")
        report.append("**平均结果：**")
        report.append("")
        report.append("| 预测长度 | 平均 MSE | 平均 MAE |")
        report.append("|----------|----------|----------|")

        for pred_len in pred_lens:
            key = f"pred_{pred_len}"
            if key in results[dataset] and len(results[dataset][key]['mse']) > 0:
                avg_mse = np.mean(results[dataset][key]['mse'])
                avg_mae = np.mean(results[dataset][key]['mae'])
                report.append(f"| {pred_len} | {avg_mse:.4f} | {avg_mae:.4f} |")

        all_mse = []
        all_mae = []
        for pred_len in pred_lens:
            key = f"pred_{pred_len}"
            if key in results[dataset]:
                all_mse.extend(results[dataset][key]['mse'])
                all_mae.extend(results[dataset][key]['mae'])

        if all_mse:
            report.append(f"| **总平均** | **{np.mean(all_mse):.4f}** | **{np.mean(all_mae):.4f}** |")

        report.append("")

    report.append("## 2. 汇总对比表（与论文结果对比）")
    report.append("")
    report.append("| 数据集 | 复现 MSE | 复现 MAE | 论文 MSE | 论文 MAE | 差异 |")
    report.append("|--------|----------|----------|----------|----------|------|")

    paper_results = {
        'ETTh1': (0.400, 0.418),
        'ETTh2': (0.331, 0.382),
        'ETTm1': (0.342, 0.375),
        'ETTm2': (0.241, 0.312),
        'weather': (0.216, 0.258),
        'electricity': (0.154, 0.249),
        'traffic': (0.388, 0.268),
        'exchange_rate': (0.344, 0.397),
        'solar_AL': (0.184, 0.245),
        'national_illness': (2.385, 1.080),
    }

    for dataset in datasets:
        if dataset not in results:
            continue

        all_mse = []
        all_mae = []
        for pred_len in pred_lens:
            key = f"pred_{pred_len}"
            if key in results[dataset]:
                all_mse.extend(results[dataset][key]['mse'])
                all_mae.extend(results[dataset][key]['mae'])

        if all_mse and dataset in paper_results:
            exp_mse = np.mean(all_mse)
            exp_mae = np.mean(all_mae)
            paper_mse, paper_mae = paper_results[dataset]
            mse_diff = abs(exp_mse - paper_mse)
            report.append(
                f"| {dataset} | {exp_mse:.4f} | {exp_mae:.4f} | {paper_mse:.3f} | {paper_mae:.3f} | {mse_diff:.4f} |"
            )

    report.append("")
    report.append("---")
    report.append("")
    report.append("*注：论文结果来自论文 Table 1*")

    output_file = Path(output_file)
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(report))

    print(f"\n[OK] 报告已生成：{output_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='SEMixer 实验结果收集脚本')
    parser.add_argument(
        '--base-dir',
        default=str(DEFAULT_BASE_DIR),
        help='实验结果根目录',
    )
    parser.add_argument(
        '--output-file',
        default=str(DEFAULT_OUTPUT_FILE),
        help='汇总报告输出路径',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    print("开始收集实验结果...")
    results = collect_results(args.base_dir)

    print("\n生成汇总报告...")
    generate_report(results, args.output_file)

    print("\n" + "=" * 60)
    print("结果收集完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()