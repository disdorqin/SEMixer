#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SEMixer 一键复现入口。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
REPO_DIR = ROOT_DIR / "repo"
RESULT_DIR = REPO_DIR / "LongTermTSF_SEMixer"


def _run_command(command: list[str], cwd: Path) -> None:
    print(f"[RUN] {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd), check=True)


def _run_repo_mode(mode: str) -> None:
    _run_command([sys.executable, "run.py", mode], REPO_DIR)


def _run_custom(dataset: str, pred_len: int, seed: int) -> None:
    code = f"from run import main; main({seed}, {pred_len}, {dataset!r})"
    _run_command([sys.executable, "-c", code], REPO_DIR)


def _collect_results(output_file: Path) -> None:
    _run_command(
        [
            sys.executable,
            str(ROOT_DIR / "collect_results.py"),
            "--base-dir",
            str(RESULT_DIR),
            "--output-file",
            str(output_file),
        ],
        ROOT_DIR,
    )


def _check_environment() -> None:
    print(f"Python: {sys.executable}")
    try:
        import torch

        print(f"Torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
    except Exception as exc:  # pragma: no cover - diagnostic output only
        print(f"Torch check failed: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SEMixer 一键复现工具")
    parser.add_argument(
        "mode",
        nargs="?",
        default="easy",
        choices=["easy", "medium", "hard", "all", "collect", "custom", "check"],
        help="运行模式",
    )
    parser.add_argument(
        "--dataset",
        default="ETTh1",
        help="自定义模式下的数据集名称",
    )
    parser.add_argument(
        "--pred-len",
        type=int,
        default=96,
        help="自定义模式下的预测长度",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="自定义模式下的随机种子",
    )
    parser.add_argument(
        "--collect-after",
        action="store_true",
        help="实验完成后自动收集结果",
    )
    parser.add_argument(
        "--output-file",
        default=str(ROOT_DIR / "实验结果汇总.md"),
        help="结果汇总报告输出路径",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "check":
        _check_environment()
        return

    if args.mode == "collect":
        _collect_results(Path(args.output_file))
        return

    if args.mode == "custom":
        _run_custom(args.dataset, args.pred_len, args.seed)
    else:
        _run_repo_mode(args.mode)

    if args.collect_after:
        _collect_results(Path(args.output_file))


if __name__ == "__main__":
    main()