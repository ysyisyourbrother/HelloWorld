#!/usr/bin/env python3
"""评估 Video-MME benchmark 结果，调用 src/benchmark/videomme 的 eval_your_results。"""
import argparse
import glob
import os
from typing import Optional

from src.benchmark.videomme import eval_your_results


def get_latest_benchmark_result(results_dir: str = "benchmark_results", pattern: str = "benchmark_Video-MME_*.json") -> Optional[str]:
    """
    检索 results_dir 下最新的 benchmark 结果文件。

    Args:
        results_dir: 结果目录，默认 benchmark_results
        pattern: 文件名模式，默认 benchmark_Video-MME_*.json

    Returns:
        最新文件的完整路径，若无匹配则返回 None
    """
    if not os.path.isdir(results_dir):
        return None
    files = glob.glob(os.path.join(results_dir, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="评估 Video-MME benchmark 结果")
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
        help="指定 benchmark 结果 JSON 路径；未指定时自动检索最新文件",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="benchmark_results",
        help="检索最新结果时的目录，默认 benchmark_results",
    )
    parser.add_argument(
        "--video_duration_type",
        type=str,
        default="short",
        help="视频时长类型，如 short/medium/long，默认 short",
    )
    parser.add_argument("--skip_missing", action="store_true", help="跳过缺失样本，不强制要求 300 条")
    parser.add_argument("--return_categories_accuracy", action="store_true")
    parser.add_argument("--return_sub_categories_accuracy", action="store_true")
    parser.add_argument("--return_task_types_accuracy", action="store_true")

    args = parser.parse_args()

    results_path = args.results_file
    if results_path is None:
        results_path = get_latest_benchmark_result(args.results_dir)
        if results_path is None:
            raise FileNotFoundError(
                f"未找到 benchmark 结果文件，目录: {args.results_dir}，"
                "请指定 --results_file 或确保目录下存在 benchmark_Video-MME_*.json"
            )
        print(f"使用最新结果: {results_path}")

    eval_your_results(
        results_path,
        video_types=args.video_duration_type,
        skip_missing=args.skip_missing,
        return_categories_accuracy=args.return_categories_accuracy,
        return_sub_categories_accuracy=args.return_sub_categories_accuracy,
        return_task_types_accuracy=args.return_task_types_accuracy,
    )


if __name__ == "__main__":
    main()
