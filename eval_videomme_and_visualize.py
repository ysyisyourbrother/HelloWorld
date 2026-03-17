#!/usr/bin/env python3
"""评估 Video-MME benchmark 结果，调用 src/benchmark/videomme 的 eval_your_results，并可视化柱状图。"""
import argparse
import glob
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.benchmark.videomme import eval_your_results, eval_your_results_then_return


def draw_subplot(
    ax: plt.Axes,
    sub_dict: Dict[str, float],
    title: str,
    overall_acc: float,
) -> None:
    """Draw a bar chart subplot with category accuracy and overall acc line."""
    categories = list(sub_dict.keys())
    accuracies = list(sub_dict.values())

    x = range(len(categories))
    ax.bar(x, accuracies, color="steelblue", edgecolor="white", linewidth=0.5)

    ax.axhline(y=overall_acc, color="red", linestyle="--", linewidth=1.5, zorder=10)
    ax.text(
        0.02, 0.98,
        f"Overall Acc: {overall_acc}%",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        color="red",
        fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)


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
        default="benchmark_results/useful/benchmark_Video-MME_medium_20260309_180348.json",
        help="指定 benchmark 结果 JSON 路径；未指定时自动检索最新文件",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="benchmark_results/useful",
        help="检索最新结果时的目录，默认 benchmark_results",
    )
    parser.add_argument(
        "--video_duration_type",
        type=str,
        default="medium",
        help="视频时长类型，如 short/medium/long，默认 short",
    )
    parser.add_argument("--skip_missing", action="store_true", help="跳过缺失样本，不强制要求 300 条")
    parser.add_argument("--return_categories_accuracy", action="store_true")
    parser.add_argument("--return_sub_categories_accuracy", action="store_true")
    parser.add_argument("--return_task_types_accuracy", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="柱状图输出路径；未指定则不保存图片",
    )
    parser.add_argument("--no_visualize", action="store_true", help="不生成可视化，仅命令行打印")

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

    # 默认开启三类准确率，用于可视化
    return_cat = args.return_categories_accuracy or not args.no_visualize
    return_sub = args.return_sub_categories_accuracy or not args.no_visualize
    return_task = args.return_task_types_accuracy or not args.no_visualize

    if args.no_visualize:
        eval_your_results(
            results_path,
            video_types=args.video_duration_type,
            skip_missing=args.skip_missing,
            return_categories_accuracy=return_cat,
            return_sub_categories_accuracy=return_sub,
            return_task_types_accuracy=return_task,
        )
    else:
        metrics = eval_your_results_then_return(
            results_path,
            video_types=args.video_duration_type,
            skip_missing=args.skip_missing,
            return_categories_accuracy=True,
            return_sub_categories_accuracy=True,
            return_task_types_accuracy=True,
        )
        overall_acc = metrics["Overall Acc"]
        print(f"Overall Acc: {overall_acc}%")

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        draw_subplot(
            axes[0],
            metrics["Video Domains"],
            "Video Domains",
            overall_acc,
        )
        draw_subplot(
            axes[1],
            metrics["Video Sub Categories"],
            "Video Sub Categories",
            overall_acc,
        )
        draw_subplot(
            axes[2],
            metrics["Task Categories"],
            "Task Categories",
            overall_acc,
        )
        plt.tight_layout()

        if args.output:
            plt.savefig(args.output, dpi=150, bbox_inches="tight")
            print(f"Saved figure to {args.output}")
        else:
            out_dir = os.path.dirname(results_path) or "."
            out_name = "videomme_accuracy_bar.png"
            out_path = os.path.join(out_dir, out_name)
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved figure to {out_path}")
        plt.close()


if __name__ == "__main__":
    main()
