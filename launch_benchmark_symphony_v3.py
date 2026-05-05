#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Symphony v3 基准测试启动（retrieve_item_type=clip 等见 configs/symconfig_v3.json）。"""

import argparse
import os
import shutil
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.config import SYSTEM_MODE_VLM_QA, SymConfig
from src.system.symphony.v3.benchmark import SymphonySystemBenchV3


def parse_args():
    parser = argparse.ArgumentParser(description="SymphonySystemBenchV3 云边集成 Benchmark")
    parser.add_argument("--skip-inject", action="store_true", help="跳过 inject，仅 query")
    parser.add_argument("--max-queries", type=int, default=None, help="最多查询条数")
    parser.add_argument("--max-videos", type=int, default=None, help="最多处理视频数")
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        choices=["short", "medium", "long"],
        help="Video-MME 子集（short/medium/long），传入后覆盖配置文件中的 benchmark.subset",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/symconfig_v3.json",
        help="配置文件路径（默认 configs/symconfig_v3.json）",
    )
    parser.add_argument(
        "--no-cloud",
        action="store_true",
        help="清除 system_mode 的 VLM 位（仅检索）",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="断点续跑：指定已有结果 JSON 路径，从中读取已处理视频并继续",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if shutil.which("ffprobe") is None:
        raise RuntimeError("benchmark 指定 ffprobe 扫描 GOP，但系统未找到 ffprobe（请安装 ffmpeg 套件）")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("benchmark 指定 ffmpeg 导出 clip，但系统未找到 ffmpeg")

    config = SymConfig(args.config)
    # v3 benchmark 显式固定后端：GOP 扫描仅用 ffprobe（V1），避免误走 GStreamer 路径。
    config.video_input_version = "V1"
    if args.subset:
        config.benchmark_subset = args.subset
    if args.no_cloud:
        config.system_mode = int(config.system_mode) & ~SYSTEM_MODE_VLM_QA
    bench = SymphonySystemBenchV3(config)
    bench.run(
        skip_inject=args.skip_inject,
        max_queries=args.max_queries,
        max_videos=args.max_videos,
        resume_path=args.resume,
    )
