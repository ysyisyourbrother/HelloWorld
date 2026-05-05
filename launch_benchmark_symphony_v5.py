#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Symphony v5 基准测试启动（MemoryAgent agentic 检索、is_local_vlm 等见 configs/symconfig_v5_moti.json）。"""

import argparse
import os
import shutil
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.config import SYSTEM_MODE_VLM_QA, SymConfig
from src.system.symphony.v5.benchmark import SymphonySystemBenchV5


def parse_args():
    parser = argparse.ArgumentParser(description="SymphonySystemBenchV5 云边集成 Benchmark")
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
        default="configs/symconfig_v5_moti.json",
        help="配置文件路径（默认 configs/symconfig_v5_moti.json，含 v5 流式与 benchmark 段）",
    )
    parser.add_argument("--no-cloud", action="store_true", help="不调用云端，仅测试边端检索")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="断点续跑：指定已有结果 JSON 路径，从中读取已处理视频并继续",
    )
    parser.add_argument(
        "--system_mode",
        type=int,
        default=None,
        help="覆盖配置中的 system_mode 十进制位掩码（见 src.config 中 SYSTEM_MODE_*）；不传则用配置文件",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if shutil.which("ffprobe") is None:
        raise RuntimeError("benchmark 指定 ffprobe 扫描 GOP，但系统未找到 ffprobe（请安装 ffmpeg 套件）")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("benchmark 指定 ffmpeg 导出 clip，但系统未找到 ffmpeg")

    config = SymConfig(args.config)
    if args.system_mode is not None:
        config.system_mode = int(args.system_mode)
    # v5 编排与 v3/v4 一致：GOP 扫描走 ffprobe（V1），避免误走 GStreamer 路径。
    config.video_input_version = "V1"
    if args.subset:
        config.benchmark_subset = args.subset
    if args.no_cloud:
        config.system_mode = int(config.system_mode) & ~SYSTEM_MODE_VLM_QA
    bench = SymphonySystemBenchV5(config)
    bench.run(
        skip_inject=args.skip_inject,
        max_queries=args.max_queries,
        max_videos=args.max_videos,
        resume_path=args.resume,
    )
