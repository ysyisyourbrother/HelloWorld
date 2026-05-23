#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Venus 基准测试启动（场景聚类注入 + Venus 检索，配置见 configs/config_moti.json 的 venus 段）。"""

import argparse
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.config import SYSTEM_MODE_VLM_QA, Config
from src.system.venus.benchmark import VenusSystemBench


def parse_args():
    parser = argparse.ArgumentParser(description="VenusSystemBench 数据集 Benchmark")
    parser.add_argument("--skip-inject", action="store_true", help="跳过 inject，仅 query（需已有 faiss/json/enhance）")
    parser.add_argument("--max-queries", type=int, default=None, help="最多查询条数")
    parser.add_argument("--max-videos", type=int, default=None, help="最多处理视频数")
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        choices=["short", "medium", "long"],
        help="Video-MME 子集，传入后覆盖配置文件 benchmark.subset",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_moti.json",
        help="配置文件路径（默认 configs/config_moti.json）",
    )
    parser.add_argument(
        "--no-cloud",
        action="store_true",
        help="清除 system_mode 的 VLM 位，仅做检索不写云端/本地 VLM 答案",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="断点续跑：指定已有结果 JSON 路径",
    )
    parser.add_argument(
        "--system_mode",
        type=int,
        default=None,
        help="覆盖配置 system_mode（见 src.config SYSTEM_MODE_*）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = Config(args.config)
    if args.system_mode is not None:
        config.system_mode = int(args.system_mode)
    if args.subset:
        config.benchmark_subset = args.subset
    if args.no_cloud:
        config.system_mode = int(config.system_mode) & ~SYSTEM_MODE_VLM_QA

    bench = VenusSystemBench(config)
    out = bench.run(
        skip_inject=args.skip_inject,
        max_queries=args.max_queries,
        max_videos=args.max_videos,
        resume_path=args.resume,
    )
    print(out)
