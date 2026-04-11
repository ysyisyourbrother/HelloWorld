#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Symphony v3 基准测试启动（retrieve_item_type=clip 等见 configs/symconfig_v3.json）。"""

import argparse
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.config import SymConfig
from src.system.symphony.v3.benchmark import SymphonySystemBenchV3


def parse_args():
    parser = argparse.ArgumentParser(description="SymphonySystemBenchV3 云边集成 Benchmark")
    parser.add_argument("--skip-inject", action="store_true", help="跳过 inject，仅 query")
    parser.add_argument("--max-queries", type=int, default=None, help="最多查询条数")
    parser.add_argument("--max-videos", type=int, default=None, help="最多处理视频数")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/symconfig_v3.json",
        help="配置文件路径（默认 configs/symconfig_v3.json）",
    )
    parser.add_argument("--no-cloud", action="store_true", help="不调用云端，仅测试边端检索")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="断点续跑：指定已有结果 JSON 路径，从中读取已处理视频并继续",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = SymConfig(args.config)
    if args.no_cloud:
        config.benchmark_use_cloud = False
    bench = SymphonySystemBenchV3(config)
    bench.run(
        skip_inject=args.skip_inject,
        max_queries=args.max_queries,
        max_videos=args.max_videos,
        resume_path=args.resume,
    )
