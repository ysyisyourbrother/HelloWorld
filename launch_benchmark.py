#!/usr/bin/env python3
"""
基准测试启动脚本
"""

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.venus_system_benchmark import VenusSystemBench
from src.config import Config

def main():
    import argparse
    parser = argparse.ArgumentParser(description="VenusSystemBench 云边集成 Benchmark")
    parser.add_argument("--skip-inject", action="store_true", help="跳过 inject，仅 query")
    parser.add_argument("--max-queries", type=int, default=None, help="最多查询条数")
    parser.add_argument("--max-videos", type=int, default=None, help="最多处理视频数")
    parser.add_argument("--config", type=str, default="configs/config.json", help="配置文件路径")
    parser.add_argument("--no-cloud", action="store_true", help="不调用云端，仅测试边端检索")
    args = parser.parse_args()

    config = Config(args.config)
    if args.no_cloud:
        config.benchmark_use_cloud = False
    bench = VenusSystemBench(config)
    bench.run(
        skip_inject=args.skip_inject,
        max_queries=args.max_queries,
        max_videos=args.max_videos,
    )


if __name__ == "__main__":
    main()
