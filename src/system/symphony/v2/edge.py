# -*- coding: utf-8 -*-
"""
Symphony 边端 v2：封装 VenusSystemEdge，作为 Symphony 专用启动入口。
流式场景使用 StreamVideoInput + MemoryManagerOnlineV3（由配置 memory_manager.online_v3 控制）。
"""

import argparse
import os
import sys

from src.system.venus.edge import VenusSystemEdge


class SymphonySystemEdge(VenusSystemEdge):
    """与 VenusSystemEdge 行为一致；后续可在本类扩展 Symphony 专用逻辑而不改 venus。"""

    pass


def parse_symphony_edge_args(argv=None):
    """
    解析边端启动参数。argv 为 None 时使用 sys.argv。
    """
    parser = argparse.ArgumentParser(
        description="启动 Symphony 边端（Jetson Orin 等，GStreamer 流 + 记忆注入）"
    )
    parser.add_argument(
        "--config",
        default="configs/symconfig_orin.json",
        help="配置文件路径（默认 configs/symconfig_orin.json）",
    )
    parser.add_argument(
        "--edge-mode",
        default=None,
        choices=[
            "query_while_inject",
            "query_with_memory",
            "retrieve_while_inject",
            "retrieve_with_memory",
            "only_inject",
            "benchmark",
        ],
        help="覆盖配置文件 edge.mode；不设则使用配置内取值",
    )
    return parser.parse_args(argv)


def main(config_cls, argv=None):
    """
    从配置类与命令行启动边端。config_cls 通常为 SymConfig。
    """
    args = parse_symphony_edge_args(argv)
    cfg_path = args.config
    if not os.path.isfile(cfg_path):
        print("配置文件不存在: %s" % cfg_path, file=sys.stderr)
        sys.exit(1)

    config = config_cls(config_path=cfg_path)
    if args.edge_mode is not None:
        config.edge_mode = args.edge_mode

    system = SymphonySystemEdge(config)
    system.start()
