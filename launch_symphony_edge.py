#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Symphony 边端启动：使用 symconfig_orin.json（或 --config 指定），
入口为 src.system.symphony.v2（StreamVideoInput + MemoryManagerOnlineV3，可配置退回 V2）。

运行前请在配置中设置 stream_uri（RTSP）或保留 video_file_path 做本地文件测试；
Orin 上请确认 frame_vectorizer / query_vectorizer 的 model_path 与设备可用。
"""

import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.config import SymConfig
from src.system.symphony.v2.edge import main as symphony_edge_main


if __name__ == "__main__":
    symphony_edge_main(SymConfig)
