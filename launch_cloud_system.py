#!/usr/bin/env python3
"""
云端服务器启动脚本
用于启动云端 gRPC API 服务器和推理模块
"""

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.venus_system_cloud import VenusSystemCloud
from src.config import Config


def main():
    """主函数"""
    # 加载配置
    config = Config()
    
    # 创建并启动云端系统
    system = VenusSystemCloud(config)
    system.start()


if __name__ == "__main__":
    main()