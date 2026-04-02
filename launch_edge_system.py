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

from src.system.venus.edge import VenusSystemEdge
from src.config import Config


def main():
    """主函数"""
    # 加载配置
    config = Config()

    # config.edge_mode = "only_inject"

    config.edge_mode = "query_with_memory"
    # config.memory_faiss_file_path = "motivation_results_symphony/memory/videomme/short/faiss/44ivpEIcBhE.faiss"
    # config.memory_databasemap_file_path = "motivation_results_symphony/memory/videomme/short/json/44ivpEIcBhE.json"
    # 创建并启动边端系统
    system = VenusSystemEdge(config)
    system.start()

if __name__ == "__main__":
    main()

