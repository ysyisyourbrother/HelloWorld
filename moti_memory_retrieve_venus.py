#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Venus 单视频 inject + 检索 +（可选）VLM 问答，用法对齐 moti_memory_retrieve_sym_v5.py。"""

import argparse
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.config import Config
from src.system.venus.motivation import VenusSystemMoti


def parse_args():
    parser = argparse.ArgumentParser(description="Venus 单视频检索并获得答案")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/venus_config_moti.json",
        help="配置文件路径",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="demo/assets/fFjv93ACGo8.mp4",
        help="待处理视频路径",
    )
    parser.add_argument(
        "--video_id",
        type=str,
        default="fFjv93ACGo8",
        help="视频 ID（faiss/json/enhance 文件名）",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="short",
        help="Video-MME 子集目录名",
    )
    parser.add_argument(
        "--dialog_id",
        type=int,
        default=0,
        help="检索对话 ID（clip 导出目录隔离）",
    )
    parser.add_argument(
        "--force_update",
        action="store_true",
        help="即使已有向量库也强制重新 inject",
    )
    parser.add_argument(
        "--system_mode",
        type=int,
        default=None,
        help="覆盖配置 system_mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = Config(config_path=args.config)
    if args.system_mode is not None:
        config.system_mode = int(args.system_mode)

    moti = VenusSystemMoti(config)
    result = moti.run_video_flow(
        video_path=args.video_path,
        video_id=args.video_id,
        questions=[
            {
                "question": (
                    "When demonstrating the Germany modern Christmas tree is initially "
                    "decorated with apples, candles and berries, which kind of the "
                    "decoration has the largest number?"
                ),
                "options": [
                    "A. Apples.",
                    "B. Candles.",
                    "C. Berries.",
                    "D. The three kinds are of the same number.",
                ],
                "answer": "C",
            }
        ],
        dataset_name="Video-MME",
        subset=args.subset,
        force_update=args.force_update,
        dialog_id=args.dialog_id,
    )

    print(result)
