#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import SymConfig
from src.system.symphony.v5.motivation import SymphonySystemMotiV5


def parse_args():
    parser = argparse.ArgumentParser(description="Symphony v5 单视频检索并获得答案")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/symconfig_v5_moti.json",
        help="配置文件路径",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="demo/assets/fFjv93ACGo8.mp4",
        help="待检索视频路径",
    )
    parser.add_argument(
        "--video_id",
        type=str,
        default="fFjv93ACGo8",
        help="视频 ID（用于向量库命名）",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="short",
        help="Video-MME 子集名",
    )
    parser.add_argument(
        "--dialog_id",
        type=int,
        default=0,
        help="检索对话 ID（v5 可用于 clip 输出目录隔离）",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="motivation_results_symphony_v5/retrieve",
        help="输出路径",
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

    captured = {}

    config = SymConfig(config_path=args.config)
    if args.system_mode is not None:
        config.system_mode = int(args.system_mode)
    moti = SymphonySystemMotiV5(config)

    result = moti.run_video_flow(
        video_path=args.video_path,
        video_id=args.video_id,
        questions=[
            {
                "question": "When demonstrating the Germany modern Christmas tree is initially decorated with apples, candles and berries, which kind of the decoration has the largest number?",
                "options": [
                "A. Apples.",
                "B. Candles.",
                "C. Berries.",
                "D. The three kinds are of the same number."
                ],
                "answer": "C",
            }
        ],
        dataset_name="Video-MME",
        subset=args.subset,
        dialog_id=args.dialog_id,
    )

    print(result)