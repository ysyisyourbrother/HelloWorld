#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
moti_inject - 测试 frame_vectorizer 编码功能

支持通过 args 传参，对视频或图片进行向量编码。
"""

import argparse
import os
import sys
import time
import cv2
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.video_input import VideoInput, FrameData
from src.frame_vectorizer import FrameVectorizer, FrameVectorData


def parse_args():
    parser = argparse.ArgumentParser(description="测试 frame_vectorizer 编码功能")
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="motivation_results/frames/frame_selected.jpg",
        help="输入路径：视频文件(.mp4/.avi 等)或图片文件(.jpg/.png 等)"
    )
    parser.add_argument(
        "-c", "--config",
        default="configs/config_moti.json",
        help="配置文件路径 (默认: configs/config_moti.json)"
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=16,
        help="批大小 (默认: 16)"
    )
    parser.add_argument(
        "-f", "--frame_interval",
        type=int,
        default=1,
        help="帧间隔，每 N 帧取一帧 (默认: 1，不跳过)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="覆盖配置中的 device (如 cuda/cpu)"
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="覆盖配置中的 model_path"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="可选：将向量保存到 .npy 文件"
    )
    return parser.parse_args()


def is_image(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_image_as_frame_data(path: str) -> List[FrameData]:
    """从图片加载为 FrameData 列表（单帧）"""
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return [
        FrameData(
            frame=frame_rgb,
            timestamp=0.0,
            frame_id=0,
            source_path=path,
            total_frames=1,
            video_fps=None,
            duration=None,
        )
    ]


def iter_frames_from_video(
    config: Config,
    video_path: str,
    batch_size: int,
    frame_interval: int,
):
    """从视频按 batch 迭代 FrameData"""
    config.video_file_path = video_path
    video_input = VideoInput(config)
    video_input.init_for_file(video_path)
    yield from video_input.iter_frames_batch(
        batch_size=batch_size,
        frame_interval=frame_interval,
    )


def main():
    args = parse_args()
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"错误: 输入路径不存在: {input_path}")
        sys.exit(1)

    # 加载配置
    config = Config(args.config)
    if args.device is not None:
        config.frame_device = args.device
    if args.model_path is not None:
        config.frame_model_path = args.model_path

    # 初始化 FrameVectorizer
    frame_vectorizer = FrameVectorizer(config)
    frame_vectorizer._initialize_vectorizer()

    total_vectors = 0
    all_vectors = []
    t0 = time.time()

    if is_image(input_path):
        # 单图模式
        frame_data_list = load_image_as_frame_data(input_path)
        vector_data_list = frame_vectorizer.encode_frames_batch(frame_data_list)
        total_vectors = len(vector_data_list)
        all_vectors = [vd.vector for vd in vector_data_list]
        print(f"图片编码完成: 1 帧 -> {total_vectors} 向量, shape={vector_data_list[0].vector.shape}")
    else:
        # 视频模式
        for batch in iter_frames_from_video(
            config, input_path, args.batch_size, args.frame_interval
        ):
            vector_data_list = frame_vectorizer.encode_frames_batch(batch)
            total_vectors += len(vector_data_list)
            all_vectors.extend([vd.vector for vd in vector_data_list])
            print(f"已处理 {total_vectors} 帧 -> {total_vectors} 向量")

    elapsed = time.time() - t0
    print(f"编码完成: 共 {total_vectors} 向量, 耗时 {elapsed:.2f}s")

    if args.output and all_vectors:
        import numpy as np
        arr = np.concatenate(all_vectors, axis=0)
        np.save(args.output, arr)
        print(f"向量已保存到: {args.output}, shape={arr.shape}")


if __name__ == "__main__":
    main()
