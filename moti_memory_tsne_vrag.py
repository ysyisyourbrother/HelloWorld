#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vrag motivation：对指定视频仅做帧向量编码与入库（不写检索 / VLM）。

向量库路径由 config 中 benchmark 的 db 目录与 subset 决定（默认与 config_moti.json
一致：motivation_results/memory/...），文件名形如「视频主文件名_frame_interval」。
"""
import argparse
import os

from src.config import Config, SYSTEM_MODE_REINJECT_MEMORY
from src.system.vrag.motivation import VragSystemMoti


def parse_args():
    parser = argparse.ArgumentParser(
        description="指定视频与 frame_interval，仅编码入库（system_mode 默认仅记忆注入）"
    )
    parser.add_argument(
        "--config",
        default="configs/config_moti.json",
        help="配置文件路径",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="待编码视频的本地路径",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=None,
        help="覆盖配置中的 frame_interval（与 frame_vectorizer.frame_interval 一致）",
    )
    parser.add_argument(
        "--dataset_name",
        default="Video-MME",
        help="数据集名，用于选择 benchmark_db_dir_* 下的子目录",
    )
    parser.add_argument(
        "--subset",
        default="short",
        help="子集目录名（如 short），库文件位于 .../memory/<dataset>/<subset>/faiss/",
    )
    parser.add_argument(
        "--system_mode",
        type=int,
        default=SYSTEM_MODE_REINJECT_MEMORY,
        help=(
            "覆盖配置 system_mode 位掩码；默认 %d 表示仅记忆重注入，"
            "不预加载 Query 文本编码器（仅帧编码）。见 src.config 中 SYSTEM_MODE_*"
            % SYSTEM_MODE_REINJECT_MEMORY
        ),
    )
    parser.add_argument(
        "--no-force-update",
        dest="no_force_update",
        action="store_true",
        help="若目标 faiss 已存在则跳过注入（默认会强制重建）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_path = os.path.abspath(os.path.expanduser(args.video))
    if not os.path.isfile(video_path):
        print("视频文件不存在: %s" % video_path)
        raise SystemExit(1)

    config = Config(config_path=args.config)
    if args.frame_interval is not None:
        config.frame_interval = int(args.frame_interval)
    frame_interval = int(config.frame_interval)

    config.system_mode = int(args.system_mode)

    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    video_id = "%s_%d" % (video_stem, frame_interval)

    moti = VragSystemMoti(config)
    faiss_path, map_path, srt_path = moti._get_db_paths(
        args.dataset_name, video_id, args.subset
    )

    force_update = not args.no_force_update
    stats = moti.inject_video(
        video_path=video_path,
        video_id=video_id,
        dataset_name=args.dataset_name,
        subset=args.subset,
        force_update=force_update,
    )

    print("frame_interval=%d system_mode=%d video_id=%s" % (frame_interval, config.system_mode, video_id))
    print("faiss: %s" % faiss_path)
    print("databasemap: %s" % map_path)
    print("srt: %s" % srt_path)
    print("inject_stats: %s" % stats)
