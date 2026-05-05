#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行 Symphony v3 的 Video-MME 检索流程，并通过钩子抓取全量相似度，绘制折线图。
支持 --topk 参数，将相似度最高的 k 个帧在图中标记为五角星。

检索结束后，若 ``system_mode`` 启用 VLM 且 ``benchmark_is_local_vlm`` 等条件满足，会经 ``SymphonySystemMotiV3``
调用大模型（与 v3 Benchmark 一致：frame 模式为 top-k 单帧，clip 模式为 GOP mp4 全帧）。
"""
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
from src.system.symphony.v3.motivation import SymphonySystemMotiV3


def parse_args():
    parser = argparse.ArgumentParser(description="Symphony v3 检索并绘制相似度折线图")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/symconfig_v3_moti.json",
        help="配置文件路径",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="将相似度最高的 k 个帧标记为五角星，0 表示不标记",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="demo/assets/44ivpEIcBhE.mp4",
        help="待检索视频路径",
    )
    parser.add_argument(
        "--video_id",
        type=str,
        default="44ivpEIcBhE",
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
        help="检索对话 ID（v3 可用于 clip 输出目录隔离）",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="motivation_results_symphony_v3/retrieve",
        help="输出路径",
    )
    return parser.parse_args()


def _question_to_folder_name(question, max_len=50):
    """将问题文本转为文件夹名，过长则截断，非法字符替换为下划线。"""
    safe_name = re.sub(r'[\\/:*?"<>|\n\r\t]', "_", question.strip())
    safe_name = safe_name.replace(" ", "_")
    if len(safe_name) > max_len:
        safe_name = safe_name[:max_len]
    return safe_name or "question"


def _load_frames_and_i_frames(map_path):
    """从 databasemap json 读取 frames/i_frames/total_frames。兼容单视频对象或视频数组。"""
    if not map_path or (not os.path.isfile(map_path)):
        return [], [], None

    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_obj = None
    if isinstance(data, dict) and "frames" in data:
        video_obj = data
    elif isinstance(data, list) and data and isinstance(data[0], dict) and "frames" in data[0]:
        video_obj = data[0]

    if not isinstance(video_obj, dict):
        return [], [], None

    frames = video_obj.get("frames") or []
    i_frames = video_obj.get("i_frames") or []
    try:
        frames = [int(x) for x in frames]
        i_frames = sorted(set(int(x) for x in i_frames))
    except (TypeError, ValueError):
        return [], [], None
    total_frames = video_obj.get("total_frames")
    try:
        total_frames = int(total_frames) if total_frames is not None else None
    except (TypeError, ValueError):
        total_frames = None
    return frames, i_frames, total_frames


if __name__ == "__main__":
    args = parse_args()

    captured = {}

    def capture_all_scores(query_vector, all_scores):
        captured["query_vector"] = query_vector.copy()
        captured["all_scores"] = all_scores.copy()

    config = SymConfig(config_path=args.config)
    moti = SymphonySystemMotiV3(config)
    moti.register_retrieve_hook(capture_all_scores)

    result = moti.run_video_flow(
        video_path=args.video_path,
        video_id=args.video_id,
        questions=[
            {
                "question": "Which instrument is the performer on the stage holding in the video?",
                # "question": "Instrument?",
                "options": [
                    "A. Trumpet.",
                    "B. Saxophone.",
                    "C. Violin.",
                    "D. Guitar.",
                ],
                "answer": "B",
            }
        ],
        dataset_name="Video-MME",
        subset=args.subset,
        dialog_id=args.dialog_id,
    )

    if not captured:
        print("未抓取到相似度数据，请检查钩子是否生效")
        raise SystemExit(1)

    all_scores = captured["all_scores"]
    # 读取 v3 向量库 databasemap，优先用原始 frame_id 作为横轴。
    _faiss_path, map_path, _srt_path = moti._get_db_paths("Video-MME", args.video_id, args.subset)
    db_frame_ids, i_frame_ids, db_total_frames = _load_frames_and_i_frames(map_path)
    if len(db_frame_ids) == len(all_scores):
        frame_ids = np.array(db_frame_ids)
    else:
        frame_ids = np.arange(len(all_scores))
        if db_frame_ids:
            print(
                "警告: databasemap 中 frames 数量({})与 all_scores 数量({})不一致，将退回索引横轴".format(
                    len(db_frame_ids), len(all_scores)
                )
            )

    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)

    # GOP 背景：按 i_frame 区间交替浅红/浅蓝，并统一使用 // hatch。
    if i_frame_ids:
        gop_edges = list(i_frame_ids)
        if db_total_frames is not None and (not gop_edges or db_total_frames > gop_edges[-1]):
            gop_edges.append(db_total_frames)

    if i_frame_ids and len(gop_edges) >= 2:
        gop_colors = ("#F9B29F", "#ACD4D0")  # 浅红 / 浅蓝
        for idx in range(len(gop_edges) - 1):
            start = gop_edges[idx]
            end = gop_edges[idx + 1]
            if end <= start:
                continue
            color = gop_colors[idx % 2]
            ax.axvspan(
                start,
                end,
                facecolor=color,
                alpha=0.50,
                hatch="//",
                edgecolor=color,
                linewidth=0.0,
                zorder=0,
            )
        print("已绘制 GOP 区间: {} 段".format(len(gop_edges) - 1))
    else:
        print("未在 databasemap 中找到可用 i_frames，跳过 GOP 区间着色")

    ax.plot(
        frame_ids,
        all_scores,
        marker="o",
        linewidth=1,
        label="Similarity",
        zorder=3,
        color="#000000",
        linestyle="--"
    )
   

    if args.topk > 0:
        topk = min(args.topk, len(all_scores))
        top_indices = np.argsort(all_scores)[-topk:][::-1]
        top_frames = frame_ids[top_indices]
        top_vals = [all_scores[i] for i in top_indices]
        ax.scatter(
            top_frames,
            top_vals,
            marker="*",
            s=200,
            c="red",
            edgecolors="darkred",
            linewidths=1.5,
            zorder=5,
            label="Top-{}".format(topk),
        )

    query_results = result.get("query_results", [])
    question = query_results[0].get("question", "") if query_results else "question"
    folder_name = _question_to_folder_name(question)
    frame_dir = os.path.join(args.output_path, folder_name)
    plot_path = os.path.join(frame_dir, "similarity_plot.png")

    ax.set_xlabel("Frame ID", fontsize=12)
    ax.set_ylabel("Similarity", fontsize=12)
    ax.set_title('FPS=25, "{}"'.format(question), fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)
    print("相似度折线图已保存至: {}".format(plot_path))

    # 兼容 frame 模式：如果返回了检索帧则落盘；clip 模式时此处会自然跳过。
    for query_item in query_results:
        retrieved_frames = query_item.get("retrieved_frames", [])
        metadata = query_item.get("retrieved_frames_metadata", [])
        if not retrieved_frames:
            continue
        os.makedirs(frame_dir, exist_ok=True)
        for i, frame in enumerate(retrieved_frames):
            frame_id = metadata[i]["frame_id"] if i < len(metadata) else i
            fps = metadata[i]["video_fps"] if i < len(metadata) else 1.0
            seconds = round(frame_id / fps, 2)
            filename = "{}_{}_{}.png".format(i, frame_id, seconds)
            output_file = os.path.join(frame_dir, filename)
            if isinstance(frame, np.ndarray):
                cv2.imwrite(output_file, frame)
            else:
                cv2.imwrite(output_file, np.array(frame))
        print("检索帧已保存至: {} ({} 帧)".format(frame_dir, len(retrieved_frames)))

    # v3 新增：打印 clip 检索信息（若 memory_retrieve_item_type=clip）
    if query_results:
        first_query = query_results[0]
        clip_paths = first_query.get("retrieve_clip_paths") or []
        clip_dir = first_query.get("retrieve_clip_dir")
        if clip_dir:
            print("检索 clip 目录: {}".format(clip_dir))
        if clip_paths:
            print("检索 clip 数量: {}".format(len(clip_paths)))
        ans = first_query.get("cloud_result")
        if ans:
            print("模型回答: {}".format(ans))
        err = first_query.get("cloud_error")
        if err and not ans:
            print("推理未返回文本: {}".format(err))
