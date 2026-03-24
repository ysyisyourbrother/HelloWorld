#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行 Video-MME 检索流程，并通过钩子抓取全量相似度，绘制折线图。
支持 --topk 参数，将相似度最高的 k 个帧在图中标记为五角星。
检索到的视频帧会保存到 output_path 下以问题命名的文件夹中。
"""
import argparse
import os
import re
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import SymConfig
from src.venus_system_motivation import SymphonySystemMoti


def parse_args():
    parser = argparse.ArgumentParser(description="检索并绘制相似度折线图")
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="将相似度最高的 k 个帧标记为五角星，0 表示不标记",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="motivation_results/retrieve",
        help="输出图片路径",
    )
    return parser.parse_args()


def _question_to_folder_name(question: str, max_len: int = 50) -> str:
    """将问题文本转为文件夹名，过长则截断，非法字符替换为下划线"""
    s = re.sub(r'[\\/:*?"<>|\n\r\t]', "_", question.strip())
    s = s.replace(" ", "_")
    if len(s) > max_len:
        s = s[:max_len]
    return s or "question"


def main():
    args = parse_args()

    captured = {}

    def capture_all_scores(query_vector, all_scores):
        captured["query_vector"] = query_vector.copy()
        captured["all_scores"] = all_scores.copy()

    config = SymConfig(config_path="configs/symconfig_moti.json")
    # config.benchmark_use_cloud = False

    moti = SymphonySystemMoti(config)
    moti.register_retrieve_hook(capture_all_scores)

    result = moti.run_video_flow(
        video_path="local_datasets/Video-MME/data/44ivpEIcBhE.mp4",
        video_id="44ivpEIcBhE",
        questions=[
            {
                # "question": "Which instrument is the performer on the stage holding in the video?",
                "question": "Instrument?",
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
        subset="short",
    )

    if not captured:
        print("未抓取到相似度数据，请检查钩子是否生效")
        return

    all_scores = captured["all_scores"]
    frames = np.arange(len(all_scores))

    plt.figure(figsize=(12, 4),dpi=300)
    plt.plot(frames, all_scores, "b-", linewidth=1, label="Similarity")

    if args.topk > 0:
        # FlatIP: 越大越相似；FlatL2: 越小越相似。config 为 FlatIP，取最大的 k 个
        topk = min(args.topk, len(all_scores))
        top_indices = np.argsort(all_scores)[-topk:][::-1]
        top_frames = top_indices
        top_vals = [all_scores[i] for i in top_indices]
        plt.scatter(
            top_frames,
            top_vals,
            marker="*",
            s=200,
            c="red",
            edgecolors="darkred",
            linewidths=1.5,
            zorder=5,
            label=f"Top-{topk}",
        )

    query_results = result.get("query_results", [])
    question = query_results[0].get("question", "")
    folder_name = _question_to_folder_name(question)
    frame_dir = os.path.join(args.output_path, folder_name)
    plot_path = os.path.join(frame_dir, "similarity_plot.png")

    plt.xlabel("Frame ID", fontsize=12)
    plt.ylabel("Similarity", fontsize=12)
    plt.title(f'FPS=25, "{question}"', fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    print(f"相似度折线图已保存至: {plot_path}")

    # 保存检索到的视频帧：检索排名_原视频帧数_原视频秒数.png
    query_results = result.get("query_results", [])
    for qr in query_results:
        frames = qr.get("retrieved_frames", [])
        metadata = qr.get("retrieved_frames_metadata", [])
        if not frames:
            continue
        os.makedirs(frame_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            frame_id = metadata[i]["frame_id"] if i < len(metadata) else i
            fps = metadata[i]["video_fps"] if i < len(metadata) else 1.0
            seconds = round(frame_id / fps, 2)
            fname = f"{i}_{frame_id}_{seconds}.png"
            path = os.path.join(frame_dir, fname)
            if isinstance(frame, np.ndarray):
                cv2.imwrite(path, frame)
            else:
                cv2.imwrite(path, np.array(frame))
        print(f"检索帧已保存至: {frame_dir} ({len(frames)} 帧)")


if __name__ == "__main__":
    main()
