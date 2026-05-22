#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adjacent encoded frames: cosine similarity of CLS-to-patch attention patterns.

Uses VragSystemMoti to encode a video, captures attention_weights via hook,
extracts CLS-to-visual-token attention per frame, then plots cosine similarity
between adjacent frames' attention patterns.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import SYSTEM_MODE_VLM_QA, Config
from src.system.vrag.motivation import VragSystemMoti


def make_accumulating_attention_hook(captured: dict, layer_idx: int):
    """
    Create a hook that accumulates attention_weights for a specific layer across batches.
    captured["attention_list"]: list of (bsz, num_heads, seq_len, seq_len)
    captured["frame_ids_list"]: list of [frame_id, ...] per batch
    """
    def hook(frame_data_list, vectors, hidden_states, attentions):
        if attentions is None:
            return
        if layer_idx >= len(attentions):
            return
        a = attentions[layer_idx]  # (bsz, num_heads, seq_len, seq_len)
        captured.setdefault("attention_list", []).append(a.copy())
        captured.setdefault("frame_ids_list", []).append(
            [fd.frame_id for fd in frame_data_list]
        )
    return hook


def cls_to_patch_attention(attn: np.ndarray) -> np.ndarray:
    """
    Extract CLS (query 0) to patches (key 1~196) attention, mean over heads.
    attn: (N, num_heads, seq_len, seq_len)
    Returns: (N, 196) flattened
    """
    cls_to_patch = attn[:, :, 0, 1:]  # [N, num_heads, 196]
    scores_flat = cls_to_patch.mean(axis=1)  # [N, 196]
    return scores_flat


def attention_pattern_cosine_similarity(
    attn_i: np.ndarray, attn_j: np.ndarray, eps: float = 1e-8
) -> float:
    """
    Compute cosine similarity between two CLS-to-patch attention patterns.
    attn_i, attn_j: (196,) each
    """
    dot = np.dot(attn_i, attn_j)
    norm_i = np.linalg.norm(attn_i) + eps
    norm_j = np.linalg.norm(attn_j) + eps
    return float(dot / (norm_i * norm_j))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot adjacent-frame CLS-to-patch attention similarity"
    )
    parser.add_argument(
        "-v", "--video_path",
        default="local_datasets/Video-MME/data/44ivpEIcBhE.mp4",
        help="Video file path",
    )
    parser.add_argument(
        "--video_id",
        default="44ivpEIcBhE_adjacent_attn",
        help="Video ID for DB (use unique ID to force fresh inject)",
    )
    parser.add_argument(
        "-l", "--layer",
        type=int,
        default=6,
        help="Layer index (0-based) to capture attention",
    )
    parser.add_argument(
        "-c", "--config",
        default="configs/config_moti.json",
        help="Config path",
    )
    parser.add_argument(
        "--output_path",
        default="motivation_results/adjacent_frame_attn_weights",
        help="Output directory for the plot",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=None,
        help="Override config frame_interval",
    )
    parser.add_argument(
        "--subset",
        default="short",
        help="Dataset subset (e.g. short, medium, long)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = Config(args.config)
    config.system_mode = int(config.system_mode) & ~SYSTEM_MODE_VLM_QA  # 跳过 VLM，加快 inject

    if args.frame_interval is not None:
        config.frame_interval = args.frame_interval
    frame_interval = config.frame_interval

    captured = {}
    hook = make_accumulating_attention_hook(captured, args.layer)
    moti = VragSystemMoti(config)
    moti.register_encode_hook(hook, need_attentions=True)

    # Run inject to encode video (use unique video_id so no skip)
    result = moti.inject_video(
        video_path=args.video_path,
        video_id=args.video_id,
        dataset_name="Video-MME",
        subset=args.subset,
    )

    if not captured.get("attention_list"):
        print("No attention captured. Check hook and inject flow.")
        return

    # Concatenate across batches
    attn_list = captured["attention_list"]
    frame_ids_list = captured["frame_ids_list"]
    attention_all = np.concatenate(attn_list, axis=0)  # (N, num_heads, seq_len, seq_len)
    frame_ids_all = [fid for lst in frame_ids_list for fid in lst]

    # Extract CLS-to-patch for each frame: (N, 196)
    cls_to_patch_all = cls_to_patch_attention(attention_all)

    N = cls_to_patch_all.shape[0]
    if N < 2:
        print("Need at least 2 encoded frames")
        return

    # Compute adjacent-frame cosine similarity of attention patterns
    sims = []
    x_frame_ids = []
    for i in range(N - 1):
        sim = attention_pattern_cosine_similarity(cls_to_patch_all[i], cls_to_patch_all[i + 1])
        sims.append(sim)
        x_frame_ids.append(frame_ids_all[i])

    # Plot
    plt.figure(figsize=(12, 4), dpi=150)
    plt.plot(x_frame_ids, sims, "b-", linewidth=1, label="CLS-to-patch attention similarity")

    plt.xlabel("Frame ID (original video)", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.title(
        f"Adjacent Frame Attention Similarity (CLS to patches) | "
        f"frame_interval={frame_interval}, layer={args.layer}",
        fontsize=14,
    )
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_dir = args.output_path
    base_name = f"adjacent_frame_attn_similarity_i{frame_interval}_l{args.layer}"
    output_path = os.path.join(output_dir, base_name + ".png")
    data_path = os.path.join(output_dir, base_name + ".npz")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    # Save plot data for later processing (same base name as image)
    np.savez(
        data_path,
        frame_ids=np.array(x_frame_ids),
        similarity=np.array(sims),
        frame_interval=np.int32(frame_interval),
        layer=np.int32(args.layer),
    )
    print(f"Plot saved to: {output_path}")
    print(f"Data saved to: {data_path}")
    print(f"  - Encoded frames: {N}")
    print(f"  - frame_interval: {frame_interval}")
    print(f"  - layer: {args.layer}")


if __name__ == "__main__":
    main()
