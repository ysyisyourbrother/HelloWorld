#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adjacent encoded frames: cosine similarity of hidden_states at a specific layer.

Uses VenusSystemMoti to encode a video, captures hidden_states via hook,
then plots token-wise cosine similarity between adjacent encoded frames.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.venus_system_motivation import VenusSystemMoti


def make_accumulating_hidden_states_hook(captured: dict, layer_idx: int):
    """
    Create a hook that accumulates hidden_states for a specific layer across batches.
    captured["hidden_states_list"]: list of (bsz, seq_len, hidden_size)
    captured["frame_ids_list"]: list of [frame_id, ...] per batch
    """
    def hook(frame_data_list, vectors, hidden_states, attentions):
        if hidden_states is None:
            return
        # hidden_states: tuple, index 0 = embeddings, index i+1 = output of layer i
        target_idx = layer_idx + 1
        if target_idx >= len(hidden_states):
            return
        h = hidden_states[target_idx]  # (bsz, seq_len, hidden_size)
        captured.setdefault("hidden_states_list", []).append(h.copy())
        captured.setdefault("frame_ids_list", []).append(
            [fd.frame_id for fd in frame_data_list]
        )
    return hook


def token_wise_cosine_similarity(
    h_i: np.ndarray, h_j: np.ndarray, eps: float = 1e-8
) -> float:
    """
    Compute cosine similarity between corresponding tokens of two frames, then average.
    h_i, h_j: (seq_len, hidden_size)
    """
    dot = np.sum(h_i * h_j, axis=-1)
    norm_i = np.linalg.norm(h_i, axis=-1) + eps
    norm_j = np.linalg.norm(h_j, axis=-1) + eps
    cos_per_token = dot / (norm_i * norm_j)
    return float(np.mean(cos_per_token))


def cls_cosine_similarity(
    h_i: np.ndarray, h_j: np.ndarray, eps: float = 1e-8
) -> float:
    """
    Compute cosine similarity between CLS tokens (position 0) of two frames.
    h_i, h_j: (seq_len, hidden_size), CLS at index 0
    """
    cls_i = h_i[0]  # (hidden_size,)
    cls_j = h_j[0]
    dot = np.dot(cls_i, cls_j)
    norm_i = np.linalg.norm(cls_i) + eps
    norm_j = np.linalg.norm(cls_j) + eps
    return float(dot / (norm_i * norm_j))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot adjacent-frame hidden_states cosine similarity"
    )
    parser.add_argument(
        "-v", "--video_path",
        default="local_datasets/Video-MME/data/44ivpEIcBhE.mp4",
        help="Video file path",
    )
    parser.add_argument(
        "--video_id",
        default="44ivpEIcBhE_adjacent_hs",
        help="Video ID for DB (use unique ID to force fresh inject)",
    )
    parser.add_argument(
        "-l", "--layer",
        type=int,
        default=6,
        help="Layer index (0-based) to capture hidden_states",
    )
    parser.add_argument(
        "-c", "--config",
        default="configs/config_moti.json",
        help="Config path",
    )
    parser.add_argument(
        "--output_path",
        default="motivation_results/adjacent_frame_hidden_states",
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
    parser.add_argument(
        "--only_cls",
        action="store_true",
        help="Use only CLS token cosine similarity; otherwise use mean over all visual tokens",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = Config(args.config)
    config.benchmark_use_cloud = False  # skip query for faster run

    if args.frame_interval is not None:
        config.frame_interval = args.frame_interval
    frame_interval = config.frame_interval
    batch_size = getattr(config, "benchmark_batch_size", 16)

    captured = {}
    hook = make_accumulating_hidden_states_hook(captured, args.layer)
    moti = VenusSystemMoti(config)
    moti.register_encode_hook(hook, need_hidden_states=True)

    # Run inject to encode video (use unique video_id so no skip)
    result = moti.inject_video(
        video_path=args.video_path,
        video_id=args.video_id,
        dataset_name="Video-MME",
        subset=args.subset,
    )

    if not captured.get("hidden_states_list"):
        print("No hidden_states captured. Check hook and inject flow.")
        return

    # Concatenate across batches
    hs_list = captured["hidden_states_list"]
    frame_ids_list = captured["frame_ids_list"]
    hidden_states_all = np.concatenate(hs_list, axis=0)  # (N, seq_len, hidden_size)
    frame_ids_all = [fid for lst in frame_ids_list for fid in lst]

    N = hidden_states_all.shape[0]
    if N < 2:
        print("Need at least 2 encoded frames")
        return

    # Compute adjacent-frame cosine similarity
    sim_fn = cls_cosine_similarity if args.only_cls else token_wise_cosine_similarity
    sim_type = "cls" if args.only_cls else "all_tokens"
    sims = []
    x_frame_ids = []
    for i in range(N - 1):
        sim = sim_fn(hidden_states_all[i], hidden_states_all[i + 1])
        sims.append(sim)
        x_frame_ids.append(frame_ids_all[i])

    # Plot
    plt.figure(figsize=(12, 4), dpi=150)
    label = "CLS cosine similarity" if args.only_cls else "Cosine similarity (mean over tokens)"
    plt.plot(x_frame_ids, sims, "b-", linewidth=1, label=label)

    plt.xlabel("Frame ID (original video)", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    title_suffix = "CLS only" if args.only_cls else "all tokens"
    plt.title(
        f"Adjacent Frame Hidden-States Similarity ({title_suffix}) | "
        f"frame_interval={frame_interval}, layer={args.layer}",
        fontsize=14,
    )
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_dir = args.output_path
    base_name = f"adjacent_frame_hidden_similarity_i{frame_interval}_l{args.layer}_{sim_type}"
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
    print(f"  - similarity type: {sim_type}")


if __name__ == "__main__":
    main()
