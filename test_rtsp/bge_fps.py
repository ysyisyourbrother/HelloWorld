"""
使用 BGE 图像编码器对随机 RGB 图像批量推理，循环测量吞吐（每轮耗时与 batch/s）。
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _ensure_project_root_on_syspath():
    if os.environ.get("PROJECT_ROOT"):
        root = Path(os.environ["PROJECT_ROOT"]).resolve()
    else:
        root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_project_root_on_syspath()
from src.memory.image_bge_vectorizer import ImageBGEVectorizer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="BGE 图像编码随机数据吞吐测试")
    p.add_argument("--model-path", type=str, required=True, help="BGE/CLIP 模型目录（from_pretrained）")
    p.add_argument("--device", type=str, default="cuda", help="推理设备，如 cuda / cuda:0 / cpu")
    p.add_argument(
        "--attn-implementation",
        type=str,
        default="eager",
        choices=("sdpa", "eager", "flash_attention_2"),
        help="CLIPModel attn_implementation",
    )
    p.add_argument("--warmup", type=int, default=0, help="正式计时的暖机轮数（不计入 repeat）")
    p.add_argument("--repeat", type=int, default=100, help="正式计时的循环次数")
    p.add_argument("--batch", type=int, default=1, help="每轮随机生成的图像张数（batch size）")
    p.add_argument(
        "--hw",
        type=int,
        nargs=2,
        default=(224, 224),
        metavar=("H", "W"),
        help="随机图像高宽（uint8 RGB），默认 224 224",
    )
    return p.parse_args()


def _random_batch(batch: int, h: int, w: int) -> list:
    return [np.random.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(batch)]


def _sync_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


if __name__ == "__main__":
    args = parse_args()
    h, w = int(args.hw[0]), int(args.hw[1])

    vec = ImageBGEVectorizer(
        device=args.device,
        model_path=args.model_path,
        attn_implementation=args.attn_implementation,
    )

    for _ in range(args.warmup):
        frames = _random_batch(args.batch, h, w)
        _ = vec.encode_batch(frames)
    _sync_device(args.device)

    times_s = []
    for i in range(args.repeat):
        frames = _random_batch(args.batch, h, w)
        _sync_device(args.device)
        t0 = time.perf_counter()
        _ = vec.encode_batch(frames)
        _sync_device(args.device)
        dt = time.perf_counter() - t0
        times_s.append(dt)
        batch_fps = 1.0 / dt if dt > 0 else float("inf")
        print(f"round={i + 1}/{args.repeat} time_s={dt:.6f} batch_per_s={batch_fps:.4f}")

    if times_s:
        arr = np.array(times_s, dtype=np.float64)
        total_s = float(arr.sum())
        mean_s = float(arr.mean())
        std_s = float(arr.std(ddof=0))
        min_s = float(arr.min())
        max_s = float(arr.max())
        mean_batch_per_s = 1.0 / mean_s if mean_s > 0 else float("inf")
        total_batches = args.repeat
        total_images = total_batches * args.batch
        mean_img_per_s = total_images / total_s if total_s > 0 else float("inf")
        print(
            "[SUMMARY] "
            f"rounds={total_batches} batch_size={args.batch} "
            f"total_s={total_s:.6f} "
            f"time_mean_s={mean_s:.6f} time_std_s={std_s:.6f} "
            f"time_min_s={min_s:.6f} time_max_s={max_s:.6f} "
            f"mean_batch_per_s={mean_batch_per_s:.4f} mean_img_per_s={mean_img_per_s:.4f}"
        )
