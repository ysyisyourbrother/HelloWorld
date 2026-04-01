"""
消费端（建议在 conda/symphony 环境运行）：
连接生产端暴露的 Queue，接收选择性解码得到的 RGB 帧，
调用 src/image_bge_vectorizer.py 里的 ImageBGEVectorizer.encode（不依赖 decord）。
"""

import argparse
import os
import sys
import time
from pathlib import Path
from multiprocessing.managers import BaseManager

import numpy as np
from PIL import Image

def _ensure_project_root_on_syspath():
    """
    允许在任意工作目录运行此脚本。
    优先使用环境变量 PROJECT_ROOT；否则按脚本位置向上推断仓库根目录。
    """
    if os.environ.get("PROJECT_ROOT"):
        root = Path(os.environ["PROJECT_ROOT"]).resolve()
    else:
        # .../repo/test_rtsp/this_file.py -> repo 根目录
        root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_project_root_on_syspath()

from src.memory.image_bge_vectorizer import ImageBGEVectorizer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="RTSP 选择性解码消费端（连接 Queue 并 encode）")
    p.add_argument("--host", type=str, default="127.0.0.1", help="生产端 Manager host")
    p.add_argument("--port", type=int, default=50050, help="生产端 Manager port")
    p.add_argument("--authkey", type=str, default="rtsp", help="生产端 Manager authkey")
    p.add_argument("--device", type=str, default="cuda", help="推理设备：cuda/cpu 等")
    p.add_argument("--model-path", type=str, required=True, help="BGE/CLIP 模型路径（from_pretrained）")
    p.add_argument(
        "--attn-implementation",
        type=str,
        default="eager",
        choices=("sdpa", "eager", "flash_attention_2"),
        help="attn_implementation，默认 sdpa",
    )
    p.add_argument("--log-every", type=float, default=1.0, help="日志间隔(秒)，默认 1")
    p.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧，0 表示不限制")
    return p.parse_args()


class _QueueManager(BaseManager):
    pass


if __name__ == "__main__":
    args = parse_args()

    _QueueManager.register("get_queue")
    mgr = _QueueManager(address=(args.host, args.port), authkey=args.authkey.encode("utf-8"))
    mgr.connect()
    q = mgr.get_queue()

    vec = ImageBGEVectorizer(
        device=args.device,
        model_path=args.model_path,
        attn_implementation=args.attn_implementation,
    )

    t0 = time.monotonic()
    last = t0
    recv = 0
    processed = 0

    print(f"[INFO] Consumer 已连接：{args.host}:{args.port}，开始接收并 encode…")
    while True:
        item = q.get()  # 阻塞等待
        recv += 1

        w = int(item["w"])
        h = int(item["h"])
        rgb = item["rgb"]  # bytes
        # 安全起见：按长度裁剪/校验
        expect = w * h * 3
        if len(rgb) < expect:
            continue
        if len(rgb) != expect:
            rgb = rgb[:expect]

        img = Image.fromarray(np.frombuffer(rgb, dtype=np.uint8).reshape(h, w, 3), mode="RGB")
        _ = vec.encode(img)
        processed += 1

        if args.max_frames > 0 and processed >= args.max_frames:
            break

        now = time.monotonic()
        if now - last >= args.log_every:
            dt = now - last
            total = now - t0
            print(
                f"[STATS] recv={recv}, processed={processed}, "
                f"proc_fps={processed/total:.2f}, interval={dt:.1f}s"
            )
            last = now

