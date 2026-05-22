"""可视化视频各帧的 pkt_size（按 I/P/B 类型区分）"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 添加 src 到路径以便导入 ffprobe_utils
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from src.video_utils.ffprobe_utils import get_frame_info_with_pkt_size, get_i_frame_indices, get_frame_types

# I=red, P=blue, B=purple（与参考脚本一致）
P_TYPE_COLOR = {"I": "red", "P": "gray", "B": "#C4C4C4"}
P_TYPE_COLOR_EDGE = {"I": "red", "P": "gray", "B": "gray"}


def parse_args():
    parser = argparse.ArgumentParser(description="可视化视频各帧的 pkt_size（按 I/P/B 类型）")
    parser.add_argument("--video_path", 
        type=str, 
        # default="local_datasets/Video-MME/data/44ivpEIcBhE.mp4", 
        default="test_rtsp/recordings/segment_00000.mp4", 
        help="视频文件路径")
    parser.add_argument("-o", "--output", type=str, default="moti_ffprobe_frame_pkt_size.png",
                        help="输出图片路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_path = args.video_path
    output_path = args.output

    if not Path(video_path).exists():
        print(f"错误: 视频文件不存在: {video_path}")
        sys.exit(1)

    print("正在提取帧信息（pict_type, pkt_pts_time, pkt_size）...")
    import time
    t1 = time.time()
    frames_info = get_frame_info_with_pkt_size(video_path)
    t2 = time.time()
    print(f"帧信息提取耗时: {t2 - t1:.3f} 秒")

    if not frames_info:
        print("未找到帧数据")
        sys.exit(1)

    pts = np.array([f["pkt_pts_time"] for f in frames_info])
    pkt_sizes = np.array([f["pkt_size"] for f in frames_info])
    p_types = [f["pict_type"] for f in frames_info]
    colors = [P_TYPE_COLOR.get(pt, "gray") for pt in p_types]

    # 柱状图参数（参考 moti_ffprobe_IPB_info_visualize.py）
    widths = np.diff(pts)
    if len(widths) > 0:
        widths = np.append(widths, widths[-1])
    else:
        widths = np.array([0.04])
    lefts = pts
    heights = np.ones(len(frames_info))

    fig, ax1 = plt.subplots(figsize=(14, 6), dpi=150)
    ax2 = ax1.twinx()

    # ax1: 折线图 - 横轴时间，纵轴 pkt_size，按 I/P/B 着色
    ax1.plot(pts, pkt_sizes, color="gray", linewidth=0.5, alpha=0.6, zorder=1)
    for pt in ["I", "P", "B"]:
        mask = np.array([p == pt for p in p_types])
        if np.any(mask):
            ax1.scatter(pts[mask], pkt_sizes[mask], c=P_TYPE_COLOR[pt], edgecolors=P_TYPE_COLOR_EDGE[pt], s=8, zorder=2, label=pt)

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("pkt_size (bytes)", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.legend(loc="upper right")

    # ax2: 柱状图，ylim 0-10，柱子高度为 1
    ax2.bar(lefts, heights, width=widths, align="edge", color=colors, linewidth=0, alpha=0.7)
    ax2.set_ylim(0, 20)
    ax2.set_ylabel("")
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.xaxis.set_major_locator(MultipleLocator(5))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"图表已保存至 {output_path}（共 {len(frames_info)} 帧）")
