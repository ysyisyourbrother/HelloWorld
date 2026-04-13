import subprocess
import json
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# video_path = "local_datasets/Video-MME/data/44ivpEIcBhE.mp4"
video_path = "test_rtsp/recordings/segment_00000.mp4"
output_json = "motion_vectors.json"
output_plot = "moti_ffprobe_frame_type.png"

# I=red, P=blue, B=purple
P_TYPE_COLOR = {"I": "red", "P": "white", "B": "gray"}

print("Extracting frame type data...")

cmd = [
    "ffprobe",
    "-v", "quiet",
    "-print_format", "json",
    "-select_streams", "v:0",
    "-show_entries", "frame=pict_type,pkt_pts_time",
    video_path
]

try:
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True
    )
    data = json.loads(result.stdout)
    frames = data.get("frames", [])

    if not frames:
        print("No frames found.")
        exit(1)

    pts = np.array([float(f.get("pkt_pts_time", 0)) for f in frames])
    p_types = [f.get("pict_type", "?") for f in frames]

    # Bar width: time span until next frame; last frame uses same as prev
    widths = np.diff(pts)
    if len(widths) > 0:
        widths = np.append(widths, widths[-1])
    else:
        widths = np.array([0.04])  # fallback

    lefts = pts
    heights = np.ones(len(frames))
    colors = [P_TYPE_COLOR.get(pt, "gray") for pt in p_types]

    # Statistics: I, P, B counts and total
    counts = Counter(p_types)
    n_i = counts.get("I", 0)
    n_p = counts.get("P", 0)
    n_b = counts.get("B", 0)
    n_total = len(frames)
    stats_text = f"I: {n_i}\nP: {n_p}\nB: {n_b}\nTotal: {n_total}"

    fig, ax = plt.subplots(figsize=(14, 2), dpi=150)
    ax.bar(lefts, heights, width=widths, align="edge", color=colors, linewidth=0)

    # X-axis ticks every 5 seconds
    ax.xaxis.set_major_locator(MultipleLocator(5))

    # Red text above I-frame bars: "No.xxx / xxx s", rotation 90
    for i, (pt, left, w) in enumerate(zip(p_types, lefts, widths)):
        if pt == "I":
            center_x = left + w / 2
            ax.text(
                center_x, 1.05, f"No.{i} / {pts[i]:.2f} s",
                color="red", rotation=90, ha="center", va="bottom",
                fontsize=7,
            )

    # Statistics on the left
    fig.text(0.06, 0.5, stats_text, fontsize=10, va="center", family="monospace")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylim(0, 2.0)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylabel("")
    ax.set_title("")
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(rect=[0.1, 0, 1, 1])
    plt.savefig(output_plot)
    plt.close()
    print(f"Plot saved to {output_plot} ({len(frames)} frames)")

    with open(output_json, "w") as f:
        json.dump(data, f)
    print(f"Data saved to {output_json}")

except subprocess.CalledProcessError as e:
    print("Error:", e.stderr)
except Exception as e:
    print("Parse error:", e)