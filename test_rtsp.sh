#!/usr/bin/env bash
# 在仓库根目录执行: bash test_rtsp.sh
# 需 apt 安装 python3-gi、python3-gst-1.0、gstreamer1.0-plugins-bad 等（装在系统 Python 上）。
# conda 环境里的 python3 通常没有 gi，故 GStreamer 脚本必须用 /usr/bin/python3（可通过环境变量 PY_SYS 覆盖）。
# 下面两条均为长驻任务，请只保留/取消注释**其中一条**再运行，或复制单行到终端执行。

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY_SYS="${PY_SYS:-/usr/bin/python3}"
RTSP_URL="rtsp://admin:smc123456@192.168.123.98:554/stream1"

# 1) GStreamer + nvv4l2decoder 硬件解码，不落盘（Ctrl+C 退出）
# XVFB_ARGS=""
# if command -v Xvfb >/dev/null 2>&1; then
#     XVFB_ARGS="--xvfb"
# else
#     echo "[WARN] 系统未安装 Xvfb：将不传 --xvfb，若仍报 EGL/display 错误请安装 xvfb" >&2
# fi
# "$PY_SYS" test_rtsp/rtsp_decode_hw.py -u "$RTSP_URL" --tcp $XVFB_ARGS

# 2) RTSP 按约 1 分钟分段保存为 MP4（输出目录可改；Ctrl+C 退出）
# "$PY_SYS" test_rtsp/rtsp_segment_record.py -u "$RTSP_URL" -o ./test_rtsp/recordings --tcp

# 3) 按 IPB 类型统计（Ctrl+C 退出）
/usr/bin/python3 test_rtsp/rtsp_record_IPB_info_simple.py \
  -u "$RTSP_URL" \
  --tcp \
  --latency 200 \
  --relaxed-caps \
  --xvfb

# 4) 打印码率
# /usr/bin/python3 test_rtsp/rtsp_print_bitrate.py -u "$RTSP_URL" --tcp