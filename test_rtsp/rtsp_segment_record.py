"""
Jetson：GStreamer 拉取 RTSP，按固定时长分段写入文件（默认每段 1 分钟）。

说明：分段保存采用「解封装 + 复用」到 MP4（splitmuxsink），不重编码、不经 YUV，
      与硬件解码并行时负载最低；若需「先硬解再重编码」再另建管线。

依赖：同 rtsp_decode_hw.py（python3-gi、GStreamer、gst-plugins-bad 含 splitmuxsink）。
"""

import argparse
import os
import signal
import sys

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="RTSP 按时间分段保存为 MP4（不重编码）"
    )
    parser.add_argument(
        "--url",
        "-u",
        type=str,
        required=True,
        help="RTSP 地址",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=".",
        help="分段文件输出目录。默认: 当前目录",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="segment_",
        help="文件名前缀。默认: segment_",
    )
    parser.add_argument(
        "--segment-minutes",
        type=float,
        default=1.0,
        help="每段时长（分钟）。默认: 1",
    )
    parser.add_argument(
        "--codec",
        "-c",
        choices=("h264", "h265"),
        default="h264",
        help="码流类型。默认: h264",
    )
    parser.add_argument(
        "--latency",
        type=int,
        default=200,
        help="rtspsrc latency (ms)。默认: 200",
    )
    parser.add_argument(
        "--tcp",
        action="store_true",
        help="RTSP over TCP",
    )
    return parser.parse_args()


def build_segment_pipeline(codec: str, latency: int, use_tcp: bool) -> str:
    # "Depayloader"。RTSP 传输的是被切分成小包的 RTP 数据包。
    # 这个元件负责将分散的 RTP 包重新组装成完整的视频帧（NALU）
    # "Parser"。解析视频流的头部信息（SPS, PPS 等），
    # 确保码流格式符合标准，并为下游元件提供正确的胶囊（Caps）信息。
    if codec == "h264":
        depay = "rtph264depay ! h264parse"
    else:
        depay = "rtph265depay ! h265parse"
    if use_tcp:
        head = f"rtspsrc name=src latency={latency} protocols=tcp ! "
    else:
        head = f"rtspsrc name=src latency={latency} ! "
    # 最后一个特殊的 Sink 元件，能够根据设定的条件自动将输出流切割成多个文件
    return f"{head}{depay} ! splitmuxsink name=mux muxer-factory=mp4mux"


if __name__ == "__main__":
    args = parse_args()
    if args.segment_minutes <= 0:
        print("[ERROR] --segment-minutes 必须为正数", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    Gst.init(None)
    pipeline_str = build_segment_pipeline(args.codec, args.latency, args.tcp)
    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except Exception as e:
        print(f"[ERROR] 无法创建管线: {e}", file=sys.stderr)
        print(f"管线字符串: {pipeline_str}", file=sys.stderr)
        sys.exit(1)

    src = pipeline.get_by_name("src")
    src.set_property("location", args.url)

    mux = pipeline.get_by_name("mux")
    # GObject 属性在 Python 中一般为下划线形式；单位为纳秒
    ns = int(args.segment_minutes * 60 * Gst.SECOND)
    mux.set_property("max_size_time", ns)
    pattern = os.path.join(
        os.path.abspath(args.output_dir), f"{args.prefix}%05d.mp4"
    )
    mux.set_property("location", pattern)

    loop = GLib.MainLoop()
    # 处理总线消息 (错误/EOS)
    def on_bus_message(_bus, message):
        if message.type == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print(f"[ERROR] {err}", file=sys.stderr)
            if dbg:
                print(dbg, file=sys.stderr)
            loop.quit()
            return
        if message.type == Gst.MessageType.EOS:
            print("[INFO] EOS")
            loop.quit()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message)

    def shutdown():
        pipeline.set_state(Gst.State.NULL)
        loop.quit()

    def on_sigint(_sig, _frame):
        print("\n[STOP] 中断，正在 finalize 当前分段并退出…")
        shutdown()

    signal.signal(signal.SIGINT, on_sigint)

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[ERROR] PLAYING 失败", file=sys.stderr)
        sys.exit(1)

    print(
        f"[INFO] 录制中：每段约 {args.segment_minutes} 分钟，目录 {args.output_dir}，"
        f"模式 {pattern}（Ctrl+C 停止）"
    )
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
