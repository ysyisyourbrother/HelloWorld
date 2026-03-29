"""
Jetson Orin：GStreamer + nvv4l2decoder 硬件解码 RTSP，解码后丢弃（不保存）。

SSH/headless 环境下，nvv4l2decoder 内部可能仍会尝试创建 EGL display。
如果你看到 `nvbufsurftransform: Could not get EGL display connection`，可用 `--xvfb` 启动虚拟显示。
默认：decoder 直连 fakesink（不经过 nvvidconv）；需要 I420 转换且已有 DISPLAY 时可加 --nvvidconv。

依赖（Jetson L4T 上通常已装）：
  - gstreamer1.0-tools, libgstreamer1.0-dev
  - python3-gi, python3-gst-1.0
  若缺 PyGObject：pip install PyGObject（需系统已装 gtk/gobject）
"""

import argparse
import signal
import sys
import time
import os
import atexit
import subprocess
import shutil

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="RTSP 硬件解码（nvv4l2decoder），仅消费帧不保存"
    )
    parser.add_argument(
        "--url",
        "-u",
        type=str,
        required=True,
        help="RTSP 地址，例如 rtsp://user:pass@host:554/stream1",
    )
    parser.add_argument(
        "--codec",
        "-c",
        choices=("h264", "h265"),
        default="h264",
        help="码流类型（与摄像头一致）。默认: h264",
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
        help="使用 RTSP over TCP（不稳定网络时建议开启）",
    )
    parser.add_argument(
        "--nvvidconv",
        action="store_true",
        help="在 decoder 后接 nvvidconv 转为 I420（需可用 EGL/显示；SSH 无 DISPLAY 时不要开）",
    )
    parser.add_argument(
        "--xvfb",
        action="store_true",
        help="无 DISPLAY/SSH headless 时自动启动 Xvfb（用于修复 EGL display connection）",
    )
    return parser.parse_args()


def build_decode_pipeline(
    codec: str, latency: int, use_tcp: bool, use_nvvidconv: bool
) -> str:
    if codec == "h264":
        depay = "rtph264depay ! h264parse"
    else:
        depay = "rtph265depay ! h265parse"
    if use_tcp:
        head = f"rtspsrc name=src latency={latency} protocols=tcp ! "
    else:
        head = f"rtspsrc name=src latency={latency} ! "
    # 注意：不同 GStreamer 版本的 fakesink 属性不完全一致。
    # 你当前报错表明没有 drop 属性，因此这里不再设置 drop=true。
    sink = "fakesink sync=false async=false name=sink"
    if use_nvvidconv:
        mid = f"nvv4l2decoder ! nvvidconv ! video/x-raw,format=I420 ! {sink}"
    else:
        mid = f"nvv4l2decoder ! {sink}"
    return f"{head}{depay} ! {mid}"


if __name__ == "__main__":
    args = parse_args()
    xfvb_proc = None
    try:
        # headless 情况下：给 EGL 找一个“可用的 display”。
        if args.xvfb and not os.environ.get("DISPLAY"):
            display = ":99"
            # 注意：Xvfb 需要系统已安装；若没安装会抛 FileNotFoundError。
            if shutil.which("Xvfb") is None:
                print(
                    "[ERROR] 启动 --xvfb 失败：系统找不到 Xvfb 可执行文件。",
                    file=sys.stderr,
                )
                print(
                    "请先安装：sudo apt update && sudo apt install -y xvfb",
                    file=sys.stderr,
                )
                sys.exit(1)
            xfvb_proc = subprocess.Popen(
                ["Xvfb", display, "-screen", "0", "1280x720x24"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            atexit.register(lambda: xfvb_proc and xfvb_proc.terminate())
            time.sleep(1.0)
            os.environ["DISPLAY"] = display

        Gst.init(None)
        pipeline_str = build_decode_pipeline(
            args.codec, args.latency, args.tcp, args.nvvidconv
        )
        try:
            pipeline = Gst.parse_launch(pipeline_str)
        except Exception as e:
            print(f"[ERROR] 无法创建管线: {e}", file=sys.stderr)
            print(f"管线字符串: {pipeline_str}", file=sys.stderr)
            sys.exit(1)

        src = pipeline.get_by_name("src")
        src.set_property("location", args.url)

        stats = {"frames": 0, "t0": time.monotonic()}

        def on_buffer(_pad, info, _data):
            stats["frames"] += 1
            return Gst.PadProbeReturn.OK

        sink = pipeline.get_by_name("sink")
        pad = sink.get_static_pad("sink")
        pad.add_probe(Gst.PadProbeType.BUFFER, on_buffer, None)

        def print_fps():
            t1 = time.monotonic()
            dt = t1 - stats["t0"]
            fps = stats["frames"] / dt if dt > 0 else 0.0
            print(
                f"[STATS] 约 {fps:.2f} fps（累计 {stats['frames']} 帧，{dt:.1f}s）"
            )
            stats["frames"] = 0
            stats["t0"] = t1
            return True

        GLib.timeout_add_seconds(1, print_fps)

        loop = GLib.MainLoop()

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
            print("\n[STOP] 中断，正在停止管线…")
            shutdown()

        signal.signal(signal.SIGINT, on_sigint)

        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print(
                "[ERROR] PLAYING 失败（检查 URL、编码、nvv4l2decoder 是否可用）",
                file=sys.stderr,
            )
            sys.exit(1)

        print("[INFO] 解码中（Ctrl+C 退出）…")
        try:
            loop.run()
        finally:
            pipeline.set_state(Gst.State.NULL)
    finally:
        if xfvb_proc is not None:
            xfvb_proc.terminate()
