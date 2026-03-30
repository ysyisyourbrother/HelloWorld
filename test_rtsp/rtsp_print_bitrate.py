"""
RTSP 拉取 H.264，经 GStreamer 解 RTP 后在 appsink 统计缓冲字节数，不解码视频。

每秒打印：过去 1 秒内的平均码率（bit/s）与累计接收字节数。

管线与 rtsp_record_IPB_info_simple 一致：手动链接 + pad-added 选 H.264 视频轨，
其它 RTP（如音频）接 fakesink；总线处理 BUFFERING 以便直播场景正常进入 PLAYING。

依赖：GStreamer、python3-gi（PyGObject），Jetson Orin 上通常已具备。
"""

import argparse
import atexit
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Optional, Tuple

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp, GLib  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="RTSP H.264 实时码率与累计字节（Gst 拉流）")
    p.add_argument("--url", "-u", type=str, required=True, help="RTSP 地址（需为 H.264 视频轨）")
    p.add_argument("--latency", type=int, default=200, help="rtspsrc 缓冲延迟（毫秒），默认 200")
    p.add_argument("--tcp", action="store_true", help="使用 RTSP over TCP")
    p.add_argument(
        "--xvfb",
        action="store_true",
        help="若无 DISPLAY 则启动虚拟显示 Xvfb（无头 Jetson 等）",
    )
    p.add_argument(
        "--relaxed-caps",
        action="store_true",
        help="不强制 h264parse alignment=au（与 IPB 脚本一致，部分 IPC 需开启）",
    )
    return p.parse_args()


def _start_xvfb_if_needed(enable: bool):
    if not enable:
        return None
    if os.environ.get("DISPLAY"):
        return None
    if shutil.which("Xvfb") is None:
        print(
            "[ERROR] 已指定 --xvfb 但未找到 Xvfb，请安装：sudo apt install -y xvfb",
            file=sys.stderr,
        )
        sys.exit(1)
    display = ":99"
    proc = subprocess.Popen(
        ["Xvfb", display, "-screen", "0", "1280x720x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(lambda: proc.terminate())
    time.sleep(1.0)
    os.environ["DISPLAY"] = display
    return proc


def _appsink_pull_sample(sink) -> Optional[object]:
    ps = getattr(sink, "pull_sample", None)
    if callable(ps):
        return ps()
    cast = getattr(GstApp.AppSink, "cast", None)
    if callable(cast):
        return cast(sink).pull_sample()
    return sink.emit("pull-sample")


def _caps_looks_h264_rtp_video(caps: Gst.Caps) -> bool:
    if caps is None or caps.is_empty():
        return False
    st = caps.get_structure(0)
    if st.get_name() != "application/x-rtp":
        return False
    s = caps.to_string().upper()
    if "H264" in s or "h264" in caps.to_string():
        return True
    if st.has_field("media") and st.has_field("clock-rate"):
        ok_m, media = st.get_string("media")
        ok_c, cr = st.get_int("clock-rate")
        if ok_m and ok_c and media == "video" and cr == 90000:
            return True
    return False


def _human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n / (1024 * 1024):.2f} MiB"


def _build_rtsp_h264_bitrate_pipeline(args) -> Tuple[Gst.Pipeline, Gst.Element]:
    pipeline = Gst.Pipeline.new("bitrate-pipe")
    src = Gst.ElementFactory.make("rtspsrc", "src")
    depay = Gst.ElementFactory.make("rtph264depay", "depay")
    h264p = Gst.ElementFactory.make("h264parse", "h264p")
    if src is None or depay is None or h264p is None:
        raise RuntimeError("无法创建 rtspsrc/rtph264depay/h264parse（请检查 GStreamer 插件）")

    queue = Gst.ElementFactory.make("queue", "q")
    cf = Gst.ElementFactory.make("capsfilter", "cf")
    parsink = Gst.ElementFactory.make("appsink", "parsink")
    if queue is None or cf is None or parsink is None:
        raise RuntimeError("无法创建 queue/capsfilter/appsink")

    if args.relaxed_caps:
        cf.set_property(
            "caps", Gst.Caps.from_string("video/x-h264,stream-format=byte-stream")
        )
    else:
        cf.set_property(
            "caps",
            Gst.Caps.from_string(
                "video/x-h264,stream-format=byte-stream,alignment=au"
            ),
        )

    for e in (src, depay, h264p, queue, cf, parsink):
        pipeline.add(e)

    if not depay.link(h264p):
        raise RuntimeError("rtph264depay -> h264parse 链接失败")
    if not h264p.link(queue):
        raise RuntimeError("h264parse -> queue 链接失败")
    if not queue.link(cf):
        raise RuntimeError("queue -> capsfilter 链接失败")
    if not cf.link(parsink):
        raise RuntimeError("capsfilter -> appsink 链接失败")

    src.set_property("location", args.url)
    src.set_property("latency", args.latency)
    if args.tcp:
        try:
            rlt = getattr(Gst, "RTSPLowerTrans", None)
            val = getattr(rlt, "TCP", 4) if rlt is not None else 4
            src.set_property("protocols", val)
        except Exception:
            try:
                src.set_property("protocols", 4)
            except Exception:
                pass
    try:
        src.set_property("drop-on-latency", True)
    except Exception:
        pass

    try:
        h264p.set_property("config-interval", -1)
    except Exception:
        pass

    parsink.set_property("emit-signals", True)
    parsink.set_property("sync", False)
    parsink.set_property("max-buffers", 8 if args.relaxed_caps else 1)
    parsink.set_property("drop", True)

    _audio_fakesink_idx = [0]

    def _resolve_pad_caps(pad: Gst.Pad) -> Optional[Gst.Caps]:
        caps = pad.get_current_caps()
        if caps is None or caps.is_empty():
            caps = pad.query_caps(None)
        if caps is None or caps.is_empty():
            return None
        return caps

    def _try_link_rtspsrc_pad(pad: Gst.Pad) -> None:
        if pad.is_linked():
            return
        caps = _resolve_pad_caps(pad)
        if caps is None or caps.is_empty():
            return
        if _caps_looks_h264_rtp_video(caps):
            sinkpad = depay.get_static_pad("sink")
            pad.link(sinkpad)
            return

        st = caps.get_structure(0)
        if st.get_name() != "application/x-rtp":
            return

        media = None
        if st.has_field("media"):
            _ok_m, media = st.get_string("media")
        enc = None
        if st.has_field("encoding-name"):
            _ok_e, enc = st.get_string("encoding-name")
        if (
            media == "video"
            and enc
            and enc.upper() not in ("H264", "H264-1998", "H264-2000")
        ):
            print(
                f"[WARN] RTP 视频编码为 {enc!r}，本脚本仅统计 H.264；"
                "HEVC 等需换用对应 depay 管线。",
                file=sys.stderr,
                flush=True,
            )

        idx = _audio_fakesink_idx[0]
        _audio_fakesink_idx[0] = idx + 1
        fake = Gst.ElementFactory.make("fakesink", f"drop_{idx}")
        if fake is None:
            return
        fake.set_property("sync", False)
        fake.set_property("async", False)
        pipeline.add(fake)
        fake.sync_state_with_parent()
        sp = fake.get_static_pad("sink")
        pad.link(sp)

    def on_notify_caps(pad: Gst.Pad, _pspec):
        if pad.is_linked():
            return
        _try_link_rtspsrc_pad(pad)

    def on_pad_added(_element, pad: Gst.Pad):
        caps = _resolve_pad_caps(pad)
        if caps is None or caps.is_empty():
            pad.connect("notify::caps", on_notify_caps)
            return
        _try_link_rtspsrc_pad(pad)

    src.connect("pad-added", on_pad_added)

    return pipeline, parsink


if __name__ == "__main__":
    args = parse_args()
    _start_xvfb_if_needed(args.xvfb)
    Gst.init(None)

    try:
        pipeline, parsink = _build_rtsp_h264_bitrate_pipeline(args)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    lock = threading.Lock()
    stats = {"last_sec_bytes": 0, "total_bytes": 0}
    stop_flag = {"stop": False}
    t0 = time.monotonic()
    loop = GLib.MainLoop()

    def print_second_stats():
        with lock:
            b_sec = stats["last_sec_bytes"]
            total = stats["total_bytes"]
            stats["last_sec_bytes"] = 0
        elapsed = time.monotonic() - t0
        bps = b_sec * 8
        mbps = bps / 1e6
        print(
            f"[{elapsed:7.1f}s] 本秒平均码率 {bps:,.0f} bit/s ({mbps:.3f} Mbps) | "
            f"本秒数据 {_human_bytes(b_sec)} | 累计 {total:,} 字节 ({_human_bytes(total)})",
            flush=True,
        )
        return not stop_flag["stop"]

    def on_bus(_bus, msg):
        if msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"[ERROR] {err}", file=sys.stderr)
            if dbg:
                print(dbg, file=sys.stderr)
            stop_flag["stop"] = True
            loop.quit()
        elif msg.type == Gst.MessageType.EOS:
            print("[INFO] EOS", flush=True)
            stop_flag["stop"] = True
            loop.quit()
        elif msg.type == Gst.MessageType.BUFFERING:
            st = msg.get_structure()
            if st and st.has_field("buffer-percent"):
                ok, pct = st.get_int("buffer-percent")
                if ok and pct >= 100:
                    pipeline.set_state(Gst.State.PLAYING)
        elif msg.type == Gst.MessageType.WARNING:
            warn, dbg = msg.parse_warning()
            print(f"[WARN] {warn}", file=sys.stderr, flush=True)
            if dbg:
                print(dbg, file=sys.stderr, flush=True)
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus)

    def on_new_sample(sink):
        if stop_flag["stop"]:
            return Gst.FlowReturn.EOS
        sample = _appsink_pull_sample(sink)
        if sample is None:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        n = buf.get_size()
        with lock:
            stats["last_sec_bytes"] += n
            stats["total_bytes"] += n
        return Gst.FlowReturn.OK

    parsink.connect("new-sample", on_new_sample)

    def on_sigint(_sig, _frame):
        print("\n[STOP] 中断，正在停止…", flush=True)
        stop_flag["stop"] = True
        loop.quit()

    signal.signal(signal.SIGINT, on_sigint)

    GLib.timeout_add(1000, print_second_stats)

    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("pipeline PLAYING 失败", file=sys.stderr)
        sys.exit(1)

    try:
        loop.run()
    finally:
        stop_flag["stop"] = True
        pipeline.set_state(Gst.State.NULL)
        with lock:
            total = stats["total_bytes"]
            rem = stats["last_sec_bytes"]
        print(
            f"[结束] 累计接收 {total:,} 字节 ({_human_bytes(total)})；"
            f"未满一秒区间约 {rem:,} 字节",
            flush=True,
        )
