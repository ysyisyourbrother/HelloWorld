"""
RTSP 拉取 H.264，经 GStreamer 解 RTP、parse 得到访问单元（AU），不解码视频。

统计方式：解析每个 AU 内的 H.264 NAL（含 slice_header 里的 slice_type），归类为
IDR / I / P / B / 其他；与解码器输出的帧类型在常见码流上一致，极端或损坏码流可能判为「其他」。

管线要点：
- 使用「手动链接 + rtspsrc 的 pad-added」而不是一条 parse_launch 字符串，因为多数 RTSP
  源同时有视频与音频等多条 RTP 流，线性链容易接错 pad，导致 appsink 永远无数据。
- 非 H.264 的 RTP（常见为音频）接到 fakesink 丢弃，否则部分相机会因 pad 无人消费而阻塞。
- 总线上处理 BUFFERING：直播场景下缓冲满后再 set PLAYING，避免长期卡在 PAUSED。

依赖：系统已安装 GStreamer 与 python3-gi（PyGObject），运行方式与 rtsp_selective_decode_producer_gst 等脚本一致。
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
from typing import List, Optional, Tuple

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp, GLib  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="RTSP H.264 I/P/B 计数（解析轨，不解码）")
    p.add_argument("--url", "-u", type=str, required=True, help="RTSP 地址（需为 H.264 视频轨）")
    p.add_argument("--latency", type=int, default=200, help="rtspsrc 缓冲延迟（毫秒），默认 200")
    p.add_argument("--tcp", action="store_true", help="使用 RTSP over TCP（弱网/防火墙下更稳）")
    p.add_argument(
        "--xvfb",
        action="store_true",
        help="若无 DISPLAY 则启动虚拟显示 Xvfb（无头 Jetson 等环境可与其它脚本行为一致）",
    )
    p.add_argument(
        "--relaxed-caps",
        action="store_true",
        help=(
            "不强制 h264parse 输出 alignment=au；部分 IPC 无法协商 AU 对齐，强制后 appsink 可能永远无缓冲。"
            "开启后仍要求 byte-stream；单 buffer 可能仅为单个 NAL，分类逻辑已兼容无起始码前缀的裸 NAL。"
        ),
    )
    return p.parse_args()


def _start_xvfb_if_needed(enable: bool):
    """按需启动 Xvfb 并设置 DISPLAY；若已有 DISPLAY 则不做任何事。"""
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
    # 屏幕配置：-screen 0 1280x720x24 定义了一个分辨率为 1280x720，色深为 24位 的虚拟屏幕。
    # 静默运行：将标准输出和错误输出重定向到空设备，避免日志污染。
    proc = subprocess.Popen(
        ["Xvfb", display, "-screen", "0", "1280x720x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # 使用 atexit 模块注册一个回调函数
    # 当主程序退出时，自动终止 Xvfb 进程，防止产生僵尸进程。
    atexit.register(lambda: proc.terminate())
    time.sleep(1.0)
    os.environ["DISPLAY"] = display
    return proc


# def _find_start_code(data: bytes, pos: int) -> Tuple[int, int]:
#     """在 pos 之后查找 Annex B 起始码，返回 (起始下标, 起始码长度 3 或 4)。"""
#     idx4 = data.find(b"\x00\x00\x00\x01", pos)
#     idx3 = data.find(b"\x00\x00\x01", pos)
#     if idx4 == -1:
#         return (idx3, 3) if idx3 != -1 else (-1, 0)
#     if idx3 == -1:
#         return (idx4, 4)
#     return (idx4, 4) if idx4 <= idx3 else (idx3, 3)


# def _iter_nal_units(au: bytes) -> List[bytes]:
#     """
#     将 Annex B 字节流拆成多个 NAL RBSP 片段（每段含 1 字节 NAL 头 + RBSP，不含起始码）。
#     若 au 中无起始码则返回空列表，由上层按「整段为单 NAL」处理。
#     """
#     def _find_start_code(data: bytes, pos: int) -> Tuple[int, int]:
#         """在 pos 之后查找 Annex B 起始码，返回 (起始下标, 起始码长度 3 或 4)。"""
#         idx4 = data.find(b"\x00\x00\x00\x01", pos)
#         idx3 = data.find(b"\x00\x00\x01", pos)
#         if idx4 == -1:
#             return (idx3, 3) if idx3 != -1 else (-1, 0)
#         if idx3 == -1:
#             return (idx4, 4)
#         return (idx4, 4) if idx4 <= idx3 else (idx3, 3)
#     n = len(au)
#     out: List[bytes] = []
#     i = 0
#     while i < n:
#         s, sl = _find_start_code(au, i)
#         if s == -1:
#             break
#         nal_start = s + sl
#         if nal_start >= n:
#             break
#         s2, _ = _find_start_code(au, nal_start + 1)
#         nal_end = s2 if s2 != -1 else n
#         chunk = au[nal_start:nal_end]
#         if chunk:
#             out.append(chunk)
#         i = nal_end if s2 != -1 else n
#     return out


# def _rbsp_unescape(rbsp: bytes) -> bytes:
#     """
#     去除 H.264 RBSP 中的 emulation prevention（0x00 0x00 0x03 后跟 0x00–0x03 时删除 0x03）。
#     必须在读 slice_header 之前做，否则 Exp-Golomb 会错位。
#     """
#     b = bytearray()
#     i = 0
#     L = len(rbsp)
#     while i < L:
#         if (
#             i + 3 < L
#             and rbsp[i] == 0
#             and rbsp[i + 1] == 0
#             and rbsp[i + 2] == 3
#             and rbsp[i + 3] <= 3
#         ):
#             b.extend(rbsp[i : i + 2])
#             i += 4
#         else:
#             b.append(rbsp[i])
#             i += 1
#     return bytes(b)


class _BitReader:
    """按位读取 RBSP，用于解析 slice_header 中的 ue(v)（如 slice_type）。"""

    def __init__(self, data: bytes):
        self._d = data
        self._bit = 0

    def read_bit(self) -> int:
        bi = self._bit // 8
        if bi >= len(self._d):
            return 0
        o = self._bit % 8
        v = (self._d[bi] >> (7 - o)) & 1
        self._bit += 1
        return v

    def read_ue(self) -> Optional[int]:
        """H.264 ue(v)：leading_zero_bits + 后缀位，与规范一致。"""
        z = 0
        while True:
            b = self.read_bit()
            if b != 0:
                break
            z += 1
            if z > 31:
                return None
        code = (1 << z) - 1
        for _ in range(z):
            code = (code << 1) | self.read_bit()
        return code


# def _slice_type_to_ipb(slice_type: int) -> str:
#     """H.264 Table 7-6：用 slice_type % 5 映射到 P/B/I 等。"""
#     m = slice_type % 5
#     if m == 0:
#         return "P"
#     if m == 1:
#         return "B"
#     if m == 2:
#         return "I"
#     if m == 3:
#         return "P"  # SP
#     if m == 4:
#         return "I"  # SI
#     return "O"


def _classify_h264_au(au: bytes) -> str:
    """
    根据 AU 内 VCL NAL 判定一帧类型，返回：
    'IDR' | 'I' | 'P' | 'B' | 'O' | 'SKIP'。

    - SKIP：仅含 SEI/SPS/PPS/AUD 等非 VCL NAL 的 buffer，不计入「帧」统计。
    - IDR：存在 type 5（IDR）且无 type 1 与之冲突（误切分可能同时出现假 5 与真 1，此时按 1 解析）。
    - I/P/B：来自 type 1 的 slice_header.slice_type。
    """
    def _iter_nal_units(au: bytes) -> List[bytes]:
        """
        将 Annex B 字节流拆成多个 NAL RBSP 片段（每段含 1 字节 NAL 头 + RBSP，不含起始码）。
        若 au 中无起始码则返回空列表，由上层按「整段为单 NAL」处理。
        """
        def _find_start_code(data: bytes, pos: int) -> Tuple[int, int]:
            """在 pos 之后查找 Annex B 起始码，返回 (起始下标, 起始码长度 3 或 4)。"""
            idx4 = data.find(b"\x00\x00\x00\x01", pos)
            idx3 = data.find(b"\x00\x00\x01", pos)
            if idx4 == -1:
                return (idx3, 3) if idx3 != -1 else (-1, 0)
            if idx3 == -1:
                return (idx4, 4)
            return (idx4, 4) if idx4 <= idx3 else (idx3, 3)
        n = len(au)
        out: List[bytes] = []
        i = 0
        while i < n:
            s, sl = _find_start_code(au, i)
            if s == -1:
                break
            nal_start = s + sl
            if nal_start >= n:
                break
            s2, _ = _find_start_code(au, nal_start + 1)
            nal_end = s2 if s2 != -1 else n
            chunk = au[nal_start:nal_end]
            if chunk:
                out.append(chunk)
            i = nal_end if s2 != -1 else n
        return out
    nals = _iter_nal_units(au)
    # relaxed 模式下 h264parse 常输出「一个 buffer = 一个 NAL」，无 00 00 01 前缀
    if not nals and len(au) >= 1:
        nals = [au]

    if not nals:
        return "O"

    # 单 NAL 且为辅助信息：不计入帧
    if len(nals) == 1 and len(nals[0]) >= 1:
        b0 = nals[0][0]
        if not (b0 & 0x80):
            nt0 = b0 & 0x1F
            if nt0 in (6, 7, 8, 9):
                return "SKIP"

    vcl: List[Tuple[int, bytes]] = []
    for nal in nals:
        if len(nal) < 1:
            continue
        b0 = nal[0]
        if b0 & 0x80:
            continue
        nt = b0 & 0x1F
        if nt in (1, 2, 3, 4, 5):
            vcl.append((nt, nal))

    has1 = any(t == 1 for t, _ in vcl)
    has5 = any(t == 5 for t, _ in vcl)

    def _from_type1(nal: bytes) -> Optional[str]:
        def _rbsp_unescape(rbsp: bytes) -> bytes:
            """
            去除 H.264 RBSP 中的 emulation prevention（0x00 0x00 0x03 后跟 0x00–0x03 时删除 0x03）。
            必须在读 slice_header 之前做，否则 Exp-Golomb 会错位。
            """
            b = bytearray()
            i = 0
            L = len(rbsp)
            while i < L:
                if (
                    i + 3 < L
                    and rbsp[i] == 0
                    and rbsp[i + 1] == 0
                    and rbsp[i + 2] == 3
                    and rbsp[i + 3] <= 3
                ):
                    b.extend(rbsp[i : i + 2])
                    i += 4
                else:
                    b.append(rbsp[i])
                    i += 1
            return bytes(b)

        def _slice_type_to_ipb(slice_type: int) -> str:
            """H.264 Table 7-6：用 slice_type % 5 映射到 P/B/I 等。"""
            m = slice_type % 5
            if m == 0:
                return "P"
            if m == 1:
                return "B"
            if m == 2:
                return "I"
            if m == 3:
                return "P"  # SP
            if m == 4:
                return "I"  # SI
            return "O"

        if len(nal) < 2:
            return None
        rbsp = _rbsp_unescape(nal[1:])
        br = _BitReader(rbsp)
        if br.read_ue() is None:
            return None
        st = br.read_ue()
        if st is None:
            return None
        return _slice_type_to_ipb(st)

    if has1 and has5:
        has5 = False

    if has1:
        for nt, nal in vcl:
            if nt == 1:
                r = _from_type1(nal)
                if r is not None:
                    return r
        return "O"

    if has5:
        return "IDR"

    return "O"


def _appsink_pull_sample(sink) -> Optional[object]:
    """兼容不同 PyGObject 绑定：有的元素可直接 pull_sample，有的需 AppSink.cast。"""
    ps = getattr(sink, "pull_sample", None)
    if callable(ps):
        return ps()
    cast = getattr(GstApp.AppSink, "cast", None)
    if callable(cast):
        return cast(sink).pull_sample()
    return sink.emit("pull-sample")


def _caps_looks_h264_rtp_video(caps: Gst.Caps) -> bool:
    """判断 RTP pad 的 caps 是否描述 H.264 视频轨（用于从多路 RTP 中选出视频）。"""
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


def _build_rtsp_h264_pipeline(args) -> Tuple[Gst.Pipeline, Gst.Element]:
    """
    构建：rtspsrc →（动态链接）rtph264depay → h264parse → queue → capsfilter → appsink。

    rtspsrc 在运行时才为每条 RTP 流创建 pad，因此必须在 pad-added 回调里把「视频 RTP」
    接到 rtph264depay；其它 RTP（如音频）接到 fakesink，避免阻塞。
    """
    pipeline = Gst.Pipeline.new("ipb-pipe")
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

    # capsfilter 约束下游为 Annex B byte-stream，便于按 00 00 01 切 NAL
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

    # 不在每个关键帧重复发 SPS/PPS，减少 AU 里非 VCL NAL 比例（可按设备兼容性调整）
    # H.264 视频流需要 SPS 和 PPS（序列参数集和图像参数集）这两个“头信息”才能解码。通常 GStreamer 会在每个关键帧（I帧）前都插入一份 SPS/PPS，以确保兼容性。
    # 含义：设置为 -1 告诉 h264parse：不要主动去插入或重复发送 SPS/PPS。
    try:
        h264p.set_property("config-interval", -1)
    except Exception:
        pass

    # 允许 appsink 发出 new-sample 信号。
    # Python 代码会监听这个信号，一旦有新视频帧，就会触发回调函数进行处理。
    parsink.set_property("emit-signals", True)
    parsink.set_property("sync", False)
    parsink.set_property("max-buffers", 8 if args.relaxed_caps else 1)
    parsink.set_property("drop", True)

    # 音频等旁路 fakesink 的自增编号（闭包内可变整数）
    _audio_fakesink_idx = [0]

    def _resolve_pad_caps(pad: Gst.Pad) -> Optional[Gst.Caps]:
        # 目的：GStreamer 的 Pad（端口）上的媒体格式信息（Caps）有时不会立刻准备好。
        # 这个函数尝试通过多种方式（get_current_caps 或 query_caps）来获取 Pad 当前协商好的格式。
        # 作用：确保后续代码能拿到准确的“数据类型说明书”。
        caps = pad.get_current_caps()
        if caps is None or caps.is_empty():
            caps = pad.query_caps(None)
        if caps is None or caps.is_empty():
            return None
        return caps

    def _try_link_rtspsrc_pad(pad: Gst.Pad) -> None:
        """根据 caps 将 pad 接到 H.264 depay 或丢弃用 fakesink。"""
        # 检查是否已连接：如果这个 Pad 已经连了别的线，直接返回，避免重复连接。
        if pad.is_linked():
            return
        # 获取格式（Caps）：调用上面的辅助函数拿到格式信息。
        caps = _resolve_pad_caps(pad)
        if caps is None or caps.is_empty():
            return
        # 判断是否为 H.264 视频
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
                f"[WARN] RTP 视频编码为 {enc!r}，本脚本仅支持 H.264；"
                "若为 H.265 需换用 HEVC 管线。",
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
        """
        部分设备在 pad-added 时尚未设好 caps，此时先挂本回调；
        caps 就绪后再尝试链接（与 pad-added 中逻辑相同）。
        """
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
        pipeline, parsink = _build_rtsp_h264_pipeline(args)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    lock = threading.Lock()
    total = {"IDR": 0, "I": 0, "P": 0, "B": 0, "O": 0}
    last_sec = {"IDR": 0, "I": 0, "P": 0, "B": 0, "O": 0}
    stop_flag = {"stop": False}
    t0 = time.monotonic()
    loop = GLib.MainLoop()

    def print_second_stats():
        with lock:
            lidr = last_sec["IDR"]
            li, lp, lb, lo = last_sec["I"], last_sec["P"], last_sec["B"], last_sec["O"]
            tidr = total["IDR"]
            ti, tp, tb, to = total["I"], total["P"], total["B"], total["O"]
            last_sec["IDR"] = last_sec["I"] = last_sec["P"] = last_sec["B"] = last_sec["O"] = 0
        elapsed = time.monotonic() - t0
        print(
            f"[{elapsed:6.1f}s] 本秒 IDR={lidr} I={li} P={lp} B={lb} 其他={lo} | "
            f"累计 IDR={tidr} I={ti} P={tp} B={tb} 其他={to}",
            flush=True,
        )
        return not stop_flag["stop"]

    def on_bus(_bus, msg):
        if msg.type == Gst.MessageType.ERROR: # 错误处理
            err, dbg = msg.parse_error()
            print(f"[ERROR] {err}", file=sys.stderr)
            if dbg:
                print(dbg, file=sys.stderr)
            stop_flag["stop"] = True
            loop.quit()
        elif msg.type == Gst.MessageType.EOS: # 播放结束处理
            print("[INFO] EOS", flush=True)
            stop_flag["stop"] = True
            loop.quit()
        elif msg.type == Gst.MessageType.BUFFERING: # 缓冲进度处理
            # 直播：缓冲未满时管线可能停在 PAUSED；满 100% 后再切 PLAYING，否则 appsink 可能一直无数据
            st = msg.get_structure()
            if st and st.has_field("buffer-percent"):
                ok, pct = st.get_int("buffer-percent")
                if ok and pct >= 100:
                    pipeline.set_state(Gst.State.PLAYING)
        elif msg.type == Gst.MessageType.WARNING: # 警告处理
            warn, dbg = msg.parse_warning()
            print(f"[WARN] {warn}", file=sys.stderr, flush=True)
            if dbg:
                print(dbg, file=sys.stderr, flush=True)
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus)

    def on_new_sample(sink):
        # 这是一个基于 GStreamer 的数据处理回调函数，
        # 专门用于从媒体流中提取、解析并统计 H.264 视频数据。
        if stop_flag["stop"]:
            return Gst.FlowReturn.EOS
        sample = _appsink_pull_sample(sink)
        if sample is None:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        
        if not ok:
            return Gst.FlowReturn.OK
        try:
            au = bytes(mapinfo.data)
        finally:
            buf.unmap(mapinfo)

        key = _classify_h264_au(au)
        if key == "SKIP":
            return Gst.FlowReturn.OK
        with lock:
            total[key] += 1
            last_sec[key] += 1
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

    # 主循环必须先运行：否则 rtspsrc 等元素的异步状态与消息无法派发，不要在 loop.run() 前长时间阻塞 get_state。
    try:
        loop.run()
    finally:
        stop_flag["stop"] = True
        pipeline.set_state(Gst.State.NULL)
        with lock:
            lidr = last_sec["IDR"]
            li, lp, lb, lo = last_sec["I"], last_sec["P"], last_sec["B"], last_sec["O"]
            tidr = total["IDR"]
            ti, tp, tb, to = total["I"], total["P"], total["B"], total["O"]
        if tidr + ti + tp + tb + to > 0:
            print(
                f"[结束] 未满一秒区间 IDR={lidr} I={li} P={lp} B={lb} 其他={lo} | "
                f"累计 IDR={tidr} I={ti} P={tp} B={tb} 其他={to}",
                flush=True,
            )
