"""
从 RTSP 拉取 H.264，经 depay/parse 得到 AU（不解码），按 NAL/slice 头统计 I/P/B，每秒打印。

实现方式对齐 test_rtsp/rtsp_selective_decode_producer_gst.py 的解析管线；帧类型由 Annex B NAL type
与首个 slice 的 slice_type（Exp-Golomb）推断，与解码器输出在常见码流上一致，极端码流可能判为「其他」。

依赖：系统 GStreamer + python3-gi（PyGObject），与 producer 脚本相同。
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
    p = argparse.ArgumentParser(description="RTSP H.264 I/P/B 计数（GStreamer 解析轨，不解码）")
    p.add_argument("--url", "-u", type=str, required=True, help="RTSP 地址（H.264）")
    p.add_argument("--latency", type=int, default=200, help="rtspsrc latency(ms)，默认 200")
    p.add_argument("--tcp", action="store_true", help="使用 RTSP over TCP")
    p.add_argument(
        "--xvfb",
        action="store_true",
        help="若无 DISPLAY 则启动 Xvfb（Jetson headless / 无显示时与 producer 一致）",
    )
    p.add_argument(
        "--verbose-bus",
        action="store_true",
        help="打印 BUFFERING / 状态变化，便于排查「有计时无帧」",
    )
    p.add_argument(
        "--relaxed-caps",
        action="store_true",
        help=(
            "不强制 h264parse 输出 alignment=au（部分 IPC 无法协商该 caps，会导致 appsink 永远无缓冲）。"
            "仍要求 byte-stream；单 buffer 可能为单 NAL，分类器会兼容无 00 00 01 前缀的裸 NAL。"
        ),
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="调试：pad/caps、notify::caps 补链、BUFFERING/WARNING、定期管线状态（可配合 --verbose-bus）",
    )
    p.add_argument(
        "--gst-debug",
        type=str,
        metavar="LEVEL",
        default=None,
        help="在 Gst.init 之前设置环境变量 GST_DEBUG（如 2、rtspsrc:5），用于排查 RTSP 无 pad-added",
    )
    return p.parse_args()


def _gst_message_src_path(msg: Gst.Message) -> str:
    try:
        s = msg.src
        if s is None:
            return "?"
        if hasattr(s, "get_path_string"):
            return s.get_path_string()
        return s.get_name()
    except Exception:
        return "?"


def _try_set_rtspsrc_io_timeout(src_el: Gst.Element, seconds: int) -> None:
    """尽量缩短「连不上却永远无 ERROR」的挂起；不同版本属性名/单位可能不同。"""
    us = int(seconds * 1_000_000)
    ns = int(seconds * Gst.SECOND)
    for prop, val in (("timeout", us), ("tcp-timeout", ns)):
        try:
            src_el.set_property(prop, val)
        except Exception:
            continue


def _mask_rtsp_url_for_log(url: str) -> str:
    """日志中隐藏 rtsp://user:pass@ 中的密码。"""
    if "@" not in url or "://" not in url:
        return url
    try:
        scheme, rest = url.split("://", 1)
        if "@" not in rest:
            return url
        cred_host = rest.split("@", 1)
        if len(cred_host) != 2:
            return url
        cred, host = cred_host
        if ":" in cred:
            user = cred.split(":", 1)[0]
            return f"{scheme}://{user}:***@{host}"
        return f"{scheme}://***@{host}"
    except Exception:
        return url


def _start_xvfb_if_needed(enable: bool):
    if not enable:
        return None
    if os.environ.get("DISPLAY"):
        return None
    if shutil.which("Xvfb") is None:
        print(
            "[ERROR] --xvfb 已开启，但系统找不到 Xvfb。请安装：sudo apt install -y xvfb",
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


def _find_start_code(data: bytes, pos: int) -> Tuple[int, int]:
    idx4 = data.find(b"\x00\x00\x00\x01", pos)
    idx3 = data.find(b"\x00\x00\x01", pos)
    if idx4 == -1:
        return (idx3, 3) if idx3 != -1 else (-1, 0)
    if idx3 == -1:
        return (idx4, 4)
    return (idx4, 4) if idx4 <= idx3 else (idx3, 3)


def _iter_nal_units(au: bytes) -> List[bytes]:
    """Annex B：每个 NAL 含 1 字节 nal header + RBSP（不含起始码）。"""
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


def _rbsp_unescape(rbsp: bytes) -> bytes:
    """
    去掉 Annex B emulation prevention：仅当出现 0x00 0x00 0x03 且第 4 字节为 0x00–0x03 时删 0x03。
    误删「任意 0x000003」会错位 slice_header 的 UE，易把 P 判成 I。
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


class _BitReader:
    def __init__(self, data: bytes):
        self._d = data
        self._bit = 0

    def _byte_i(self) -> int:
        return self._bit // 8

    def read_bit(self) -> int:
        if self._byte_i() >= len(self._d):
            return 0
        o = self._bit % 8
        v = (self._d[self._byte_i()] >> (7 - o)) & 1
        self._bit += 1
        return v

    def read_ue(self) -> Optional[int]:
        # H.264 ue(v)：与参考软件一致，先数 leading zero，再读 suffix
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


def _slice_type_to_ipb(slice_type: int) -> str:
    """Table 7-6：用 slice_type % 5 归类。"""
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


def _classify_h264_au(au: bytes) -> str:
    """
    返回 'IDR' | 'I' | 'P' | 'B' | 'O' | 'SKIP'。
    SKIP：仅 SEI/SPS/PPS/AUD 等非 VCL NAL 的缓冲，不计入「帧」统计。

    规范上：IDR 访问单元里 VCL 只有 type 5；非 IDR 只有 type 1（等）。
    若在 RBSP 里误切出假起始码，会同时出现「假 5」与真 type 1，此时不能「任见 5 就当 I」，
    应优先按 type 1 解析 slice_type。NAL 首字节须满足 forbidden_zero_bit==0。
    """
    nals = _iter_nal_units(au)
    # h264parse 在 relaxed 模式下常输出「单 buffer = 单 NAL」，无 Annex B 起始码
    if not nals and len(au) >= 1:
        nals = [au]

    if not nals:
        return "O"

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

    # 同时出现 5 与 1：多为误切分产生的假 IDR，按非 IDR 处理
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
    """兼容不同 PyGObject：有的无 AppSink.cast，有的元素上直接有 pull_sample。"""
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


def _build_rtsp_h264_pipeline(args) -> Tuple[Gst.Pipeline, Gst.Element, dict]:
    """
    手动组管线 + rtspsrc pad-added：多路 RTSP（视频+音频）时 parse_launch 线性链
    常无法把「视频 RTP」接到 rtph264depay，导致 appsink 永远无缓冲。
    """
    pipeline = Gst.Pipeline.new("ipb-pipe")
    src = Gst.ElementFactory.make("rtspsrc", "src")
    depay = Gst.ElementFactory.make("rtph264depay", "depay")
    h264p = Gst.ElementFactory.make("h264parse", "h264p")
    if src is None or depay is None or h264p is None:
        raise RuntimeError("无法创建 rtspsrc/rtph264depay/h264parse（缺插件？）")

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

    if args.debug:
        _try_set_rtspsrc_io_timeout(src, 45)

    try:
        h264p.set_property("config-interval", -1)
    except Exception:
        pass

    parsink.set_property("emit-signals", True)
    parsink.set_property("sync", False)
    parsink.set_property("max-buffers", 8 if args.relaxed_caps else 1)
    parsink.set_property("drop", True)

    _audio_fakesink_idx = [0]
    diag = {
        "video_linked": False,
        "pad_added_n": 0,
        "notify_caps_n": 0,
        "rtspsrc": src,
    }

    def _resolve_pad_caps(pad: Gst.Pad) -> Optional[Gst.Caps]:
        caps = pad.get_current_caps()
        if caps is None or caps.is_empty():
            caps = pad.query_caps(None)
        if caps is None or caps.is_empty():
            return None
        return caps

    def _try_link_rtspsrc_pad(pad: Gst.Pad, why: str) -> None:
        if pad.is_linked():
            return
        caps = _resolve_pad_caps(pad)
        if caps is None or caps.is_empty():
            if args.debug:
                print(
                    f"[DEBUG] pad {pad.get_name()} ({why}) caps 仍为空，依赖 notify::caps",
                    flush=True,
                )
            return

        if args.debug:
            try:
                caps_s = caps.to_string()
                if len(caps_s) > 280:
                    caps_s = caps_s[:277] + "..."
                print(f"[DEBUG] {why} pad={pad.get_name()} caps={caps_s}", flush=True)
            except Exception:
                pass

        if _caps_looks_h264_rtp_video(caps):
            sinkpad = depay.get_static_pad("sink")
            ret = pad.link(sinkpad)
            diag["video_linked"] = ret == Gst.PadLinkReturn.OK
            if args.verbose_bus or args.debug:
                print(f"[BUS] 视频 RTP pad -> rtph264depay: {ret}", flush=True)
            return

        st = caps.get_structure(0)
        if st.get_name() != "application/x-rtp":
            if args.debug:
                print(
                    f"[DEBUG] 忽略非 RTP pad: {pad.get_name()} structure={st.get_name()}",
                    flush=True,
                )
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
                f"[WARN] RTP 视频编码为 {enc!r}，本脚本仅接 H.264 depay；"
                "若为 H.265/HEVC 将不会有 H.264 AU。",
                file=sys.stderr,
                flush=True,
            )

        # 非 H264 的 RTP（如音频）：接到 fakesink，避免部分设备在 pad 未消费时阻塞
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
        ret = pad.link(sp)
        if args.verbose_bus or args.debug:
            print(f"[BUS] 旁路 RTP pad -> fakesink drop_{idx}: {ret}", flush=True)

    def on_notify_caps(pad: Gst.Pad, _pspec):
        diag["notify_caps_n"] += 1
        if pad.is_linked():
            return
        _try_link_rtspsrc_pad(pad, "notify::caps")

    def on_pad_added(_element, pad: Gst.Pad):
        diag["pad_added_n"] += 1
        if args.debug:
            print(
                f"[DEBUG] pad-added #{diag['pad_added_n']} name={pad.get_name()} "
                f"peer={'Y' if pad.get_peer() else 'N'}",
                flush=True,
            )

        caps = _resolve_pad_caps(pad)
        if caps is None or caps.is_empty():
            if args.debug:
                print(
                    "[DEBUG] pad-added 时 caps 为空，已挂 notify::caps（设备晚协商时常见）",
                    flush=True,
                )
            pad.connect("notify::caps", on_notify_caps)
            return

        _try_link_rtspsrc_pad(pad, "pad-added")

    src.connect("pad-added", on_pad_added)

    return pipeline, parsink, diag


if __name__ == "__main__":
    args = parse_args()
    _start_xvfb_if_needed(args.xvfb)
    if args.gst_debug:
        os.environ["GST_DEBUG"] = args.gst_debug
    Gst.init(None)

    try:
        pipeline, parsink, diag = _build_rtsp_h264_pipeline(args)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    rtspsrc_el = diag["rtspsrc"]
    element_dbg_left = {"n": 24}

    if args.debug:
        print(
            f"[DEBUG] 启动 url={_mask_rtsp_url_for_log(args.url)} latency_ms={args.latency} "
            f"tcp={args.tcp} relaxed_caps={args.relaxed_caps} xvfb={args.xvfb} "
            f"rtspsrc_io_timeout≈45s(若插件支持)",
            flush=True,
        )
        if args.gst_debug:
            print(f"[DEBUG] GST_DEBUG={args.gst_debug!r}", flush=True)

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
        if msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"[ERROR] {_gst_message_src_path(msg)} | {err}", file=sys.stderr)
            if dbg:
                print(dbg, file=sys.stderr)
            stop_flag["stop"] = True
            loop.quit()
        elif msg.type == Gst.MessageType.EOS:
            print("[INFO] EOS", flush=True)
            stop_flag["stop"] = True
            loop.quit()
        elif msg.type == Gst.MessageType.BUFFERING:
            # 无此处理时，部分 RTSP 管线会长期停在 PAUSED（pending=playing），appsink 永远无帧
            st = msg.get_structure()
            if st and st.has_field("buffer-percent"):
                ok, pct = st.get_int("buffer-percent")
                if ok:
                    if args.verbose_bus or args.debug:
                        extra = ""
                        if args.debug and st.has_field("avg-input-rate"):
                            ok_ar, ar = st.get_double("avg-input-rate")
                            if ok_ar:
                                extra = f" avg-in={ar:.1f}"
                        print(f"[BUS] BUFFERING {pct}%{extra}", flush=True)
                    # 直播流：不在缓冲中主动 PAUSED，仅在满缓冲后再推一次 PLAYING（与官方教程 live 建议一致）
                    if pct >= 100:
                        pipeline.set_state(Gst.State.PLAYING)
        elif msg.type == Gst.MessageType.WARNING:
            warn, dbg = msg.parse_warning()
            print(f"[WARN] {_gst_message_src_path(msg)} | {warn}", file=sys.stderr, flush=True)
            if args.debug and dbg:
                print(dbg, file=sys.stderr, flush=True)
        elif msg.type == Gst.MessageType.INFO and args.debug:
            info, dbg = msg.parse_info()
            print(f"[BUS] INFO {_gst_message_src_path(msg)} | {info}", flush=True)
            if dbg:
                print(dbg, flush=True)
        elif msg.type == Gst.MessageType.ELEMENT and args.debug:
            if element_dbg_left["n"] > 0:
                st = msg.get_structure()
                if st:
                    element_dbg_left["n"] -= 1
                    s = st.to_string()
                    if len(s) > 240:
                        s = s[:237] + "..."
                    print(f"[BUS] ELEMENT {_gst_message_src_path(msg)} {s}", flush=True)
        elif msg.type == Gst.MessageType.STATE_CHANGED and (args.verbose_bus or args.debug):
            if msg.src == pipeline:
                old, new, pending = msg.parse_state_changed()
                print(f"[BUS] STATE_CHANGED {old.value_nick}->{new.value_nick} pending={pending.value_nick}", flush=True)
            elif args.debug and msg.src == rtspsrc_el:
                old, new, pending = msg.parse_state_changed()
                print(
                    f"[BUS] rtspsrc STATE {old.value_nick}->{new.value_nick} "
                    f"pending={pending.value_nick}",
                    flush=True,
                )
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

    def _debug_periodic():
        if stop_flag["stop"]:
            return False
        _r, state, pending = pipeline.get_state(0)
        with lock:
            tsum = sum(total.values())
        elapsed = time.monotonic() - t0
        rs = rp = None
        try:
            _rv, rs, rp = rtspsrc_el.get_state(0)
        except Exception:
            pass
        rtsps = ""
        if rs is not None and rp is not None:
            rtsps = f" rtspsrc={rs.value_nick}/{rp.value_nick}"
        print(
            f"[DEBUG] +{elapsed:.1f}s pipeline={state.value_nick} pending={pending.value_nick}"
            f"{rtsps} video_linked={diag['video_linked']} pad_added={diag['pad_added_n']} "
            f"notify_caps={diag['notify_caps_n']} 累计帧计数={tsum}",
            flush=True,
        )
        return not stop_flag["stop"]

    if args.debug:
        GLib.timeout_add(5000, _debug_periodic)

    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("pipeline PLAYING 失败", file=sys.stderr)
        sys.exit(1)

    # 勿在 loop.run() 之前阻塞 get_state()：主循环未跑时 GLib 源不派发，rtspsrc 常停在 PAUSED
    # （pending=playing），会误报「10s 内未到 PLAYING」。真正状态在 loop 跑起来后才可靠。
    def _warn_if_still_not_playing():
        if stop_flag["stop"]:
            return False
        _r, state, pending = pipeline.get_state(0)
        if state != Gst.State.PLAYING:
            print(
                f"[WARN] 启动约 10s 后仍未 PLAYING（当前 {state.value_nick}，pending={pending.value_nick}）。"
                " RTSP 不可达、一直 BUFFERING，或 caps 协商失败时 appsink 不会有帧。",
                file=sys.stderr,
            )
            print(
                f"[WARN] 诊断: video_linked={diag['video_linked']} pad_added={diag['pad_added_n']} "
                f"notify_caps={diag['notify_caps_n']}",
                file=sys.stderr,
            )
            if diag["pad_added_n"] == 0:
                print(
                    "[WARN] pad_added=0：rtspsrc 从未发出 pad-added，问题在 RTSP 握手之前"
                    "（网络/554 端口/认证/相机路数/URL），与 H.264 解析链无关。",
                    file=sys.stderr,
                )
                print(
                    "[HINT] 可试: ping 与 nc -zv <IP> 554；减少其他拉流客户端；"
                    " 加参数 --gst-debug rtspsrc:5 看握手；或用 gst-launch-1.0 -v rtspsrc … ! fakesink。",
                    file=sys.stderr,
                )
            print(
                "[HINT] 若已有 pad 仍无帧：相机可能是 H.265；"
                " 可用 gst-discoverer-1.0 <url> 对照编码。",
                file=sys.stderr,
            )
        return False

    GLib.timeout_add(10_000, _warn_if_still_not_playing)

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
