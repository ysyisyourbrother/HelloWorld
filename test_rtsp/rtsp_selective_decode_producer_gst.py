"""
生产端（建议用系统 /usr/bin/python3 运行）：
1) GStreamer 拉取 RTSP（只 depay/parse，不做全量解码）
2) 参考 smart_frame_extractor 逻辑：必解 I 帧；非 I 帧按 budget 触发解码
3) 仅在触发时把“该帧(AU)”送入解码管线，得到 RGB 后通过共享队列发送给推理进程
4) 若队列满则丢弃本次帧（不阻塞，避免延迟堆积）

跨解释器/跨独立脚本共享队列：使用 multiprocessing.managers.BaseManager 暴露一个 Queue。
推理端（conda/symphony）用同样的 host/port/authkey 连接即可。
"""

import argparse
import atexit
import os
import queue
import shutil
import signal
import subprocess
import sys
import time
from multiprocessing.managers import BaseManager
from typing import Optional, Tuple

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp, GLib  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="RTSP 选择性解码生产端（GStreamer + Queue）")
    p.add_argument("--url", "-u", type=str, required=True, help="RTSP 地址（H.264）")
    p.add_argument("--latency", type=int, default=200, help="rtspsrc latency(ms)，默认 200")
    p.add_argument("--tcp", action="store_true", help="使用 RTSP over TCP")
    p.add_argument("--byte-budget", "-b", type=int, default=50000, help="非 I 帧累计字节预算，默认 50000")
    p.add_argument("--max-frames", type=int, default=0, help="最多发送的触发帧数，0 表示不限制")
    p.add_argument("--queue-size", type=int, default=8, help="队列最大长度（满则丢帧），默认 8")
    p.add_argument("--bind", type=str, default="127.0.0.1", help="Manager 绑定地址，默认 127.0.0.1")
    p.add_argument("--port", type=int, default=50050, help="Manager 端口，默认 50050")
    p.add_argument("--authkey", type=str, default="rtsp", help="Manager authkey（明文），默认 rtsp")
    p.add_argument(
        "--xvfb",
        action="store_true",
        help="若无 DISPLAY 则启动 Xvfb（用于 Jetson headless 下 EGL 相关问题）",
    )
    p.add_argument(
        "--decode-timeout-ms",
        type=int,
        default=1500,
        help="单帧解码等待超时(ms)，默认 1500",
    )
    return p.parse_args()


def _start_xvfb_if_needed(enable: bool) -> Optional[subprocess.Popen]:
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


def _h264_is_idr(au_bytes: bytes) -> bool:
    """
    从 AnnexB/byte-stream 里扫描 NALU type，存在 type=5 视为 IDR(I帧)。
    注意：这里是“尽力而为”的轻量判断，足够用于 I 帧优先触发。
    """
    data = au_bytes
    i = 0
    n = len(data)

    def find_start(pos: int) -> Tuple[int, int]:
        idx4 = data.find(b"\x00\x00\x00\x01", pos)
        idx3 = data.find(b"\x00\x00\x01", pos)
        if idx4 == -1:
            return (idx3, 3) if idx3 != -1 else (-1, 0)
        if idx3 == -1:
            return (idx4, 4)
        return (idx4, 4) if idx4 <= idx3 else (idx3, 3)

    while i < n:
        s, sl = find_start(i)
        if s == -1:
            break
        nal_start = s + sl
        if nal_start >= n:
            break
        # NAL header 第一个字节低 5bit 为 type
        nal_type = data[nal_start] & 0x1F
        if nal_type == 5:
            return True
        i = nal_start + 1
    return False


class _QueueManager(BaseManager):
    pass


def _make_manager_queue(maxsize: int):
    import multiprocessing as mp

    q = mp.Queue(maxsize=maxsize)
    _QueueManager.register("get_queue", callable=lambda: q)
    return q


class SelectiveDecodeProducer:
    def __init__(self, args):
        self.args = args
        self.byte_budget = args.byte_budget
        self.pending_bytes = 0
        self.sent_frames = 0
        self.last_stat_t = time.monotonic()
        self.rx_aus = 0

        self._stop = False

        # Queue + Manager（供推理端连接）
        self._queue = _make_manager_queue(args.queue_size)
        self._mgr = _QueueManager(address=(args.bind, args.port), authkey=args.authkey.encode("utf-8"))
        self._mgr.start()
        atexit.register(lambda: self._mgr.shutdown())

        # 解析管线（只到 AU，不解码）
        proto = "protocols=tcp" if args.tcp else ""
        head = f"rtspsrc name=src latency={args.latency} {proto} ! "
        parse_pipe = (
            head
            + "rtph264depay ! h264parse ! video/x-h264,stream-format=byte-stream,alignment=au ! "
            + "appsink name=parsink emit-signals=true sync=false max-buffers=1 drop=true"
        )
        self.parse_pipeline = Gst.parse_launch(parse_pipe)
        self.src = self.parse_pipeline.get_by_name("src")
        self.src.set_property("location", args.url)
        self.parsink = self.parse_pipeline.get_by_name("parsink")

        # 解码管线（仅在触发时推送 AU）
        dec_pipe = (
            "appsrc name=appsrc is-live=false format=time do-timestamp=true "
            "caps=video/x-h264,stream-format=byte-stream,alignment=au ! "
            "h264parse ! "
            "avdec_h264 ! videoconvert ! video/x-raw,format=RGB ! "
            "appsink name=decsink emit-signals=false sync=false max-buffers=1 drop=true"
        )
        self.dec_pipeline = Gst.parse_launch(dec_pipe)
        self.appsrc = self.dec_pipeline.get_by_name("appsrc")
        self.decsink = self.dec_pipeline.get_by_name("decsink")

        self._wire_bus(self.parse_pipeline, "parse")
        self._wire_bus(self.dec_pipeline, "decode")

        self.parsink.connect("new-sample", self._on_new_au_sample)

    def _wire_bus(self, pipeline: Gst.Pipeline, tag: str):
        bus = pipeline.get_bus()
        bus.add_signal_watch()

        def on_msg(_bus, msg):
            if msg.type == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                print(f"[ERROR:{tag}] {err}", file=sys.stderr)
                if dbg:
                    print(dbg, file=sys.stderr)
                self._stop = True
            elif msg.type == Gst.MessageType.EOS:
                print(f"[INFO:{tag}] EOS")
                self._stop = True

        bus.connect("message", on_msg)

    def _decode_au_to_rgb(self, au_bytes: bytes) -> Optional[Tuple[int, int, bytes]]:
        buf = Gst.Buffer.new_allocate(None, len(au_bytes), None)
        buf.fill(0, au_bytes)
        ret = self.appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            return None

        # 拉取一帧
        timeout_ns = int(self.args.decode_timeout_ms) * 1_000_000
        sample = None
        # 不同 JetPack / PyGObject 版本，appsink 的方法暴露不一致：
        # - 有的版本有 .try_pull_sample()
        # - 有的版本需要用 signal: emit('try-pull-sample', timeout_ns)
        if hasattr(self.decsink, "try_pull_sample"):
            sample = self.decsink.try_pull_sample(timeout_ns)
        else:
            try:
                sample = self.decsink.emit("try-pull-sample", timeout_ns)
            except Exception:
                sample = None
        if sample is None:
            return None
        caps = sample.get_caps()
        s = caps.get_structure(0)
        w = s.get_value("width")
        h = s.get_value("height")
        out_buf = sample.get_buffer()
        ok, mapinfo = out_buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            rgb = bytes(mapinfo.data)
        finally:
            out_buf.unmap(mapinfo)
        return int(w), int(h), rgb

    def _try_send(self, payload) -> bool:
        try:
            # BaseManager QueueProxy 支持 block=False
            self._queue.put(payload, block=False)
            return True
        except queue.Full:
            return False

    def _maybe_print_stats(self):
        now = time.monotonic()
        if now - self.last_stat_t < 1.0:
            return
        self.last_stat_t = now
        print(
            f"[STATS] rx_au={self.rx_aus}/s, sent={self.sent_frames}, pending_bytes={self.pending_bytes}"
        )
        self.rx_aus = 0

    def _on_new_au_sample(self, sink):
        if self._stop:
            return Gst.FlowReturn.EOS
        sample = sink.emit("pull-sample")
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

        self.rx_aus += 1
        is_idr = _h264_is_idr(au)
        trigger_reason = None

        if is_idr:
            trigger_reason = "I_FRAME_DETECTED"
        else:
            self.pending_bytes += len(au)
            if self.pending_bytes >= self.byte_budget:
                trigger_reason = "BUDGET_REACHED"

        if trigger_reason:
            decoded = self._decode_au_to_rgb(au)
            # 重置预算累积（与 smart_frame_extractor 一致：触发后重置）
            self.pending_bytes = 0

            if decoded is not None:
                w, h, rgb = decoded
                payload = {
                    "ts": time.time(),
                    "reason": trigger_reason,
                    "w": w,
                    "h": h,
                    "rgb": rgb,  # bytes，长度应为 w*h*3
                }
                if self._try_send(payload):
                    self.sent_frames += 1
                # 队列满则丢弃本次 payload（按你的要求）

                if self.args.max_frames > 0 and self.sent_frames >= self.args.max_frames:
                    self._stop = True

        self._maybe_print_stats()
        return Gst.FlowReturn.OK

    def run(self):
        # 启动解码管线（待命）
        if self.dec_pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("dec_pipeline PLAYING 失败")
        if self.parse_pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("parse_pipeline PLAYING 失败")

        print(
            "[INFO] Producer 已启动。推理端请连接："
            f" host={self.args.bind} port={self.args.port} authkey={self.args.authkey!r}"
        )
        print(
            f"[INFO] 触发规则：I 帧必解；非 I 帧累计字节 >= {self.byte_budget} 触发解码并发送。"
        )

        loop = GLib.MainLoop()

        def on_sigint(_sig, _frame):
            print("\n[STOP] 中断，正在停止…")
            self._stop = True
            loop.quit()

        signal.signal(signal.SIGINT, on_sigint)

        # 轮询 stop 标志
        def tick():
            if self._stop:
                loop.quit()
                return False
            return True

        GLib.timeout_add(200, tick)

        try:
            loop.run()
        finally:
            self.parse_pipeline.set_state(Gst.State.NULL)
            self.dec_pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    args = parse_args()
    _start_xvfb_if_needed(args.xvfb)
    Gst.init(None)
    SelectiveDecodeProducer(args).run()

