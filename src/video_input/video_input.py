import os
import glob
import multiprocessing as mp
import queue
import numpy as np
import time
import torch
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from typing import Optional, List, Iterator, Dict, Tuple
# 本项目
from src.config import Config
from src.video_utils.file_video_reader import open_file_video_reader
from src.video_utils.ffprobe_utils import get_frame_info_for_stream

@dataclass
class FrameData:
    """帧数据结构体, 包含帧numpy数组数据、时间戳、帧ID、视频来源、视频总帧数、视频FPS和视频时长"""
    frame: Optional[np.ndarray]     # 帧numpy数组数据；跨进程链路默认不传像素，仅传引用
    timestamp: float                # 时间戳, 用于系统测时
    frame_id: int                   # 帧ID
    source_path: str                # 视频来源: "camera"或视频文件路径
    total_frames: Optional[int] = None  # 视频总帧数，仅视频文件模式有值
    video_fps: Optional[float] = None   # 视频FPS，仅视频文件模式有值
    duration: Optional[float] = None     # 视频总时长(秒)，由 decord/cv2 读取得到，仅视频文件模式有值
    trace_ts: Optional[Dict[str, float]] = None  # 链路时延埋点（秒）

class VideoInputBase:
    def __init__(self, config=None):
        """
        初始化 VideoInput 模块
        负责从本地视频文件提取帧并送入队列。

        Args:
            config (Config): 配置对象实例
        """
        if config is None:
            config = Config()

        # 从Config对象获取配置
        self.video_file_path = config.video_file_path
        self.log_file = config.video_log_file
        self.is_original_fps = config.video_original_fps
        self.target_fps = config.video_target_fps

        self.cap = None  # cv2视频捕获对象
        self.vr = None  # 文件视频读取器（decord/cv2 适配，与 decord.VideoReader 索引接口一致）
        self._file_reader_backend = getattr(config, "video_reader_backend", "auto")
        self.current_frame_idx = 0 # 读取到的帧索引
        self.total_frames = 0
        self.video_fps = None
        self.video_duration = 0
        # 边端实时默认仅跨进程传引用，避免传输 ndarray 的 pickle 拷贝成本
        self.ipc_send_frame = getattr(config, "video_ipc_send_frame", False)
        
    def _set_logger(self):
        """设置日志记录器"""
        log_file = self.log_file

        pattern = log_file.replace(".log", "*")
        log_files = glob.glob(pattern)
        for f in log_files:
            try:
                os.remove(f)
            except Exception as e:
                pass

        self.logger = logging.getLogger(name=self.__class__.__name__)
        # 清除旧的处理器，避免重复添加
        self.logger.handlers.clear()
        # 设置logger本身的级别，确保所有级别日志都能被处理
        self.logger.setLevel(logging.DEBUG)
        # 配置日志输出到控制台
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)

        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, mode='w')
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False
    
    def _initialize_video_source(self):
        """初始化视频源"""
        self._open_file_video_reader(self.video_file_path)
        self.current_frame_idx = 0
    
    def _open_file_video_reader(self, video_file_path: str) -> bool:
        """使用 file_video_reader（decord 或 cv2）加载本地视频文件"""
        self.vr = open_file_video_reader(video_file_path, self._file_reader_backend)
        
        self.video_fps = self.vr.get_avg_fps()
        self.total_frames = len(self.vr)
        self.video_duration = self.total_frames / self.video_fps if self.video_fps and self.video_fps > 0 else 0
        
        self.logger.info(f"视频加载成功: 总帧数 {self.total_frames}, FPS: {self.video_fps}, 时长: {self.video_duration:.2f}s")
        return True
    
    def _to_frame_data(self, frame, timestamp, frame_id=None, source_path="unknown", total_frames=None, video_fps=None, duration=None):
        # 如果提供了帧索引，使用原始帧索引作为frame_id，否则使用时间戳生成
        if frame_id is None:
            frame_id = int(timestamp * 1000)
        
        return FrameData(
            frame=frame,
            timestamp=timestamp,
            frame_id=frame_id,
            source_path=source_path,
            total_frames=total_frames,
            video_fps=video_fps,
            duration=duration,
            trace_ts={"video_extracted_at": timestamp}
        )

    
    def _extract_video_frame(self, current_time):
        # 检查是否超出视频范围
        if self.current_frame_idx >= self.total_frames:
            return None
            
        # 保存当前帧索引，用于设置frame_id
        original_frame_idx = self.current_frame_idx
        
        if self.vr is not None:
            frame = self.vr[original_frame_idx].asnumpy() # RGB, [H,W,C], uint8
        self.logger.debug(f"成功提取帧 {original_frame_idx}, 帧形状: {frame.shape}, 帧数据类型: {frame.dtype}")
        
        # 传入原始帧索引作为frame_id、视频文件路径作为source_path，以及视频总帧数、FPS和时长
        # 多进程实时链路默认仅传引用；同步/benchmark 路径保留帧像素
        should_send_frame_payload = self.ipc_send_frame or (not hasattr(self, "running_event"))
        frame_data = self._to_frame_data(
            frame if should_send_frame_payload else None,
            current_time, 
            original_frame_idx, 
            source_path=self.video_file_path,
            total_frames=self.total_frames,
            video_fps=self.video_fps,
            duration=self.video_duration
        )
        
        # 每次只前进一帧
        self.current_frame_idx += 1
        
        return frame_data

    def get_video_info(self):
        """获取视频信息"""
        return {
            'total_frames': self.total_frames,
            'video_fps': self.video_fps,
            'duration': self.total_frames / self.video_fps if self.video_fps and self.video_fps > 0 else 0,
            'source': 'file'
        }
    
    def init_for_file(self, video_file_path: str):
        """为 benchmark 同步模式初始化视频文件源（不启动子进程）"""
        if not hasattr(self, "logger") or self.logger is None:
            self._set_logger()
        self.video_file_path = video_file_path
        self._initialize_video_source()

    def iter_frames_batch(self, batch_size: int, frame_interval: int = 1) -> Iterator[List[FrameData]]:
        """
        按 batch 同步迭代帧，供 benchmark 使用。
        需先调用 init_for_file(video_path)。
        按索引跳帧读取，frame_interval > 1 时直接跳到目标帧，不读取中间帧。

        Args:
            batch_size: 每批帧数量
            frame_interval: 帧间隔，每 frame_interval 帧取一帧（1 表示不跳过）

        Yields:
            每批 FrameData 列表
        """
        batch = []
        step = max(1, frame_interval)
        current_time = time.time()
        while self.current_frame_idx < self.total_frames:
            frame_data = self._extract_video_frame(current_time)
            if frame_data is None:
                break
            batch.append(frame_data)
            if step > 1:
                # _extract_video_frame 已 +1，再跳过 (step-1) 帧，下次读取目标索引
                self.current_frame_idx += step - 1
            if len(batch) >= batch_size:
                yield batch
                batch = []
            current_time = time.time()
        if batch:
            yield batch


class VideoInputOnline(VideoInputBase):
    """在线输入类：在 Base 同步能力上提供进程与队列能力。"""

    def __init__(self, config=None):
        super().__init__(config)
        self.frame_queue = mp.Queue(maxsize=100)

    def _extract_frames(self):
        """提取帧的线程函数（不跳过任何帧）"""
        frames_processed = 0
        start_time = time.time()

        frame_interval = 1.0 / self.video_fps if self.video_fps and self.video_fps > 0 else 1.0 / 30.0

        while self.running_event.is_set():
            frame_start = time.time()
            frame_data = self._extract_video_frame(time.time())
            if frame_data is None:
                self.logger.info("帧数据为None, 可能已到达视频末尾, 跳出循环")
                break

            self._put_frame_safely(frame_data)
            frames_processed += 1

            if self.is_original_fps:
                frame_elapsed = time.time() - frame_start
                if frame_elapsed < frame_interval:
                    sleep_time = frame_interval - frame_elapsed
                    time.sleep(sleep_time)
                else:
                    additional_wait_time = frame_elapsed - frame_interval
                    self.logger.debug(f"发生阻塞, 阻塞时间: {additional_wait_time:.4f}s")

        if frames_processed > 0:
            elapsed = time.time() - start_time
            self.logger.info(f"处理完成，共入队列 {frames_processed} 帧，耗时: {elapsed:.2f}s")

    def start(self):
        """启动帧提取子进程"""
        self.running_event = mp.Event()
        self.running_event.set()
        self.process = mp.Process(target=self._process_main, daemon=True)
        if hasattr(self.process, "name"):
            self.process.name = f"{self.__class__.__name__}-Extractor"
        self.process.start()

    def start_single_process(self):
        """启动单线程处理模式"""
        self.running_event = mp.Event()
        self.running_event.set()
        self._process_main()

    def _process_main(self):
        """子进程主函数，负责初始化视频源和提取帧"""
        self._set_logger()
        self.logger.info(f"子进程启动, 进程ID: {mp.current_process().pid}")
        self._initialize_video_source()
        self._extract_frames()

    def _put_frame_safely(self, frame_data: FrameData, timeout=None):
        """安全地将帧放入队列"""
        try:
            if timeout is None:
                self.frame_queue.put(frame_data)
            else:
                self.frame_queue.put(frame_data, timeout=timeout)
            self.logger.debug(f"成功将帧 {frame_data.frame_id} 放入队列")
        except Exception as e:
            self.logger.error(f"丢包: {e}")

    def _is_process_parent(self):
        """当前进程是否为子进程的父进程（只有父进程才能安全调用 is_alive/join）"""
        if not hasattr(self, "process"):
            return False
        parent_pid = getattr(self.process, "_parent_pid", None)
        return parent_pid is not None and parent_pid == os.getpid()

    def stop(self):
        """停止帧提取子进程"""
        if hasattr(self, "running_event"):
            self.running_event.clear()
        if not hasattr(self, "process"):
            return
        try:
            if not self._is_process_parent():
                return
            if self.process.is_alive():
                self.process.join(timeout=5)
        except (AssertionError, ValueError) as e:
            logging.getLogger(__name__).debug("停止子进程时跳过 join: %s", e)
        self.cap = None
        if self.vr is not None and hasattr(self.vr, "close"):
            try:
                self.vr.close()
            except Exception:
                pass
        self.vr = None

    def get_frame(self):
        """获取一帧数据"""
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def get_frame_queue(self):
        """获取帧队列供其他模块使用"""
        return self.frame_queue

# pict_type 到整数的映射：I=0, P=1, B=2
P_TYPE_I, P_TYPE_P, P_TYPE_B = 0, 1, 2
PICT_TYPE_TO_INT = {"I": P_TYPE_I, "P": P_TYPE_P, "B": P_TYPE_B}


@dataclass
class SymFrameData:
    frame: Optional[np.ndarray]     # 帧numpy数组数据；可为 None（仅传引用）
    frame_id: int                   # 帧ID
    p_type: int                     # 在压缩算法中的类型 (0=I, 1=P, 2=B)
    source_path: str                # 视频来源: "camera"或视频文件路径
    pkt_size: Optional[int] = None  # 该包压缩后的大小
    total_frames: Optional[int] = None  # 视频总帧数，仅视频文件模式有值
    video_fps: Optional[float] = None   # 视频FPS，仅视频文件模式有值
    duration: Optional[float] = None    # 视频总时长(秒)，由 decord/cv2 读取得到，仅视频文件模式有值
    trace_ts: Optional[Dict[str, float]] = None  # 链路时延埋点（秒）

class SymVideoInput(VideoInputBase):
    """
    继承 VideoInputBase，按 p_type 分组返回帧：每次迭代返回两个 I 帧之间的帧（留头去尾）。
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.i_frame_indices: List[int] = []
        self.frame_types: List[str] = []
        self.pkt_sizes: List[int] = []

    def _to_sym_frame_data(
        self,
        frame: Optional[np.ndarray],
        frame_id: int,
        p_type: int,
        source_path: str,
        pkt_size: Optional[int] = None,
        total_frames: Optional[int] = None,
        video_fps: Optional[float] = None,
        duration: Optional[float] = None,
        trace_ts: Optional[Dict[str, float]] = None,
    ) -> SymFrameData:
        """构造 SymFrameData，包含 p_type 与 pkt_size 信息"""
        return SymFrameData(
            frame=frame,
            frame_id=frame_id,
            p_type=p_type,
            source_path=source_path,
            pkt_size=pkt_size,
            total_frames=total_frames,
            video_fps=video_fps,
            duration=duration,
            trace_ts=trace_ts,
        )

    def _extract_video_frame_at(self, frame_idx: int) -> Optional[SymFrameData]:
        """按索引提取单帧，返回 SymFrameData（含 p_type），不修改 current_frame_idx"""
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None
        if frame_idx >= len(self.frame_types):
            p_type_int = P_TYPE_I
        else:
            p_type_int = PICT_TYPE_TO_INT.get(self.frame_types[frame_idx], P_TYPE_I)

        if self.vr is not None:
            frame = self.vr[frame_idx].asnumpy()
        else:
            return None

        pkt_size = None
        if hasattr(self, "pkt_sizes") and self.pkt_sizes and frame_idx < len(self.pkt_sizes):
            pkt_size = self.pkt_sizes[frame_idx]

        return self._to_sym_frame_data(
            frame=frame,
            frame_id=frame_idx,
            p_type=p_type_int,
            source_path=self.video_file_path,
            pkt_size=pkt_size,
            total_frames=self.total_frames,
            video_fps=self.video_fps,
            duration=self.video_duration,
            trace_ts={"video_extracted_at": time.time()},
        )

    def init_for_file(self, video_file_path: str):
        """为 SymVideoInput 同步模式初始化视频文件源，并获取 ffprobe 的 I 帧、pict_type、pkt_size 信息"""
        super().init_for_file(video_file_path)
        self.i_frame_indices, self.frame_types, self.pkt_sizes = get_frame_info_for_stream(video_file_path)
        self.logger.info(f"ffprobe: I 帧数 {len(self.i_frame_indices)}, 总帧类型数 {len(self.frame_types)}, 每帧压缩大小数 {len(self.pkt_sizes)}")

    def iter_gop_ranges(self) -> Iterator[Tuple[int, int]]:
        """
        按 GOP 迭代，仅返回 (start, end) 索引范围，不解码帧。
        需先调用 init_for_file(video_path)。

        Yields:
            (start, end) 元组，表示该 GOP 的帧索引范围 [start, end)
        """
        if not self.i_frame_indices:
            self.logger.warning("无 I 帧信息，无法按 GOP 迭代")
            return

        for k in range(len(self.i_frame_indices)):
            start = self.i_frame_indices[k]
            end = self.i_frame_indices[k + 1] if k + 1 < len(self.i_frame_indices) else self.total_frames
            yield start, end

    def iter_frames_by_gop(self) -> Iterator[List[SymFrameData]]:
        """
        按 GOP（两个 I 帧之间）迭代帧，留头去尾：含起始 I 帧，不含下一 I 帧。
        会解码该 GOP 内每一帧，仅当需要完整帧数据时使用；否则优先用 iter_gop_ranges + 按需解码。

        Yields:
            每个 GOP 的 SymFrameData 列表，即 [I, P, P, B, ...] 直到下一 I 之前
        """
        if not self.i_frame_indices:
            self.logger.warning("无 I 帧信息，无法按 GOP 迭代")
            return

        for start, end in self.iter_gop_ranges():
            group: List[SymFrameData] = []
            for idx in range(start, end):
                fd = self._extract_video_frame_at(idx)
                if fd is not None:
                    group.append(fd)
            if group:
                yield group