import cv2
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
from typing import Optional, List, Iterator
from decord import VideoReader

# 本项目
from src.config import Config
from src.video_utils.ffprobe_utils import get_frame_info_for_stream

@dataclass
class FrameData:
    """帧数据结构体, 包含帧numpy数组数据、时间戳、帧ID、视频来源、视频总帧数、视频FPS和视频时长"""
    frame: np.ndarray               # 帧numpy数组数据
    timestamp: float                # 时间戳, 用于系统测时
    frame_id: int                   # 帧ID
    source_path: str                # 视频来源: "camera"或视频文件路径
    total_frames: Optional[int] = None  # 视频总帧数，仅视频文件模式有值
    video_fps: Optional[float] = None   # 视频FPS，仅视频文件模式有值
    duration: Optional[float] = None     # 视频总时长(秒)，由 decord/cv2 读取得到，仅视频文件模式有值

class StreamInput:
    def __init__(self, config=None):
        """
        初始化StreamInput模块
        负责从视频流中提取帧，支持摄像头实时视频流或本地视频文件
        
        Args:
            config (Config): 配置对象实例
        """
        if config is None:
            config = Config()

        # 从Config对象获取配置
        self.video_source = config.stream_video_source
        self.video_file_path = config.stream_video_file_path
        # 获取视频读取器类型
        self.reader_type = config.stream_reader_type
        self.log_file = config.stream_log_file
        self.is_original_fps = config.stream_original_fps
        self.target_fps = config.stream_target_fps

        # 使用multiprocessing.Queue以支持多进程间通信
        self.frame_queue = mp.Queue(maxsize=100)
        self.cap = None  # cv2视频捕获对象
        self.vr = None  # decord视频读取器对象
        self.current_frame_idx = 0 # 读取到的帧索引
        self.total_frames = 0
        self.video_fps = None
        self.video_duration = 0
        
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

        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, mode='w')
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False
    
    def _initialize_video_source(self):
        """初始化视频源"""
        if self.video_source == "camera":
            # 摄像头仍然使用cv2，因为decord不支持实时流
            self.logger.info(f"使用cv2初始化摄像头视频源")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception(f"无法打开摄像头: {self.video_source}")
            
        elif self.video_source == "file":
            self.logger.info(f"使用{self.reader_type}初始化文件视频源")
            if self.reader_type == 'decord':
                self._decord_load_video(self.video_file_path)
            else:
                self._cv2_load_video(self.video_file_path)
        self.current_frame_idx = 0

    def _cv2_load_video(self, video_file_path: str) -> bool:
        """使用cv2加载视频文件"""
        self.cap = cv2.VideoCapture(video_file_path)
        
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = self.total_frames / self.video_fps if self.video_fps and self.video_fps > 0 else 0
        
        self.logger.info(f"视频加载成功: 总帧数 {self.total_frames}, FPS: {self.video_fps}, 时长: {self.video_duration:.2f}s")
        return True
    
    def _decord_load_video(self, video_file_path: str) -> bool:
        """使用decord加载视频文件"""
        # 使用decord的VideoReader加载视频
        self.vr = VideoReader(video_file_path)
        
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
            duration=duration
        )
    
    def _extract_camera_frame(self, current_time):
        """
        从摄像头提取帧
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        # 摄像头模式没有原始帧索引，所以不传入frame_index
        # 传入"camera"作为source_path
        frame_data = self._to_frame_data(frame, current_time, source_path="camera")
        
        return frame_data
    
    def _extract_video_frame(self, current_time):
        # 检查是否超出视频范围
        if self.current_frame_idx >= self.total_frames:
            return None
            
        # 保存当前帧索引，用于设置frame_id
        original_frame_idx = self.current_frame_idx
        
        if self.reader_type == 'decord' and self.vr is not None:
            frame = self.vr[original_frame_idx].asnumpy() # RGB, [H,W,C], uint8
        elif self.reader_type == 'cv2' and self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()  # BGR, [H,W,C], uint8
            if not ret:
                self.logger.error(f"cv2读取帧失败, 返回ret={ret}")
                return None
            # 转换为RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            self.logger.error(f"未知的reader_type: {self.reader_type}")
            return None
        
        self.logger.debug(f"成功提取帧 {original_frame_idx}, 帧形状: {frame.shape}, 帧数据类型: {frame.dtype}")
        
        # 传入原始帧索引作为frame_id、视频文件路径作为source_path，以及视频总帧数、FPS和时长
        frame_data = self._to_frame_data(
            frame, 
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

    def _extract_frames(self):
        """提取帧的线程函数（不跳过任何帧）"""
        frames_processed = 0
        start_time = time.time()
        
        # 对于视频文件，使用原始视频的fps来控制帧率
        # 对于摄像头，使用默认的1/30秒间隔
        frame_interval = 1.0 / self.video_fps if self.video_source == "file" and self.video_fps and self.video_fps > 0 else 1.0 / 30.0
        
        if self.video_source == "camera":
            while self.running_event.is_set():
                frame_start = time.time()
                frame_data = self._extract_camera_frame(time.time())
                try:
                    self._put_frame_safely(frame_data, timeout=1.0)
                    frames_processed += 1
                    break
                except Exception as e:
                    self.logger.error(f"将摄像头帧放入队列时出错: {e}")
                    # 队列仍然满，继续尝试，不跳过帧
                if frame_data is None:
                    # 摄像头读取失败，短暂等待后重试
                    time.sleep(0.1)


        if self.video_source == "file":
            while self.running_event.is_set():
                frame_start = time.time()
                frame_data = self._extract_video_frame(time.time())
                if frame_data is None:
                    # 视频读取完毕，跳出循环
                    self.logger.info(f"帧数据为None, 可能已到达视频末尾, 跳出循环")
                    break
                
                # TODO: 这里可以有两种处理方式：
                # 1. 确保所有帧都被处理, 不跳过任何帧, 队列满了就阻塞等待
                # 2. 模拟视频播放, 按视频原始帧率处理帧, 队列满了就丢包
                if True:
                    # 队列满了就会卡在这里, 这是正常的
                    self._put_frame_safely(frame_data)
                    frames_processed += 1
                else:
                    # 满了就丢包
                    pass
                
                # 帧率控制 - 确保不超过视频原始FPS
                if self.is_original_fps:
                    frame_elapsed = time.time() - frame_start
                    if frame_elapsed < frame_interval:
                        sleep_time = frame_interval - frame_elapsed
                        time.sleep(sleep_time)
                    else:
                        additional_wait_time = frame_elapsed - frame_interval
                        self.logger.debug(f"发生阻塞, 阻塞时间: {additional_wait_time:.4f}s")

        
        # 线程结束时打印最终统计信息
        if frames_processed > 0:
            elapsed = time.time() - start_time
            self.logger.info(f"处理完成，共入队列 {frames_processed} 帧，耗时: {elapsed:.2f}s")
    
    def start(self):
        """启动帧提取子进程"""
        # 创建一个共享变量来控制子进程运行
        self.running_event = mp.Event()
        self.running_event.set()
        # 设置线程为daemon模式，确保主程序退出时线程也会退出
        # 注意：我们将视频源初始化移到子进程内部，确保资源在子进程上下文中正确创建
        self.process = mp.Process(target=self._process_main, daemon=True)
        if hasattr(self.process, 'name'):
            self.process.name = f"{self.__class__.__name__}-Extractor"
        self.process.start()
    
    def start_single_process(self):
        """启动单线程处理模式"""
        self.running_event = mp.Event()
        self.running_event.set()
        self._process_main()

    def _process_main(self):
        """子进程主函数，负责初始化视频源和提取帧"""
        # 注意：在子进程中需要重新初始化logger
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
        if not hasattr(self, 'process'):
            return False
        parent_pid = getattr(self.process, '_parent_pid', None)
        return parent_pid is not None and parent_pid == os.getpid()

    def stop(self):
        """停止帧提取子进程"""
        # 使用running_event来停止子进程
        if hasattr(self, 'running_event'):
            self.running_event.clear()
        # 等待子进程结束（仅当当前进程是父进程时才可安全 join）
        if not hasattr(self, 'process'):
            return
        try:
            if not self._is_process_parent():
                return
            if self.process.is_alive():
                self.process.join(timeout=5)
        except (AssertionError, ValueError) as e:
            logging.getLogger(__name__).debug("停止子进程时跳过 join: %s", e)
        # 注意：在父进程中不释放视频资源，因为它们在子进程中已经被释放
        # 重置状态以便可能的重新启动
        self.cap = None
        self.vr = None
    
    def get_video_info(self):
        """获取视频信息"""
        if self.video_source == "file":
            return {
                'total_frames': self.total_frames,
                'video_fps': self.video_fps,
                'duration': self.total_frames / self.video_fps if self.video_fps and self.video_fps > 0 else 0,
                'source': 'file',
                'reader_type': self.reader_type
            }
        elif self.video_source == "camera":
            return {
                'source': 'camera',
                'reader_type': 'cv2'  # 摄像头始终使用cv2
            }
        return None
    
    def get_frame(self):
        """获取一帧数据"""
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None
    
    def get_frame_queue(self):
        """获取帧队列供其他模块使用"""
        return self.frame_queue

    def init_for_file(self, video_file_path: str):
        """为 benchmark 同步模式初始化视频文件源（不启动子进程）"""
        if not hasattr(self, "logger") or self.logger is None:
            self._set_logger()
        self.video_source = "file"
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

# pict_type 到整数的映射：I=0, P=1, B=2
P_TYPE_I, P_TYPE_P, P_TYPE_B = 0, 1, 2
PICT_TYPE_TO_INT = {"I": P_TYPE_I, "P": P_TYPE_P, "B": P_TYPE_B}


@dataclass
class SymFrameData:
    frame: np.ndarray               # 帧numpy数组数据
    frame_id: int                   # 帧ID
    p_type: int                     # 在压缩算法中的类型 (0=I, 1=P, 2=B)
    source_path: str                # 视频来源: "camera"或视频文件路径
    pkt_size: Optional[int] = None  # 该包压缩后的大小
    total_frames: Optional[int] = None  # 视频总帧数，仅视频文件模式有值
    video_fps: Optional[float] = None   # 视频FPS，仅视频文件模式有值
    duration: Optional[float] = None    # 视频总时长(秒)，由 decord/cv2 读取得到，仅视频文件模式有值

class SymStreamInput(StreamInput):
    """
    继承 StreamInput，按 p_type 分组返回帧：每次迭代返回两个 I 帧之间的帧（留头去尾）。
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.i_frame_indices: list[int] = []
        self.frame_types: list[str] = []
        self.pkt_sizes: list[int] = []

    def _to_sym_frame_data(
        self,
        frame: np.ndarray,
        frame_id: int,
        p_type: int,
        source_path: str,
        pkt_size: int | None = None,
        total_frames: int | None = None,
        video_fps: float | None = None,
        duration: float | None = None,
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
        )

    def _extract_video_frame_at(self, frame_idx: int) -> SymFrameData | None:
        """按索引提取单帧，返回 SymFrameData（含 p_type），不修改 current_frame_idx"""
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None
        if frame_idx >= len(self.frame_types):
            p_type_int = P_TYPE_I
        else:
            p_type_int = PICT_TYPE_TO_INT.get(self.frame_types[frame_idx], P_TYPE_I)

        if self.reader_type == "decord" and self.vr is not None:
            frame = self.vr[frame_idx].asnumpy()
        elif self.reader_type == "cv2" and self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                return None
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        )

    def init_for_file(self, video_file_path: str):
        """为 SymStreamInput 同步模式初始化视频文件源，并获取 ffprobe 的 I 帧、pict_type、pkt_size 信息"""
        super().init_for_file(video_file_path)
        self.i_frame_indices, self.frame_types, self.pkt_sizes = get_frame_info_for_stream(video_file_path)
        self.logger.info(f"ffprobe: I 帧数 {len(self.i_frame_indices)}, 总帧类型数 {len(self.frame_types)}, 每帧压缩大小数 {len(self.pkt_sizes)}")

    def iter_gop_ranges(self) -> Iterator[tuple[int, int]]:
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