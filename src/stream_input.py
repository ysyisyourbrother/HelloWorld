import cv2
import multiprocessing as mp
import queue
import numpy as np
import time
import torch
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from typing import Optional
from decord import VideoReader

# 本项目
from src.config import Config

@dataclass
class FrameData:
    """帧数据结构体, 包含帧numpy数组数据、时间戳、帧ID、视频来源、视频总帧数和视频FPS"""
    frame: np.ndarray               # 帧numpy数组数据
    timestamp: float                # 时间戳, 用于系统测时
    frame_id: int                   # 帧ID
    source_path: str                # 视频来源: "camera"或视频文件路径
    total_frames: Optional[int] = None  # 视频总帧数，仅视频文件模式有值
    video_fps: Optional[float] = None   # 视频FPS，仅视频文件模式有值

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
        self.config = config

        # 从Config对象获取配置
        self.video_source = self.config.stream_video_source
        self.video_file_path = self.config.stream_video_file_path
        # 获取视频读取器类型
        self.reader_type = self.config.stream_reader_type
        
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
        log_file = self.config.stream_log_file

        self.logger = logging.getLogger(name='StreamInput')
        # 设置logger本身的级别，确保所有级别日志都能被处理
        self.logger.setLevel(logging.DEBUG)
        # 配置日志输出到控制台
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)

        # 配置日志输出到文件，设置日志回滚
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
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
        # self.video_fps = float(self.vr.metadata.get('video', {}).get('fps', 30))
        self.total_frames = len(self.vr)            
        self.video_duration = self.total_frames / self.video_fps if self.video_fps and self.video_fps > 0 else 0
        
        self.logger.info(f"视频加载成功: 总帧数 {self.total_frames}, FPS: {self.video_fps}, 时长: {self.video_duration:.2f}s")
        return True
    
    def _to_frame_data(self, frame, timestamp, frame_id=None, source_path="unknown", total_frames=None, video_fps=None):
        # 如果提供了帧索引，使用原始帧索引作为frame_id，否则使用时间戳生成
        if frame_id is None:
            frame_id = int(timestamp * 1000)
        
        return FrameData(
            frame=frame,
            timestamp=timestamp,
            frame_id=frame_id,
            source_path=source_path,
            total_frames=total_frames,
            video_fps=video_fps
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
        self.logger.debug(f"正在提取帧 {original_frame_idx}/{self.total_frames}")
        
        if self.reader_type == 'decord' and self.vr is not None:
            frame = self.vr[original_frame_idx].asnumpy() # RGB, [H,W,C], uint8
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()  # RGB, [H,W,C], uint8
            if not ret:
                self.logger.error(f"cv2读取帧失败，返回ret={ret}")
                return None
            # 转换为RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        self.logger.debug(f"成功提取帧 {original_frame_idx}, 帧形状: {frame.shape}, 帧数据类型: {frame.dtype}")
        
        # 传入原始帧索引作为frame_id、视频文件路径作为source_path，以及视频总帧数和FPS
        frame_data = self._to_frame_data(
            frame, 
            current_time, 
            original_frame_idx, 
            source_path=self.video_file_path,
            total_frames=self.total_frames,
            video_fps=self.video_fps
        )
        
        # 每次只前进一帧
        self.current_frame_idx += 1
        self.logger.debug(f"帧索引递增，下一帧索引: {self.current_frame_idx}")
        
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
                
                self.logger.debug(f"准备将帧 {frame_data.frame_id} 放入队列")
                
                # TODO: 这里可以有两种处理方式：
                # 1. 确保所有帧都被处理, 不跳过任何帧, 队列满了就阻塞等待
                # 2. 模拟视频播放, 按视频原始帧率处理帧, 队列满了就丢包
                if True:
                    # 队列满了就会卡在这里, 这是正常的
                    self._put_frame_safely(frame_data)
                    frames_processed += 1
                    self.logger.info(f"已处理帧数: {frames_processed}")
                else:
                    # 满了就丢包
                    pass
                
                # 帧率控制 - 确保不超过视频原始FPS
                frame_elapsed = time.time() - frame_start
                if frame_elapsed < frame_interval:
                    sleep_time = frame_interval - frame_elapsed
                    time.sleep(sleep_time)
                else:
                    additional_wait_time = frame_elapsed - frame_interval
                    self.logger.info(f"发生阻塞, 阻塞时间: {additional_wait_time:.4f}s")

        
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
            self.process.name = "StreamInput-Extractor"
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
            self.logger.debug(f"尝试将帧 {frame_data.frame_id} 放入队列，帧形状: {frame_data.frame.shape}")
            if timeout is None:
                self.frame_queue.put(frame_data)
            else:
                self.frame_queue.put(frame_data, timeout=timeout)
            self.logger.debug(f"成功将帧 {frame_data.frame_id} 放入队列")
        except Exception as e:
            self.logger.error(f"丢包: {e}")

    def stop(self):
        """停止帧提取子进程"""
        # 使用running_event来停止子进程
        if hasattr(self, 'running_event'):
            self.running_event.clear()
            self.logger.info("已清除running_event标志")
        # 等待子进程结束
        if hasattr(self, 'process') and self.process.is_alive():
            self.logger.info("等待子进程结束...")
            self.process.join(timeout=5)
            self.logger.info("子进程已结束或超时")
        # 注意：在父进程中不释放视频资源，因为它们在子进程中已经被释放
        # 重置状态以便可能的重新启动
        self.cap = None
        self.vr = None
        self.logger.info("子进程已停止")
    
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
