import cv2
import threading as mp
import queue
import numpy as np
import time
import torch
from PIL import Image
from dataclasses import dataclass
from typing import Optional
from .config import Config

@dataclass
class FrameData:
    """帧数据结构体, 包含帧张量数据、时间戳、帧ID和视频来源"""
    frame_tensor: torch.Tensor      # 帧张量数据
    timestamp: float                # 时间戳
    frame_id: int                   # 帧ID
    source_path: str                # 视频来源: "camera"或视频文件路径
    
    def __post_init__(self):
        """初始化后的验证"""
        if not isinstance(self.frame_tensor, torch.Tensor):
            raise TypeError("frame_tensor must be a torch.Tensor")
        if not isinstance(self.timestamp, (int, float)):
            raise TypeError("timestamp must be a number")
        if not isinstance(self.frame_id, int):
            raise TypeError("frame_id must be an integer")
        if not isinstance(self.source_path, str):
            raise TypeError("source_path must be a string")

class StreamInput:
    def __init__(self, config=None):
        """
        初始化StreamInput模块
        负责从视频流中提取帧，支持摄像头实时视频流或本地视频文件
        
        Args:
            config (Config): 配置对象实例
        """
        self.config = config if config is not None else Config()
        
        # 从Config对象获取配置
        self.fps = self.config.stream_fps # 决定了系统应该以什么频率从视频源提取和处理帧
        self.video_source = self.config.stream_video_source
        self.video_file_path = self.config.stream_video_file_path
        
        # 队列用于线程间传递帧数据
        self.frame_queue = queue.Queue(maxsize=100)
        self.running = False
        self.cap = None  # 视频捕获对象
        self.current_frame_idx = 0
        self.total_frames = 0
        self.video_fps = None
        self.video_duration = 0
        
    def initialize_video_source(self):
        """初始化视频源"""
        if self.video_source == "camera":
            # TODO: 需要后续适配和开发
            # 摄像头仍然使用cv2，因为torchvision.io不支持实时流
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception(f"无法打开摄像头: {self.video_source}")
        elif self.video_source == "file":
            if not self._load_video(self.video_file_path):
                raise Exception(f"无法使用cv2打开视频文件: {self.video_file_path}")
            
        self.current_frame_idx = 0
    
    def _load_video(self, file_path: str) -> bool:
        """使用cv2加载视频文件"""
        try:
            print(f"使用cv2加载视频文件: {file_path}")
            self.cap = cv2.VideoCapture(file_path)
            
            if not self.cap.isOpened():
                raise Exception(f"无法打开视频文件: {file_path}")
                
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_duration = self.total_frames / self.video_fps if self.video_fps and self.video_fps > 0 else 0
            
            print(f"视频加载成功: 总帧数 {self.total_frames}, FPS: {self.video_fps}, 时长: {self.video_duration:.2f}s")
            return True
        except Exception as e:
            raise Exception(f"无法使用cv2打开视频文件 {file_path}: {str(e)}")
    
    def _tensor_to_frame_data(self, frame_tensor, timestamp, frame_index=None, source_path="unknown"):
        """
        将torch张量转换为FrameData结构体
        
        Args:
            frame_tensor: torch.Tensor, shape (H, W, C) 或 (C, H, W)
            timestamp: float, 时间戳
            frame_index: int, 原始视频帧索引（如果有）
            source_path: str, 视频来源路径（"camera"或视频文件路径）
            
        Returns:
            FrameData: 帧数据结构体，输出标准CHW格式
        """
        # 确保张量格式为 (H, W, C)
        if frame_tensor.shape[0] in [1, 3]:  # 如果是 (C, H, W) 格式
            frame_tensor = frame_tensor.permute(1, 2, 0)
        
        # 确保数据类型为float32
        if frame_tensor.dtype != torch.float32:
            frame_tensor = frame_tensor.float()
        
        # 确保值在[0, 1]范围内
        if frame_tensor.max() > 1.0:
            frame_tensor = frame_tensor / 255.0
        
        # 转换为标准CHW格式 (C, H, W) - PyTorch标准格式
        frame_tensor = frame_tensor.permute(2, 0, 1)
        
        # 如果提供了帧索引，使用原始帧索引作为frame_id，否则使用时间戳生成
        if frame_index is not None:
            frame_id = frame_index
        else:
            frame_id = int(timestamp * 1000)
        
        return FrameData(
            frame_tensor=frame_tensor,
            timestamp=timestamp,
            frame_id=frame_id,
            source_path=source_path
        )
    
    def _extract_camera_frame(self, current_time):
        """
        从摄像头提取帧
        
        Args:
            current_time: float, 当前时间戳
            
        Returns:
            FrameData or None: 帧数据结构体，失败时返回None
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame_tensor = torch.from_numpy(frame).float() / 255.0
        # 摄像头模式没有原始帧索引，所以不传入frame_index
        # 传入"camera"作为source_path
        frame_data = self._tensor_to_frame_data(frame_tensor, current_time, source_path="camera")
        
        return frame_data
    
    def _extract_video_frame(self, current_time):
        """
        从视频文件提取帧
        
        Args:
            current_time: float, 当前时间戳
            
        Returns:
            FrameData or None: 帧数据结构体，视频结束时返回None
        """
        # 优化：动态调整跳帧间隔，确保队列增长速度
        if self.video_fps and self.video_fps > 0:
            # 如果队列较小，减少跳帧数以提高提取速度
            # 当且仅当视频速度快于系统处理速度时，才需要跳帧
            if self.frame_queue.qsize() < 30:
                frame_skip = max(1, int(self.video_fps / (self.fps * 1.5)))  # 提高提取速度
            else:
                frame_skip = max(1, int(self.video_fps / self.fps))
        else:
            frame_skip = 1
        
        # 检查是否超出视频范围
        if self.current_frame_idx >= self.total_frames:
            return None
            
        # 优化：减少set()操作的频率，因为这是一个昂贵的操作
        if self.current_frame_idx % (frame_skip * 5) == 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        
        ret, frame = self.cap.read()
        if not ret:
            # 尝试重新定位
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                return None
        
        # 保存当前帧索引，用于设置frame_id
        original_frame_idx = self.current_frame_idx
        
        # 转换为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转换为torch张量 (H, W, C) 格式，值范围[0, 1]
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        # 传入原始帧索引作为frame_id和视频文件路径作为source_path
        frame_data = self._tensor_to_frame_data(frame_tensor, current_time, original_frame_idx, source_path=self.video_file_path)
        
        # 更新下一帧位置
        self.current_frame_idx += frame_skip
        
        return frame_data

    def _extract_frames(self):
        """提取帧的线程函数"""
        frame_interval = 1.0 / self.fps
        frames_processed = 0
        frames_skipped = 0
        start_time = time.time()
        
        # 批量读取优化参数
        batch_size = 5  # 一次尝试处理的帧数
        
        while self.running:
            frame_batch_start = time.time()
            
            # 批量处理帧以提高效率
            for _ in range(batch_size):
                if not self.running:
                    break
                    
                # 根据视频源类型提取帧
                if self.video_source == "camera":
                    frame_data = self._extract_camera_frame(time.time())
                    if frame_data is None:
                        # 摄像头读取失败，短暂等待后重试
                        time.sleep(0.1)
                        continue
                else:  # video file
                    # 优化：直接计算应该读取的帧位置，避免逐帧读取
                    frame_data = self._extract_video_frame(time.time())
                    if frame_data is None:
                        print(f"视频结束，共处理 {frames_processed} 帧，跳过 {frames_skipped} 帧")
                        return
                
                frames_processed += 1
                
                # 将帧数据放入队列 - 使用非阻塞方式提高效率
                try:
                    # 使用较短的超时时间
                    self.frame_queue.put(frame_data, timeout=0.1)
                except:
                    # 队列满，跳过这一帧
                    frames_skipped += 1
                    if frames_processed % 100 == 0:
                        print(f"队列已满，跳过第 {frame_data.frame_id} 帧")
                    continue
            
            # 每处理100帧打印一次状态
            if frames_processed % 100 == 0:
                elapsed = time.time() - start_time
                print(f"已处理 {frames_processed} 帧，当前队列大小: {self.frame_queue.qsize()}，耗时: {elapsed:.2f}s")
            
            # 优化的帧率控制 - 计算批量处理后的睡眠时间
            batch_elapsed = time.time() - frame_batch_start
            target_batch_time = batch_size * frame_interval
            
            if batch_elapsed < target_batch_time:
                # 使用更精确的睡眠方式
                sleep_time = target_batch_time - batch_elapsed
                # 对于短睡眠，使用较小的时间片来提高精度
                if sleep_time > 0.005:
                    time.sleep(sleep_time * 0.9)  # 先睡90%
                    # 使用自旋等待剩余时间以提高精度
                    spin_end = time.time() + sleep_time * 0.1
                    while time.time() < spin_end:
                        pass
    
    def start(self):
        """启动帧提取线程"""
        self.running = True
        self.initialize_video_source()
        
        # 设置线程为daemon模式，确保主程序退出时线程也会退出
        self.process = mp.Thread(target=self._extract_frames, daemon=True)
        # 提高线程优先级（如果系统支持）
        if hasattr(self.process, 'name'):
            self.process.name = "StreamInput-Extractor"
        
        self.process.start()
        print(f"StreamInput线程已启动, FPS: {self.fps}, 目标队列增长: 高速度模式")
    
    def stop(self):
        """停止帧提取线程"""
        self.running = False
        if hasattr(self, 'process') and self.process.is_alive():
            self.process.join(timeout=5)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("StreamInput线程已停止")
    
    def get_video_info(self):
        """获取视频信息"""
        if self.video_source == "file":
            return {
                'total_frames': self.total_frames,
                'video_fps': self.video_fps,
                'target_fps': self.fps,
                'duration': self.total_frames / self.video_fps if self.video_fps and self.video_fps > 0 else 0,
                'source': 'file'
            }
        elif self.video_source == "camera":
            return {
                'target_fps': self.fps,
                'source': 'camera'
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

if __name__ == "__main__":
    # 测试代码
    config = Config()
    # 修改配置为摄像头模式进行测试
    config.stream_video_source = "file"
    stream_input = StreamInput(config)
    
    try:
        stream_input.start()
        
        frame_count = 0
        while frame_count < 10:  # 只读取10帧进行测试
            frame_data = stream_input.get_frame()
            if frame_data:
                print(f"获取到帧 {frame_count}: ID={frame_data.frame_id}, "
                      f"时间戳={frame_data.timestamp:.3f}, "
                      f"张量形状={frame_data.frame_tensor.shape}")
                frame_count += 1
            else:
                break
                
    except Exception as e:
        print(f"测试失败: {e}")
    finally:
        stream_input.stop()