import multiprocessing as mp
import numpy as np
import torch
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from config import Config
from stream_input import FrameData

@dataclass
class VectorData:
    """向量数据结构体, 包含向量张量、时间戳、帧ID、视频来源、视频总帧数和视频FPS"""
    vec_tensor: torch.Tensor        # 向量张量数据
    timestamp: float                # 时间戳
    frame_id: int                   # 帧ID
    source_path: str                # 视频来源: "camera"或视频文件路径
    total_frames: Optional[int] = None  # 视频总帧数，仅视频文件模式有值
    video_fps: Optional[float] = None   # 视频FPS，仅视频文件模式有值
    
    def __post_init__(self):
        """初始化后的验证"""
        if not isinstance(self.vec_tensor, torch.Tensor):
            raise TypeError("vec_tensor must be a torch.Tensor")
        if not isinstance(self.timestamp, (int, float)):
            raise TypeError("timestamp must be a number")
        if not isinstance(self.frame_id, int):
            raise TypeError("frame_id must be an integer")
        if not isinstance(self.source_path, str):
            raise TypeError("source_path must be a string")
        if self.total_frames is not None and not isinstance(self.total_frames, int):
            raise TypeError("total_frames must be an integer or None")
        if self.video_fps is not None and not isinstance(self.video_fps, (int, float)):
            raise TypeError("video_fps must be a number or None")
    
    @classmethod
    def from_frame(cls, frame_data: FrameData, vec_tensor: torch.Tensor = None, is_keyframe: bool = False):
        """
        从FrameData创建VectorData
        
        Args:
            frame_data: FrameData对象
            vec_tensor: 向量张量，如果为None则创建一个随机向量张量
            is_keyframe: 是否为关键帧
            
        Returns:
            VectorData对象
        """
        # 如果没有提供vec_tensor，则创建一个默认的向量张量
        if vec_tensor is None:
            # 默认创建一个768维的随机向量（ViT-B/16的维度）
            vec_tensor = torch.rand(768)
        
        return cls(
            vec_tensor=vec_tensor,
            timestamp=frame_data.timestamp,
            frame_id=frame_data.frame_id,
            source_path=frame_data.source_path,
            total_frames=frame_data.total_frames,
            video_fps=frame_data.video_fps,
        )



class VectorizerBase(ABC):
    """向量化基类"""
    @abstractmethod
    def encode(self, frame):
        pass

class ViTVectorizer(VectorizerBase):
    """ViT向量化器"""
    def __init__(self):
        # TODO: 初始化ViT模型
        self.model = None  # 这里应该加载实际的ViT模型
        print("ViT向量化器已初始化")
    
    def encode(self, frame):
        # TODO: 实现实际的ViT编码
        # 返回模拟的向量
        return np.random.rand(768)  # ViT-B/16的输出维度

class VLMVectorizer(VectorizerBase):
    """VLM向量化器, 提取KV Cache均值"""
    def __init__(self):
        # TODO: 初始化VLM模型
        self.model = None  # 这里应该加载实际的VLM模型
        print("VLM向量化器已初始化")
    
    def encode(self, frame):
        # TODO: 实现实际的VLM KV Cache提取
        # 返回模拟的向量
        return np.random.rand(1024)

class FrameVectorizer:
    def __init__(self, config=None):
        """
        初始化FrameVectorizer模块
        负责把提取到的帧转换为语义向量和时间索引
        
        Args:
            config (Config): 配置对象实例
        """
        if config is None:
            config = Config()
        
        self.config = config
        
        # 从Config对象获取配置
        self.model_type = config.frame_model_type
        self.extraction_strategy = config.frame_extraction_strategy
        self.frame_interval = config.frame_interval
        self.use_vlm = config.frame_use_vlm
        
        self.vectorizer = self._initialize_vectorizer()
        self.frame_queue = None
        self.vector_queue = mp.Queue(maxsize=200)  # 使用默认队列大小
        self.running = False
        self.frame_count = 0
        
    def _initialize_vectorizer(self):
        """初始化向量化器"""
        if self.use_vlm:
            return VLMVectorizer()
        elif self.model_type == "ViT":
            return ViTVectorizer()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def set_frame_queue(self, frame_queue):
        """设置帧队列"""
        self.frame_queue = frame_queue
    
    def process_frames(self):
        """处理帧的主循环"""
        while self.running:
            try:
                if self.frame_queue and not self.frame_queue.empty():
                    frame_data = self.frame_queue.get(timeout=1)
                    
                    # 根据提取策略决定是否处理当前帧
                    if self._should_process_frame():
                        frame = frame_data['frame']
                        timestamp = frame_data['timestamp']
                        frame_id = frame_data['frame_id']
                        
                        # 向量化
                        vector_np = self.vectorizer.encode(frame)
                        # 转换为torch张量
                        vector_tensor = torch.from_numpy(vector_np)
                        
                        # 创建VectorData对象
                        vector_data = VectorData(
                            vec_tensor=vector_tensor,
                            timestamp=timestamp,
                            frame_id=frame_id,
                            source_path=frame_data.get('source_path', 'unknown'),
                            total_frames=frame_data.get('total_frames'),
                            video_fps=frame_data.get('video_fps'),
                            is_keyframe=self._is_keyframe()
                        )
                        
                        if not self.vector_queue.full():
                            self.vector_queue.put(vector_data)
                        
                        self.frame_count += 1
                
                time.sleep(0.001)
            except Exception as e:
                print(f"处理帧时出错: {e}")
                time.sleep(0.1)
    
    def _should_process_frame(self):
        """根据策略决定是否处理当前帧"""
        if self.extraction_strategy == "every_frame":
            return True
        elif self.extraction_strategy == "interval":
            return self.frame_count % self.frame_interval == 0
        elif self.extraction_strategy == "early_exit":
            # TODO: 实现Early-exit策略
            return True
        else:
            return True
    
    def _is_keyframe(self):
        """判断是否为关键帧"""
        # TODO: 实现关键帧检测逻辑
        return self.frame_count % (self.frame_interval * 10) == 0
    
    def start(self):
        """启动向量化进程"""
        self.running = True
        self.process = mp.Process(target=self.process_frames)
        self.process.start()
        print(f"FrameVectorizer进程已启动，模型: {self.model_type}")
    
    def stop(self):
        """停止向量化进程"""
        self.running = False
        if hasattr(self, 'process'):
            self.process.join(timeout=5)
        print("FrameVectorizer进程已停止")
    
    def get_vector_queue(self):
        """获取向量队列供MemoryManager使用"""
        return self.vector_queue

if __name__ == "__main__":
    # 测试代码
    frame_vectorizer = FrameVectorizer()
    try:
        frame_vectorizer.start()
        time.sleep(5)  # 运行5秒进行测试
    except KeyboardInterrupt:
        pass
    finally:
        frame_vectorizer.stop()