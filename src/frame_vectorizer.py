import multiprocessing as mp
import numpy as np
import torch
import logging
from logging.handlers import RotatingFileHandler
import time
import queue
from dataclasses import dataclass
from typing import Optional
import glob
import os
# 本项目
from src.config import Config
from src.stream_input import FrameData
from models.bge.modeling_MMRet_CLIP import CLIPModel

@dataclass
class VectorData:
    """向量数据结构体, 包含向量张量、时间戳、帧ID、视频来源、视频总帧数和视频FPS"""
    vector: np.ndarray        # 向量张量数据
    timestamp: float                # 时间戳
    frame_id: int                   # 帧ID
    source_path: str                # 视频来源: "camera"或视频文件路径
    total_frames: Optional[int] = None  # 视频总帧数，仅视频文件模式有值
    video_fps: Optional[float] = None   # 视频FPS，仅视频文件模式有值
    
    @classmethod
    def from_frame(cls, frame_data: FrameData, vector: np.ndarray = None):
        """
        从FrameData创建VectorData
        
        Args:
            frame_data: FrameData对象
            vec_tensor: 向量张量，如果为None则创建一个随机向量张量
            
        Returns:
            VectorData对象
        """
        return cls(
            vector=vector,  
            timestamp=frame_data.timestamp,
            frame_id=frame_data.frame_id,
            source_path=frame_data.source_path,
            total_frames=frame_data.total_frames,
            video_fps=frame_data.video_fps,
        )

class BGEVectorizer():
    """BGE模型向量化器"""
    def __init__(self, config: Config):
        # 加载BGE模型到GPU
        self.device = config.frame_device
        model_path = config.frame_model_path
        print(f"正在加载BGE模型, 路径: {model_path}")
        print(f"使用设备: {self.device}")
        
        # 加载CLIPModel
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.model.set_processor(model_path)
        self.processor = self.model.processor  # 确保processor作为类属性存在
        self.model.eval()
        print("BGE模型向量化器已成功初始化")
    
    def encode(self, frame):
        # 使用processor处理图像
        img = self.processor(images=frame, return_tensors="pt")['pixel_values'].to(self.device)
        with torch.no_grad():
            vector = self.model.encode_image(images=img)
        return vector
    
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
        
        self.vectorizer = None
        self.frame_queue = None
        # 创建向量队列，使用multiprocessing.Queue以支持多进程间通信
        self.vector_queue = mp.Queue(maxsize=100)
        self.running = False
        self.vectorized_frame_count = 0
        self.all_frame_count = 0
        
    def _set_logger(self):
        """设置日志记录器"""
        log_file = self.config.frame_log_file
        pattern = log_file.replace(".log", "*")
        log_files = glob.glob(pattern)
        for f in log_files:
            try:
                os.remove(f)
            except Exception as e:
                pass

        self.logger = logging.getLogger(name='FrameVectorizer')
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

    def _initialize_vectorizer(self):
        """初始化向量化器"""
        if self.model_type == "BGE":
            self.vectorizer = BGEVectorizer(self.config)
        elif self.model_type == "ViT":
            # 为了兼容性保留ViT选项，但实际上使用BGE
            print("注意: 当前配置为ViT: 但将使用BGE模型")
            self.vectorizer = BGEVectorizer(self.config)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _preprocess_single_frame(self, frame):
        """
        处理单帧
        进来时是[H, W, C]的uint8格式的RGB的numpy的array
        转换成(待定)
        """
        # frame = Image.fromarray(frame)
        return frame

    def _vectorize_frames(self):
        """处理帧的主循环"""
        start_time = time.time()
        last_vectorized_time = start_time
        while self.running_event.is_set():
            self.logger.info(f"尝试从帧队列获取数据... 当前队列大小估计: {self.frame_queue.qsize()}")
            frame_data: FrameData = self.frame_queue.get() 
            # 直接尝试访问FrameData对象的属性
            frame = frame_data.frame
            timestamp = frame_data.timestamp
            frame_id = frame_data.frame_id
            
            # 根据提取策略决定是否处理当前帧
            if self._should_process_frame(frame_data):
                self.logger.info(f"处理帧数据: frame_id={frame_id}")
                # 预处理
                frame = self._preprocess_single_frame(frame)
                # 向量化
                vector_tensor = self.vectorizer.encode(frame)
                vector = vector_tensor.cpu().numpy()
                self.logger.debug(f"帧 {frame_id} 向量化完成，向量形状: {vector.shape}")

                # 创建VectorData对象
                vector_data = VectorData(
                    vector=vector,
                    timestamp=timestamp,
                    frame_id=frame_id,
                    source_path=frame_data.source_path,
                    total_frames=frame_data.total_frames,
                    video_fps=frame_data.video_fps
                )
                
                # 放入向量队列
                self.logger.debug(f"尝试将帧 {frame_id} 的向量化数据放入向量队列...")
                self.vector_queue.put(vector_data)  # 增加超时时间
                self.logger.debug(f"帧 {frame_id} 的向量化数据成功放入向量队列")
                self.vectorized_frame_count += 1
                # 计算并打印处理速度
                current_time = time.time()
                
                frames_per_second = 1.0 / (current_time - last_vectorized_time)
                self.logger.debug(f"当前编码速度(FPS): {frames_per_second:.2f} 帧/秒")
                self.last_vectorized_frame_count = self.vectorized_frame_count
                last_vectorized_time = current_time
            else:
                self.logger.debug(f"跳过帧数据: frame_id={frame_id}")
            self.all_frame_count += 1

        return

    def _process_main(self):
        """处理帧的主循环"""
        self._set_logger()
        self.logger.info(f"子进程启动, 进程ID: {mp.current_process().pid}")
        self._initialize_vectorizer()
        assert self.frame_queue is not None
        self._vectorize_frames()

    
    def _should_process_frame(self, frame_data: FrameData = None):
        """根据策略决定是否处理当前帧"""
        if self.extraction_strategy == "every_frame":
            return True
        elif self.extraction_strategy == "interval":
            return self.vectorized_frame_count % self.frame_interval == 0
        else:
            return True
    
    def _is_keyframe(self):
        """判断是否为关键(该函数还没有使用)"""
        # TODO: 实现关键帧检测逻辑
        is_keyframe = self.vectorized_frame_count % (self.frame_interval * 10) == 0
        return is_keyframe
    
    def start(self):
        """启动向量化进程"""
        # 创建一个共享变量来控制子进程运行
        self.running_event = mp.Event()
        self.running_event.set()
        # 设置线程为daemon模式，确保主程序退出时线程也会退出
        self.process = mp.Process(target=self._process_main, daemon=True)
        if hasattr(self.process, 'name'):
            self.process.name = "FrameVectorizer-Processor"
        self.process.start()
    
    def start_single_process(self):
        """启动单进程向量化"""
        self._process_main()

    def stop(self):
        """停止向量化进程"""
        if hasattr(self, 'process') and self.process.is_alive():
            self.process.join(timeout=5)
        print("FrameVectorizer进程已停止")
    
    def set_frame_queue(self, frame_queue):
        """设置帧队列"""
        self.frame_queue = frame_queue

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