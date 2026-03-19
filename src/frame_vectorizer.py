import multiprocessing as mp
import numpy as np
import random
import torch
import logging
from logging.handlers import RotatingFileHandler
import time
import queue
from dataclasses import dataclass
from typing import Any, Optional, List, Callable, Tuple
import glob
import os
# 本项目
from src.config import Config
from src.stream_input import FrameData, SymFrameData
from models.bge.modeling_MMRet_CLIP import CLIPModel

@dataclass
class FrameVectorData:
    """向量数据结构体, 包含向量张量、时间戳、帧ID、视频来源、视频总帧数、视频FPS和视频时长"""
    vector: np.ndarray        # 向量张量数据
    timestamp: float                # 时间戳
    frame_id: int                   # 帧ID
    source_path: str                # 视频来源: "camera"或视频文件路径
    total_frames: Optional[int] = None  # 视频总帧数，仅视频文件模式有值
    video_fps: Optional[float] = None   # 视频FPS，仅视频文件模式有值
    duration: Optional[float] = None     # 视频总时长(秒)，仅视频文件模式有值
    
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
            duration=frame_data.duration,
        )

class ImageBGEVectorizer():
    """BGE模型向量化器"""
    def __init__(self, device: str, model_path: str, attn_implementation: str = "sdpa"):
        self.device = device
        self.attn_implementation = attn_implementation

        # 加载CLIPModel，attn_implementation="eager" 时支持 output_attentions
        self.model = CLIPModel.from_pretrained(
            model_path,
            attn_implementation=attn_implementation,
        ).to(self.device)
        self.model.set_processor(model_path)
        self.processor = self.model.processor  # 确保processor作为类属性存在
        self.model.eval()

    def encode(self, frame):
        # 使用processor处理图像
        img = self.processor(images=frame, return_tensors="pt")['pixel_values'].to(self.device)
        with torch.no_grad():
            vector = self.model.encode_image(images=img)
        return vector

    def encode_batch(self, frames: list):
        """批量编码多帧图像，frames 为 numpy 数组列表 [H,W,C] RGB"""
        if not frames:
            return torch.empty(0)
        img = self.processor(images=frames, return_tensors="pt")['pixel_values'].to(self.device)
        with torch.no_grad():
            vectors = self.model.encode_image(images=img)
        return vectors

    def encode_batch_with_vision_outputs(
        self,
        frames: list,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[object]]:
        """
        批量编码并返回 vision_model 的 hidden_states 和 attentions。
        当 output_attentions=True 时，需使用 attn_implementation="eager" 初始化。

        Returns:
            (vectors, vision_outputs): vectors 为归一化后的嵌入；vision_outputs 含 .hidden_states 和 .attentions
        """
        if not frames:
            return torch.empty(0), None
        img = self.processor(images=frames, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.no_grad():
            vision_outputs = self.model.vision_model(
                pixel_values=img,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            pooled_output = vision_outputs[1]
            image_features = self.model.visual_projection(pooled_output)
            vectors = torch.nn.functional.normalize(image_features, dim=-1)
        return vectors, vision_outputs
    
class FrameVectorizer:
    def __init__(self, config: Config = None):
        """
        初始化FrameVectorizer模块
        负责把提取到的帧转换为语义向量和时间索引
        
        Args:
            config (Config): 配置对象实例
        """
        if config is None:
            config = Config()
        
        # 从Config对象获取配置
        self.model_type = config.frame_model_type
        self.frame_interval = config.frame_interval
        self.log_file = config.frame_log_file
        self.frame_device = config.frame_device
        self.frame_model_path = config.frame_model_path
        
        self.vectorizer = None
        self.frame_queue = None  # 需要由stream_input设置
        # 创建向量队列，使用multiprocessing.Queue以支持多进程间通信
        self.frame_vector_queue = mp.Queue(maxsize=100)
        self.running = False
        self.vectorized_frame_count = 0
        self.all_frame_count = 0

        # 编码钩子：在 encode_frames_batch 时调用，fn(frame_data_list, vectors, hidden_states, attentions)
        self._encode_hooks: List[Callable] = []
        self._encode_hooks_need_hidden_states = False
        self._encode_hooks_need_attentions = False
        
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
        self.logger.handlers.clear()
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

    def register_encode_hook(
        self,
        fn: Callable,
        need_hidden_states: bool = False,
        need_attentions: bool = False,
    ):
        """
        注册编码钩子，在 encode_frames_batch 时调用。

        fn(frame_data_list, vectors, hidden_states, attentions)：
          - frame_data_list: List[FrameData]
          - vectors: np.ndarray, shape (N, dim)
          - hidden_states: Optional[Tuple], 每层 (bsz, seq_len, hidden_size)，仅当 need_hidden_states=True 时有值
          - attentions: Optional[Tuple], 每层 (bsz, num_heads, seq_len, seq_len) softmax(QK^T)，仅当 need_attentions=True 时有值

        need_attentions=True 时，需使用 attn_implementation="eager" 加载模型，否则可能报错。
        """
        self._encode_hooks.append(fn)
        if need_hidden_states:
            self._encode_hooks_need_hidden_states = True
        if need_attentions:
            self._encode_hooks_need_attentions = True

    def _initialize_vectorizer(self):
        """初始化向量化器"""
        attn_impl = "eager" if self._encode_hooks_need_attentions else "sdpa"
        if self.model_type == "BGE":
            self.vectorizer = ImageBGEVectorizer(
                self.frame_device, self.frame_model_path, attn_implementation=attn_impl
            )
        elif self.model_type == "ViT":
            # 为了兼容性保留ViT选项，但实际上使用BGE
            print("注意: 当前配置为ViT: 但将使用BGE模型")
            self.vectorizer = ImageBGEVectorizer(
                self.frame_device, self.frame_model_path, attn_implementation=attn_impl
            )
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
            self.logger.debug(f"尝试从帧队列获取数据... 当前队列大小: {self.frame_queue.qsize()}")
            frame_data: FrameData = self.frame_queue.get() 
            # 直接尝试访问FrameData对象的属性
            frame = frame_data.frame
            timestamp = frame_data.timestamp
            frame_id = frame_data.frame_id
            
            # 根据提取策略决定是否处理当前帧
            if self._should_process_frame(frame_data):
                # 预处理
                frame = self._preprocess_single_frame(frame)
                # 向量化
                vector_tensor = self.vectorizer.encode(frame)
                vector = vector_tensor.cpu().numpy()
                self.logger.debug(f"帧 {frame_id} 向量化完成, {vector.shape}, {vector.dtype}, {type(vector)}")

                # 创建VectorData对象
                vector_data = FrameVectorData(
                    vector=vector,
                    timestamp=timestamp,
                    frame_id=frame_id,
                    source_path=frame_data.source_path,
                    total_frames=frame_data.total_frames,
                    video_fps=frame_data.video_fps,
                    duration=frame_data.duration
                )
                
                # 放入向量队列
                self.frame_vector_queue.put(vector_data)  
                self.logger.debug(f"帧 {frame_id} 的向量化数据成功放入向量队列")
                self.vectorized_frame_count += 1

                # 计算并打印处理速度
                current_time = time.time()
                frames_per_second = 1.0 / (current_time - last_vectorized_time)
                self.logger.debug(f"当前编码速度(FPS): {frames_per_second:.2f} 帧/秒")
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
        """根据帧间隔决定是否处理当前帧"""
        return self.all_frame_count % self.frame_interval == 0
    
    def start(self):
        """启动向量化进程"""
        # 创建一个共享变量来控制子进程运行
        self.running_event = mp.Event()
        self.running_event.set()
        # 设置线程为daemon模式，确保主程序退出时线程也会退出
        self.process = mp.Process(target=self._process_main, daemon=True)
        if hasattr(self.process, 'name'):
            self.process.name = f"{self.__class__.__name__}-Processor"
        self.process.start()
    
    def start_single_process(self):
        """启动单进程向量化"""
        self.running_event = mp.Event()
        self.running_event.set()
        self._process_main()

    def _is_process_parent(self):
        """当前进程是否为子进程的父进程（只有父进程才能安全调用 is_alive/join）"""
        if not hasattr(self, 'process'):
            return False
        parent_pid = getattr(self.process, '_parent_pid', None)
        return parent_pid is not None and parent_pid == os.getpid()

    def stop(self):
        """停止向量化进程"""
        if not hasattr(self, 'process'):
            return
        try:
            if not self._is_process_parent():
                return
            if self.process.is_alive():
                self.process.join(timeout=5)
        except (AssertionError, ValueError) as e:
            # 非父进程调用 is_alive/join 会触发 "can only test/join a child process"
            logging.getLogger(__name__).debug("停止子进程时跳过 join: %s", e)
    
    def set_frame_queue(self, frame_queue):
        """设置帧队列"""
        self.frame_queue = frame_queue

    def get_vector_queue(self):
        """获取向量队列供MemoryManager使用"""
        return self.frame_vector_queue

    def encode_frames_batch(self, frame_data_list: List[FrameData]) -> List[FrameVectorData]:
        """
        同步批量编码帧，供 benchmark 使用。需先调用 _initialize_vectorizer()。
        
        Args:
            frame_data_list: FrameData 列表
            
        Returns:
            FrameVectorData 列表
        """
        if not frame_data_list:
            return []
        if self.vectorizer is None:
            self._initialize_vectorizer()
        frames = [self._preprocess_single_frame(fd.frame) for fd in frame_data_list]

        if self._encode_hooks:
            need_hs = self._encode_hooks_need_hidden_states
            need_attn = self._encode_hooks_need_attentions
            vector_tensors, vision_outputs = self.vectorizer.encode_batch_with_vision_outputs(
                frames, output_hidden_states=need_hs, output_attentions=need_attn
            )
            vectors_np = vector_tensors.cpu().numpy()
            hidden_states = None
            attentions = None
            if vision_outputs is not None:
                if need_hs and vision_outputs.hidden_states is not None:
                    hidden_states = tuple(h.cpu().numpy() for h in vision_outputs.hidden_states)
                if need_attn and vision_outputs.attentions is not None:
                    attentions = tuple(a.cpu().numpy() for a in vision_outputs.attentions)
            for fn in self._encode_hooks:
                try:
                    fn(frame_data_list, vectors_np, hidden_states, attentions)
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.warning(f"编码钩子执行异常: {e}")
                    else:
                        logging.getLogger(__name__).warning(f"编码钩子执行异常: {e}")
        else:
            vector_tensors = self.vectorizer.encode_batch(frames)
            vectors_np = vector_tensors.cpu().numpy()

        result = []
        for i, fd in enumerate(frame_data_list):
            vec = vectors_np[i] if len(vectors_np.shape) > 1 else vectors_np
            if len(vec.shape) == 1:
                vec = vec.reshape(1, -1)
            result.append(FrameVectorData(
                vector=vec,
                timestamp=fd.timestamp,
                frame_id=fd.frame_id,
                source_path=fd.source_path,
                total_frames=fd.total_frames,
                video_fps=fd.video_fps,
                duration=fd.duration
            ))
        return result


class SymFrameVectorizer(FrameVectorizer):
    """
    继承 FrameVectorizer，支持按 GOP 编码：通过 select_frame_in_gop 筛选帧后仅编码选中的帧。
    """

    def __init__(self, config: Config = None):
        super().__init__(config)
        self.select_strategy = config.frame_select_strategy  # "random" | "first" | "pktsize"

    def select_frame_in_gop(self, gop_frames: List[SymFrameData]) -> List[SymFrameData]:
        """
        根据 select_strategy 从 GOP 帧列表中筛选要编码的帧。

        Args:
            gop_frames: 一个 GOP 内的帧列表（SymFrameData）

        Returns:
            筛选后的帧列表
        """
        if not gop_frames:
            return []

        if self.select_strategy == "first":
            return [gop_frames[0]]
        elif self.select_strategy == "random":
            return [random.choice(gop_frames)]
        elif self.select_strategy == "pktsize":
            raise NotImplementedError("select_strategy 'pktsize' 暂未实现")
        else:
            if hasattr(self, "logger"):
                self.logger.warning(f"未知的 select_strategy '{self.select_strategy}'，回退为 first")
            return [gop_frames[0]]

    def encode_frames_by_gop(self, gop_frames: List[SymFrameData]) -> List[FrameVectorData]:
        """
        对 GOP 内帧先按 select_frame_in_gop 筛选，再仅编码筛选出的帧。
        不依赖 encode_frames_batch，不触发编码钩子。

        Args:
            gop_frames: 一个 GOP 的 SymFrameData 列表

        Returns:
            编码后的 FrameVectorData 列表
        """
        selected = self.select_frame_in_gop(gop_frames)
        if not selected:
            return []

        if self.vectorizer is None:
            self._initialize_vectorizer()

        frames = [self._preprocess_single_frame(sf.frame) for sf in selected]
        vector_tensors = self.vectorizer.encode_batch(frames)
        vectors_np = vector_tensors.cpu().numpy()

        result = []
        for i, sf in enumerate(selected):
            vec = vectors_np[i] if len(vectors_np.shape) > 1 else vectors_np
            if len(vec.shape) == 1:
                vec = vec.reshape(1, -1)
            ts = sf.frame_id / sf.video_fps if sf.video_fps else 0.0
            result.append(FrameVectorData(
                vector=vec,
                timestamp=ts,
                frame_id=sf.frame_id,
                source_path=sf.source_path,
                total_frames=sf.total_frames,
                video_fps=sf.video_fps,
                duration=sf.duration,
            ))
        return result
