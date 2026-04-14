import numpy as np
import random
import torch
import logging
from logging.handlers import RotatingFileHandler
import time
from dataclasses import dataclass
from typing import Any, Optional, List, Callable, Tuple, Dict
import glob
import os
# 本项目
from src.config import Config
from src.memory.image_bge_vectorizer import ImageBGEVectorizer
from src.video_input.video_input import FrameData, SymFrameData, SymVideoInputByGOP
from src.video_utils.file_video_reader import open_file_video_reader

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
    trace_ts: Optional[Dict[str, float]] = None  # 链路时延埋点（秒）
    
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
        self.frame_attn_implementation = config.frame_attn_implementation

        self.vectorizer = None
        self.vectorized_frame_count = 0
        self.all_frame_count = 0
        self._video_readers = {}

        # 编码钩子：在 encode_frames_batch 时调用，fn(frame_data_list, vectors, hidden_states, attentions)
        self._encode_hooks: List[Callable] = []
        self._encode_hooks_need_hidden_states = False
        self._encode_hooks_need_attentions = False
        self._video_reader_backend = getattr(config, "video_reader_backend", "auto")

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
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
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
        attn_impl = self.frame_attn_implementation
        if self._encode_hooks_need_attentions:
            attn_impl = "eager"
        if self.model_type == "BGE":
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

    def _resolve_frame_from_reference(self, frame_data: FrameData) -> Optional[np.ndarray]:
        """当队列中仅传引用时，在本进程按 source_path+frame_id 解码帧。"""
        if frame_data.frame is not None:
            return frame_data.frame
        source_path = frame_data.source_path
        frame_id = frame_data.frame_id
        if not source_path or frame_id is None:
            return None
        try:
            vr = self._video_readers.get(source_path)
            if vr is None:
                vr = open_file_video_reader(source_path, self._video_reader_backend)
                self._video_readers[source_path] = vr
            return vr[frame_id].asnumpy()
        except Exception as e:
            self.logger.error(f"按引用解码帧失败 source={source_path}, frame_id={frame_id}: {e}")
            return None

    def _should_process_frame(self, frame_data: FrameData = None):
        """根据帧间隔决定是否处理当前帧"""
        return self.all_frame_count % self.frame_interval == 0

    def encode_frame_from_stream(self, frame_data: FrameData) -> Optional[FrameVectorData]:
        """
        对单条入队帧做间隔筛选与编码，供 MemoryManager 注入线程调用。
        需先调用 _set_logger() 与 _initialize_vectorizer()（由 MemoryManager 在子进程内完成）。
        """
        if not hasattr(self, "logger") or self.logger is None:
            self._set_logger()
        if self.vectorizer is None:
            self._initialize_vectorizer()

        frame = self._resolve_frame_from_reference(frame_data)
        timestamp = frame_data.timestamp
        frame_id = frame_data.frame_id
        trace_ts = dict(frame_data.trace_ts or {})
        trace_ts["frame_vectorizer_dequeue_at"] = time.time()
        if "video_extracted_at" in trace_ts:
            self.logger.info(
                f"[Latency][Inject] video->memory_frame_vectorizer frame_id={frame_id} "
                f"{(trace_ts['frame_vectorizer_dequeue_at'] - trace_ts['video_extracted_at']) * 1000:.2f} ms"
            )
        if frame is None:
            self.logger.warning(f"跳过无像素帧 frame_id={frame_id}, source={frame_data.source_path}")
            self.all_frame_count += 1
            return None

        if self._should_process_frame(frame_data):
            frame = self._preprocess_single_frame(frame)
            encode_start = time.time()
            vector_tensor = self.vectorizer.encode(frame)
            vector = vector_tensor.cpu().numpy()
            trace_ts["frame_vectorizer_encoded_at"] = time.time()
            self.logger.info(
                f"[Latency][Inject] vectorize frame_id={frame_id} "
                f"{(trace_ts['frame_vectorizer_encoded_at'] - encode_start) * 1000:.2f} ms"
            )
            self.logger.debug(f"帧 {frame_id} 向量化完成, {vector.shape}, {vector.dtype}, {type(vector)}")
            self.vectorized_frame_count += 1
            self.all_frame_count += 1
            return FrameVectorData(
                vector=vector,
                timestamp=timestamp,
                frame_id=frame_id,
                source_path=frame_data.source_path,
                total_frames=frame_data.total_frames,
                video_fps=frame_data.video_fps,
                duration=frame_data.duration,
                trace_ts=trace_ts,
            )
        self.logger.debug(f"跳过帧数据: frame_id={frame_id}")
        self.all_frame_count += 1
        return None

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


class SymFrameVectorizerByGOP(FrameVectorizer):
    """
    继承 FrameVectorizer，支持按 GOP 编码：通过 select_frame_in_gop 筛选帧后仅编码选中的帧。
    """

    def __init__(self, config: Config = None):
        super().__init__(config)
        self.select_strategy = config.frame_select_strategy  # "random" | "first" | "pktsize"

    def _resolve_sym_frame_from_reference(self, frame_data: SymFrameData) -> Optional[np.ndarray]:
        """Sym 帧在仅传引用时，按 source_path+frame_id 解码。"""
        if frame_data.frame is not None:
            return frame_data.frame
        source_path = frame_data.source_path
        frame_id = frame_data.frame_id
        if not source_path or frame_id is None:
            return None
        try:
            vr = self._video_readers.get(source_path)
            if vr is None:
                vr = open_file_video_reader(source_path, self._video_reader_backend)
                self._video_readers[source_path] = vr
            return vr[frame_id].asnumpy()
        except Exception:
            return None

    def select_frame_indices_in_gop(
        self,
        gop_start: int,
        gop_end: int,
        frame_types: List[str],
        pkt_sizes: List[int],
    ) -> List[int]:
        """
        根据 select_strategy 返回该 GOP 内要解码的帧索引（先筛选再解码，避免解码全部帧）。

        Args:
            gop_start: GOP 起始帧索引（含）
            gop_end: GOP 结束帧索引（不含）
            frame_types: 全视频的 pict_type 列表
            pkt_sizes: 全视频的 pkt_size 列表

        Returns:
            要提取的帧索引列表
        """
        n = gop_end - gop_start
        if n <= 0:
            return []

        if self.select_strategy == "first":
            return [gop_start]
        elif self.select_strategy == "random":
            idx = random.randint(0, n - 1)
            return [gop_start + idx]
        elif self.select_strategy == "pktsize":
            raise NotImplementedError("select_strategy 'pktsize' 暂未实现")
        else:
            if hasattr(self, "logger"):
                self.logger.warning(f"未知的 select_strategy '{self.select_strategy}'，回退为 first")
            return [gop_start]

    def select_frame_in_gop(self, gop_frames: List[SymFrameData]) -> List[SymFrameData]:
        """
        根据 select_strategy 从 GOP 帧列表中筛选要编码的帧。
        注意：会先解码全部帧再筛选，仅当已有完整帧数据时使用；否则用 encode_frames_by_gop_from_video_input。
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

    def encode_frames_by_gop_from_video_input(
        self, video_input: SymVideoInputByGOP, gop_start: int, gop_end: int
    ) -> List[FrameVectorData]:
        """
        对单个 GOP 先按 select_strategy 选索引，仅解码选中帧，再编码。
        避免解码全部 2227 帧，大幅提升速度。

        Args:
            video_input: SymVideoInput 实例（含 frame_types、pkt_sizes）
            gop_start: GOP 起始帧索引
            gop_end: GOP 结束帧索引

        Returns:
            编码后的 FrameVectorData 列表
        """
        indices = self.select_frame_indices_in_gop(
            gop_start, gop_end,
            video_input.frame_types, video_input.pkt_sizes,
        )
        if not indices:
            return []

        selected = [video_input._extract_video_frame_at(i) for i in indices]
        selected = [s for s in selected if s is not None]
        if not selected:
            return []

        if self.vectorizer is None:
            self._initialize_vectorizer()

        resolved_frames = [self._resolve_sym_frame_from_reference(sf) for sf in selected]
        selected_and_frames = [(sf, f) for sf, f in zip(selected, resolved_frames) if f is not None]
        if not selected_and_frames:
            return []
        selected = [x[0] for x in selected_and_frames]
        frames = [self._preprocess_single_frame(x[1]) for x in selected_and_frames]
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

        resolved_frames = [self._resolve_sym_frame_from_reference(sf) for sf in selected]
        selected_and_frames = [(sf, f) for sf, f in zip(selected, resolved_frames) if f is not None]
        if not selected_and_frames:
            return []
        selected = [x[0] for x in selected_and_frames]
        frames = [self._preprocess_single_frame(x[1]) for x in selected_and_frames]
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


class SymFrameVectorizerForV3(SymFrameVectorizerByGOP):
    """
    Symphony v3：在 GOP 版向量器之上，提供流式窗已选帧的批量编码（不做 GOP 内 random/first）。
    """

    def encode_sym_frames_list(self, sym_frames: List[SymFrameData]) -> List[FrameVectorData]:
        """
        对已选中的 SymFrameData 列表编码；调用方（如 SymVideoInputByStreamWindow）负责选帧策略。
        """
        if not sym_frames:
            return []
        if self.vectorizer is None:
            self._initialize_vectorizer()
        resolved_frames = [self._resolve_sym_frame_from_reference(sf) for sf in sym_frames]
        selected_and_frames = [(sf, f) for sf, f in zip(sym_frames, resolved_frames) if f is not None]
        if not selected_and_frames:
            return []
        frames = [self._preprocess_single_frame(x[1]) for x in selected_and_frames]
        vector_tensors = self.vectorizer.encode_batch(frames)
        vectors_np = vector_tensors.cpu().numpy()
        result = []
        for i, (sf, _) in enumerate(selected_and_frames):
            vec = vectors_np[i] if len(vectors_np.shape) > 1 else vectors_np
            if len(vec.shape) == 1:
                vec = vec.reshape(1, -1)
            ts = sf.frame_id / sf.video_fps if sf.video_fps else 0.0
            result.append(
                FrameVectorData(
                    vector=vec,
                    timestamp=ts,
                    frame_id=sf.frame_id,
                    source_path=sf.source_path,
                    total_frames=sf.total_frames,
                    video_fps=sf.video_fps,
                    duration=sf.duration,
                    trace_ts=dict(sf.trace_ts) if sf.trace_ts else None,
                )
            )
        return result
