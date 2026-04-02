import numpy as np
import time
import torch
import logging
from logging.handlers import RotatingFileHandler
import glob
import os
from dataclasses import dataclass
from typing import Optional, Dict
# 本项目
from src.config import Config
from models.bge.modeling_MMRet_CLIP import CLIPModel, CLIPProcessor

@dataclass
class QueryData:
    """查询数据结构体, 包含查询文本、查询ID和时间戳"""
    query: str                # 查询文本
    query_id: int             # 查询ID
    dialog_id: int            # 对话ID
    timestamp: float          # 时间戳, 用于系统测时
    trace_ts: Optional[Dict[str, float]] = None  # 链路时延埋点（秒）

@dataclass
class QueryVectorData:
    """向量数据结构体, 包含向量张量、查询ID、对话ID和时间戳"""
    vector: np.ndarray        # 向量张量数据
    query_id: int             # 查询ID
    dialog_id: int            # 对话ID
    timestamp: float          # 时间戳
    trace_ts: Optional[Dict[str, float]] = None  # 链路时延埋点（秒）

class TextBGEVectorizer:
    """BGE模型向量化器"""
    def __init__(self, device: str, model_path: str):
        self.device = device
        
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.model.set_processor(model_path)
        self.processor = self.model.processor
        self.model.eval()
    
    def encode(self, query: str):
        """
        对查询文本进行向量化
        TODO: 目前只支持单次 77 token 的编码
        """
        txt = self.processor(text=query, 
                            return_tensors="pt", # Return PyTorch `torch.Tensor` objects.
                            padding=True, 
                            truncation=True, 
                            max_length=77)
        txt = {k: v.to(self.device) for k, v in txt.items()}
        with torch.no_grad():
            vector = self.model.encode_text(txt)
            return vector.cpu().numpy()

class QueryVectorizer:
    """查询向量化器"""
    def __init__(self, config: Config = None):
        """
        初始化QueryVectorizer模块
        负责把查询文本转换为语义向量
        
        Args:
            config (Config): 配置对象实例
        """
        if config is None:
            config = Config()
        
        # 从Config对象获取配置
        self.model_type = config.query_model_type
        self.log_file = config.query_log_file
        self.query_device = config.query_device
        self.query_model_path = config.query_model_path
        
        self.vectorizer = None
        self.vectorized_query_count = 0
        self.all_query_count = 0
    
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

        self.logger = logging.getLogger(name='QueryVectorizer')
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
    
    def _initialize_vectorizer(self):
        """初始化向量化器"""
        if self.model_type == "BGE":
            self.vectorizer = TextBGEVectorizer(self.query_device, self.query_model_path)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def encode_query_data(self, query_data: QueryData) -> QueryVectorData:
        """
        将 QueryData 编码为 QueryVectorData，供 MemoryManager 查询线程调用。
        需先调用 _set_logger() 与 _initialize_vectorizer()（由 MemoryManager 在子进程内完成）。
        """
        if not hasattr(self, "logger") or self.logger is None:
            self._set_logger()
        if self.vectorizer is None:
            self._initialize_vectorizer()

        query = query_data.query
        query_id = query_data.query_id
        dialog_id = query_data.dialog_id
        timestamp = query_data.timestamp
        trace_ts = dict(query_data.trace_ts or {})
        trace_ts["query_vectorizer_dequeue_at"] = time.time()
        if "api_query_enqueued_at" in trace_ts:
            self.logger.info(
                f"[Latency][Query] api->memory_query_vectorizer query_id={query_id} "
                f"{(trace_ts['query_vectorizer_dequeue_at'] - trace_ts['api_query_enqueued_at']) * 1000:.2f} ms"
            )

        encode_start = time.time()
        vector = self.vectorizer.encode(query)
        trace_ts["query_vectorizer_encoded_at"] = time.time()
        self.logger.info(
            f"[Latency][Query] query_vectorize query_id={query_id} "
            f"{(trace_ts['query_vectorizer_encoded_at'] - encode_start) * 1000:.2f} ms"
        )
        self.logger.debug(f"查询 {query_id} 向量化完成, {vector.shape}, {vector.dtype}, {type(vector)}")
        self.vectorized_query_count += 1
        self.all_query_count += 1

        return QueryVectorData(
            vector=vector,
            query_id=query_id,
            dialog_id=dialog_id,
            timestamp=timestamp,
            trace_ts=trace_ts,
        )

    def encode_query_sync(self, query_text: str) -> np.ndarray:
        """同步编码查询文本，供 benchmark 使用。需先调用 _initialize_vectorizer()。"""
        if self.vectorizer is None:
            self._initialize_vectorizer()
        return self.vectorizer.encode(query_text)

if __name__ == "__main__":
    config = Config()
    vectorizer = TextBGEVectorizer(config.query_device, config.query_model_path)
    query = "你好"
    vector = vectorizer.encode(query)
    print(vector.shape)