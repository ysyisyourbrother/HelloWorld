import multiprocessing as mp
import numpy as np
import time
import torch
import logging
from logging.handlers import RotatingFileHandler
import queue
import glob
import os
from dataclasses import dataclass
from typing import Optional
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

@dataclass
class QueryVectorData:
    """向量数据结构体, 包含向量张量、查询ID、对话ID和时间戳"""
    vector: np.ndarray        # 向量张量数据
    query_id: int             # 查询ID
    dialog_id: int            # 对话ID
    timestamp: float          # 时间戳

class TextBGEVectorizer:
    """BGE模型向量化器"""
    def __init__(self, config: Config):
        self.device = config.query_device
        model_path = config.query_model_path
        
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
        self.config = config
        
        # 从Config对象获取配置
        self.model_type = config.query_model_type
        
        self.vectorizer = None
        self.query_queue = None
        # 创建向量队列，使用multiprocessing.Queue以支持多进程间通信
        self.query_vector_queue = mp.Queue(maxsize=100)
        self.running = False
        self.vectorized_query_count = 0
        self.all_query_count = 0
    
    def _set_logger(self):
        """设置日志记录器"""
        log_file = self.config.query_log_file
        pattern = log_file.replace(".log", "*")
        log_files = glob.glob(pattern)
        for f in log_files:
            try:
                os.remove(f)
            except Exception as e:
                pass

        self.logger = logging.getLogger(name='QueryVectorizer')
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
            self.vectorizer = TextBGEVectorizer(self.config)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _vectorize_queries(self):
        """处理查询的主循环"""
        start_time = time.time()
        last_vectorized_time = start_time
        while self.running_event.is_set():
            self.logger.debug(f"尝试从查询队列获取数据... 当前队列大小: {self.query_queue.qsize()}")
            query_data: QueryData = self.query_queue.get()
            query = query_data.query
            query_id = query_data.query_id
            dialog_id = query_data.dialog_id
            timestamp = query_data.timestamp
            
            # 向量化
            vector = self.vectorizer.encode(query)
            self.logger.debug(f"查询 {query_id} 向量化完成, {vector.shape}, {vector.dtype}, {type(vector)}")
            
            # 创建向量数据对象
            vector_data = QueryVectorData(
                vector=vector,
                query_id=query_id,
                dialog_id=dialog_id,
                timestamp=timestamp
            )
            
            # 放入向量队列
            self.logger.debug(f"尝试将查询 {query_id} 的向量化数据放入向量队列...")
            self.query_vector_queue.put(vector_data)
            self.logger.debug(f"查询 {query_id} 的向量化数据成功放入向量队列")
            self.vectorized_query_count += 1
            
            # 计算并打印处理速度
            current_time = time.time()
            queries_per_second = 1.0 / (current_time - last_vectorized_time)
            self.logger.debug(f"当前编码速度: {queries_per_second:.2f} 查询/秒")
            self.last_vectorized_query_count = self.vectorized_query_count
            last_vectorized_time = current_time
            
            self.all_query_count += 1
    
    def _process_main(self):
        """处理查询的主循环"""
        self._set_logger()
        self.logger.info(f"子进程启动, 进程ID: {mp.current_process().pid}")
        self._initialize_vectorizer()
        assert self.query_queue is not None
        self._vectorize_queries()
    
    def start(self):
        """启动向量化进程"""
        # 创建一个共享变量来控制子进程运行
        self.running_event = mp.Event()
        self.running_event.set()
        # 设置线程为daemon模式，确保主程序退出时线程也会退出
        self.process = mp.Process(target=self._process_main, daemon=True)
        if hasattr(self.process, 'name'):
            self.process.name = "QueryVectorizer-Processor"
        self.process.start()
    
    def start_single_process(self):
        """启动单进程向量化"""
        # 创建一个事件对象来控制进程运行
        self.running_event = mp.Event()
        self.running_event.set()
        self._process_main()
    
    def stop(self):
        """停止向量化进程"""
        # 清除running_event标志，通知子进程停止
        if hasattr(self, 'running_event'):
            self.running_event.clear()
        # 等待子进程结束
        if hasattr(self, 'process') and self.process.is_alive():
            self.process.join(timeout=5)
        print("QueryVectorizer进程已停止")
    
    def set_query_queue(self, query_queue):
        """设置查询队列"""
        self.query_queue = query_queue
    
    def get_vector_queue(self):
        """获取向量队列供后续处理使用"""
        return self.query_vector_queue

if __name__ == "__main__":
    config = Config()
    vectorizer = TextBGEVectorizer(config)
    query = "你好"
    vector = vectorizer.encode(query)
    print(vector.shape)