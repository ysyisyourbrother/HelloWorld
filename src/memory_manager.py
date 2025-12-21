import threading
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import time
import queue
import json
import glob
import os
import faiss
from dataclasses import dataclass
from typing import List, Optional, Tuple
from contextlib import contextmanager
import multiprocessing as mp

# 本项目
from src.config import Config
from src.frame_vectorizer import FrameVectorData
from src.query_vectorizer import QueryVectorData

# 视频读取库
import cv2
import decord

@dataclass
class QueryResult:
    """查询结果结构体, 包含查询ID、对话ID和匹配的向量ID列表"""
    query_id: int             # 查询ID
    dialog_id: int            # 对话ID
    scores: List[float]       # 匹配分数列表
    frame_data_list: List[any]  # 匹配的帧数据列表
    timestamp: float          # 时间戳

class ThreadSafeFaiss:
    """线程安全的Faiss索引类"""
    def __init__(self, index: faiss.Index):
        self._index = index
        self._lock = threading.RLock()

    @contextmanager
    def acquire(self):
        """上下文管理器，用于自动获取和释放锁"""
        try:
            self._lock.acquire()
            yield self._index
        finally:
            self._lock.release()

    def save_local(self, path: str = None):
        with self.acquire():
            faiss.write_index(self._index, path)

    def delete(self):
        ret = []
        with self.acquire():
            # Faiss的IndexFlatL2/IndexFlatIP没有docstore属性，直接重置索引
            self._index.reset()
        return ret

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """线程安全的搜索方法"""
        with self.acquire():
            return self._index.search(query_vector, top_k)


class ThreadSafeMap:
    """线程安全的id-frame映射类, 用于管理databasemap"""
    def __init__(self, map_data: list = None):
        self._map = map_data if map_data is not None else []
        self._lock = threading.RLock()

    @contextmanager
    def acquire(self):
        """上下文管理器，用于自动获取和释放锁"""
        try:
            self._lock.acquire()
            yield self._map
        finally:
            self._lock.release()

    def save_local(self, path: str = None):
        """保存databasemap到本地文件"""
        with self.acquire():
            # 确保目录存在
            dir_path = os.path.dirname(path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
            # 保存到文件
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self._map, f, ensure_ascii=False, indent=2)

    def load_local(self, path: str = None):
        """从本地文件加载databasemap"""
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                map_data = json.load(f)
            if not isinstance(map_data, list):
                return False
            with self.acquire():
                self._map = map_data
            return True
        return False

    def append(self, item: dict):
        """添加一个项目到databasemap"""
        with self.acquire():
            self._map.append(item)

    def get(self, index: int):
        """获取指定索引的项目"""
        with self.acquire():
            if 0 <= index < len(self._map):
                return self._map[index]
            return None

    def clear(self):
        """清空databasemap"""
        with self.acquire():
            self._map.clear()

    def __len__(self):
        """获取databasemap的长度"""
        with self.acquire():
            return len(self._map)

    def __getitem__(self, index: int):
        """获取指定索引的项目"""
        with self.acquire():
            if 0 <= index < len(self._map):
                return self._map[index]
            else:
                raise IndexError("databasemap index out of range")

    def __setitem__(self, index: int, item: dict):
        """设置指定索引的项目"""
        with self.acquire():
            if 0 <= index < len(self._map):
                self._map[index] = item
            else:
                raise IndexError("databasemap index out of range")

    def __delitem__(self, index: int):
        """删除指定索引的项目"""
        with self.acquire():
            if 0 <= index < len(self._map):
                del self._map[index]
            else:
                raise IndexError("databasemap index out of range")

    def append(self, item: dict):
        """添加一个项目到databasemap"""
        with self.acquire():
            self._map.append(item)


class MemoryManager:
    """内存管理器"""
    def __init__(self, config: Config = None):
        """
        初始化MemoryManager模块
        负责维护Video向量数据库, 包括加载、更新、保存和查询
        
        Args:
            config (Config): 配置对象实例
        """
        if config is None:
            config = Config()
        
        # 从Config对象获取配置
        self.database_type = config.memory_database_type # 向量 或 其他
        
        # logger配置
        self.log_file = config.memory_log_file

        # 向量数据库相关
        self.index = None
        self.dimension = None
        self.faiss_index_type = config.memory_faiss_index_type 
        self.faiss_file_path = config.memory_faiss_file_path  # faiss文件路径
        self.dimension = config.memory_dimension  # 向量维度
        self.databasemap_file_path = config.memory_databasemap_file_path  # databasemap文件路径
        self.databasemap = None  # 线程安全的databasemap
        self.retrieval_strategy = config.memory_retrieval_strategy
        self.max_size = config.memory_max_size
        self.memory_topk = config.memory_topk  # topk参数
        
        # 队列相关
        self.memory_mode = config.memory_mode
        self.frame_vector_queue = None  # 需要由frame_vectorizer设置
        self.query_vector_queue = None  # 需要由query_vectorizer设置
        self.query_result_queue = mp.Queue(maxsize=100)  # 用于返回查询结果
        
        self.running = False
        self.vector_count = 0
        
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

        self.logger = logging.getLogger(name='MemoryManager')
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
    
    def _initialize_database(self):
        """初始化向量数据库和databasemap"""
        # 尝试从本地加载faiss文件
        if os.path.isfile(self.faiss_file_path):
            self.logger.info(f"从本地文件 {self.faiss_file_path} 加载向量数据库")
            local_faiss = faiss.read_index(self.faiss_file_path)
            self.index = ThreadSafeFaiss(local_faiss)
            self.dimension = local_faiss.d
            self.vector_count = local_faiss.ntotal
            self.logger.info(f"数据库加载完成，包含 {self.vector_count} 个向量，维度 {self.dimension}")
            
            # 加载databasemap文件
            self.databasemap = ThreadSafeMap()  # 线程安全的databasemap
            if self.databasemap.load_local(self.databasemap_file_path):
                self.logger.info(f"databasemap加载完成, 包含 {len(self.databasemap)} 条记录")
                # 验证databasemap与faiss索引的一致性
                if len(self.databasemap) != self.vector_count:
                    self.logger.warning(f"databasemap记录数({len(self.databasemap)})与faiss索引向量数({self.vector_count})不一致")
            else:
                self.logger.warning(f"databasemap文件 {self.databasemap_file_path} 不存在或加载失败")
        else:
            self.logger.info(f"本地文件 {self.faiss_file_path} 不存在，将创建新的向量数据库")
            if self.faiss_index_type == "FlatL2":
                local_faiss = faiss.IndexFlatL2(self.dimension)
            elif self.faiss_index_type == "FlatIP":
                local_faiss = faiss.IndexFlatIP(self.dimension)
            else:
                raise ValueError(f"不支持的faiss索引类型: {self.faiss_index_type}")
            self.index = ThreadSafeFaiss(local_faiss)
            # 初始化空的databasemap
            self.databasemap = ThreadSafeMap()

    def _save_database(self):
        """保存向量数据库到本地"""
        if self.index is not None and self.vector_count > 0:
            # 保存faiss索引
            self.index.save_local(self.faiss_file_path)
            # 保存databasemap
            self.databasemap.save_local(self.databasemap_file_path)
            
            self.logger.info(f"向量数据库已保存到 {self.faiss_file_path}，包含 {self.vector_count} 个向量")
            self.logger.info(f"databasemap已保存到 {self.databasemap_file_path}，包含 {len(self.databasemap)} 条记录")
    
    def _add_vector(self, vector_data: FrameVectorData):
        """添加单个向量到数据库"""
        vector = vector_data.vector
        # 确保向量是二维数组 [1, dim]
        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)
        
        # 线程安全地添加向量到索引
        with self.index.acquire() as faiss_index:
            faiss_index.add(vector)
        
        # 使用ThreadSafeMap的append方法添加记录
        # 创建databasemap记录
        db_record = {
            "source_path": vector_data.source_path,
            "frame_id": vector_data.frame_id,
            "timestamp": vector_data.timestamp,
            "total_frames": vector_data.total_frames,
            "video_fps": vector_data.video_fps
        }
        self.databasemap.append(db_record)
        
        self.vector_count += 1
        
        self.logger.info(f"向量添加成功, ID: {self.vector_count-1}, 总向量数: {self.vector_count}")
    
    def _thread_frame_vectors(self):
        """处理帧向量的线程"""
        self.logger.info(f"帧向量处理线程启动, 线程名: {threading.current_thread().name}")
        
        save_interval = 30  # 默认30秒保存一次
        
        while self.running_event.is_set():
            try:
                # 尝试从队列获取数据，超时时间设置为save_interval
                vector_data: FrameVectorData = self.frame_vector_queue.get(timeout=save_interval)
                self._add_vector(vector_data)
            except queue.Empty:
                # 队列超时，保存数据库
                self.logger.debug("帧向量队列超时，保存数据库")
                self._save_database()
            except Exception as e:
                self.logger.error(f"处理帧向量时出错: {e}")
    
    def _query_faiss(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[List[int], List[float]]:
        """查询向量数据库
        
        Args:
            query_vector: 查询向量
            top_k: 返回前k个最相似的向量
            
        Returns:
            tuple: (向量ID列表, 相似度分数列表)
        """
        if self.index is None or self.vector_count == 0:
            self.logger.warning("向量数据库为空，无法执行查询")
            return [], []
        
        # 确保查询向量是二维数组 [1, dim]
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # 执行查询
        distances, indices = self.index.search(query_vector, top_k)
        
        # 转换为列表
        vector_ids = indices[0].tolist()
        scores = distances[0].tolist()
        
        self.logger.info(f"查询完成，返回 {len(vector_ids)} 个结果")
        return vector_ids, scores
    
    def _read_frame_from_video(self, vector_ids: List[int]) -> Optional[np.ndarray]:
        """根据vector_ids读取实际帧数据
        """
        frames = []
        for vector_id in vector_ids:
            db_record = self.databasemap[vector_id]

            cap = cv2.VideoCapture(db_record["source_path"])
            if not cap.isOpened():
                self.logger.error(f"无法打开视频文件: {db_record['source_path']}")
                return None

            cap.set(cv2.CAP_PROP_POS_FRAMES, db_record["frame_id"])
            ret, frame = cap.read()
            if not ret:
                self.logger.error(f"无法读取视频帧 {db_record['frame_id']}")
                continue
            
            frames.append(frame)

        return frames
    
    def _retrieve(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[List[FrameVectorData], List[float]]:
        """根据查询向量检索匹配的帧数据
        
        Args:
            query_vector: 查询向量
            top_k: 返回前k个最相似的帧数据
            
        Returns:
            tuple: (匹配的帧数据列表, 相似度分数列表)
        """
        vector_ids, scores = self._query_faiss(query_vector, top_k)
        self.logger.info(f"查询完成，返回 {len(vector_ids)} 个结果")
        # 获取对应的帧数据
        frame_data_list = self._read_frame_from_video(vector_ids)
        
        self.logger.debug(f"成功读取 {len(frame_data_list)} 个实际帧")
        return frame_data_list, scores
    
    def _thread_query_vectors(self):
        """处理查询向量的线程"""
        self.logger.info(f"查询向量处理线程启动, 线程名: {threading.current_thread().name}")
        
        while self.running_event.is_set():
            # 从队列获取查询向量数据
            self.logger.info(f"等待查询向量...")
            query_data: QueryVectorData = self.query_vector_queue.get()
            query_vector = query_data.vector
            query_id = query_data.query_id
            dialog_id = query_data.dialog_id
            timestamp = query_data.timestamp
            
            # 执行查询
            frame_data_list, scores = self._retrieve(query_vector, self.memory_topk)
            
            # 创建查询结果
            result = QueryResult(
                query_id=query_id,
                dialog_id=dialog_id,
                scores=scores,
                frame_data_list=frame_data_list,
                timestamp=timestamp
            )
            
            # 将结果放入结果队列
            self.query_result_queue.put(result)
            self.logger.info(f"查询 {query_id} 处理完成，返回 {len(frame_data_list)} 个结果")
    
    def _process_main(self):
        """子进程，根据配置启动相应的线程"""
        self._set_logger()
        self.logger.info(f"MemoryManager启动, 进程ID: {os.getpid()}")
        
        # 初始化数据库
        self._initialize_database()
        
        # 根据memory_mode决定启动哪些线程
        memory_mode = self.memory_mode
        
        self.logger.info(f"MemoryManager模式: {memory_mode}")
        
        threads = []
        
        # 启动帧向量处理线程（用于inject）
        if memory_mode in ["only_inject", "both"]:
            frame_thread = threading.Thread(target=self._thread_frame_vectors, daemon=True)
            frame_thread.name = "FrameVectorThread"
            threads.append(frame_thread)
            self.logger.info("帧向量处理线程已创建")
        
        # 启动查询向量处理线程（用于query）
        if memory_mode in ["only_query", "both"]:
            query_thread = threading.Thread(target=self._thread_query_vectors, daemon=True)
            query_thread.name = "QueryThread"
            threads.append(query_thread)
            self.logger.info("查询向量处理线程已创建")
        
        # 验证至少启动了一个线程
        if len(threads) == 0:
            self.logger.error("没有启动任何线程，请检查memory_mode配置")
            return
        
        # 启动所有创建的线程
        for thread in threads:
            thread.start()
            self.logger.info(f"线程 {thread.name} 已启动")
        
        # 保持子进程运行，等待线程完成
        while self.running_event.is_set():
            time.sleep(1.0)
    
    def start(self):
        """启动MemoryManager"""
        # 创建一个事件对象来控制线程运行
        self.running_event = mp.Event()
        self.running_event.set()
        
        # 在子进程中运行_process_main
        self.process = mp.Process(target=self._process_main, daemon=True)
        if hasattr(self.process, 'name'):
            self.process.name = "MemoryManager-Processor"
        self.process.start()
    
    def start_single_thread(self):
        """启动单线程运行MemoryManager"""
        # 创建一个事件对象来控制线程运行
        self.running_event = threading.Event()
        self.running_event.set()
        
        # 直接调用_process_main，在当前线程中运行
        self._process_main()
    
    def stop(self):
        """停止MemoryManager"""
        if hasattr(self, 'running_event'):
            self.running_event.clear()
        
        # 保存数据库 - 只有在当前进程有index的情况下才保存
        if self.index is not None:
            self._save_database()
        
        # 等待子进程结束
        if hasattr(self, 'process') and self.process.is_alive():
            # 只有在logger已初始化的情况下才记录日志
            self.process.join(timeout=5)
        
    
    def set_frame_vector_queue(self, frame_vector_queue: mp.Queue):
        """设置帧向量队列"""
        self.frame_vector_queue = frame_vector_queue
    
    def set_query_vector_queue(self, query_vector_queue: mp.Queue):
        """设置查询向量队列"""
        self.query_vector_queue = query_vector_queue
    
    def get_query_result_queue(self):
        """获取查询结果队列"""
        return self.query_result_queue