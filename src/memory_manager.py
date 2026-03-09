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
from typing import List, Optional, Tuple, Sequence
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
class MemoryResult:
    """查询结果结构体, 包含查询ID、对话ID和匹配的向量ID列表"""
    frame_data_list: List[any]  # 匹配的帧数据列表
    timestamp: float          # 时间戳
    query_id: int             # 查询ID
    dialog_id: int            # 对话ID
    scores: List[float]       # 匹配分数列表

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
    """线程安全的 id-frame 映射类，按视频聚合存储，减少冗余。
    内部结构: [_videos] 每项为 {"source_path", "total_frames", "video_fps", "duration", "frames": [frame_id, ...]}
    FAISS 索引 i 对应: 按顺序遍历 _videos，累加 frames 长度，定位到对应视频和 frame_id。
    """
    def __init__(self, videos: list = None):
        self._videos = videos if videos is not None else []
        self._lock = threading.RLock()

    @contextmanager
    def acquire(self):
        """上下文管理器，用于自动获取和释放锁"""
        try:
            self._lock.acquire()
            yield self._videos
        finally:
            self._lock.release()

    def _total_frames_count(self):
        """返回总帧数（向量数）"""
        return sum(len(v["frames"]) for v in self._videos)

    def _index_to_record(self, index: int) -> dict:
        """将 FAISS 索引转换为 {source_path, frame_id, total_frames, video_fps, duration}"""
        offset = 0
        for v in self._videos:
            n = len(v["frames"])
            if index < offset + n:
                return {
                    "source_path": v["source_path"],
                    "frame_id": v["frames"][index - offset],
                    "total_frames": v["total_frames"],
                    "video_fps": v["video_fps"],
                    "duration": v.get("duration"),
                }
            offset += n
        raise IndexError("databasemap index out of range")

    def save_local(self, path: str = None):
        """保存 databasemap 到本地，新格式：单视频为对象，多视频为数组"""
        with self.acquire():
            dir_path = os.path.dirname(path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
            if len(self._videos) == 1:
                data = self._videos[0]
            else:
                data = self._videos
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def load_local(self, path: str = None) -> bool:
        """从本地加载 databasemap，支持新格式和旧格式（数组逐条记录）"""
        if not os.path.isfile(path):
            return False
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # 新格式：单视频对象 {"source_path", "total_frames", "video_fps", "duration", "frames": [...]}
        if isinstance(raw, dict) and "frames" in raw:
            videos = [self._ensure_duration(raw)]
        # 新格式：多视频数组 [{...}, {...}]
        elif isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict) and "frames" in raw[0]:
            videos = [self._ensure_duration(v) for v in raw]
        # 旧格式：逐条记录 [{"source_path", "frame_id", ...}, ...]
        elif isinstance(raw, list) and len(raw) > 0 and "frame_id" in raw[0]:
            videos = self._convert_legacy_to_videos(raw)
        else:
            return False

        with self.acquire():
            self._videos = videos
        return True

    def _ensure_duration(self, v: dict) -> dict:
        """确保视频对象包含 duration，若缺失则根据 total_frames/video_fps 计算"""
        if "duration" not in v or v["duration"] is None:
            tf = v.get("total_frames")
            fps = v.get("video_fps")
            v = dict(v)
            v["duration"] = (tf / fps if tf is not None and fps and fps > 0 else None)
        return v

    def _convert_legacy_to_videos(self, records: list) -> list:
        """将旧格式 [{"source_path", "frame_id", ...}, ...] 转为按视频聚合的新格式，保持插入顺序"""
        from collections import OrderedDict
        by_path = OrderedDict()
        for r in records:
            key = (r["source_path"], r["total_frames"], r["video_fps"])
            if key not in by_path:
                entry = {"source_path": r["source_path"], "total_frames": r["total_frames"],
                         "video_fps": r["video_fps"], "frames": []}
                # 旧格式可能带 duration
                if "duration" in r:
                    entry["duration"] = r["duration"]
                by_path[key] = entry
            by_path[key]["frames"].append(r["frame_id"])
        return [self._ensure_duration(v) for v in by_path.values()]

    def append(self, item: dict):
        """添加一条记录，自动聚合到对应视频的 frames 中"""
        with self.acquire():
            sp = item["source_path"]
            tf = item["total_frames"]
            fps = item["video_fps"]
            fid = item["frame_id"]
            duration = item.get("duration")
            if self._videos and self._videos[-1]["source_path"] == sp:
                self._videos[-1]["frames"].append(fid)
                if duration is not None and "duration" not in self._videos[-1]:
                    self._videos[-1]["duration"] = duration
            else:
                self._videos.append({
                    "source_path": sp,
                    "total_frames": tf,
                    "video_fps": fps,
                    "duration": duration,
                    "frames": [fid],
                })

    def get(self, index: int):
        """获取指定索引的记录"""
        with self.acquire():
            if 0 <= index < self._total_frames_count():
                return self._index_to_record(index)
            return None

    def clear(self):
        """清空 databasemap"""
        with self.acquire():
            self._videos.clear()

    def __len__(self):
        """总向量数"""
        with self.acquire():
            return self._total_frames_count()

    def __getitem__(self, index: int) -> dict:
        """按 FAISS 索引获取 {source_path, frame_id, total_frames, video_fps, duration}"""
        with self.acquire():
            return self._index_to_record(index)

    def __setitem__(self, index: int, item: dict):
        """不支持按索引修改，请通过 append 添加"""
        raise NotImplementedError("按视频聚合格式不支持按索引覆盖，请使用 append")

    def __delitem__(self, index: int):
        """不支持按索引删除"""
        raise NotImplementedError("按视频聚合格式不支持按索引删除")

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
        self.current_video_name = None
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
        # 清除已有 handler，避免 benchmark 多次 _init_components 时重复添加导致日志重复输出
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
            self.logger.info(f"本地文件 {self.faiss_file_path} 不存在，将根据*视频文件名*创建新的向量数据库")
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
            # 按当前视频名生成保存路径
            if self.current_video_name:
                base_dir = os.path.dirname(self.faiss_file_path)
                save_faiss_path = os.path.join(base_dir, f"{self.current_video_name}.faiss")
                save_map_path = os.path.join(base_dir, f"{self.current_video_name}.json")
            else:
                save_faiss_path = self.faiss_file_path
                save_map_path = self.databasemap_file_path

            # 保存faiss索引
            self.index.save_local(save_faiss_path)
            # 保存databasemap
            self.databasemap.save_local(save_map_path)

            self.logger.info(f"向量数据库已保存到 {save_faiss_path}，包含 {self.vector_count} 个向量")
            self.logger.info(f"databasemap已保存到 {save_map_path}，包含 {len(self.databasemap)} 条记录")
    
    def _add_vector(self, vector_data: FrameVectorData):
        """添加单个向量到数据库"""
        current_video_name = os.path.splitext(os.path.basename(vector_data.source_path))[0]
        if not self.current_video_name:
            self.current_video_name = current_video_name
        elif self.current_video_name != current_video_name:
            raise RuntimeError(
                f"当前处理视频为 {self.current_video_name}，收到来自 {current_video_name} 的帧；"
                "跨视频文件转换的逻辑尚未开发。"
            )

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
            "video_fps": vector_data.video_fps,
            "duration": vector_data.duration
        }
        self.databasemap.append(db_record)
        
        self.vector_count += 1
        
        self.logger.debug(f"向量添加成功, ID: {self.vector_count-1}, 总向量数: {self.vector_count}")
    
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
    
    def _read_frame_from_video(self, vector_ids: List[int]) -> Optional[List[np.ndarray]]:
        """根据 vector_ids 读取实际帧数据，并保存到 database/ 目录。
        按视频分组、复用 VideoCapture，避免重复打开同一视频。
        """
        from collections import defaultdict

        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database")
        os.makedirs(save_dir, exist_ok=True)

        # 按视频分组: source_path -> [(vector_id, frame_id), ...]
        by_video = defaultdict(list)
        for vector_id in vector_ids:
            db_record = self.databasemap[vector_id]
            by_video[db_record["source_path"]].append((vector_id, db_record["frame_id"]))

        # 预分配结果，按 vector_ids 顺序
        result = [None] * len(vector_ids)
        vid_to_idx = {vid: i for i, vid in enumerate(vector_ids)}

        for source_path, items in by_video.items():
            cap = cv2.VideoCapture(source_path)
            if not cap.isOpened():
                self.logger.error(f"无法打开视频文件: {source_path}")
                return None
            try:
                video_name = os.path.splitext(os.path.basename(source_path))[0]
                fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
                for vector_id, frame_id in items:
                    sec = round(frame_id / fps, 2)
                    self.logger.info(
                        f"读取帧: 视频={source_path}, 帧数={frame_id}, 时间={sec}秒"
                    )
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.error(f"无法读取视频帧 {frame_id}")
                        continue
                    result[vid_to_idx[vector_id]] = frame
                    # save_path = os.path.join(save_dir, f"{video_name}_{frame_id}.png")
                    # cv2.imwrite(save_path, frame)
                    # self.logger.debug(f"已保存帧: {save_path}")
            finally:
                cap.release()

        return [f for f in result if f is not None]
    
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
            result = MemoryResult(
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
        
        # 等待子进程结束（仅当当前进程是子进程的父进程时才可安全 join）
        if not hasattr(self, 'process'):
            return
        try:
            parent_pid = getattr(self.process, '_parent_pid', None)
            if parent_pid is None or parent_pid != os.getpid():
                return
            if self.process.is_alive():
                self.process.join(timeout=5)
        except (AssertionError, ValueError) as e:
            logging.getLogger(__name__).debug("停止子进程时跳过 join: %s", e)
        
    
    def set_frame_vector_queue(self, frame_vector_queue: mp.Queue):
        """设置帧向量队列"""
        self.frame_vector_queue = frame_vector_queue
    
    def set_query_vector_queue(self, query_vector_queue: mp.Queue):
        """设置查询向量队列"""
        self.query_vector_queue = query_vector_queue
    
    def get_query_result_queue(self):
        """获取查询结果队列"""
        return self.query_result_queue

    def init_sync(self):
        """同步初始化数据库（在主进程调用，供 benchmark 使用）"""
        self._set_logger()
        self._initialize_database()

    def add_vectors_batch(self, vector_data_list: Sequence[FrameVectorData]):
        """批量同步添加向量，供 benchmark 使用。需先调用 init_sync()。"""
        for vd in vector_data_list:
            self._add_vector(vd)

    def retrieve_sync(self, query_vector: np.ndarray, top_k: int = None) -> Tuple[List, List[float]]:
        """同步检索，供 benchmark 使用。返回 (帧列表, 分数列表)。"""
        k = top_k if top_k is not None else self.memory_topk
        return self._retrieve(query_vector, k)

    def save_database_sync(self):
        """同步保存数据库"""
        self._save_database()