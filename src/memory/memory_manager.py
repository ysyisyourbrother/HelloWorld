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
from typing import List, Optional, Tuple, Sequence, Callable, Dict
from contextlib import contextmanager
import multiprocessing as mp

# 本项目
from src.config import Config
from src.memory.frame_vectorizer import FrameVectorData, FrameVectorizer
from src.memory.query_vectorizer import QueryData, QueryVectorizer
from src.video_input.video_input import FrameData
from src.video_utils.about_frame import extract_save_frame_by_index

# 视频读取库
import cv2

@dataclass
class MemoryResult:
    """查询结果结构体, 包含查询ID、对话ID和匹配的向量ID列表"""
    metadata_list: List[dict]   # 匹配结果元数据列表（不含像素帧）
    timestamp: float          # 时间戳
    query_id: int             # 查询ID
    dialog_id: int            # 对话ID
    scores: List[float]       # 匹配分数列表
    trace_ts: Optional[Dict[str, float]] = None  # 链路时延埋点（秒）

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

class MemoryManagerBase:
    """记忆管理器"""
    def __init__(self, config: Config = None):
        """
        初始化MemoryManager模块
        负责维护Video向量数据库, 包括加载、更新、保存和查询
        
        Args:
            config (Config): 配置对象实例
        """
        if config is None:
            config = Config()
        self._config = config

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
        self.memory_topk = config.memory_topk  # topk参数
        self.memory_save_retrieved_frames = config.memory_save_retrieved_frames
        self.memory_save_injected_frames = config.memory_save_injected_frames
        self.memory_retrieve_save_dir = os.path.join("logs", "memory", "retrieve")
        self.memory_inject_save_dir = os.path.join("logs", "memory", "inject")
        
        # 队列相关（与 VideoInput.frame_queue 对接，由编排层注入）
        self.memory_mode = config.memory_mode
        self.frame_queue = None  # multiprocessing.Queue[FrameData]，原先进 FrameVectorizer

        self._frame_encoder: Optional[FrameVectorizer] = None
        self._query_encoder: Optional[QueryVectorizer] = None
        self._ready_event = threading.Event()

        self.running = False
        self.current_video_name = None
        self.vector_count = 0

        # 检索钩子：每次 retrieve 时调用，传入 (query_vector, all_scores)
        # all_scores: List[float]，长度为 vector_count，all_scores[i] 为向量 i 与 query 的距离
        self._retrieve_hooks: List[Callable[[np.ndarray, List[float]], None]] = []

    def register_retrieve_hook(self, fn: Callable[[np.ndarray, List[float]], None]):
        """注册检索钩子，在每次 retrieve 时调用。fn(query_vector, all_scores)"""
        self._retrieve_hooks.append(fn)

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
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False

        # 启动时清空 inject 目录旧图片；retrieve 目录仍在每次检索前清理
        os.makedirs(self.memory_inject_save_dir, exist_ok=True)
        for name in os.listdir(self.memory_inject_save_dir):
            lower_name = name.lower()
            if not lower_name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            file_path = os.path.join(self.memory_inject_save_dir, name)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    self.logger.warning(f"删除旧注入帧失败: {file_path}, err={e}")
    
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
        new_vector_id = self.vector_count - 1
        if self.memory_save_injected_frames:
            self._save_injected_frame(vector_data, new_vector_id)
        trace_ts = dict(vector_data.trace_ts or {})
        trace_ts["memory_inject_added_at"] = time.time()
        if "frame_vectorizer_encoded_at" in trace_ts:
            self.logger.info(
                f"[Latency][Inject] frame_encode->faiss_add frame_id={vector_data.frame_id} "
                f"{(trace_ts['memory_inject_added_at'] - trace_ts['frame_vectorizer_encoded_at']) * 1000:.2f} ms"
            )
        
        self.logger.debug(f"向量添加成功, ID: {new_vector_id}, 总向量数: {self.vector_count}")
    
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

    def _query_faiss_all_scores(self, query_vector: np.ndarray) -> List[float]:
        """查询所有向量与 query 的距离，返回长度为 vector_count 的列表，all_scores[i] 为向量 i 的距离"""
        if self.index is None or self.vector_count == 0:
            return []
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        k = self.vector_count
        distances, indices = self.index.search(query_vector, k)
        # 构建 all_scores[i] = 向量 i 与 query 的距离
        all_scores = [0.0] * self.vector_count
        for idx, dist in zip(indices[0].tolist(), distances[0].tolist()):
            if 0 <= idx < self.vector_count:
                all_scores[idx] = float(dist)
        return all_scores
    
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

        return result

    def _save_injected_frame(self, vector_data: FrameVectorData, vector_id: int):
        """将本次入库对应的帧保存到 logs/memory/inject。"""
        os.makedirs(self.memory_inject_save_dir, exist_ok=True)
        try:
            source_path = vector_data.source_path
            frame_id = vector_data.frame_id
            video_fps = vector_data.video_fps or 1.0
            video_name = os.path.splitext(os.path.basename(source_path or "unknown"))[0]
            second = float(frame_id) / float(video_fps) if frame_id is not None else 0.0

            filename = (
                f"fid{int(frame_id)}"
                f"_sec{second:.2f}"
                f"_{video_name}"
                f"_vid{vector_id:06d}.jpg"
            )
            save_path = os.path.join(self.memory_inject_save_dir, filename)
            ok = extract_save_frame_by_index(
                video_path=str(source_path),
                output_path=save_path,
                frame_index=int(frame_id),
                backend="cv2",
            )
            if not ok:
                self.logger.warning(
                    f"保存注入帧失败（工具函数返回False）: {source_path}, frame_id={frame_id}"
                )
        except Exception as e:
            self.logger.warning(f"保存注入帧失败，vector_id={vector_id}, err={e}")

    def _save_retrieved_frames(self, vector_ids: List[int], scores: List[float]):
        """按检索结果保存帧图片到 logs/memory 目录。"""
        if not vector_ids:
            return
        os.makedirs(self.memory_retrieve_save_dir, exist_ok=True)
        for name in os.listdir(self.memory_retrieve_save_dir):
            lower_name = name.lower()
            if not lower_name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            file_path = os.path.join(self.memory_retrieve_save_dir, name)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    self.logger.warning(f"删除旧检索帧失败: {file_path}, err={e}")

        for rank, (vector_id, score) in enumerate(zip(vector_ids, scores), start=1):
            try:
                rec = self.databasemap[vector_id]
                source_path = rec.get("source_path")
                frame_id = rec.get("frame_id")
                video_fps = rec.get("video_fps") or 1.0
                video_name = os.path.splitext(os.path.basename(source_path or "unknown"))[0]
                second = float(frame_id) / float(video_fps) if frame_id is not None else 0.0

                filename = (
                    f"rank{rank:02d}"
                    f"_score{float(score):.6f}"
                    f"_sec{second:.2f}"
                    f"_{video_name}"
                    f"_fid{int(frame_id)}.jpg"
                )
                save_path = os.path.join(self.memory_retrieve_save_dir, filename)
                ok = extract_save_frame_by_index(
                    video_path=str(source_path),
                    output_path=save_path,
                    frame_index=int(frame_id),
                    backend="cv2",
                )
                if not ok:
                    self.logger.warning(
                        f"保存检索帧失败（工具函数返回False）: {source_path}, frame_id={frame_id}"
                    )
            except Exception as e:
                self.logger.warning(f"保存检索帧失败，vector_id={vector_id}, err={e}")

    def _retrieve(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> Tuple[List[float], List[dict]]:
        """根据查询向量检索匹配的帧数据

        Returns:
            tuple: (分数列表, 元数据列表)，元数据每项为 {frame_id, video_fps, source_path}
        """
        # 若有注册的钩子，计算全量相似度并调用
        if self._retrieve_hooks:
            all_scores = self._query_faiss_all_scores(query_vector)
            for fn in self._retrieve_hooks:
                try:
                    fn(query_vector.copy(), all_scores)
                except Exception as e:
                    self.logger.warning(f"检索钩子执行异常: {e}")

        vector_ids, scores = self._query_faiss(query_vector, top_k)
        self.logger.info(f"查询完成，返回 {len(vector_ids)} 个结果")
        if self.memory_save_retrieved_frames:
            self._save_retrieved_frames(vector_ids, scores)

        # 构建元数据（frame_id, video_fps）
        metadata_list = []
        for vid in vector_ids:
            rec = self.databasemap[vid]
            metadata_list.append({
                "frame_id": rec["frame_id"],
                "video_fps": rec.get("video_fps") or 1.0,
                "source_path": rec.get("source_path"),
            })

        # 边端实时优先：查询阶段只返回元数据，不读取像素帧
        self.logger.debug(f"查询阶段返回元数据 {len(metadata_list)} 条（不含像素帧）")
        return scores, metadata_list
    
    def init_sync(self):
        """同步初始化数据库（在主进程调用，供 benchmark 使用）"""
        self._set_logger()
        self._initialize_database()

    def add_vectors_batch(self, vector_data_list: Sequence[FrameVectorData]):
        """批量同步添加向量，供 benchmark 使用。需先调用 init_sync()。"""
        for vd in vector_data_list:
            self._add_vector(vd)

    def retrieve_sync(
        self, query_vector: np.ndarray, top_k: int = None
    ) -> Tuple[List[float], List[dict]]:
        """同步检索，供 benchmark 使用。返回 (分数列表, 元数据列表)。"""
        k = top_k if top_k is not None else self.memory_topk
        return self._retrieve(query_vector, k)

    def query_text_sync(
        self,
        query_text: str,
        query_id: int,
        dialog_id: int,
        timestamp: float,
        trace_ts: Optional[Dict[str, float]] = None,
    ) -> MemoryResult:
        """
        同步查询：文本编码 + 检索，供 APIServerE 直接调用（无 query 队列往返）。
        """
        if not self._ready_event.is_set():
            # query_with_memory 等场景未启动后台线程时，按需同步初始化数据库
            if self.index is None or self.databasemap is None:
                self.init_sync()
            self._ready_event.set()

        if self._query_encoder is None:
            self._query_encoder = QueryVectorizer(self._config)
            self._query_encoder._set_logger()
            self._query_encoder._initialize_vectorizer()

        qd = QueryData(
            query=query_text,
            query_id=query_id,
            dialog_id=dialog_id,
            timestamp=timestamp,
            trace_ts=dict(trace_ts or {}),
        )
        qvd = self._query_encoder.encode_query_data(qd)
        merged_trace = dict(qd.trace_ts or {})
        merged_trace.update(qvd.trace_ts or {})
        merged_trace["memory_query_dequeue_at"] = time.time()
        retrieve_start = time.time()
        scores, metadata_list = self._retrieve(qvd.vector, self.memory_topk)
        merged_trace["memory_query_retrieved_at"] = time.time()
        self.logger.info(
            f"[Latency][Query] memory_retrieve query_id={qvd.query_id} "
            f"{(merged_trace['memory_query_retrieved_at'] - retrieve_start) * 1000:.2f} ms"
        )
        return MemoryResult(
            metadata_list=metadata_list,
            timestamp=qvd.timestamp,
            query_id=qvd.query_id,
            dialog_id=qvd.dialog_id,
            scores=scores,
            trace_ts=merged_trace,
        )

    def save_database_sync(self):
        """同步保存数据库"""
        self._save_database()


class MemoryManagerOnline(MemoryManagerBase):
    """在线记忆管理：在 Base 同步能力上扩展线程和队列流水线。"""

    def _thread_frame_vectors(self):
        """处理帧向量的线程（从 frame_queue 取 FrameData，经 FrameVectorizer 编码后入库）"""
        self.logger.info(f"帧向量处理线程启动, 线程名: {threading.current_thread().name}")

        save_interval = 30

        while self.running_event.is_set():
            try:
                frame_data: FrameData = self.frame_queue.get(timeout=save_interval)
                vector_data = self._frame_encoder.encode_frame_from_stream(frame_data)
                if vector_data is not None:
                    self._add_vector(vector_data)
            except queue.Empty:
                self.logger.debug("帧输入队列超时，保存数据库")
                self._save_database()
            except Exception as e:
                self.logger.error(f"处理帧向量时出错: {e}")

    def _process_main(self):
        """在线主循环：按 memory_mode 启动对应线程/编码器。"""
        self._set_logger()
        self.logger.info(f"MemoryManager启动, 进程ID: {os.getpid()}")

        self._initialize_database()
        memory_mode = self.memory_mode
        self.logger.info(f"MemoryManager模式: {memory_mode}")

        if memory_mode in ("only_inject", "both"):
            self._frame_encoder = FrameVectorizer(self._config)
            self._frame_encoder._set_logger()
            self._frame_encoder._initialize_vectorizer()
            if self.frame_queue is None:
                raise RuntimeError("inject 模式需要设置 frame_queue（通常为 VideoInput.frame_queue）")
        if memory_mode in ("only_query", "both"):
            self._query_encoder = QueryVectorizer(self._config)
            self._query_encoder._set_logger()
            self._query_encoder._initialize_vectorizer()

        self._ready_event.set()
        threads = []

        if memory_mode in ["only_inject", "both"]:
            frame_thread = threading.Thread(target=self._thread_frame_vectors, daemon=True)
            frame_thread.name = "FrameVectorThread"
            threads.append(frame_thread)
            self.logger.info("帧向量处理线程已创建")

        if len(threads) == 0 and memory_mode == "only_query":
            self.logger.info("only_query 模式：使用同步 query_text_sync，不启动后台线程")
        elif len(threads) == 0:
            self.logger.error("没有启动任何线程，请检查memory_mode配置")
            return

        for thread in threads:
            thread.start()
            self.logger.info(f"线程 {thread.name} 已启动")

        while self.running_event.is_set():
            time.sleep(1.0)

    def start(self):
        """启动MemoryManager（同进程后台线程）"""
        self.running_event = threading.Event()
        self.running_event.set()
        self._ready_event.clear()
        self.worker_thread = threading.Thread(
            target=self._process_main,
            daemon=True,
            name="MemoryManager-Main",
        )
        self.worker_thread.start()

    def start_single_thread(self):
        """启动单线程运行MemoryManager"""
        self.running_event = threading.Event()
        self.running_event.set()
        self._process_main()

    def stop(self):
        """停止MemoryManager"""
        if hasattr(self, "running_event"):
            self.running_event.clear()

        if self.index is not None:
            self._save_database()

        if hasattr(self, "worker_thread") and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def set_frame_queue(self, frame_queue: mp.Queue):
        """设置帧输入队列（VideoInput 产出 FrameData，原 FrameVectorizer 消费端）"""
        self.frame_queue = frame_queue
