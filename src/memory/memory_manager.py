import threading
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import time
import queue
import glob
import os
import shutil
import faiss
from dataclasses import dataclass
from typing import List, Optional, Tuple, Sequence, Callable, Dict, Any
import multiprocessing as mp

# 本项目
from src.config import Config
from src.memory.frame_vectorizer import FrameVectorData, FrameVectorizer
from src.memory.index.faiss import ThreadSafeFaiss
from src.memory.index.map import ThreadSafeMap
from src.memory.query_vectorizer import QueryData, QueryVectorizer
from src.video_input.video_input import FrameData
from src.video_utils.about_frame import extract_save_frame_by_index, extract_frame_by_index

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
    # 与 metadata_list 等长；BGR uint8 ndarray，供边端跳过按路径再解码（MemoryManagerOnlineV3）
    retrieval_frames: Optional[List[Any]] = None
    # retrieve_item_type=clip 时：GOP mp4 路径列表与输出目录（见 logs/memory/retrieve/clips/）
    retrieve_clip_paths: Optional[List[Dict[str, Any]]] = None
    retrieve_clip_dir: Optional[str] = None

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

        self.memory_retrieve_item_type = config.memory_retrieve_item_type   
       
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
        # clip 导出：按 dialog_id 分目录，新对话时删除上一对话子目录
        self._last_retrieve_clip_dialog_id = None  # type: Optional[int]
        self._retrieve_clip_lock = threading.RLock()

    def register_retrieve_hook(self, fn: Callable[[np.ndarray, List[float]], None]):
        """注册检索钩子，在每次 retrieve 时调用。fn(query_vector, all_scores)"""
        self._retrieve_hooks.append(fn)

    def register_i_frames(self, source_path, i_frame_indices):
        """
        编排层在开始处理某个本地视频 source_path 时调用，将 I 帧索引写入 databasemap
        （与 SymVideoInputByGOP.i_frame_indices 或 ffprobe 扫描结果一致）。

        须保证与后续入库向量的 source_path 字符串一致（含绝对/相对路径）。
        """
        if self.databasemap is None:
            log = getattr(self, "logger", None)
            if log:
                log.warning("register_i_frames: databasemap 未初始化，请先 init_sync 或 start")
            return
        self.databasemap.register_i_frames(source_path, i_frame_indices)

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
            self.logger.debug(f"从本地文件 {self.faiss_file_path} 加载向量数据库")
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
            self.logger.debug(f"本地文件 {self.faiss_file_path} 不存在，将根据*视频文件名*创建新的向量数据库")
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

            out_dir = os.path.dirname(os.path.abspath(save_faiss_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            map_dir = os.path.dirname(os.path.abspath(save_map_path))
            if map_dir and map_dir != out_dir:
                os.makedirs(map_dir, exist_ok=True)

            # 保存faiss索引
            self.index.save_local(save_faiss_path)
            # 保存databasemap
            self.databasemap.save_local(save_map_path)

            self.logger.debug(f"向量数据库已保存到 {save_faiss_path}，包含 {self.vector_count} 个向量")
            self.logger.debug(f"数据库索引已保存到 {save_map_path}，包含 {len(self.databasemap)} 条记录")
    
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

    def _prepare_retrieve_clip_output_dir(self, dialog_id: int) -> str:
        """
        在 logs/memory/retrieve/clips/dialog_{dialog_id}/ 下创建空目录并写入 GOP mp4。
        若 dialog_id 与上次不同，则删除上一对话对应子目录；同一对话内每次检索会先清空本子目录再写入。
        """
        base = os.path.abspath(self.memory_retrieve_save_dir)
        os.makedirs(base, exist_ok=True)
        clips_root = os.path.join(base, "clips")
        d = int(dialog_id)
        with self._retrieve_clip_lock:
            last = self._last_retrieve_clip_dialog_id
            if last is not None and int(last) != d:
                prev = os.path.join(clips_root, "dialog_%d" % int(last))
                if os.path.isdir(prev):
                    shutil.rmtree(prev, ignore_errors=True)
            self._last_retrieve_clip_dialog_id = d
            out = os.path.join(clips_root, "dialog_%d" % d)
            if os.path.isdir(out):
                shutil.rmtree(out, ignore_errors=True)
            os.makedirs(out, exist_ok=True)
        return out

    def _export_retrieve_clips_if_needed(
        self, metadata_list: List[dict], dialog_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        clip 模式下将 topk 元数据导出为 GOP mp4；返回 {"paths": [...], "dir": str} 或 None。
        """
        if self.memory_retrieve_item_type != "clip" or not metadata_list:
            return None
        from src.video_utils import ffmpeg_utils

        out_dir = self._prepare_retrieve_clip_output_dir(dialog_id)
        try:
            pairs = ffmpeg_utils.export_unique_gop_mp4s_from_memory_records(
                metadata_list, output_dir=out_dir, prefix="gop"
            )
            paths = [
                {"gop_start": int(rng[0]), "gop_end": int(rng[1]), "path": p}
                for rng, p in pairs
            ]
            return {"paths": paths, "dir": out_dir}
        except Exception as e:
            log = getattr(self, "logger", None)
            if log:
                log.warning("检索 clip 导出失败: %s", e)
            return {"paths": [], "dir": out_dir, "error": str(e)}

    def _retrieve(
        self, query_vector: np.ndarray, top_k: int = 5, dialog_id: int = 0
    ) -> Tuple[List[float], List[dict], Optional[Dict[str, Any]]]:
        """根据查询向量检索匹配的帧数据

        Returns:
            tuple: (分数列表, 元数据列表, clip_info)。
            clip_info 为 None（非 clip）或 {"paths": [...], "dir": str}；paths 每项含 gop_start、gop_end、path。
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

        # 构建元数据（frame_id, video_fps, total_frames, i_frames 供 GOP 导出等）
        metadata_list = []
        for vid in vector_ids:
            rec = self.databasemap[vid]
            meta = {
                "frame_id": rec["frame_id"],
                "video_fps": rec.get("video_fps") or 1.0,
                "source_path": rec.get("source_path"),
                "total_frames": rec.get("total_frames"),
                "i_frames": rec.get("i_frames") or [],
            }
            metadata_list.append(meta)

        clip_info = self._export_retrieve_clips_if_needed(metadata_list, dialog_id)

        # 边端实时优先：查询阶段只返回元数据，不读取像素帧
        self.logger.debug(f"查询阶段返回元数据 {len(metadata_list)} 条（不含像素帧）")
        return scores, metadata_list, clip_info
    
    def init_sync(self):
        """同步初始化数据库（在主进程调用，供 benchmark 使用）"""
        self._set_logger()
        self._initialize_database()

    def add_vectors_batch(self, vector_data_list: Sequence[FrameVectorData]):
        """批量同步添加向量，供 benchmark 使用。需先调用 init_sync()。"""
        for vd in vector_data_list:
            self._add_vector(vd)

    def retrieve_sync(
        self, query_vector: np.ndarray, top_k: int = None, dialog_id: int = 0
    ) -> Tuple[List[float], List[dict], Optional[Dict[str, Any]]]:
        """同步检索，供 benchmark 使用。

        Returns:
            (分数列表, 元数据列表, clip_info)；clip_info 见 ``_retrieve``。
        """
        k = top_k if top_k is not None else self.memory_topk
        return self._retrieve(query_vector, k, dialog_id=dialog_id)

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
        scores, metadata_list, clip_info = self._retrieve(
            qvd.vector, self.memory_topk, dialog_id=qvd.dialog_id
        )
        merged_trace["memory_query_retrieved_at"] = time.time()
        self.logger.info(
            f"[Latency][Query] memory_retrieve query_id={qvd.query_id} "
            f"{(merged_trace['memory_query_retrieved_at'] - retrieve_start) * 1000:.2f} ms"
        )
        clip_paths = None
        clip_dir = None
        if clip_info:
            clip_paths = clip_info.get("paths")
            clip_dir = clip_info.get("dir")
        return MemoryResult(
            metadata_list=metadata_list,
            timestamp=qvd.timestamp,
            query_id=qvd.query_id,
            dialog_id=qvd.dialog_id,
            scores=scores,
            trace_ts=merged_trace,
            retrieve_clip_paths=clip_paths,
            retrieve_clip_dir=clip_dir,
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
                try:
                    self._save_database()
                except Exception:
                    self.logger.exception("定时保存向量库失败")
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
            try:
                self._save_database()
            except Exception:
                self.logger.exception("停止时保存向量库失败")

        if hasattr(self, "worker_thread") and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def set_frame_queue(self, frame_queue: mp.Queue):
        """设置帧输入队列（VideoInput 产出 FrameData，原 FrameVectorizer 消费端）"""
        self.frame_queue = frame_queue


class MemoryManagerOnlineV2(MemoryManagerOnline):
    """
    网络流 / StreamVideoInput 专用在线记忆：
    - stop() 时先停止 StreamVideoInput（若已绑定），再保存 faiss 与 databasemap；
    - 可选 memory_stream_max_seconds：后台计时，到点自动 stop()（保存向量库并结束会话）；
    - 可与 StreamVideoInput.set_on_session_end 配合（流因 max 时长结束时由回调触发 stop）。
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._stream_input = None
        self._session_timer_thread = None

    def set_stream_input(self, stream_input):
        """绑定 StreamVideoInput，便于 stop 顺序与计时。"""
        self._stream_input = stream_input

    def start(self):
        if self._stream_input is not None and hasattr(
            self._stream_input, "set_on_session_end"
        ):
            self._stream_input.set_on_session_end(self._on_stream_session_end)
        super().start()
        max_sec = float(getattr(self._config, "memory_stream_max_seconds", 0.0) or 0.0)
        if max_sec > 0:
            self._session_timer_thread = threading.Thread(
                target=self._run_memory_session_timer,
                name="MemoryStreamMaxSeconds",
                daemon=True,
            )
            self._session_timer_thread.start()

    def _on_stream_session_end(self):
        """由 Gst 线程在达到 stream_record_max_seconds 时触发，勿阻塞。"""

        def _run():
            self.logger.info("流输入会话结束，保存向量库并停止 MemoryManagerOnlineV2")
            try:
                self.stop()
            except Exception:
                self.logger.exception("MemoryManagerOnlineV2.stop 异常")

        threading.Thread(target=_run, daemon=True).start()

    def _run_memory_session_timer(self):
        max_sec = float(getattr(self._config, "memory_stream_max_seconds", 0.0) or 0.0)
        if max_sec <= 0:
            return
        t0 = time.time()
        while time.time() - t0 < max_sec:
            if not self.running_event.is_set():
                return
            time.sleep(0.25)
        self.logger.info(
            "memory_stream_max_seconds=%.0f 已到，停止流并保存向量库", max_sec
        )
        try:
            self.stop()
        except Exception:
            self.logger.exception("MemoryManagerOnlineV2 计时停止异常")

    def stop(self):
        if self._stream_input is not None:
            try:
                self._stream_input.stop()
            except Exception as e:
                self.logger.warning("停止 StreamVideoInput 时: %s", e)
            self._stream_input = None
        super().stop()


class ShortMemoryStore(object):
    """
    按向量 id 缓存最近注入帧的 BGR 像素；TTL 略大于分段时长，供检索短路。
    """

    def __init__(self, ttl_sec, logger):
        self.ttl_sec = max(1.0, float(ttl_sec))
        self.logger = logger
        self._lock = threading.RLock()
        self._by_id = {}  # type: Dict[int, Tuple[float, Any]]

    def put(self, vector_id, bgr, mono_ts):
        if bgr is None:
            return
        try:
            payload = np.ascontiguousarray(bgr)
        except Exception:
            return
        with self._lock:
            self._purge_locked(mono_ts)
            self._by_id[int(vector_id)] = (float(mono_ts), payload)

    def get(self, vector_id, mono_ts):
        with self._lock:
            self._purge_locked(mono_ts)
            ent = self._by_id.get(int(vector_id))
            if ent is None:
                return None
            return ent[1]

    def clear(self):
        with self._lock:
            self._by_id.clear()

    def _purge_locked(self, mono_ts):
        cutoff = float(mono_ts) - self.ttl_sec
        dead = [vid for vid, (t0, _) in self._by_id.items() if t0 < cutoff]
        for vid in dead:
            del self._by_id[vid]


class StreamSegmentFileMap(object):
    """
    线程安全：已知 segment MP4 路径 -> {fps, frame_count}，按路径分组取帧，支持跨文件。
    """

    def __init__(self, logger):
        self.logger = logger
        self._lock = threading.RLock()
        self._meta = {}  # type: Dict[str, Dict[str, Any]]

    def clear(self):
        with self._lock:
            self._meta.clear()

    def refresh_session_dir(self, session_dir):
        # type: (Optional[str]) -> List[str]
        if not session_dir or not os.path.isdir(session_dir):
            return []
        pattern = os.path.join(os.path.abspath(session_dir), "segment_*.mp4")
        paths = sorted(glob.glob(pattern))
        with self._lock:
            for p in paths:
                if p not in self._meta:
                    self._meta[p] = self._probe(p)
        return paths

    def _probe(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return {"fps": 25.0, "frame_count": 0}
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            return {"fps": fps, "frame_count": max(0, n)}
        finally:
            cap.release()

    def get_meta(self, path):
        # type: (str) -> Dict[str, Any]
        with self._lock:
            return dict(self._meta.get(path, {"fps": 25.0, "frame_count": 0}))

    def read_frame_bgr(self, path, frame_index):
        # type: (str, int) -> Optional[np.ndarray]
        if not path or not os.path.isfile(path):
            return None
        try:
            return extract_frame_by_index(
                path, int(frame_index), backend="cv2"
            )
        except Exception as e:
            self.logger.debug("segment 取帧失败 path=%s idx=%s err=%s", path, frame_index, e)
            return None


class MemoryManagerOnlineV3(MemoryManagerOnlineV2):
    """
    在 V2 基础上：
    - 短期记忆：按 stream 分段时长 × 系数保留最近向量的 BGR 帧，检索时优先命中；
    - 长期：本地文件路径仍按 frame_id 解码；RTSP 等非常规文件路径时，可选按录制目录
      segment_*.mp4 + 墙钟时间启发式取帧（依赖 stream_record_fps / 分段配置）。
    - query_text_sync 填充 MemoryResult.retrieval_frames，与 metadata_list 等长（BGR）。
    """

    def __init__(self, config=None):
        super().__init__(config)
        seg_min = float(getattr(self._config, "stream_record_segment_minutes", 1.0) or 1.0)
        ratio = float(getattr(self._config, "memory_short_memory_ttl_ratio", 1.1) or 1.1)
        ttl = max(60.0, seg_min * 60.0 * ratio)
        self._short_memory = ShortMemoryStore(ttl, logging.getLogger("ShortMemoryStore"))
        self._segment_map = StreamSegmentFileMap(
            logging.getLogger("StreamSegmentFileMap")
        )
        self._v3_extra_lock = threading.RLock()
        self._vector_wall_ts = {}  # type: Dict[int, float]
        self._session_first_wall_ts = None  # type: Optional[float]
        self._segment_pixel_heuristic = bool(
            getattr(self._config, "memory_segment_pixel_heuristic", True)
        )

    def _set_logger(self):
        super()._set_logger()
        self._short_memory.logger = self.logger
        self._segment_map.logger = self.logger

    def _add_vector(self, vector_data, pixel_bgr=None):
        super()._add_vector(vector_data)
        vid = self.vector_count - 1
        wall = float(vector_data.timestamp)
        mono = time.time()
        with self._v3_extra_lock:
            self._vector_wall_ts[vid] = wall
            if self._session_first_wall_ts is None:
                self._session_first_wall_ts = wall
        self._short_memory.put(vid, pixel_bgr, mono)

    def _thread_frame_vectors(self):
        self.logger.info(
            "帧向量处理线程启动(V3), 线程名: %s", threading.current_thread().name
        )
        save_interval = 30
        while self.running_event.is_set():
            try:
                frame_data = self.frame_queue.get(timeout=save_interval)
                vector_data = self._frame_encoder.encode_frame_from_stream(frame_data)
                if vector_data is not None:
                    bgr = None
                    fd = getattr(frame_data, "frame", None)
                    if fd is not None:
                        try:
                            bgr = cv2.cvtColor(fd, cv2.COLOR_RGB2BGR)
                        except Exception as e:
                            self.logger.warning("RGB->BGR 失败 frame_id=%s: %s", frame_data.frame_id, e)
                    self._add_vector(vector_data, pixel_bgr=bgr)
            except queue.Empty:
                self.logger.debug("帧输入队列超时，保存数据库")
                try:
                    self._save_database()
                except Exception:
                    self.logger.exception("定时保存向量库失败")
            except Exception as e:
                self.logger.error("处理帧向量时出错: %s", e)

    def _long_term_frame_bgr(self, vector_id, rec):
        # type: (int, dict) -> Optional[np.ndarray]
        sp = rec.get("source_path") or ""
        fid = rec.get("frame_id")
        if sp and os.path.isfile(sp):
            try:
                return extract_frame_by_index(sp, int(fid), backend="cv2")
            except Exception as e:
                self.logger.debug("按 source_path 取帧失败: %s", e)

        if not self._segment_pixel_heuristic:
            return None
        if not sp.startswith("rtsp://"):
            return None

        session_dir = None
        if self._stream_input is not None and hasattr(
            self._stream_input, "get_recording_dir"
        ):
            session_dir = self._stream_input.get_recording_dir()
        if not session_dir:
            return None

        with self._v3_extra_lock:
            wall_ts = self._vector_wall_ts.get(vector_id)
            anchor = self._session_first_wall_ts
        if wall_ts is None:
            return None
        if anchor is None:
            anchor = wall_ts

        paths = self._segment_map.refresh_session_dir(session_dir)
        if not paths:
            return None

        segment_sec = max(
            1.0, float(getattr(self._config, "stream_record_segment_minutes", 1.0) or 1.0) * 60.0
        )
        fps = float(getattr(self._config, "stream_record_fps", 25.0) or 25.0) or 25.0

        dt = max(0.0, float(wall_ts) - float(anchor))
        seg_i = int(dt // segment_sec)
        if seg_i < 0:
            seg_i = 0
        if seg_i >= len(paths):
            seg_i = len(paths) - 1
        offset = dt - float(seg_i) * segment_sec
        local_fid = int(offset * fps)
        path = paths[seg_i]
        meta = self._segment_map.get_meta(path)
        nfm = int(meta.get("frame_count") or 0)
        if nfm > 0 and local_fid >= nfm:
            local_fid = max(0, nfm - 1)
        return self._segment_map.read_frame_bgr(path, local_fid)

    def _build_retrieval_frames(self, vector_ids, mono_now):
        # type: (List[int], float) -> List[Any]
        out = []
        for vid in vector_ids:
            if vid < 0 or vid >= self.vector_count:
                out.append(None)
                continue
            bgr = self._short_memory.get(vid, mono_now)
            if bgr is not None:
                out.append(bgr)
                continue
            rec = self.databasemap[vid]
            out.append(self._long_term_frame_bgr(vid, rec))
        return out

    def _save_retrieved_frames(self, vector_ids, scores):
        """
        与 _build_retrieval_frames 一致：短期记忆中有 BGR 则直接 imwrite；
        否则长期路径（本地视频按 frame_id，RTSP 则分段 MP4 启发式）取帧后再存。
        不再对 source_path 调用 extract_save_frame_by_index（避免 pathlib 破坏 rtsp://）。
        """
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

        mono = time.time()
        bgr_list = self._build_retrieval_frames(vector_ids, mono)
        for rank, (vector_id, score, bgr) in enumerate(
            zip(vector_ids, scores, bgr_list), start=1
        ):
            try:
                if vector_id < 0 or vector_id >= self.vector_count:
                    self.logger.warning(
                        "保存检索帧跳过: 无效 vector_id=%s", vector_id
                    )
                    continue
                rec = self.databasemap[vector_id]
                source_path = rec.get("source_path")
                frame_id = rec.get("frame_id")
                video_fps = rec.get("video_fps") or 1.0
                video_name = os.path.splitext(
                    os.path.basename(source_path or "unknown")
                )[0]
                second = (
                    float(frame_id) / float(video_fps)
                    if frame_id is not None
                    else 0.0
                )
                filename = (
                    f"rank{rank:02d}"
                    f"_score{float(score):.6f}"
                    f"_sec{second:.2f}"
                    f"_{video_name}"
                    f"_fid{int(frame_id)}.jpg"
                )
                save_path = os.path.join(self.memory_retrieve_save_dir, filename)
                if bgr is None:
                    self.logger.warning(
                        "保存检索帧跳过: 无像素（短期已过期且长期取帧失败）"
                        " vector_id=%s path=%s frame_id=%s",
                        vector_id,
                        source_path,
                        frame_id,
                    )
                    continue
                if not cv2.imwrite(save_path, bgr):
                    self.logger.warning("cv2.imwrite 失败: %s", save_path)
            except Exception as e:
                self.logger.warning(
                    "保存检索帧失败，vector_id=%s, err=%s", vector_id, e
                )

    def query_text_sync(
        self,
        query_text,
        query_id,
        dialog_id,
        timestamp,
        trace_ts=None,
    ):
        if not self._ready_event.is_set():
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

        qv = qvd.vector
        top_k = self.memory_topk
        if self._retrieve_hooks:
            all_scores = self._query_faiss_all_scores(qv)
            for fn in self._retrieve_hooks:
                try:
                    fn(qv.copy(), all_scores)
                except Exception as e:
                    self.logger.warning("检索钩子执行异常: %s", e)

        vector_ids, scores = self._query_faiss(qv, top_k)
        self.logger.info("查询完成，返回 %d 个结果", len(vector_ids))
        if self.memory_save_retrieved_frames:
            self._save_retrieved_frames(vector_ids, scores)

        metadata_list = []
        for vid in vector_ids:
            if vid < 0 or vid >= self.vector_count:
                metadata_list.append(
                    {
                        "frame_id": -1,
                        "video_fps": 1.0,
                        "source_path": None,
                        "total_frames": None,
                        "i_frames": [],
                    }
                )
                continue
            rec = self.databasemap[vid]
            metadata_list.append(
                {
                    "frame_id": rec["frame_id"],
                    "video_fps": rec.get("video_fps") or 1.0,
                    "source_path": rec.get("source_path"),
                    "total_frames": rec.get("total_frames"),
                    "i_frames": rec.get("i_frames") or [],
                }
            )

        clip_info = self._export_retrieve_clips_if_needed(metadata_list, dialog_id)
        clip_paths = None
        clip_dir = None
        if clip_info:
            clip_paths = clip_info.get("paths")
            clip_dir = clip_info.get("dir")

        mono = time.time()
        retrieval_frames = self._build_retrieval_frames(vector_ids, mono)
        merged_trace["memory_query_retrieved_at"] = time.time()
        self.logger.info(
            "[Latency][Query] memory_retrieve query_id=%s %.2f ms",
            qvd.query_id,
            (merged_trace["memory_query_retrieved_at"] - retrieve_start) * 1000.0,
        )
        return MemoryResult(
            metadata_list=metadata_list,
            timestamp=qvd.timestamp,
            query_id=qvd.query_id,
            dialog_id=qvd.dialog_id,
            scores=scores,
            trace_ts=merged_trace,
            retrieval_frames=retrieval_frames,
            retrieve_clip_paths=clip_paths,
            retrieve_clip_dir=clip_dir,
        )

    def stop(self):
        self._short_memory.clear()
        self._segment_map.clear()
        with self._v3_extra_lock:
            self._vector_wall_ts.clear()
            self._session_first_wall_ts = None
        super().stop()
