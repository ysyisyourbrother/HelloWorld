#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VenusSystemMoti - Motivation 试验系统

支持：
- 指定 Video-MME 的某一个视频执行流程（inject + query）
- 指定某一个特定的图片进行推理
- 使用 config_moti.json 初始化
"""

import os
import sys
import json
import time
import logging
import faiss
import cv2
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.stream_input import StreamInput
from src.benchmark.utils import build_rag_prompt
from src.frame_vectorizer import FrameVectorizer
from src.memory_manager import MemoryManager
from src.query_vectorizer import QueryVectorizer
from src.reasoner import Reasoner, QueryRequest


class VenusSystemMoti:
    """Motivation 试验系统 - 指定视频或单图推理"""

    def __init__(self, config: Config = None):
        if config is None:
            config = Config(config_path="configs/config_moti.json")
        self.config = config
        self._setup_logger()

        # 组件（同步模式）
        self.stream_input: Optional[StreamInput] = None
        self.frame_vectorizer: Optional[FrameVectorizer] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.query_vectorizer: Optional[QueryVectorizer] = None
        self.reasoner: Optional[Reasoner] = None

        # 从 config 读取（兼容 benchmark 配置段）
        self.dataset_path = getattr(config, "benchmark_dataset_path", "local_datasets")
        self.batch_size = getattr(config, "benchmark_batch_size", 16)
        self.frame_interval = getattr(config, "frame_interval", 10)
        self.use_cloud = getattr(config, "benchmark_use_cloud", True)

        # 检索钩子列表（每次 init 新 memory_manager 时会重新挂入）
        self._retrieve_hooks: List[Callable] = []

        # 编码钩子列表（每次 init 新 frame_vectorizer 时会重新挂入）
        self._encode_hooks: List[tuple] = []  # [(fn, need_hidden_states, need_attentions), ...]

    def register_encode_hook(
        self,
        fn: Callable,
        need_hidden_states: bool = False,
        need_attentions: bool = False,
    ):
        """
        注册编码钩子，在 frame_vectorizer.encode_frames_batch 时调用。
        fn(frame_data_list, vectors, hidden_states, attentions)：
          - frame_data_list: List[FrameData]
          - vectors: np.ndarray, shape (N, dim)
          - hidden_states: Optional[Tuple]，仅当 need_hidden_states=True 时有值
          - attentions: Optional[Tuple]，仅当 need_attentions=True 时有值（softmax(QK^T)）
        """
        self._encode_hooks.append((fn, need_hidden_states, need_attentions))
        if self.frame_vectorizer is not None:
            self.frame_vectorizer.register_encode_hook(
                fn, need_hidden_states=need_hidden_states, need_attentions=need_attentions
            )

    def register_retrieve_hook(self, fn: Callable):
        """
        注册检索钩子，在每次 retrieve 时调用。
        fn(query_vector, all_scores)：
          - query_vector: np.ndarray, 查询向量
          - all_scores: List[float], 长度为 vector_count，all_scores[i] 为向量 i 与 query 的距离
        """
        self._retrieve_hooks.append(fn)
        if self.memory_manager is not None:
            self.memory_manager.register_retrieve_hook(fn)

    def _setup_logger(self):
        self.logger = logging.getLogger("VenusSystemMoti")
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

    def _resolve_path(self, p: str) -> Path:
        """解析相对路径为绝对路径"""
        base = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = base / p
        if not path.exists():
            path = Path(p)
        return path.resolve()

    def _get_video_path(self, dataset_name: str, video_id: str) -> Optional[str]:
        """
        根据数据集和视频 ID 获取本地视频路径。
        Video-MME: video_id 为 videoID（YouTube ID）；egoschema: video_id 为 video_idx（UUID）
        """
        if dataset_name == "egoschema":
            video_dir = getattr(
                self.config, "benchmark_video_dir_egoschema", "local_datasets/egoschema/videos"
            )
        elif dataset_name == "Video-MME":
            video_dir = getattr(
                self.config, "benchmark_video_dir_videomme", "local_datasets/Video-MME/data"
            )
        else:
            return None
        video_dir = self._resolve_path(video_dir)
        path = video_dir / f"{video_id}.mp4"
        return str(path) if path.exists() else None

    def _get_db_paths(self, dataset_name: str, video_id: str, subset: Optional[str] = None) -> tuple:
        """获取该视频的 faiss 和 databasemap 路径。
        支持配置中的绝对路径（如 /mnt/share/...）或相对路径（相对于项目根）。"""
        base = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if dataset_name == "egoschema":
            db_dir = getattr(self.config, "benchmark_db_dir_egoschema", "database/egoschema")
        elif dataset_name == "Video-MME":
            db_dir = getattr(self.config, "benchmark_db_dir_videomme", "database/videomme")
        else:
            db_dir = "database/benchmark"
        db_path = Path(db_dir) if Path(db_dir).is_absolute() else base / db_dir
        if subset:
            db_path = db_path / subset
        faiss_dir = db_path / "faiss"
        json_dir = db_path / "json"
        faiss_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        faiss_path = str(faiss_dir / f"{video_id}.faiss")
        map_path = str(json_dir / f"{video_id}.json")
        return faiss_path, map_path

    def _init_components(
        self,
        video_path: Optional[str] = None,
        faiss_path: Optional[str] = None,
        map_path: Optional[str] = None,
    ):
        """
        初始化各组件（同步模式）。
        video_path 为 None 时仅初始化 query 相关组件（用于 skip_inject）。
        """
        self.config.memory_mode = "both"
        if faiss_path is not None:
            self.config.memory_faiss_file_path = faiss_path
        if map_path is not None:
            self.config.memory_databasemap_file_path = map_path

        self.memory_manager = MemoryManager(self.config)
        self.query_vectorizer = QueryVectorizer(self.config)

        self.memory_manager.init_sync()
        for fn in self._retrieve_hooks:
            self.memory_manager.register_retrieve_hook(fn)
        self.query_vectorizer._initialize_vectorizer()

        if video_path:
            self.config.stream_video_source = "file"
            self.config.stream_video_file_path = video_path
            self.stream_input = StreamInput(self.config)
            self.frame_vectorizer = FrameVectorizer(self.config)
            for fn, need_hs, need_attn in self._encode_hooks:
                self.frame_vectorizer.register_encode_hook(
                    fn, need_hidden_states=need_hs, need_attentions=need_attn
                )
            self.stream_input.init_for_file(video_path)
            self.frame_vectorizer._initialize_vectorizer()
        else:
            self.stream_input = None
            self.frame_vectorizer = None

    def _run_inject_phase(
        self,
        video_path: str,
        video_id: str,
        dataset_name: str,
        subset: Optional[str] = None,
        force_update: bool = True,
    ) -> Dict[str, Any]:
        """Inject 阶段：按 batch 读取、向量化、插入，按视频名保存"""
        faiss_path, map_path = self._get_db_paths(dataset_name, video_id, subset)
        if os.path.isfile(faiss_path) and not force_update:
            self.logger.info(f"向量库已存在，跳过 inject: {faiss_path}")
            self._init_components(video_path=None, faiss_path=faiss_path, map_path=map_path)
            idx = faiss.read_index(faiss_path)
            return {
                "total_frames": idx.ntotal,
                "total_vectors": idx.ntotal,
                "elapsed_sec": 0,
                "batch_size": self.batch_size,
                "skipped": True,
            }

        if os.path.isfile(faiss_path) and force_update:
            try:
                os.remove(faiss_path)
                self.logger.info(f"已删除旧向量库，将重新 inject: {faiss_path}")
            except OSError as e:
                self.logger.warning(f"删除 faiss 文件失败: {e}")
            if os.path.isfile(map_path):
                try:
                    os.remove(map_path)
                except OSError:
                    pass

        self._init_components(video_path=video_path, faiss_path=faiss_path, map_path=map_path)

        total_frames = 0
        total_vectors = 0
        t0 = time.time()

        step = max(1, self.frame_interval)
        num_encoded = (self.stream_input.total_frames + step - 1) // step
        batch_iter = self.stream_input.iter_frames_batch(
            batch_size=self.batch_size, frame_interval=self.frame_interval
        )
        with tqdm(total=num_encoded, unit="frame", desc="Encoding") as pbar:
            for batch in batch_iter:
                total_frames += len(batch)
                vector_data_list = self.frame_vectorizer.encode_frames_batch(batch)
                self.memory_manager.add_vectors_batch(vector_data_list)
                total_vectors += len(vector_data_list)
                pbar.update(len(batch))
                self.logger.debug(f"已处理 {total_frames} 帧，插入 {total_vectors} 向量")

        self.memory_manager.current_video_name = None
        self.memory_manager.save_database_sync()
        elapsed = time.time() - t0
        self.logger.info(
            f"Inject 完成: {total_frames} 帧 -> {total_vectors} 向量, 耗时 {elapsed:.2f}s"
        )
        return {
            "total_frames": total_frames,
            "total_vectors": total_vectors,
            "elapsed_sec": elapsed,
            "batch_size": self.batch_size,
            "skipped": False,
        }

    def _get_reasoner(self) -> Reasoner:
        """懒加载 Reasoner"""
        if self.reasoner is None:
            self.reasoner = Reasoner(self.config)
            self.logger.info("已初始化 Reasoner（同步推理）")
        return self.reasoner

    def _get_video_time(self, map_path: Optional[str] = None) -> Optional[float]:
        """获取视频时长（秒）"""
        if self.stream_input is not None and hasattr(self.stream_input, "video_duration"):
            return getattr(self.stream_input, "video_duration", None) or 0
        if map_path and os.path.isfile(map_path):
            try:
                with open(map_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "frames" in data:
                    duration = data.get("duration")
                    if duration is not None:
                        return float(duration)
                    tf = data.get("total_frames", 0)
                    fps = data.get("video_fps", 1)
                    return tf / fps if fps and fps > 0 else None
                if isinstance(data, list) and data and isinstance(data[0], dict) and "frames" in data[0]:
                    v = data[0]
                    duration = v.get("duration")
                    if duration is not None:
                        return float(duration)
                    tf = v.get("total_frames", 0)
                    fps = v.get("video_fps", 1)
                    return tf / fps if fps and fps > 0 else None
            except (json.JSONDecodeError, IOError):
                pass
        return None

    def _run_query_single(
        self,
        question: str,
        sample_id: str = "",
        sample: Optional[Dict[str, Any]] = None,
        video_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """单次查询：编码 -> 检索 -> 推理"""
        t0 = time.time()
        query_vector = self.query_vectorizer.encode_query_sync(question)
        frame_list, scores, frames_metadata = self.memory_manager.retrieve_sync(query_vector)
        retrieve_time = time.time() - t0

        result = {
            "question": question,
            "retrieve_time_sec": retrieve_time,
            "scores": scores,
            "retrieved_frames": frame_list if frame_list else [],
            "retrieved_frames_metadata": frames_metadata if frames_metadata else [],
        }

        query_text = question
        select_frame_num = len(frame_list) if frame_list else 0
        if video_time is not None and frame_list and sample is not None:
            options = sample.get("options", [])
            if not isinstance(options, list):
                options = list(options) if options else []
            query_text = build_rag_prompt(
                video_time=video_time,
                num_selected_frame=len(frame_list),
                question=question,
                options=options,
            )
        result["rag_question"] = query_text
        result["select_frame_num"] = select_frame_num

        if self.use_cloud and frame_list:
            frames_rgb = [
                cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if f.ndim == 3 else cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
                for f in frame_list
            ]
            query_request = QueryRequest(
                query_text=query_text,
                memory_results=frames_rgb,
                query_id=hash(sample_id) % (2**31) if sample_id else int(time.time()),
                dialog_id=0,
            )
            reasoner = self._get_reasoner()
            response = reasoner.infer_sync(query_request)
            result["cloud_result"] = response.result
            result["cloud_error"] = response.error
            result["total_time_sec"] = time.time() - t0
        else:
            result["cloud_result"] = None
            result["cloud_error"] = "use_cloud=False 或 无检索帧"
            result["total_time_sec"] = retrieve_time

        return result

    def inject_video(
        self,
        video_path: str,
        video_id: str,
        dataset_name: str = "Video-MME",
        subset: Optional[str] = None,
        force_update: bool = True,
    ) -> Dict[str, Any]:
        """
        对指定视频执行 inject（向量化并写入 faiss）。

        Args:
            video_path: 视频文件路径
            video_id: 视频 ID（用于保存 faiss/databasemap 文件名）
            dataset_name: 数据集名，用于确定 db 目录
            subset: 子集（如 short/medium/long）
            force_update: 若为 True（默认），即使 faiss 已存在也强制重新 inject；若为 False 则跳过

        Returns:
            inject 统计信息
        """
        return self._run_inject_phase(
            video_path, video_id, dataset_name, subset, force_update=force_update
        )

    def query_video(
        self,
        video_id: str,
        question: str,
        sample: Optional[Dict[str, Any]] = None,
        dataset_name: str = "Video-MME",
        subset: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        对已有向量库的视频执行单次 query。

        Args:
            video_id: 视频 ID（用于定位 faiss 文件）
            question: 问题文本
            sample: 可选，包含 options 等，用于构造 RAG 提示
            dataset_name: 数据集名
            subset: 子集

        Returns:
            查询结果
        """
        faiss_path, map_path = self._get_db_paths(dataset_name, video_id, subset)
        if not os.path.isfile(faiss_path):
            raise FileNotFoundError(f"向量库不存在: {faiss_path}")
        self._init_components(None, faiss_path=faiss_path, map_path=map_path)
        video_time = self._get_video_time(map_path=map_path)
        if video_time is None and sample is not None:
            video_time = 0.0  # 无时长时用 0
        return self._run_query_single(
            question,
            sample_id=sample.get("question_id", sample.get("question_idx", "")) if sample else "",
            sample=sample,
            video_time=video_time,
        )

    def run_video_flow(
        self,
        video_path: str,
        video_id: str,
        questions: List[Dict[str, Any]],
        dataset_name: str = "Video-MME",
        subset: Optional[str] = None,
        force_update: bool = True,
    ) -> Dict[str, Any]:
        """
        对指定视频执行完整流程：inject + 多次 query。

        Args:
            video_path: 视频文件路径（可为本地路径或通过 _get_video_path 解析）
            video_id: 视频 ID
            questions: 问题列表，每项为 dict，需含 "question"，可选 "options"、"answer" 等
            dataset_name: 数据集名
            subset: 子集
            force_update: 若为 True（默认），即使 faiss 已存在也强制重新 inject

        Returns:
            包含 inject_stats 和 query_results 的字典
        """
        if self.use_cloud:
            reasoner = self._get_reasoner()
            if not reasoner.test_mode and reasoner.model is None:
                reasoner._set_logger()
                reasoner._initialize_model()
                self.logger.info("已预加载 LLaVA 模型")

        inject_stats = self._run_inject_phase(
            video_path, video_id, dataset_name, subset, force_update=force_update
        )
        video_time = self._get_video_time()

        query_results = []
        for i, q in enumerate(questions):
            question = q.get("question", "")
            if not question:
                continue
            sample = q if isinstance(q, dict) else {"question": q, "options": []}
            r = self._run_query_single(
                question,
                sample_id=str(i),
                sample=sample,
                video_time=video_time or 0.0,
            )
            query_results.append(r)

        return {"inject_stats": inject_stats, "query_results": query_results}

    def run_image(self, image_path: str, question: str) -> Dict[str, Any]:
        """
        对指定图片执行推理（无 RAG，直接传入图片与问题）。

        Args:
            image_path: 图片文件路径
            question: 问题文本

        Returns:
            包含 cloud_result 等的字典
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        query_request = QueryRequest(
            query_text=question,
            memory_results=[img_rgb],
            query_id=int(time.time()),
            dialog_id=0,
        )
        reasoner = self._get_reasoner()
        response = reasoner.infer_sync(query_request)

        return {
            "question": question,
            "cloud_result": response.result,
            "cloud_error": response.error,
        }
