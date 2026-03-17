#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VenusSystemBench - 串行执行的云边集成 Benchmark 系统

支持：
- 按 batch 读取视频帧、向量化、插入记忆
- 使用 local_datasets 中的数据集（egoschema / Video-MME）进行测试
- 通过 config 控制数据集和子集
- 记录测试结果并保存
"""

import os
import sys
import json
import time
import logging
import faiss
import cv2
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.stream_input import StreamInput, FrameData
from src.benchmark.utils import build_rag_prompt
from src.frame_vectorizer import FrameVectorizer, FrameVectorData
from src.memory_manager import MemoryManager
from src.query_vectorizer import QueryVectorizer
from src.reasoner import Reasoner, QueryRequest


class VenusSystemBench:
    """云边集成 Benchmark 系统 - 串行执行，无进程/队列开销"""

    def __init__(self, config: Config = None):
        if config is None:
            config = Config()
        self.config = config
        self._setup_logger()

        # 组件（同步模式，不启动子进程）
        self.stream_input: Optional[StreamInput] = None
        self.frame_vectorizer: Optional[FrameVectorizer] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.query_vectorizer: Optional[QueryVectorizer] = None
        self.reasoner: Optional[Reasoner] = None  # 云端推理，benchmark 直接变量传递

        # 数据集路径（支持 local_datasets 软链接）
        self.dataset_path = getattr(
            config, "benchmark_dataset_path", "local_datasets"
        )
        self.batch_size = getattr(config, "benchmark_batch_size", 16)
        self.frame_interval = getattr(config, "frame_interval", 10)
        self.use_cloud = getattr(config, "benchmark_use_cloud", True)
        self.result_dir = getattr(config, "benchmark_result_dir", "benchmark_results")

    def _setup_logger(self):
        self.logger = logging.getLogger("VenusSystemBench")
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
        """获取该视频的 faiss 和 databasemap 路径，按数据集和 subset 分目录，faiss 存 faiss 子目录、json 存 json 子目录"""
        base = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if dataset_name == "egoschema":
            db_dir = getattr(self.config, "benchmark_db_dir_egoschema", "database/egoschema")
        elif dataset_name == "Video-MME":
            db_dir = getattr(self.config, "benchmark_db_dir_videomme", "database/videomme")
        else:
            db_dir = "database/benchmark"
        db_path = base / db_dir
        if subset:
            db_path = db_path / subset
        faiss_dir = db_path / "faiss"
        json_dir = db_path / "json"
        faiss_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        faiss_path = str(faiss_dir / f"{video_id}.faiss")
        map_path = str(json_dir / f"{video_id}.json")
        return faiss_path, map_path

    def _load_dataset(self):
        """根据 config 加载数据集并过滤子集"""
        dataset_name = getattr(self.config, "benchmark_dataset", "Video-MME")
        subset = getattr(self.config, "benchmark_subset", "short")

        base = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        parent = base / self.dataset_path
        if not parent.exists():
            parent = Path(self.dataset_path)
        ds_path = parent / dataset_name
        if not ds_path.exists():
            ds_path = parent
        ds_path = str(ds_path.resolve())

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("benchmark 需要 datasets 库: pip install datasets")

        if dataset_name == "egoschema":
            config_name = subset if subset else "Subset"
            ds = load_dataset(ds_path, config_name, split="test")
        elif dataset_name == "Video-MME":
            ds = load_dataset(ds_path, "videomme", split="test")
            if subset and subset in ("short", "medium", "long"):
                ds = ds.filter(lambda x: x["duration"] == subset)
        else:
            raise ValueError(f"不支持的 benchmark 数据集: {dataset_name}")

        self.logger.info(f"加载数据集 {dataset_name} 子集 {subset}，共 {len(ds)} 条")
        return ds, dataset_name, subset

    def _group_by_video(self, dataset, dataset_name: str) -> Dict[str, List[Dict]]:
        """按视频分组，返回 {video_id: [sample, ...]}"""
        video_key = "videoID" if dataset_name == "Video-MME" else "video_idx"
        groups = {}
        for i in range(len(dataset)):
            sample = dict(dataset[i])
            vid = sample.get(video_key, "")
            if vid not in groups:
                groups[vid] = []
            groups[vid].append(sample)
        return groups

    def _init_components(
        self,
        video_path: Optional[str] = None,
        faiss_path: Optional[str] = None,
        map_path: Optional[str] = None,
    ):
        """
        初始化各组件（同步模式）。
        video_path 为 None 时仅初始化 query 相关组件（用于 skip_inject）。
        faiss_path/map_path 指定向量库路径，不传则需在调用前设置到 config。
        """
        self.config.memory_mode = "both"
        if faiss_path is not None:
            self.config.memory_faiss_file_path = faiss_path
        if map_path is not None:
            self.config.memory_databasemap_file_path = map_path

        self.memory_manager = MemoryManager(self.config)
        self.query_vectorizer = QueryVectorizer(self.config)

        self.memory_manager.init_sync()
        self.query_vectorizer._initialize_vectorizer()

        if video_path:
            self.config.stream_video_source = "file"
            self.config.stream_video_file_path = video_path
            self.stream_input = StreamInput(self.config)
            self.frame_vectorizer = FrameVectorizer(self.config)
            self.stream_input.init_for_file(video_path)
            self.frame_vectorizer._initialize_vectorizer()
        else:
            self.stream_input = None
            self.frame_vectorizer = None

    def _run_inject_phase(
        self, video_path: str, video_id: str, dataset_name: str, subset: Optional[str] = None
    ) -> Dict[str, Any]:
        """Inject 阶段：按 batch 读取、向量化、插入，按视频名保存到数据集对应目录"""
        faiss_path, map_path = self._get_db_paths(dataset_name, video_id, subset)
        if os.path.isfile(faiss_path):
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

        self._init_components(video_path=video_path, faiss_path=faiss_path, map_path=map_path)

        total_frames = 0
        total_vectors = 0
        t0 = time.time()

        for batch in self.stream_input.iter_frames_batch(
            batch_size=self.batch_size, frame_interval=self.frame_interval
        ):
            total_frames += len(batch)
            vector_data_list = self.frame_vectorizer.encode_frames_batch(batch)
            self.memory_manager.add_vectors_batch(vector_data_list)
            total_vectors += len(vector_data_list)
            self.logger.debug(f"已处理 {total_frames} 帧，插入 {total_vectors} 向量")

        # benchmark 使用配置的路径保存，不按视频名覆盖
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
        """懒加载 Reasoner（benchmark 云边一体，直接变量传递，无需 gRPC）"""
        if self.reasoner is None:
            self.reasoner = Reasoner(self.config)
            self.logger.info("已初始化 Reasoner（同步推理，无 gRPC）")
        return self.reasoner

    def _get_video_time(self, map_path: Optional[str] = None) -> Optional[float]:
        """获取视频时长（秒）。优先从 stream_input，否则从 databasemap 文件读取。"""
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
        """单次查询：编码 -> 检索 -> 推理（直接变量传递，无 gRPC）。若提供 sample 和 video_time，则用 build_rag_prompt 构造 RAG 提示传给推理。"""
        t0 = time.time()
        query_vector = self.query_vectorizer.encode_query_sync(question)
        frame_list, scores, _ = self.memory_manager.retrieve_sync(query_vector)
        retrieve_time = time.time() - t0

        result = {"question": question, "retrieve_time_sec": retrieve_time, "scores": scores}

        # 构造传给 reasoner 的 query_text：有 RAG 参数则用 build_rag_prompt，否则用原始 question
        query_text = question
        select_frame_num = len(frame_list) if frame_list else 0
        assert video_time, "video_time 必须要有才能创建ragprompt"
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
            # 帧为 BGR，Reasoner 需要 RGB
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

    def _build_video_entry(
        self,
        video_id: str,
        query_results: List[Dict],
        create_v_db_time: float = 0,
    ) -> Dict[str, Any]:
        """根据单视频的 query 结果构建一个视频的 JSON 条目"""
        if not query_results:
            return {
                "video_id": video_id,
                "duration": "",
                "domain": "",
                "sub_category": "",
                "url": video_id,
                "create_v_db_time": create_v_db_time,
                "questions": [],
            }
        first = query_results[0]
        questions = []
        for r in query_results:
            q = {
                "question_id": r.get("sample_id", ""),
                "task_type": r.get("task_type", ""),
                "question": r.get("question", ""),
                "options": r.get("options", []),
                "answer": r.get("ground_truth", ""),
                "response": r.get("cloud_result") or "",
                "rag_question": r.get("rag_question", ""),
                "select_frame_num": r.get("select_frame_num", 0),
            }
            questions.append(q)
        return {
            "video_id": video_id,
            "duration": first.get("duration", ""),
            "domain": first.get("domain", ""),
            "sub_category": first.get("sub_category", ""),
            "url": first.get("url", video_id),
            "create_v_db_time": create_v_db_time,
            "questions": questions,
        }

    def _save_results(
        self,
        inject_stats_list: List[Dict],
        query_results: List[Dict],
        dataset_name: str,
        subset: str,
        video_paths: List[str],
        out_path: Optional[str] = None,
    ):
        """保存测试结果到 JSON，结构为按视频分组的列表。若指定 out_path 则写入该路径，否则生成新文件"""
        os.makedirs(self.result_dir, exist_ok=True)
        if out_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"benchmark_{dataset_name}_{subset}_{timestamp}.json"
            out_path = os.path.join(self.result_dir, fname)

        inject_by_video = {s["video_id"]: s.get("elapsed_sec", 0) for s in inject_stats_list}
        groups: Dict[str, List[Dict]] = {}
        for r in query_results:
            vid = r.get("video_id", "")
            if vid not in groups:
                groups[vid] = []
            groups[vid].append(r)

        result_list = []
        for video_id, results in groups.items():
            entry = self._build_video_entry(
                video_id, results, inject_by_video.get(video_id, 0)
            )
            result_list.append(entry)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_list, f, ensure_ascii=False, indent=2)

        self.logger.info(f"结果已保存: {out_path}")
        return out_path

    def _append_video_and_save(
        self,
        result_list: List[Dict],
        out_path: str,
        inject_stat: Dict[str, Any],
        query_results_for_video: List[Dict],
        video_id: str,
    ) -> List[Dict]:
        """将单个视频的结果追加到 result_list 并保存到 out_path。返回更新后的 result_list"""
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        create_v_db_time = inject_stat.get("elapsed_sec", 0)
        entry = self._build_video_entry(video_id, query_results_for_video, create_v_db_time)
        result_list.append(entry)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_list, f, ensure_ascii=False, indent=2)
        self.logger.info(f"结果已保存（增量）: {out_path}")
        return result_list

    def _load_resume_result(self, resume_path: str) -> tuple:
        """
        加载 resume JSON 文件，返回 (result_list, processed_video_ids)。
        若文件不存在或解析失败，返回 ([], set())。
        """
        path = Path(resume_path)
        if not path.is_absolute():
            base = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            path = (base / resume_path).resolve()
        if not path.exists():
            self.logger.warning(f"Resume 文件不存在: {path}")
            return [], set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                result_list = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.warning(f"Resume 文件解析失败: {e}")
            return [], set()
        if not isinstance(result_list, list):
            return [], set()
        processed = {e.get("video_id") for e in result_list if isinstance(e, dict) and e.get("video_id")}
        return result_list, processed

    def run(
        self,
        skip_inject: bool = False,
        max_queries: Optional[int] = None,
        max_videos: Optional[int] = None,
        resume_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        运行完整 benchmark 流程。视频路径从数据集配置的目录自动解析。

        Args:
            skip_inject: 若为 True，跳过 inject，仅做 query（需已有向量库）
            max_queries: 最多执行的查询数量，None 表示全部
            max_videos: 最多处理的视频数量，None 表示全部
            resume_path: 断点续跑用的 JSON 文件路径，将从中读取已处理的视频并跳过，从下一个未处理的视频继续

        Returns:
            包含 summary 和 result_path 的字典
        """
        # 非 no-cloud 模式下，启动时预加载 Reasoner 和大模型
        if self.use_cloud:
            reasoner = self._get_reasoner()
            if not reasoner.test_mode and reasoner.model is None:
                reasoner._set_logger()
                reasoner._initialize_model()
                self.logger.info("已预加载 LLaVA 模型")

        ds, dataset_name, subset = self._load_dataset()
        id_key = "question_id" if dataset_name == "Video-MME" else "question_idx"
        video_key = "videoID" if dataset_name == "Video-MME" else "video_idx"

        all_inject_stats = []
        all_query_results = []
        video_paths_used = []
        query_count = 0

        # Resume 支持：加载已处理结果，确定输出路径
        result_list: List[Dict] = []
        result_path: Optional[str] = None
        processed_video_ids: set = set()
        if resume_path:
            result_list, processed_video_ids = self._load_resume_result(resume_path)
            p = Path(resume_path)
            if not p.is_absolute():
                base = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                result_path = str((base / resume_path).resolve())
            else:
                result_path = resume_path
            if processed_video_ids:
                self.logger.info(f"Resume 模式：已加载 {len(processed_video_ids)} 个已处理视频，从下一未处理视频继续")

        if skip_inject:
            self.logger.info("跳过 Inject，使用已有向量库")
            groups = self._group_by_video(ds, dataset_name)
            if max_videos is not None:
                groups = dict(list(groups.items())[:max_videos])
            self.logger.info("Phase 2: Query（遍历数据集查询）")
            for video_id, samples in groups.items():
                if resume_path and video_id in processed_video_ids:
                    continue
                faiss_path, map_path = self._get_db_paths(dataset_name, video_id, subset)
                if not os.path.isfile(faiss_path):
                    self.logger.warning(f"向量库不存在，跳过视频 {video_id}: {faiss_path}")
                    continue
                self._init_components(None, faiss_path=faiss_path, map_path=map_path)
                video_time = self._get_video_time(map_path=map_path)
                if max_queries is not None:
                    remaining = max_queries - query_count
                    samples = samples[:remaining]
                video_query_results = []
                for sample in samples:
                    question = sample.get("question", "")
                    if not question:
                        continue
                    sample_id = sample.get(id_key, "")
                    gt = sample.get("answer", "")
                    r = self._run_query_single(
                        question, sample_id=str(sample_id), sample=sample, video_time=video_time
                    )
                    r["sample_id"] = sample_id
                    r["ground_truth"] = gt
                    r["video_id"] = video_id
                    r["duration"] = sample.get("duration", "")
                    r["domain"] = sample.get("domain", "")
                    r["sub_category"] = sample.get("sub_category", "")
                    r["url"] = sample.get("url", video_id)
                    r["task_type"] = sample.get("task_type", "")
                    r["options"] = sample.get("options", [])
                    if not isinstance(r["options"], list):
                        r["options"] = list(r["options"]) if r["options"] else []
                    all_query_results.append(r)
                    video_query_results.append(r)
                    query_count += 1
                    if query_count % 10 == 0:
                        self.logger.info(f"已查询 {query_count} 条")
                # 每完成一个视频即保存
                if video_query_results:
                    if result_path is None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        result_path = os.path.join(
                            self.result_dir, f"benchmark_{dataset_name}_{subset}_{timestamp}.json"
                        )
                    inject_stat = {"video_id": video_id, "elapsed_sec": 0}
                    result_list = self._append_video_and_save(
                        result_list, result_path, inject_stat, video_query_results, video_id
                    )
                if max_queries is not None and query_count >= max_queries:
                    break
        else:
            groups = self._group_by_video(ds, dataset_name)
            if max_videos is not None:
                groups = dict(list(groups.items())[:max_videos])

            for video_id, samples in groups.items():
                if resume_path and video_id in processed_video_ids:
                    continue
                video_path = self._get_video_path(dataset_name, video_id)
                if not video_path:
                    self.logger.warning(f"视频不存在，跳过: {video_id}")
                    continue

                if max_queries is not None and query_count >= max_queries:
                    break

                self.logger.info(f"处理视频: {video_id} ({len(samples)} 条问题)")
                self.logger.info("=" * 50)
                self.logger.info(f"Inject: {video_path}")
                self.logger.info("=" * 50)
                inject_stats = self._run_inject_phase(video_path, video_id, dataset_name, subset)
                inject_stat = {"video_id": video_id, "path": video_path, **inject_stats}
                all_inject_stats.append(inject_stat)
                video_paths_used.append(video_path)

                video_time = self._get_video_time()
                if max_queries is not None:
                    remaining = max_queries - query_count
                    samples = samples[:remaining]

                self.logger.info("Phase 2: Query")
                video_query_results = []
                for sample in samples:
                    question = sample.get("question", "")
                    if not question:
                        continue
                    sample_id = sample.get(id_key, "")
                    gt = sample.get("answer", "")
                    r = self._run_query_single(
                        question, sample_id=str(sample_id), sample=sample, video_time=video_time
                    )
                    r["sample_id"] = sample_id
                    r["ground_truth"] = gt
                    r["video_id"] = video_id
                    r["duration"] = sample.get("duration", "")
                    r["domain"] = sample.get("domain", "")
                    r["sub_category"] = sample.get("sub_category", "")
                    r["url"] = sample.get("url", video_id)
                    r["task_type"] = sample.get("task_type", "")
                    r["options"] = sample.get("options", [])
                    if not isinstance(r["options"], list):
                        r["options"] = list(r["options"]) if r["options"] else []
                    all_query_results.append(r)
                    video_query_results.append(r)
                    query_count += 1
                    if query_count % 10 == 0:
                        self.logger.info(f"已查询 {query_count} 条")

                # 每完成一个视频即保存
                if video_query_results:
                    if result_path is None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        result_path = os.path.join(
                            self.result_dir, f"benchmark_{dataset_name}_{subset}_{timestamp}.json"
                        )
                    result_list = self._append_video_and_save(
                        result_list, result_path, inject_stat, video_query_results, video_id
                    )

                if max_queries is not None and query_count >= max_queries:
                    break

        # 汇总 inject 统计（仅本次运行处理的视频）
        if all_inject_stats:
            agg = {
                "total_frames": sum(s["total_frames"] for s in all_inject_stats),
                "total_vectors": sum(s["total_vectors"] for s in all_inject_stats),
                "elapsed_sec": sum(s["elapsed_sec"] for s in all_inject_stats),
                "batch_size": self.batch_size,
                "videos": all_inject_stats,
            }
        else:
            agg = {}

        return {
            "summary": agg,
            "query_count": len(all_query_results),
            "result_path": result_path or "",
        }
