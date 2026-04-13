#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Symphony motivation system (v3)."""

import logging
import os
import sys
import time
from typing import Any, Dict, Optional

import faiss

_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
sys.path.insert(0, _project_root)

from src.benchmark.utils import build_rag_prompt
from src.config import SymConfig
from src.memory.frame_vectorizer import SymFrameVectorizer
from src.memory.memory_manager import MemoryManagerBase
from src.memory.query_vectorizer import QueryVectorizer
from src.system.symphony.v1.motivation import SymphonySystemMoti
from src.video_input.video_input import make_sym_video_input


class SymphonySystemMotiV3(SymphonySystemMoti):
    """v3 Motivation 编排：沿用 GOP 注入，并补齐 clip 检索结果字段。"""

    def __init__(self, config=None):
        if config is None:
            config = SymConfig(config_path="configs/symconfig_moti.json")
        super().__init__(config)

    def _setup_logger(self):
        self.logger = logging.getLogger("SymphonySystemMotiV3")
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

    def _init_components(
        self,
        video_path: Optional[str] = None,
        faiss_path: Optional[str] = None,
        map_path: Optional[str] = None,
    ):
        """初始化组件：SymVideoInput + SymFrameVectorizer（支持编码/检索钩子重挂载）。"""
        self.config.memory_mode = "both"
        if faiss_path is not None:
            self.config.memory_faiss_file_path = faiss_path
        if map_path is not None:
            self.config.memory_databasemap_file_path = map_path

        self.memory_manager = MemoryManagerBase(self.config)
        self.query_vectorizer = QueryVectorizer(self.config)

        self.memory_manager.init_sync()
        for fn in self._retrieve_hooks:
            self.memory_manager.register_retrieve_hook(fn)
        self.query_vectorizer._initialize_vectorizer()

        if video_path:
            self.config.video_file_path = video_path
            self.video_input = make_sym_video_input(self.config)
            self.frame_vectorizer = SymFrameVectorizer(self.config)
            for fn, need_hs, need_attn in self._encode_hooks:
                self.frame_vectorizer.register_encode_hook(
                    fn, need_hidden_states=need_hs, need_attentions=need_attn
                )
            self.video_input.init_for_file(video_path)
            ixs = getattr(self.video_input, "i_frame_indices", None)
            if ixs:
                self.memory_manager.register_i_frames(video_path, list(ixs))
            self.frame_vectorizer._initialize_vectorizer()
        else:
            self.video_input = None
            self.frame_vectorizer = None

    def _run_inject_phase(
        self,
        video_path: str,
        video_id: str,
        dataset_name: str,
        subset: Optional[str] = None,
        force_update: bool = True,
    ) -> Dict[str, Any]:
        """Inject：按 GOP 选帧编码并写入向量库。"""
        faiss_path, map_path = self._get_db_paths(dataset_name, video_id, subset)
        if os.path.isfile(faiss_path) and not force_update:
            self.logger.info("向量库已存在，跳过 inject: %s", faiss_path)
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
                self.logger.info("已删除旧向量库，将重新 inject: %s", faiss_path)
            except OSError as e:
                self.logger.warning("删除 faiss 文件失败: %s", e)
            if os.path.isfile(map_path):
                try:
                    os.remove(map_path)
                except OSError:
                    pass

        self._init_components(video_path=video_path, faiss_path=faiss_path, map_path=map_path)

        total_frames = 0
        total_vectors = 0
        t0 = time.time()

        for gop_start, gop_end in self.video_input.iter_gop_ranges():
            total_frames += gop_end - gop_start
            vector_data_list = self.frame_vectorizer.encode_frames_by_gop_from_video_input(
                self.video_input, gop_start, gop_end
            )
            self.memory_manager.add_vectors_batch(vector_data_list)
            total_vectors += len(vector_data_list)
            self.logger.debug("已处理 %s 帧，插入 %s 向量", total_frames, total_vectors)

        self.memory_manager.current_video_name = None
        self.memory_manager.save_database_sync()
        elapsed = time.time() - t0
        self.logger.info(
            "Inject 完成: %s 帧 -> %s 向量, 耗时 %.2fs",
            total_frames,
            total_vectors,
            elapsed,
        )
        return {
            "total_frames": total_frames,
            "total_vectors": total_vectors,
            "elapsed_sec": elapsed,
            "batch_size": self.batch_size,
            "skipped": False,
        }

    def _run_query_single(
        self,
        question: str,
        sample_id: str = "",
        sample: Optional[Dict[str, Any]] = None,
        video_time: Optional[float] = None,
        dialog_id: int = 0,
    ) -> Dict[str, Any]:
        """单次查询：编码 -> 检索（支持 dialog_id）-> 返回 Motivation 结果。"""
        t0 = time.time()
        query_vector = self.query_vectorizer.encode_query_sync(question)
        scores, frames_metadata, clip_info = self.memory_manager.retrieve_sync(
            query_vector, dialog_id=dialog_id
        )
        retrieve_time = time.time() - t0

        result = {
            "question": question,
            "retrieve_time_sec": retrieve_time,
            "scores": scores,
            "retrieved_frames": [],
            "retrieved_frames_metadata": frames_metadata if frames_metadata else [],
        }

        rit = getattr(self.memory_manager, "memory_retrieve_item_type", None) or getattr(
            self.config, "memory_retrieve_item_type", "frame"
        )
        if isinstance(rit, str):
            rit = rit.strip().lower()
        else:
            rit = "frame"
        if rit not in ("frame", "clip"):
            rit = "frame"
        result["retrieve_item_type"] = rit
        result["retrieve_dialog_id"] = int(dialog_id)

        if clip_info:
            result["retrieve_clip_paths"] = clip_info.get("paths") or []
            result["retrieve_clip_dir"] = clip_info.get("dir")
            if clip_info.get("error"):
                result["retrieve_clip_error"] = clip_info.get("error")
        else:
            result["retrieve_clip_paths"] = []
            result["retrieve_clip_dir"] = None

        query_text = question
        select_frame_num = len(frames_metadata) if frames_metadata else 0
        if video_time is not None and frames_metadata and sample is not None:
            options = sample.get("options", [])
            if not isinstance(options, list):
                options = list(options) if options else []
            query_text = build_rag_prompt(
                video_time=video_time,
                num_selected_frame=len(frames_metadata),
                question=question,
                options=options,
            )
        result["rag_question"] = query_text
        result["select_frame_num"] = select_frame_num

        result["cloud_result"] = None
        result["cloud_error"] = "实时模式已切换为仅返回检索元数据，不再返回检索帧"
        result["total_time_sec"] = retrieve_time
        return result

    def query_video(
        self,
        video_id: str,
        question: str,
        sample: Optional[Dict[str, Any]] = None,
        dataset_name: str = "Video-MME",
        subset: Optional[str] = None,
        dialog_id: int = 0,
    ) -> Dict[str, Any]:
        """对已有向量库执行单次 query（支持 dialog_id）。"""
        faiss_path, map_path = self._get_db_paths(dataset_name, video_id, subset)
        if not os.path.isfile(faiss_path):
            raise FileNotFoundError("向量库不存在: {}".format(faiss_path))
        self._init_components(None, faiss_path=faiss_path, map_path=map_path)
        video_time = self._get_video_time(map_path=map_path)
        if video_time is None and sample is not None:
            video_time = 0.0
        return self._run_query_single(
            question,
            sample_id=sample.get("question_id", sample.get("question_idx", "")) if sample else "",
            sample=sample,
            video_time=video_time,
            dialog_id=dialog_id,
        )


# 兼容旧命名：保持与 v1 风格一致
SymphonySystemMoti = SymphonySystemMotiV3
