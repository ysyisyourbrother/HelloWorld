#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Symphony benchmark system."""

import logging
import os
import sys
import time
from typing import Dict, Any, Optional

import faiss

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.memory.frame_vectorizer import SymFrameVectorizer
from src.memory.memory_manager import MemoryManager
from src.memory.query_vectorizer import QueryVectorizer
from src.system.venus.benchmark import VenusSystemBench
from src.video_input.video_input import SymVideoInput

class SymphonySystemBench(VenusSystemBench):
    """
    继承 VenusSystemBench，使用 SymVideoInput 和 SymFrameVectorizer 按 GOP 进行 inject。
    按 select_strategy 从每个 GOP 中选帧后编码，不依赖钩子逻辑。
    """

    def _setup_logger(self):
        self.logger = logging.getLogger("SymphonySystemBench")
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
        """初始化各组件，使用 SymVideoInput 和 SymFrameVectorizer"""
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
            self.config.video_file_path = video_path
            self.video_input = SymVideoInput(self.config)
            self.frame_vectorizer = SymFrameVectorizer(self.config)
            self.video_input.init_for_file(video_path)
            self.frame_vectorizer._initialize_vectorizer()
        else:
            self.video_input = None
            self.frame_vectorizer = None

    def _run_inject_phase(
        self, video_path: str, video_id: str, dataset_name: str, subset: Optional[str] = None
    ) -> Dict[str, Any]:
        """Inject 阶段：按 GOP 迭代、select_frame_in_gop 选帧、encode_frames_by_gop 编码、插入"""
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

        for gop_start, gop_end in self.video_input.iter_gop_ranges():
            total_frames += gop_end - gop_start
            vector_data_list = self.frame_vectorizer.encode_frames_by_gop_from_video_input(
                self.video_input, gop_start, gop_end
            )
            self.memory_manager.add_vectors_batch(vector_data_list)
            total_vectors += len(vector_data_list)
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
