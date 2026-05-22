#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Symphony motivation system (v1)."""

import logging
import os
import sys
import time
from typing import Dict, Any, Optional

import faiss
from tqdm import tqdm

_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
sys.path.insert(0, _project_root)

from src.config import SymConfig, system_mode_wants_memory_reinject
from src.memory.frame.frame_vectorizer import SymFrameVectorizerByGOP
from src.memory.memory_manager import MemoryManagerBase
from src.memory.query.query_vectorizer import QueryVectorizer
from src.system.vrag.motivation import VragSystemMoti
from src.video_input.video_input import make_sym_video_input

class SymphonySystemMoti(VragSystemMoti):
    """
    继承 VragSystemMoti，使用 SymVideoInput / SymVideoInputV2（由 config video_input.version 选择）和 SymFrameVectorizer 按 GOP 进行 inject。
    按 select_strategy 从每个 GOP 中选帧后编码，不依赖编码钩子。
    """

    def __init__(self, config=None):
        if config is None:
            config = SymConfig(config_path="configs/symconfig_moti.json")
        super().__init__(config)

    def _setup_logger(self):
        self.logger = logging.getLogger("SymphonySystemMoti")
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
        srt_path: Optional[str] = None,
    ):
        """初始化各组件，使用 make_sym_video_input（V1/V2）和 SymFrameVectorizer（不注册编码钩子）"""
        if faiss_path is not None:
            self.config.memory_faiss_file_path = faiss_path
        if map_path is not None:
            self.config.memory_databasemap_file_path = map_path
        if srt_path is not None:
            self.config.memory_srt_file_path = srt_path

        self.memory_manager = MemoryManagerBase(self.config)
        self.query_vectorizer = QueryVectorizer(self.config)

        self.memory_manager.init_sync()
        for fn in self._retrieve_hooks:
            self.memory_manager.register_retrieve_hook(fn)
        self.query_vectorizer._initialize_vectorizer()

        if video_path:
            self.config.video_file_path = video_path
            self.video_input = make_sym_video_input(self.config)
            self.frame_vectorizer = SymFrameVectorizerByGOP(self.config)
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
        """Inject 阶段：按 GOP 迭代、select_frame_in_gop 选帧、encode_frames_by_gop 编码、插入"""
        faiss_path, map_path, srt_path = self._get_db_paths(dataset_name, video_id, subset)
        if not system_mode_wants_memory_reinject(self.config.system_mode):
            if os.path.isfile(faiss_path):
                self.logger.info(
                    "system_mode 未启用记忆重注入，加载已有向量库: %s",
                    faiss_path,
                )
                self._init_components(
                    video_path=None,
                    faiss_path=faiss_path,
                    map_path=map_path,
                    srt_path=srt_path,
                )
                idx = faiss.read_index(faiss_path)
                return {
                    "total_frames": idx.ntotal,
                    "total_vectors": idx.ntotal,
                    "elapsed_sec": 0,
                    "batch_size": self.batch_size,
                    "skipped": True,
                }
            self.logger.error(
                "system_mode 未启用记忆重注入但本地无 faiss: %s",
                faiss_path,
            )
            return {
                "total_frames": 0,
                "total_vectors": 0,
                "elapsed_sec": 0.0,
                "batch_size": self.batch_size,
                "skipped": False,
                "error": "missing_faiss_for_local_only_mode",
            }
        if os.path.isfile(faiss_path) and not force_update:
            self.logger.info(f"向量库已存在，跳过 inject: {faiss_path}")
            self._init_components(
                video_path=None, faiss_path=faiss_path, map_path=map_path, srt_path=srt_path
            )
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
            if os.path.isfile(srt_path):
                try:
                    os.remove(srt_path)
                except OSError:
                    pass

        self._init_components(
            video_path=video_path, faiss_path=faiss_path, map_path=map_path, srt_path=srt_path
        )

        total_frames = 0
        total_vectors = 0
        t0 = time.time()

        num_gops = len(self.video_input.i_frame_indices)
        with tqdm(total=num_gops, unit="gop", desc="Encoding") as pbar:
            for gop_start, gop_end in self.video_input.iter_gop_ranges():
                total_frames += gop_end - gop_start
                vector_data_list = self.frame_vectorizer.encode_frames_by_gop_from_video_input(
                    self.video_input, gop_start, gop_end
                )
                self.memory_manager.add_vectors_batch(vector_data_list)
                total_vectors += len(vector_data_list)
                pbar.update(1)
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
