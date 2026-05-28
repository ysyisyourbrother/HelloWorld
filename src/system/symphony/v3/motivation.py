#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Symphony motivation system (v3)."""

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import cv2
import faiss

_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
sys.path.insert(0, _project_root)

from src.benchmark.retrieve_clip_frames import (
    count_existing_clip_mp4s,
    count_reported_frames_in_clip_info,
    decode_clip_info_all_frames_bgr,
)
from src.agent.prompts_for_symphony import (
    rag_prompt_with_clips,
    rag_prompt_with_frames,
)
from src.config import (
    SymConfig,
    system_mode_wants_any_memory_inject,
    system_mode_wants_frame_inject,
    system_mode_wants_vlm_qa,
)
from src.llm.reasoner import QueryRequest
from src.memory.frame.frame_vectorizer import SymFrameVectorizerForV3
from src.memory.memory_manager import MemoryManagerBase
from src.memory.query.query_vectorizer import QueryVectorizer
from src.system.symphony.v1.motivation import SymphonySystemMoti
from src.video_input.video_input import SymVideoInputByStreamWindow
from src.video_utils.about_frame import extract_frame_by_index


class SymphonySystemMotiV3(SymphonySystemMoti):
    """v3 Motivation 编排：流式时间窗选帧注入；clip 模式补齐 GOP mp4 路径，RAG 与 frame 计数字段与 Benchmark 语义对齐。"""

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
        srt_path: Optional[str] = None,
    ):
        """初始化组件：SymVideoInputByStreamWindow + SymFrameVectorizerForV3（支持编码/检索钩子重挂载）。"""
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
            self.video_input = SymVideoInputByStreamWindow(self.config)
            self.frame_vectorizer = SymFrameVectorizerForV3(self.config)
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
        """Inject：按媒体时间窗选帧编码并写入向量库。"""
        faiss_path, map_path, srt_path = self._get_db_paths(dataset_name, video_id, subset)
        if not system_mode_wants_any_memory_inject(self.config.system_mode):
            if os.path.isfile(faiss_path):
                self.logger.info(
                    "system_mode 未启用记忆注入，加载已有向量库: %s",
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
                "system_mode 未启用记忆注入但本地无 faiss: %s",
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

        if not system_mode_wants_frame_inject(self.config.system_mode):
            self._init_components(
                video_path=video_path,
                faiss_path=faiss_path,
                map_path=map_path,
                srt_path=srt_path,
            )
            ntotal = 0
            if os.path.isfile(faiss_path):
                idx = faiss.read_index(faiss_path)
                ntotal = idx.ntotal
            self.logger.info(
                "system_mode 未启用帧注入，跳过帧编码（保留已有 faiss/srt）: %s",
                faiss_path,
            )
            return {
                "total_frames": ntotal,
                "total_vectors": ntotal,
                "elapsed_sec": 0.0,
                "batch_size": self.batch_size,
                "skipped": False,
                "frame_inject_skipped": True,
            }

        if os.path.isfile(faiss_path) and not force_update:
            self.logger.info("向量库已存在，跳过 inject: %s", faiss_path)
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
                self.logger.info("已删除旧向量库，将重新 inject: %s", faiss_path)
            except OSError as e:
                self.logger.warning("删除 faiss 文件失败: %s", e)
            if os.path.isfile(map_path):
                try:
                    os.remove(map_path)
                except OSError:
                    pass

        self._init_components(
            video_path=video_path, faiss_path=faiss_path, map_path=map_path, srt_path=srt_path
        )

        total_video_frames = int(getattr(self.video_input, "total_frames", 0) or 0)
        total_vectors = 0
        t0 = time.time()

        for batch in self.video_input.iter_frames_by_stream_window():
            if not batch:
                continue
            vector_data_list = self.frame_vectorizer.encode_sym_frames_list(batch)
            if vector_data_list:
                self.memory_manager.add_vectors_batch(vector_data_list)
                total_vectors += len(vector_data_list)
            self.logger.debug("已插入 %s 向量（视频总帧约 %s）", total_vectors, total_video_frames)

        self.memory_manager.current_video_name = None
        self.memory_manager.save_database_sync()
        elapsed = time.time() - t0
        self.logger.info(
            "Inject 完成(流式窗): 视频总帧约 %s -> %s 向量, 耗时 %.2fs",
            total_video_frames,
            total_vectors,
            elapsed,
        )
        return {
            "total_frames": total_video_frames,
            "total_vectors": total_vectors,
            "elapsed_sec": elapsed,
            "batch_size": self.batch_size,
            "skipped": False,
        }

    def _decode_retrieval_frames_bgr(
        self, frames_metadata: Optional[List[Dict[str, Any]]]
    ):
        """与 Benchmark 一致：按元数据从整段视频解 top-k 单帧（BGR）。"""
        out = []
        for m in frames_metadata or []:
            source_path = m.get("source_path")
            frame_id = m.get("frame_id")
            if not source_path or frame_id is None:
                continue
            if not os.path.isfile(source_path):
                continue
            try:
                frame = extract_frame_by_index(
                    video_path=source_path,
                    frame_index=int(frame_id),
                    backend="cv2",
                )
            except Exception as e:
                self.logger.debug(
                    "motivation 解码检索帧失败 path=%s idx=%s: %s",
                    source_path,
                    frame_id,
                    e,
                )
                continue
            if frame is not None:
                out.append(frame)
        return out

    def _run_query_single(
        self,
        question: str,
        sample_id: str = "",
        sample: Optional[Dict[str, Any]] = None,
        video_time: Optional[float] = None,
        dialog_id: int = 0,
    ) -> Dict[str, Any]:
        """单次查询：编码 -> 检索（支持 dialog_id）；若 ``self.use_cloud`` 为真则调用 Reasoner。"""
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

        if clip_info and clip_info.get("error"):
            result["retrieve_clip_error"] = clip_info.get("error")

        query_text = question
        select_frame_num = len(frames_metadata) if frames_metadata else 0
        num_existing_clips = count_existing_clip_mp4s(clip_info) if clip_info else 0
        clip_paths_ok = (
            rit == "clip"
            and clip_info
            and not clip_info.get("error")
            and num_existing_clips > 0
        )
        clip_bgr_list = (
            decode_clip_info_all_frames_bgr(clip_info, logger=self.logger)
            if clip_paths_ok
            else []
        )

        if video_time is not None and frames_metadata and sample is not None:
            options = sample.get("options", [])
            if not isinstance(options, list):
                options = list(options) if options else []
            options_text = " ".join(options)
            if clip_paths_ok and clip_bgr_list:
                query_text = rag_prompt_with_clips.format(
                    video_time=float(video_time),
                    num_selected_clips=int(num_existing_clips),
                    question=question,
                    options_text=options_text,
                )
            else:
                query_text = rag_prompt_with_frames.format(
                    video_time=float(video_time),
                    num_selected_frame=int(len(frames_metadata)),
                    question=question,
                    options_text=options_text,
                )
        result["rag_question"] = query_text
        result["select_frame_num"] = select_frame_num
        if rit == "clip":
            result["select_clip_num"] = num_existing_clips
            if clip_bgr_list:
                result["reasoner_input_frame_count"] = len(clip_bgr_list)
            else:
                result["reasoner_input_frame_count"] = (
                    count_reported_frames_in_clip_info(clip_info) if clip_paths_ok else 0
                )
        else:
            result["select_clip_num"] = 0
            result["reasoner_input_frame_count"] = select_frame_num

        if not system_mode_wants_vlm_qa(self.config.system_mode):
            result["cloud_result"] = None
            result["cloud_error"] = "system_mode 未启用 VLM 问答，仅检索"
            result["total_time_sec"] = time.time() - t0
            return result

        if self._benchmark_uses_api_vlm():
            if rit == "clip" and clip_paths_ok and clip_info:
                paths = self.memory_manager.list_retrieved_media_paths(clip_info)
                self._fill_reasoner_from_media_paths(
                    t0, retrieve_time, query_text, paths, str(sample_id), result
                )
            else:
                self._fill_reasoner_result(
                    t0,
                    retrieve_time,
                    query_text,
                    frames_metadata,
                    str(sample_id),
                    result,
                    clip_info=clip_info,
                )
        else:
            if rit == "clip" and clip_bgr_list:
                frame_list_bgr = clip_bgr_list
            else:
                frame_list_bgr = self._decode_retrieval_frames_bgr(frames_metadata)

            if self.use_cloud and frame_list_bgr:
                frames_rgb = [
                    cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    if f.ndim == 3
                    else cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
                    for f in frame_list_bgr
                ]
                qid = (
                    (hash(sample_id) % (2**31))
                    if sample_id
                    else int(time.time())
                )
                query_request = QueryRequest(
                    query_text=query_text,
                    memory_results=frames_rgb,
                    query_id=qid,
                    dialog_id=int(dialog_id),
                )
                reasoner = self._get_reasoner()
                response = reasoner.infer_sync(query_request)
                result["cloud_result"] = response.result
                result["cloud_error"] = response.error
                result["total_time_sec"] = time.time() - t0
            elif self.use_cloud:
                result["cloud_result"] = None
                result["cloud_error"] = (
                    "use_cloud=True 但无可用图像（clip 解码失败且无检索帧）"
                )
                result["total_time_sec"] = time.time() - t0
            else:
                result["cloud_result"] = None
                result["cloud_error"] = "use_cloud=False，跳过推理"
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
        faiss_path, map_path, srt_path = self._get_db_paths(dataset_name, video_id, subset)
        if not os.path.isfile(faiss_path):
            raise FileNotFoundError("向量库不存在: {}".format(faiss_path))
        self._init_components(None, faiss_path=faiss_path, map_path=map_path, srt_path=srt_path)
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
