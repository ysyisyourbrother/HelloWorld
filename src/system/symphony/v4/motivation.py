#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Symphony motivation system (v4)."""

import os
import time
from typing import Any, Dict, List, Optional

import cv2

from src.agent.prompts_for_symphony import rag_prompt_with_frames_and_subtitles
from src.benchmark.retrieve_clip_frames import (
    count_existing_clip_mp4s,
    count_reported_frames_in_clip_info,
    decode_clip_info_all_frames_bgr,
)
from src.llm.reasoner import QueryRequest
from src.memory.subtitle.asr import SymASR, SymStreamASR
from src.system.symphony.v3.motivation import SymphonySystemMotiV3
from src.video_input.audio_input import AudioInputBase


class SymphonySystemMotiV4(SymphonySystemMotiV3):
    """v4 Motivation 编排：在 v3 基础上增加 ASR 注入与字幕增强 RAG。"""

    def _inject_srt_by_asr(self, video_path: str) -> int:
        audio_input = AudioInputBase(self.config)
        audio_input.init_source(video_path)

        if bool(getattr(self.config, "audio_stream_as_chunks", False)):
            asr = SymStreamASR(self.config)
            total = 0
            for audio_data in audio_input.iter_audio_data():
                segments = asr.transcribe_audio_data(audio_data)
                if not segments:
                    continue
                items = [
                    {
                        "start_time": float(seg.start_time),
                        "end_time": float(seg.end_time),
                        "text": str(seg.text),
                    }
                    for seg in segments
                ]
                self.memory_manager._add_subtitle(items)
                total += len(items)
            tail_segments = asr.finish()
            if tail_segments:
                tail_items = [
                    {
                        "start_time": float(seg.start_time),
                        "end_time": float(seg.end_time),
                        "text": str(seg.text),
                    }
                    for seg in tail_segments
                ]
                self.memory_manager._add_subtitle(tail_items)
                total += len(tail_items)
            return total

        asr = SymASR(self.config)
        all_items = []
        for audio_data in audio_input.iter_audio_data():
            segments = asr.transcribe_audio_data(audio_data)
            if not segments:
                continue
            all_items.extend(
                {
                    "start_time": float(seg.start_time),
                    "end_time": float(seg.end_time),
                    "text": str(seg.text),
                }
                for seg in segments
            )
        if all_items:
            self.memory_manager._add_subtitle(all_items)
        return len(all_items)

    def _build_subtitle_context(
        self,
        frames_metadata: Optional[List[Dict[str, Any]]],
        video_time: Optional[float],
    ) -> str:
        if not frames_metadata:
            return ""
        top1 = frames_metadata[0]
        frame_id = top1.get("frame_id")
        video_fps = top1.get("video_fps") or 1.0
        if frame_id is None:
            return ""
        center_sec = float(frame_id) / float(video_fps)
        start_t = max(0.0, center_sec - 15.0)
        end_cap = float(video_time) if video_time is not None else (center_sec + 15.0)
        end_t = min(end_cap, center_sec + 15.0)
        rows = self.memory_manager.retrieve_segment_by_period(start_t=start_t, end_t=end_t)
        if not rows:
            return ""
        gathered_subtitles = " ".join(str(r.get("text") or "").strip() for r in rows).strip()
        if not gathered_subtitles:
            return ""
        if float(rows[0].get("start_time", 0.0)) > start_t:
            gathered_subtitles = "... " + gathered_subtitles
        if float(rows[-1].get("end_time", 0.0)) < end_t:
            gathered_subtitles = gathered_subtitles + " ..."
        return gathered_subtitles

    def _run_inject_phase(
        self,
        video_path: str,
        video_id: str,
        dataset_name: str,
        subset: Optional[str] = None,
        force_update: bool = True,
    ) -> Dict[str, Any]:
        result = super()._run_inject_phase(
            video_path=video_path,
            video_id=video_id,
            dataset_name=dataset_name,
            subset=subset,
            force_update=force_update,
        )
        if result.get("skipped"):
            return result

        srt_path = getattr(self.memory_manager, "srt_file_path", "") or getattr(
            self.config, "memory_srt_file_path", ""
        )
        if srt_path and os.path.isfile(srt_path):
            os.remove(srt_path)
            if self.memory_manager.srt is not None:
                self.memory_manager.srt.latest_idx = 0
                self.memory_manager.srt.start_time = []
                self.memory_manager.srt.end_time = []
                self.memory_manager.srt.texts = []
                self.memory_manager.srt.whole_texts = ""
        subtitle_count = self._inject_srt_by_asr(video_path)
        self.memory_manager.save_database_sync()
        result["subtitle_count"] = subtitle_count
        return result

    def _run_query_single(
        self,
        question: str,
        sample_id: str = "",
        sample: Optional[Dict[str, Any]] = None,
        video_time: Optional[float] = None,
        dialog_id: int = 0,
    ) -> Dict[str, Any]:
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
        rit = rit.strip().lower() if isinstance(rit, str) else "frame"
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
            gathered_subtitles = self._build_subtitle_context(frames_metadata, video_time)
            query_text = rag_prompt_with_frames_and_subtitles.format(
                video_time=float(video_time),
                num_selected_frame=int(len(frames_metadata)),
                gathered_subtitles=gathered_subtitles,
                question=question,
                options_text=" ".join(options),
            )
            result["gathered_subtitles"] = gathered_subtitles

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

        is_local_vlm = bool(getattr(self.config, "benchmark_is_local_vlm", True))
        if not is_local_vlm:
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
            return result

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
            qid = (hash(sample_id) % (2**31)) if sample_id else int(time.time())
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
            result["cloud_error"] = "use_cloud=True 但无可用图像（clip 解码失败且无检索帧）"
            result["total_time_sec"] = time.time() - t0
        else:
            result["cloud_result"] = None
            result["cloud_error"] = "use_cloud=False，跳过推理"
            result["total_time_sec"] = retrieve_time

        return result
