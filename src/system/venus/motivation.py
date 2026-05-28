#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Venus motivation system：继承 VragSystemMoti，场景聚类注入 + Venus 检索。"""

import importlib.util
import inspect
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from decord import VideoReader, cpu
from PIL import Image
from scenedetect import ContentDetector, detect
from sklearn.cluster import KMeans
from tqdm import tqdm

_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, _project_root)

from src.agent.prompt_for_venus import (
    build_ocr_text,
    build_rag_question,
    build_single_frame_inject_text,
    build_yolo_text,
)
from src.config import Config, system_mode_online_memory_needs_query_encoder, system_mode_wants_frame_inject
from src.memory.frame.frame_vectorizer import FrameVectorData
from src.memory.frame.image_bge_vectorizer import ImageBGEVectorizer
from src.memory.memory_manager import MemoryManagerBase
from src.memory.query.query_vectorizer import QueryVectorizer
from src.memory.query.text_bge_vectorizer import TextBGEVectorizer
from src.system.vrag.motivation import VragSystemMoti

_VENUS_READ_BATCH = 32
_VENUS_ENCODE_BATCH = 8
_VENUS_KMEANS_NITER = 25
_VENUS_SCENE_THRESHOLD = 27.0
_VENUS_SCENE_MIN_LEN = 15
_VENUS_SIMU_KMEANS = True

class VenusInjectRetrieveMixin:
    """Venus 注入与检索逻辑，供 Moti / Bench 共用。"""

    _venus_enhance_path: Optional[str] = None
    _venus_enhance_doc: Dict[str, Any] = None
    _venus_image_encoder: Optional[ImageBGEVectorizer] = None
    _venus_text_encoder: Optional[TextBGEVectorizer] = None
    _venus_ocr_model = None
    _venus_yolo_model = None

    def _get_db_paths(
        self, dataset_name: str, video_id: str, subset: Optional[str] = None
    ) -> tuple:
        faiss_path, map_path, srt_path = super()._get_db_paths(dataset_name, video_id, subset)
        map_p = Path(map_path)
        enhance_path = str(map_p.parent.parent / "enhance" / map_p.name)
        enhance_dir = Path(enhance_path).parent
        enhance_dir.mkdir(parents=True, exist_ok=True)
        self._venus_enhance_path = enhance_path
        return faiss_path, map_path, srt_path

    def _venus_remove_db_files(self, faiss_path: str, map_path: str, srt_path: str):
        ep = getattr(self, "_venus_enhance_path", None)
        for p in (faiss_path, map_path, srt_path, ep):
            if p and os.path.isfile(p):
                os.remove(p)

    def _venus_load_enhance_doc(self):
        self._venus_enhance_doc = {}
        ep = getattr(self, "_venus_enhance_path", None)
        if ep and os.path.isfile(ep):
            with open(ep, "r", encoding="utf-8") as f:
                self._venus_enhance_doc = json.load(f)

    def _get_video_time(self, map_path: Optional[str] = None) -> Optional[float]:
        mp = map_path or getattr(self.config, "memory_databasemap_file_path", None)
        if mp and os.path.isfile(mp):
            with open(mp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if data.get("duration") is not None:
                    return float(data["duration"])
                tf = data.get("total_frames", 0)
                fps = data.get("video_fps", 1)
                if fps and fps > 0:
                    return float(tf) / float(fps)
        return super()._get_video_time(map_path=map_path)

    def _venus_save_enhance_doc(self, doc: Dict[str, Any]):
        ep = getattr(self, "_venus_enhance_path", None)
        if not ep:
            return
        with open(ep, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        self._venus_enhance_doc = doc

    def _init_components(
        self,
        video_path: Optional[str] = None,
        faiss_path: Optional[str] = None,
        map_path: Optional[str] = None,
        srt_path: Optional[str] = None,
    ):
        if faiss_path is not None:
            self.config.memory_faiss_file_path = faiss_path
        if map_path is not None:
            self.config.memory_databasemap_file_path = map_path
        if srt_path is not None:
            self.config.memory_srt_file_path = srt_path

        self.memory_manager = MemoryManagerBase(self.config)
        if system_mode_online_memory_needs_query_encoder(self.config.system_mode):
            self.query_vectorizer = QueryVectorizer(self.config)
            self.query_vectorizer._initialize_vectorizer()
        else:
            self.query_vectorizer = None

        self.memory_manager.init_sync()
        for fn in getattr(self, "_retrieve_hooks", []):
            self.memory_manager.register_retrieve_hook(fn)
        self._venus_load_enhance_doc()
        self.video_input = None
        self.frame_vectorizer = None

    def _venus_subsample_indices(
        self, total_frames: int, video_fps: float, target_fps: float
    ) -> List[int]:
        if total_frames < 1:
            return []
        if target_fps <= 0 or video_fps <= 0:
            return list(range(total_frames))
        step = float(video_fps) / float(target_fps)
        indices = []
        pos = 0.0
        while int(pos) < total_frames:
            idx = int(pos)
            if not indices or indices[-1] != idx:
                indices.append(idx)
            pos += step
        if indices[-1] != total_frames - 1:
            indices.append(total_frames - 1)
        return indices

    def _venus_scene_frame_range(self, start_tc, end_tc) -> Tuple[int, int]:
        f0 = int(start_tc.frame_num)
        f1 = int(end_tc.frame_num) - 1
        return f0, f1

    def _venus_ensure_enhance_models(self):
        if self._venus_image_encoder is None:
            self._venus_image_encoder = ImageBGEVectorizer(
                self.config.frame_device,
                self.config.frame_model_path,
                attn_implementation=self.config.frame_attn_implementation,
            )
        if self._venus_text_encoder is None:
            self._venus_text_encoder = TextBGEVectorizer(
                self.config.query_device, self.config.query_model_path
            )
        if self._venus_ocr_model is None:
            if importlib.util.find_spec("easyocr") is None:
                raise ImportError("未安装 easyocr")
            if importlib.util.find_spec("ultralytics") is None:
                raise ImportError("未安装 ultralytics")
            import easyocr
            from ultralytics import YOLO

            langs = [str(self.config.ocr_language).strip() or "en"]
            self._venus_ocr_model = easyocr.Reader(langs, gpu=False, verbose=False)
            self._venus_yolo_model = YOLO(self.config.yolo_model_path)

    def _venus_ocr_yolo_text(self, frame_rgb: np.ndarray, frame_bgr: np.ndarray) -> Tuple[str, str, str]:
        self._venus_ensure_enhance_models()
        rows = self._venus_ocr_model.readtext(frame_rgb)
        ocr_t = build_ocr_text(rows, confidence_thre=self.config.ocr_conf_threshold)
        yolo_res = self._venus_yolo_model(frame_bgr, verbose=False)
        yolo_t = ""
        if yolo_res:
            yolo_t = build_yolo_text(yolo_res[0], confidence_thre=self.config.yolo_conf_threshold)
        inject_t = build_single_frame_inject_text(ocr_t, yolo_t)
        return ocr_t, yolo_t, inject_t

    def _venus_load_and_encode_frames(
        self, vr: VideoReader, frames: List[int]
    ) -> Tuple[List[Image.Image], List[int], np.ndarray]:
        pil_list: List[Image.Image] = []
        frame_ids: List[int] = []
        enc_parts = []
        j = 0
        while j < len(frames):
            sub = frames[j : j + _VENUS_READ_BATCH]
            j += _VENUS_READ_BATCH
            batch = vr.get_batch(np.array(sub, dtype=np.int64)).asnumpy()
            for t in range(len(sub)):
                pil_list.append(Image.fromarray(batch[t]))
                frame_ids.append(int(sub[t]))
            enc_parts.append(self._venus_encode_pils(pil_list[-len(sub) :]))
        if enc_parts:
            X = np.concatenate(enc_parts, axis=0)
        else:
            X = np.zeros((0, 0), dtype=np.float32)
        return pil_list, frame_ids, X

    def _venus_encode_pils(self, pil_list: List[Image.Image]) -> np.ndarray:
        chunks = []
        bs = _VENUS_ENCODE_BATCH
        for i in range(0, len(pil_list), bs):
            part = pil_list[i : i + bs]
            t = self._venus_image_encoder.encode_batch_no_norm(part)
            chunks.append(t.detach().float().cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)

    def _venus_kmeans_centers(
        self, vectors: np.ndarray, k: int
    ) -> List[int]:
        n = int(vectors.shape[0])
        if n < 1:
            return []
        X = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
        faiss.normalize_L2(X)
        ncent = min(int(k), n)
        if n < int(k):
            return list(range(n))
        km = KMeans(
            n_clusters=ncent,
            max_iter=_VENUS_KMEANS_NITER,
            random_state=1234,
            n_init=1,
        )
        km.fit(X)
        C = np.ascontiguousarray(km.cluster_centers_, dtype=np.float32)
        faiss.normalize_L2(C)
        out = []
        for ci in range(ncent):
            dists = np.sum((X - C[ci : ci + 1]) ** 2, axis=1)
            out.append(int(np.argmin(dists)))
        return out

    def _venus_kmeans_centers_simul(self, n: int, k: int) -> List[int]:
        """
        在场景内 n 个已编码帧上按 k 均匀取局部下标，模拟聚类中心选取（不做 KMeans）。
        返回长度为 min(k, n) 的下标列表。
        """
        n = int(n)
        k = int(k)
        if n < 1:
            return []
        ncent = min(k, n)
        if ncent < k:
            return list(range(n))
        if ncent == 1:
            return [n // 2]
        out = []
        for i in range(ncent):
            idx = int(round(i * (n - 1) / float(ncent - 1)))
            if not out or out[-1] != idx:
                out.append(idx)
        return out

    def _venus_fuse_vectors(
        self, frame_vec: np.ndarray, text_vec: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """未归一化的图像/文本特征相加后，再对最后一维做 L2 归一化。"""
        fv = np.asarray(frame_vec, dtype=np.float32).reshape(1, -1)
        if text_vec is not None:
            tv = np.asarray(text_vec, dtype=np.float32).reshape(1, -1)
            fused = fv + tv
        else:
            fused = fv.copy()
        faiss.normalize_L2(fused)
        return fused

    def _venus_run_inject(self, video_path: str) -> Tuple[List[FrameVectorData], Dict[str, Any], int]:
        self._venus_ensure_enhance_models()
        vr = VideoReader(str(video_path), ctx=cpu(0))
        n_video = len(vr)
        if n_video < 1:
            raise ValueError("视频可读帧数为 0")
        video_fps = float(vr.get_avg_fps()) or 30.0
        duration = n_video / video_fps
        subsampled = self._venus_subsample_indices(
            n_video, video_fps, float(self.config.video_target_fps)
        )
        detector = ContentDetector(
            threshold=_VENUS_SCENE_THRESHOLD, min_scene_len=_VENUS_SCENE_MIN_LEN
        )
        sig = inspect.signature(detect)
        kw = {}
        if "start_in_scene" in sig.parameters:
            kw["start_in_scene"] = True
        scenes = detect(str(video_path), detector, show_progress=False, **kw)
        if not scenes:
            raise ValueError("场景列表为空")

        k_target = int(self.config.venus_k_clusters)
        vector_list: List[FrameVectorData] = []
        enhance_scenes: List[Dict[str, Any]] = []
        for st_tc, en_tc in scenes:
            f0, f1 = self._venus_scene_frame_range(st_tc, en_tc)
            f0 = max(0, min(f0, n_video - 1))
            f1 = max(0, min(f1, n_video - 1))
            if f1 < f0:
                continue
            scene_frames = [f for f in subsampled if f0 <= f <= f1]
            if not scene_frames:
                continue

            if _VENUS_SIMU_KMEANS:
                center_local = self._venus_kmeans_centers_simul(len(scene_frames), k_target)
                picked_frames = [scene_frames[i] for i in center_local]
                pil_list, frame_ids, X = self._venus_load_and_encode_frames(vr, picked_frames)
                encode_indices = list(range(len(frame_ids)))
            else:
                pil_list, frame_ids, X = self._venus_load_and_encode_frames(vr, scene_frames)
                encode_indices = self._venus_kmeans_centers(X, k_target)

            center_ids = [frame_ids[i] for i in encode_indices]
            ocr_texts = []
            yolo_texts = []

            for li in encode_indices:
                fid = frame_ids[li]
                rgb = np.array(pil_list[li])
                bgr = rgb[:, :, ::-1].copy()
                ocr_t, yolo_t, inject_t = self._venus_ocr_yolo_text(rgb, bgr)
                ocr_texts.append(ocr_t)
                yolo_texts.append(yolo_t)
                fv = X[li : li + 1].copy()
                if inject_t.strip():
                    tv = self._venus_text_encoder.encode_no_norm(inject_t)
                    tv = tv.detach().float().cpu().numpy()
                    if len(tv.shape) == 1:
                        tv = tv.reshape(1, -1)
                    fused = self._venus_fuse_vectors(fv, tv)
                else:
                    fused = self._venus_fuse_vectors(fv, None)
                vector_list.append(
                    FrameVectorData(
                        vector=fused,
                        timestamp=float(fid) / video_fps,
                        frame_id=int(fid),
                        source_path=str(video_path),
                        total_frames=n_video,
                        video_fps=video_fps,
                        duration=duration,
                    )
                )

            enhance_scenes.append(
                {
                    "scene_start": int(f0),
                    "scene_end": int(f1),
                    "scene_frames": scene_frames,
                    "center_frames": center_ids,
                    "ocr_texts": ocr_texts,
                    "yolo_texts": yolo_texts,
                }
            )

        return vector_list, {"scenes": enhance_scenes}, n_video

    def _venus_all_scores(self, query_vector: np.ndarray) -> List[float]:
        n = self.memory_manager.vector_count
        if n < 1:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        distances, indices = self.memory_manager.index.search(q, n)
        all_scores = [0.0] * n
        for idx, dist in zip(indices[0].tolist(), distances[0].tolist()):
            if 0 <= idx < n:
                all_scores[idx] = float(dist)
        return all_scores

    def _venus_topk_ids(self, all_scores: List[float], top_k: int) -> List[int]:
        k = min(int(top_k), len(all_scores))
        if k < 1:
            return []
        order = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)
        return order[:k]

    def _venus_softmax(self, scores: List[float], tau: float) -> List[float]:
        t = max(float(tau), 1e-8)
        exps = [math.exp(float(s) / t) for s in scores]
        z = sum(exps)
        return [e / z for e in exps] if z > 0 else [1.0 / len(scores)] * len(scores)

    def _venus_progressive_pick(self, topk_ids: List[int], topk_scores: List[float]) -> Dict[int, int]:
        probs = self._venus_softmax(topk_scores, self.config.venus_retrieve_tau)
        p_max = max(probs) if probs else 1.0
        theta = float(self.config.venus_retrieve_theta)
        beta = max(float(self.config.venus_retrieve_beta), 1e-8)
        n_min = max(1, int(math.ceil(beta * math.ceil(theta / max(p_max, 1e-12)))))
        n_max = int(self.config.venus_retrieve_n_max)
        counts: Dict[int, int] = {}
        picked = set()
        rng = random.Random(1234)
        n_draws = 0
        while n_draws < n_max:
            pick = rng.choices(range(len(topk_ids)), weights=probs, k=1)[0]
            vid = topk_ids[pick]
            picked.add(vid)
            counts[vid] = counts.get(vid, 0) + 1
            n_draws += 1
            cum_p = sum(probs[i] for i, v in enumerate(topk_ids) if v in picked)
            if n_draws >= n_min and cum_p / beta >= theta:
                break
        while n_draws < n_min:
            pick = rng.choices(range(len(topk_ids)), weights=probs, k=1)[0]
            vid = topk_ids[pick]
            counts[vid] = counts.get(vid, 0) + 1
            n_draws += 1
        return counts

    def _venus_vector_scene_map(self) -> Dict[int, int]:
        mapping = {}
        vid = 0
        for si, scene in enumerate((self._venus_enhance_doc or {}).get("scenes") or []):
            for _ in scene.get("center_frames") or []:
                mapping[vid] = si
                vid += 1
        return mapping

    def _venus_uniform_scene_frames(self, scene_frames: List[int], n_pick: int) -> List[int]:
        if n_pick < 1 or not scene_frames:
            return []
        frames = sorted(scene_frames)
        if n_pick >= len(frames):
            return list(frames)
        return sorted(random.Random(1234).sample(frames, n_pick))

    def _venus_retrieve_frame_ids(self, query_vector: np.ndarray) -> Tuple[List[int], List[float]]:
        all_scores = self._venus_all_scores(query_vector)
        top_k = int(self.config.memory_topk)
        doc = self._venus_enhance_doc or {}
        strategy = str(self.config.venus_retrieve_strategy)

        if strategy == "threshold":
            th = float(self.config.venus_retrieve_threshold)
            above = [(i, all_scores[i]) for i in range(len(all_scores)) if all_scores[i] > th]
            if above:
                v2s = self._venus_vector_scene_map()
                scenes = doc.get("scenes") or []
                frames = []
                for vid, sc in above:
                    si = v2s.get(vid)
                    if si is None:
                        continue
                    centers = scenes[si].get("center_frames") or []
                    off = sum(len(scenes[j].get("center_frames") or []) for j in range(si))
                    loc = vid - off
                    if 0 <= loc < len(centers):
                        frames.append(int(centers[loc]))
                uniq = sorted(set(frames))
                return uniq, [1.0] * len(uniq)
            topk_ids = self._venus_topk_ids(all_scores, top_k)
            if not topk_ids:
                return [], []
            v2s = self._venus_vector_scene_map()
            scenes = doc.get("scenes") or []
            frames = []
            for vid in topk_ids:
                si = v2s.get(vid)
                if si is None:
                    continue
                centers = scenes[si].get("center_frames") or []
                off = sum(len(scenes[j].get("center_frames") or []) for j in range(si))
                loc = vid - off
                if 0 <= loc < len(centers):
                    frames.append(int(centers[loc]))
            uniq = sorted(set(frames))
            sc0 = all_scores[topk_ids[0]] if topk_ids else 0.0
            return uniq, [sc0] * len(uniq)

        topk_ids = self._venus_topk_ids(all_scores, top_k)
        if not topk_ids:
            return [], []
        topk_scores = [all_scores[i] for i in topk_ids]
        counts = self._venus_progressive_pick(topk_ids, topk_scores)
        v2s = self._venus_vector_scene_map()
        scenes = doc.get("scenes") or []
        picked: List[int] = []
        for si, n_pick in self._venus_scene_pick_counts(counts, v2s).items():
            if 0 <= si < len(scenes):
                sf = scenes[si].get("scene_frames") or []
                picked.extend(self._venus_uniform_scene_frames(sf, n_pick))
        uniq = sorted(set(picked))
        return uniq, [1.0] * len(uniq)

    def _venus_scene_pick_counts(
        self, vec_counts: Dict[int, int], v2s: Dict[int, int]
    ) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for vid, cnt in vec_counts.items():
            si = v2s.get(vid)
            if si is not None:
                out[si] = out.get(si, 0) + int(cnt)
        return out

    def _venus_metadata_from_frame_ids(
        self, frame_ids: List[int], scores: List[float]
    ) -> Tuple[List[float], List[dict]]:
        base = self.memory_manager.databasemap[0] if self.memory_manager.vector_count > 0 else {}
        meta_list = []
        sc_list = []
        for fid, sc in zip(frame_ids, scores):
            meta_list.append(
                {
                    "frame_id": int(fid),
                    "video_fps": base.get("video_fps") or 1.0,
                    "source_path": base.get("source_path"),
                    "total_frames": base.get("total_frames"),
                    "i_frames": base.get("i_frames") or [],
                }
            )
            sc_list.append(float(sc))
        return sc_list, meta_list

    def _venus_ocr_yolo_for_frames(self, frame_ids: List[int]) -> Tuple[List[str], List[str]]:
        ocr_out, yolo_out = [], []
        doc = self._venus_enhance_doc or {}
        for fid in frame_ids:
            ocr_t, yolo_t = "", ""
            for scene in doc.get("scenes") or []:
                centers = [int(x) for x in (scene.get("center_frames") or [])]
                if int(fid) not in centers:
                    continue
                idx = centers.index(int(fid))
                ocr_rows = scene.get("ocr_texts") or []
                yolo_rows = scene.get("yolo_texts") or []
                if idx < len(ocr_rows):
                    ocr_t = ocr_rows[idx]
                if idx < len(yolo_rows):
                    yolo_t = yolo_rows[idx]
                break
            ocr_out.append(ocr_t)
            yolo_out.append(yolo_t)
        return ocr_out, yolo_out

    def _run_inject_phase_venus(
        self,
        video_path: str,
        video_id: str,
        dataset_name: str,
        subset: Optional[str] = None,
        skip_if_exists: bool = False,
    ) -> Dict[str, Any]:
        faiss_path, map_path, srt_path = self._get_db_paths(dataset_name, video_id, subset)
        if not system_mode_wants_frame_inject(self.config.system_mode):
            if os.path.isfile(faiss_path):
                self._init_components(None, faiss_path, map_path, srt_path)
                idx = faiss.read_index(faiss_path)
                return {
                    "total_frames": idx.ntotal,
                    "total_vectors": idx.ntotal,
                    "elapsed_sec": 0,
                    "batch_size": self.batch_size,
                    "skipped": True,
                }
            return {
                "total_frames": 0,
                "total_vectors": 0,
                "elapsed_sec": 0.0,
                "batch_size": self.batch_size,
                "skipped": False,
                "error": "missing_faiss_for_local_only_mode",
            }

        if skip_if_exists and os.path.isfile(faiss_path):
            self.logger.info("向量库已存在，跳过 inject: %s", faiss_path)
            self._init_components(None, faiss_path, map_path, srt_path)
            idx = faiss.read_index(faiss_path)
            return {
                "total_frames": idx.ntotal,
                "total_vectors": idx.ntotal,
                "elapsed_sec": 0,
                "batch_size": self.batch_size,
                "skipped": True,
            }

        if os.path.isfile(faiss_path):
            self.logger.info("删除旧库，重新 Venus inject: %s", faiss_path)
            self._venus_remove_db_files(faiss_path, map_path, srt_path)

        self._init_components(None, faiss_path, map_path, srt_path)
        t0 = time.time()
        vector_list, enhance_doc, n_video = self._venus_run_inject(video_path)
        for vd in vector_list:
            self.memory_manager.add_vectors_batch([vd])
        self.memory_manager.current_video_name = None
        self.memory_manager.save_database_sync()
        self._venus_save_enhance_doc(enhance_doc)
        elapsed = time.time() - t0
        self.logger.info(
            "Venus Inject: %d 向量, %d 场景, %.2fs",
            len(vector_list),
            len(enhance_doc.get("scenes") or []),
            elapsed,
        )
        return {
            "total_frames": n_video,
            "total_vectors": len(vector_list),
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
        from src.config import system_mode_wants_vlm_qa

        t0 = time.time()
        if self.query_vectorizer is None:
            self.query_vectorizer = QueryVectorizer(self.config)
            self.query_vectorizer._initialize_vectorizer()
        query_vector = self.query_vectorizer.encode_query_sync(question)

        hooks = getattr(self, "_retrieve_hooks", None) or []
        if hooks:
            all_scores = self._venus_all_scores(query_vector)
            for fn in hooks:
                fn(query_vector.copy(), all_scores)

        frame_ids, retrieve_scores = self._venus_retrieve_frame_ids(query_vector)
        scores, frames_metadata = self._venus_metadata_from_frame_ids(frame_ids, retrieve_scores)

        if self.memory_manager.memory_save_retrieved_frames and frames_metadata:
            vids, scs = [], []
            for meta in frames_metadata:
                fid = int(meta["frame_id"])
                for i in range(self.memory_manager.vector_count):
                    rec = self.memory_manager.databasemap[i]
                    if int(rec.get("frame_id", -1)) == fid:
                        vids.append(i)
                        scs.append(1.0)
                        break
            if vids:
                self.memory_manager._save_retrieved_frames(vids, scs)

        clip_info = self.memory_manager._export_retrieve_clips_if_needed(
            frames_metadata, dialog_id
        )
        retrieve_time = time.time() - t0

        result = {
            "question": question,
            "retrieve_time_sec": retrieve_time,
            "scores": scores,
            "retrieved_frames_metadata": frames_metadata,
        }

        query_text = question
        select_frame_num = len(frames_metadata)
        if video_time is not None and frames_metadata and sample is not None:
            options = sample.get("options", [])
            if not isinstance(options, list):
                options = list(options) if options else []
            ocr_list, yolo_list = self._venus_ocr_yolo_for_frames(
                [int(m["frame_id"]) for m in frames_metadata]
            )
            query_text = build_rag_question(
                video_time=float(video_time),
                len_selected_frame=select_frame_num,
                selected_det=yolo_list,
                selected_ocr=ocr_list,
                selected_asr=[],
                question=question,
                options=options,
            )
        result["rag_question"] = query_text
        result["select_frame_num"] = select_frame_num

        if not system_mode_wants_vlm_qa(self.config.system_mode):
            result["cloud_result"] = None
            result["cloud_error"] = "system_mode 未启用 VLM 问答，仅检索"
            result["total_time_sec"] = time.time() - t0
            return result

        if self.use_cloud:
            sid = sample_id or str(abs(hash(question)) % (2**31))
            self._fill_reasoner_result(
                t0, retrieve_time, query_text, frames_metadata, sid, result, clip_info=clip_info
            )
        else:
            result["cloud_result"] = None
            result["cloud_error"] = "use_cloud=False，跳过推理"
            result["total_time_sec"] = retrieve_time
        return result


class VenusSystemMoti(VenusInjectRetrieveMixin, VragSystemMoti):
    """Motivation：继承 VragSystemMoti，覆盖 Venus 注入与检索。"""

    def __init__(self, config: Config = None):
        if config is None:
            config = Config(config_path="configs/config_moti.json")
        super().__init__(config)
        self._venus_enhance_doc = {}

    def _setup_logger(self):
        self.logger = logging.getLogger("VenusSystemMoti")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(h)
        self.logger.propagate = False

    def _run_inject_phase(
        self,
        video_path: str,
        video_id: str,
        dataset_name: str,
        subset: Optional[str] = None,
        force_update: bool = True,
    ) -> Dict[str, Any]:
        return self._run_inject_phase_venus(
            video_path, video_id, dataset_name, subset, skip_if_exists=not force_update
        )
