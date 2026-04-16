import json
import os
import threading
from contextlib import contextmanager


class ThreadSafeMap:
    """线程安全的 id-frame 映射类，按视频聚合存储，减少冗余。
    内部结构: [_videos] 每项为 {"source_path", "total_frames", "video_fps", "duration",
    "frames": [frame_id, ...], "i_frames": [I 帧索引, ...]}（i_frames 可选，由编排层 register_i_frames 写入）
    FAISS 索引 i 对应: 按顺序遍历 _videos，累加 frames 长度，定位到对应视频和 frame_id。
    """

    def __init__(self, videos: list = None):
        self._videos = videos if videos is not None else []
        self._lock = threading.RLock()
        # source_path -> I 帧索引列表；在首条向量 append 前由 register_i_frames 暂存
        self._pending_i_frames = {}

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
        """将 FAISS 索引转换为 {source_path, frame_id, total_frames, video_fps, duration, i_frames?}"""
        offset = 0
        for v in self._videos:
            n = len(v["frames"])
            if index < offset + n:
                rec = {
                    "source_path": v["source_path"],
                    "frame_id": v["frames"][index - offset],
                    "total_frames": v["total_frames"],
                    "video_fps": v["video_fps"],
                    "duration": v.get("duration"),
                }
                ixs = v.get("i_frames")
                if ixs:
                    rec["i_frames"] = ixs
                else:
                    rec["i_frames"] = []
                return rec
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
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def load_local(self, path: str = None) -> bool:
        """从本地加载 databasemap，支持新格式和旧格式（数组逐条记录）"""
        if not os.path.isfile(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
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
        """确保视频对象包含 duration，若缺失则根据 total_frames/video_fps 计算；规范化 i_frames。"""
        v = dict(v)
        if "duration" not in v or v["duration"] is None:
            tf = v.get("total_frames")
            fps = v.get("video_fps")
            v = dict(v)
            v["duration"] = tf / fps if tf is not None and fps and fps > 0 else None
        raw_ix = v.get("i_frames")
        if raw_ix is None:
            v["i_frames"] = []
        else:
            try:
                v["i_frames"] = sorted(set(int(x) for x in raw_ix))
            except (TypeError, ValueError):
                v["i_frames"] = []
        return v

    def _convert_legacy_to_videos(self, records: list) -> list:
        """将旧格式 [{"source_path", "frame_id", ...}, ...] 转为按视频聚合的新格式，保持插入顺序"""
        from collections import OrderedDict

        by_path = OrderedDict()
        for r in records:
            key = (r["source_path"], r["total_frames"], r["video_fps"])
            if key not in by_path:
                entry = {
                    "source_path": r["source_path"],
                    "total_frames": r["total_frames"],
                    "video_fps": r["video_fps"],
                    "frames": [],
                    "i_frames": [],
                }
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
                self._videos[-1].setdefault("i_frames", [])
                self._videos[-1]["frames"].append(fid)
                if duration is not None and "duration" not in self._videos[-1]:
                    self._videos[-1]["duration"] = duration
            else:
                pending_ix = self._pending_i_frames.pop(sp, None)
                init_ix = list(pending_ix) if pending_ix is not None else []
                self._videos.append(
                    {
                        "source_path": sp,
                        "total_frames": tf,
                        "video_fps": fps,
                        "duration": duration,
                        "frames": [fid],
                        "i_frames": init_ix,
                    }
                )

    def register_i_frames(self, source_path, i_frame_indices):
        """
        由编排层在开始处理某个 source_path 时调用，写入该视频的 I 帧索引列表（与 SymVideoInputByGOP 一致）。

        若该路径已在 _videos 中存在，则就地更新 i_frames；否则暂存，待首条 append 同路径时并入。
        """
        with self.acquire():
            if not source_path:
                return
            try:
                indices = sorted(set(int(x) for x in (i_frame_indices or [])))
            except (TypeError, ValueError):
                indices = []
            updated = False
            for v in self._videos:
                if v.get("source_path") == source_path:
                    v["i_frames"] = list(indices)
                    updated = True
            if not updated:
                self._pending_i_frames[source_path] = list(indices)

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
            self._pending_i_frames.clear()

    def __len__(self):
        """总向量数"""
        with self.acquire():
            return self._total_frames_count()

    def __getitem__(self, index: int) -> dict:
        """按 FAISS 索引获取 {source_path, frame_id, total_frames, video_fps, duration, i_frames}"""
        with self.acquire():
            return self._index_to_record(index)

    def __setitem__(self, index: int, item: dict):
        """不支持按索引修改，请通过 append 添加"""
        raise NotImplementedError("按视频聚合格式不支持按索引覆盖，请使用 append")

    def __delitem__(self, index: int):
        """不支持按索引删除"""
        raise NotImplementedError("按视频聚合格式不支持按索引删除")
