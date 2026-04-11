# -*- coding: utf-8 -*-
"""
与在线 StreamVideoInput 一致的「按窗触发 + 按包长累积抽样」纯策略（无 GStreamer 依赖）。

离线分窗使用每帧媒体时间（秒），由 ffprobe 的 pkt_pts_time / best_effort_timestamp_time 等得到。
"""

from __future__ import absolute_import, division, print_function

from typing import Any, Dict, Iterator, List, Optional, Tuple


def cumulative_sample_indices_by_pkt_sizes(pkt_sizes, k):
    # type: (List[int], int) -> List[int]
    """
    在窗口内按包大小前缀和，在总大小的 (k+1) 等分点处各取一帧（与累积到 S*i/(k+1) 对齐）。
    """
    if k <= 0 or not pkt_sizes:
        return []
    total = float(sum(pkt_sizes))
    if total <= 0:
        return []
    thresholds = [total * float(i) / float(k + 1) for i in range(1, k + 1)]
    cum = 0.0
    out = []
    ti = 0
    for j, p in enumerate(pkt_sizes):
        cum += float(p)
        while ti < len(thresholds) and cum >= thresholds[ti]:
            out.append(j)
            ti += 1
        if ti >= len(thresholds):
            break
    return out


def select_frames_for_window(
    window_rows,
    target_decode_fps,
    window_duration_sec,
    baseline_non_i,
    trigger_ratio,
    warmup_windows,
    windows_seen,
    alpha_baseline,
):
    # type: (List[Dict[str, Any]], float, float, Optional[float], float, int, int, float) -> Tuple[List[int], Optional[float], bool]
    """
    根据窗口内帧元数据决定是否触发，并返回应编码的帧在 window_rows 中的下标列表。

    Returns:
        (selected_indices_in_window, new_baseline_non_i, triggered)
    """
    if not window_rows:
        return [], baseline_non_i, False

    pkt_sizes = [int(r["pkt_size"]) for r in window_rows]
    is_key = [bool(r["is_keyframe"]) for r in window_rows]

    non_i_pkts = [pkt_sizes[i] for i in range(len(pkt_sizes)) if not is_key[i]]
    if not non_i_pkts:
        non_i_mean = 0.0
    else:
        non_i_mean = float(sum(non_i_pkts)) / float(len(non_i_pkts))

    triggered = False
    new_baseline = baseline_non_i

    # windows_seen 为已结束的窗口数（从 1 起）；前 warmup_windows 个窗口仅更新基线、不触发
    if windows_seen <= warmup_windows:
        if non_i_mean > 0:
            if new_baseline is None:
                new_baseline = non_i_mean
            else:
                new_baseline = alpha_baseline * non_i_mean + (1.0 - alpha_baseline) * new_baseline
        return [], new_baseline, False

    if new_baseline is not None and new_baseline > 0 and non_i_mean > new_baseline * trigger_ratio:
        triggered = True

    if not triggered:
        if non_i_mean > 0:
            if new_baseline is None:
                new_baseline = non_i_mean
            else:
                new_baseline = alpha_baseline * non_i_mean + (1.0 - alpha_baseline) * new_baseline
        return [], new_baseline, False

    k = int(target_decode_fps * window_duration_sec)
    if k < 1:
        k = 1

    cum_idx = cumulative_sample_indices_by_pkt_sizes(pkt_sizes, k)
    key_idx = [i for i in range(len(window_rows)) if is_key[i]]
    merged = sorted(set(cum_idx) | set(key_idx))
    return merged, new_baseline, True


def iter_window_index_groups_by_media_time(
    media_times,  # type: List[float]
    window_duration_sec,  # type: float
    frames_per_window_fallback=None,  # type: Optional[int]
):
    # type: (...) -> Iterator[List[int]]
    """
    按媒体时间将全局帧索引 0..N-1 分组为若干窗口，供离线复刻 StreamVideoInput 策略。

    以首帧时间 t_ref = media_times[0] 为锚点，第 k 个时间窗覆盖
    [t_ref + k*D, t_ref + (k+1)*D)（D = window_duration_sec），窗口内帧按解码/显示顺序归入该窗。

    若相邻帧的窗号从 k 跳到 k+m（m>1），中间会产出 m-1 个空列表，与「墙钟分窗但某秒无帧」时
    仍推进 windows_seen 的语义一致。

    若时间戳退化（几乎全相同）且提供 frames_per_window_fallback（>=1），则按固定帧数切块，
    避免单窗吞掉整段视频。

    Yields:
        每个元素为该窗内的全局帧索引列表（可能为空）。
    """
    n = len(media_times)
    if n == 0 or window_duration_sec <= 0:
        return

    span = 0.0
    if n > 1:
        span = float(max(media_times) - min(media_times))
    degenerate = span < 1e-9
    if degenerate and frames_per_window_fallback is not None and frames_per_window_fallback >= 1:
        step = int(frames_per_window_fallback)
        for start in range(0, n, step):
            yield list(range(start, min(start + step, n)))
        return

    t_ref = float(media_times[0])

    def wid_of(i):
        w = int((float(media_times[i]) - t_ref) / window_duration_sec)
        return w if w >= 0 else 0

    cur_wid = wid_of(0)
    cur_indices = [0]
    for i in range(1, n):
        w = wid_of(i)
        if w == cur_wid:
            cur_indices.append(i)
        else:
            yield cur_indices
            for _ in range(cur_wid + 1, w):
                yield []
            cur_wid = w
            cur_indices = [i]
    yield cur_indices
