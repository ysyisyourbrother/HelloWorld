# -*- coding: utf-8 -*-
"""从单次 benchmark 结果 JSON 中汇总答错与无法解析为选项的题目。"""

import argparse
import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.benchmark.videomme import extract_characters_regex


def _response_as_str(question):
    r = question.get("response")
    if r is None:
        return ""
    if isinstance(r, str):
        return r
    return str(r)


def collect_incorrect_by_video(benchmark_result_path):
    """
    读取 benchmark 结果文件，按 video_id 汇总答错（解析出选项但与标答不一致）
    与空白（无法从 response 中解析出 A/B/C/D，与 eval 逻辑一致）。

    返回 dict: { video_id: { "wrong": [...], "blank": [...] } }，仅包含有问题的视频；
    子键仅在对应列表非空时出现。
    """
    benchmark_result_path = os.path.abspath(benchmark_result_path)
    with open(benchmark_result_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    out = {}
    for item in rows:
        video_id = item["video_id"]
        wrong_ids = []
        blank_ids = []
        for q in item.get("questions", []):
            qid = q["question_id"]
            gt = q.get("answer", "")
            raw = _response_as_str(q)
            pred = extract_characters_regex(raw) if raw else ""
            if pred == "":
                blank_ids.append(qid)
            elif pred != gt:
                wrong_ids.append(qid)

        entry = {}
        if wrong_ids:
            entry["wrong"] = wrong_ids
        if blank_ids:
            entry["blank"] = blank_ids
        if entry:
            out[video_id] = entry
    return out


def incorrect_report_path(benchmark_result_path):
    """与结果同目录，文件名为 incorrect_ + 原文件名。"""
    benchmark_result_path = os.path.abspath(benchmark_result_path)
    d = os.path.dirname(benchmark_result_path)
    base = os.path.basename(benchmark_result_path)
    return os.path.join(d, "incorrect_" + base)


def parse_args():
    p = argparse.ArgumentParser(description="从 benchmark 结果 JSON 生成 incorrect_*.json")
    p.add_argument(
        "benchmark_json",
        type=str,
        help="单次 benchmark 结果文件路径，例如 benchmark_results/.../benchmark_*.json",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="输出路径（默认：同目录 incorrect_原文件名）",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = collect_incorrect_by_video(args.benchmark_json)
    out_path = args.output if args.output else incorrect_report_path(args.benchmark_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(out_path)
