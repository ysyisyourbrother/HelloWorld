#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="加载 Video-MME test 集并按 task_type 汇总问题"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="local_datasets/Video-MME",
        help="Video-MME 数据集目录（支持传入 local_datasets 或 local_datasets/Video-MME）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="videomme_questions_by_task_type.json",
        help="输出 JSON 文件路径",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = (script_dir / dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError("数据集路径不存在: {}".format(dataset_path))

    # 与 benchmark.py 保持一致：使用 load_dataset + videomme 配置加载 test split
    load_path = dataset_path
    if not (load_path / "videomme.py").exists() and (load_path / "Video-MME").exists():
        load_path = load_path / "Video-MME"

    dataset = load_dataset(str(load_path), "videomme", split="test")

    grouped_questions = defaultdict(list)
    for sample in dataset:
        task_type = sample.get("task_type") or "Unknown"
        question = sample.get("question")
        if question:
            grouped_questions[task_type].append(question)

    output_data = dict(sorted(grouped_questions.items(), key=lambda item: item[0]))

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (script_dir / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(
        "已输出 {} 个任务类别，共 {} 条问题到 {}".format(
            len(output_data),
            sum(len(v) for v in output_data.values()),
            output_path,
        )
    )
