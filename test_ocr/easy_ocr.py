#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import os
import sys
import time


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    parser = argparse.ArgumentParser(description="EasyOCR text recognition (test script).")
    parser.add_argument(
        "--image_path",
        default=os.path.join(project_root, "test3.png"),
        help="图片路径（必须是存在的图片文件）。",
    )
    parser.add_argument(
        "--langs",
        default="ch_sim",
        help="语言列表，逗号分隔，例如 en,ch_sim 或 en,ko。",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.0,
        help="置信度阈值，低于该值的结果会被过滤（默认 0.0）。",
    )
    parser.add_argument(
        "--download_enabled",
        action="store_true",
        help="允许 EasyOCR 在首次运行时下载模型（默认不允许）。",
    )
    parser.add_argument(
        "--detector",
        default="craft",
        help="文本检测器，可选 craft 或 dbnet18（默认 dbnet18）。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if importlib.util.find_spec("easyocr") is None:
        sys.stderr.write(
            "未检测到依赖 easyocr。请先安装：pip install easyocr\n"
        )
        raise SystemExit(2)

    import easyocr  # noqa: E402
    import torch  # noqa: E402

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    if not langs:
        langs = ["en"]

    image_path = os.path.abspath(args.image_path)
    if os.path.isdir(image_path):
        sys.stderr.write("image_path 不能是目录：%s\n" % args.image_path)
        raise SystemExit(2)
    if not os.path.isfile(image_path):
        sys.stderr.write("image_path 不存在或不是文件：%s\n" % args.image_path)
        raise SystemExit(2)

    # 说明：model_storage_directory 用于缓存模型文件，避免每次重复下载/初始化。
    use_gpu = bool(torch.cuda.is_available())
    reader = easyocr.Reader(
        langs,
        gpu=use_gpu,
        download_enabled=bool(args.download_enabled),
        detect_network=args.detector,
    )

    print("==== image: %s" % image_path)
    t0 = time.time()
    results = reader.readtext(image_path)
    elapsed = time.time() - t0
    print("识别耗时: %.3f 秒" % elapsed)

    if not results:
        print("(no text)")
        raise SystemExit(0)

    kept = 0
    for i, item in enumerate(results, start=1):
        # item: (bbox, text, conf)
        text = item[1]
        conf = float(item[2])
        if conf < args.conf_threshold:
            continue
        kept += 1
        print("%03d\tconf=%.4f\t%s" % (i, conf, text))

    if kept == 0:
        print("(no text after conf filter)")

