#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试StreamInput和FrameVectorizer的多进程交互
"""
import sys
import time
import multiprocessing as mp
from src.stream_input import StreamInput
from src.frame_vectorizer import FrameVectorizer
from src.config import Config

if __name__ == "__main__":
    config = Config()
    stream_input = StreamInput(config)
    frame_vectorizer = FrameVectorizer(config)
    frame_vectorizer.set_frame_queue(stream_input.frame_queue)

    stream_input.start()
    # stream_input.start_single_process()
    frame_vectorizer.start()

    time.sleep(20) # 测试2秒
    # frame_vectorizer.start()
