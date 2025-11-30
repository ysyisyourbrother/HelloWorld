import multiprocessing as mp
import numpy as np
import time
import torch
import cv2
# 本项目
from src.config import Config
from src.frame_vectorizer import ImageBGEVectorizer


if __name__ == "__main__":
    config = Config()
    vectorizer = ImageBGEVectorizer(config)
    frame = cv2.imread("test_frame.png")
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    vector = vectorizer.encode(frame)
    print(vector.shape)