import multiprocessing as mp
import numpy as np
import time
import torch
# 本项目
from src.config import Config
from src.query_vectorizer import TextBGEVectorizer


if __name__ == "__main__":
    config = Config()
    vectorizer = TextBGEVectorizer(config)
    query = "你好, 我是一个学生"
    vector = vectorizer.encode(query)
    print(vector.shape)