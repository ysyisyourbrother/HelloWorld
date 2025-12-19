#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Reasoner 模块
验证 LLaVA 模型能否正确理解图片
"""
import sys
import time
import cv2
import os
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.reasoner import Reasoner
from src.config import Config


def test_reasoner_with_image(image_path: str, question: str):
    """
    测试 Reasoner 对单张图片的理解能力
    
    Args:
        image_path: 图片路径
        question: 要问的问题
    """
    print("=" * 80)
    print("测试 Reasoner 模块")
    print("=" * 80)
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"❌ 错误: 图片文件不存在: {image_path}")
        return
    
    print(f"\n📷 图片路径: {image_path}")
    print(f"❓ 问题: {question}")
    
    # 读取图片
    print("\n⏳ 正在读取图片...")
    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if frame is None:
        print(f"❌ 错误: 无法读取图片: {image_path}")
        return
    
    # 转换为RGB格式
    if frame.ndim == 2:  # 灰度图
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:  # BGR彩色图
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    print(f"✅ 图片读取成功，形状: {frame.shape}, 数据类型: {frame.dtype}")
    
    # 初始化配置和Reasoner
    print("\n⏳ 正在初始化 Reasoner（加载模型可能需要几分钟）...")
    config = Config()
    reasoner = Reasoner(config)
    
    # 启动Reasoner（会在子进程中加载模型）
    print("⏳ 启动 Reasoner 进程...")
    reasoner.start()
    
    # 等待模型加载完成（给一些时间让子进程初始化）
    print("⏳ 等待模型加载...")
    time.sleep(5)
    
    try:
        # 添加查询
        query_id = int(time.time() * 1000)
        print(f"\n📤 发送查询 (ID: {query_id})...")
        
        reasoner.add_query(
            query_text=question,
            memory_results=[frame],
            query_id=query_id
        )
        
        # 等待结果
        print("⏳ 等待推理结果（这可能需要几秒钟）...")
        result = reasoner.result_queue.get(timeout=60)  # 60秒超时
        
        # 打印结果
        print("\n" + "=" * 80)
        print("📊 推理结果")
        print("=" * 80)
        
        if result.error:
            print(f"❌ 错误: {result.error}")
        else:
            print(f"✅ 查询ID: {result.query_id}")
            print(f"✅ 回答: {result.result}")
            print(f"✅ 时间戳: {result.timestamp}")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 停止Reasoner
        print("\n⏳ 正在停止 Reasoner...")
        reasoner.stop()
        print("✅ Reasoner 已停止")


def test_multiple_questions():
    """
    测试多个问题
    """
    image_path = "test_frame.png"
    
    questions = [
        "Hello, please describe this image.",
        "What objects can you see in this image?",
        "What colors are dominant in this image?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n\n{'#' * 80}")
        print(f"测试 {i}/{len(questions)}")
        print(f"{'#' * 80}\n")
        test_reasoner_with_image(image_path, question)
        
        if i < len(questions):
            print("\n⏳ 等待 5 秒后继续下一个测试...\n")
            time.sleep(5)


if __name__ == "__main__":
    # 默认测试图片和问题
    default_image = "test_frame.png"
    default_question = "Hello, please describe this image."
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        question = sys.argv[2] if len(sys.argv) > 2 else default_question
    else:
        image_path = default_image
        question = default_question
    
    # 运行测试
    try:
        test_reasoner_with_image(image_path, question)
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ 测试完成！")
