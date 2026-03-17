"""
使用 BGE 模型计算指定图像与指定文本的向量相似度。
"""
import argparse
import os
import sys

import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import Config
from models.bge.modeling_MMRet_CLIP import CLIPModel


def compute_similarity(image_path: str, text: str, model_path: str, device: str = "cuda") -> float:
    """
    使用 BGE 模型对图像和文本编码，并计算余弦相似度。

    Args:
        image_path: 图像文件路径
        text: 文本内容
        model_path: BGE 模型路径
        device: 运行设备

    Returns:
        图像向量与文本向量的余弦相似度
    """
    model = CLIPModel.from_pretrained(model_path).to(device)
    model.set_processor(model_path)
    processor = model.processor
    model.eval()

    # 加载图像 (BGR -> RGB)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 图像编码
    pixel_values = processor(images=img_rgb, return_tensors="pt")["pixel_values"].to(device)
    # 文本编码
    text_inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    with torch.no_grad():
        image_embedding = model.encode_image(images=pixel_values)
        text_embedding = model.encode_text(text_inputs)

    # 余弦相似度 (向量已归一化时等价于内积)
    similarity = F.cosine_similarity(image_embedding, text_embedding, dim=1).item()
    return similarity


def main():
    parser = argparse.ArgumentParser(description="BGE 模型：计算图像与文本的向量相似度")
    parser.add_argument("-i", "--image", default="motivation_results/frames/1500/1500.png", help="图像路径")
    parser.add_argument("-t", "--text", default="instrument", help="文本内容")
    parser.add_argument("-c", "--config", default="configs/config_moti.json", help="配置文件（用于读取模型路径）")
    parser.add_argument("--model-path", default=None, help="BGE 模型路径（覆盖 config 中的配置）")
    parser.add_argument("--device", default=None, help="运行设备（默认从 config 读取）")
    args = parser.parse_args()

    config = Config(args.config)
    model_path = args.model_path or config.frame_model_path
    device = args.device or config.frame_device

    similarity = compute_similarity(
        image_path=args.image,
        text=args.text,
        model_path=model_path,
        device=device,
    )
    print(f"图像: {args.image}")
    print(f"文本: {args.text}")
    print(f"相似度: {similarity:.4f}")


if __name__ == "__main__":
    main()
