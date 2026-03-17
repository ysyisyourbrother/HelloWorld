"""
可视化 BGE 推理前，指定 token 的位置编码和语义编码。
在同一幅 figure 中分为上下两个子图展示。
"""
import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# 抑制中文字体缺失时的警告（若系统无中文字体，标签可能显示为方框）
warnings.filterwarnings("ignore", message=".*missing from font.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import Config
from models.bge.modeling_MMRet_CLIP import CLIPModel


def get_token_encodings(
    text: str,
    token_idx: int,
    model_path: str,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    获取 BGE 推理前，指定位置 token 的位置编码和语义编码。

    Args:
        text: 输入文本
        token_idx: token 位置索引（0-based）
        model_path: BGE 模型路径
        device: 运行设备

    Returns:
        position_enc: 位置编码向量 shape (embed_dim,)
        token_enc: 语义编码向量 shape (embed_dim,)
        token_str: 该位置 token 的字符串表示
    """
    model = CLIPModel.from_pretrained(model_path).to(device)
    model.set_processor(model_path)
    model.eval()
    processor = model.processor

    text_inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    input_ids = text_inputs["input_ids"]  # [1, seq_len]

    embeddings_module = model.text_model.embeddings
    seq_length = input_ids.shape[-1]

    with torch.no_grad():
        # 语义编码: token_embedding(input_ids)
        token_embeds = embeddings_module.token_embedding(input_ids)  # [1, seq_len, embed_dim]
        # 位置编码: position_embedding(position_ids)
        position_ids = embeddings_module.position_ids[:, :seq_length]
        position_embeds = embeddings_module.position_embedding(position_ids)  # [1, seq_len, embed_dim]

    if token_idx >= seq_length:
        raise ValueError(f"token_idx={token_idx} 超出序列长度 {seq_length}")

    position_enc = position_embeds[0, token_idx].cpu().numpy()
    token_enc = token_embeds[0, token_idx].cpu().numpy()

    # 解码 token 字符串
    tokenizer = processor.tokenizer
    token_id = input_ids[0, token_idx].item()
    token_str = tokenizer.decode([token_id])

    return position_enc, token_enc, token_str


def visualize_token_encodings(
    text: str,
    token_idx: int,
    model_path: str,
    output_path: str,
    device: str = "cuda",
):
    """
    在同一幅 figure 中，上下两个子图分别展示指定 token 的位置编码和语义编码。
    """
    position_enc, token_enc, token_str = get_token_encodings(
        text=text,
        token_idx=token_idx,
        model_path=model_path,
        device=device,
    )

    embed_dim = len(position_enc)
    x = np.arange(embed_dim)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # 上子图: 位置编码
    ax1.bar(x, position_enc, color="steelblue", alpha=0.8)
    ax1.set_ylabel("position embedding")
    ax1.set_title(f"Position {token_idx}")
    ax1.grid(axis="y", alpha=0.3)

    # 下子图: 语义编码
    ax2.bar(x, token_enc, color="coral", alpha=0.8)
    ax2.set_xlabel("dimension id")
    ax2.set_ylabel("semantics embedding")
    ax2.set_title(f"Semantics (token: {repr(token_str)}, pos={token_idx})")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f'Text: "{text}"', fontsize=10, y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_path, f"{token_str}_{token_idx}.png")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"可视化已保存到: {output_path}")
    print(f"  - Token 位置: {token_idx}")
    print(f"  - Token 字符串: {repr(token_str)}")
    print(f"  - 嵌入维度: {embed_dim}")


def main():
    parser = argparse.ArgumentParser(
        description="可视化 BGE 推理前指定 token 的位置编码和语义编码"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="A saxophone",
        help="输入文本",
    )
    parser.add_argument(
        "--token-idx",
        type=int,
        default=2,
        help="要可视化的 token 位置索引 (0-based)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_moti.json",
        help="配置文件路径",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="motivation_results/text_embedding",
        help="输出图片路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="运行设备",
    )
    args = parser.parse_args()

    config = Config(args.config)
    model_path = config.query_model_path

    visualize_token_encodings(
        text=args.text,
        token_idx=args.token_idx,
        model_path=model_path,
        output_path=args.output_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
