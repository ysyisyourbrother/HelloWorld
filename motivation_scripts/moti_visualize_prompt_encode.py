"""
可视化 BGE 最后一层每个 token 的编码与 pooling 后的最终输出。
在同一幅图中：各 token 编码用柱状图（带透明度重叠），pooling 输出用折线图。
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


def get_last_layer_and_pooled(
    text: str,
    model_path: str,
    device: str = "cuda",
    output_attentions: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], tuple | None]:
    """
    获取 BGE 最后一层每个 token 的编码和 pooling 后的最终输出。

    Args:
        text: 输入文本
        model_path: BGE 模型路径
        device: 运行设备
        output_attentions: 是否返回各层注意力权重（需使用 eager 实现）

    Returns:
        token_encodings: 各 token 的最后一层编码 shape (seq_len, embed_dim)
        pooled_output: pooling 后的输出 shape (embed_dim,)
        token_strs: 各位置 token 的字符串表示
        attentions: 各层注意力 (num_heads, seq_len, seq_len) 或 None
    """
    load_kwargs = {}
    if output_attentions:
        load_kwargs["attn_implementation"] = "eager"
    model = CLIPModel.from_pretrained(model_path, **load_kwargs).to(device)
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

    with torch.no_grad():
        text_outputs = model.text_model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs.get("attention_mask"),
            return_dict=True,
            output_attentions=output_attentions,
        )

    last_hidden_state = text_outputs.last_hidden_state  # [1, seq_len, embed_dim]
    pooled_output = text_outputs.pooler_output  # [1, embed_dim]

    token_encodings = last_hidden_state[0].cpu().numpy()  # [seq_len, embed_dim]
    pooled_output_np = pooled_output[0].cpu().numpy()  # [embed_dim]

    # 解码各 token 字符串
    tokenizer = processor.tokenizer
    token_strs = []
    for i in range(input_ids.shape[1]):
        token_id = input_ids[0, i].item()
        token_strs.append(tokenizer.decode([token_id]))

    attentions_out = None
    if output_attentions and text_outputs.attentions is not None:
        attentions_out = text_outputs.attentions

    return token_encodings, pooled_output_np, token_strs, attentions_out


def visualize_token_encode(
    text: str,
    model_path: str,
    output_path: str,
    device: str = "cuda",
    attn_layer: int | None = -1,
):
    """
    上中下三个子图：上图为各 token 编码与 pooling 输出；中图为各 token 与 EOT 的余弦相似度；
    下图为指定层 EOT 对各 token 的注意力权重（柱状图）。
    """
    need_attn = attn_layer is not None
    token_encodings, pooled_output, token_strs, attentions = get_last_layer_and_pooled(
        text=text,
        model_path=model_path,
        device=device,
        output_attentions=need_attn,
    )

    embed_dim = token_encodings.shape[1]
    num_tokens = token_encodings.shape[0]
    x = np.arange(embed_dim)

    num_subplots = 3 if need_attn else 2
    fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 4 * num_subplots), sharex=False)
    if num_subplots == 2:
        ax1, ax2 = axes
    else:
        ax1, ax2, ax3 = axes

    # ========== 上子图：token 编码 + pooling 折线 ==========
    bar_alpha = 0.6
    colors = plt.cm.rainbow(np.linspace(0, 1, num_tokens))
    for i in range(num_tokens):
        ax1.bar(x, token_encodings[i], alpha=bar_alpha, color=colors[i], width=0.9, label=token_strs[i])
    ax1.plot(x, pooled_output, color="red", linewidth=0.3, label="pooled output", zorder=10)
    ax1.set_ylim(-8, 8)
    ax1.set_xlabel("dimension id")
    ax1.set_ylabel("embedding value")
    ax1.set_title("Last layer token encodings (bars) vs pooled output (line)")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(loc="upper right")

    # ========== 中子图：各 token 与 EOT 的余弦相似度（排除 startoftext）==========
    plot_indices = [i for i in range(num_tokens) if "<|startoftext|>" not in token_strs[i]]
    plot_token_strs = [token_strs[i] for i in plot_indices]
    cos_sims = []
    for i in plot_indices:
        a, b = token_encodings[i], pooled_output
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        cos_sim = np.dot(a, b) / (norm_a * norm_b) if (norm_a > 0 and norm_b > 0) else 0.0
        cos_sims.append(cos_sim)

    n_plot = len(plot_indices)
    x_tokens = np.arange(n_plot)
    ax2.bar(x_tokens, cos_sims, color="steelblue", alpha=0.8)
    ax2.set_xticks(x_tokens)
    ax2.set_xticklabels(plot_token_strs, rotation=45, ha="right")
    ax2.set_ylabel("cosine similarity with EOT")
    ax2.set_title("Token–EOT cosine similarity")
    ax2.grid(axis="y", alpha=0.3)

    # ========== 下子图：指定层 EOT 对各 token 的注意力权重（排除 startoftext）==========
    if need_attn and attentions is not None:
        layer_idx = attn_layer if attn_layer >= 0 else len(attentions) + attn_layer
        # 需要 input_ids 来定位 EOT，从 get_last_layer_and_pooled 无法直接拿到，需在函数内获取
        # 简化：用 token_strs 找 EOT 位置（含 endoftext 的 token）
        eot_pos = None
        for i, s in enumerate(token_strs):
            if "<|endoftext|>" in s or "endoftext" in s.lower():
                eot_pos = i
                break
        if eot_pos is None:
            eot_pos = num_tokens - 1
        attn = attentions[layer_idx]
        eot_to_all = attn[0, :, eot_pos, :].cpu().numpy().mean(axis=0)
        attn_plot_indices = [i for i in range(num_tokens) if "<|startoftext|>" not in token_strs[i]]
        attn_plot_strs = [token_strs[i] for i in attn_plot_indices]
        attn_weights = [eot_to_all[i] for i in attn_plot_indices]
        n_attn = len(attn_plot_indices)
        x_attn = np.arange(n_attn)
        ax3.bar(x_attn, attn_weights, color="coral", alpha=0.8)
        ax3.set_xticks(x_attn)
        ax3.set_xticklabels(attn_plot_strs, rotation=45, ha="right")
        ax3.set_ylabel("attention weight")
        ax3.set_title(f"EOT→Token attention weights (layer {layer_idx})")
        ax3.grid(axis="y", alpha=0.3)

    fig.suptitle(f'Text: "{text}"', fontsize=10, y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"可视化已保存到: {output_path}")
    print(f"  - Token 数量: {num_tokens}")
    print(f"  - 嵌入维度: {embed_dim}")
    print(f"  - Tokens: {token_strs}")


def main():
    parser = argparse.ArgumentParser(
        description="可视化 BGE 最后一层 token 编码与 pooling 输出"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Instrument?",
        help="输入文本",
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
        default="motivation_results/text_encode",
        help="输出图片路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="运行设备",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="注意力权重的层索引，-1 表示最后一层；设为 None 可禁用注意力子图（需配合 --no_attn）",
    )
    parser.add_argument(
        "--no_attn",
        action="store_true",
        help="不绘制注意力权重子图",
    )
    args = parser.parse_args()

    config = Config(args.config)
    model_path = config.query_model_path

    attn_layer = None if args.no_attn else args.layer

    # 输出文件名以输入文本前若干字符命名
    safe_text = "".join(c if c.isalnum() or c in " -_" else "_" for c in args.text)[:30].strip("_")
    if not safe_text:
        safe_text = "text"
    layer_suffix = f"_attn_l{args.layer}" if attn_layer is not None else ""
    output_file = os.path.join(args.output_path, f"{safe_text}_last_layer_vs_pooled{layer_suffix}.png")

    visualize_token_encode(
        text=args.text,
        model_path=model_path,
        output_path=output_file,
        device=args.device,
        attn_layer=attn_layer,
    )


if __name__ == "__main__":
    main()
