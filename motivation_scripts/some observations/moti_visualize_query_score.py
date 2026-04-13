import cv2
import argparse
import json
import numpy as np
import os
import sys
import torch
from typing import Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import Config
from models.bge.modeling_MMRet_CLIP import CLIPModel


def _hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    """#RRGGBB -> (B, G, R)"""
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return (b, g, r)


def _get_color_from_colormap(
    colormap: Union[int, dict],
    v_norm: float,
) -> Tuple[int, int, int]:
    """
    根据归一化值 v_norm in [0, 1] 获取 BGR 颜色。

    colormap 支持两种形式:
    - int: cv2 预定义 colormap，如 cv2.COLORMAP_JET
    - dict: 自定义分段，如 {"0": "#0000FF", "0.5": "#00FF00", "1": "#FF0000"}
            key 为归一化位置 (0~1)，value 为 hex 颜色，中间线性插值
    """
    v_norm = max(0.0, min(1.0, v_norm))
    if isinstance(colormap, dict):
        stops = sorted([(float(k), _hex_to_bgr(v)) for k, v in colormap.items()])
        positions = [s[0] for s in stops]
        colors_bgr = [s[1] for s in stops]
        if v_norm <= positions[0]:
            return colors_bgr[0]
        if v_norm >= positions[-1]:
            return colors_bgr[-1]
        for idx in range(len(positions) - 1):
            p0, p1 = positions[idx], positions[idx + 1]
            if p0 <= v_norm <= p1:
                t = (v_norm - p0) / (p1 - p0) if p1 > p0 else 0
                c0, c1 = colors_bgr[idx], colors_bgr[idx + 1]
                return tuple(int(c0[i] + t * (c1[i] - c0[i])) for i in range(3))
        return colors_bgr[-1]
    else:
        v_uint8 = int(v_norm * 255) & 0xFF
        return tuple(cv2.applyColorMap(np.array([[v_uint8]], dtype=np.uint8), colormap)[0, 0])


def _load_bge_patch_params(config_path: str) -> Tuple[int, int]:
    """从 BGE 模型配置中读取 image_size 和 patch_size"""
    config = Config(config_path)
    model_path = config.frame_model_path
    vision_config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(vision_config_path):
        return 224, 16
    with open(vision_config_path, "r") as f:
        cfg = json.load(f)
    vc = cfg.get("vision_config", cfg)
    return (
        vc.get("image_size", 224),
        vc.get("patch_size", 16),
    )


def visualize_patch_grid(
    image_path: str,
    output_path: str,
    image_size: int = 224,
    patch_size: int = 16,
    line_color: tuple = (255, 255, 255),
    line_thickness: int = 1,
):
    """
    根据 BGE 模型的 patchify 方式，在图像上绘制 patch 分隔线并保存。
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        image_size: 模型输入尺寸 (BGE 默认 224)
        patch_size: patch 大小 (BGE 默认 16)
        line_color: 分隔线颜色，BGR 格式，默认白色
        line_thickness: 线宽
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    
    # 缩放到模型输入尺寸，与 processor 一致
    img_resized = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    
    # 在 patch 边界画线
    # 垂直线: x = patch_size, 2*patch_size, ..., (num_patches_per_side-1)*patch_size
    # 水平线: y = 同上
    num_patches_per_side = image_size // patch_size
    
    for i in range(1, num_patches_per_side):
        x = i * patch_size
        y = i * patch_size
        # 垂直线
        cv2.line(img_resized, (x, 0), (x, image_size), line_color, line_thickness)
        # 水平线
        cv2.line(img_resized, (0, y), (image_size, y), line_color, line_thickness)
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img_resized)
    print(f"Patch 可视化已保存到: {output_path}")
    print(f"  - 图像尺寸: {image_size}x{image_size}")
    print(f"  - Patch 大小: {patch_size}x{patch_size}")
    print(f"  - Patch 数量: {num_patches_per_side}x{num_patches_per_side} = {num_patches_per_side**2}")

def get_query_patch_similarity(
    image_path: str,
    model_path: str,
    query_text: str,
    device: str = "cuda",
    image_size: int = 224,
    patch_size: int = 16,
) -> Tuple[np.ndarray, float]:
    """
    计算特定文本与各视觉 patch 的余弦相似度。
    视觉 token（含 CLS）先经 post_layernorm 和 visual_projection，再与文本向量计算相似度。

    Returns:
        patch_scores: shape (num_patches_per_side, num_patches_per_side)，即 14x14，用于可视化
        cls_score: CLS token 与文本的相似度，仅用于打印
    """
    model = CLIPModel.from_pretrained(model_path).to(device)
    model.set_processor(model_path)
    model.eval()
    processor = model.processor

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pixel_values = processor(images=img_rgb, return_tensors="pt")["pixel_values"].to(device)
    text_inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        last_hidden = vision_outputs.last_hidden_state  # [1, 197, 768]

        # 所有视觉 token 经 post_layernorm
        post_ln = model.vision_model.post_layernorm
        tokens_after_ln = post_ln(last_hidden)  # [1, 197, 768]

        # 经 visual_projection
        tokens_proj = model.visual_projection(tokens_after_ln)  # [1, 197, proj_dim]

        cls_token_proj = tokens_proj[:, 0, :]  # [1, proj_dim]
        patch_tokens_proj = tokens_proj[:, 1:, :]  # [1, 196, proj_dim]

        # 文本特征（已含 text_projection）
        text_features = model.get_text_features(**text_inputs)  # [1, proj_dim]

        # 归一化后计算余弦相似度
        text_norm = torch.nn.functional.normalize(text_features, dim=-1)
        patch_norm = torch.nn.functional.normalize(patch_tokens_proj, dim=-1)
        cls_norm = torch.nn.functional.normalize(cls_token_proj, dim=-1)

        patch_similarity = (text_norm @ patch_norm.squeeze(0).T).squeeze(0).cpu().numpy()  # [196]
        cls_similarity = (text_norm @ cls_norm.T).squeeze().item()

    num_per_side = image_size // patch_size
    patch_scores = patch_similarity.reshape(num_per_side, num_per_side)
    return patch_scores, cls_similarity


def visualize_cls_patch_scores_on_original(
    image_path: str,
    output_path: str,
    scores: np.ndarray,
    image_size: int = 224,
    patch_size: int = 16,
    line_color: tuple = (255, 255, 255),
    line_thickness: int = 2,
    colormap: Union[int, dict] = cv2.COLORMAP_JET,
):
    """
    以与 visualize_patch_grid_on_original 相同的画法，将 patch 填充为冷暖色热力图。
    分数越低越冷（蓝），越高越暖（红），白线分隔 patch。

    colormap: int 为 cv2 预定义，dict 为自定义如 {"0":"#0000FF","0.5":"#00FF00","1":"#FF0000"}
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    h, w = img.shape[:2]
    num_patches_per_side = image_size // patch_size

    scale_x = w / image_size
    scale_y = h / image_size

    # 创建空白画布，与原图同尺寸
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # 将 scores 归一化到 [0, 1]
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min > 1e-8:
        scores_norm_01 = (scores - s_min) / (s_max - s_min)
    else:
        scores_norm_01 = np.full_like(scores, 0.5, dtype=np.float64)

    for i in range(num_patches_per_side):
        for j in range(num_patches_per_side):
            x1 = int(j * patch_size * scale_x)
            x2 = int((j + 1) * patch_size * scale_x)
            y1 = int(i * patch_size * scale_y)
            y2 = int((i + 1) * patch_size * scale_y)
            x2 = min(x2, w)
            y2 = min(y2, h)
            color_bgr = _get_color_from_colormap(colormap, float(scores_norm_01[i, j]))
            canvas[y1:y2, x1:x2] = color_bgr

    # 画白线分隔 patch
    for k in range(1, num_patches_per_side):
        x_224 = k * patch_size
        y_224 = k * patch_size
        x_orig = int(x_224 * scale_x)
        y_orig = int(y_224 * scale_y)
        cv2.line(canvas, (x_orig, 0), (x_orig, h), line_color, line_thickness)
        cv2.line(canvas, (0, y_orig), (w, y_orig), line_color, line_thickness)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, canvas)
    print(f"Query-patch 相似度热力图已保存到: {output_path}")
    print(f"  - 分数范围: [{s_min:.4f}, {s_max:.4f}]")


def visualize_cls_patch_scores_with_labels(
    image_path: str,
    output_path: str,
    scores: np.ndarray,
    image_size: int = 224,
    patch_size: int = 16,
    line_color: tuple = (255, 255, 255),
    line_thickness: int = 2,
    colormap: Union[int, dict] = cv2.COLORMAP_JET,
    text_color: tuple = (255, 255, 255),
    font_scale: float = 0.35,
    font_thickness: int = 1,
):
    """
    在 visualize_cls_patch_scores_on_original 基础上，在每个 patch 中央标注相似度分数（保留三位小数）。
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    h, w = img.shape[:2]
    num_patches_per_side = image_size // patch_size

    scale_x = w / image_size
    scale_y = h / image_size

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min > 1e-8:
        scores_norm_01 = (scores - s_min) / (s_max - s_min)
    else:
        scores_norm_01 = np.full_like(scores, 0.5, dtype=np.float64)

    for i in range(num_patches_per_side):
        for j in range(num_patches_per_side):
            x1 = int(j * patch_size * scale_x)
            x2 = int((j + 1) * patch_size * scale_x)
            y1 = int(i * patch_size * scale_y)
            y2 = int((i + 1) * patch_size * scale_y)
            x2 = min(x2, w)
            y2 = min(y2, h)
            color_bgr = _get_color_from_colormap(colormap, float(scores_norm_01[i, j]))
            canvas[y1:y2, x1:x2] = color_bgr

    # 画白线分隔 patch
    for k in range(1, num_patches_per_side):
        x_224 = k * patch_size
        y_224 = k * patch_size
        x_orig = int(x_224 * scale_x)
        y_orig = int(y_224 * scale_y)
        cv2.line(canvas, (x_orig, 0), (x_orig, h), line_color, line_thickness)
        cv2.line(canvas, (0, y_orig), (w, y_orig), line_color, line_thickness)

    # 在每个 patch 中央标注分数（保留三位小数）
    for i in range(num_patches_per_side):
        for j in range(num_patches_per_side):
            x1 = int(j * patch_size * scale_x)
            x2 = int((j + 1) * patch_size * scale_x)
            y1 = int(i * patch_size * scale_y)
            y2 = int((i + 1) * patch_size * scale_y)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            text = f"{scores[i, j]:.3f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            tx = cx - tw // 2
            ty = cy + th // 2
            cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, canvas)
    print(f"Query-patch 相似度热力图（含分数标注）已保存到: {output_path}")
    print(f"  - 分数范围: [{s_min:.4f}, {s_max:.4f}]")


def visualize_cls_patch_scores_overlay(
    image_path: str,
    output_path: str,
    scores: np.ndarray,
    image_size: int = 224,
    patch_size: int = 16,
    alpha: float = 0.4,
    line_color: tuple = (255, 255, 255),
    line_thickness: int = 2,
    colormap: Union[int, dict] = cv2.COLORMAP_JET,
):
    """
    将热力图以半透明方式叠加在划了白线的原图上。
    原图作为底图，热力图透明度由 alpha 控制（0=完全透明只看原图，1=完全不透明只看热力图）。

    colormap: int 为 cv2 预定义，dict 为自定义如 {"0":"#0000FF","0.5":"#00FF00","1":"#FF0000"}
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    h, w = img.shape[:2]
    num_patches_per_side = image_size // patch_size

    scale_x = w / image_size
    scale_y = h / image_size

    # 构建热力图（与原图同尺寸）
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min > 1e-8:
        scores_norm_01 = (scores - s_min) / (s_max - s_min)
    else:
        scores_norm_01 = np.full_like(scores, 0.5, dtype=np.float64)

    for i in range(num_patches_per_side):
        for j in range(num_patches_per_side):
            x1 = int(j * patch_size * scale_x)
            x2 = int((j + 1) * patch_size * scale_x)
            y1 = int(i * patch_size * scale_y)
            y2 = int((i + 1) * patch_size * scale_y)
            x2 = min(x2, w)
            y2 = min(y2, h)
            color_bgr = _get_color_from_colormap(colormap, float(scores_norm_01[i, j]))
            heatmap[y1:y2, x1:x2] = color_bgr

    # 底图：原图 + 白线
    base = img.copy()
    for k in range(1, num_patches_per_side):
        x_224 = k * patch_size
        y_224 = k * patch_size
        x_orig = int(x_224 * scale_x)
        y_orig = int(y_224 * scale_y)
        cv2.line(base, (x_orig, 0), (x_orig, h), line_color, line_thickness)
        cv2.line(base, (0, y_orig), (w, y_orig), line_color, line_thickness)

    # 叠加：result = alpha * heatmap + (1 - alpha) * base
    blended = cv2.addWeighted(heatmap, alpha, base, 1.0 - alpha, 0)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, blended)
    print(f"Query 热力图叠加图已保存到: {output_path} (alpha={alpha})")


def visualize_patch_grid_on_original(
    image_path: str,
    output_path: str,
    image_size: int = 224,
    patch_size: int = 16,
    line_color: tuple = (255, 255, 255),
    line_thickness: int = 2,
):
    """在原始尺寸图像上绘制 patch 网格（按比例缩放）"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    
    h, w = img.shape[:2]
    num_patches_per_side = image_size // patch_size
    
    # 将 patch 边界从 224x224 空间映射到原图
    scale_x = w / image_size
    scale_y = h / image_size
    
    for i in range(1, num_patches_per_side):
        x_224 = i * patch_size
        y_224 = i * patch_size
        x_orig = int(x_224 * scale_x)
        y_orig = int(y_224 * scale_y)
        cv2.line(img, (x_orig, 0), (x_orig, h), line_color, line_thickness)
        cv2.line(img, (0, y_orig), (w, y_orig), line_color, line_thickness)
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Patch 可视化（原图尺寸）已保存到: {output_path}")

def _sanitize_query_for_filename(q: str, max_len: int = 25) -> str:
    """将 query 转为安全的文件名片段"""
    s = "".join(c if c.isalnum() or c in " -_" else "_" for c in q)
    s = "_".join(s.split())[:max_len].strip("_")
    return s or "query"


def main():
    parser = argparse.ArgumentParser(description="BGE patch 网格与 文本-patch 相似度可视化")
    parser.add_argument("-i", "--input", default="motivation_results/frames/1500/1500.png", help="输入图片路径")
    # parser.add_argument("-q", "--query", default="Saxophone", help="查询文本，用于与视觉 token 计算相似度")
    # parser.add_argument("-q", "--query", default="Which instrument is the performer on the stage holding in the video?", help="查询文本，用于与视觉 token 计算相似度")
    parser.add_argument("-q", "--query", default="A person holding a saxophone", help="查询文本，用于与视觉 token 计算相似度")
    # parser.add_argument("-q", "--query", default="A person with a blue shirt holding a saxophone", help="查询文本，用于与视觉 token 计算相似度")

    
    parser.add_argument("--overlay_alpha", type=float, default=0.4, help="叠加时热力图不透明度 (默认 0.4)")
    parser.add_argument("--no_patch_grid", action="store_true", help="不生成 patch 网格图，仅生成 query 热力图")
    parser.add_argument("-c", "--config", default="configs/config_moti.json", help="配置文件")
    parser.add_argument("--device", default=None, help="device (默认从 config 读取)")
    parser.add_argument("--image_size", type=int, default=None, help="模型输入尺寸 (默认从 config 读取)")
    parser.add_argument("--patch_size", type=int, default=None, help="Patch 大小 (默认从 config 读取)")
    args = parser.parse_args()

    config = Config(args.config)
    image_size = args.image_size
    patch_size = args.patch_size
    if image_size is None or patch_size is None:
        isz, psz = _load_bge_patch_params(args.config)
        image_size = image_size if image_size is not None else isz
        patch_size = patch_size if patch_size is not None else psz
    device = args.device or config.frame_device
    model_path = config.frame_model_path

    # 所有输出路径基于输入路径 -i 和 query 自动生成
    base, ext = os.path.splitext(args.input)
    q_suffix = _sanitize_query_for_filename(args.query)
    output = f"{base}_patch_grid.jpg"
    output_original = f"{base}_patch_grid_original_size.jpg"
    output_query = f"{base}_query_{q_suffix}_score.jpg"
    output_overlay = f"{base}_query_{q_suffix}_overlay.jpg"
    output_labels = f"{base}_query_{q_suffix}_labels.jpg"

    # 自定义 colormap，可改为 dict 如 {"0":"#0000FF","0.5":"#00FF00","1":"#FF0000"}
    colormap: Union[int, dict] = cv2.COLORMAP_JET
    colormap = {"0": "#18354E", "0.80": "#B1CEE7", "1": "#CC0300"}

    if not args.no_patch_grid:
        visualize_patch_grid(
            image_path=args.input,
            output_path=output,
            image_size=image_size,
            patch_size=patch_size,
        )
        visualize_patch_grid_on_original(
            image_path=args.input,
            output_path=output_original,
            image_size=image_size,
            patch_size=patch_size,
        )

    patch_scores, cls_score = get_query_patch_similarity(
        image_path=args.input,
        model_path=model_path,
        query_text=args.query,
        device=device,
        image_size=image_size,
        patch_size=patch_size,
    )
    print(f"CLS token 与文本 \"{args.query}\" 的相似度: {cls_score:.4f}")

    visualize_cls_patch_scores_on_original(
        image_path=args.input,
        output_path=output_query,
        scores=patch_scores,
        image_size=image_size,
        patch_size=patch_size,
        colormap=colormap,
    )
    visualize_cls_patch_scores_with_labels(
        image_path=args.input,
        output_path=output_labels,
        scores=patch_scores,
        image_size=image_size,
        patch_size=patch_size,
        colormap=colormap,
    )
    visualize_cls_patch_scores_overlay(
        image_path=args.input,
        output_path=output_overlay,
        scores=patch_scores,
        image_size=image_size,
        patch_size=patch_size,
        alpha=args.overlay_alpha,
        colormap=colormap,
    )

if __name__ == "__main__":
    main()