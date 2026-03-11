import cv2
import argparse
import json
import numpy as np
import os
import sys
import torch
from typing import Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import Config
from models.bge.modeling_MMRet_CLIP import CLIPModel


def _hex_to_bgr(hex_str: str) -> tuple[int, int, int]:
    """#RRGGBB -> (B, G, R)"""
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return (b, g, r)


def _get_color_from_colormap(
    colormap: Union[int, dict],
    v_norm: float,
) -> tuple[int, int, int]:
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


def _load_bge_patch_params(config_path: str) -> tuple[int, int]:
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

def get_cls_patch_similarity(
    image_path: str,
    model_path: str,
    device: str = "cuda",
    image_size: int = 224,
    patch_size: int = 16,
) -> np.ndarray:
    """
    计算 BGE ViT 中 CLS token 与各 patch 的余弦相似度。
    
    Returns:
        scores: shape (num_patches_per_side, num_patches_per_side)，即 14x14
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

    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
    last_hidden = vision_outputs.last_hidden_state  # [1, 197, 768]
    cls_token = last_hidden[:, 0, :]  # [1, 768]
    patch_tokens = last_hidden[:, 1:, :]  # [1, 196, 768]

    cls_norm = torch.nn.functional.normalize(cls_token, dim=-1)
    patch_norm = torch.nn.functional.normalize(patch_tokens, dim=-1)
    similarity = (cls_norm @ patch_norm.squeeze(0).T).squeeze(0).cpu().numpy()  # [196]

    num_per_side = image_size // patch_size
    scores = similarity.reshape(num_per_side, num_per_side)
    return scores


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
    print(f"CLS-patch 相似度热力图已保存到: {output_path}")
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
    print(f"CLS-patch 相似度热力图（含分数标注）已保存到: {output_path}")
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
    print(f"CLS 热力图叠加图已保存到: {output_path} (alpha={alpha})")


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

def main():
    parser = argparse.ArgumentParser(description="BGE patch 网格与 CLS-patch 相似度可视化")
    parser.add_argument("-i", "--input", default="motivation_results/frames/frame_selected.jpg", help="输入图片路径")
    parser.add_argument("-o", "--output", default="motivation_results/frames/frame_patch_grid.jpg", help="patch 网格输出路径 (224x224)")
    parser.add_argument(
        "-o2", "--output_original",
        default=None,
        help="patch 网格原图尺寸输出路径 (默认: 在 -o 基础上加 _original_size)"
    )
    parser.add_argument(
        "--output_cls",
        default=None,
        help="CLS-patch 相似度热力图输出路径 (默认: 在 -o 基础上加 _CLS_score)"
    )
    parser.add_argument(
        "--output_overlay",
        default=None,
        help="热力图叠加在原图上的输出路径 (默认: 在 -o 基础上加 _CLS_overlay)"
    )
    parser.add_argument(
        "--output_labels",
        default=None,
        help="带分数标注的热力图输出路径 (默认: 在 -o 基础上加 _CLS_labels)"
    )
    parser.add_argument("--overlay_alpha", type=float, default=0.4, help="叠加时热力图不透明度 (默认 0.4)")
    parser.add_argument("--no_patch_grid", action="store_true", help="不生成 patch 网格图，仅生成 CLS 热力图")
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

    output_original = args.output_original
    if output_original is None:
        base, ext = os.path.splitext(args.output)
        output_original = f"{base}_original_size{ext}"

    output_cls = args.output_cls
    if output_cls is None:
        base, ext = os.path.splitext(args.output)
        output_cls = f"{base}_CLS_score{ext}"

    output_overlay = args.output_overlay
    if output_overlay is None:
        base, ext = os.path.splitext(args.output)
        output_overlay = f"{base}_CLS_overlay{ext}"

    output_labels = args.output_labels
    if output_labels is None:
        base, ext = os.path.splitext(args.output)
        output_labels = f"{base}_CLS_labels{ext}"

    # 自定义 colormap，可改为 dict 如 {"0":"#0000FF","0.5":"#00FF00","1":"#FF0000"}
    colormap: Union[int, dict] = cv2.COLORMAP_JET
    colormap= {"0":"#18354E","0.80":"#B1CEE7","1":"#CC0300"}

    if not args.no_patch_grid:
        visualize_patch_grid(
            image_path=args.input,
            output_path=args.output,
            image_size=image_size,
            patch_size=patch_size,
        )
        visualize_patch_grid_on_original(
            image_path=args.input,
            output_path=output_original,
            image_size=image_size,
            patch_size=patch_size,
        )

    scores = get_cls_patch_similarity(
        image_path=args.input,
        model_path=model_path,
        device=device,
        image_size=image_size,
        patch_size=patch_size,
    )
    visualize_cls_patch_scores_on_original(
        image_path=args.input,
        output_path=output_cls,
        scores=scores,
        image_size=image_size,
        patch_size=patch_size,
        colormap=colormap,
    )
    visualize_cls_patch_scores_with_labels(
        image_path=args.input,
        output_path=output_labels,
        scores=scores,
        image_size=image_size,
        patch_size=patch_size,
        colormap=colormap,
    )
    visualize_cls_patch_scores_overlay(
        image_path=args.input,
        output_path=output_overlay,
        scores=scores,
        image_size=image_size,
        patch_size=patch_size,
        alpha=args.overlay_alpha,
        colormap=colormap,
    )

if __name__ == "__main__":
    main()