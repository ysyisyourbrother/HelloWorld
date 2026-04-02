import cv2
import argparse
import json
import os
import sys
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import Config


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
    parser = argparse.ArgumentParser(description="BGE patch 网格可视化")
    parser.add_argument("-i", "--input", default="motivation_results/frames/frame_selected.jpg", help="输入图片路径")
    parser.add_argument("-o", "--output", default="motivation_results/frames/frame_patch_grid.jpg", help="输出路径 (224x224)")
    parser.add_argument(
        "-o2", "--output_original",
        default=None,
        help="原图尺寸输出路径 (默认: 在 -o 基础上加 _original_size)"
    )
    parser.add_argument("-c", "--config", default="configs/config_moti.json", help="配置文件，用于读取 BGE 模型路径")
    parser.add_argument("--image_size", type=int, default=None, help="模型输入尺寸 (默认从 config 读取，否则 224)")
    parser.add_argument("--patch_size", type=int, default=None, help="Patch 大小 (默认从 config 读取，否则 16)")
    args = parser.parse_args()

    image_size = args.image_size
    patch_size = args.patch_size
    if image_size is None or patch_size is None:
        isz, psz = _load_bge_patch_params(args.config)
        image_size = image_size if image_size is not None else isz
        patch_size = patch_size if patch_size is not None else psz

    output_original = args.output_original
    if output_original is None:
        base, ext = os.path.splitext(args.output)
        output_original = f"{base}_original_size{ext}"

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

if __name__ == "__main__":
    main()