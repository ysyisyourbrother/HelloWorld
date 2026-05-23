"""
仅包含 BGE 图像向量化器，避免通过 frame_vectorizer 间接导入 video_input/decord。
Jetson 等环境可不装 decord 而单独使用本模块。
"""

from typing import List, Optional, Tuple

import torch

from models.bge.modeling_MMRet_CLIP import CLIPModel


class ImageBGEVectorizer:
    """BGE模型向量化器"""

    def __init__(self, device: str, model_path: str, attn_implementation: str = "sdpa"):
        self.device = device
        self.attn_implementation = attn_implementation

        self.model = CLIPModel.from_pretrained(
            model_path,
            attn_implementation=attn_implementation,
        ).to(self.device)
        self.model.set_processor(model_path)
        self.processor = self.model.processor
        self.model.eval()

    def encode(self, frame):
        img = self.processor(images=frame, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.no_grad():
            vector = self.model.encode_image(images=img)
        return vector

    def encode_no_norm(self, frame):
        img = self.processor(images=frame, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.no_grad():
            vector = self.model.get_image_features(images=img)
        return vector

    def encode_batch(self, frames: list):
        """批量编码多帧图像，frames 为 numpy 数组列表 [H,W,C] RGB"""
        if not frames:
            return torch.empty(0)
        img = self.processor(images=frames, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.no_grad():
            vectors = self.model.encode_image(images=img)
        return vectors

    def encode_batch_no_norm(self, frames: list):
        """批量编码多帧图像，但是不归一化，frames 为 numpy 数组列表 [H,W,C] RGB"""
        if not frames:
            return torch.empty(0)
        img = self.processor(images=frames, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.no_grad():
            vectors = self.model.get_image_features(images=img)
        return vectors

    def encode_batch_with_vision_outputs(
        self,
        frames: List,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[object]]:
        if not frames:
            return torch.empty(0), None
        img = self.processor(images=frames, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.no_grad():
            vision_outputs = self.model.vision_model(
                pixel_values=img,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            pooled_output = vision_outputs[1]
            image_features = self.model.visual_projection(pooled_output)
            vectors = torch.nn.functional.normalize(image_features, dim=-1)
        return vectors, vision_outputs
