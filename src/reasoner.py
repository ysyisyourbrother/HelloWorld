from models.llava.model.builder import load_pretrained_model
from models.llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from models.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from models.llava.conversation import conv_templates, SeparatorStyle
import torch
import os
from PIL import Image
import requests
from io import BytesIO
# 设置为离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HUGGINGFACE_HUB_OFFLINE"] = "1"

# class LLaVA:
#     def __init__(self, model_path="/mnt/share/cache/models/LLaVA-Video-7B-Qwen2", device="cuda:0"):
#         self.model_path = model_path
#         self.device = device
#         self.model = None
#         self.tokenizer = None
#         self.image_processor = None
#         self.model_name = None
#         self.initialize_model()
    
#     def initialize_model(self):
#         """使用load_pretrained_model函数初始化模型"""
#         self.model_name = get_model_name_from_path(self.model_path)
#         self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
#             model_path=self.model_path,
#             model_base=None,
#             model_name=self.model_name
#         )
#         self.model = self.model.to(self.device)
    
#     def generate_response(self, prompt, images=None, temperature=0.1, max_new_tokens=512):
#         """生成响应"""
#         if images is None:
#             images = []
        
#         # 处理对话模板
#         conv = conv_templates["llava_v1"].copy()
        
#         # 如果有图片，添加图片token
#         if images:
#             # 处理图片
#             image_sizes = [img.size for img in images]
#             images = process_images(images, self.image_processor, self.model.config)
#             images = images.to(self.model.device, dtype=torch.float16)
            
#             # 在prompt前添加图片token
#             image_token_len = self.model.config.num_image_embeds
#             img_tokens = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
#             if self.model.config.mm_use_im_start_end:
#                 img_tokens = DEFAULT_IM_START_TOKEN + img_tokens + DEFAULT_IM_END_TOKEN
#             prompt = img_tokens + "\n" + prompt
        
#         # 添加prompt到对话
#         conv.append_message(conv.roles[0], prompt)
#         conv.append_message(conv.roles[1], "")
#         prompt = conv.get_prompt()
        
#         # 处理输入
#         input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        
#         # 生成响应
#         with torch.inference_mode():
#             output_ids = self.model.generate(
#                 input_ids, 
#                 images=images, 
#                 temperature=temperature, 
#                 max_new_tokens=max_new_tokens,
#                 use_cache=True
#             )
        
#         # 解码响应
#         output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
#         response = output.split(conv.roles[1])[-1].strip()
        
#         return response
    
#     def load_image_from_url(self, url):
#         """从URL加载图片"""
#         response = requests.get(url)
#         return Image.open(BytesIO(response.content))
    
#     def load_image_from_path(self, path):
#         """从本地路径加载图片"""
#         return Image.open(path)

