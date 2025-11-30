import multiprocessing as mp
import numpy as np
import time
import torch
# 本项目
from src.config import Config
from models.bge.modeling_MMRet_CLIP import CLIPModel, CLIPProcessor

class TextBGEVectorizer:
    """BGE模型向量化器"""
    def __init__(self, config: Config):
        self.device = config.query_device
        model_path = config.query_model_path
        
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.model.set_processor(model_path)
        self.processor = self.model.processor
        self.model.eval()
    
    def encode(self, query: str):
        """对查询文本进行向量化"""
        txt = self.processor(text=query, 
                            return_tensors="pt", # Return PyTorch `torch.Tensor` objects.
                            padding=True, 
                            truncation=True, 
                            max_length=77)
        txt = {k: v.to(self.device) for k, v in txt.items()}
        with torch.no_grad():
            vector = self.model.encode_text(txt)
            return vector.cpu().numpy()


# if __name__ == "__main__":
#     config = Config()
#     vectorizer = TextBGEVectorizer(config)
#     query = "你好"
#     vector = vectorizer.encode(query)
#     print(vector.shape)