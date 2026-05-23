from models.bge.modeling_MMRet_CLIP import CLIPModel
import torch

class TextBGEVectorizer:
    """BGE模型向量化器"""
    def __init__(self, device: str, model_path: str):
        self.device = device
        
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.model.set_processor(model_path)
        self.processor = self.model.processor
        self.model.eval()
    
    def encode(self, query: str):
        """
        对文本进行向量化
        TODO: 目前只支持单次 77 token 的编码
        """
        txt = self.processor(text=query, 
                            return_tensors="pt", # Return PyTorch `torch.Tensor` objects.
                            padding=True, 
                            truncation=True, 
                            max_length=77)
        txt = {k: v.to(self.device) for k, v in txt.items()}
        with torch.no_grad():
            vector = self.model.encode_text(txt)
            return vector.cpu().numpy()

    def encode_no_norm(self, query: str):
        """
        对文本进行向量化, 且不归一化
        TODO: 目前只支持单次 77 token 的编码
        """
        txt = self.processor(text=query, 
                            return_tensors="pt", # Return PyTorch `torch.Tensor` objects.
                            padding=True, 
                            truncation=True, 
                            max_length=77)
        txt = {k: v.to(self.device) for k, v in txt.items()}
        with torch.no_grad():
            vector = self.model.get_text_features(**txt)
            return vector