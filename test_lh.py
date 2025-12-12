#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import torch
import cv2
import copy
import multiprocessing as mp
from decord import VideoReader

from models.llava.model.builder import load_pretrained_model
from models.llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from models.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from models.llava.conversation import SeparatorStyle 
from models.llava.conversation import conv_qwen

from src.stream_input import StreamInput
from src.frame_vectorizer import FrameVectorizer
from src.config import Config

def llava_inference(model, tokenizer,qs, video, conv_template = "qwen_1_5", device="cuda"):
    if video is not None:
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        question = qs
    else:
        question = qs
    print("======================================================")
    print("question:", question)
    conv = copy.deepcopy(conv_qwen)
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, 
                    return_tensors="pt").unsqueeze(0).to(device)
    if tokenizer.pad_token_id is None:
        if "qwen" in tokenizer.name_or_path.lower():
            print("Setting pad token to bos token for qwen model.")
            tokenizer.pad_token_id = 151643
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids, images=video, 
            attention_mask=attention_masks, 
            modalities="video", 
            do_sample=False, 
            temperature=0.0,
            max_new_tokens=16, #NOTE 对齐 generate_videomme.py
            top_p=0.1,
            num_beams=1, 
            use_cache=True)
    text_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return text_outputs

if __name__ == "__main__":
    # 有关模型的参数
    model_path = "/mnt/share/cache/models/LLaVA-Video-7B-Qwen2"
    
    overwrite_config = {}
    overwrite_config["mm_spatial_pool_mode"] =  "average"
    # 加载模型
    mem_before = torch.cuda.max_memory_allocated()
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path=model_path, 
        model_base=None, 
        model_name="llava_qwen", 
        torch_dtype="bfloat16", 
        load_in_8bit=False,
        load_in_4bit=False, 
        device_map="auto",
        # device_map=device_map,
        attn_implementation="eager",
        overwrite_config=overwrite_config)  # Add any other thing you want to pass in llava_model_args
    mem_after = torch.cuda.max_memory_allocated()
    print("LLaVA-Video-7B-Qwen2 memory usage: {:.2f} GB".format((mem_after - mem_before) / 1024 / 1024/ 1024))
    model.eval()
    # 读取图片
    img_path = "test_frame.png"
    frame = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 强制读取为彩色图像（BGR格式的numpy数组）
    if frame.ndim == 2:  # 如果是灰度图像，转换为RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:  # 如果是彩色图像，转换为RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 开始推理
    question_text = "Hello, please describe this image."
    frames = [frame]
    processed_frames = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    text_outputs = llava_inference(model, tokenizer, question_text, processed_frames)
    print("======================================================")
    print("text_outputs: ", text_outputs)