import torch
import argparse
from medusa.pipeline_model.llama_config import LlamaConfig
from medusa.pipeline_model.mistral_config import MistralConfig

import os
def main(args):
    if 'vicuna' in args.config_file:
        config = LlamaConfig.from_pretrained( args.config_file) # 包含vicuna-7b-v1.3 config和medusa head config的内容
        if '7b' in args.config_file:
            model_path = "temp_vicuna_7b_world_1_rank_0/stage.bin"  # 在运行weight_split.py得到的权重路径,要将两个路径权重合并到一个文件
        else:
            model_path =  config.base_model_name_or_path # 里面是 vicuna+medusa_head 不需要合并
        from medusa.pipeline_model.medusa_llama import MedusaLlamaForCausalLM as MedusaModel
    elif 'zephyr' in args.config_file:
        config = MistralConfig.from_pretrained( args.config_file) # 包含vicuna-7b-v1.3 config和medusa head config的内容
        model_path = config.base_model_name_or_path # 里面是 zephyr+medusa_head 不需要合并
        from medusa.pipeline_model.medusa_mistral import  MedusaMistralForCausalLM as MedusaModel
    else:
        raise NotImplementedError
    model_path= "model/medusa-1.0-vicuna-13b-v1.5"
    mem_before = torch.cuda.memory_allocated() 
    if config.device == "cuda":
        with torch.device("cuda"):
            model =  MedusaModel.from_pretrained(
                pretrained_model_name_or_path=  model_path,
                config=config, 
                use_safetensors=False ,
                torch_dtype=config.torch_dtype,
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit
            ) 
    else:
        model =  MedusaModel.from_pretrained(
                pretrained_model_name_or_path=  model_path,
                config=config, 
                use_safetensors=False ,
                torch_dtype=config.torch_dtype,
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit
            )
    model.eval()
    if not args.load_in_8bit and not args.load_in_4bit:
        model = model.to(config.device)
    print(model)
    mem_after = torch.cuda.memory_allocated()
    print("after load model: {}".format((mem_after - mem_before)/1024/1024)  )
    total_params = sum(p.numel() for p in model.parameters())
    print("total_params", total_params)
    print(total_params*2 /(1024*1024))
    print( "model memory footprint",  model.get_memory_footprint()/(1024*1024))
    print(model.dtype)
    tokenizer = model.get_tokenizer()
    prompt ="""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, 
        detailed, and polite answers to the user's questions. USER: Tell me what do you know about Jupiter? . ASSISTANT:"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
            model.device
        )
    print("input_ids",input_ids.shape)
    output_stream = model.medusa_generate(
            input_ids,
            temperature=args.temperature,
            max_steps=args.max_steps,
        )
    # 从generator中获取输出结果
    pre = 0
    for outputs in output_stream:
        output_text = outputs["text"]
        output_text = output_text.strip().split(" ")
        now = len(output_text) - 1
        if now > pre:
            print(" ".join(output_text[pre:now]), end=" ", flush=True)
            pre = now
    print(" ".join(output_text[pre:]), flush=True)
    max_memory = torch.cuda.max_memory_allocated(device= model.device)
    print("Max memory:  {} ( {} MB ) ".format( max_memory , max_memory /(1024*1024) ))    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config/vicuna_7b_config.json", help="Config file path.")

    parser.add_argument(
        "--load_in_8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--load_in_4bit", action="store_true", help="Use 4-bit quantization"
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()
    main(args)
