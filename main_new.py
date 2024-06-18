import torch
import argparse
from medusa.pipeline_model.medusa_llama import MedusaLlamaForCausalLM
from medusa.pipeline_model.llama_config import LlamaConfig
from medusa.pipeline_model.dis_utils import initialize_distributed
from   medusa.pipeline_model.dis_utils  import   get_medusa_model_state_dict

def main(args):
    config = LlamaConfig.from_pretrained( args.base_model_path ) # 包含vicuna-7b-v1.3 config和medusa head config
    all_state_dict = get_medusa_model_state_dict(args.base_model_path, args.medusa_head_path)    
    model = MedusaLlamaForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=  None,
                    config=config, 
                    state_dict = all_state_dict, 
                    use_safetensors=False ,
                    torch_dtype=torch.float16,
    )
    model.to("cuda")
    print(model)
    print(model.dtype)
    tokenizer = model.get_tokenizer()
    prompt ="""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, 
        detailed, and polite answers to the user's questions. USER: What is Jupiter? ASSISTANT:"""
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
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="vicuna-7b-v1.3", help="Model name or path.")
    parser.add_argument("--medusa_head_path", type=str, default="medusa-vicuna-7b-v1.3", help="Model name or path.")

    parser.add_argument(
        "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=512)
    args = parser.parse_args()
    main(args)