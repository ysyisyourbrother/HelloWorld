import argparse
import time
import inspect

import torch
import torch.distributed as dist

from tasks.medusa_llama.llama_config import LlamaConfig
from tools.utils import initialize_distributed,get_max_memory,get_model_type
from tools.sot import get_skeleton_prompt,get_point_expanding_prompt
from jupiter.utils import jupiter_prefilling,normal_decoding,jupiter_prefilling_no_finish,point_prefilling
import tasks.medusa_llama.outline_decoding_controller   as outline_decoding_controller
def main(args):
    if get_model_type(args.config_file) == 'vicuna_7b' or get_model_type(args.config_file) == 'vicuna_13b':
        config = LlamaConfig.from_pretrained(args.config_file) # 包含vicuna-7b-v1.3 config和medusa head config的内容
        temp_path = "temp_{}_world_{}_rank_{}/stage.bin".format(get_model_type(args.config_file), args.world,  args.rank)
        from tasks.medusa_llama.medusa_llama_pp import PPMedusaLlamaForCausalLM as PPMedusaModel
    else:
        raise NotImplementedError("暂不支持该模型")
    print("temp_path:", temp_path)
    initialize_distributed(config, args)
    config.update_pp_stage_config(args)
    start = time.time()
    mem_before =  get_max_memory(config)
    # load model
    if config.device == "cuda":
        with torch.device("cuda"):
            model =  PPMedusaModel.from_pretrained(
                                        pretrained_model_name_or_path=  temp_path,
                                        config=config, 
                                        use_safetensors=False ,
                                        torch_dtype= config.torch_dtype,
                                        load_in_4bit=args.load_in_4bit,
                                        load_in_8bit=args.load_in_8bit
            )
    else:
        model =  PPMedusaModel.from_pretrained(
                                        pretrained_model_name_or_path=  temp_path,
                                        config=config, 
                                        use_safetensors=False ,
                                        torch_dtype= config.torch_dtype,
                                        load_in_4bit=args.load_in_4bit,
                                        load_in_8bit=args.load_in_8bit
            ) 
    model.eval()
    # for quantization
    if not args.load_in_8bit and not args.load_in_4bit:
        model = model.to(config.device)
    mem_after =  get_max_memory(config)
    print("Model device:", model.device)
    print("Load time:", time.time() - start)
    print("After load model: {}".format((mem_after - mem_before)/1024/1024)  )
    print( "Model memory footprint",  model.get_memory_footprint()/(1024*1024))
    question = "What are the most effective ways to deal with stress?"
    prompt = get_skeleton_prompt(question)
    # Step 1: prefilling with sequence slicing
    medusa_logits, logits = jupiter_prefilling(prompt,model,config,args)
    dist.barrier()
    # Step 2: normal decoding
    answer = normal_decoding(prompt,model,config,medusa_logits,logits)
    skeleton = "\n".join([line.lstrip() for line in answer.splitlines()])
    print("skeleton:\n", skeleton)
    points,shared_perfix,prompts_for_points = get_point_expanding_prompt(skeleton, question)
    point_num = len(prompts_for_points)
    print("point number:", point_num)
    tokenizer = model.get_tokenizer()
    input_ids = tokenizer.encode(shared_perfix, return_tensors="pt")     
    print(input_ids.shape)
    # Step 3: shared perfix prefiling
    jupiter_prefilling_no_finish(shared_perfix ,model,config,args)
    dist.barrier()
    # Step 4: point request prefiling, and get medusa_logits, logits for every request 
    outline_decoding_controller.controller=outline_decoding_controller.OutlineDecodingController(point_num,config,model)
    outline_decoding_controller.controller.prepare_point_kv_cache()
    point_prefilling(prompts_for_points ,model,config,args)
    outline_decoding_controller.controller.check()
    dist.barrier()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--world', default=2, type=int)
    parser.add_argument("--config_file", type=str, default="config/vicuna_7b_config.json", help="Model name or path.")
    parser.add_argument(
        "--load_in_8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--load_in_4bit", action="store_true", help="Use 4-bit quantization"
    )
    args = parser.parse_args()
    #TODO: config里增加和 dtype

    main(args)  