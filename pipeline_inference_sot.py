import argparse
import time
import torch
import torch.distributed as dist
from tasks.medusa_llama.llama_config import LlamaConfig
from tools.utils import initialize_distributed,get_max_memory,get_model_type
from tools.sot import get_skeleton_prompt,get_point_expanding_prompt
from jupiter.utils import jupiter_prefilling,normal_decoding,jupiter_prefilling_no_finish,point_prefilling,outline_based_decoding
from  tasks.medusa_llama.outline_decoding_controller  import get_controller,OutlineDecodingController,set_controller  #[modified]

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
    print("Data_type:", model.dtype)
    # for quantization
    if not args.load_in_8bit and not args.load_in_4bit:
        model = model.to(config.device)
    question = "What are the most effective ways to deal with stress?"
    prompt = get_skeleton_prompt(question)
    # Step 1: prefilling with sequence slicing
    medusa_logits, logits = jupiter_prefilling(prompt,model,config,args)
    dist.barrier()
    # # Step 2: normal decoding
    answer = normal_decoding(prompt,model,config,medusa_logits,logits)
    skeleton = "\n".join([line.lstrip() for line in answer.splitlines()])
    print("===========================================\n")
    print("Skeleton:\n", skeleton)
    points,shared_perfix,prompts_for_points = get_point_expanding_prompt(skeleton, question)
    print("===========================================\n")
    print("Shared perfix:\n",shared_perfix)
    print("===========================================\n")
    tokenizer = model.tokenizer
    print("prompts_for_points: ")
    for i in prompts_for_points:
        print(i)
    # Step 3: shared perfix prefiling
    input_ids_1 = tokenizer.encode( shared_perfix, return_tensors="pt")
    jupiter_prefilling_no_finish(shared_perfix ,model,config,args,input_ids =input_ids_1 )
    dist.barrier()
    # # Step 4: point request prefiling, and get medusa_logits, logits for every point 
    set_controller(OutlineDecodingController(points,config,model))
    print("==============================\n point_prefilling")
    medusa_logits_list,logits_list = point_prefilling(prompts_for_points ,model,config,args )
    dist.barrier()
    if config.is_last_stage:
        logits = logits_list[0]
        print(logits.shape)
        print("logits", logits[:,-1,-10:])
        torch.save(logits,"logits_split.pt")
        
    # # prepare reuquets
    if config.is_last_stage:
        get_controller().add_requests(medusa_logits_list,logits_list)
    # prepare inout_ids

    input_ids_for_point=[]
    for i in range(len(prompts_for_points)): 
        input_ids_1 = tokenizer.encode( shared_perfix, return_tensors="pt")
        input_ids_2 = tokenizer.encode(prompts_for_points[i], return_tensors="pt")   
        input_ids = torch.cat([input_ids_1, input_ids_2[:,2:] ], dim=1)
        if config.device == "cuda":
            input_ids = input_ids.cuda()
        input_ids_for_point.append(input_ids)
    get_controller().set_up_input_ids_for_point(input_ids_for_point)
    dist.barrier()
    # Step 5: jupiter decoding
    print("==============================\n outline_based_decoding")
    outline_based_decoding(model,config,args)
    dist.barrier()
    get_controller().get_output(tokenizer) 
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