import argparse
import time
import torch
import torch.distributed as dist
import os
import inspect
from medusa.pipeline_model.llama_config import LlamaConfig
from medusa.pipeline_model.mistral_config import MistralConfig

from medusa.pipeline_model.dis_utils import initialize_distributed,get_module_memory,get_max_memory,get_model_type
from medusa.pipeline_model.PrefillingPipeline import PrefillingPipeline

def main(args):
    if get_model_type(args.config_file) == 'vicuna_7b' or get_model_type(args.config_file) == 'vicuna_13b':
        config = LlamaConfig.from_pretrained( args.config_file) # 包含vicuna-7b-v1.3 config和medusa head config的内容
        temp_path = "temp_{}_world_{}_rank_{}/stage.bin".format(get_model_type(args.config_file), args.world,  args.rank)
        from medusa.pipeline_model.medusa_llama_pp import PPMedusaLlamaForCausalLM as PPMedusaModel
    elif get_model_type(args.config_file) == 'zephyr':
        config = MistralConfig.from_pretrained( args.config_file) # 包含vicuna-7b-v1.3 config和medusa head config的内容
        temp_path = "temp_zephyr_world_{}_rank_{}/stage.bin".format( args.world,  args.rank)
        from medusa.pipeline_model.medusa_mistral_pp import  PPMedusaMistralForCausalLM as PPMedusaModel
    else:
        raise NotImplementedError("暂不支持该模型")
    print("temp_path:", temp_path)
    initialize_distributed(config, args)
    config.update_pp_stage_config(args)
    start = time.time()
    mem_before =  get_max_memory(config)
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
    print(model)
    if not args.load_in_8bit and not args.load_in_4bit:
        model = model.to(config.device)
    mem_after =  get_max_memory(config)
    print("model device:", model.device)
    print("load time:", time.time() - start)
    print("after load model: {}".format((mem_after - mem_before)/1024/1024)  )
    print( "Model   memory footprint",  model.get_memory_footprint()/(1024*1024))


    if config.is_last_stage: 
        print( "medusa head mem", get_module_memory(model.medusa_head)/1024/1024)
        print( "one medusa head mem", get_module_memory(model.medusa_head[0])/1024/1024)
        print("lm_head mem", get_module_memory(model.lm_head)/1024/1024)
    if config.is_first_stage:
        print("embedding mem", get_module_memory(model.model.embed_tokens)/1024/1024)
    print(model.model.layers[0].self_attn.q_proj.weight.dtype)
    if config.is_last_stage:
        print("model.lm_head.weight.dtype", model.lm_head.weight.dtype)
        print("model.medusa_head[0][0].linear.weight.dtype", model.medusa_head[0][0].linear.weight.dtype)
    tokenizer = model.get_tokenizer()
    

    prompt ="""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, 
        detailed, and polite answers to the user's questions. USER: Tell me what do you know about Jupiter? . ASSISTANT:"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")     
    if config.device == "cuda":
        input_ids = input_ids.cuda()
    start = time.time()
    ######################################################################
    # Pipelined Prefilling Stage
    start = time.time()
    runtime = PrefillingPipeline(model, config, args)

    if config.is_last_stage:    # last stage会产生medusa logits供decoding阶段使用
        medusa_logits, logits = runtime.pipeline_with_sequence_slicing( )
    else:
        if config.is_first_stage:
            runtime.pipeline_with_sequence_slicing(input_ids) # or use runtime.pipeline_forward
        else:
            runtime.pipeline_with_sequence_slicing()
    print("prefilling time:", time.time() - start)
    ######################################################################
    # Decoding stage 
    new_token=0
    start = time.time()
    for idx in range( config.max_steps):
        # Step 1 : generate_candidates
        if config.is_last_stage:
            candidates, tree_candidates = model.generate_candidates(
                    medusa_logits, 
                    logits, 
            )
        if config.total_stage > 1:
            if config.is_first_stage:
                recv_tensor = torch.zeros(1, 64, dtype=torch.int64) # int
                dist.recv(tensor=recv_tensor, src= config.total_stage-1, tag=1) 
                if config.device == "cuda":
                    tree_candidates = recv_tensor.to("cuda")
                else:   
                    tree_candidates = recv_tensor
            if config.is_last_stage:
                if config.device == "cuda":
                    send_tensor = tree_candidates.cpu()
                else:
                    send_tensor = tree_candidates
                dist.send(tensor= send_tensor, dst=0, tag=1)
        # Step 2 tree decoding
        # TODO:tree_decoding所有stage 都需要 input_ids, 需要广播input_ids, 每一次decoding都会修改inputs_ids
        # TODO: 是否需要用到inputs_ids，还是只需要用到input_ids.shape[1]?
        if config.is_first_stage:
            if not config.is_last_stage:
                hidden_states = model.tree_decoding(
                    tree_candidates = tree_candidates,
                    tree_candidates_embeds = None,
                    input_ids = input_ids
                )
                # 不是最后一个stage就要往前传播数据
                send_tensor = hidden_states.cpu()
                dist.send(tensor= send_tensor, dst= config.next_rank, tag=1)
            else: # world == 1
                raise NotImplementedError("暂不支持单机推理")
        else:
            # TODO:candidates:torch.Size([42, 5]) tree_candidates torch.Size([1, 64]) Size可能和不同medusa choice 有关
            recv_tensor = torch.zeros(1,  64, config.hidden_size, dtype= config.torch_dtype)
            dist.recv(tensor=recv_tensor, src= config.pre_rank, tag=1) 
            if config.device == "cuda":
                hidden_states = recv_tensor.to("cuda")
            else:
                hidden_states = recv_tensor
            if not config.is_last_stage:
                hidden_states = model.tree_decoding(
                    tree_candidates = None,
                    tree_candidates_embeds = hidden_states,
                    input_ids = input_ids
                )
                send_tensor = hidden_states.cpu()
                dist.send(tensor= send_tensor, dst= config.next_rank, tag=1)
            else:
                medusa_logits, logits  = model.tree_decoding(
                    tree_candidates = None,
                    tree_candidates_embeds = hidden_states,
                    input_ids = input_ids
                )
        # 完成decoding阶段的前向传播推理后，我们在最后一个stage进行evaluate_posterior 和 update_inference_inputs
        # Step 3: 选择出best candidate作为采样的结果，同时更新inputs_ids
        if config.is_last_stage:
            best_candidate, accept_length = model.evaluate_posterior(logits,
                            candidates)
            input_ids, logits, medusa_logits, new_token , select_indices= model.update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                logits,
                medusa_logits,
                new_token,
            )  
            new_input_ids = input_ids[:,  -select_indices.shape[0]:]
            print(model.tokenizer.decode(
                        new_input_ids[0,  :],
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                        clean_up_tokenization_spaces=True,
                    ) )
        # Step 4: 前面model.update_inference_inputs 调用更新了最后一个stage的kvcache,现在同步decoding的结果，并更新其他stage的kv cache和inputs_ids
        if config.total_stage > 1:
            if  config.is_last_stage: #scatter new_token_len
                new_token_len =  torch.tensor(select_indices.shape)
                dist.broadcast(new_token_len,  src= config.total_stage-1)
                # select_indices 和 new_inputs_ids
                #TODO: 通信合并为一次， new_token_len 是一个数,shape [1], select_indices.shape = [1], new_input_ids.shape = [1,n]
                # 把三个拼在一起,第一个是new_token_len， 后面第new_token_len个元素是select_indices，在后面new_token_len元素是new_input_ids
                # 三个拼在一起长度为  1+2*new_token_len, new_token_length 不会很长 现在最多是5
                dist.broadcast(select_indices,    src= config.total_stage-1) 
                dist.broadcast(new_input_ids,    src= config.total_stage-1)
            else:
                recv_tensor = torch.zeros( 1,   dtype=torch.int64)
                dist.broadcast(recv_tensor,   src= config.total_stage-1) 
                if config.device == "cuda":
                    new_token_len = recv_tensor.cuda()
                else:
                    new_token_len = recv_tensor
                select_indices =  torch.zeros( new_token_len ,   dtype=torch.int64)
                new_input_ids =  torch.zeros( 1,new_token_len ,   dtype=torch.int64)
                dist.broadcast(select_indices,   src= config.total_stage-1) 
                dist.broadcast(new_input_ids,   src= config.total_stage-1) 
                if config.device == "cuda":
                    select_indices = select_indices.cuda()
                model.update_kv_cache(input_ids,select_indices)
                if config.device == "cuda":
                    new_input_ids = new_input_ids.cuda()
                input_ids = torch.cat([input_ids, new_input_ids], dim=-1    )
                print(model.tokenizer.decode(
                            new_input_ids[0,  :],
                            skip_special_tokens=True,
                            spaces_between_special_tokens=False,
                            clean_up_tokenization_spaces=True,
                        ) )
        if model.tokenizer.eos_token_id in new_input_ids[0,  :]:
            print("Final step: ", idx+1 )
            print("\nfinish decoding")
            break
    end = time.time()
    print("time", end - start)
    print(model.dtype)


    total_params = sum(p.numel() for p in model.parameters())
    print("total_params", total_params)
    print(f"{inspect.currentframe().f_code.co_filename} line {inspect.currentframe().f_lineno}  Max memory allocated: { get_max_memory(config) / (1024 * 1024)}")
    print(f"{inspect.currentframe().f_code.co_filename} line {inspect.currentframe().f_lineno} Max memory allocated: {  torch.cuda.max_memory_allocated() / (1024 * 1024)}")

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
 