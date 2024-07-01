import argparse

import torch
import torch.distributed as dist

from medusa.pipeline_model.medusa_llama_pp import PPMedusaLlamaForCausalLM
from medusa.pipeline_model.llama_config import LlamaConfig
from medusa.pipeline_model.dis_utils import initialize_distributed,get_stage_state_dict,get_medusa_model_state_dict

from medusa.pipeline_model.PrefillingPipeline import PrefillingPipeline

def main(args):
    config = LlamaConfig.from_pretrained(f"./config_{args.rank}.json")
    initialize_distributed(config, args)
    stage_state_dict = get_stage_state_dict(
        config.base_model_name_or_path,
        config.medusa_head_path,
        config.stage_num_hidden_layers_list,
        args.rank
    )
    config.update_pp_stage_config()
    model = PPMedusaLlamaForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=None,
                    config=config, 
                    state_dict = stage_state_dict, 
                    use_safetensors=False ,
                    torch_dtype=torch.float16,
    ).to("cuda")
    tokenizer = model.get_tokenizer()

    prompt ="""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, 
        detailed, and polite answers to the user's questions. USER: What is Jupiter? ASSISTANT:"""
    #TODO: 或许应该第一个stage广播prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")


    # Pipelined Prefilling Stage
    runtime = PrefillingPipeline(model, config, args)
    if config.is_last_stage:    # last stage会产生medusa logits供decoding阶段使用
        medusa_logits, logits = runtime.pipeline_forward(input_ids)
    else:
        runtime.pipeline_forward(input_ids)


    # Decoding stage 
    # generate_candidates
    new_token=0
    for idx in range(config.max_steps):
        # 在最后一个stage生成candidates
        if config.is_last_stage:
            candidates, tree_candidates = model.generate_candidates(
                    medusa_logits, 
                    logits, 
            )

        # last stage 将 tree_candidates 传给first stage, TODO:tree_candidates torch.Size([1, 64]) 可能和不同medusa choice 有关
        if config.total_stage > 1:
            if config.is_first_stage:
                recv_tensor = torch.zeros(1, 64, dtype=torch.int64) # int
                dist.recv(tensor=recv_tensor, src= config.total_stage-1, tag=1) 
                tree_candidates = recv_tensor.to("cuda")
                # print("Stage {} revecive {} from Stage {}".format( config.stage,tree_candidates.shape,config.total_stage-1 ))
            if config.is_last_stage:
                send_tensor = tree_candidates.cpu()
                dist.send(tensor= send_tensor, dst=0, tag=1)
                # print("Stage {} send {} to Stage {}".format( config.stage,tree_candidates.shape,0))

        # tree decoding
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
                # print("send to next stage", config.next_rank)
            else: # world == 1
                raise NotImplementedError("暂不支持单机推理")
        else:
            # TODO:candidates:torch.Size([42, 5]) tree_candidates torch.Size([1, 64]) Size可能和不同medusa choice 有关
            recv_tensor = torch.zeros(1,  64, config.hidden_size, dtype=torch.float16)
            dist.recv(tensor=recv_tensor, src= config.pre_rank, tag=1) 
            # print( "receive from previous stage", config.pre_rank)
            hidden_states = recv_tensor.to("cuda")
            if not config.is_last_stage:
                hidden_states = model.tree_decoding(
                    tree_candidates = None,
                    tree_candidates_embeds = hidden_states,
                    input_ids = input_ids
                )
                send_tensor = hidden_states.cpu()
                dist.send(tensor= send_tensor, dst= config.next_rank, tag=1)
                # print("send to next stage", config.next_rank)
            else:
                medusa_logits, logits  = model.tree_decoding(
                    tree_candidates = None,
                    tree_candidates_embeds = hidden_states,
                    input_ids = input_ids
                )
                
        # 完成decoding阶段的前向传播推理后，我们在最后一个stage进行evaluate_posterior 和 update_inference_inputs
        # 这个过程会选择出best candidate作为采样的结果，同时更新inputs_ids
        if config.is_last_stage:
            best_candidate, accept_length = model.evaluate_posterior(logits,
                            candidates)
            # print("best_candidate {} accept_length {}".format(best_candidate, accept_length ))
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

        # 前面model.update_inference_inputs 调用更新了最后一个stage的kvcache
        # 现在同步decoding的结果，并更新其他stage的kv cache和inputs_ids
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
                new_token_len = recv_tensor.cuda()
                select_indices =  torch.zeros( new_token_len ,   dtype=torch.int64)
                new_input_ids =  torch.zeros( 1,new_token_len ,   dtype=torch.int64)
                dist.broadcast(select_indices,   src= config.total_stage-1) 
                dist.broadcast(new_input_ids,   src= config.total_stage-1) 
                select_indices = select_indices.cuda()
                model.update_kv_cache(input_ids,select_indices)
                new_input_ids = new_input_ids.cuda()
                input_ids = torch.cat([input_ids, new_input_ids], dim=-1    )
                print(model.tokenizer.decode(
                            new_input_ids[0,  :],
                            skip_special_tokens=True,
                            spaces_between_special_tokens=False,
                            clean_up_tokenization_spaces=True,
                        ) )
        if model.tokenizer.eos_token_id in new_input_ids[0,  :]:
            print("\n")
            print("finish decoding")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--world', default=2, type=int)
    parser.add_argument(
        "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
    )
    args = parser.parse_args()
    #TODO: config里增加device和 dtype

    main(args)