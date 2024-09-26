from .prefilling_pipeline import PrefillingPipeline
from .decoding_pipeline import DecodingPipeline
import time
import torch
import copy
import torch.distributed as dist
from .core.tag_manager   import Tag
def jupiter_prefilling(input_ids,model,config,args ):
    # prefilling with sequence slicing  
    if config.device == "cuda" and input_ids.device != "cuda":
        input_ids = input_ids.cuda()
    print("input_ids shape:",input_ids.shape)
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
    # runtime.comm_handler.stop_helper_threads()
    if config.is_last_stage:
        return medusa_logits, logits
    else:
        return None, None
def jupiter_prefilling_no_finish(input_ids,model,config,args   ):
    # prefilling with sequence slicing  
    # 用于share_perfix填充
    # 最终hidden_states不经过lm_heads,medusa_head
    # 并且不需要set_mask_for_medusa_decoding
    if config.device == "cuda" and input_ids.device != "cuda":
        input_ids = input_ids.cuda()
    print("input_ids shape:",input_ids.shape)
    ######################################################################
    runtime = PrefillingPipeline(model, config, args)
    if config.is_last_stage:   
        runtime.pipeline_with_sequence_slicing_no_finish( )
    else:
        if config.is_first_stage:
            runtime.pipeline_with_sequence_slicing_no_finish(input_ids)  
        else:
            runtime.pipeline_with_sequence_slicing_no_finish()
            
def point_prefilling(points,model,config,args ):
    '''
    prefiling points for every request,
    e.g.  
    points[0] =  `1. [/INST]1. Exercise`
    points[1] =  '2. [/INST]2. Mindfulness`
    and then get medusa_logits, logits for decoding phase (only for last stage)
    '''
    tokenizer = model.get_tokenizer()
    runtime = PrefillingPipeline(model, config, args)
    points_input_ids = [tokenizer.encode(point, return_tensors="pt") for point in  points]
    points_input_ids = [point[:,2:] for point in points_input_ids]
    if config.device == "cuda":
        points_input_ids = [p.cuda() for p in points_input_ids ]
    if config.is_last_stage: 
        medusa_logits_list,logits_list = runtime.points_saturation(points_input_ids)
    else:
        runtime.points_saturation(points_input_ids)
    # runtime.comm_handler.stop_helper_threads()
    if config.is_last_stage:
        return medusa_logits_list,logits_list
    else:
        return None ,None
def normal_decoding(prompt,model,config,medusa_logits=None,logits=None,input_ids=None):
    # normal decoding, no pipeline
    tag_manager = Tag()
    tensor_tag = {"tree_decoding":  tag_manager.get_next_tag(),
                  "tree_candidates":  tag_manager.get_next_tag(),
                  "new_token":  tag_manager.get_next_tag()}
    tensor_shape = {"tree_decoding": (1,  64, config.hidden_size), 
                    "tree_candidates": (1,64),
                    "new_token":(1,1+2*config.medusa_num_heads) }  
    tensor_type = {"tree_decoding": config.torch_dtype,
                    "tree_candidates":torch.int64,
                    "new_token":torch.int64}
    tokenizer = model.get_tokenizer()
    if input_ids is None:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")     
    if config.device == "cuda":
        input_ids = input_ids.cuda()
    input_len = input_ids.shape[1]
    new_token=0
    text = ""
    for idx in range( config.max_steps):
        # Step 1 : generate_candidates


        if config.is_last_stage:
            candidates, tree_candidates = model.generate_candidates(
                    medusa_logits, 
                    logits, 
            )
            
            # print("medusa_logits: ", medusa_logits[:,:,-1,-10:])
            # print("logits", logits[:,-1,-10:])
        if config.is_first_stage:
            recv_tensor = torch.zeros(tensor_shape["tree_candidates"], 
                                    dtype=tensor_type["tree_candidates"]) # int64
            dist.recv(tensor=recv_tensor, src= config.total_stage-1, tag= tensor_tag["tree_candidates"]) 
            if config.device == "cuda":
                tree_candidates = recv_tensor.to("cuda")
            else:   
                tree_candidates = recv_tensor

        if config.is_last_stage:
            if config.device == "cuda":
                send_tensor = tree_candidates.cpu()
            else:
                send_tensor = tree_candidates
            dist.send(tensor= send_tensor, dst=0, tag= tensor_tag["tree_candidates"])

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

                dist.send(tensor= send_tensor, dst= config.next_rank, tag= tensor_tag["tree_decoding"])
            else: # world == 1
                raise NotImplementedError("暂不支持单机推理")
        else:
            # TODO:candidates:torch.Size([42, 5]) tree_candidates torch.Size([1, 64]) Size可能和不同medusa choice 有关
            recv_tensor = torch.zeros(tensor_shape["tree_decoding"], dtype=  tensor_type["tree_decoding"])
            dist.recv(tensor=recv_tensor, src= config.pre_rank, tag= tensor_tag["tree_decoding"]) 
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
                dist.send(tensor= send_tensor, dst= config.next_rank,tag= tensor_tag["tree_decoding"]) 
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
            text = model.tokenizer.decode(
                        input_ids[0,  input_len:],
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                        clean_up_tokenization_spaces=True,
                    ) 
        # Step 4: Step 3 中model.update_inference_inputs 调用更新了最后一个stage的kvcache,现在同步decoding的结果，并更新其他stage的kv cache和inputs_ids
        if  config.is_last_stage: 
            new_token_len =  torch.tensor(select_indices.shape)
            # 三个拼在一起长度为  1+2*new_token_len, new_token_length <= config.medusa_num_heads
            select_indices_and_new_inputs_ids = torch.cat((new_token_len.unsqueeze(0), select_indices.unsqueeze(0).cpu(), new_input_ids.cpu()), dim=1)
            dist.broadcast(select_indices_and_new_inputs_ids,  src= config.total_stage-1)
        else:
            recv_tensor = torch.zeros(tensor_shape["new_token"],   dtype= tensor_type["new_token"]) #最多长度11
            dist.broadcast(recv_tensor,   src= config.total_stage-1) 
            if config.device == "cuda":
                select_indices_and_new_inputs_ids = recv_tensor.cuda()
            else:
                select_indices_and_new_inputs_ids = recv_tensor
            new_token_len = select_indices_and_new_inputs_ids[0,0].item()
            select_indices = select_indices_and_new_inputs_ids[:,1:new_token_len+1].view(-1)   
            new_input_ids = select_indices_and_new_inputs_ids[:, new_token_len+1:2*new_token_len+1] 
            if config.device == "cuda":
                select_indices = select_indices.cuda()
            model.update_kv_cache(input_ids,select_indices)
            if config.device == "cuda":
                new_input_ids = new_input_ids.cuda()
            input_ids = torch.cat([input_ids, new_input_ids], dim=-1    )
            text = model.tokenizer.decode(
                        input_ids[0, input_len :],
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                        clean_up_tokenization_spaces=True,
                    ) 
        if model.tokenizer.eos_token_id in input_ids[0, input_len:]:
            break
    return text
def outline_based_decoding( model,config,args ):
    model.set_mask_for_medusa_decoding()
    runtime = DecodingPipeline(model, config, args)
    runtime.jupiter_decoding_pipeline()