import torch
import argparse
import torch.distributed as dist

from medusa.pipeline_model.medusa_llama_pp import PPMedusaLlamaForCausalLM
from medusa.pipeline_model.llama_config import LlamaConfig
from medusa.pipeline_model.dis_utils import initialize_distributed,get_stage_state_dict,get_medusa_model_state_dict
def main(args):
    config = LlamaConfig.from_pretrained(  args.config_file ) # 包含vicuna-7b-v1.3 config和medusa head config
    initialize_distributed(config, args)
    stage_state_dict = get_stage_state_dict(
        config.base_model_name_or_path,
        config.medusa_head_path,
        config.stage_num_hidden_layers_list,
        args.rank
    )
    config.update_pp_stage_config()
    config.print_config()
    model = PPMedusaLlamaForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=  None,
                    config=config, 
                    state_dict = stage_state_dict, 
                    use_safetensors=False ,
                    torch_dtype=torch.float16,
    )
    model.to("cuda")
    print(model)
    tokenizer = model.get_tokenizer()
  
    prompt ="""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, 
    detailed, and polite answers to the user's questions. USER: hello. ASSISTANT:"""

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
        model.base_model.device
    )
    # prefilling 
    bs,seq_len =input_ids.shape
    assert bs == 1
    if config.is_first_stage:
        if not config.is_last_stage:
            hidden_states = model.prefilling(input_ids=input_ids, inputs_embeds=None,temperature=args.temperature)
            print("send to next stage", config.next_rank)
            send_tensor = hidden_states.cpu()
            dist.send(tensor= send_tensor, dst= config.next_rank)
        else:
            medusa_logits, logits =  model.prefilling(input_ids=input_ids, inputs_embeds=None,temperature=args.temperature)
            print("medusa_logits{} logits{}".format(medusa_logits.shape, logits.shape))
    else:
        recv_tensor = torch.zeros( bs,  seq_len,  config.hidden_size, dtype=torch.float16)
        print( "receive from previous stage", config.pre_rank)
        dist.recv(tensor=recv_tensor, src= config.pre_rank) 
        hidden_states = recv_tensor.to("cuda")
        if not config.is_last_stage:
            hidden_states = model.prefilling(input_ids= None, inputs_embeds=hidden_states, temperature=args.temperature)
            print("send to next stage", config.next_rank)
            send_tensor = hidden_states.cpu()
            dist.send(tensor= send_tensor, dst= config.next_rank)
        else:
            medusa_logits, logits =model.prefilling(input_ids= None, inputs_embeds=hidden_states, temperature=args.temperature)
            print("medusa_logits{} logits{}".format(medusa_logits.shape, logits.shape))
    # 将medusa_logits logits 从最后一个stage 给第一个stage
    if config.total_stage > 1:
        if config.is_first_stage:
            recv_tensor1 = torch.zeros(  config.medusa_num_heads, bs ,seq_len,  config.vocab_size, dtype=torch.float16)
            recv_tensor2 = torch.zeros( bs ,seq_len,  config.vocab_size, dtype=torch.float16)
            dist.recv(tensor=recv_tensor1, src= config.total_stage-1) 
            dist.recv(tensor=recv_tensor2, src= config.total_stage-1) 
            medusa_logits = recv_tensor1.to("cuda")
            logits = recv_tensor2.to("cuda")
            print("medusa_logits{} logits{}".format(medusa_logits.shape, logits.shape))
        if config.is_last_stage:
            send_tensor1 = medusa_logits.cpu()
            send_tensor2 = logits.cpu()
            dist.send(tensor= send_tensor1, dst=  0)
            dist.send(tensor= send_tensor2, dst=  0)
    # decoding stage 
    new_token = 0
    input_len = input_ids.shape[1]
    # generate_candidates
    if config.is_first_stage:
        candidates, tree_candidates  = model.generate_candidates(
                medusa_logits, 
                logits, 
        )
        print("candidates{} tree_candidates{}".format(candidates.shape, tree_candidates.shape))
    # tree decoding
    # TODO:tree_decoding所有stage 都需要 input_ids, 需要广播input_ids, 每一次decoding都会修改inputs_ids
    if config.is_first_stage:
        if not config.is_last_stage:
            hidden_states = model.tree_decoding(
                tree_candidates = tree_candidates,
                tree_candidates_embeds = None,
                input_ids = input_ids
            )
            send_tensor = hidden_states.cpu()
            dist.send(tensor= send_tensor, dst= config.next_rank)
            print("send to next stage", config.next_rank)
        else:
            medusa_logits, logits  = model.tree_decoding(
                tree_candidates = tree_candidates,
                tree_candidates_embeds = None,
                input_ids = input_ids
            )
    else:
        # TODO:candidatestorch.Size([42, 5]) tree_candidates torch.Size([1, 64]) 可能和不同medusa choice 有关
        recv_tensor = torch.zeros( 1,  64,  config.hidden_size, dtype=torch.float16)
        dist.recv(tensor=recv_tensor, src= config.pre_rank) 
        print( "receive from previous stage", config.pre_rank)
        hidden_states = recv_tensor.to("cuda")
        if not config.is_last_stage:
            hidden_states = model.tree_decoding(
                tree_candidates = None,
                tree_candidates_embeds = hidden_states,
                input_ids = input_ids
            )
            send_tensor = hidden_states.cpu()
            dist.send(tensor= send_tensor, dst= config.next_rank)
            print("send to next stage", config.next_rank)
        else:
            medusa_logits, logits  = model.tree_decoding(
                tree_candidates = None,
                tree_candidates_embeds = hidden_states,
                input_ids = input_ids
            )
    if config.total_stage > 1:
        if config.is_first_stage:
            recv_tensor1 = torch.zeros(  config.medusa_num_heads,  42, config.medusa_num_heads  ,  config.vocab_size, dtype=torch.float16)
            recv_tensor2 = torch.zeros(  42 , config.medusa_num_heads,  config.vocab_size, dtype=torch.float16)
            dist.recv(tensor=recv_tensor1, src= config.total_stage-1) 
            dist.recv(tensor=recv_tensor2, src= config.total_stage-1) 
            medusa_logits = recv_tensor1.to("cuda")
            logits = recv_tensor2.to("cuda")
            print("after tree decoding medusa_logits{} logits{}".format(medusa_logits.shape, logits.shape))
        if config.is_last_stage:
            send_tensor1 = medusa_logits.cpu()
            send_tensor2 = logits.cpu()
            dist.send(tensor= send_tensor1, dst=  0)
            dist.send(tensor= send_tensor2, dst=  0)
    #  evaluate_posterior and update_inference_inputs
    if config.is_first_stage:
        best_candidate, accept_length = model.evaluate_posterior(logits,
                           candidates)
        print("best_candidate {} accept_length {}".format(best_candidate, accept_length ))
        input_ids, logits, medusa_logits, new_token = model.update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            logits,
            medusa_logits,
            new_token,
        ) #TODO: 其他stage的kv cache 需要根据select_indices更新自己的kvcache
        print(model.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                ))







        



    # output_stream = model.medusa_generate(
    #     input_ids,
    #     temperature=args.temperature,
    #     max_steps=args.max_steps,
    # )

    # # 从generator中获取输出结果
    # pre = 0
    # for outputs in output_stream:
    #     output_text = outputs["text"]
    #     output_text = output_text.strip().split(" ")
    #     now = len(output_text) - 1
    #     if now > pre:
    #         print(" ".join(output_text[pre:now]), end=" ", flush=True)
    #         pre = now
    # print(" ".join(output_text[pre:]), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--world', default=2, type=int)
    parser.add_argument('--config_file',   type=str, default="medusa/pipeline_model/config.json", help="Config file path")
    parser.add_argument(
        "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=512)
    args = parser.parse_args()
        #TODO: 把generate的参数加到config里面

    main(args)