import torch
from jupiter.core.decoding_communication import CommunicationHandler
import time
from  tasks.medusa_llama.outline_decoding_controller  import get_controller   #[modified]


class DecodingPipeline():
    def __init__(self,  stage_model, config, args):
        self.config = config
        self.args = args
        self.stage = config.stage
        self.next_rank = config.next_rank
        self.pre_rank = config.pre_rank
        self.total_stage = config.total_stage
        self.stage_model = stage_model
        self.comm_handler = CommunicationHandler(config)
        
    def tree_decoding_send(self, tensor, point_id):
        assert self.stage != self.total_stage -1
        self.comm_handler.send(tensor, tag = self.comm_handler.tensor_tag["tree_decoding"], point_id=point_id)
        
    def tree_decoding_recv(self):
        assert self.stage != 0
        return self.comm_handler.recv(tag = self.comm_handler.tensor_tag["tree_decoding"])
    
    def tree_candidates_send(self, tensor, point_id):
        assert self.comm_handler.if_last_rank
        self.comm_handler.send(tensor, tag = self.comm_handler.tensor_tag["tree_candidates"], point_id=point_id)
        
    def tree_candidates_recv(self):
        assert self.comm_handler.if_first_rank
        return self.comm_handler.recv(tag = self.comm_handler.tensor_tag["tree_candidates"])
    
    def new_token_send(self, tensor, point_id):
        assert self.comm_handler.if_last_rank
        self.comm_handler.send(tensor,tag = self.comm_handler.tensor_tag["new_token"], point_id=point_id)
    def new_token_recv(self):
        assert not self.comm_handler.if_last_rank
        return self.comm_handler.recv(self.comm_handler.tensor_tag["new_token"])
    def jupiter_decoding_pipeline(self):
        #use point_id to select point_kv_cache  in forward computation or modify point_kv_cache based on selectd token
        extra_kwargs = {
                'is_point': True,
                'point_id': 0,
                } 
        for idx in range(  self.config.max_steps):
            if  get_controller().check_finish():
                break #TODO: 停不下来 
            else:
                print("not finish ... ")
                print("=========================================\n")
                get_controller().get_output( ) 
                print("=========================================\n")
            # step 1: get request and generate_candidates
            new_token=0# no use
            if self.config.is_last_stage:
                request = get_controller().get_request()
                candidates, tree_candidates = self.stage_model.generate_candidates(
                    request["medusa_logits"], 
                    request["logits"], 
                )
                input_ids = get_controller().get_input_ids( request["point_id"] )
                self.tree_candidates_send(tree_candidates, request["point_id"] )
            if self.config.is_first_stage:
                tree_candidates, point_id = self.tree_candidates_recv()
                input_ids = get_controller().get_input_ids( point_id)
            # Step 2: tree decoding
            if self.config.is_first_stage:
                if  self.config.is_last_stage:
                    raise NotImplementedError("暂不支持单机推理")
                extra_kwargs["point_id"]=point_id
                hidden_states = self.stage_model.tree_decoding(
                    tree_candidates = tree_candidates,
                    tree_candidates_embeds = None,
                    input_ids = input_ids,
                    **extra_kwargs  # 传递额外参数
                )
                self.tree_decoding_send(hidden_states,point_id)
            else:
                hidden_states, point_id = self.tree_decoding_recv()   
                # print("stage recv hidden_states: ", hidden_states[:,-1,-10:])
                input_ids = get_controller().get_input_ids(point_id)
                extra_kwargs["point_id"]=point_id
                if not self.config.is_last_stage:
                    hidden_states = self.stage_model.tree_decoding(
                    tree_candidates = None,
                    tree_candidates_embeds = hidden_states,
                    input_ids = input_ids,
                    **extra_kwargs  # 传递额外参数
                    )
                    self.tree_decoding_send(hidden_states,point_id)
                else:
                    medusa_logits, logits  = self.stage_model.tree_decoding(
                    tree_candidates = None,
                    tree_candidates_embeds = hidden_states,
                    input_ids = input_ids,
                    **extra_kwargs  # 传递额外参数
                    )
            # Step 3: 选择出best candidate作为采样的结果，同时更新inputs_ids
            if self.config.is_last_stage:   
                best_candidate, accept_length = self.stage_model.evaluate_posterior(logits,
                            candidates)
                extra_kwargs["point_id"]=point_id
                input_ids, logits, medusa_logits, new_token , select_indices= self.stage_model.update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                logits,
                medusa_logits,
                new_token,
                **extra_kwargs
                )  
                new_input_ids = input_ids[:,  -select_indices.shape[0]:]
                # update input_ids request
                get_controller().update_input_ids(input_ids,point_id)
                input_len = get_controller().get_input_len(point_id)
                if self.stage_model.tokenizer.eos_token_id in input_ids[0, input_len:]:
                    # set point finish
                    get_controller().set_point_finish(point_id)
                    self.comm_handler.stop_helper_threads()
                    print("Point: {} finish".format(point_id))
                else:
                    # update request
                    get_controller().add_request(medusa_logits,logits,point_id)
                #TODO:判断是否结束
            # Step 4: Step 3 中self.stage_model.update_inference_inputs 调用更新了最后一个stage的kvcache,现在同步decoding的结果，并更新其他stage的kv cache和inputs_ids
            if  self.config.is_last_stage: 
                new_token_len =  torch.tensor(select_indices.shape)
                select_indices_and_new_inputs_ids = torch.cat((new_token_len.unsqueeze(0), select_indices.unsqueeze(0).cpu(), new_input_ids.cpu()), dim=1)
                self.new_token_send(select_indices_and_new_inputs_ids,point_id)
            else:
                select_indices_and_new_inputs_ids, point_id = self.new_token_recv()
                new_token_len = select_indices_and_new_inputs_ids[0,0].item()
                select_indices = select_indices_and_new_inputs_ids[:,1:new_token_len+1].view(-1)   
                new_input_ids = select_indices_and_new_inputs_ids[:, new_token_len+1:2*new_token_len+1] 
                extra_kwargs["point_id"]=point_id
                self.stage_model.update_kv_cache(input_ids,select_indices,**extra_kwargs)
                input_ids = torch.cat([input_ids, new_input_ids], dim=-1    )#必须在update_kv_cache之后执行
                # update input_ids
                get_controller().update_input_ids(input_ids,point_id)
                input_len = get_controller().get_input_len(point_id)
                if self.stage_model.tokenizer.eos_token_id in input_ids[0, input_len:]:
                    # set point finish
                    get_controller().set_point_finish(point_id)
                    self.comm_handler.stop_helper_threads()
                    print("Point: {} finish".format(point_id))
