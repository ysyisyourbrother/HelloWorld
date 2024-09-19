import torch
from jupiter.core.decoding_communication import CommunicationHandler
import time


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
        
    def tree_decoding_send(self, tensor):
        assert self.stage != self.total_stage -1
        self.comm_handler.send(tensor, tag = self.comm_handler.tensor_tag["tree_decoding"])
        
    def tree_decoding_recv(self):
        assert self.stage != 0
        return self.comm_handler.recv(tag = self.comm_handler.tensor_tag["tree_decoding"])
    
    def tree_candidates_send(self, tensor):
        assert self.comm_handler.if_last_rank
        self.comm_handler.send(tensor, tag = self.comm_handler.tensor_tag["tree_candidates"])
        
    def tree_candidates_recv(self):
        assert self.comm_handler.if_first_rank
        return self.comm_handler.recv(tag = self.comm_handler.tensor_tag["tree_candidates"])
    
    def new_token_send(self, tensor):
        assert self.comm_handler.if_last_rank
        self.comm_handler.send(tensor,tag = self.comm_handler.tensor_tag["new_token"])
    def new_token_recv(self):
        assert not self.comm_handler.if_last_rank
        return self.comm_handler.recv(self.comm_handler.tensor_tag["new_token"])
    
    def decoding_pipeline(self ,input_ids, medusa_logits=None,logits=None):
    # no sot
        input_len = input_ids.shape[1]
        new_token=0
        for idx in range( self.config.max_steps):
            # Step 1 : generate_candidates
            if self.config.is_last_stage:
                assert medusa_logits is not None
                assert logits is not None
                candidates, tree_candidates = self.stage_model.generate_candidates(
                    medusa_logits, 
                    logits, 
                )
                self.tree_candidates_send(tree_candidates)
            if self.config.is_first_stage:
                tree_candidates = self.tree_candidates_recv()
            # # Step 2 tree decoding
            if self.config.is_first_stage:
                if not self.config.is_last_stage:
                    hidden_states = self.stage_model.tree_decoding(
                        tree_candidates = tree_candidates,
                        tree_candidates_embeds = None,
                        input_ids = input_ids
                    )
                    self.tree_decoding_send(hidden_states)
                else:
                    raise NotImplementedError("暂不支持单机推理")
            else:
                # reveive activations from pre_rank
                hidden_states = self.tree_decoding_recv()   
                if not self.config.is_last_stage:
                    hidden_states = self.stage_model.tree_decoding(
                    tree_candidates = None,
                    tree_candidates_embeds = hidden_states,
                    input_ids = input_ids
                )
                    # send activations to next_rank
                    self.tree_decoding_send(hidden_states)
                else:
                    medusa_logits, logits  = self.stage_model.tree_decoding(
                    tree_candidates = None,
                    tree_candidates_embeds = hidden_states,
                    input_ids = input_ids
                )
            # 完成decoding阶段的前向传播推理后，我们在最后一个stage进行evaluate_posterior 和 update_inference_inputs
            # Step 3: 选择出best candidate作为采样的结果，同时更新inputs_ids
            if self.config.is_last_stage:        
                best_candidate, accept_length = self.stage_model.evaluate_posterior(logits,
                            candidates)
                input_ids, logits, medusa_logits, new_token , select_indices= self.stage_model.update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                logits,
                medusa_logits,
                new_token,
                )  
                new_input_ids = input_ids[:,  -select_indices.shape[0]:]
                text = self.stage_model.tokenizer.decode(
                            input_ids[0,  input_len:],
                            skip_special_tokens=True,
                            spaces_between_special_tokens=False,
                            clean_up_tokenization_spaces=True,
                        ) 
            # Step 4: Step 3 中self.stage_model.update_inference_inputs 调用更新了最后一个stage的kvcache,现在同步decoding的结果，并更新其他stage的kv cache和inputs_ids
            if  self.config.is_last_stage: 
                new_token_len =  torch.tensor(select_indices.shape)
                select_indices_and_new_inputs_ids = torch.cat((new_token_len.unsqueeze(0), select_indices.unsqueeze(0).cpu(), new_input_ids.cpu()), dim=1)
                self.new_token_send(select_indices_and_new_inputs_ids)
            else:
                select_indices_and_new_inputs_ids = self.new_token_recv()
                new_token_len = select_indices_and_new_inputs_ids[0,0].item()
                select_indices = select_indices_and_new_inputs_ids[:,1:new_token_len+1].view(-1)   
                new_input_ids = select_indices_and_new_inputs_ids[:, new_token_len+1:2*new_token_len+1] 
                self.stage_model.update_kv_cache(input_ids,select_indices)
                input_ids = torch.cat([input_ids, new_input_ids], dim=-1    )
                text = self.stage_model.tokenizer.decode(
                        input_ids[0, input_len :],
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                        clean_up_tokenization_spaces=True,
                    ) 
            if self.stage_model.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break
        return  text