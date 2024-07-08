from medusa.pipeline_model.dis_utils import get_stage_state_dict ,save_state_dict
from medusa.pipeline_model.llama_config import LlamaConfig
import time 
config = LlamaConfig.from_pretrained(f"./config.json") # 修改stage_num_hidden_layers_list 参数
start=time.time()
world = len(config.stage_num_hidden_layers_list)
for rank in range(world ):
    stage_state_dict = get_stage_state_dict(
        config.base_model_name_or_path,
        config.medusa_head_path,
        config.stage_num_hidden_layers_list,
        rank
    )
    save_path = "temp_{}/stage.bin".format( rank)
    save_state_dict(stage_state_dict, save_path)
end = time.time()
print("cost time:", end - start)

