from medusa.pipeline_model.dis_utils import get_stage_state_dict,save_state_dict,get_model_type
from medusa.pipeline_model.llama_config import LlamaConfig
from medusa.pipeline_model.mistral_config import MistralConfig

import time 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="config/vicuna_7b_config.json", help="Config file path.")
args = parser.parse_args()

if get_model_type(args.config_file) == 'vicuna':
    config = LlamaConfig.from_pretrained( args.config_file) # 包含vicuna-7b-v1.3 config和medusa head config的内容
elif get_model_type(args.config_file) == 'zephyr':
    config = MistralConfig.from_pretrained(args.config_file)
else:
    raise NotImplementedError

start=time.time()
world = len(config.stage_num_hidden_layers_list)
for rank in range(world ):
    if 'vicuna' in args.config_file:
        stage_state_dict = get_stage_state_dict(
            config.base_model_name_or_path,
            config.medusa_head_path,
            config.stage_num_hidden_layers_list,
            rank
        )
    elif 'zephyr' in args.config_file:
        stage_state_dict = get_stage_state_dict(
            config.base_model_name_or_path,
            None,
            config.stage_num_hidden_layers_list,
            rank
        )
    save_path = "temp_{}_world_{}_rank_{}/stage.bin".format( get_model_type(args.config_file),world, rank)
    save_state_dict(stage_state_dict, save_path)
end = time.time()
print("cost time:", end - start)

