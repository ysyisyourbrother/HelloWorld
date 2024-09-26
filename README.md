1. 模型切分

   模型划分方式由 config 中`stage_num_hidden_layers_list`决定

```bash
python tools/model_partition.py --config_file tasks/medusa_llama/config/vicuna_7b_config.json
```

2. 运行 pipeline inference

```bash
python pipeline_inference_sot.py  --world 4 --rank xxx  --config_file  tasks/medusa_llama/config/vicuna_7b_config.json  | tee xxx.txt
```
