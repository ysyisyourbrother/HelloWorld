1. 模型切分
```bash
python tools/model_partition.py --config_file tasks/medusa_llama/config/vicuna_7b_config.json
```

2. 运行pipeline inference 
```bash
python pipeline_inference.py  --world 4 --rank xxx --config_file  tasks/medusa_llama/config/vicuna_7b_config.json  --load_in_8bit
```