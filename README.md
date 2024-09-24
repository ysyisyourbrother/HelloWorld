1. 模型切分

```bash
python tools/model_partition.py --config_file tasks/medusa_llama/config/vicuna_7b_config.json
```

2. 运行 pipeline inference / pipeline inference with outline_based_decoding

```bash
python pipeline_inference.py  --world 4 --rank xxx --config_file  tasks/medusa_llama/config/vicuna_7b_config.json  --load_in_8bit
```

```bash
python pipeline_inference_sot.py  --world 4 --rank xxx  --config_file  tasks/medusa_llama/config/vicuna_7b_config.json  | tee xxx.txt
```
