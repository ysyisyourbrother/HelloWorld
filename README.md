# Jupiter

```shell
CUDA_VISIBLE_DEVICES=1 python    main_new.py
CUDA_VISIBLE_DEVICES=1 python    main_new.py --load_in_8bit
CUDA_VISIBLE_DEVICES=1 python    main_new.py --load_in_4bit
```

将模型合并
model: `MedusaLlamaForCausalLM`

```
MedusaLlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096, padding_idx=0)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
  (medusa_head): ModuleList(
    (0-4): 5 x Sequential(
      (0): ResBlock(
        (linear): Linear(in_features=4096, out_features=4096, bias=True)
        (act): SiLU()
      )
      (1): Linear(in_features=4096, out_features=32000, bias=False)
    )
  )
)

```

内存占用: 设置 `torch_dtype=bfloat16`

| data type | model memory | max memory |
| :-------- | :----------: | :--------: |
| bp16      |   14294 MB   |  14620 MB  |
| int8      |   7519 MB    |  7895 MB   |
| int4      |   4434 MB    |  4923 MB   |

## Pipeline

首先运行`weight_split.py` 划分模型参数,保存到新的文件
然后 `pipe_main` 从保存路径中读取权重

参数意义:

config_file: `./config.json`

- 模型参数( `medusa_head_path`和 `base_model_name_or_path`)
- pipeline 参数: `stage_num_hidden_layers_list`
- 分布式参数: `init_method` `distributed_backend`

```
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 2 --rank 0

CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 2 --rank 1

```

修改 config_file 参数 `stage_num_hidden_layers_list` 和 `--world` 参数,支持不同数量 stage
(这样只需要一个 `config.json`文件)

例如：划分 3 个 stage, 修改`stage_num_hidden_layers_list` = [10,10,12]

```
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 3 --rank 0
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 3 --rank 1
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 3 --rank 2

```

### Quantization

**Note**:

- 如果设备选择`cpu`: 半精度数量类型必须为`bfloat16 `
- 如果使用`bitsandbytes`量化，不支持 cpu

```
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 2 --rank 0 --load_in_8bit

CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 2 --rank 1 --load_in_8bit

```

[8,8,8,8]
data type: bp16
| | stage 0 <br> (max) | stage 0 <br> (model)| stage 1 <br> (max) | stage 1 <br> (model)| stage 2 <br> (max) | stage 2 <br> (model)| stage 3 <br> (max) | stage 3 <br> (model)|
| :-------- | :-----: | ------: |:-----: | ------: | :-----: | ------: |:-----: | ------: |
| bp16 |3392|3346|3142|3096|3142|3096|4986|4756|
| int8 | 1957|1957|1707|1707|1707|1707|2791|2665|
| int4 |1266|1220|1014|969|1014|969|1923| 1617|

[8,9,9,6]
data type: bp16
| | stage 0 <br> (max) | stage 0 <br> (model)| stage 1 <br> (max) | stage 1 <br> (model)| stage 2 <br> (max) | stage 2 <br> (model)| stage 3 <br> (max) | stage 3 <br> (model)|
| :-------- | :-----: | ------: |:-----: | ------: | :-----: | ------: |:-----: | ------: |
| bp16 |3392|3346|3533|3529|3533|3529|4204|3982|
| int8 | 1957|1957|1904|1904|1904|1904|2388|2270|
| int4 |1266|1220|1129|1080|1129|1080|1694|1395 |
