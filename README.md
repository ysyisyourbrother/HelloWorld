# Jupiter

```shell
CUDA_VISIBLE_DEVICES=1 python    main_new.py
```

将模型合并且手动 load 权重
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

## Pipeline

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
这样只需要一个 config 文件

例如：划分 3 个 stage, 修改`stage_num_hidden_layers_list` = [10,10,12]

```
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 3 --rank 0
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 3 --rank 1
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 3 --rank 2

```
