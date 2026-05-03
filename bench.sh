# nohup python launch_benchmark_venus.py --config configs/config.json > bench.out 2>&1 & echo $! > bench.pid

# nohup python launch_benchmark_symphony_v1.py \
#     --resume benchmark_results/symphony_v1/benchmark_Video-MME_short_20260319_153714.json \
#     --config configs/symconfig.json > symphony_v1_short_resume.out 2>&1 & echo $! > symphony_v1_short_resume.pid

# nohup python launch_benchmark_symphony_v3.py \
#     --skip-inject \
#     --subset short \
#     --config configs/symconfig_v3.json > "symphony_v3_short(top2_frame).out" 2>&1 & echo $! > symphony_v3_short.pid
# nohup python launch_benchmark_symphony_v3.py \
#     --subset medium \
#     --config configs/symconfig_v3.json > symphony_v3_medium.out 2>&1 & echo $! > symphony_v3_medium.pid
# nohup python launch_benchmark_symphony_v3.py \
#     --subset long \
#     --config configs/symconfig_v3.json > symphony_v3_long.out 2>&1 & echo $! > symphony_v3_long.pid
# 断点续跑 v3 时取消下行注释并填写已有 JSON 路径：

# nohup python launch_benchmark_symphony_v3.py \
#     --resume benchmark_results/symphony_v3/benchmark_Video-MME_short_YYYYMMDD_HHMMSS.json \
#     --config configs/symconfig_v3.json > symphony_v3_short_resume.out 2>&1 & echo $! > symphony_v3_short_resume.pid

# V5
echo "V5"
export PYTHONWARNINGS="ignore::FutureWarning"
export PYTHONWARNINGS="ignore::UserWarning"
export DASHSCOPE_API_KEY="sk-4b8261173870474290a052c60f183316"
export DEEPSEEK_API_KEY="sk-e6f0a114a0534d5b95ff692f0fc0af49"
export ZHIPU_API_KEY="210bd394c54d427c9546e15086da4c33.4AsYXT0R9D7Tiagl"


nohup python launch_benchmark_symphony_v5.py \
    --subset short \
    --config configs/symconfig_v5.json > symphony_v5_short.out 2>&1 & echo $! > symphony_v5_short.pid