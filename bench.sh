# nohup python launch_benchmark_venus.py --config configs/config.json > bench.out 2>&1 & echo $! > bench.pid
nohup python launch_benchmark_symphony.py \
    --resume benchmark_results/symphony/benchmark_Video-MME_short_20260319_153714.json \
    --config configs/symconfig.json > symphony_short_resume.out 2>&1 & echo $! > symphony_short_resume.pid