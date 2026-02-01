#!/bin/bash

# 使用 nohup 调用当前目录下的 HuggingFace 下载脚本
# 数据集：lmms-lab/Video-MME
# 日志输出到当前目录的 nohup_video_mme.out

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

nohup python "$SCRIPT_DIR/download.py" \
  --repo lmms-lab/egoschema \
  --type datasets \
  > "$SCRIPT_DIR/nohup_EgoSchema.out" 2>&1 &

PID=$!
echo "$PID" > "$SCRIPT_DIR/EgoSchema.pid"

echo "已使用 nohup 后台启动 lmms-lab/egoschema 下载, PID: $PID, 日志文件: $SCRIPT_DIR/nohup_EgoSchema.out"