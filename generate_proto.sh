#!/bin/bash
# 生成 gRPC Python 代码的脚本

# 确保 proto 目录存在
mkdir -p proto

# 生成 Python 代码
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/query_service.proto

echo "gRPC 代码生成完成！"
echo "生成的文件："
echo "  - proto/query_service_pb2.py"
echo "  - proto/query_service_pb2_grpc.py"
