# gRPC 服务说明

## 生成 gRPC Python 代码

在使用 gRPC 服务之前，需要先从 proto 文件生成 Python 代码：

```bash
# 方法1: 使用提供的脚本
./generate_proto.sh

# 方法2: 手动运行
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/query_service.proto
```

生成的文件：
- `proto/query_service_pb2.py` - 消息定义
- `proto/query_service_pb2_grpc.py` - 服务定义

## 服务接口

### 一元RPC (Query)
- 用于单轮对话
- 客户端发送一个请求，服务器返回一个响应

### 服务器端流式RPC (QueryStream) - 预留
- 用于多轮对话
- 客户端发送一个请求，服务器返回多个响应流

### 双向流式RPC (QueryBidiStream) - 预留
- 用于多轮对话
- 客户端和服务器都可以发送多个消息

## 配置

在 `configs/config.json` 中配置：

```json
{
  "api_server_e": {
    "cloud_server_url": "localhost:9000",
    "dialog_mode": "single",  // "single" 或 "multi"
    "communication_mode": "unary"  // "unary", "server_stream", "bidi_stream"
  },
  "api_server_c": {
    "host": "0.0.0.0",
    "port": 9000
  }
}
```
