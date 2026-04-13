# gRPC 服务说明

## 版本兼容

- 生成的 `query_service_pb2.py` 依赖 **protobuf >= 6.31.1**。若运行时报 `TypeError: Couldn't parse file content!`，请执行：`pip install -U 'protobuf>=6.31.1'`。
- 修改 `.proto` 后需重新执行生成脚本，使 _pb2 与当前 proto 定义一致。

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

## QueryRequest 记忆载荷（帧 pickle / clip 二选一）

- **旧客户端**：只填 `memory_results`（pickle 帧列表），字段 `5–7` 留空，与历史行为一致。
- **新客户端（省带宽）**：填 `memory_clip_inputs`（每段 `MemoryClipInput.clip_payload` 为完整 MP4 等容器字节）或填 `memory_clip_uris`；`memory_results` 必须为空。详见 `proto/query_service.proto` 文件头注释中的互斥与错误语义。
- **服务端**：若 `memory_results` 与 clip 字段同时非空，应返回明确 `error`，勿静默取舍。

## 服务接口

### 一元RPC (Query)
- 用于单轮对话
- 客户端发送一个请求，服务器返回一个响应

## 配置

在 `configs/config.json` 中配置：

```json
{
  "api_server_e": {
    "cloud_server_url": "localhost:9000"
  },
  "api_server_c": {
    "host": "0.0.0.0",
    "port": 9000
  }
}
```
