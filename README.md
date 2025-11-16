# 代码结构
## (按代码功能和模块)
把整个系统按阶段分为：
```text
- (端侧):
    - perception(感知模块, 在线和离线的处理)
        - (视频流存取相关)
        - (视频流CV处理)
        - (视频流编码处理)
    - memory(记忆模块, 离线)
        - (记忆维护)
    - communication(通信模块)
        - (与服务器端通信)
    - ui(图形化前端界面, 可选, JavaScript开发, WebSocket通信)
- (云侧, Simple Realtime Server):
    - reason(推理)
    - communication(通信模块)
        - (与客户端通信)
- (共有模块):
    - model
        - (云端大模型相关)
        - (端侧embedding相关)
- configs
    - (存储配置和超参)
- resource
    - (存储demo视频)
- demo
    - (端到端运行)
```
## 问题
- 用什么结构存储记忆？即使是两级结构(Key frame + Raw video)
- 记忆包括KVcache？那KVCache从哪里来
- 记忆包括语义特征编码？如果用，岂不是要在端侧部署MEM，然后用提取的特征构建“语义索引”
- 单纯的语义特征索引，岂不是只有“更好地在端侧索引”的作用？对云端推理没有任何帮助？
- 考不考虑“端-边-云”？在边端部署SLM，云端部署LLM？推测解码？

## 想法
- 固定镜头的视频, 可不可以由边端执行top-k tokens的选择, 然后交由服务器端执行稀疏attention？
- 服务器端执行的稀疏attention, 其top-k tokens的信息是不是可以反过来指导记忆的存取？