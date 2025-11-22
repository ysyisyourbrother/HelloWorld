# StreamInput 模块使用指南

## 1. 概述

StreamInput 模块是一个用于从视频源提取帧数据的核心组件，支持实时摄像头流和本地视频文件两种模式。该模块负责高效地读取视频帧，并将其转换为标准格式的张量数据，供后续处理模块（如FrameVectorizer）使用。

## 2. 核心类结构

### 2.1 FrameData 类

`FrameData` 是一个数据类，用于封装从视频源提取的帧数据及其元信息。

#### 主要属性

| 属性名 | 类型 | 描述 |
|--------|------|------|
| `frame_tensor` | `torch.Tensor` | 帧数据张量，格式为(C, H, W)，值范围[0, 1]，数据类型为float32 |
| `timestamp` | `float` | 帧的时间戳 |
| `frame_id` | `int` | 帧的唯一标识符<br>- 视频文件模式：表示原始视频帧序号<br>- 摄像头模式：基于时间戳生成的ID |
| `source_path` | `str` | 视频来源路径<br>- 视频文件模式：文件路径<br>- 摄像头模式："camera" |
| `total_frames` | `Optional[int]` | 视频总帧数，仅在视频文件模式下有值 |
| `video_fps` | `Optional[float]` | 视频帧率，仅在视频文件模式下有值 |

## 3. StreamInput 类

### 3.1 初始化

```python
def __init__(self, config=None):
    """
    初始化StreamInput模块
    
    Args:
        config (Config): 配置对象实例，如果为None则使用默认配置
    """
```

### 3.2 核心方法

#### 3.2.1 启动帧提取

```python
def start(self):
    """启动帧提取线程"""
```

- 功能：启动后台线程开始从视频源提取帧
- 自动初始化视频源（摄像头或文件）
- 创建并启动名为"StreamInput-Extractor"的守护线程

#### 3.2.2 获取单帧数据

```python
def get_frame(self):
    """获取一帧数据"""
    
    # 返回值：FrameData对象或None（队列为空时）
```

- 功能：从内部帧队列中获取一帧数据
- 使用timeout=1.0秒的非阻塞方式获取数据
- 适用于单帧处理场景

#### 3.2.3 获取帧队列

```python
def get_frame_queue(self):
    """获取帧队列供其他模块使用"""
    
    # 返回值：queue.Queue对象
```

- 功能：返回内部帧队列的引用
- 主要用于其他模块（如FrameVectorizer）直接访问帧队列
- 这是StreamInput与FrameVectorizer交互的主要接口

#### 3.2.4 获取视频信息

```python
def get_video_info(self):
    """获取视频信息"""
    
    # 返回值：包含视频信息的字典
```

- 功能：返回当前视频源的信息
- 视频文件模式返回：total_frames、video_fps、target_fps、duration、source
- 摄像头模式返回：target_fps、source

#### 3.2.5 停止帧提取

```python
def stop(self):
    """停止帧提取线程"""
```

- 功能：安全停止帧提取线程
- 释放视频捕获资源
- 等待线程终止（最多5秒）

## 4. 配置选项

StreamInput通过Config对象进行配置，主要配置项包括：

| 配置项 | 描述 | 默认值 |
|--------|------|--------|
| `stream_fps` | 目标帧率，决定从视频源提取帧的频率 | 由Config类定义 |
| `stream_video_source` | 视频来源类型，可选值："camera"或"file" | 由Config类定义 |
| `stream_video_file_path` | 视频文件路径，当stream_video_source="file"时使用 | 由Config类定义 |

## 5. 与FrameVectorizer的交互方式

StreamInput与FrameVectorizer通过队列进行交互，主要步骤如下：

1. StreamInput初始化并启动后，开始向内部帧队列写入FrameData对象
2. FrameVectorizer通过`set_frame_queue()`方法接收StreamInput的帧队列引用
3. FrameVectorizer从队列中获取帧数据进行向量化处理

### 注意事项

- FrameVectorizer期望从队列获取的是字典格式数据，但StreamInput产生的是FrameData对象
- 需要在两者之间进行数据格式转换，确保FrameVectorizer能正确处理数据

## 6. 使用示例

### 6.1 基本使用示例

```python
from src.stream_input import StreamInput
from src.config import Config

# 初始化配置
config = Config()
config.stream_video_source = "file"  # 设置为视频文件模式
config.stream_video_file_path = "demo/assets/cooking_small.mp4"  # 设置视频文件路径

# 创建StreamInput实例
stream_input = StreamInput(config)

# 启动帧提取
stream_input.start()

try:
    # 获取并处理帧
    for i in range(10):  # 处理10帧
        frame_data = stream_input.get_frame()
        if frame_data:
            print(f"获取到帧 {i}: 帧ID={frame_data.frame_id}, 来源={frame_data.source_path}")
            # 访问帧张量数据
            frame_tensor = frame_data.frame_tensor
            print(f"  张量形状: {frame_tensor.shape}")
            
            # 在视频文件模式下，还可以访问视频信息
            if frame_data.total_frames is not None:
                print(f"  视频总帧数: {frame_data.total_frames}")
                print(f"  视频FPS: {frame_data.video_fps}")
        else:
            print("未获取到帧数据")
            break
        
finally:
    # 确保停止
    stream_input.stop()
```

### 6.2 与FrameVectorizer集成使用示例

```python
from src.stream_input import StreamInput
from src.frame_vectorizer import FrameVectorizer
from src.config import Config
import time

# 初始化配置
config = Config()
config.stream_video_source = "file"
config.stream_video_file_path = "demo/assets/cooking_small.mp4"

# 创建并启动StreamInput
stream_input = StreamInput(config)
stream_input.start()

# 创建FrameVectorizer
frame_vectorizer = FrameVectorizer(config)

# 设置FrameVectorizer的帧队列为StreamInput的帧队列
# 注意：这里需要进行格式转换，因为FrameVectorizer期望的是字典格式
# 实际使用时可能需要一个适配层来进行数据格式转换
frame_vectorizer.set_frame_queue(stream_input.get_frame_queue())

# 启动FrameVectorizer
frame_vectorizer.start()

try:
    # 运行一段时间进行测试
    time.sleep(10)
    
finally:
    # 确保清理资源
    frame_vectorizer.stop()
    stream_input.stop()
```

## 7. 最佳实践

1. **资源管理**：始终在使用完毕后调用`stop()`方法释放资源
2. **异常处理**：在生产环境中添加适当的异常处理机制
3. **队列管理**：监控队列大小，避免内存溢出
4. **格式转换**：与其他模块集成时，确保进行必要的数据格式转换
5. **配置优化**：根据具体需求调整`stream_fps`等参数以获得最佳性能

## 8. 性能考虑

- StreamInput使用后台线程提取帧，不阻塞主线程
- 采用批量处理和动态跳帧等优化策略，提高处理效率
- 使用非阻塞队列操作，避免线程阻塞
- 支持动态调整帧率，适应不同的处理需求

## 9. 故障排除

### 常见问题及解决方案

1. **无法打开视频文件**
   - 检查文件路径是否正确
   - 确认视频格式是否受支持
   - 检查文件权限

2. **无法打开摄像头**
   - 确认摄像头是否可用
   - 检查是否被其他程序占用
   - 确保cv2库正确安装

3. **队列满导致丢帧**
   - 降低`stream_fps`参数
   - 增加消费者（如FrameVectorizer）的处理速度
   - 考虑使用更大的队列容量