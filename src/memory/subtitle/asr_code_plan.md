# Goal
我希望在"src/memory/subtitle/asr.py"写代码，这个代码的内容是进行语音转文字识别。
# Code requirements
## Style
代码的风格上有以下要求：
1. "src/memory/frame/frame_vectorizer.py"和该模块是并列的模块，都会被"src/memory/memory_manager.py"作为一个模块调用，所以代码风格应当有一定相似度。（虽然要添加到memory_manager中，但是目前只用写好该模块的功能就行，先不用改Memory的编排）
## Incremental requirements for other code files
在写这个代码文件之前，请你先完成以下的事项：
1. 给src/config.py添加asr_config, 你可以根据实现的需要添加一些必要的config参数，但是切忌冗余。

## Requiremented classes and their features
我需要以下的类：
1. SymASRBase:
    - 这个类作为基类被SymASR和SymStreamASR继承。
    - 初始化如同其他所有组件一样，仅传入可选的config进行初始化，如果没有传入，就自己实例化一个config
    - 初始化的时候，需要有一个config参数来选择whisper模型的版本, 该参数在config中默认为tiny。还要有一个属性标识这是流式的还是整段音频理解，来自config.audio_stream_as_chunks。需要有一个config属性标识模型路径。
    - 除了初始化暂时不要写任何函数

2. SymASR: 
    - 这个类继承SymASRBase的，初始化函数也仅有 config
    - 这个类的模型加载，需要判断模型文件夹的合法性, whisper模型的版本是不是符合要求。
    - 这个类的作用是对完整的音频（即audio_type为False的AudioData）返回识别到的文本和其起止时间。参考"test_whisper/whisper_tiny.py"
    - 起止时间需要校验, 如果遇到了起始时间变成0的情况需要更改。参考"test_whisper/whisper_tiny.py"

3. SymStreamASR:
    - 这个类继承SymASRBase的，初始化函数也仅有 config
    - 这个类的一些初始化属性与audio_config有重复，比如chunk_duration_sec
    - 这个类的模型加载，需要判断模型文件夹的合法性, whisper模型的版本是不是符合要求, 以及是不是-ct2结尾, 如果从文件夹名字看出了whisper模型的版本符合了要求但是不是-ct2结尾，需要看同文件夹下有没有一个同名的-ct2结尾的模型文件夹并尝试从里面加载模型。
    - 这个类最主要的逻辑是流式地对音频进行文本识别，参考"test_whisper_streaming/whisper_online.py"中faster-whisper的online版本进行实现，你也可以对照这个文件中的传入参数，添加一些必要的参数到config中，比如--buffer_trimming默认为sentence，但是切忌重复冗余。
    - 从函数功能性上来讲，他会像"test_whisper_streaming/whisper_online.py"中faster-whisper 的 online 版本一样，每次返回已经确认的稳定文本和起止时间给memory模块。
