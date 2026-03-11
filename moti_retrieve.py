from src.config import Config
from src.venus_system_motivation import VenusSystemMoti

# 使用 config_moti_retrieve.json 初始化

moti = VenusSystemMoti(config=Config(config_path="configs/config_moti_retrieve.json"))

# 指定 Video-MME 视频执行流程
result = moti.run_video_flow(
    video_path="local_datasets/Video-MME/data/44ivpEIcBhE.mp4",
    video_id="44ivpEIcBhE",
    questions=[
            {
                "question": "Which instrument is the performer on the stage holding in the video?", 
                "options": [
                    "A. Trumpet.",
                    "B. Saxophone.",
                    "C. Violin.",
                    "D. Guitar."],
                "answer": "B"
            }
        ],
    dataset_name="Video-MME",
    subset="short",
)

# 或单图推理
# result = moti.run_image("path/to/image.jpg", "描述这张图片")