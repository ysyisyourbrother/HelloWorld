"""从视频中提取单帧并保存的示例脚本"""

from src.video_utils.about_frame import extract_frame_by_index, extract_frame_by_time

if __name__ == "__main__":
    video_path = "local_datasets/Video-MME/data/44ivpEIcBhE.mp4"
    output_path = "motivation_results/frames/frame.jpg"
    extract_frame_by_time(video_path, output_path, 24.0)
    print(f"已保存到 {output_path}")
