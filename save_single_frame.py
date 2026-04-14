"""从视频中提取单帧并保存的示例脚本"""

from src.video_utils.about_frame import extract_save_frame_by_index, extract_save_frame_by_time

if __name__ == "__main__":
    video_path = "demo/assets/44ivpEIcBhE.mp4"
    frame_id = 485
    # output_path = f"frame_{frame_id}.png"
    output_path = f"frame.png"
    # extract_save_frame_by_time(video_path, output_path, 24.0)
    extract_save_frame_by_index(video_path, output_path, frame_id)
    print(f"已保存到 {output_path}")
