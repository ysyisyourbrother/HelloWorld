"""从视频中提取单帧并保存的示例脚本"""

from src.video_utils.about_frame import extract_save_frame_by_index, extract_save_frame_by_time

if __name__ == "__main__":
    # video_path = "/mnt/share/cache/datasets/Video-MME/data/_cZXyj6rYVg.mp4"
    video_path = "demo/assets/44ivpEIcBhE.mp4"
    frame_id = list(range(500, 616, 5))

    for fid in frame_id:
        output_path = f"rgb_frame_{fid}.png"
        extract_save_frame_by_index(video_path, output_path, fid, resize=[100])
        print(f"已保存到 {output_path}")
