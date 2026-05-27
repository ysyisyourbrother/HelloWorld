"""从视频中提取单帧并保存的示例脚本"""

from src.video_utils.about_frame import extract_save_frame_by_index, extract_save_frame_by_time
import os
import glob

if __name__ == "__main__":
    # video_path = "/mnt/share/cache/datasets/Video-MME/data/_cZXyj6rYVg.mp4"
    video_path = "/mnt/share/cache/datasets/Video-MME/data/0ay2Qy3wBe8.mp4"
    # frame_id = list(range(500, 616, 5))
    frame_id = [375,312,298,255,437,]


    # 删除当前目录下所有rgb_frame_*.png文件
    for f in glob.glob("rgb_frame_*.png"):
        try:
            os.remove(f)
            print(f"已删除文件 {f}")
        except Exception as e:
            print(f"删除文件 {f} 时出错: {e}")
            
    for fid in frame_id:
        output_path = f"rgb_frame_{fid}.png"
        
        extract_save_frame_by_index(
            video_path, 
            output_path, fid, 
            # resize=[100]
            )
        print(f"已保存到 {output_path}")
