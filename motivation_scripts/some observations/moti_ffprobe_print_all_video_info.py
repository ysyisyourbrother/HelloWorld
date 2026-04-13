import subprocess
import argparse
from pathlib import Path
from collections import Counter
from typing import Optional

# 常见视频扩展名
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v"}


def get_video_codec(video_path: str) -> Optional[str]:
    """使用 ffprobe 获取视频编码方式，失败返回 None"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True,
        )
        # 输出格式为 "codec_name=xxx"，取等号后的部分
        line = result.stdout.strip()
        if line.startswith("codec_name="):
            return line.split("=", 1)[1].strip()
        return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main():
    parser = argparse.ArgumentParser(description="统计文件夹下所有视频的编码方式")
    parser.add_argument(
        "video_path",
        type=str,
        default="local_datasets/egoschema/videos",
        nargs="?",
        help="视频文件夹路径",
    )
    args = parser.parse_args()

    video_dir = Path(args.video_path)
    if not video_dir.exists():
        print(f"错误：路径不存在 {video_dir}")
        return
    if not video_dir.is_dir():
        print(f"错误：{video_dir} 不是文件夹")
        return

    # 收集所有视频文件（递归子目录）
    video_files = [
        p for p in video_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]

    codec_counter: Counter[str] = Counter()
    failed_count = 0

    for vf in video_files:
        codec = get_video_codec(str(vf))
        if codec:
            codec_counter[codec] += 1
        else:
            failed_count += 1

    # 输出结果
    print(f"1. 总视频个数: {len(video_files)}")
    print(f"\n2. 各编码方式对应的视频个数:")
    for codec, count in codec_counter.most_common():
        print(f"   {codec}: {count}")

    if failed_count > 0:
        print(f"\n（无法获取编码的视频: {failed_count} 个）")


if __name__ == "__main__":
    main()
