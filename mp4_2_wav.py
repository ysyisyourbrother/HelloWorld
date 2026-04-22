import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将指定的 MP4 文件音频提取并转换为 WAV。"
    )
    parser.add_argument(
        "input_mp4",
        type=Path,
        default="demo/assets/fFjv93ACGo8.mp4",
        help="输入 MP4 文件路径",
    )
    parser.add_argument(
        "-o",
        "--output-wav",
        type=Path,
        default=None,
        help="输出 WAV 文件路径（默认与输入同名，仅后缀改为 .wav）",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="输出音频采样率，默认 16000",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="输出声道数，默认 1（单声道）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("未找到 ffmpeg，请先安装并确保其在 PATH 中。")

    input_mp4 = args.input_mp4.expanduser().resolve()
    if not input_mp4.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_mp4}")
    if input_mp4.suffix.lower() != ".mp4":
        raise ValueError(f"输入文件不是 .mp4: {input_mp4}")

    output_wav = (
        args.output_wav.expanduser().resolve()
        if args.output_wav is not None
        else input_mp4.with_suffix(".wav")
    )
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_mp4),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(args.sample_rate),
        "-ac",
        str(args.channels),
        str(output_wav),
    ]

    print("执行命令:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"转换完成: {output_wav}")
