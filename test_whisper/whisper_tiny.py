import argparse
from pathlib import Path
from time import perf_counter

import torch
from transformers import pipeline


def format_srt_timestamp(seconds):
    if seconds is None:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_args():
    parser = argparse.ArgumentParser(description="Run ASR with local whisper-tiny model.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/mnt/share/cache/models/whisper-tiny",
        help="Local whisper model directory.",
    )
    parser.add_argument(
        "--input-video",
        type=str,
        default="demo/assets/fFjv93ACGo8.mp4",
        help="Input video file path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_whisper/output",
        help="Directory to save ASR text result.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="fFjv93ACGo8_asr.srt",
        help="Output subtitle file name (.srt).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_dir = Path(args.model_dir)
    input_video = Path(args.input_video)
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_name

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    output_dir.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    asr_pipe = pipeline(
        task="automatic-speech-recognition",
        model=str(model_dir),
        tokenizer=str(model_dir),
        feature_extractor=str(model_dir),
        torch_dtype=dtype,
        device=device,
    )

    start_process_time = perf_counter()
    result = asr_pipe(str(input_video), return_timestamps=True)
    elapsed_seconds = perf_counter() - start_process_time
    chunks = result.get("chunks", [])

    srt_lines = []
    offset_seconds = 0.0
    prev_raw_start = None
    prev_global_end = 0.0
    subtitle_index = 1

    for chunk in chunks:
        timestamp = chunk.get("timestamp", (None, None))
        start_time, end_time = timestamp
        text = chunk.get("text", "").strip()
        if not text:
            continue
        if start_time is None:
            start_time = 0.0
        if end_time is None:
            end_time = start_time

        # Some whisper chunks may reset local timestamps to 0; stitch them into one timeline.
        if prev_raw_start is not None and start_time + 0.2 < prev_raw_start:
            offset_seconds = prev_global_end

        global_start = start_time + offset_seconds
        global_end = end_time + offset_seconds
        if global_end < global_start:
            global_end = global_start

        srt_lines.append(str(subtitle_index))
        srt_lines.append(
            f"{format_srt_timestamp(global_start)} --> {format_srt_timestamp(global_end)}"
        )
        srt_lines.append(text)
        srt_lines.append("")
        subtitle_index += 1
        prev_raw_start = start_time
        prev_global_end = global_end

    output_content = "\n".join(srt_lines).rstrip() + "\n"
    output_path.write_text(output_content, encoding="utf-8")

    print(f"ASR done. Output saved to: {output_path}")
    print(f"音频处理耗时: {elapsed_seconds:.2f} 秒")
