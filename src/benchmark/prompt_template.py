

def build_rag_prompt_with_frames(video_time: float, 
                     num_selected_frame: int, 
                     question: str, 
                     options: list) -> str:
    """
    根据筛选的帧、问题和选项构造 RAG 提示词文本。
    
    Args:
        video_time: 视频时长（秒）
        num_selected_frame: 筛选出的帧数目
        question: 问题文本
        options: 选项列表，如 ["A. xxx", "B. xxx", ...]
    
    Returns:
        构造好的提示词字符串
    """
    prompt = f"The video lasts for {video_time:.2f} seconds, and {num_selected_frame} frames are selected from it."
    prompt += "\nPlease answer the following questions related to this video."
    prompt += " Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option. Question: "
    prompt += question + "\n" + " ".join(options) + "\nThe best answer is:"
    return prompt

def build_rag_prompt_with_clips(
    video_time: float,
    num_selected_clips: int,
    question: str,
    options: list,
) -> str:
    """
    clip 检索模式下：说明原视频时长、选中的 GOP 片段数量，并强调后续图像序列为
    这些片段中的**全部帧**（与按 top-k 单帧抽样的 frame 模板区分）。
    """
    prompt = (
        f"The video lasts for {video_time:.2f} seconds. "
        f"{num_selected_clips} clip(s) were retrieved from it; "
        "the following images are **all frames** from these clips, in temporal order."
    )
    prompt += "\nPlease answer the following questions related to this video."
    prompt += (
        " Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option. Question: "
    )
    prompt += question + "\n" + " ".join(options) + "\nThe best answer is:"
    return prompt