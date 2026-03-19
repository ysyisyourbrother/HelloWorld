import ffmpeg

video_path = "local_datasets/Video-MME/data/44ivpEIcBhE.mp4"
output_residual = "residuals_vis.mp4"

print("正在生成残差可视化视频...")
print("画面越黑，说明压缩效果越好（差异小）；画面越亮/噪点越多，说明变化大或压缩损失大。")

try:
    # 使用 codecview 滤镜显示预测误差 (ep = error prediction / residual)
    # 注意：这依然依赖 FFmpeg 的 debug 编译选项，如果不行，可以用简单的帧差法替代
    (
        ffmpeg
        .input(video_path)
        .filter('codecview', vis='ep') # ep 显示预测误差（残差）
        .output(output_residual, vcodec='libx264', pix_fmt='yuv420p', crf=18) # 高画质保存
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )
    print(f"成功！请播放 {output_residual}。")
    print("观察：静止背景应该是纯黑的，只有运动物体边缘会有灰白色线条。")

except ffmpeg.Error as e:
    print("当前 FFmpeg 可能不支持 codecview 滤镜。")
    print("尝试使用纯 Python 计算帧差作为替代方案...")
    
    # 备选方案：用 OpenCV 计算简单的帧差 (模拟残差概念)
    import cv2
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('simple_diff.mp4', fourcc, 25.0, (640, 360))
    
    ret, prev_frame = cap.read()
    if not ret:
        print("无法读取视频")
    else:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 计算绝对差值
            diff = cv2.absdiff(prev_gray, curr_gray)
            
            # 阈值化，让细微噪声变黑，突出明显变化
            _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
            
            # 转回彩色以便保存
            diff_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            out.write(diff_color)
            
            prev_gray = curr_gray
            count += 1
            if count % 100 == 0: print(f"处理了 {count} 帧...")
        
        cap.release()
        out.release()
        print("简单帧差视频已生成：simple_diff.mp4")