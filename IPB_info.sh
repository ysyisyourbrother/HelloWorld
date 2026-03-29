# GStreamer + PyGObject（gi）装在系统 Python 上；不要用 conda 里的 python3，否则会 ModuleNotFoundError: gi
/usr/bin/python3 test_rtsp/rtsp_record_IPB_info.py \
  -u "rtsp://admin:smc123456@192.168.123.98:554/stream1" \
  --tcp \
  --latency 200 \
  --relaxed-caps \
  --verbose-bus \
  --xvfb

# 命令行自检（应看到 fakesink 收到 buffer；若只有 rtpsession stats 无 downstream caps，多半是 depay/caps 未协商）:
# gst-launch-1.0 -v rtspsrc location='...' protocols=tcp latency=200 ! rtph264depay ! h264parse ! video/x-h264,stream-format=byte-stream ! fakesink sync=false num-buffers=300