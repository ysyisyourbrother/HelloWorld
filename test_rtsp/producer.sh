/usr/bin/python3 test_rtsp/rtsp_selective_decode_producer_gst.py \
  -u 'rtsp://admin:smc123456@192.168.123.98:554/stream1' \
  --tcp \
  --byte-budget 50000 \
  --queue-size 8 \
  --bind 127.0.0.1 --port 50050 --authkey rtsp \
  --xvfb