import argparse
import os

import cv2
from ultralytics import YOLO


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(project_root, "models/yolo26l.pt"),
        help="YOLO model path",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=os.path.join(project_root, "test3.png"),
        help="Input image path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(project_root, "test3_yolo.png"),
        help="Output visualization image path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = YOLO(args.model)
    results = model(args.image)

    print("results_count:", len(results))

    if len(results) == 0:
        print("no detections")
        raise SystemExit(0)

    boxes = results[0].boxes
    names = model.names

    # ultralytics 返回的boxes一般是：boxes.cls, boxes.conf, boxes.xyxy
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        if isinstance(names, dict):
            cls_name = names.get(cls_id, str(cls_id))
        elif isinstance(names, (list, tuple)):
            cls_name = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
        else:
            cls_name = str(cls_id)
        print(
            "det",
            i,
            "class:",
            cls_name,
            "conf:",
            round(conf, 6),
            "xyxy:",
            [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        )

    annotated = results[0].plot()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ok = cv2.imwrite(args.output, annotated)
    print("saved:", args.output, "ok:", ok)