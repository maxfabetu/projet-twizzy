import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import ultralytics.models.yolo as yolo_new
sys.modules["ultralytics.yolo"] = yolo_new
from ultralytics import YOLO


def rotate_image(img, angle):
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += (nw / 2) - cx
    M[1, 2] += (nh / 2) - cy
    return cv2.warpAffine(img, M, (nw, nh), borderValue=(128, 128, 128))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--v1", default="models/yolov8_finetuned_v1.pt")
    ap.add_argument("--v2", default="models/yolov8_finetuned.pt")
    ap.add_argument("--angles", default="-30,-20,-10,0,10,20,30")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--out", default="reports/rotation_compare")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    m1 = YOLO(args.v1)
    m2 = YOLO(args.v2)
    print(f"v1 classes: {m1.names}")
    print(f"v2 classes: {m2.names}")

    angles = [float(a) for a in args.angles.split(",")]
    print(f"\n{'angle':>6} | {'v1 detections':<40} | {'v2 detections':<40}")
    print("-" * 92)
    for a in angles:
        rot = rotate_image(img, a)
        r1 = m1.predict(rot, conf=args.conf, verbose=False)[0]
        r2 = m2.predict(rot, conf=args.conf, verbose=False)[0]
        d1 = ", ".join(f"{m1.names[int(b.cls)]} {float(b.conf):.2f}" for b in r1.boxes) or "(rien)"
        d2 = ", ".join(f"{m2.names[int(b.cls)]} {float(b.conf):.2f}" for b in r2.boxes) or "(rien)"
        print(f"{a:>6.1f} | {d1:<40} | {d2:<40}")

        annotated_v1 = r1.plot()
        annotated_v2 = r2.plot()
        h = max(annotated_v1.shape[0], annotated_v2.shape[0])
        w = annotated_v1.shape[1] + annotated_v2.shape[1] + 10
        canvas = np.full((h + 30, w, 3), 255, dtype=np.uint8)
        canvas[30:30 + annotated_v1.shape[0], 0:annotated_v1.shape[1]] = annotated_v1
        canvas[30:30 + annotated_v2.shape[0], annotated_v1.shape[1] + 10:] = annotated_v2
        cv2.putText(canvas, f"v1 (deg=5) | rot={a}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(canvas, f"v2 (deg=30) | rot={a}", (annotated_v1.shape[1] + 20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.imwrite(str(out_dir / f"compare_a{int(a):+04d}.jpg"), canvas)


if __name__ == "__main__":
    main()
