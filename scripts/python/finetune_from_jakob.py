import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import ultralytics.models.yolo as yolo_new
sys.modules["ultralytics.yolo"] = yolo_new
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo/data.yaml")
    ap.add_argument("--weights", default="models/pretrained_candidates/jakobjfl_dk.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/finetune")
    ap.add_argument("--name", default="vision_ensem_v1")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--export-onnx", default="models/yolov8_finetuned.onnx")
    args = ap.parse_args()

    print(f"Base weights: {args.weights}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}, imgsz: {args.imgsz}, batch: {args.batch}, device: {args.device}")

    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save=True,
        verbose=True,
        plots=True,
        cos_lr=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.4,
        fliplr=0.0,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=10,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt not found at {best}")

    print(f"\nBest weights: {best}")
    final = YOLO(str(best))
    onnx_path = final.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=12)
    onnx_path = Path(onnx_path)
    target = Path(args.export_onnx)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.replace(target)
    print(f"ONNX exported to: {target}")


if __name__ == "__main__":
    main()
