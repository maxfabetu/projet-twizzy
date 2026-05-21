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
    ap.add_argument("--weights", default="models/yolov8_finetuned.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/finetune")
    ap.add_argument("--name", default="vision_ensem_rot")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--degrees", type=float, default=30.0)
    ap.add_argument("--export-onnx", default="models/yolov8_finetuned.onnx")
    args = ap.parse_args()

    print(f"Base weights: {args.weights}")
    print(f"Rotation augmentation: degrees=±{args.degrees}")

    model = YOLO(args.weights)
    model.train(
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
        degrees=args.degrees,
        translate=0.1,
        scale=0.5,
        shear=5.0,
        perspective=0.0005,
        fliplr=0.0,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=10,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if not best.exists():
        candidates = list(Path("runs").rglob(f"{args.name}/weights/best.pt"))
        if candidates:
            best = candidates[0]
            print(f"Resolved best.pt at: {best}")
        else:
            raise FileNotFoundError(f"best.pt not found at {best}")

    final = YOLO(str(best))
    onnx_path = final.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=12)
    onnx_path = Path(onnx_path)
    target = Path(args.export_onnx)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.replace(target)
    import shutil
    shutil.copy2(str(best), "models/yolov8_finetuned.pt")
    print(f"ONNX exported to: {target}")


if __name__ == "__main__":
    main()
