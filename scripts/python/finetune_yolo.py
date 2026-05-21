import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name", default="vision_ensem")
    p.add_argument("--export-onnx", default="models/yolov8n_signs.onnx")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

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
        resume=args.resume,
        save=True,
        verbose=True,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt not found at {best}")
    final = YOLO(str(best))
    onnx_path = final.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=12)
    onnx_path = Path(onnx_path)
    target = Path(args.export_onnx)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.replace(target)
    print(f"Fine-tuned ONNX exported to: {target}")


if __name__ == "__main__":
    main()
