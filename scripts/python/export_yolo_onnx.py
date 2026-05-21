import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--simplify", action="store_true", default=True)
    p.add_argument("--dynamic", action="store_true", default=False)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    model = YOLO(args.weights)
    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=args.simplify,
        dynamic=args.dynamic,
        opset=args.opset,
    )
    onnx_path = Path(onnx_path)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        onnx_path.replace(target)
        onnx_path = target
    print(f"ONNX exported to: {onnx_path}")


if __name__ == "__main__":
    main()
