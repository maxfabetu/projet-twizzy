import argparse
import csv
import os
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import ultralytics.models.yolo as yolo_new
sys.modules["ultralytics.yolo"] = yolo_new
from ultralytics import YOLO


YOLO_ID_TO_SHORT = {11: "50", 15: "70", 17: "90", 5: "110"}
CSV_COL_TO_SHORT = {
    "Speed_limit_50_km_h": "50",
    "Speed_limit_70_km_h": "70",
    "Speed_limit_90_km_h": "90",
    "Speed_limit_110_km_h": "110",
}


def read_csv_index(csv_path):
    idx = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = row["filename"].strip()
            positives = set()
            for col, short in CSV_COL_TO_SHORT.items():
                if row.get(col, "0").strip() in ("1", "1.0"):
                    positives.add(short)
            idx[fname] = positives
    return idx


def is_scene_image(fname: str) -> bool:
    lf = fname.lower()
    if "video" in lf or "road" in lf or "ipm" in lf or "scene" in lf:
        return True
    base = lf.split(".")[0]
    parts = base.split("_")
    if parts[0].isdigit() and len(parts[0]) == 5:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/pretrained_candidates/jakobjfl_dk.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--root", default="data/raw/BDD_Roboflow_Tazi")
    ap.add_argument("--out", default="reports/annotated_full")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--only-scenes", action="store_true", default=True)
    ap.add_argument("--keep-empty", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    split_dir = Path(args.root) / args.split
    idx = read_csv_index(split_dir / "_classes.csv")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "positives").mkdir(exist_ok=True)
    (out_dir / "negatives").mkdir(exist_ok=True)

    candidates = [f for f in idx if (not args.only_scenes) or is_scene_image(f)]
    pos = [f for f in candidates if idx[f]]
    neg = [f for f in candidates if not idx[f]]
    rng = random.Random(args.seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    take_pos = pos[: args.limit]
    take_neg = neg[: max(0, args.limit // 4)]
    selection = take_pos + take_neg

    print(f"Loading model {args.model} ...")
    model = YOLO(args.model)
    target_ids = list(YOLO_ID_TO_SHORT.keys())
    print(f"Will process {len(selection)} images ({len(take_pos)} positives, {len(take_neg)} negatives)")

    written = 0
    skipped_empty = 0
    for i, fname in enumerate(selection, 1):
        img_path = split_dir / fname
        res = model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf,
                            classes=target_ids, verbose=False, save=False)[0]
        if len(res.boxes) == 0 and not args.keep_empty:
            skipped_empty += 1
            continue
        annotated = res.plot()
        gt = idx[fname]
        gt_str = "_".join(sorted(gt)) if gt else "neg"
        subdir = out_dir / ("positives" if gt else "negatives")
        out_name = f"{gt_str}_{Path(fname).stem[:80]}.jpg"
        cv2.imwrite(str(subdir / out_name), annotated)
        written += 1
        if i % 25 == 0:
            print(f"  [{i}/{len(selection)}] written={written} skipped_empty={skipped_empty}")

    print(f"\nDone. Wrote {written} annotated images to {out_dir}")
    print(f"Skipped empty (no detection): {skipped_empty}")
    print(f"  positives/: images marked in CSV")
    print(f"  negatives/: images NOT marked in CSV (false positives if any)")


if __name__ == "__main__":
    main()
