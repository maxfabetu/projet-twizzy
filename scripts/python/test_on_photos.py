import argparse
import csv
import json
import random
import shutil
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import ultralytics.models.yolo as yolo_new
sys.modules["ultralytics.yolo"] = yolo_new
from ultralytics import YOLO


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
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--root", default="data/raw/BDD_Roboflow_Tazi")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="reports/test_finetuned")
    ap.add_argument("--max-pos", type=int, default=400)
    ap.add_argument("--max-neg", type=int, default=200)
    ap.add_argument("--only-scenes", action="store_true", default=False)
    ap.add_argument("--save-images", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    split_dir = Path(args.root) / args.split
    idx = read_csv_index(split_dir / "_classes.csv")
    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "annotated").mkdir(exist_ok=True)

    pool = [f for f in idx if (not args.only_scenes) or is_scene_image(f)]
    pos = [f for f in pool if idx[f]]
    neg = [f for f in pool if not idx[f]]
    rng = random.Random(args.seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    sample = pos[: args.max_pos] + neg[: args.max_neg]

    print(f"Model: {args.model}")
    print(f"Sample: {len(sample)} = {min(len(pos), args.max_pos)} pos + {min(len(neg), args.max_neg)} neg")

    model = YOLO(args.model)

    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int); tn = defaultdict(int)
    saved = 0
    per_image = []

    for i, fname in enumerate(sample, 1):
        img_path = split_dir / fname
        res = model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf,
                            verbose=False, save=False)[0]
        gt_set = idx[fname]
        det_set = set()
        det_details = []
        for b in res.boxes:
            cls_id = int(b.cls.item())
            short = model.names.get(cls_id, str(cls_id))
            det_set.add(short)
            det_details.append({"class": short, "conf": float(b.conf.item()),
                                "bbox": [round(v, 1) for v in b.xyxy[0].tolist()]})
        per_image.append({"file": fname, "gt": sorted(gt_set), "pred": sorted(det_set),
                          "boxes": det_details})

        for short in CSV_COL_TO_SHORT.values():
            in_gt = short in gt_set
            in_pred = short in det_set
            if in_gt and in_pred: tp[short] += 1
            elif in_gt and not in_pred: fn[short] += 1
            elif not in_gt and in_pred: fp[short] += 1
            else: tn[short] += 1

        if saved < args.save_images and det_details:
            annotated = res.plot()
            gt_label = "_".join(sorted(gt_set)) if gt_set else "neg"
            out_name = f"{gt_label}__{Path(fname).stem[:80]}.jpg"
            cv2.imwrite(str(out_dir / "annotated" / out_name), annotated)
            saved += 1

        if i % 50 == 0:
            print(f"  [{i}/{len(sample)}] saved={saved}")

    metrics = {}
    for short in CSV_COL_TO_SHORT.values():
        t = tp[short]; f = fp[short]; n = fn[short]; tnv = tn[short]
        precision = t / (t + f) if (t + f) > 0 else 0.0
        recall = t / (t + n) if (t + n) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[short] = {"tp": t, "fp": f, "fn": n, "tn": tnv,
                          "precision": round(precision, 4),
                          "recall": round(recall, 4),
                          "f1": round(f1, 4)}

    macro = {
        "precision": round(sum(m["precision"] for m in metrics.values()) / len(metrics), 4),
        "recall": round(sum(m["recall"] for m in metrics.values()) / len(metrics), 4),
        "f1": round(sum(m["f1"] for m in metrics.values()) / len(metrics), 4),
    }
    summary = {"model": args.model, "split": args.split, "imgsz": args.imgsz,
               "conf": args.conf, "sampled": len(sample), "per_class": metrics,
               "macro": macro}

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_image.json", "w", encoding="utf-8") as f:
        json.dump(per_image, f, indent=2)

    print("\n=== Per-class metrics ===")
    print(f"{'class':<6} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  {'P':>6} {'R':>6} {'F1':>6}")
    for short, m in metrics.items():
        print(f"{short:<6} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} {m['tn']:>5}  "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")
    print(f"\nMacro: P={macro['precision']:.3f} R={macro['recall']:.3f} F1={macro['f1']:.3f}")
    print(f"Annotated images saved to {out_dir/'annotated'}")
    print(f"Full report: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
