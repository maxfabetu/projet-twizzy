import argparse
import csv
import json
import os
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

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


def read_csv_index(csv_path: Path):
    idx = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"].strip()
            positives = set()
            for col, short in CSV_COL_TO_SHORT.items():
                if row.get(col, "0").strip() in ("1", "1.0"):
                    positives.add(short)
            idx[fname] = positives
    return idx


def select_sample(idx, max_pos, max_neg, seed=42):
    rng = random.Random(seed)
    pos = [k for k, v in idx.items() if v]
    neg = [k for k, v in idx.items() if not v]
    rng.shuffle(pos)
    rng.shuffle(neg)
    pos = pos[: max_pos] if max_pos > 0 else pos
    neg = neg[: max_neg] if max_neg >= 0 else neg
    return pos + neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/pretrained_candidates/jakobjfl_dk.pt")
    ap.add_argument("--split", default="test", choices=["test", "valid", "train"])
    ap.add_argument("--root", default="data/raw/BDD_Roboflow_Tazi")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-pos", type=int, default=200)
    ap.add_argument("--max-neg", type=int, default=200)
    ap.add_argument("--out-dir", default="reports/zero_shot_eval")
    ap.add_argument("--save-images", type=int, default=30)
    args = ap.parse_args()

    split_dir = Path(args.root) / args.split
    csv_path = split_dir / "_classes.csv"
    idx = read_csv_index(csv_path)
    sample = select_sample(idx, args.max_pos, args.max_neg)
    print(f"Split={args.split} total={len(idx)} sampled={len(sample)} positives={sum(1 for f in sample if idx[f])}")

    model = YOLO(args.model)
    target_yolo_ids = list(YOLO_ID_TO_SHORT.keys())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annot_dir = out_dir / "samples"
    annot_dir.mkdir(parents=True, exist_ok=True)

    # Per-class confusion
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    tn = defaultdict(int)

    saved = 0
    by_class_samples = defaultdict(list)

    def iter_predict():
        for f in sample:
            res = model.predict(
                str(split_dir / f), imgsz=args.imgsz, conf=args.conf,
                classes=target_yolo_ids, verbose=False, save=False,
            )
            yield res[0]

    per_image = []
    for fname, r in zip(sample, iter_predict()):
        gt_set = idx[fname]
        det_set = set()
        det_details = []
        for b in r.boxes:
            yid = int(b.cls.item())
            short = YOLO_ID_TO_SHORT.get(yid)
            if short:
                det_set.add(short)
                det_details.append({"class": short, "conf": float(b.conf.item()),
                                    "bbox": [round(v, 1) for v in b.xyxy[0].tolist()]})
        per_image.append({"file": fname, "gt": sorted(gt_set), "pred": sorted(det_set), "boxes": det_details})

        for short in CSV_COL_TO_SHORT.values():
            in_gt = short in gt_set
            in_pred = short in det_set
            if in_gt and in_pred: tp[short] += 1
            elif in_gt and not in_pred: fn[short] += 1
            elif not in_gt and in_pred: fp[short] += 1
            else: tn[short] += 1

        if saved < args.save_images and det_details:
            for d in det_details:
                if len(by_class_samples[d["class"]]) < args.save_images // 4:
                    by_class_samples[d["class"]].append(fname)
                    saved += 1
                    # save an annotated image
                    annotated = r.plot()
                    import cv2
                    out_img = annot_dir / f"{d['class']}_{Path(fname).stem}.jpg"
                    cv2.imwrite(str(out_img), annotated)
                    break

    metrics = {}
    for short in CSV_COL_TO_SHORT.values():
        t = tp[short]; f = fp[short]; n = fn[short]; tnv = tn[short]
        precision = t / (t + f) if (t + f) > 0 else 0.0
        recall = t / (t + n) if (t + n) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[short] = {
            "tp": t, "fp": f, "fn": n, "tn": tnv,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    overall_tp = sum(tp.values())
    overall_fp = sum(fp.values())
    overall_fn = sum(fn.values())
    overall = {
        "macro_precision": round(sum(m["precision"] for m in metrics.values()) / len(metrics), 4),
        "macro_recall": round(sum(m["recall"] for m in metrics.values()) / len(metrics), 4),
        "macro_f1": round(sum(m["f1"] for m in metrics.values()) / len(metrics), 4),
        "micro_tp": overall_tp,
        "micro_fp": overall_fp,
        "micro_fn": overall_fn,
    }

    summary = {
        "model": args.model,
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "sampled": len(sample),
        "per_class": metrics,
        "overall": overall,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_image.json", "w", encoding="utf-8") as f:
        json.dump(per_image, f, indent=2)

    print("\n=== Per-class metrics (image-level: CSV ground truth) ===")
    print(f"{'class':<6} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  {'P':>6} {'R':>6} {'F1':>6}")
    for short, m in metrics.items():
        print(f"{short:<6} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} {m['tn']:>5}  "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")
    print(f"\nMacro: P={overall['macro_precision']:.3f} R={overall['macro_recall']:.3f} F1={overall['macro_f1']:.3f}")
    print(f"Annotated samples saved to {annot_dir}")
    print(f"Full report: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
