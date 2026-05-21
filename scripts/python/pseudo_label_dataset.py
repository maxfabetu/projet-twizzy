import argparse
import csv
import os
import shutil
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import ultralytics.models.yolo as yolo_new
sys.modules["ultralytics.yolo"] = yolo_new
from ultralytics import YOLO


CSV_COL_TO_INTERNAL = {
    "Speed_limit_50_km_h": 0,
    "Speed_limit_70_km_h": 1,
    "Speed_limit_90_km_h": 2,
    "Speed_limit_110_km_h": 3,
}
YOLO_ID_TO_INTERNAL = {11: 0, 15: 1, 17: 2, 5: 3}
INTERNAL_TO_SHORT = {0: "50", 1: "70", 2: "90", 3: "110"}


def read_csv_index(csv_path):
    idx = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = row["filename"].strip()
            positives = set()
            for col, internal in CSV_COL_TO_INTERNAL.items():
                if row.get(col, "0").strip() in ("1", "1.0"):
                    positives.add(internal)
            idx[fname] = positives
    return idx


def write_yolo_labels(label_path: Path, detections, img_w: int, img_h: int):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for det in detections:
        cls, x1, y1, x2, y2 = det
        cx = ((x1 + x2) / 2.0) / img_w
        cy = ((y1 + y2) / 2.0) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src.resolve(), dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/pretrained_candidates/jakobjfl_dk.pt")
    ap.add_argument("--raw-root", default="data/raw/BDD_Roboflow_Tazi")
    ap.add_argument("--out-root", default="data/yolo")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--max-per-split", type=int, default=-1)
    ap.add_argument("--neg-ratio", type=float, default=0.15,
                    help="Fraction max d'images negatives (sans panneau cible) a inclure")
    ap.add_argument("--require-csv-match", action="store_true", default=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    print(f"Loading model {args.model} ...")
    model = YOLO(args.model)
    target_yolo_ids = list(YOLO_ID_TO_INTERNAL.keys())

    overall_stats = defaultdict(int)

    for split in ("train", "valid", "test"):
        split_in = Path(args.raw_root) / split
        if not split_in.exists():
            print(f"  Skip {split}: directory not found")
            continue
        csv_path = split_in / "_classes.csv"
        idx = read_csv_index(csv_path)
        images = sorted([f for f in idx if (split_in / f).exists()])
        if args.max_per_split > 0:
            images = images[: args.max_per_split]
        n_total = len(images)
        print(f"\n=== {split} : {n_total} images ===")

        out_imgs = Path(args.out_root) / split / "images"
        out_lbls = Path(args.out_root) / split / "labels"
        out_imgs.mkdir(parents=True, exist_ok=True)
        out_lbls.mkdir(parents=True, exist_ok=True)

        stats = defaultdict(int)
        pos_kept = 0
        neg_kept = 0
        max_negs = int(n_total * args.neg_ratio)

        t_start = time.time()
        for i in range(0, n_total, args.batch):
            batch_files = images[i:i + args.batch]
            batch_paths = [str(split_in / f) for f in batch_files]
            try:
                results = model.predict(batch_paths, imgsz=args.imgsz, conf=args.conf,
                                        classes=target_yolo_ids, verbose=False, save=False)
            except Exception as e:
                print(f"  Batch error at {i}: {e!r}, retrying single-image...")
                results = []
                for p in batch_paths:
                    try:
                        r = model.predict(p, imgsz=args.imgsz, conf=args.conf,
                                          classes=target_yolo_ids, verbose=False, save=False)
                        results.append(r[0])
                    except Exception as ee:
                        print(f"    single failed: {p}: {ee!r}")
                        results.append(None)

            for fname, res in zip(batch_files, results):
                if res is None:
                    stats["error"] += 1
                    continue
                csv_pos = idx[fname]
                detections = []
                for b in res.boxes:
                    yid = int(b.cls.item())
                    internal = YOLO_ID_TO_INTERNAL.get(yid)
                    if internal is None:
                        continue
                    if args.require_csv_match and csv_pos and internal not in csv_pos:
                        stats["filtered_csv_mismatch"] += 1
                        continue
                    if args.require_csv_match and not csv_pos:
                        stats["det_on_negative_image"] += 1
                        continue
                    xyxy = b.xyxy[0].tolist()
                    detections.append((internal, *xyxy))

                src_img = split_in / fname
                img_h, img_w = res.orig_shape if hasattr(res, "orig_shape") else (640, 640)

                if detections:
                    write_yolo_labels(out_lbls / (Path(fname).stem + ".txt"),
                                      detections, img_w, img_h)
                    link_or_copy(src_img, out_imgs / fname)
                    pos_kept += 1
                    for d in detections:
                        stats[f"kept_{INTERNAL_TO_SHORT[d[0]]}"] += 1
                elif not csv_pos and neg_kept < max_negs:
                    (out_lbls / (Path(fname).stem + ".txt")).write_text("", encoding="utf-8")
                    link_or_copy(src_img, out_imgs / fname)
                    neg_kept += 1
                    stats["kept_negative"] += 1
                else:
                    stats["skipped_no_detection"] += 1

            if (i // args.batch) % 20 == 0:
                elapsed = time.time() - t_start
                fps = (i + args.batch) / max(0.01, elapsed)
                print(f"  [{min(i+args.batch, n_total)}/{n_total}] pos={pos_kept} neg={neg_kept} fps={fps:.1f}")

        print(f"  -- {split} done: pos={pos_kept} neg={neg_kept}")
        print(f"  -- counts per class: " + " ".join(
            f"{k}={v}" for k, v in sorted(stats.items()) if k.startswith("kept_")))
        for k, v in stats.items():
            overall_stats[f"{split}_{k}"] = v

    out_root = Path(args.out_root)
    classes_txt = out_root / "classes.txt"
    classes_txt.write_text("\n".join(f"{i} {INTERNAL_TO_SHORT[i]}" for i in range(4)),
                           encoding="utf-8")
    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        f"path: {out_root.absolute().as_posix()}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"nc: 4\n"
        f"names: ['50', '70', '90', '110']\n",
        encoding="utf-8")
    print(f"\nWrote {classes_txt} and {data_yaml}")
    print("\nOverall stats:")
    for k, v in sorted(overall_stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
