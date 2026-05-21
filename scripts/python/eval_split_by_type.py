import json
from collections import defaultdict
from pathlib import Path


def categorize(fname: str) -> str:
    lf = fname.lower()
    if "video" in lf:
        return "scene_video"
    if lf.startswith("00") and ("_00" in lf or "_jpg" in lf or "_png" in lf):
        if any(seg in lf for seg in ("_00026", "_00029", "_00014")):
            return "crop_gtsrb"
        if "_00" in lf.split(".")[0]:
            return "crop_gtsrb"
    if "road" in lf or "ipm" in lf or "scene" in lf:
        return "scene_other"
    return "scene_other"


def metrics(stats):
    p = stats["tp"] / (stats["tp"] + stats["fp"]) if stats["tp"] + stats["fp"] > 0 else 0
    r = stats["tp"] / (stats["tp"] + stats["fn"]) if stats["tp"] + stats["fn"] > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return p, r, f1


data = json.load(open("reports/zero_shot_eval/per_image.json", encoding="utf-8"))
print(f"Total images: {len(data)}")

by_cat = defaultdict(lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}))
cat_counts = defaultdict(int)

for d in data:
    cat = categorize(d["file"])
    cat_counts[cat] += 1
    gt = set(d["gt"])
    pr = set(d["pred"])
    for c in ("50", "70", "90", "110"):
        in_gt = c in gt
        in_pr = c in pr
        if in_gt and in_pr: by_cat[cat][c]["tp"] += 1
        elif in_gt and not in_pr: by_cat[cat][c]["fn"] += 1
        elif not in_gt and in_pr: by_cat[cat][c]["fp"] += 1
        else: by_cat[cat][c]["tn"] += 1

print("\nImage counts per category:")
for cat, n in sorted(cat_counts.items()):
    print(f"  {cat}: {n}")

for cat in sorted(by_cat.keys()):
    print(f"\n=== Category: {cat} ({cat_counts[cat]} images) ===")
    print(f"{'class':<6} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  {'P':>6} {'R':>6} {'F1':>6}")
    macro_p, macro_r, macro_f = 0, 0, 0
    for c in ("50", "70", "90", "110"):
        s = by_cat[cat][c]
        p, r, f = metrics(s)
        macro_p += p; macro_r += r; macro_f += f
        print(f"{c:<6} {s['tp']:>5} {s['fp']:>5} {s['fn']:>5} {s['tn']:>5}  {p:>6.3f} {r:>6.3f} {f:>6.3f}")
    print(f"Macro: P={macro_p/4:.3f} R={macro_r/4:.3f} F1={macro_f/4:.3f}")
