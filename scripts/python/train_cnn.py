import argparse
import csv
from pathlib import Path
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


CLASS_MAP = {
    "Speed_limit_50_km_h": 0,
    "Speed_limit_70_km_h": 1,
    "Speed_limit_90_km_h": 2,
    "Speed_limit_110_km_h": 3,
}


class RoiDataset(Dataset):
    def __init__(self, images_dir: Path, labels_dir: Path, classes_csv: Path, size: int, augment: bool):
        self.size = size
        self.augment = augment
        index = {}
        with open(classes_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pos = []
                for k, v in row.items():
                    k = k.strip()
                    if k in CLASS_MAP and v.strip() in ("1", "1.0"):
                        pos.append(CLASS_MAP[k])
                if pos:
                    index[row["filename"].strip()] = pos[0]
        self.samples = []
        for img in images_dir.glob("*.jpg"):
            cls = index.get(img.name)
            if cls is None:
                continue
            lbl = labels_dir / (img.stem + ".txt")
            if not lbl.exists():
                continue
            self.samples.append((img, lbl, cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path, cls = self.samples[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            return self._fallback(cls)
        h, w = img.shape[:2]
        boxes = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                c, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                if c != cls:
                    continue
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
        if not boxes:
            return self._fallback(cls)
        x1, y1, x2, y2 = random.choice(boxes) if self.augment else boxes[0]
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return self._fallback(cls)
        if self.augment and random.random() < 0.5:
            roi = cv2.flip(roi, 1)
        roi = cv2.resize(roi, (self.size, self.size))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        roi = np.transpose(roi, (2, 0, 1))
        return torch.from_numpy(roi), torch.tensor(cls, dtype=torch.long)

    def _fallback(self, cls):
        arr = np.zeros((3, self.size, self.size), dtype=np.float32)
        return torch.from_numpy(arr), torch.tensor(cls, dtype=torch.long)


class SmallCnn(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-images", required=True)
    p.add_argument("--train-labels", required=True)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-images", required=True)
    p.add_argument("--val-labels", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="models/cnn_classifier.onnx")
    args = p.parse_args()

    train_ds = RoiDataset(Path(args.train_images), Path(args.train_labels), Path(args.train_csv), args.size, True)
    val_ds = RoiDataset(Path(args.val_images), Path(args.val_labels), Path(args.val_csv), args.size, False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    model = SmallCnn(len(CLASS_MAP)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for ep in range(args.epochs):
        model.train()
        tl = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            tl += loss.item() * xb.size(0)
            n += xb.size(0)
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        acc = correct / max(1, total)
        print(f"epoch {ep+1}/{args.epochs} loss={tl/max(1,n):.4f} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            dummy = torch.zeros(1, 3, args.size, args.size, device=device)
            torch.onnx.export(
                model, dummy, args.out,
                input_names=["input"], output_names=["logits"],
                opset_version=12, do_constant_folding=True,
                dynamic_axes=None,
            )
            print(f"saved {args.out} (val_acc={acc:.4f})")

    print(f"best val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
