"""
Train EfficientNet to classify vehicle models (multi‑scale friendly).

Additions vs. previous version
──────────────────────────────
* **Wide‑scale augmentation** – `RandomResizedCrop(scale=(0.4,1.0))` handles very small crops.
* **Progressive resizing** – start small (`--image-size`), then automatically switch to a larger
  `--progressive-size` after the head‑freeze phase, re‑building DataLoaders on the fly.
* Minor refactor – helper to (re)build transforms & loaders.

Run example
```
python train_vehicle_model_efficientnet.py \
    --src-dir "E:/윤경섭/상세차종_re-id_datasets" \
    --dst-dir "E:/윤경섭/vehicle_model" \
    --batch-size 32 --epochs 50 --freeze-epochs 10 \
    --progressive-size 320
```
"""

import argparse
import os
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# ────────────────────────────────────────────────
# Utility
# ────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Stop training when `val_loss` hasn’t improved for `patience` epochs."""

    def __init__(self, patience: int = 7):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.stop = False

    def step(self, val_loss: float):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────


class VehicleDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.samples: List[tuple[str, int]] = []
        self.class_to_idx = {}

        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            self.class_to_idx[class_name] = idx
            for fname in os.listdir(class_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.samples.append((os.path.join(class_path, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ────────────────────────────────────────────────
# Train / Val loops
# ────────────────────────────────────────────────

def accuracy(pred, target):
    return (pred == target).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True):
    model.train() if train else model.eval()
    epoch_loss = 0.0
    epoch_acc = 0.0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            epoch_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            epoch_acc += (preds == labels).sum().item()

    total = len(loader.dataset)
    return epoch_loss / total, epoch_acc / total


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EfficientNet vehicle model classifier (multi-scale)")

    # Paths
    parser.add_argument("--src-dir", required=True, help="Original dataset root")
    parser.add_argument("--dst-dir", required=True, help="Root folder containing <train>/<valid>")
    parser.add_argument("--prepare", action="store_true", help="Run dataset prepare util before training")

    # Hyper-parameters
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--freeze-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)

    # Multi‑scale options
    parser.add_argument("--image-size", type=int, default=224, help="Initial image size")
    parser.add_argument("--progressive-size", type=int, default=0,
                        help="If >0, switch input size to this value after freeze phase")

    args = parser.parse_args()
    set_seed(args.seed)

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)

    if args.prepare:
        from vehicle_model_dataset_prepare import run_vmodel_dataset_prepare
        run_vmodel_dataset_prepare(src_dir, dst_dir, train_ratio=0.8)

    train_root = dst_dir / "train"
    val_root = dst_dir / "valid"
    if not (train_root.exists() and val_root.exists()):
        raise FileNotFoundError("train/valid 폴더가 없습니다 – --prepare 옵션을 확인하세요.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Helper to build transforms & loaders given a size
    def build_loaders(img_size: int):
        train_tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomResizedCrop(img_size, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
            #T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.RandomRotation(15),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
            T.RandomPerspective(distortion_scale=0.2, p=0.5),
            T.GaussianBlur(3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        val_tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        tr_ds = VehicleDataset(str(train_root), transform=train_tf)
        vl_ds = VehicleDataset(str(val_root), transform=val_tf)
        num_workers = min(os.cpu_count(), 8)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, persistent_workers=True)
        vl_loader = DataLoader(vl_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True, persistent_workers=True)
        return tr_ds, tr_loader, vl_loader

    # Initial loaders
    train_ds, train_loader, val_loader = build_loaders(args.image_size)
    num_classes = len(train_ds.class_to_idx)

    # Model & optimiser
    model = EfficientNet.from_pretrained("efficientnet-b4", num_classes=num_classes)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7)

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    stopper = EarlyStopping(patience=args.patience)

    best_acc = 0.0
    best_model_path = dst_dir / "efficientnet_vehicle_model_best.pth"
    log_fp = open(dst_dir / "train_log.txt", "w", encoding="utf-8")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Freeze/unfreeze
        if epoch == 1 or epoch == args.freeze_epochs + 1:
            trainable = []
            for name, p in model.named_parameters():
                if epoch <= args.freeze_epochs:
                    p.requires_grad = name.startswith("_fc")
                else:
                    p.requires_grad = True
                if p.requires_grad:
                    trainable.append(p)
            optimizer.param_groups = []
            optimizer.add_param_group({"params": trainable})
            print(f"Epoch {epoch}: trainable params updated → {len(trainable)} tensors")

        # Progressive resizing: rebuild loaders once after freeze phase
        if args.progressive_size > 0 and epoch == args.freeze_epochs + 1:
            print(f"↗ Switching image size {args.image_size} → {args.progressive_size}")
            train_ds, train_loader, val_loader = build_loaders(args.progressive_size)

        # Run epoch
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
        scheduler.step(val_loss)

        msg = (f"Epoch {epoch:02d}/{args.epochs} | "
               f"Train {train_loss:.4f}/{train_acc:.4f} | Val {val_loss:.4f}/{val_acc:.4f} | "
               f"LR {optimizer.param_groups[0]['lr']:.2e}")
        print(msg)
        log_fp.write(msg + "\n"); log_fp.flush()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✔ Saved new best (acc={best_acc:.4f})")

        stopper.step(val_loss)
        if stopper.stop:
            print("✖ Early stopping")
            break

    log_fp.close()

    # Rename best model to include acc and save class names
    final_model = best_model_path.with_name(f"efficientnet_vehicle_model_best_acc_{best_acc:.4f}.pth")
    best_model_path.rename(final_model)

    classes_txt = dst_dir / "vehicle_classes.txt"
    with classes_txt.open("w", encoding="utf-8") as f:
        for cls in sorted(train_ds.class_to_idx, key=train_ds.class_to_idx.get):
            f.write(cls + "\n")

    print(f"Training finished – best={best_acc:.4f}\nModel saved → {final_model}\nClass list → {classes_txt}")


if __name__ == "__main__":
    main()
