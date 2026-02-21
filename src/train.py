"""
Training pipeline for VinBigData CXR Object Detection (Faster R-CNN).
"""
import math
import sys
import os
import torch
import torchvision.transforms as T

import utils
from dataset import VinBigDataDataset
from model import get_model
from evaluate import evaluate
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
TRAIN_IMG_DIR  = r"d:/dlpro/archive (1)/vinbigdata-cxr-ad-coco/images/train"
TRAIN_ANN_FILE = r"d:/dlpro/archive (1)/vinbigdata-cxr-ad-coco/annotations/instances_train.json"

NUM_CLASSES    = 15          # 14 abnormalities + 1 background
BATCH_SIZE     = 4
NUM_EPOCHS     = 10          # increase as needed
LR             = 0.005
CHECKPOINT_DIR = r"d:/dlpro/checkpoints"


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────
def get_transform():
    return T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
    ])


# ──────────────────────────────────────────────
# Training loop (one epoch)
# ──────────────────────────────────────────────
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    losses_per_iter = []
    header = f"Epoch [{epoch}]"

    # Warm-up LR scheduler for first epoch
    warmup_scheduler = None
    if epoch == 0:
        warmup_iters = min(1000, len(data_loader) - 1)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0 / 1000, total_iters=warmup_iters
        )

    for i, (images, targets) in enumerate(data_loader):
        images  = [img.to(device)  for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss for loss in loss_dict.values())
        loss_val  = losses.item()

        if not math.isfinite(loss_val):
            print(f"Loss is {loss_val}, stopping training.")
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if warmup_scheduler is not None:
            warmup_scheduler.step()

        losses_per_iter.append(loss_val)

        if (i + 1) % 50 == 0 or (i + 1) == len(data_loader):
            avg = sum(losses_per_iter) / len(losses_per_iter)
            print(f"  {header} step [{i+1}/{len(data_loader)}] avg loss: {avg:.4f}")

    avg_loss = sum(losses_per_iter) / len(losses_per_iter)
    print(f"{header} finished — avg loss: {avg_loss:.4f}")
    return avg_loss


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    transform = get_transform()

    full_dataset      = VinBigDataDataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE, transforms=transform)
    full_dataset_eval = VinBigDataDataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE, transforms=transform)

    # 90 / 10 train-val split (fixed seed for reproducibility)
    g          = torch.Generator().manual_seed(42)
    indices    = torch.randperm(len(full_dataset), generator=g).tolist()
    split      = int(0.9 * len(full_dataset))
    train_idx  = indices[:split]
    val_idx    = indices[split:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset   = torch.utils.data.Subset(full_dataset_eval, val_idx)

    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=utils.collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=utils.collate_fn
    )

    model = get_model(NUM_CLASSES)
    model.to(device)

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)
    lr_sched  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(NUM_EPOCHS):
        # --- Train ---
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_sched.step()

        # --- Evaluate (mAP) ---
        print(f"\nRunning validation for epoch {epoch}...")
        metrics = evaluate(model, val_loader, device, TRAIN_ANN_FILE)
        if metrics:
            print(f"  AP@[0.50:0.95] = {metrics.get('AP@[0.50:0.95]', 0):.4f}")
            print(f"  AP@0.50        = {metrics.get('AP@0.50', 0):.4f}")

        # --- Save checkpoint ---
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_loss": avg_loss,
            "metrics": metrics,
        }, ckpt_path)
        print(f"Checkpoint saved → {ckpt_path}\n")

    print("Training complete!")


if __name__ == "__main__":
    main()
