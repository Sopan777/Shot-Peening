"""
Training script for Shot Peening Surface Deformation Detection.

Implements two-stage transfer learning with a U-Net segmentation model:
  Stage 1: Freeze encoder, train decoder only (epochs 1 to freeze_epochs).
  Stage 2: Unfreeze all, fine-tune with differential learning rates.

Usage:
    python src/train.py --data_dir data/
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Local imports
from dataset import get_dataloaders
from model import create_model, set_encoder_trainable, get_parameter_groups, get_model_summary
from utils import (
    calculate_metrics,
    MetricTracker,
    EarlyStopping,
    save_checkpoint,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Seed all random number generators for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs * targets_flat).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (
            probs.sum() + targets_flat.sum() + self.smooth
        )


class CombinedLoss(nn.Module):
    """Weighted sum of Binary Cross-Entropy and Dice losses."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


# ---------------------------------------------------------------------------
# Training & validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> Dict[str, float]:
    """Run one training epoch and return average metrics."""
    model.train()
    tracker = MetricTracker()

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Compute per-batch metrics
        batch_metrics = calculate_metrics(logits.detach(), masks.detach())
        batch_metrics["loss"] = loss.item()
        tracker.update(batch_metrics, n=images.size(0))

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return tracker.get_averages()


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Run validation and return average metrics."""
    model.eval()
    tracker = MetricTracker()

    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast():
            logits = model(images)
            loss = criterion(logits, masks)

        batch_metrics = calculate_metrics(logits, masks)
        batch_metrics["loss"] = loss.item()
        tracker.update(batch_metrics, n=images.size(0))

    return tracker.get_averages()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train shot-peening deformation detection model"
    )
    # Data
    p.add_argument("--data_dir", type=str, required=True, help="Root data directory with train/val/test splits")
    p.add_argument("--img_size", type=int, default=512, help="Input image size (default: 512)")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (default: 4)")

    # Model
    p.add_argument("--architecture", type=str, default="Unet", choices=["Unet", "UnetPlusPlus", "DeepLabV3Plus"])
    p.add_argument("--encoder", type=str, default="resnet34", help="Encoder backbone (default: resnet34)")

    # Training
    p.add_argument("--epochs", type=int, default=100, help="Max epochs (default: 100)")
    p.add_argument("--lr", type=float, default=1e-4, help="Encoder learning rate (default: 1e-4)")
    p.add_argument("--decoder_lr", type=float, default=5e-4, help="Decoder learning rate (default: 5e-4)")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    p.add_argument("--freeze_encoder_epochs", type=int, default=30, help="Epochs to freeze encoder (default: 30)")
    p.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")

    # Loss
    p.add_argument("--bce_weight", type=float, default=0.5, help="BCE loss weight (default: 0.5)")
    p.add_argument("--dice_weight", type=float, default=0.5, help="Dice loss weight (default: 0.5)")

    # Output
    p.add_argument("--output_dir", type=str, default="outputs", help="Output directory (default: outputs)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Directories
    model_dir = Path(args.output_dir) / "models"
    log_dir = Path(args.output_dir) / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_file=str(log_dir / "training.log"))
    logger.info("=" * 60)
    logger.info("Shot Peening Deformation Detection — Training")
    logger.info("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Data
    dataloaders = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders.get("test")

    logger.info(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
    if test_loader:
        logger.info(f"Test batches:  {len(test_loader)}")

    # Model
    model = create_model(
        architecture=args.architecture,
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        classes=1,
    ).to(device)
    total_params = get_model_summary(model)
    logger.info(f"Model: {args.architecture} / {args.encoder}  ({total_params:,} params)")

    # Stage 1: freeze encoder
    set_encoder_trainable(model, trainable=False)
    logger.info(f"Stage 1 — Encoder FROZEN for first {args.freeze_encoder_epochs} epochs")

    # Optimizer with differential LR
    param_groups = get_parameter_groups(model, encoder_lr=args.lr, decoder_lr=args.decoder_lr)
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

    # Scheduler & scaler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler()

    # Loss
    criterion = CombinedLoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)

    # Early stopping
    best_model_path = str(model_dir / "best_model.pth")
    early_stopping = EarlyStopping(patience=args.patience, delta=0.001, save_path=best_model_path)

    # TensorBoard
    tb_writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_iou = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        # --- Stage transition ---
        if epoch == args.freeze_encoder_epochs + 1:
            set_encoder_trainable(model, trainable=True)
            # Reset optimizer LRs for stage 2
            optimizer.param_groups[0]["lr"] = args.lr       # encoder
            optimizer.param_groups[1]["lr"] = args.decoder_lr  # decoder
            logger.info("Stage 2 — Encoder UNFROZEN, fine-tuning all layers")

        # In stage 1 the encoder LR is irrelevant (params frozen), but
        # we set the decoder to a higher warm-up LR.
        if epoch <= args.freeze_encoder_epochs:
            optimizer.param_groups[1]["lr"] = 1e-3  # decoder warm-up LR

        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Log
        logger.info(
            f"  train_loss={train_metrics['loss']:.4f}  |  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_iou={val_metrics['iou']:.4f}  "
            f"val_dice={val_metrics['dice']:.4f}"
        )

        tb_writer.add_scalar("Loss/Train", train_metrics["loss"], epoch)
        tb_writer.add_scalar("Loss/Val", val_metrics["loss"], epoch)
        tb_writer.add_scalar("Metrics/Val_IoU", val_metrics["iou"], epoch)
        tb_writer.add_scalar("Metrics/Val_Dice", val_metrics["dice"], epoch)
        tb_writer.add_scalar("LR/Encoder", optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar("LR/Decoder", optimizer.param_groups[1]["lr"], epoch)

        # Checkpoint best
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint(model, optimizer, epoch, val_metrics, best_model_path)
            logger.info(f"  ★ New best model saved (IoU={best_iou:.4f})")

        # Early stopping (uses negative loss internally)
        early_stopping(-val_metrics["iou"], model, optimizer, epoch)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    elapsed = time.time() - start_time
    logger.info(f"\nTraining complete in {elapsed / 60:.1f} min  |  Best Val IoU: {best_iou:.4f}")

    # ------------------------------------------------------------------
    # Final evaluation on test set
    # ------------------------------------------------------------------
    if test_loader:
        logger.info("\n--- Evaluating on TEST set ---")
        # Reload best weights
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics = validate(model, test_loader, criterion, device)
        logger.info(
            f"  Test Loss={test_metrics['loss']:.4f}  "
            f"IoU={test_metrics['iou']:.4f}  "
            f"Dice={test_metrics['dice']:.4f}  "
            f"Precision={test_metrics['precision']:.4f}  "
            f"Recall={test_metrics['recall']:.4f}"
        )

    tb_writer.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
