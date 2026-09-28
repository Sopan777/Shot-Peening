import sys
import argparse
import json
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src import config
from src.dataset import prepare_datasets
from src.transforms import get_train_transforms, get_val_transforms
from src.model import build_model

def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    """Runs evaluation on validation or test set, returns metrics in true % coverage units."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * len(targets)

            # Convert 0..1 scale back to 0..100%
            all_preds.extend((outputs.cpu().numpy() * 100.0).tolist())
            all_targets.extend((targets.cpu().numpy() * 100.0).tolist())

    total_samples = len(all_preds)
    mean_loss = total_loss / max(1, total_samples)

    preds_np = np.array(all_preds)
    targets_np = np.array(all_targets)

    mae = float(np.mean(np.abs(preds_np - targets_np)))
    rmse = float(np.sqrt(np.mean((preds_np - targets_np) ** 2)))
    max_err = float(np.max(np.abs(preds_np - targets_np)))

    # R2 Score
    ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
    ss_res = np.sum((targets_np - preds_np) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-6 else 0.0

    return {
        "loss": mean_loss,
        "mae_pct": mae,
        "rmse_pct": rmse,
        "max_err_pct": max_err,
        "r2_score": r2
    }


def train_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    device: torch.device,
    stage_name: str,
    writer: SummaryWriter,
    checkpoint_path: Path,
    patience: int = 10,
    use_bf16: bool = True
) -> Dict[str, Any]:
    """Executes a single training stage (Frozen linear probe or Fine-tuning)."""
    best_val_mae = float("inf")
    patience_counter = 0
    history = []

    print(f"\n{'='*25} STARTING {stage_name.upper()} ({epochs} EPOCHS) {'='*25}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"[{stage_name}] Epoch {epoch:02d}/{epochs:02d}")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Intel CPU BFloat16 mixed precision
            if device.type == "cpu" and use_bf16:
                with torch.cpu.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(targets)
            pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})

        if scheduler is not None:
            scheduler.step()

        epoch_train_loss = train_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        # TensorBoard logging
        step = epoch
        writer.add_scalar(f"{stage_name}/train_loss", epoch_train_loss, step)
        writer.add_scalar(f"{stage_name}/val_loss", val_metrics["loss"], step)
        writer.add_scalar(f"{stage_name}/val_mae_pct", val_metrics["mae_pct"], step)
        writer.add_scalar(f"{stage_name}/val_rmse_pct", val_metrics["rmse_pct"], step)
        writer.add_scalar(f"{stage_name}/val_r2", val_metrics["r2_score"], step)

        print(
            f"Epoch {epoch:02d} ({elapsed:.1f}s) | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val MAE: {val_metrics['mae_pct']:.2f}% | "
            f"Val RMSE: {val_metrics['rmse_pct']:.2f}% | "
            f"R²: {val_metrics['r2_score']:.3f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            **val_metrics
        })

        # Checkpoint if MAE improved
        if val_metrics["mae_pct"] < best_val_mae:
            best_val_mae = val_metrics["mae_pct"]
            patience_counter = 0
            torch.save({
                "stage": stage_name,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": {
                    "image_size": config.IMAGE_SIZE,
                    "model_name": config.MODEL_NAME
                }
            }, checkpoint_path)
            print(f"  [BEST] Saved new best checkpoint with Val MAE: {best_val_mae:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [!] Early stopping triggered after {patience} epochs without improvement.")
                break

    return {"history": history, "best_val_mae": best_val_mae}


def main():
    parser = argparse.ArgumentParser(description="Train Shot Peening Coverage Prediction Model")
    parser.add_argument("--model", type=str, default=config.MODEL_NAME, help="Model architecture")
    parser.add_argument("--csv", type=str, default=str(config.LABELS_FILE), help="Path to labels.csv")
    parser.add_argument("--images", type=str, default=str(config.IMAGE_DIR), help="Directory of .tif images")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs1", type=int, default=config.NUM_EPOCHS_STAGE1, help="Stage 1 (probe) epochs")
    parser.add_argument("--epochs2", type=int, default=config.NUM_EPOCHS_STAGE2, help="Stage 2 (fine-tune) epochs")
    parser.add_argument("--no_ipex", action="store_true", help="Disable Intel IPEX optimization")
    parser.add_argument("--no_stage2", action="store_true", help="Skip Stage 2 fine-tuning (probe only)")
    args = parser.parse_args()

    device = config.DEVICE
    print(f"[*] Training Device: {device} ({'Intel Core Ultra CPU' if device.type == 'cpu' else 'GPU'})")

    # 1. Prepare Datasets and DataLoaders
    print(f"[*] Loading data from CSV: {args.csv}")
    train_transform = get_train_transforms(config.IMAGE_SIZE)
    val_transform = get_val_transforms(config.IMAGE_SIZE)

    train_ds, val_ds, test_ds, split_summary = prepare_datasets(
        csv_path=args.csv,
        image_dir=args.images,
        train_transform=train_transform,
        val_transform=val_transform,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        random_seed=config.RANDOM_SEED
    )

    print("\nDataset Split Summary:")
    print(split_summary.to_string(index=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    # 2. Build Model
    model = build_model(model_name=args.model, dropout=config.DROPOUT_RATE, freeze_backbone=True)
    model.to(device)

    # Smooth L1 (Huber) Loss is robust to noise and outliers in measurements
    criterion = nn.SmoothL1Loss(beta=0.05)
    writer = SummaryWriter(log_dir=str(config.LOG_DIR))

    # Optional Intel Extension for PyTorch (IPEX)
    use_ipex = config.USE_IPEX and (not args.no_ipex) and (device.type == "cpu")
    if use_ipex:
        try:
            import intel_extension_for_pytorch as ipex
            print("[IPEX] Intel Extension for PyTorch (IPEX) loaded and enabled.")
        except ImportError:
            print("[i] IPEX not found. Proceeding with standard PyTorch CPU execution.")
            use_ipex = False

    # ==================== STAGE 1: LINEAR PROBING ====================
    # Fast training of regression head (~5-10 min)
    optimizer_s1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE_STAGE1,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s1, T_max=args.epochs1)

    if use_ipex:
        import intel_extension_for_pytorch as ipex
        model, optimizer_s1 = ipex.optimize(model, optimizer=optimizer_s1)

    ckpt_s1 = config.CHECKPOINT_DIR / "best_stage1_probe.pth"
    res_s1 = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_s1,
        scheduler=scheduler_s1,
        epochs=args.epochs1,
        device=device,
        stage_name="stage1_probe",
        writer=writer,
        checkpoint_path=ckpt_s1,
        patience=config.PATIENCE,
        use_bf16=config.USE_BF16
    )

    # Load best checkpoint from Stage 1
    if ckpt_s1.exists():
        best_ckpt = torch.load(ckpt_s1, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        print(f"\n[OK] Loaded Stage 1 best weights (Val MAE: {best_ckpt['val_metrics']['mae_pct']:.2f}%)")

    # ==================== STAGE 2: PARTIAL FINE-TUNING ====================
    if not args.no_stage2 and args.epochs2 > 0:
        if hasattr(model, "unfreeze"):
            model.unfreeze(num_blocks=4)
        elif hasattr(model, "backbone"):
            for p in model.backbone.parameters():
                p.requires_grad = True

        optimizer_s2 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.LEARNING_RATE_STAGE2,
            weight_decay=config.WEIGHT_DECAY
        )
        scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_s2, T_0=10, T_mult=2
        )

        if use_ipex:
            import intel_extension_for_pytorch as ipex
            model, optimizer_s2 = ipex.optimize(model, optimizer=optimizer_s2)

        ckpt_s2 = config.CHECKPOINT_DIR / "best_stage2_finetuned.pth"
        res_s2 = train_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer_s2,
            scheduler=scheduler_s2,
            epochs=args.epochs2,
            device=device,
            stage_name="stage2_finetune",
            writer=writer,
            checkpoint_path=ckpt_s2,
            patience=config.PATIENCE,
            use_bf16=config.USE_BF16
        )

        final_ckpt_to_evaluate = ckpt_s2 if ckpt_s2.exists() else ckpt_s1
    else:
        final_ckpt_to_evaluate = ckpt_s1

    # Save overall best model to best_model.pth
    best_final_path = config.CHECKPOINT_DIR / "best_model.pth"
    torch.save(torch.load(final_ckpt_to_evaluate, map_location=device), best_final_path)
    print(f"\n[BEST] Saved final operational model to: {best_final_path}")

    # ==================== FINAL TEST SET EVALUATION ====================
    print(f"\n{'='*25} EVALUATING ON INDEPENDENT TEST SET {'='*25}")
    best_saved = torch.load(best_final_path, map_location=device)
    model.load_state_dict(best_saved["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test MAE (Mean Absolute Error):   {test_metrics['mae_pct']:.2f}% Coverage")
    print(f"Test RMSE (Root Mean Sq Error):    {test_metrics['rmse_pct']:.2f}% Coverage")
    print(f"Test Max Error:                    {test_metrics['max_err_pct']:.2f}% Coverage")
    print(f"Test R^2 Score:                    {test_metrics['r2_score']:.4f}")

    # Save metrics JSON
    metrics_path = config.OUTPUT_DIR / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=4)
    print(f"[OK] Saved test metrics to {metrics_path}")

    writer.close()
    print("[OK] Training completed successfully!")

if __name__ == "__main__":
    main()
