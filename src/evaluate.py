import sys
import argparse
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.dataset import prepare_datasets
from src.model import build_model
from src.transforms import get_val_transforms

def run_evaluation(
    checkpoint_path: Path,
    csv_path: Path,
    image_dir: Path,
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = config.DEVICE

    print(f"[*] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("config", {}).get("model_name", config.MODEL_NAME)
    image_size = checkpoint.get("config", {}).get("image_size", config.IMAGE_SIZE)

    # Rebuild model architecture
    model = build_model(model_name=model_name, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load test set
    val_transform = get_val_transforms(image_size)
    _, _, test_ds, _ = prepare_datasets(
        csv_path=csv_path,
        image_dir=image_dir,
        train_transform=val_transform,
        val_transform=val_transform,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        random_seed=config.RANDOM_SEED
    )

    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    predictions = []
    actuals = []
    file_paths = [str(p.name) for p in test_ds.image_paths]

    print("[*] Generating predictions on test set...")
    with torch.no_grad():
        for images, targets in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)

            # Denormalize to percentage
            preds_pct = outputs.cpu().numpy() * 100.0
            actual_pct = targets.cpu().numpy() * 100.0

            predictions.extend(preds_pct.tolist())
            actuals.extend(actual_pct.tolist())

    preds_np = np.array(predictions)
    actuals_np = np.array(actuals)
    residuals = preds_np - actuals_np
    abs_errors = np.abs(residuals)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_tot = np.sum((actuals_np - np.mean(actuals_np)) ** 2)
    ss_res = np.sum(residuals ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-6 else 0.0

    print("\n" + "="*50)
    print("           DETAILED TEST EVALUATION REPORT")
    print("="*50)
    print(f"Total Test Samples:   {len(actuals_np)}")
    print(f"MAE:                  {mae:.2f}% Coverage")
    print(f"RMSE:                 {rmse:.2f}% Coverage")
    print(f"R² Score:             {r2:.4f}")
    print(f"Mean Residual:        {np.mean(residuals):+.2f}%")
    print(f"Max Absolute Error:   {np.max(abs_errors):.2f}%")
    print("="*50)

    # 1. Scatter Plot: Actual vs Predicted
    plt.figure(figsize=(7, 6))
    sns.set_style("whitegrid")
    plt.scatter(actuals_np, preds_np, alpha=0.7, color="#1976D2", edgecolors="k", s=50)

    # 45 degree perfect line
    min_val = min(float(np.min(actuals_np)), float(np.min(preds_np))) - 5
    max_val = max(float(np.max(actuals_np)), float(np.max(preds_np))) + 5
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal (y = x)")

    plt.xlabel("Actual Coverage (%)", fontsize=12)
    plt.ylabel("Predicted Coverage (%)", fontsize=12)
    plt.title(f"Shot Peening Surface Coverage: Actual vs Predicted\n(R² = {r2:.3f}, MAE = {mae:.2f}%)", fontsize=13)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend(loc="upper left")
    plt.tight_layout()

    scatter_path = output_dir / "scatter_predicted_vs_actual.png"
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"[OK] Saved scatter plot to: {scatter_path}")

    # 2. Residual Distribution Plot
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True, color="#43A047", bins=20)
    plt.axvline(0, color="red", linestyle="--", lw=1.5, label="Zero Error")
    plt.xlabel("Residual (Predicted - Actual) %", fontsize=12)
    plt.ylabel("Sample Count", fontsize=12)
    plt.title("Error Residual Distribution", fontsize=13)
    plt.legend()
    plt.tight_layout()

    residual_path = output_dir / "residual_distribution.png"
    plt.savefig(residual_path, dpi=300)
    plt.close()
    print(f"[OK] Saved residual distribution to: {residual_path}")

    # 3. Save Detailed Results DataFrame
    results_df = pd.DataFrame({
        "Filename": file_paths,
        "Actual_Coverage_%": np.round(actuals_np, 2),
        "Predicted_Coverage_%": np.round(preds_np, 2),
        "Residual_%": np.round(residuals, 2),
        "Absolute_Error_%": np.round(abs_errors, 2)
    })
    results_df = results_df.sort_values(by="Absolute_Error_%", ascending=False)
    csv_results_path = output_dir / "test_predictions_detailed.csv"
    results_df.to_csv(csv_results_path, index=False)
    print(f"[OK] Saved per-sample predictions to: {csv_results_path}")

    print("\nTop 5 Largest Error Samples (Outlier Inspection):")
    print(results_df.head(5).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Shot Peening Model")
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "best_model.pth"))
    parser.add_argument("--csv", type=str, default=str(config.LABELS_FILE))
    parser.add_argument("--images", type=str, default=str(config.IMAGE_DIR))
    parser.add_argument("--output", type=str, default=str(config.PLOT_DIR))
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=Path(args.checkpoint),
        csv_path=Path(args.csv),
        image_dir=Path(args.images),
        output_dir=Path(args.output)
    )

if __name__ == "__main__":
    main()
