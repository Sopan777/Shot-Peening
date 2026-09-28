import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from typing import List, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src import config
from src.model import build_model
from src.transforms import get_val_transforms

def load_inference_model(checkpoint_path: Path, device: torch.device):
    """Loads model weights for inference."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("config", {}).get("model_name", config.MODEL_NAME)
    image_size = checkpoint.get("config", {}).get("image_size", config.IMAGE_SIZE)

    model = build_model(model_name=model_name, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = get_val_transforms(image_size)
    return model, transform


def predict_single_image(model, transform, image_path: Path, device: torch.device) -> float:
    """Predicts coverage % for a single SEM image file."""
    with Image.open(image_path) as pil_img:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        img_np = np.array(pil_img)

    augmented = transform(image=img_np)
    tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred_pct = float(output.cpu().item() * 100.0)

    return round(pred_pct, 2)


def interpret_coverage(coverage_pct: float) -> str:
    """Provides engineering context for coverage measurement."""
    if coverage_pct < 60.0:
        return "Under-peened (Sparse crater distribution, low dimple overlap)"
    elif 60.0 <= coverage_pct < 85.0:
        return "Moderate peening (Approaching nominal dimple density)"
    elif 85.0 <= coverage_pct < 98.0:
        return "Nominal full coverage (High density overlapping dimples)"
    else:
        return "100%+ Full saturation / Over-peened surface"


def main():
    parser = argparse.ArgumentParser(description="Run Inference on Shot Peened SEM Images")
    parser.add_argument("--image", type=str, help="Path to a single SEM .tif image")
    parser.add_argument("--folder", type=str, help="Directory containing SEM .tif images for batch inference")
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "best_model.pth"))
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Output CSV for batch predictions")
    args = parser.parse_args()

    device = config.DEVICE
    checkpoint_path = Path(args.checkpoint)

    print(f"[*] Loading model on {device}...")
    model, transform = load_inference_model(checkpoint_path, device)

    # 1. Single Image Mode
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"[!] Error: Image not found at {img_path}")
            sys.exit(1)

        pct = predict_single_image(model, transform, img_path, device)
        interpretation = interpret_coverage(pct)

        print("\n" + "="*55)
        print(f"Image:          {img_path.name}")
        print(f"Coverage:       {pct:.2f}%")
        print(f"Interpretation: {interpretation}")
        print("="*55)

    # 2. Batch Directory Mode
    elif args.folder:
        folder_path = Path(args.folder)
        if not folder_path.exists():
            print(f"[!] Error: Folder not found at {folder_path}")
            sys.exit(1)

        image_extensions = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
        image_files = [f for f in folder_path.iterdir() if f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"[!] No valid image files found in {folder_path}")
            sys.exit(1)

        print(f"[*] Running inference on {len(image_files)} images...")
        records = []
        for f in image_files:
            pct = predict_single_image(model, transform, f, device)
            interp = interpret_coverage(pct)
            records.append({
                "Filename": f.name,
                "Predicted_Coverage_%": pct,
                "Interpretation": interp
            })

        df = pd.DataFrame(records)
        df.to_csv(args.output_csv, index=False)
        print(f"[OK] Batch inference complete! Saved results to {args.output_csv}")
        print("\nFirst 5 predictions:")
        print(df.head(5).to_string(index=False))

    else:
        print("[!] Please provide either --image <path> or --folder <path>")
        parser.print_help()

if __name__ == "__main__":
    main()
