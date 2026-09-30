"""
Inference script for Shot Peening Surface Deformation Detection.

Scans input images (single or batch) and outputs:
  1. Binary segmentation mask
  2. Overlay visualization (green=deformed, red=non-deformed)
  3. Text coverage report

Usage:
    # Single image
    python src/predict.py --model_path outputs/models/best_model.pth \
                          --input path/to/image.tif \
                          --output_dir results/

    # Batch directory
    python src/predict.py --model_path outputs/models/best_model.pth \
                          --input path/to/images/ \
                          --output_dir results/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import tifffile
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from model import create_model
from utils import create_overlay, generate_coverage_report, visualize_prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ImageNet normalization constants
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """Load an image from disk as RGB uint8 numpy array."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        img = tifffile.imread(str(path))
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        raise ValueError(f"Could not read image: {path}")

    # Convert grayscale → RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Ensure uint8
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    return img


def preprocess(image: np.ndarray, img_size: int) -> torch.Tensor:
    """Resize, normalize, and convert to (1, 3, H, W) tensor."""
    img = cv2.resize(image, (img_size, img_size)).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def predict_single(
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    img_size: int = 512,
    threshold: float = 0.5,
    output_dir: Optional[Path] = None,
    pixel_size_mm: Optional[float] = None,
) -> Dict:
    """Run inference on a single image and save outputs.

    Returns:
        Coverage report dictionary.
    """
    # Load & preprocess
    orig_img = load_image(image_path)
    orig_h, orig_w = orig_img.shape[:2]
    tensor = preprocess(orig_img, img_size).to(device)

    # Inference
    with torch.no_grad(), autocast():
        logits = model(tensor)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Binary mask at original resolution
    mask = (probs > threshold).astype(np.uint8)
    mask_full = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Coverage report
    report = generate_coverage_report(mask_full, pixel_size_mm=pixel_size_mm)

    # Save outputs
    if output_dir is not None:
        stem = image_path.stem

        # 1. Binary mask (0/255)
        mask_save = (mask_full * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"{stem}_mask.png"), mask_save)

        # 2. Overlay visualization
        overlay = create_overlay(orig_img.copy(), mask_full)
        cv2.imwrite(
            str(output_dir / f"{stem}_overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )

        # 3. Text report
        report_path = output_dir / f"{stem}_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Image: {image_path.name}\n")
            f.write(f"Resolution: {orig_w} x {orig_h}\n")
            f.write("-" * 40 + "\n")
            for key, val in report.items():
                f.write(f"{key}: {val}\n")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference — shot peening deformation detection"
    )
    p.add_argument("--model_path", type=str, required=True, help="Path to trained model .pth checkpoint")
    p.add_argument("--input", type=str, required=True, help="Single image path or directory of images")
    p.add_argument("--output_dir", type=str, default="results", help="Output directory (default: results)")
    p.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold (default: 0.5)")
    p.add_argument("--img_size", type=int, default=512, help="Model input size (default: 512)")
    p.add_argument("--architecture", type=str, default="Unet", choices=["Unet", "UnetPlusPlus", "DeepLabV3Plus"])
    p.add_argument("--encoder", type=str, default="resnet34", help="Encoder name (must match training)")
    p.add_argument("--pixel_size_mm", type=float, default=None, help="Physical size of one pixel in mm (optional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    model = create_model(
        architecture=args.architecture,
        encoder_name=args.encoder,
        encoder_weights=None,  # will load trained weights
        classes=1,
    ).to(device)

    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded model from {args.model_path}")

    # Collect input images
    input_path = Path(args.input)
    image_paths: List[Path] = []

    if input_path.is_file():
        image_paths.append(input_path)
    elif input_path.is_dir():
        for f in sorted(input_path.iterdir()):
            if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_paths.append(f)
    else:
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    if not image_paths:
        logger.error("No supported images found.")
        sys.exit(1)

    logger.info(f"Found {len(image_paths)} image(s) to process.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process
    all_reports: List[Dict] = []

    for img_path in tqdm(image_paths, desc="Processing"):
        try:
            report = predict_single(
                image_path=img_path,
                model=model,
                device=device,
                img_size=args.img_size,
                threshold=args.threshold,
                output_dir=output_dir,
                pixel_size_mm=args.pixel_size_mm,
            )
            report["image"] = img_path.name
            all_reports.append(report)
        except Exception as e:
            logger.error(f"Failed on {img_path.name}: {e}")

    # Summary
    if all_reports:
        coverages = [r["coverage_percentage"] for r in all_reports]
        defect_counts = [r["num_defect_regions"] for r in all_reports]

        logger.info("\n" + "=" * 50)
        logger.info("BATCH SUMMARY")
        logger.info("=" * 50)
        logger.info(f"  Images processed : {len(all_reports)}")
        logger.info(f"  Avg coverage     : {np.mean(coverages):.1f}%")
        logger.info(f"  Min coverage     : {np.min(coverages):.1f}%")
        logger.info(f"  Max coverage     : {np.max(coverages):.1f}%")
        logger.info(f"  Avg defect regions: {np.mean(defect_counts):.1f}")

        # Save aggregate JSON report
        with open(output_dir / "batch_summary.json", "w") as f:
            json.dump(all_reports, f, indent=2, default=str)
        logger.info(f"  Saved batch summary to {output_dir / 'batch_summary.json'}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
