"""
Step 2 — Prepare Data.

Converts LabelMe JSON annotations → binary mask PNGs, then
splits all images and masks into train / val / test folders.

Usage:
    python run_prepare_data.py
    python run_prepare_data.py --image_dir data/raw/images --anno_dir data/raw/annotations
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "data" / "raw" / "images"       # original TIF images
ANNO_DIR  = BASE_DIR / "data" / "raw" / "annotations"  # LabelMe JSON files
MASK_DIR  = BASE_DIR / "data" / "raw" / "masks"        # output binary masks
DATA_DIR  = BASE_DIR / "data"                           # final split output


def print_banner():
    print()
    print("=" * 60)
    print("  Shot Peening AI — Step 2: Prepare Data")
    print("=" * 60)
    print()


def convert_annotations(anno_dir: Path, mask_dir: Path) -> bool:
    """Run tools/labelme_to_masks.py to convert JSON → PNG masks."""
    json_files = list(anno_dir.glob("*.json"))
    if not json_files:
        print(f"[ERROR] No annotation JSON files found in {anno_dir}")
        print(f"        Please annotate images first:  python run_annotation.py")
        return False

    print(f"[INFO] Converting {len(json_files)} annotation(s) to binary masks...")
    result = subprocess.run(
        [sys.executable,
         str(BASE_DIR / "tools" / "labelme_to_masks.py"),
         "--json_dir",    str(anno_dir),
         "--output_dir",  str(mask_dir)],
        check=False,
    )
    if result.returncode != 0:
        print("[ERROR] Annotation conversion failed. See messages above.")
        return False

    masks = list(mask_dir.glob("*.png"))
    print(f"[INFO] {len(masks)} mask(s) created in {mask_dir}")
    return True


def split_data(image_dir: Path, mask_dir: Path, data_dir: Path) -> bool:
    """Run src/prepare_data.py to create train/val/test splits."""
    images = (
        list(image_dir.glob("*.tif")) +
        list(image_dir.glob("*.tiff")) +
        list(image_dir.glob("*.png"))
    )
    masks  = list(mask_dir.glob("*.png"))

    if not images:
        print(f"[ERROR] No images found in {image_dir}")
        return False
    if not masks:
        print(f"[ERROR] No masks found in {mask_dir}")
        return False

    print()
    print(f"[INFO] Splitting {len(images)} images + {len(masks)} masks into train/val/test...")
    result = subprocess.run(
        [sys.executable,
         str(BASE_DIR / "src" / "prepare_data.py"),
         "--images_dir", str(image_dir),
         "--masks_dir",  str(mask_dir),
         "--output_dir", str(data_dir)],
        check=False,
    )
    if result.returncode != 0:
        print("[ERROR] Data split failed. See messages above.")
        return False
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert annotations and split data for training")
    p.add_argument("--image_dir", type=str, default=str(IMAGE_DIR))
    p.add_argument("--anno_dir",  type=str, default=str(ANNO_DIR))
    p.add_argument("--mask_dir",  type=str, default=str(MASK_DIR))
    p.add_argument("--data_dir",  type=str, default=str(DATA_DIR))
    p.add_argument("--skip_convert", action="store_true",
                   help="Skip annotation→mask conversion (if masks already exist)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    anno_dir  = Path(args.anno_dir)
    mask_dir  = Path(args.mask_dir)
    data_dir  = Path(args.data_dir)

    mask_dir.mkdir(parents=True, exist_ok=True)

    print_banner()

    # Step A: annotations → masks
    if not args.skip_convert:
        ok = convert_annotations(anno_dir, mask_dir)
        if not ok:
            sys.exit(1)
    else:
        print("[INFO] Skipping annotation conversion (--skip_convert).")

    # Step B: split into train/val/test
    ok = split_data(image_dir, mask_dir, data_dir)
    if not ok:
        sys.exit(1)

    print()
    print("=" * 60)
    print("  Data preparation complete!")
    print()
    print("  Next steps:")
    print("    Option A — Train LOCALLY (slow, CPU only):")
    print("      python run_train.py")
    print()
    print("    Option B — Train on Google Colab (recommended, free GPU):")
    print("      1. ZIP your data/ folder")
    print("      2. Open shot_peening_training.ipynb in Colab")
    print("      3. Upload the ZIP and run all cells")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
