"""
TIF → PNG Converter for LabelMe Compatibility.

LabelMe does not natively support .tif/.tiff files.
Run this script FIRST to convert all your TIF images to PNG,
then annotate the PNGs in LabelMe.

The original TIF files are NOT deleted — they are kept for training
(higher quality). Only PNGs are used for annotation purposes.

Usage:
    python tools/tif_to_png.py
    python tools/tif_to_png.py --input_dir data/raw/images --output_dir data/raw/images_png
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def convert_tif_to_png(tif_path: Path, output_dir: Path) -> Path:
    """
    Convert a single TIF image to PNG.

    Handles:
      - Grayscale, RGB, RGBA TIF images
      - 16-bit TIFs (normalised to 8-bit for LabelMe)
      - Multi-page TIFs (first page only)

    Args:
        tif_path:   Path to source .tif file.
        output_dir: Directory where the .png will be saved.

    Returns:
        Path to the saved PNG file.
    """
    img = tifffile.imread(str(tif_path))

    # Multi-page TIF — take the first page
    if img.ndim == 4:
        img = img[0]

    # 16-bit → 8-bit normalisation
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        # Normalise any other dtype to 0–255
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)

    # Grayscale → RGB (LabelMe works better with RGB)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        # RGBA → RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    out_path = output_dir / (tif_path.stem + ".png")
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TIF images to PNG for LabelMe annotation"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw/images",
        help="Folder containing .tif images (default: data/raw/images)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw/images_png",
        help="Folder where PNGs will be saved (default: data/raw/images_png)",
    )
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(
        list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    )

    if not tif_files:
        logger.warning(f"No .tif/.tiff files found in {input_dir}")
        sys.exit(0)

    logger.info(f"Found {len(tif_files)} TIF file(s). Converting to PNG...")
    logger.info(f"Output → {output_dir.resolve()}")
    logger.info("")

    success, failed = 0, 0
    for tif in tif_files:
        try:
            out = convert_tif_to_png(tif, output_dir)
            logger.info(f"  ✓  {tif.name:50s} → {out.name}")
            success += 1
        except Exception as e:
            logger.error(f"  ✗  {tif.name}: {e}")
            failed += 1

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Done.  ✓ {success} converted   ✗ {failed} failed")
    logger.info("")
    logger.info("Next step:")
    logger.info(f"  python run_annotation.py")
    logger.info(f"  (opens LabelMe on {output_dir.resolve()})")


if __name__ == "__main__":
    main()
