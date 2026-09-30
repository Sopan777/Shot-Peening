"""
Annotation Helper — converts LabelMe JSON annotations to binary mask images.

After you annotate images with LabelMe and export JSON files, run this script
to convert them into binary PNG masks compatible with the training pipeline.

Usage:
    # Convert a single JSON annotation
    python tools/labelme_to_masks.py --json_dir annotations/ --output_dir data/raw/masks/

    # Also copy the source images alongside
    python tools/labelme_to_masks.py --json_dir annotations/ --output_dir data/raw/masks/ --image_dir data/raw/images/

LabelMe annotation convention:
    - Label "non_deformed" (or "defect") → WHITE pixels (255) in mask = non-deformed area
    - Everything else                    → BLACK pixels (0)   in mask = properly peened
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Labels that indicate NON-DEFORMED (defect) regions
NON_DEFORMED_LABELS = {"non_deformed", "defect", "unpeened", "non_peened", "no_deformation"}


def polygon_to_mask(polygon_points: list, height: int, width: int) -> np.ndarray:
    """Convert a list of [x, y] polygon points to a binary mask."""
    pts = np.array(polygon_points, dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], color=255)
    return mask


def convert_json_to_mask(json_path: Path, output_dir: Path) -> bool:
    """
    Convert one LabelMe JSON annotation file to a binary mask PNG.

    Args:
        json_path:  Path to the .json file exported by LabelMe.
        output_dir: Directory where the mask PNG will be saved.

    Returns:
        True on success, False on error.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Cannot read {json_path.name}: {e}")
        return False

    height = data.get("imageHeight")
    width  = data.get("imageWidth")
    if not height or not width:
        logger.error(f"Missing imageHeight/imageWidth in {json_path.name}")
        return False

    # Start with all-black mask (everything assumed deformed/peened)
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    shapes = data.get("shapes", [])
    matched = 0
    for shape in shapes:
        label = shape.get("label", "").strip().lower()
        shape_type = shape.get("shape_type", "polygon")
        points = shape.get("points", [])

        if label not in NON_DEFORMED_LABELS:
            continue  # skip non-relevant labels

        if shape_type == "polygon" and len(points) >= 3:
            region_mask = polygon_to_mask(points, height, width)
            combined_mask = cv2.bitwise_or(combined_mask, region_mask)
            matched += 1
        elif shape_type == "rectangle" and len(points) == 2:
            x1, y1 = int(points[0][0]), int(points[0][1])
            x2, y2 = int(points[1][0]), int(points[1][1])
            combined_mask[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)] = 255
            matched += 1
        else:
            logger.warning(f"  Skipping unsupported shape type '{shape_type}' in {json_path.name}")

    if matched == 0:
        logger.warning(
            f"  No matching labels found in {json_path.name}. "
            f"Expected one of: {NON_DEFORMED_LABELS}. "
            f"Found labels: {[s.get('label','') for s in shapes]}"
        )

    # Save mask
    stem = Path(data.get("imagePath", json_path.stem)).stem
    out_path = output_dir / f"{stem}.png"
    cv2.imwrite(str(out_path), combined_mask)
    logger.info(f"  ✓ {json_path.name}  →  {out_path.name}  ({matched} regions)")
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert LabelMe JSON annotations to binary mask PNGs"
    )
    p.add_argument(
        "--json_dir", type=str, required=True,
        help="Directory containing LabelMe .json annotation files"
    )
    p.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for binary mask PNGs"
    )
    p.add_argument(
        "--labels", type=str, nargs="+",
        default=list(NON_DEFORMED_LABELS),
        help=f"Labels that map to non-deformed regions. Default: {list(NON_DEFORMED_LABELS)}"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Update label set if custom labels provided
    global NON_DEFORMED_LABELS
    NON_DEFORMED_LABELS = {l.strip().lower() for l in args.labels}
    logger.info(f"Non-deformed labels: {NON_DEFORMED_LABELS}")

    json_dir = Path(args.json_dir)
    output_dir = Path(args.output_dir)

    if not json_dir.exists():
        logger.error(f"JSON directory does not exist: {json_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        logger.error(f"No .json files found in {json_dir}")
        sys.exit(1)

    logger.info(f"Found {len(json_files)} annotation file(s). Converting...")

    success, failed = 0, 0
    for jf in json_files:
        if convert_json_to_mask(jf, output_dir):
            success += 1
        else:
            failed += 1

    logger.info("=" * 50)
    logger.info(f"Done. ✓ {success} converted   ✗ {failed} failed")
    logger.info(f"Masks saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
