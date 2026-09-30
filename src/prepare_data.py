import os
import shutil
import argparse
import logging
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict
import random

import numpy as np
import tifffile
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def extract_group_key(filename: str) -> str:
    """
    Extract a grouping key from the filename to prevent data leakage.
    Parses filenames like 'SP Over NB 09-05-25 Stn-1---2.tif'.
    
    Args:
        filename (str): Name of the file.
        
    Returns:
        str: Grouping key (e.g., date and station).
    """
    # Simple heuristic: remove trailing numbers, extensions, etc., to find the base group
    # Using regex to capture Date and Station
    # Example match: 09-05-25 Stn-1
    pattern = r"(\d{2}-\d{2}-\d{2})\s*(Stn-\d+)"
    match = re.search(pattern, filename, re.IGNORECASE)
    
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    
    # Fallback to the whole name if regex doesn't match
    base = Path(filename).stem
    # Remove things like "---2", "---1"
    base = re.sub(r'---.*$', '', base)
    return base

def group_files(image_paths: List[Path]) -> Dict[str, List[Path]]:
    """Group files by their extracted group key."""
    groups = defaultdict(list)
    for p in image_paths:
        key = extract_group_key(p.name)
        groups[key].append(p)
    return groups

def create_placeholder_mask(image_path: Path, output_mask_path: Path) -> None:
    """Create a black placeholder mask of the same spatial dimensions as the image."""
    try:
        if image_path.suffix.lower() in ['.tif', '.tiff']:
            img = tifffile.imread(str(image_path))
        else:
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Save as PNG by default for placeholder if not forced otherwise
        cv2.imwrite(str(output_mask_path), mask)
    except Exception as e:
        logger.error(f"Failed to create placeholder mask for {image_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for shot peening deformation detection.")
    parser.add_argument("--src_images", type=str, required=True, help="Directory containing raw images.")
    parser.add_argument("--src_masks", type=str, default=None, help="Directory containing raw masks. If not provided, black masks will be generated.")
    parser.add_argument("--out_dir", type=str, default="../data", help="Output directory for train/val/test splits.")
    parser.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15], help="Train/val/test split ratios.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)
    
    src_images_dir = Path(args.src_images)
    src_masks_dir = Path(args.src_masks) if args.src_masks else None
    out_dir = Path(args.out_dir)

    if not src_images_dir.exists():
        logger.error(f"Source images directory {src_images_dir} does not exist.")
        return

    # Find all images
    image_paths = []
    for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(src_images_dir.glob(ext))
    
    if not image_paths:
        logger.error(f"No images found in {src_images_dir}")
        return

    # Group files to prevent leakage
    grouped_files = group_files(image_paths)
    groups = list(grouped_files.keys())
    random.shuffle(groups)
    
    logger.info(f"Found {len(image_paths)} images across {len(groups)} distinct groups.")

    # Calculate split indices
    n_groups = len(groups)
    train_end = int(n_groups * args.split[0])
    val_end = train_end + int(n_groups * args.split[1])
    
    train_groups = groups[:train_end]
    val_groups = groups[train_end:val_end]
    test_groups = groups[val_end:]
    
    split_mapping = {
        "train": train_groups,
        "val": val_groups,
        "test": test_groups
    }

    # Process and copy files
    stats = {"train": 0, "val": 0, "test": 0}
    
    for split_name, split_groups in split_mapping.items():
        split_img_dir = out_dir / split_name / "images"
        split_msk_dir = out_dir / split_name / "masks"
        
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_msk_dir.mkdir(parents=True, exist_ok=True)
        
        for grp in split_groups:
            for img_path in grouped_files[grp]:
                # Copy image
                dst_img = split_img_dir / img_path.name
                shutil.copy2(img_path, dst_img)
                
                # Handle mask
                if src_masks_dir:
                    # Find corresponding mask
                    mask_found = False
                    for ext in [img_path.suffix, ".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
                        potential_mask = src_masks_dir / (img_path.stem + ext)
                        if potential_mask.exists():
                            shutil.copy2(potential_mask, split_msk_dir / potential_mask.name)
                            mask_found = True
                            break
                    
                    if not mask_found:
                        logger.warning(f"Mask not found for {img_path.name}, generating placeholder.")
                        create_placeholder_mask(img_path, split_msk_dir / (img_path.stem + ".png"))
                else:
                    # Generate placeholder
                    create_placeholder_mask(img_path, split_msk_dir / (img_path.stem + ".png"))
                    
                stats[split_name] += 1

    # Print statistics
    logger.info("Dataset preparation complete.")
    logger.info("Split Statistics:")
    logger.info(f"  Train: {stats['train']} images")
    logger.info(f"  Val:   {stats['val']} images")
    logger.info(f"  Test:  {stats['test']} images")
    logger.info(f"Data saved to {out_dir.absolute()}")

if __name__ == "__main__":
    main()
