"""
Step 1 — Annotation Launcher.

Converts TIF images to PNG (LabelMe compatible), then opens
LabelMe so you can annotate the non-deformed surface areas.

Usage:
    python run_annotation.py
    python run_annotation.py --image_dir data/raw/images --anno_dir data/raw/annotations
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── paths (relative to this script's location) ──────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
IMAGE_DIR  = BASE_DIR / "data" / "raw" / "images"        # original TIFs
PNG_DIR    = BASE_DIR / "data" / "raw" / "images_png"    # PNGs for LabelMe
ANNO_DIR   = BASE_DIR / "data" / "raw" / "annotations"   # LabelMe JSON output


def print_banner():
    print()
    print("=" * 60)
    print("  Shot Peening AI — Step 1: Annotation")
    print("=" * 60)
    print()


def print_instructions():
    print("HOW TO ANNOTATE IN LABELME")
    print("-" * 40)
    print("1. An image opens in LabelMe automatically.")
    print("2. Click  Edit → Create Polygons  (or press 'P').")
    print("3. Click around every NON-DEFORMED (smooth/unpeened) area.")
    print("4. When asked for a label, type:  non_deformed")
    print("5. Press Ctrl+S to save (autosave is also ON).")
    print("6. Use the right arrow key ▶ to move to the next image.")
    print()
    print("TIP: Use  Edit → Create Rectangle  for simple square regions.")
    print("TIP: Zoom with Ctrl+Scroll.  Pan by holding Ctrl and dragging.")
    print()
    print("When done annotating all images:")
    print("  → Close LabelMe and run:  python run_prepare_data.py")
    print()
    print("-" * 40)
    print()


def convert_tifs_to_png(image_dir: Path, png_dir: Path) -> int:
    """Convert TIF images to PNG using tools/tif_to_png.py."""
    tif_files = list(image_dir.glob("*.tif")) + list(image_dir.glob("*.tiff"))
    if not tif_files:
        print(f"[INFO] No TIF files found in {image_dir}")
        print(f"       Please copy your .tif images into:  {image_dir}")
        print()
        input("Press Enter after copying images to continue...")
        tif_files = list(image_dir.glob("*.tif")) + list(image_dir.glob("*.tiff"))

    png_dir.mkdir(parents=True, exist_ok=True)

    # Check which TIFs still need converting
    to_convert = []
    for tif in tif_files:
        png = png_dir / (tif.stem + ".png")
        if not png.exists():
            to_convert.append(tif)

    if to_convert:
        print(f"[INFO] Converting {len(to_convert)} TIF(s) to PNG for LabelMe...")
        result = subprocess.run(
            [sys.executable,
             str(BASE_DIR / "tools" / "tif_to_png.py"),
             "--input_dir", str(image_dir),
             "--output_dir", str(png_dir)],
            check=False,
        )
        if result.returncode != 0:
            print("[ERROR] TIF conversion failed. See messages above.")
            sys.exit(1)
    else:
        print(f"[INFO] All {len(tif_files)} TIF(s) already converted to PNG. Skipping.")

    return len(list(png_dir.glob("*.png")))


def launch_labelme(png_dir: Path, anno_dir: Path) -> None:
    """Launch LabelMe GUI on the PNG image directory."""
    anno_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Opening LabelMe...")
    print(f"       Images : {png_dir}")
    print(f"       Saving : {anno_dir}")
    print()

    cmd = [
        "labelme",
        str(png_dir),
        "--output", str(anno_dir),
        "--labels", "non_deformed,deformed",
        "--autosave",
        "--nodata",          # don't embed image bytes in JSON (saves space)
        "--keep-prev",       # remember previous polygon while navigating
    ]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("[ERROR] LabelMe not found. Install it with:")
        print("        pip install labelme")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] LabelMe exited with code {e.returncode}")
        sys.exit(e.returncode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch LabelMe annotation for shot peening images")
    p.add_argument("--image_dir", type=str, default=str(IMAGE_DIR),
                   help=f"Directory with original TIF images (default: {IMAGE_DIR})")
    p.add_argument("--anno_dir",  type=str, default=str(ANNO_DIR),
                   help=f"Directory to save LabelMe JSON files (default: {ANNO_DIR})")
    p.add_argument("--skip_convert", action="store_true",
                   help="Skip TIF→PNG conversion (use if PNGs already exist)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    anno_dir  = Path(args.anno_dir)

    print_banner()
    print_instructions()

    if not args.skip_convert:
        n = convert_tifs_to_png(image_dir, PNG_DIR)
        print(f"[INFO] {n} PNG(s) ready in {PNG_DIR}")
        print()

    launch_labelme(PNG_DIR, anno_dir)

    print()
    print("[INFO] LabelMe closed.")
    print("[INFO] Next step → python run_prepare_data.py")
    print()


if __name__ == "__main__":
    main()
