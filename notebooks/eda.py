import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from src import config

def inspect_dataset(image_dir: Path, labels_csv: Path):
    """
    Analyzes SEM image dataset properties (resolutions, channels, intensity ranges)
    and coverage percentage distribution.
    """
    print(f"\n{'='*55}")
    print("      SHOT PEENING DATASET EXPLORATORY ANALYSIS")
    print(f"{'='*55}")

    # 1. Analyze Images
    image_extensions = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]

    print(f"[*] Total image files found in {image_dir}: {len(image_files)}")

    if not image_files:
        print("[!] No images found! Place your .tif files into data/images/")
        return

    # Sample up to 50 images for metadata checks
    sample_sizes = []
    modes = []
    bit_depths = []

    for f in image_files[:50]:
        try:
            with Image.open(f) as img:
                sample_sizes.append(img.size)  # (width, height)
                modes.append(img.mode)
        except Exception as e:
            print(f"[!] Warning reading {f.name}: {e}")

    unique_sizes = list(set(sample_sizes))
    unique_modes = list(set(modes))

    print(f"[*] Detected Image Dimensions (Width x Height): {unique_sizes}")
    print(f"[*] Detected Color Modes:                       {unique_modes}")

    # 2. Analyze Labels if CSV exists
    if labels_csv.exists():
        print(f"\n[*] Found labels CSV at: {labels_csv}")
        df = pd.read_csv(labels_csv)
        print(f"[*] CSV Columns: {list(df.columns)}")
        print(f"[*] Total rows in CSV: {len(df)}")

        # Auto-detect label col
        from src.dataset import find_label_column, find_filename_column
        try:
            label_col = find_label_column(df)
            fname_col = find_filename_column(df)
            print(f"[*] Using '{fname_col}' for Filenames and '{label_col}' for Coverage %.")

            coverage_vals = df[label_col].astype(float)

            print("\nCoverage % Statistics:")
            stats = coverage_vals.describe()
            print(stats.to_string())

            # Plot distribution
            plt.figure(figsize=(8, 5))
            sns.set_style("whitegrid")
            sns.histplot(coverage_vals, kde=True, color="#0288D1", bins=20)
            plt.axvline(coverage_vals.mean(), color="red", linestyle="--", label=f"Mean: {coverage_vals.mean():.1f}%")
            plt.axvline(coverage_vals.median(), color="orange", linestyle=":", label=f"Median: {coverage_vals.median():.1f}%")
            plt.xlabel("Coverage Percentage (%)", fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.title("Distribution of Shot Peening Coverage Values", fontsize=13)
            plt.legend()
            plt.tight_layout()

            plot_path = config.PLOT_DIR / "eda_coverage_distribution.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"[OK] Saved coverage distribution plot to: {plot_path}")

        except Exception as e:
            print(f"[!] Error parsing labels CSV: {e}")
    else:
        print(f"\n[i] Labels CSV not found at {labels_csv}.")
        print("[i] Generating a template 'labels_template.csv' for you...")
        template_records = [{"filename": f.name, "coverage_percentage": ""} for f in image_files]
        template_df = pd.DataFrame(template_records)
        template_path = config.DATA_DIR / "labels_template.csv"
        template_df.to_csv(template_path, index=False)
        print(f"[OK] Created template at: {template_path}")
        print("    You can fill in the 'coverage_percentage' column and rename it to 'labels.csv'!")

def main():
    parser = argparse.ArgumentParser(description="EDA for Shot Peening SEM Dataset")
    parser.add_argument("--images", type=str, default=str(config.IMAGE_DIR))
    parser.add_argument("--csv", type=str, default=str(config.LABELS_FILE))
    args = parser.parse_args()

    inspect_dataset(Path(args.images), Path(args.csv))

if __name__ == "__main__":
    main()
