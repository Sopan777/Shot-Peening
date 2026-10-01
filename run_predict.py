"""
Step 4 — Run Predictions.

Scans shot-peened surface images and outputs:
  - *_mask.png     : binary mask (white = non-deformed areas)
  - *_overlay.png  : colour overlay (red = non-deformed, green = deformed)
  - *_report.txt   : per-image coverage statistics
  - batch_summary.json : summary across all images

CPU-optimised for Intel Core Ultra 5 235:
  - Uses all 16 threads
  - Default img_size=256 for speed (matches training default)

Usage:
    python run_predict.py
    python run_predict.py --input path/to/image.tif
    python run_predict.py --input path/to/images_dir --output_dir results/
    python run_predict.py --model_path custom/model.pth --threshold 0.4
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent
MODEL_PATH  = BASE_DIR / "outputs" / "models" / "best_model.pth"
INPUT_DIR   = BASE_DIR / "data" / "raw" / "images"
OUTPUT_DIR  = BASE_DIR / "results"
CPU_THREADS = "16"


def print_banner(model_path: Path, input_path: Path, output_dir: Path) -> None:
    print()
    print("=" * 60)
    print("  Shot Peening AI — Step 4: Predict Deformation")
    print("=" * 60)
    print()
    print(f"  Model   : {model_path}")
    print(f"  Input   : {input_path}")
    print(f"  Output  : {output_dir}")
    print()
    print("  Output files per image:")
    print("    *_overlay.png  → colour overlay (red=non-deformed)")
    print("    *_mask.png     → binary mask (white=non-deformed)")
    print("    *_report.txt   → coverage % and defect statistics")
    print("    batch_summary.json → all results combined")
    print()


def check_model(model_path: Path) -> None:
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print()
        print("  You need to train first. Options:")
        print("  A) Local CPU training:  python run_train.py")
        print("  B) Google Colab (faster):")
        print("       https://colab.research.google.com/github/Sopan777/Shot-Peening/blob/ai-model/shot_peening_training.ipynb")
        print("     Then download best_model.pth and place it at:")
        print(f"       {model_path}")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run shot peening deformation detection on images")
    p.add_argument("--model_path",   type=str, default=str(MODEL_PATH),
                   help=f"Path to trained model .pth (default: {MODEL_PATH})")
    p.add_argument("--input",        type=str, default=str(INPUT_DIR),
                   help=f"Single image or directory (default: {INPUT_DIR})")
    p.add_argument("--output_dir",   type=str, default=str(OUTPUT_DIR),
                   help=f"Directory for results (default: {OUTPUT_DIR})")
    p.add_argument("--threshold",    type=float, default=0.5,
                   help="Binarization threshold 0–1 (default: 0.5)")
    p.add_argument("--img_size",     type=int,   default=256,
                   help="Model input size — must match training size (default: 256)")
    p.add_argument("--encoder",      type=str,   default="resnet34")
    p.add_argument("--architecture", type=str,   default="Unet",
                   choices=["Unet", "UnetPlusPlus", "DeepLabV3Plus"])
    p.add_argument("--pixel_size_mm", type=float, default=None,
                   help="Physical size of one pixel in mm (optional, for area calculations)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    print_banner(model_path, input_path, output_dir)
    check_model(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set CPU threading
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = CPU_THREADS
    env["MKL_NUM_THREADS"] = CPU_THREADS

    cmd = [
        sys.executable,
        str(BASE_DIR / "src" / "predict.py"),
        "--model_path",   str(model_path),
        "--input",        str(input_path),
        "--output_dir",   str(output_dir),
        "--threshold",    str(args.threshold),
        "--img_size",     str(args.img_size),
        "--architecture", args.architecture,
        "--encoder",      args.encoder,
    ]
    if args.pixel_size_mm is not None:
        cmd += ["--pixel_size_mm", str(args.pixel_size_mm)]

    print("[INFO] Running inference...")
    print()

    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode == 0:
        print()
        print("=" * 60)
        print(f"  Results saved to: {output_dir.resolve()}")
        print("=" * 60)
    else:
        print("[ERROR] Prediction failed. See messages above.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
