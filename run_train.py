"""
Step 3 — Train the Model (Local CPU).

Launches training directly on this machine (CPU).
For Intel Core Ultra 5 235: sets OMP/MKL threads to use all 16 cores.

NOTE: Training on CPU is SLOW (~10-20× slower than GPU).
For faster training, use Google Colab:
    Open shot_peening_training.ipynb at:
    https://colab.research.google.com/github/Sopan777/Shot-Peening/blob/ai-model/shot_peening_training.ipynb

Usage:
    python run_train.py
    python run_train.py --batch_size 4 --img_size 256   # smaller for less RAM
    python run_train.py --epochs 50 --patience 10       # quicker experiment
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = BASE_DIR / "outputs"

# Intel Core Ultra 5 235 — use all 16 threads
CPU_THREADS = "16"


def print_banner(args: argparse.Namespace) -> None:
    print()
    print("=" * 60)
    print("  Shot Peening AI — Step 3: Train Model (CPU)")
    print("=" * 60)
    print()
    print("  Architecture : U-Net + ResNet-34 (ImageNet pretrained)")
    print(f"  Image size   : {args.img_size} × {args.img_size}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Max epochs   : {args.epochs}  (early stop patience={args.patience})")
    print(f"  CPU threads  : {CPU_THREADS}")
    print()
    print("  ⚠️  CPU training is slow. Estimated time on Intel Core Ultra 5 235:")
    if args.img_size <= 256:
        print("      ~30–60 min for 100 epochs at 256×256")
    else:
        print("      ~3–6 hours for 100 epochs at 512×512")
    print()
    print("  💡 Faster alternative: use Google Colab (free GPU, ~1–2 hrs)")
    print("     Open: shot_peening_training.ipynb")
    print()
    choice = input("  Continue on CPU? [y/N]: ").strip().lower()
    if choice != "y":
        print()
        print("  Cancelled. Open shot_peening_training.ipynb in Colab instead.")
        sys.exit(0)
    print()


def check_data(data_dir: Path) -> None:
    train_images = list((data_dir / "train" / "images").glob("*")) if (data_dir / "train" / "images").exists() else []
    if not train_images:
        print(f"[ERROR] No training images found in {data_dir / 'train' / 'images'}")
        print("        Run  python run_prepare_data.py  first.")
        sys.exit(1)
    print(f"[INFO] Training images found: {len(train_images)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train shot peening deformation model on CPU")
    p.add_argument("--data_dir",    type=str, default=str(DATA_DIR))
    p.add_argument("--output_dir",  type=str, default=str(OUT_DIR))
    p.add_argument("--encoder",     type=str, default="resnet34")
    p.add_argument("--img_size",    type=int, default=256,
                   help="Image size (default 256 for CPU — use 512 on GPU)")
    p.add_argument("--batch_size",  type=int, default=4,
                   help="Batch size (default 4 for CPU — use 8-16 on GPU)")
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--patience",    type=int, default=15)
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0 = main process, safer on Windows)")
    p.add_argument("--yes", "-y",   action="store_true",
                   help="Skip the CPU speed warning prompt")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Set Intel CPU threading environment variables
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = CPU_THREADS
    env["MKL_NUM_THREADS"] = CPU_THREADS

    if not args.yes:
        print_banner(args)

    check_data(Path(args.data_dir))

    print("[INFO] Starting training...")
    print("[INFO] Monitor with TensorBoard:")
    print(f"         tensorboard --logdir {args.output_dir}/logs/tensorboard")
    print()

    cmd = [
        sys.executable,
        str(BASE_DIR / "src" / "train.py"),
        "--data_dir",    args.data_dir,
        "--output_dir",  args.output_dir,
        "--encoder",     args.encoder,
        "--img_size",    str(args.img_size),
        "--batch_size",  str(args.batch_size),
        "--epochs",      str(args.epochs),
        "--patience",    str(args.patience),
        "--num_workers", str(args.num_workers),
    ]

    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode == 0:
        print()
        print("=" * 60)
        print("  Training complete!")
        print(f"  Best model saved to: {args.output_dir}/models/best_model.pth")
        print()
        print("  Next step → python run_predict.py")
        print("=" * 60)
    else:
        print()
        print("[ERROR] Training failed. Common fixes:")
        print("  - Reduce --batch_size to 2 if you get memory errors")
        print("  - Reduce --img_size to 128 for very limited RAM")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
