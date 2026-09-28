from pathlib import Path
import os
import torch

# ==================== PATH CONFIGURATION ====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
LABELS_FILE = DATA_DIR / "labels.csv"

# Fallback to repository "Shot Pinning" folder if data/images contains no image files
alt_image_dir = PROJECT_ROOT / "Shot Pinning"
image_extensions = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]
has_images = any(len(list(IMAGE_DIR.glob(ext))) > 0 for ext in image_extensions) if IMAGE_DIR.exists() else False
if alt_image_dir.exists() and not has_images:
    IMAGE_DIR = alt_image_dir

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
PLOT_DIR = OUTPUT_DIR / "plots"
OPENVINO_DIR = OUTPUT_DIR / "openvino"

# Ensure all output directories exist
for path in [DATA_DIR, IMAGE_DIR, OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, PLOT_DIR, OPENVINO_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# ==================== MODEL ARCHITECTURE ====================
# Options: "dinov2_base" (recommended), "dinov2_small", "efficientnet_b3"
MODEL_NAME = "dinov2_base"
NUM_CLASSES = 1  # Single continuous value (Coverage Percentage: 0.0 to 100.0)
DROPOUT_RATE = 0.2

# ==================== TRAINING HYPERPARAMETERS ====================
IMAGE_SIZE = 224  # DINOv2 default patch resolution (224x224)
BATCH_SIZE = 8    # Balanced for CPU RAM / cache
NUM_WORKERS = 0   # Safe default for Windows multiprocessing

# Two-stage training strategy
NUM_EPOCHS_STAGE1 = 25     # Phase 1: Train regression head with frozen backbone (~5-10 min)
LEARNING_RATE_STAGE1 = 1e-3

NUM_EPOCHS_STAGE2 = 30     # Phase 2: Unfreeze top transformer blocks for fine-tuning
LEARNING_RATE_STAGE2 = 1e-5

WEIGHT_DECAY = 1e-4
PATIENCE = 10              # Early stopping patience in epochs

# ==================== DATASET SPLIT ====================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ==================== HARDWARE & ACCELERATION ====================
# Device detection: Intel Core Ultra 5 235 (CPU for training, NPU via OpenVINO for inference)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Intel Extension for PyTorch (IPEX) & BFloat16 mixed precision on CPU
USE_IPEX = True
USE_BF16 = (DEVICE.type == "cpu")  # BFloat16 supported on Intel Core Ultra
