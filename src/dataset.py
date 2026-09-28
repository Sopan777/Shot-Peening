import os
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class ShotPeeningDataset(Dataset):
    """
    Custom PyTorch Dataset for Shot-Peened SEM surface images.
    Target labels are normalized to [0.0, 1.0] for stable training with Sigmoid output.
    """
    def __init__(self, image_paths: List[Path], labels: List[float], transform=None):
        self.image_paths = [Path(p) for p in image_paths]
        # Normalize 0..100 coverage to 0..1 range
        self.labels = [float(lbl) / 100.0 for lbl in labels]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        
        # Load SEM image (TIF/TIFF or other formats)
        try:
            with Image.open(img_path) as pil_img:
                # Convert grayscale / 16-bit to standard 8-bit RGB
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                img_np = np.array(pil_img)
        except Exception as e:
            raise RuntimeError(f"Error loading image '{img_path}': {e}")

        # Target label as float tensor
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Apply Albumentations transform
        if self.transform is not None:
            augmented = self.transform(image=img_np)
            image_tensor = augmented["image"]
        else:
            # Fallback basic normalization HWC -> CHW
            image_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        return image_tensor, label_tensor


def find_label_column(df: pd.DataFrame) -> str:
    """Intelligently detects the coverage/deformation label column name."""
    candidates = [
        "coverage_percentage", "coverage_pct", "coverage", "deformation",
        "peened_area", "coverage_%", "percentage", "target", "label"
    ]
    for col in df.columns:
        if col.strip().lower() in candidates:
            return col
    # Fallback to second column if there are only 2 columns
    if len(df.columns) == 2:
        return df.columns[1]
    raise ValueError(
        f"Could not identify coverage label column in CSV. Found columns: {list(df.columns)}. "
        f"Please ensure one column is named 'coverage_percentage' or 'coverage'."
    )


def find_filename_column(df: pd.DataFrame) -> str:
    """Intelligently detects the image filename column name."""
    candidates = ["filename", "file_name", "image", "image_name", "path", "id"]
    for col in df.columns:
        if col.strip().lower() in candidates:
            return col
    return df.columns[0]


def prepare_datasets(
    csv_path: Union[str, Path],
    image_dir: Union[str, Path],
    train_transform,
    val_transform,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[ShotPeeningDataset, ShotPeeningDataset, ShotPeeningDataset, pd.DataFrame]:
    """
    Reads labels CSV, verifies existence of image files, and creates Train/Val/Test Datasets.
    """
    csv_path = Path(csv_path)
    image_dir = Path(image_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"Labels CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    fname_col = find_filename_column(df)
    label_col = find_label_column(df)

    valid_paths = []
    valid_labels = []
    missing_files = []

    for _, row in df.iterrows():
        fname = str(row[fname_col]).strip()
        val = float(row[label_col])
        full_path = image_dir / fname

        # Check exact path or case-insensitive match
        if full_path.exists():
            valid_paths.append(full_path)
            valid_labels.append(val)
        else:
            missing_files.append(fname)

    if missing_files:
        print(f"[!] Warning: {len(missing_files)} / {len(df)} images from CSV not found in {image_dir}")
        if len(missing_files) <= 5:
            print(f"    Missing: {missing_files}")
        else:
            print(f"    First 5 missing: {missing_files[:5]}")

    if len(valid_paths) == 0:
        raise ValueError(f"No matching images found in '{image_dir}' from '{csv_path}'!")

    print(f"[OK] Loaded {len(valid_paths)} verified images and labels.")

    # Convert to numpy arrays for split
    paths_arr = np.array(valid_paths)
    labels_arr = np.array(valid_labels)

    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths_arr, labels_arr,
        test_size=temp_ratio,
        random_state=random_seed
    )

    # Second split: Val vs Test
    test_share_of_temp = test_ratio / temp_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=test_share_of_temp,
        random_state=random_seed
    )

    train_ds = ShotPeeningDataset(train_paths, train_labels, transform=train_transform)
    val_ds = ShotPeeningDataset(val_paths, val_labels, transform=val_transform)
    test_ds = ShotPeeningDataset(test_paths, test_labels, transform=val_transform)

    split_summary = pd.DataFrame([
        {"Split": "Train", "Count": len(train_ds), "Ratio": f"{len(train_ds)/len(valid_paths):.1%}"},
        {"Split": "Validation", "Count": len(val_ds), "Ratio": f"{len(val_ds)/len(valid_paths):.1%}"},
        {"Split": "Test", "Count": len(test_ds), "Ratio": f"{len(test_ds)/len(valid_paths):.1%}"}
    ])

    return train_ds, val_ds, test_ds, split_summary
