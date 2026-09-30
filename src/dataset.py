import os
import glob
from pathlib import Path
from typing import Optional, Tuple, Callable, Dict, Any, List
import logging

import numpy as np
import cv2
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader

from augmentations import get_train_augmentations, get_val_augmentations, get_test_augmentations

logger = logging.getLogger(__name__)

class ShotPeeningDataset(Dataset):
    """
    PyTorch Dataset for shot peening surface deformation detection.
    
    Expects directory structure:
        data_dir/
            images/
                img1.tif
                img2.png
            masks/
                img1.tif
                img2.png
    """
    
    def __init__(
        self,
        data_dir: str,
        transforms: Optional[Callable] = None
    ) -> None:
        """
        Initialize the dataset.

        Args:
            data_dir (str): Path to the split directory (e.g., 'data/train').
            transforms (Callable, optional): Albumentations transform pipeline.
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "masks"
        self.transforms = transforms
        
        # Collect all supported image files
        self.image_paths = []
        for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]:
            self.image_paths.extend(self.images_dir.glob(ext))
        self.image_paths = sorted(self.image_paths)
        
        if not self.image_paths:
            logger.warning(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
        
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image from path, handle tif and standard formats."""
        if path.suffix.lower() in [".tif", ".tiff"]:
            try:
                img = tifffile.imread(str(path))
            except Exception as e:
                logger.error(f"Failed to load TIFF {path}: {e}")
                raise
        else:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to load image {path}")
            if len(img.shape) == 3:
                # Convert BGR to RGB for standard formats
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
        # Ensure RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
        return img

    def _load_mask(self, path: Path) -> np.ndarray:
        """Load mask from path."""
        # Check for mask with various extensions
        mask_path = None
        for ext in [path.suffix, ".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
            p = self.masks_dir / (path.stem + ext)
            if p.exists():
                mask_path = p
                break
                
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for {path.name} in {self.masks_dir}")
            
        if mask_path.suffix.lower() in [".tif", ".tiff"]:
            mask = tifffile.imread(str(mask_path))
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
        if mask is None:
            raise ValueError(f"Failed to load mask {mask_path}")
            
        # Ensure binary (0=deformed, 255=non-deformed) -> mapping to 0 and 1
        mask = (mask > 127).astype(np.float32)
        return mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image and mask tensor pair.
        
        Args:
            idx (int): Index of the item.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image tensor and Mask tensor.
        """
        img_path = self.image_paths[idx]
        
        image = self._load_image(img_path)
        mask = self._load_mask(img_path)
        
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            
        # Ensure mask has a channel dimension
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
            
        return image, mask

def get_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    img_size: int = 512,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train, val, and test splits.
    
    Args:
        data_dir (str): Base data directory containing 'train', 'val', 'test' subdirs.
        batch_size (int): Batch size.
        img_size (int): Target image size.
        num_workers (int): Number of dataloader workers.
        
    Returns:
        Dict[str, DataLoader]: Dictionary containing the DataLoaders.
    """
    base_dir = Path(data_dir)
    
    dataloaders = {}
    
    # Train
    train_dir = base_dir / "train"
    if train_dir.exists():
        train_ds = ShotPeeningDataset(
            data_dir=str(train_dir),
            transforms=get_train_augmentations(img_size)
        )
        dataloaders["train"] = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
        
    # Val
    val_dir = base_dir / "val"
    if val_dir.exists():
        val_ds = ShotPeeningDataset(
            data_dir=str(val_dir),
            transforms=get_val_augmentations(img_size)
        )
        dataloaders["val"] = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
    # Test
    test_dir = base_dir / "test"
    if test_dir.exists():
        test_ds = ShotPeeningDataset(
            data_dir=str(test_dir),
            transforms=get_test_augmentations(img_size)
        )
        dataloaders["test"] = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
    return dataloaders
