"""
Utility functions for shot peening surface deformation detection.
"""
import logging
import sys
import os
from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- Metrics ---

def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculates various segmentation metrics for a batch of tensors.
    
    Args:
        pred: Predictions (B, C, H, W), logits or probabilities.
        target: Ground truth mask (B, C, H, W), binary (0 or 1).
        threshold: Threshold for binarizing predictions.
        
    Returns:
        Dictionary containing metric values.
    """
    # Assuming sigmoid is applied if not already probabilities
    if pred.max() > 1 or pred.min() < 0:
        pred = torch.sigmoid(pred)
        
    pred = (pred > threshold).float()
    target = target.float()
    
    tp = torch.sum(pred * target)
    fp = torch.sum(pred * (1 - target))
    fn = torch.sum((1 - pred) * target)
    tn = torch.sum((1 - pred) * (1 - target))
    
    eps = 1e-7
    
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    
    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }

class MetricTracker:
    """Class to accumulate metrics across batches."""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.metrics = {}
        self.counts = {}
        
    def update(self, batch_metrics: Dict[str, float], n: int = 1):
        for k, v in batch_metrics.items():
            if k not in self.metrics:
                self.metrics[k] = 0.0
                self.counts[k] = 0
            self.metrics[k] += v * n
            self.counts[k] += n
            
    def get_averages(self) -> Dict[str, float]:
        return {k: v / self.counts[k] for k, v in self.metrics.items()}

class EarlyStopping:
    """Early stopping to stop the training when validation loss doesn't improve."""
    def __init__(self, patience: int = 7, delta: float = 0.0, save_path: str = 'checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss: float, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
        """Saves model when validation loss decrease."""
        save_checkpoint(model, optimizer, epoch, {'val_loss': val_loss}, self.save_path)
        self.val_loss_min = val_loss

# --- Visualization and Reporting ---

def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5, 
                   color_deformed: Tuple[int, int, int] = (0, 255, 0),
                   color_non_deformed: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """Creates a colored overlay of the mask on the image."""
    # Ensure image is RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[mask == 1] = color_deformed
    overlay[mask == 0] = color_non_deformed
    
    blend = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return blend

def visualize_prediction(image: np.ndarray, mask_true: np.ndarray, mask_pred: np.ndarray, save_path: Optional[str] = None):
    """Creates a 3-panel figure comparing original, ground truth, and prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(image)
    axes[1].imshow(mask_true, alpha=0.5, cmap='jet')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(image)
    axes[2].imshow(mask_pred, alpha=0.5, cmap='jet')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def generate_coverage_report(mask_pred: np.ndarray, pixel_size_mm: Optional[float] = None) -> Dict[str, Any]:
    """Generates a physical coverage report based on the prediction mask."""
    # mask_pred: binary mask (1 for deformed/peened, 0 for non-deformed)
    total_pixels = mask_pred.size
    deformed_pixels = np.sum(mask_pred)
    non_deformed_pixels = total_pixels - deformed_pixels
    coverage_pct = (deformed_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    
    # Find connected components (defect/unpeened regions) -> looking for 0s
    inverted_mask = (1 - mask_pred).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_mask, connectivity=8)
    
    # Ignore background component (label 0, which is the peened area in inverted mask)
    defect_areas_px = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else []
    largest_defect_px = np.max(defect_areas_px) if len(defect_areas_px) > 0 else 0
    
    report = {
        'coverage_percentage': float(coverage_pct),
        'non_deformed_area_px': int(non_deformed_pixels),
        'num_defect_regions': max(0, int(num_labels - 1)),
        'largest_defect_area_px': int(largest_defect_px)
    }
    
    if pixel_size_mm is not None:
        area_per_px = pixel_size_mm ** 2
        report['non_deformed_area_mm2'] = float(non_deformed_pixels * area_per_px)
        report['largest_defect_area_mm2'] = float(largest_defect_px * area_per_px)
        
    return report

# --- Checkpointing and Logging ---

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: Dict[str, Any], filepath: str):
    """Saves model checkpoint."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """Loads model checkpoint."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {})
    }

def setup_logging(log_file: str = 'training.log', level: int = logging.INFO) -> logging.Logger:
    """Sets up logging configuration."""
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    
    logger = logging.getLogger('ShotPeening')
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times if called again
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(sh)
        
    return logger
