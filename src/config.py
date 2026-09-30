import json
from dataclasses import dataclass, field, asdict
from typing import List, Tuple
import os

@dataclass
class Config:
    # Data
    data_dir: str = 'data'
    img_size: int = 512
    batch_size: int = 8
    num_workers: int = 4
    
    # Model
    encoder_name: str = 'resnet34'
    encoder_weights: str = 'imagenet'
    architecture: str = 'Unet'
    classes: int = 1
    
    # Training
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 15
    freeze_encoder_epochs: int = 30
    
    # Loss
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    
    # Output
    output_dir: str = 'outputs'
    model_dir: str = 'outputs/models'
    log_dir: str = 'outputs/logs'
    
    # Inference
    threshold: float = 0.5
    
    # ImageNet normalization
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    
    # Reproducibility
    seed: int = 42

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: dict) -> 'Config':
        """Create a configuration instance from a dictionary."""
        # Filter out invalid keys to ensure forward compatibility
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        
        # Handle tuple conversions for mean/std if they were loaded as lists
        if 'mean' in filtered_data and isinstance(filtered_data['mean'], list):
            filtered_data['mean'] = tuple(filtered_data['mean'])
        if 'std' in filtered_data and isinstance(filtered_data['std'], list):
            filtered_data['std'] = tuple(filtered_data['std'])
            
        return cls(**filtered_data)
        
    def save(self, filepath: str) -> None:
        """Save configuration to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
            
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """Load configuration from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
