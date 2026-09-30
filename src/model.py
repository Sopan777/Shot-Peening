"""
Model definitions and utilities for shot peening surface deformation detection.
"""
from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def create_model(
    architecture: str = 'Unet',
    encoder_name: str = 'resnet34',
    encoder_weights: Optional[str] = 'imagenet',
    classes: int = 1,
    activation: Optional[str] = None
) -> nn.Module:
    """
    Creates a segmentation model.

    Args:
        architecture: Architecture to use ('Unet', 'UnetPlusPlus', 'DeepLabV3Plus').
        encoder_name: Name of the encoder network.
        encoder_weights: Pre-trained weights to use ('imagenet' or None).
        classes: Number of output classes.
        activation: Activation function to apply to the output.

    Returns:
        A segmentation model instance.
    """
    architectures = {
        'Unet': smp.Unet,
        'UnetPlusPlus': smp.UnetPlusPlus,
        'DeepLabV3Plus': smp.DeepLabV3Plus
    }

    if architecture not in architectures:
        raise ValueError(f"Unsupported architecture '{architecture}'. Supported: {list(architectures.keys())}")

    model_class = architectures[architecture]
    model = model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=classes,
        activation=activation,
    )
    return model

def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 512, 512)) -> int:
    """
    Prints model summary and returns total parameter count.

    Args:
        model: PyTorch model.
        input_size: Expected input size (C, H, W).

    Returns:
        Total number of parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    return total_params

def set_encoder_trainable(model: nn.Module, trainable: bool = True) -> None:
    """
    Freezes or unfreezes the encoder of the model.

    Args:
        model: Segmentation model from smp.
        trainable: True to unfreeze, False to freeze.
    """
    if not hasattr(model, 'encoder'):
        raise AttributeError("Model does not have an 'encoder' attribute.")
    
    for param in model.encoder.parameters():
        param.requires_grad = trainable

def get_parameter_groups(model: nn.Module, encoder_lr: float, decoder_lr: float) -> List[Dict[str, Any]]:
    """
    Gets separate parameter groups for encoder and decoder for differential learning rates.

    Args:
        model: Segmentation model from smp.
        encoder_lr: Learning rate for the encoder.
        decoder_lr: Learning rate for the decoder and head.

    Returns:
        A list of dictionaries for optimizer parameter groups.
    """
    if not hasattr(model, 'encoder'):
        raise AttributeError("Model does not have an 'encoder' attribute.")
    
    encoder_params = list(model.encoder.parameters())
    
    # Everything else is considered decoder/head
    decoder_params = []
    for name, param in model.named_parameters():
        if not name.startswith('encoder'):
            decoder_params.append(param)
            
    return [
        {'params': encoder_params, 'lr': encoder_lr},
        {'params': decoder_params, 'lr': decoder_lr}
    ]
