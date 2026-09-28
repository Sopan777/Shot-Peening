import torch
import torch.nn as nn
from typing import Optional
import timm

class DINOv2CoverageModel(nn.Module):
    """
    DINOv2 (Vision Transformer) for Shot Peening Surface Coverage Regression.
    
    Architecture:
    - Backbone: Pretrained DINOv2 ViT (Self-supervised on 142M images, optimal for microstructure SEM)
    - Head: LayerNorm -> Dropout -> FC(256) -> GELU -> FC(1) -> Sigmoid
    
    Training Strategy:
    - Stage 1: Backbone is completely frozen (❄️). Only the ~200k-param regression head is trained.
               Takes ~5-10 minutes on CPU with zero overfitting risk.
    - Stage 2 (optional): Unfreeze the last few transformer blocks for fine-tuning.
    """
    def __init__(self, variant: str = "dinov2_vitb14", dropout: float = 0.2, freeze_backbone: bool = True):
        super().__init__()
        self.variant = variant
        self.freeze_backbone = freeze_backbone

        print(f"[*] Initializing DINOv2 backbone ({variant})...")
        try:
            # First attempt: load official Facebook Research DINOv2 weights via torch.hub
            self.backbone = torch.hub.load("facebookresearch/dinov2", variant)
            # Feature dimensions: vit_small: 384, vit_base: 768, vit_large: 1024
            self.feature_dim = self.backbone.embed_dim
            self.is_timm = False
        except Exception as e:
            print(f"[!] torch.hub failed ({e}), falling back to timm DINOv2...")
            # Fallback to timm equivalent
            timm_name = "vit_base_patch14_dinov2.lvd142m" if "vitb" in variant else "vit_small_patch14_dinov2.lvd142m"
            self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)
            self.feature_dim = self.backbone.num_features
            self.is_timm = True

        if freeze_backbone:
            self.freeze()

        # Multi-layer perceptron regression head
        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Strictly bounds output to [0.0, 1.0] (0% to 100% coverage)
        )

    def freeze(self):
        """Freeze all parameters in the backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[FROZEN] DINOv2 backbone frozen. Training only regression head.")

    def unfreeze(self, num_blocks: int = 4):
        """Unfreeze the top N transformer blocks for fine-tuning."""
        if hasattr(self.backbone, "blocks"):
            blocks = list(self.backbone.blocks)
            for block in blocks[-num_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            # Also unfreeze norm layer
            if hasattr(self.backbone, "norm"):
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
            print(f"[UNFROZEN] Unfroze top {num_blocks} transformer blocks for fine-tuning.")
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("[UNFROZEN] Unfroze entire backbone for fine-tuning.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If backbone is frozen, compute features without building gradient graph
        if not any(p.requires_grad for p in self.backbone.parameters()):
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)

        # In timm num_classes=0 returns pooled/CLS representation
        if isinstance(features, tuple):
            features = features[0]

        output = self.head(features)
        return output.squeeze(-1)  # Shape: [batch_size]


class EfficientNetCoverageModel(nn.Module):
    """
    EfficientNet-B3 baseline for comparison.
    """
    def __init__(self, model_name: str = "efficientnet_b3", dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        n_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, 512),
            nn.SiLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features).squeeze(-1)


def build_model(
    model_name: str = "dinov2_base",
    dropout: float = 0.2,
    freeze_backbone: bool = True
) -> nn.Module:
    """
    Factory helper to instantiate chosen model.
    """
    model_name_lower = model_name.lower()
    if "dinov2_base" in model_name_lower or "dinov2-b" in model_name_lower:
        return DINOv2CoverageModel(variant="dinov2_vitb14", dropout=dropout, freeze_backbone=freeze_backbone)
    elif "dinov2_small" in model_name_lower or "dinov2-s" in model_name_lower:
        return DINOv2CoverageModel(variant="dinov2_vits14", dropout=dropout, freeze_backbone=freeze_backbone)
    elif "efficientnet" in model_name_lower:
        return EfficientNetCoverageModel(model_name=model_name, dropout=dropout)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from: 'dinov2_base', 'dinov2_small', 'efficientnet_b3'")
