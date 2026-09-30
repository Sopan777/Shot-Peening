import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_augmentations(img_size: int = 512) -> A.Compose:
    """
    Get albumentations composition for training.
    Applies aggressive augmentations.

    Args:
        img_size (int): Target size to resize images.

    Returns:
        A.Compose: Albumentations transforms pipeline.
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

def get_val_augmentations(img_size: int = 512) -> A.Compose:
    """
    Get albumentations composition for validation.
    Only resizes and normalizes.

    Args:
        img_size (int): Target size to resize images.

    Returns:
        A.Compose: Albumentations transforms pipeline.
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

def get_test_augmentations(img_size: int = 512) -> A.Compose:
    """
    Get albumentations composition for testing.
    Same as validation.

    Args:
        img_size (int): Target size to resize images.

    Returns:
        A.Compose: Albumentations transforms pipeline.
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
