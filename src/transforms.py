import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size: int = 224):
    """
    Data augmentation pipeline tailored specifically for SEM microstructures & shot peening:
    - Rotations, flips, affine scaling for orientation invariance
    - CLAHE & Contrast/Brightness for SEM detector lighting variations
    - Subtle Gaussian noise/blur for microscope defocus simulation
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=30,
            border_mode=0,
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 40.0)),
            A.GaussianBlur(blur_limit=(3, 5)),
        ], p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.4
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])

def get_val_transforms(image_size: int = 224):
    """
    Deterministic preprocessing for validation and test sets.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
