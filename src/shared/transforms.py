"""
src/shared/transforms.py
-------------------------
Albumentations augmentation pipelines for training and validation.

get_train_transforms(img_size) – spatial + photometric augmentations
get_val_transforms(img_size)   – resize + normalize only (no randomness)

All pipelines end with Normalize (ImageNet stats) + ToTensorV2 so the
output is a float32 torch Tensor with shape [C, H, W] in [~-2, ~2] range.
The mask output is float32 [H, W] — SegmentationDataset.unsqueeze(0) it.
"""

import sys
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


def get_train_transforms(img_size: tuple[int, int] = (config.IMG_HEIGHT, config.IMG_WIDTH)) -> A.Compose:
    """
    Training pipeline with spatial + photometric augmentations.

    Augmentations applied (all with p < 1 so the model sees originals too):
      - Resize to target size
      - HorizontalFlip               (p=0.5)  – flip horizontally
      - RandomRotate90               (p=0.5)  – useful for crack detection
      - RandomBrightnessContrast     (p=0.3)  – simulates lighting variation
      - GaussNoise                   (p=0.2)  – sensor noise robustness
      - Normalize (ImageNet mean/std)
      - ToTensorV2 → float32 Tensor
    """
    h, w = img_size
    aug = config.AUG
    return A.Compose([
        A.Resize(h, w),
        A.HorizontalFlip(p=aug["hflip_p"]),
        A.RandomRotate90(p=aug["rotate90_p"]),
        A.RandomBrightnessContrast(p=aug["brightness_contrast_p"]),
        A.GaussNoise(p=aug["gauss_noise_p"]),
        A.Normalize(mean=aug["mean"], std=aug["std"]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: tuple[int, int] = (config.IMG_HEIGHT, config.IMG_WIDTH)) -> A.Compose:
    """
    Validation / test pipeline — deterministic resize + normalize only.
    No random ops so results are reproducible across runs.
    """
    h, w = img_size
    aug = config.AUG
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=aug["mean"], std=aug["std"]),
        ToTensorV2(),
    ])
