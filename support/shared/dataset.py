"""
support/shared/dataset.py
----------------------
PyTorch Dataset for binary crack segmentation.

Reads image-mask pairs from a CSV manifest produced by preprocess.py:
    image_path, mask_path
    /abs/path/to/img.jpg, /abs/path/to/mask.png
    ...

Images are loaded in RGB, masks as grayscale (binarised at 127).
An optional albumentations transform pipeline is applied jointly to
both image and mask to keep them spatially consistent.
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """
    Reads a CSV manifest and yields (image_tensor [3, H, W], mask_tensor [1, H, W]) pairs.

    Args:
        manifest_csv : Path to the split CSV (train.csv / val.csv / test.csv).
        transform    : An albumentations Compose pipeline (or None for no augmentation).
        img_size     : (height, width) tuple used as a fallback resize if no transform given.
    """

    def __init__(self,
                 manifest_csv: str | Path,
                 transform=None,
                 img_size: tuple[int, int] = (256, 256)):
        self.transform  = transform
        self.img_size   = img_size
        self.pairs: list[tuple[str, str]] = []

        with open(manifest_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.pairs.append((row["image_path"], row["mask_path"]))

        if not self.pairs:
            raise ValueError(f"No pairs found in {manifest_csv}. Check the CSV file.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]

        # Load image (BGR → RGB)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask as grayscale, binarise at 127
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        mask = (mask > 127).astype(np.uint8)

        # Apply albumentations transform (handles image + mask jointly)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]   # already a tensor (ToTensorV2)
            mask  = augmented["mask"].float()
            # Ensure mask shape is [1, H, W]
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            # Fallback: manual resize + to-tensor
            h, w = self.img_size
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            mask  = cv2.resize(mask,  (w, h), interpolation=cv2.INTER_NEAREST)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask  = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask

    def __repr__(self) -> str:
        return (f"SegmentationDataset(n={len(self)}, "
                f"img_size={self.img_size}, transform={'yes' if self.transform else 'no'})")


