"""
src/shared/deeplabv3.py
------------------------
DeepLabv3 wrapper for binary segmentation comparison against U-Net.

Features:
  - Uses torchvision's deeplabv3_resnet50 (or deeplabv3_mobilenet_v3_large)
  - Replaces final classifier head with a 1-channel output + Sigmoid
  - Compatible with Trainer, BCEDiceLoss, and SegmentationDataset

Usage:
    from src.shared.deeplabv3 import DeepLabV3Segmentation
    model = DeepLabV3Segmentation(in_channels=3, out_channels=1, backbone="resnet50", pretrained=True)
"""

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights, DeepLabV3_MobileNet_V3_Large_Weights


class DeepLabV3Segmentation(nn.Module):
    """
    DeepLabv3 model for binary semantic segmentation.

    Args:
        in_channels  (int): Number of input channels (default: 3 for RGB).
        out_channels (int): Number of output classes (default: 1 for binary mask).
        backbone     (str): 'resnet50' or 'mobilenet_v3'.
        pretrained  (bool): Whether to use ImageNet/COCO pretrained weights.
    """

    def __init__(self,
                 in_channels:  int = 3,
                 out_channels: int = 1,
                 backbone:     str = "resnet50",
                 pretrained:  bool = True):
        super().__init__()
        self.backbone_type = backbone

        if backbone == "resnet50":
            weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
            self.model = deeplabv3_resnet50(weights=weights)
            # Replace classifier head (21 COCO classes -> out_channels)
            self.model.classifier[4] = nn.Conv2d(256, out_channels, kernel_size=1)
        elif backbone == "mobilenet_v3":
            weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.model = deeplabv3_mobilenet_v3_large(weights=weights)
            self.model.classifier[4] = nn.Conv2d(256, out_channels, kernel_size=1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'resnet50' or 'mobilenet_v3'.")

        # Auxiliary classifier head replacement if present
        if hasattr(self.model, "aux_classifier") and self.model.aux_classifier is not None:
            self.model.aux_classifier[4] = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)["out"]
        return torch.sigmoid(out)

    def count_parameters(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = DeepLabV3Segmentation(backbone="resnet50", pretrained=False)
    print(f"DeepLabV3 parameters: {model.count_parameters():,}")
    dummy = torch.randn(2, 3, 256, 256)
    out   = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (2, 1, 256, 256)
    print("✓ DeepLabV3 sanity check passed.")
