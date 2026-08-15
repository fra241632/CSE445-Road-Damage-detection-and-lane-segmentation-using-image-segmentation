"""
support/shared/unet.py
-------------------
Standard U-Net for binary segmentation.

Architecture (Ronneberger et al., 2015):
  Encoder:     4 downsampling stages (DoubleConv + MaxPool)
  Bottleneck:  DoubleConv at the lowest resolution
  Decoder:     4 upsampling stages (Upsample + skip concat + DoubleConv)
  Output:      1×1 Conv → Sigmoid → probability map [0, 1]

Parameters are fully configurable via __init__ args:
  in_channels   – number of input image channels (3 for RGB)
  out_channels  – number of output channels (1 for binary segmentation)
  base_features – feature width of the first encoder block (doubles each level)

With base_features=32 and img_size=256×256:
  Encoder feature maps:  32 → 64 → 128 → 256
  Bottleneck:            512
  Decoder feature maps:  256 → 128 → 64 → 32
  Parameters:            ~7.8 M   (lightweight, trains well on free Colab GPU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """Conv→BN→ReLU → Conv→BN→ReLU  (the repeated unit in U-Net)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """MaxPool2d (↓2) followed by DoubleConv — one encoder stage."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Bilinear upsample (↑2) + skip-connection concat + DoubleConv — one decoder stage.

    Uses bilinear upsampling rather than ConvTranspose2d:
      - Fewer parameters, less checkerboard artifacts
      - Handles non-power-of-two spatial sizes gracefully
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # After concat: channels = in_ch//2 (from up) + in_ch//2 (skip) = in_ch
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad x to match skip's spatial dimensions if needed
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh > 0 or dw > 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    """
    U-Net for binary semantic segmentation.

    Args:
        in_channels   (int): Input channels. Default 3 (RGB).
        out_channels  (int): Output channels. Default 1 (binary map).
        base_features (int): Features in the first encoder block (doubles each stage).
                             Typical values: 16 (tiny), 32 (standard), 64 (large).
    """

    def __init__(self,
                 in_channels:   int = 3,
                 out_channels:  int = 1,
                 base_features: int = 32):
        super().__init__()
        f = base_features                # shorthand

        # ── Encoder ──────────────────────────────────────────────
        self.enc1 = DoubleConv(in_channels, f)       # [B, f,   H,   W]
        self.enc2 = Down(f,     f * 2)               # [B, 2f,  H/2, W/2]
        self.enc3 = Down(f * 2, f * 4)               # [B, 4f,  H/4, W/4]
        self.enc4 = Down(f * 4, f * 8)               # [B, 8f,  H/8, W/8]

        # ── Bottleneck ───────────────────────────────────────────
        self.bottleneck = Down(f * 8, f * 16)        # [B, 16f, H/16, W/16]

        # ── Decoder ──────────────────────────────────────────────
        # Up expects in_ch = decoder_ch + skip_ch
        self.dec4 = Up(f * 16 + f * 8,  f * 8)      # [B, 8f,  H/8, W/8]
        self.dec3 = Up(f * 8  + f * 4,  f * 4)      # [B, 4f,  H/4, W/4]
        self.dec2 = Up(f * 4  + f * 2,  f * 2)      # [B, 2f,  H/2, W/2]
        self.dec1 = Up(f * 2  + f,      f)           # [B, f,   H,   W]

        # ── Output head ──────────────────────────────────────────
        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b  = self.bottleneck(e4)

        # Decode (with skip connections)
        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # Output probability map
        return torch.sigmoid(self.out_conv(d1))   # [B, 1, H, W] in [0, 1]

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1, base_features=32)
    print(f"U-Net parameters: {model.count_parameters():,}")
    dummy = torch.randn(2, 3, 256, 256)
    out   = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")   # expect [2, 1, 256, 256]
    assert out.shape == (2, 1, 256, 256), "Shape mismatch!"
    assert out.min() >= 0 and out.max() <= 1, "Output not in [0,1]!"
    print("✓ U-Net sanity check passed.")


