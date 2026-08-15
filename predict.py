"""
predict.py
----------
Single-image crack detection and severity analysis.

Usage:
    python predict.py --image path/to/road_image.jpg
    python predict.py --image path/to/road_image.jpg --threshold 0.5 --save output.png
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

import config
from support.shared.unet import UNet
from support.shared.transforms import get_val_transforms


def load_model(checkpoint_path: Path = None, device: str = None) -> torch.nn.Module:
    """Load the trained U-Net model from checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if checkpoint_path is None:
        checkpoint_path = config.EXP_DIR / config.CRACK_UNET["run_name"] / "best_model.pth"

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}. Train the model first!")

    cfg = config.CRACK_UNET
    model = UNet(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        base_features=cfg["base_features"]
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[✓] Loaded trained model from {checkpoint_path} (Device: {device})")
    return model, device


def predict_image(image_path: str | Path, model: torch.nn.Module, device: torch.device, threshold: float = 0.5):
    """
    Run inference on a single road image.

    Returns:
        orig_rgb (np.ndarray): Original image (RGB).
        pred_mask (np.ndarray): Binary crack mask (0 or 1).
        overlay (np.ndarray): Original image with cracks highlighted in red.
        crack_ratio (float): Percentage of image covered by cracks.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Read original image
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = orig_rgb.shape[:2]

    # Transform for model
    transform = get_val_transforms((config.IMG_HEIGHT, config.IMG_WIDTH))
    transformed = transform(image=orig_rgb)
    tensor = transformed["image"].unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        pred_prob = model(tensor).squeeze().cpu().numpy()

    # Resize prediction back to original image size
    pred_prob_resized = cv2.resize(pred_prob, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    pred_mask = (pred_prob_resized > threshold).astype(np.uint8)

    # Overlay: Highlight detected cracks in bright red
    overlay = orig_rgb.copy()
    overlay[pred_mask == 1] = [230, 25, 25]  # Red highlight

    # Blended transparent overlay
    alpha = 0.6
    blended = cv2.addWeighted(overlay, alpha, orig_rgb, 1 - alpha, 0)

    # Damage calculation
    crack_pixels = np.sum(pred_mask)
    total_pixels = pred_mask.size
    crack_ratio = (crack_pixels / total_pixels) * 100.0

    return orig_rgb, pred_mask, blended, crack_ratio


def assess_severity(crack_ratio: float) -> str:
    if crack_ratio == 0:
        return "No Crack Detected (Safe Road)"
    elif crack_ratio < 0.5:
        return "Minor Surface Cracking"
    elif crack_ratio < 2.0:
        return "Moderate Pavement Degradation"
    else:
        return "Severe Structural Damage (Repair Recommended)"


def main():
    parser = argparse.ArgumentParser(description="Predict road cracks for a single image.")
    parser.add_argument("--image", type=str, default=None, help="Path to input road image (optional: auto-picks sample if omitted)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_model.pth")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification probability threshold (default: 0.5)")
    parser.add_argument("--save", type=str, default="prediction_result.png", help="Path to save output visual comparison")
    args = parser.parse_args()

    # Auto-detect sample image if not passed
    image_path = args.image
    if image_path is None:
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        candidates = [p for p in config.CRACK_IMG_DIR.glob("*.*") if p.suffix.lower() in valid_exts]
        if not candidates:
            raise FileNotFoundError(f"No sample images found in {config.CRACK_IMG_DIR}. Please specify --image <path>.")
        image_path = candidates[0]
        print(f"[i] No image specified. Auto-selected sample: {image_path.name}")

    model, device = load_model(args.checkpoint)
    orig_rgb, pred_mask, overlay, crack_ratio = predict_image(image_path, model, device, args.threshold)

    has_crack = crack_ratio > 0.05
    severity = assess_severity(crack_ratio)

    print("\n" + "="*50)
    print("           ROAD DAMAGE ANALYSIS REPORT")
    print("="*50)
    print(f"  File               : {Path(image_path).name}")
    print(f"  Crack Detected     : {'YES ⚠️' if has_crack else 'NO ✅'}")
    print(f"  Crack Surface Area : {crack_ratio:.3f}% of total area")
    print(f"  Damage Severity    : {severity}")
    print("="*50 + "\n")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Road Surface Crack Detection (Severity: {severity})", fontsize=13, fontweight="bold")

    axes[0].imshow(orig_rgb)
    axes[0].set_title("Input Road Image")
    axes[0].axis("off")

    axes[1].imshow(pred_mask, cmap="gray")
    axes[1].set_title(f"Predicted Binary Mask (Area: {crack_ratio:.2f}%)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Detection Overlay (Red = Cracks)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"[✓] Visual report saved to: {args.save}")
    plt.show()


if __name__ == "__main__":
    main()
