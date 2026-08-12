"""
src/shared/metrics.py
----------------------
Evaluation metrics for binary segmentation.

All metrics:
  - Operate on raw probability tensors (not thresholded by default)
  - Accept a `threshold` parameter (default 0.5) to binarise predictions
  - Are computed over the entire batch at once for efficiency
  - Return Python floats (not tensors) for easy logging

Metrics implemented:
  iou_score      – Intersection over Union (Jaccard index)
  dice_score     – F1 score for segmentation (= 2×IoU / (1 + IoU))
  pixel_accuracy – fraction of correctly classified pixels
  compute_all_metrics – returns all three as a dict

Note: IoU is the primary metric for segmentation in CSE445 evaluation.
"""

import torch

SMOOTH = 1e-6


def iou_score(pred:      torch.Tensor,
              target:    torch.Tensor,
              threshold: float = 0.5) -> float:
    """
    Intersection over Union (Jaccard Index).

    IoU = |pred ∩ target| / |pred ∪ target|

    Args:
        pred      : float tensor [B, 1, H, W] with values in [0, 1]
        target    : float tensor [B, 1, H, W] with values in {0.0, 1.0}
        threshold : probability threshold for binarising pred

    Returns:
        Mean IoU over the batch as a Python float.
    """
    with torch.no_grad():
        pred_bin = (pred > threshold).float()
        inter    = (pred_bin * target).sum(dim=(1, 2, 3))
        union    = (pred_bin + target).clamp(0, 1).sum(dim=(1, 2, 3))
        iou      = (inter + SMOOTH) / (union + SMOOTH)
    return iou.mean().item()


def dice_score(pred:      torch.Tensor,
               target:    torch.Tensor,
               threshold: float = 0.5) -> float:
    """
    Dice coefficient (F1 score for segmentation).

    Dice = 2 * |pred ∩ target| / (|pred| + |target|)

    Mathematically related to IoU by: Dice = 2*IoU / (1 + IoU)
    Dice is more sensitive to false negatives than IoU.

    Returns:
        Mean Dice over the batch as a Python float.
    """
    with torch.no_grad():
        pred_bin     = (pred > threshold).float()
        intersection = (pred_bin * target).sum(dim=(1, 2, 3))
        denom        = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice         = (2.0 * intersection + SMOOTH) / (denom + SMOOTH)
    return dice.mean().item()


def pixel_accuracy(pred:      torch.Tensor,
                   target:    torch.Tensor,
                   threshold: float = 0.5) -> float:
    """
    Pixel-wise accuracy: fraction of pixels classified correctly.

    Note: this metric is misleading for imbalanced tasks (a model predicting
    all-background gets ~97% accuracy on CRACK500). Always report alongside IoU.

    Returns:
        Accuracy as a Python float in [0, 1].
    """
    with torch.no_grad():
        pred_bin = (pred > threshold).float()
        correct  = (pred_bin == target).float().sum()
        total    = torch.numel(target)
    return (correct / total).item()


def compute_all_metrics(pred:      torch.Tensor,
                        target:    torch.Tensor,
                        threshold: float = 0.5) -> dict[str, float]:
    """
    Compute IoU, Dice, and Pixel Accuracy in one call.

    Returns:
        dict with keys: 'iou', 'dice', 'pixel_acc'
    """
    return {
        "iou"       : iou_score(pred, target, threshold),
        "dice"      : dice_score(pred, target, threshold),
        "pixel_acc" : pixel_accuracy(pred, target, threshold),
    }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    # Perfect prediction → all metrics should be 1.0
    target   = (torch.rand(4, 1, 64, 64) > 0.9).float()
    pred_perfect = target.clone()

    m = compute_all_metrics(pred_perfect, target)
    print("Perfect prediction:", m)
    assert abs(m["iou"]       - 1.0) < 1e-3
    assert abs(m["dice"]      - 1.0) < 1e-3
    assert abs(m["pixel_acc"] - 1.0) < 1e-3

    # Random prediction → metrics should be much lower
    pred_random = torch.rand_like(target)
    m2 = compute_all_metrics(pred_random, target)
    print("Random prediction :", m2)
    assert m2["iou"] < 0.5

    print("✓ Metrics sanity check passed.")
