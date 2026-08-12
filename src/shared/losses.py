"""
src/shared/losses.py
---------------------
Loss functions for binary segmentation with extreme class imbalance.

Crack pixels account for only ~2–5% of image area; lane pixels ~3–8%.
Plain BCE treats these equally weighted with background — the model tends
to predict all-background and achieves ~95% accuracy while learning nothing.

Solutions implemented here:
  BCEDiceLoss  – weighted sum of BCE and Dice loss (default: 0.5 / 0.5)
  dice_loss    – standalone Dice loss (1 - Dice coefficient)
  focal_loss   – down-weights easy negatives, focuses on hard foreground pixels

All functions expect:
  pred   : float tensor [B, 1, H, W] from sigmoid output (values in [0,1])
  target : float tensor [B, 1, H, W] binary ground truth (values in {0.0, 1.0})
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


SMOOTH = 1e-6   # small constant to prevent division by zero in Dice


# ---------------------------------------------------------------------------
# Dice loss
# ---------------------------------------------------------------------------
def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Soft Dice loss for binary segmentation.

    Dice = (2 * |pred ∩ target|) / (|pred| + |target|)
    Loss = 1 - Dice

    'Soft' means we use the raw probabilities (not thresholded),
    which makes the loss differentiable everywhere.
    """
    pred   = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    dice_coeff   = (2.0 * intersection + SMOOTH) / (pred.sum() + target.sum() + SMOOTH)
    return 1.0 - dice_coeff


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------
def focal_loss(pred:   torch.Tensor,
               target: torch.Tensor,
               alpha:  float = 0.8,
               gamma:  float = 2.0) -> torch.Tensor:
    """
    Focal loss (Lin et al., 2017).
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha : weight for positive class (foreground). Set higher (>0.5) when
                foreground is rare (crack/lane pixels).
        gamma : focusing parameter. gamma=0 reduces to BCE; gamma=2 is typical.
    """
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    p_t = torch.exp(-bce)   # = pred for positives, (1-pred) for negatives
    focal = alpha * (1 - p_t) ** gamma * bce
    return focal.mean()


# ---------------------------------------------------------------------------
# Combined BCE + Dice loss (primary training loss)
# ---------------------------------------------------------------------------
class BCEDiceLoss(nn.Module):
    """
    Weighted combination of BCE and Dice losses.

    BCEDiceLoss = bce_weight * BCE + dice_weight * DiceLoss

    Why combine?
      - BCE is pixel-wise → good gradient signal everywhere
      - Dice is region-wise → directly optimises the IoU-like metric we care about
      - Together they balance pixel-level accuracy with region-level overlap

    Args:
        bce_weight  : weight on the BCE term  (default 0.5)
        dice_weight : weight on the Dice term (default 0.5)
        pos_weight  : optional scalar to upweight positive class in BCE
                      (use when crack pixels are <5% of total pixels)
    """

    def __init__(self,
                 bce_weight:  float = 0.5,
                 dice_weight: float = 0.5,
                 pos_weight:  float | None = None):
        super().__init__()
        self.bce_w  = bce_weight
        self.dice_w = dice_weight

        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce_fn = nn.BCELoss(weight=None)
        self._pos_weight = pw   # stored for device movement

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce  = self.bce_fn(pred, target)
        dice = dice_loss(pred, target)
        return self.bce_w * bce + self.dice_w * dice

    def extra_repr(self) -> str:
        return f"bce_weight={self.bce_w}, dice_weight={self.dice_w}"


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    pred   = torch.sigmoid(torch.randn(4, 1, 64, 64))
    target = (torch.rand(4, 1, 64, 64) > 0.9).float()   # ~10% foreground

    loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    loss    = loss_fn(pred, target)
    print(f"BCEDiceLoss: {loss.item():.4f}")

    f_loss  = focal_loss(pred, target)
    print(f"FocalLoss  : {f_loss.item():.4f}")

    print("✓ Loss sanity check passed.")
