"""
support/crack/train_2stage.py
-----------------------------
Two-Stage Training Pipeline for Road Surface Crack Segmentation:
  Stage 1: Pre-train U-Net on high-contrast Close-Up crack dataset (e.g. CRACK500, DeepCrack, macro road shots).
  Stage 2: Fine-tune U-Net on Dashcam / Drone survey dataset with a reduced learning rate.

Usage:
  python support/crack/train_2stage.py                  # Runs Stage 1 then Stage 2
  python support/crack/train_2stage.py --stage 1        # Runs Stage 1 only
  python support/crack/train_2stage.py --stage 2        # Runs Stage 2 only (requires Stage 1 checkpoint)
  python support/crack/train_2stage.py --stage combined # Runs joint training on combined dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from support.shared.dataset import SegmentationDataset
from support.shared.transforms import get_train_transforms, get_val_transforms
from support.shared.unet import UNet
from support.shared.losses import BCEDiceLoss
from support.shared.trainer import Trainer


# ---------------------------------------------------------------------------
# Helper: Create DataLoaders
# ---------------------------------------------------------------------------
def build_dataloaders(train_csv: Path, val_csv: Path, batch_size: int, num_workers: int = 0):
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"Split manifests missing:\n  Train: {train_csv}\n  Val: {val_csv}\n"
            f"Please run `python support/crack/preprocess.py` first!"
        )

    img_size = (config.IMG_HEIGHT, config.IMG_WIDTH)
    train_ds = SegmentationDataset(train_csv, transform=get_train_transforms(img_size))
    val_ds   = SegmentationDataset(val_csv,   transform=get_val_transforms(img_size))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    print(f"  [DataLoader] Train: {len(train_ds)} samples ({len(train_loader)} batches)")
    print(f"  [DataLoader] Val  : {len(val_ds)} samples ({len(val_loader)} batches)")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Stage 1: Pre-training on Close-up dataset
# ---------------------------------------------------------------------------
def run_stage1(epochs: int | None = None) -> Path:
    cfg = config.CRACK_UNET_STAGE1.copy()
    if epochs:
        cfg["epochs"] = epochs

    print("\n" + "=" * 70)
    print("  STAGE 1: Pre-training U-Net on Close-Up Crack Dataset")
    print(f"  Run name : {cfg['run_name']}")
    print(f"  Epochs   : {cfg['epochs']} | LR: {cfg['lr']} | BCE/Dice: {cfg['bce_weight']}/{cfg['dice_weight']}")
    print("=" * 70 + "\n")

    train_csv = config.CRACK_SPLIT_DIR / "stage1_train.csv"
    val_csv   = config.CRACK_SPLIT_DIR / "stage1_val.csv"
    if not train_csv.exists():
        train_csv = config.CRACK_SPLIT_DIR / "closeup_train.csv"
        val_csv   = config.CRACK_SPLIT_DIR / "closeup_val.csv"

    train_loader, val_loader = build_dataloaders(train_csv, val_csv, cfg["batch_size"])

    model = UNet(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        base_features=cfg["base_features"]
    )
    loss_fn = BCEDiceLoss(bce_weight=cfg["bce_weight"], dice_weight=cfg["dice_weight"])

    trainer = Trainer(model, loss_fn, cfg, train_loader, val_loader)
    trainer.fit()

    print(f"\n[OK] Stage 1 pre-training complete! Best model saved to: {trainer.best_model_path}")
    return trainer.best_model_path


# ---------------------------------------------------------------------------
# Stage 2: Fine-tuning on Dashcam/Drone dataset
# ---------------------------------------------------------------------------
def run_stage2(pretrained_path: Path | str | None = None, epochs: int | None = None) -> Path:
    cfg = config.CRACK_UNET_STAGE2.copy()
    if epochs:
        cfg["epochs"] = epochs

    # Resolve pretrained weights
    if pretrained_path is None:
        pretrained_path = cfg.get("pretrained_path", config.EXP_DIR / "crack_stage1_unet" / "best_model.pth")
    pretrained_path = Path(pretrained_path).resolve()

    if not pretrained_path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at: {pretrained_path}\n"
            f"Please run Stage 1 first: python support/crack/train_2stage.py --stage 1"
        )

    print("\n" + "=" * 70)
    print("  STAGE 2: Fine-Tuning U-Net on Dashcam / Drone Dataset")
    print(f"  Run name   : {cfg['run_name']}")
    print(f"  Pretrained : {pretrained_path}")
    print(f"  Epochs     : {cfg['epochs']} | LR: {cfg['lr']} (Fine-tuning) | BCE/Dice: {cfg['bce_weight']}/{cfg['dice_weight']}")
    print("=" * 70 + "\n")

    train_csv = config.CRACK_SPLIT_DIR / "train.csv"
    val_csv   = config.CRACK_SPLIT_DIR / "val.csv"
    if not train_csv.exists():
        train_csv = config.CRACK_SPLIT_DIR / "dashcam_train.csv"
        val_csv   = config.CRACK_SPLIT_DIR / "dashcam_val.csv"

    train_loader, val_loader = build_dataloaders(train_csv, val_csv, cfg["batch_size"])

    model = UNet(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        base_features=cfg["base_features"]
    )

    # Load Stage 1 weights
    device = torch.device("cuda" if (cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[Stage 2] Loading weights from Stage 1: {pretrained_path}")
    state = torch.load(pretrained_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    print("[Stage 2] Stage 1 weights loaded successfully into U-Net!")

    loss_fn = BCEDiceLoss(bce_weight=cfg["bce_weight"], dice_weight=cfg["dice_weight"])
    trainer = Trainer(model, loss_fn, cfg, train_loader, val_loader)
    trainer.fit()

    print(f"\n[OK] Stage 2 fine-tuning complete! Final model saved to: {trainer.best_model_path}")
    return trainer.best_model_path


# ---------------------------------------------------------------------------
# Combined: Joint Training on Combined Dataset
# ---------------------------------------------------------------------------
def run_combined(epochs: int | None = None) -> Path:
    cfg = config.CRACK_UNET.copy()
    cfg["run_name"] = "crack_combined_unet"
    if epochs:
        cfg["epochs"] = epochs

    print("\n" + "=" * 70)
    print("  JOINT TRAINING: U-Net on Combined Dataset (Closeup + Dashcam)")
    print(f"  Run name : {cfg['run_name']}")
    print(f"  Epochs   : {cfg['epochs']} | LR: {cfg['lr']} | BCE/Dice: {cfg['bce_weight']}/{cfg['dice_weight']}")
    print("=" * 70 + "\n")

    train_csv = config.CRACK_SPLIT_DIR / "combined_train.csv"
    val_csv   = config.CRACK_SPLIT_DIR / "combined_val.csv"
    train_loader, val_loader = build_dataloaders(train_csv, val_csv, cfg["batch_size"])

    model = UNet(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        base_features=cfg["base_features"]
    )
    loss_fn = BCEDiceLoss(bce_weight=cfg["bce_weight"], dice_weight=cfg["dice_weight"])
    trainer = Trainer(model, loss_fn, cfg, train_loader, val_loader)
    trainer.fit()

    print(f"\n[OK] Joint training complete! Model saved to: {trainer.best_model_path}")
    return trainer.best_model_path


# ---------------------------------------------------------------------------
# Main CLI entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="2-Stage Crack Segmentation Training")
    parser.add_argument(
        "--stage", type=str, default="both",
        choices=["both", "1", "2", "combined"],
        help="Which training stage to run: 'both' (Stage 1 then Stage 2), '1' (Stage 1 only), '2' (Stage 2 only), 'combined'"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for the selected stage")
    parser.add_argument("--pretrained", type=str, default=None, help="Stage 1 checkpoint path (for stage 2)")
    args = parser.parse_args()

    if args.stage == "1":
        run_stage1(epochs=args.epochs)
    elif args.stage == "2":
        run_stage2(pretrained_path=args.pretrained, epochs=args.epochs)
    elif args.stage == "combined":
        run_combined(epochs=args.epochs)
    else:  # "both"
        print("Starting 2-Stage Training Workflow (Stage 1 -> Stage 2)...")
        stage1_model_path = run_stage1(epochs=args.epochs)
        run_stage2(pretrained_path=stage1_model_path, epochs=args.epochs)
        print("\n=======================================================")
        print("  [OK] Full 2-Stage Training Pipeline Completed!")
        print("=======================================================")


if __name__ == "__main__":
    main()
