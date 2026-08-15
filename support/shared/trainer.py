"""
support/shared/trainer.py
----------------------
Training engine for U-Net segmentation models.

Features:
  - Per-epoch train + validation loops with tqdm progress bars
  - Metrics logged each epoch: loss, IoU, Dice, Pixel Accuracy
  - ReduceLROnPlateau scheduler (monitors val IoU)
  - Early stopping (patience-based, monitors val IoU)
  - Best checkpoint saving (by val IoU)
  - CSV training log (one row per epoch)
  - Experiment directory creation with config snapshot

Usage (in a notebook or script):
    from support.shared import UNet, BCEDiceLoss, Trainer
    import config

    cfg     = config.CRACK_UNET
    model   = UNet(cfg["in_channels"], cfg["out_channels"], cfg["base_features"])
    loss_fn = BCEDiceLoss(cfg["bce_weight"], cfg["dice_weight"])
    trainer = Trainer(model, loss_fn, cfg, train_loader, val_loader)
    trainer.fit()
"""

import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from .metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------
class Trainer:
    """
    Manages the full training lifecycle for a segmentation model.

    Args:
        model        : nn.Module (e.g. UNet)
        loss_fn      : callable(pred, target) → scalar loss tensor
        cfg          : dict from config.CRACK_UNET
        train_loader : DataLoader for training split
        val_loader   : DataLoader for validation split
        exp_dir      : optional override for experiment output directory
                       (defaults to config.EXP_DIR / cfg["run_name"])
    """

    def __init__(self,
                 model:        nn.Module,
                 loss_fn,
                 cfg:          dict,
                 train_loader: DataLoader,
                 val_loader:   DataLoader,
                 exp_dir:      Path | None = None):

        self.cfg          = cfg
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.loss_fn      = loss_fn

        # Device
        requested = cfg.get("device", "cuda")
        self.device = torch.device("cuda" if (requested == "cuda" and torch.cuda.is_available()) else "cpu")
        print(f"[Trainer] Using device: {self.device}")
        self.model = model.to(self.device)

        # Optimizer
        self.optimizer = Adam(self.model.parameters(), lr=cfg["lr"])

        # LR scheduler (ReduceLROnPlateau — fires when val IoU stops improving)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",       # maximise val IoU
            factor=cfg["lr_factor"],
            patience=cfg["lr_patience"],
        )

        # Experiment directory
        self.exp_dir = Path(exp_dir) if exp_dir else config.EXP_DIR / cfg["run_name"]
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.best_model_path = self.exp_dir / "best_model.pth"
        self.log_csv_path    = self.exp_dir / "train_log.csv"

        # Internal state
        self.best_val_iou      = 0.0
        self.epochs_no_improve = 0
        self.history           = []   # list of dicts, one per epoch

        # Save config snapshot
        snapshot_path = self.exp_dir / "config_snapshot.json"
        with open(snapshot_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[Trainer] Config snapshot → {snapshot_path}")

        # Initialise CSV log
        with open(self.log_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "lr",
                "train_loss", "train_iou", "train_dice", "train_pixel_acc",
                "val_loss",   "val_iou",   "val_dice",   "val_pixel_acc",
                "epoch_time_s",
            ])

    # -----------------------------------------------------------------------
    # Single epoch: training
    # -----------------------------------------------------------------------
    def _train_epoch(self) -> dict:
        self.model.train()
        total_loss, total_iou, total_dice, total_acc = 0.0, 0.0, 0.0, 0.0
        n_batches = len(self.train_loader)

        loop = tqdm(self.train_loader, desc="  Train", leave=False, ncols=90)
        for images, masks in loop:
            images = images.to(self.device)
            masks  = masks.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(images)
            loss  = self.loss_fn(preds, masks)
            loss.backward()
            self.optimizer.step()

            m = compute_all_metrics(preds.detach(), masks)
            total_loss += loss.item()
            total_iou  += m["iou"]
            total_dice += m["dice"]
            total_acc  += m["pixel_acc"]
            loop.set_postfix(loss=f"{loss.item():.4f}", iou=f"{m['iou']:.4f}")

        return {
            "train_loss"      : total_loss / n_batches,
            "train_iou"       : total_iou  / n_batches,
            "train_dice"      : total_dice / n_batches,
            "train_pixel_acc" : total_acc  / n_batches,
        }

    # -----------------------------------------------------------------------
    # Single epoch: validation
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def _val_epoch(self) -> dict:
        self.model.eval()
        total_loss, total_iou, total_dice, total_acc = 0.0, 0.0, 0.0, 0.0
        n_batches = len(self.val_loader)

        loop = tqdm(self.val_loader, desc="  Val  ", leave=False, ncols=90)
        for images, masks in loop:
            images = images.to(self.device)
            masks  = masks.to(self.device)

            preds = self.model(images)
            loss  = self.loss_fn(preds, masks)

            m = compute_all_metrics(preds, masks)
            total_loss += loss.item()
            total_iou  += m["iou"]
            total_dice += m["dice"]
            total_acc  += m["pixel_acc"]
            loop.set_postfix(loss=f"{loss.item():.4f}", iou=f"{m['iou']:.4f}")

        return {
            "val_loss"      : total_loss / n_batches,
            "val_iou"       : total_iou  / n_batches,
            "val_dice"      : total_dice / n_batches,
            "val_pixel_acc" : total_acc  / n_batches,
        }

    # -----------------------------------------------------------------------
    # Full training loop
    # -----------------------------------------------------------------------
    def fit(self, n_epochs: int | None = None) -> list[dict]:
        """
        Train for n_epochs (or cfg["epochs"] if None).

        Returns:
            self.history — list of per-epoch metric dicts.
        """
        n_epochs = n_epochs or self.cfg["epochs"]
        patience = self.cfg["early_stop_patience"]

        print(f"\n{'═'*70}")
        print(f"  Starting training: {self.cfg['run_name']}")
        print(f"  Epochs: {n_epochs}   Device: {self.device}   LR: {self.cfg['lr']}")
        print(f"  Early stopping patience: {patience}")
        print(f"  Output dir: {self.exp_dir}")
        print(f"{'═'*70}\n")

        # Header row for pretty-print table
        hdr = (f"{'Ep':>4} {'LR':>8} | "
               f"{'TrLoss':>8} {'TrIoU':>7} | "
               f"{'VaLoss':>8} {'VaIoU':>7} {'VaDice':>7} | {'Best?':>6}")
        print(hdr)
        print("-" * len(hdr))

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            train_m = self._train_epoch()
            val_m   = self._val_epoch()

            elapsed = time.time() - t0
            lr_now  = self.optimizer.param_groups[0]["lr"]

            # LR scheduler step
            self.scheduler.step(val_m["val_iou"])

            # Check if best
            is_best = val_m["val_iou"] > self.best_val_iou
            if is_best:
                self.best_val_iou = val_m["val_iou"]
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                best_flag = "✓ NEW"
            else:
                self.epochs_no_improve += 1
                best_flag = ""

            # Print row
            print(
                f"{epoch:>4} {lr_now:>8.2e} | "
                f"{train_m['train_loss']:>8.4f} {train_m['train_iou']:>7.4f} | "
                f"{val_m['val_loss']:>8.4f} {val_m['val_iou']:>7.4f} "
                f"{val_m['val_dice']:>7.4f} | {best_flag:>6}"
            )

            # Merge and store
            row = {"epoch": epoch, "lr": lr_now, "epoch_time_s": round(elapsed, 1)}
            row.update(train_m)
            row.update(val_m)
            self.history.append(row)

            # Write to CSV
            with open(self.log_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, lr_now,
                    train_m["train_loss"], train_m["train_iou"],
                    train_m["train_dice"], train_m["train_pixel_acc"],
                    val_m["val_loss"],     val_m["val_iou"],
                    val_m["val_dice"],     val_m["val_pixel_acc"],
                    round(elapsed, 1),
                ])

            # Early stopping
            if self.epochs_no_improve >= patience:
                print(f"\n[Trainer] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs).")
                break

        print(f"\n[Trainer] Training complete. Best val IoU: {self.best_val_iou:.4f}")
        print(f"[Trainer] Best checkpoint → {self.best_model_path}")
        print(f"[Trainer] Training log    → {self.log_csv_path}\n")

        return self.history

    # -----------------------------------------------------------------------
    # Convenience: load best checkpoint
    # -----------------------------------------------------------------------
    def load_best(self) -> None:
        """Load the best saved checkpoint into self.model."""
        if self.best_model_path.exists():
            self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
            print(f"[Trainer] Loaded best checkpoint from {self.best_model_path}")
        else:
            print(f"[!] No checkpoint found at {self.best_model_path}")


