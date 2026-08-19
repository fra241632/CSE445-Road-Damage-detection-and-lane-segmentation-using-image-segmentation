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
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.model = model.to(self.device)

        # Mixed precision (AMP)
        self.use_amp = cfg.get("use_amp", True) and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        if self.use_amp:
            print("[Trainer] Mixed precision (AMP) enabled.")

        # Optimizer
        self.optimizer = Adam(self.model.parameters(), lr=cfg["lr"])

        # LR scheduler (ReduceLROnPlateau — fires when val IoU stops improving)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",       # maximise val IoU
            factor=cfg.get("lr_factor", 0.5),
            patience=cfg.get("lr_patience", 4),
        )

        # Experiment directory
        self.exp_dir = Path(exp_dir) if exp_dir else config.EXP_DIR / cfg["run_name"]
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.best_model_path = self.exp_dir / "best_model.pth"
        self.last_model_path = self.exp_dir / "last_model.pth"
        self.log_csv_path    = self.exp_dir / "train_log.csv"

        # Internal state
        self.best_val_iou      = 0.0
        self.epochs_no_improve = 0
        self.history           = []   # list of dicts, one per epoch
        self.start_epoch       = 1

        # Check for resumption
        resume_from = cfg.get("resume_checkpoint", None)
        if resume_from and Path(resume_from).exists():
            self._load_resume_checkpoint(Path(resume_from))

        # Save config snapshot
        snapshot_path = self.exp_dir / "config_snapshot.json"
        with open(snapshot_path, "w") as f:
            json.dump(cfg, f, indent=2, default=str)
        print(f"[Trainer] Config snapshot -> {snapshot_path}")

        # Initialise CSV log if new
        if not self.log_csv_path.exists() or self.log_csv_path.stat().st_size == 0:
            with open(self.log_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "lr",
                    "train_loss", "train_iou", "train_dice", "train_pixel_acc",
                    "val_loss",   "val_iou",   "val_dice",   "val_pixel_acc",
                    "epoch_time_s",
                ])

    def _load_resume_checkpoint(self, ckpt_path: Path):
        """Loads weights and state from an existing checkpoint."""
        print(f"[Trainer] Resuming from checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
            if "optimizer_state_dict" in state:
                try:
                    self.optimizer.load_state_dict(state["optimizer_state_dict"])
                except Exception:
                    pass
            if "best_val_iou" in state:
                self.best_val_iou = state["best_val_iou"]
            if "epoch" in state:
                self.start_epoch = state["epoch"] + 1
        elif isinstance(state, dict):
            # Pure model state dict
            self.model.load_state_dict(state)

        # Sync start_epoch and best_val_iou from train_log.csv if present
        if self.log_csv_path.exists():
            try:
                import pandas as pd
                df_log = pd.read_csv(self.log_csv_path)
                if not df_log.empty and "epoch" in df_log.columns:
                    self.start_epoch = int(df_log["epoch"].max()) + 1
                    if "val_iou" in df_log.columns:
                        self.best_val_iou = float(df_log["val_iou"].max())
            except Exception:
                pass

        print(f"[Trainer] Resumed model state. Baseline best Val IoU: {self.best_val_iou:.4f}, Start epoch: {self.start_epoch}")

    # -----------------------------------------------------------------------
    # Single epoch: training
    # -----------------------------------------------------------------------
    def _train_epoch(self) -> dict:
        self.model.train()
        total_loss, total_iou, total_dice, total_acc = 0.0, 0.0, 0.0, 0.0
        n_batches = len(self.train_loader)

        loop = tqdm(self.train_loader, desc="  Train", leave=False, ncols=90)
        for images, masks in loop:
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                preds = self.model(images)
                loss  = self.loss_fn(preds, masks)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

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
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
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
            self.history - list of per-epoch metric dicts.
        """
        total_epochs = n_epochs or self.cfg["epochs"]
        patience     = self.cfg["early_stop_patience"]
        start_ep     = self.start_epoch

        if start_ep > total_epochs:
            print(f"[Trainer] Model already trained up to epoch {start_ep - 1} >= target {total_epochs}. Extending total epochs to {start_ep + total_epochs - 1}.")
            total_epochs = start_ep + total_epochs - 1

        print(f"\n{'='*70}")
        print(f"  Starting training: {self.cfg['run_name']}")
        print(f"  Epochs: {start_ep} to {total_epochs} (Total: {total_epochs}) | Device: {self.device} | Initial LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Early stopping patience: {patience}")
        print(f"  Output dir: {self.exp_dir}")
        print(f"{'='*70}\n")

        # Header row for pretty-print table
        hdr = (f"{'Ep':>4} {'LR':>8} | "
               f"{'TrLoss':>8} {'TrIoU':>7} | "
               f"{'VaLoss':>8} {'VaIoU':>7} {'VaDice':>7} | {'Best?':>6}")
        print(hdr)
        print("-" * len(hdr))

        for epoch in range(start_ep, total_epochs + 1):
            t0 = time.time()

            train_m = self._train_epoch()
            val_m   = self._val_epoch()

            elapsed = time.time() - t0
            lr_now  = self.optimizer.param_groups[0]["lr"]

            # LR scheduler step
            self.scheduler.step(val_m["val_iou"])

            # Check if best (guarded against degenerate all-ones predictions where pixel_acc is near 0)
            is_valid = (val_m["val_pixel_acc"] >= 0.70)
            is_best  = is_valid and (val_m["val_iou"] > self.best_val_iou)
            
            # Checkpoint dict
            ckpt_state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_iou": max(self.best_val_iou, val_m["val_iou"]),
                "val_iou": val_m["val_iou"],
                "val_dice": val_m["val_dice"],
                "train_iou": train_m["train_iou"],
            }

            if is_best:
                self.best_val_iou = val_m["val_iou"]
                self.epochs_no_improve = 0
                torch.save(ckpt_state, self.best_model_path)
                best_flag = "* NEW"
            else:
                self.epochs_no_improve += 1
                best_flag = ""

            # Always save last checkpoint
            torch.save(ckpt_state, self.last_model_path)

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
        print(f"[Trainer] Best checkpoint -> {self.best_model_path}")
        print(f"[Trainer] Training log    -> {self.log_csv_path}\n")

        return self.history

    # -----------------------------------------------------------------------
    # Convenience: load best checkpoint
    # -----------------------------------------------------------------------
    def load_best(self) -> None:
        """Load the best saved checkpoint into self.model."""
        if self.best_model_path.exists():
            state = torch.load(self.best_model_path, map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            self.model.load_state_dict(state)
            print(f"[Trainer] Loaded best checkpoint from {self.best_model_path}")
        else:
            print(f"[!] No checkpoint found at {self.best_model_path}")



