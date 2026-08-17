"""
config.py – Single source of truth for all paths and hyperparameters.

To switch between Colab and local, set the RUN_ENV environment variable:
    export RUN_ENV=colab   (Google Colab with Drive mounted at /content/drive)
    export RUN_ENV=local   (default — paths relative to this file's location)
Or just edit LOCAL_ROOT / COLAB_ROOT below.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
_ENV = os.environ.get("RUN_ENV", "local").lower()

_REPO_ROOT = Path(__file__).resolve().parent      # project root (wherever this file lives)
_COLAB_ROOT = Path("/content/drive/MyDrive/Road_Damage_Project")

if _ENV == "colab":
    ROOT = _COLAB_ROOT
else:
    ROOT = _REPO_ROOT

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR        = ROOT / "data"

CRACK_DIR       = DATA_DIR / "crack"
CRACK_IMG_DIR   = CRACK_DIR / "images"
CRACK_MASK_DIR  = CRACK_DIR / "masks"
CRACK_CLOSEUP_DIR = CRACK_DIR / "closeup"
CRACK_DASHCAM_DIR = CRACK_DIR / "dashcam"
CRACK_SPLIT_DIR = CRACK_DIR / "splits"   # train.csv / val.csv / test.csv


# ---------------------------------------------------------------------------
# Experiment output paths
# ---------------------------------------------------------------------------
EXP_DIR = ROOT / "experiments"

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
RANDOM_SEED     = 42
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15   # must sum to 1.0 with the above two

# ---------------------------------------------------------------------------
# Image dimensions (resize all images to this before feeding the model)
# ---------------------------------------------------------------------------
IMG_HEIGHT      = 256
IMG_WIDTH       = 256

# ---------------------------------------------------------------------------
# Training – U-Net crack detection (Single & 2-Stage Fine-Tuning)
# ---------------------------------------------------------------------------
CRACK_UNET = dict(
    run_name        = "crack_unet_run1",
    in_channels     = 3,
    out_channels    = 1,
    base_features   = 32,
    epochs          = 50,
    batch_size      = 32,
    lr              = 2e-4,
    lr_patience     = 5,       # ReduceLROnPlateau patience
    lr_factor       = 0.5,
    early_stop_patience = 10,
    bce_weight      = 0.5,
    dice_weight     = 0.5,
    device          = "cuda",  # falls back to cpu in trainer.py if unavailable
)

# Stage 1: Pre-training on Close-up crack dataset
CRACK_UNET_STAGE1 = dict(
    run_name        = "crack_stage1_unet",
    in_channels     = 3,
    out_channels    = 1,
    base_features   = 32,
    epochs          = 40,
    batch_size      = 32,
    lr              = 2e-4,
    lr_patience     = 4,
    lr_factor       = 0.5,
    early_stop_patience = 8,
    bce_weight      = 0.5,
    dice_weight     = 0.5,
    device          = "cuda",
)

# Stage 2: Fine-tuning on Dashcam / Drone survey dataset
CRACK_UNET_STAGE2 = dict(
    run_name        = "crack_stage2_unet_finetuned",
    in_channels     = 3,
    out_channels    = 1,
    base_features   = 32,
    epochs          = 75,      # Total epochs (35 initial + 40 extended)
    batch_size      = 32,
    lr              = 1e-4,    # Balanced learning rate for fine-tuning
    lr_patience     = 4,       # ReduceLROnPlateau patience
    lr_factor       = 0.5,
    early_stop_patience = 12,
    bce_weight      = 0.5,
    dice_weight     = 0.5,     # Balanced BCE/Dice
    device          = "cuda",
    pretrained_path = EXP_DIR / "crack_stage1_unet" / "best_model.pth",
    resume_checkpoint = EXP_DIR / "crack_stage2_unet_finetuned" / "best_model.pth",
)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
AUG = dict(
    hflip_p             = 0.5,
    rotate90_p          = 0.5,
    brightness_contrast_p = 0.3,
    gauss_noise_p       = 0.2,
    mean                = (0.485, 0.456, 0.406),   # ImageNet stats
    std                 = (0.229, 0.224, 0.225),
)
