"""
support/crack/preprocess.py
------------------------
Preprocessing and EDA utilities for multi-source crack detection datasets:
  - Closeup datasets (e.g., CRACK500, DeepCrack, macro road shots) -> Stage 1
  - Dashcam / Drone datasets (wide-angle road survey footage) -> Stage 2 / Final

Supports directory structures under data/crack/:
  Option A (Recommended multi-dataset structure):
    data/crack/closeup/images/ (or image/) and data/crack/closeup/masks/ (or mask/)
    data/crack/dashcam/images/ (or image/) and data/crack/dashcam/masks/ (or mask/)

  Option B (Legacy single-folder structure):
    data/crack/images/ and data/crack/masks/

Generates split manifests in data/crack/splits/:
  - Closeup -> stage1_train.csv, stage1_val.csv, stage1_test.csv (alias: closeup_*.csv)
  - Dashcam -> train.csv, val.csv, test.csv (alias: dashcam_*.csv)
  - Combined (if both present) -> combined_train.csv, combined_val.csv, combined_test.csv

Usage:
    python support/crack/preprocess.py
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

CRACK_DIR       = config.CRACK_DIR
CRACK_IMG_DIR   = config.CRACK_IMG_DIR
CRACK_MASK_DIR  = config.CRACK_MASK_DIR
CRACK_CLOSEUP_DIR = getattr(config, "CRACK_CLOSEUP_DIR", CRACK_DIR / "closeup")
CRACK_DASHCAM_DIR = getattr(config, "CRACK_DASHCAM_DIR", CRACK_DIR / "dashcam")
CRACK_SPLIT_DIR = config.CRACK_SPLIT_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Helper: Locate image and mask directories inside a dataset folder
# ---------------------------------------------------------------------------
def find_image_mask_dirs(base_dir: Path) -> tuple[Path, Path] | None:
    """
    Search base_dir for images/masks subdirectories.
    Checks common naming patterns:
      - images / masks
      - image / mask
      - imgs / msks
      - img / msk
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return None

    img_candidates = ["images", "image", "imgs", "img"]
    mask_candidates = ["masks", "mask", "msks", "msk", "labels", "label", "groundtruth", "gt"]

    found_img_dir = None
    found_mask_dir = None

    for name in img_candidates:
        p = base_dir / name
        if p.is_dir() and any(f.suffix.lower() in IMAGE_EXTS for f in p.iterdir()):
            found_img_dir = p
            break

    for name in mask_candidates:
        p = base_dir / name
        if p.is_dir() and any(f.suffix.lower() in IMAGE_EXTS for f in p.iterdir()):
            found_mask_dir = p
            break

    if found_img_dir and found_mask_dir:
        return found_img_dir, found_mask_dir

    # Check if base_dir itself contains both images and masks directly
    all_files = [f for f in base_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    if len(all_files) >= 2:
        return base_dir, base_dir

    return None


# ---------------------------------------------------------------------------
# 1. Pair loading
# ---------------------------------------------------------------------------
def load_pairs(images_dir: Path, masks_dir: Path, tag: str = "dataset") -> list[tuple[Path, Path]]:
    """
    Match images with masks by filename stem (ignoring extension).
    Returns a sorted list of (image_path, mask_path) tuples.
    """
    images_dir = Path(images_dir)
    masks_dir  = Path(masks_dir)

    if not images_dir.exists() or not masks_dir.exists():
        return []

    # Map filename stem to Path
    img_map  = {p.stem: p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    mask_map = {p.stem: p for p in masks_dir.iterdir()  if p.is_file() and p.suffix.lower() in IMAGE_EXTS}

    # Handle cases where masks have suffixes like '_mask', '_gt', etc.
    common = sorted(set(img_map) & set(mask_map))
    if not common:
        # Try matching stems by stripping common mask suffixes
        matched_pairs = []
        for img_stem, img_path in img_map.items():
            for suffix in ["_mask", "_gt", "-mask", "_label", "_crack"]:
                if (img_stem + suffix) in mask_map:
                    matched_pairs.append((img_path, mask_map[img_stem + suffix]))
                    break
        if matched_pairs:
            print(f"[{tag}] Loaded {len(matched_pairs)} matched pairs using mask suffix matching.")
            return matched_pairs

    only_img  = set(img_map) - set(mask_map)
    only_mask = set(mask_map) - set(img_map)

    if only_img and len(img_map) != len(common):
        print(f"[{tag}] {len(only_img)} images have no matching mask (skipped).")
    if only_mask and len(mask_map) != len(common):
        print(f"[{tag}] {len(only_mask)} masks have no matching image (skipped).")

    pairs = [(img_map[stem], mask_map[stem]) for stem in common]
    print(f"[{tag}] Loaded {len(pairs)} matched image-mask pairs from {images_dir.name}/{masks_dir.name}.")
    return pairs


# ---------------------------------------------------------------------------
# 2. EDA statistics
# ---------------------------------------------------------------------------
def report_stats(pairs: list[tuple[Path, Path]], title: str = "Dataset", n_sample: int = 5) -> None:
    """
    Print sample resolution and foreground crack pixel ratio.
    """
    if not pairs:
        print(f"[!] No pairs found for {title}.")
        return

    sample = pairs[:n_sample]
    heights, widths, fg_ratios = [], [], []

    print(f"\n--- {title} (Sample of {min(n_sample, len(pairs))} / {len(pairs)} pairs) ---")
    print(f"{'File':<32} {'H x W':>12} {'Crack %':>10}")
    print("-" * 58)

    for img_path, mask_path in sample:
        img  = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        h, w = img.shape[:2]
        fg   = float((mask > 127).mean())
        heights.append(h)
        widths.append(w)
        fg_ratios.append(fg)
        print(f"  {img_path.name:<30} {h}x{w:>5}   {fg:8.3%}")

    if heights:
        print(f"  Avg crack pixel area: {np.mean(fg_ratios):.3%} (background: {1-np.mean(fg_ratios):.3%})")
        print(f"  Total pairs in {title}: {len(pairs)}")
    print("-" * 58 + "\n")


# ---------------------------------------------------------------------------
# 3. Dataset splitting
# ---------------------------------------------------------------------------
def split_dataset(pairs: list[tuple[Path, Path]],
                  train: float = config.TRAIN_RATIO,
                  val:   float = config.VAL_RATIO,
                  test:  float = config.TEST_RATIO,
                  seed:  int   = config.RANDOM_SEED
                  ) -> dict[str, list[tuple[Path, Path]]]:
    """
    Deterministic train/val/test split.
    """
    if not pairs:
        return {"train": [], "val": [], "test": []}

    assert abs(train + val + test - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    n       = len(shuffled)
    n_train = max(1, int(n * train))
    n_val   = max(1, int(n * val)) if (n - n_train) >= 2 else 0

    splits = {
        "train": shuffled[:n_train],
        "val"  : shuffled[n_train : n_train + n_val],
        "test" : shuffled[n_train + n_val :],
    }

    for name, subset in splits.items():
        print(f"  {name:>5}: {len(subset):>5} pairs")

    return splits


# ---------------------------------------------------------------------------
# 4. Save manifests
# ---------------------------------------------------------------------------
def save_splits(splits: dict[str, list[tuple[Path, Path]]],
                prefix: str = "",
                output_dir: Path = CRACK_SPLIT_DIR) -> None:
    """
    Write CSV manifests with columns: image_path, mask_path.
    If prefix is 'stage1_', filenames will be 'stage1_train.csv', etc.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, pairs in splits.items():
        if not pairs:
            continue
        filename = f"{prefix}{split_name}.csv" if prefix else f"{split_name}.csv"
        csv_path = output_dir / filename
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "mask_path"])
            for img_p, mask_p in pairs:
                writer.writerow([str(img_p.resolve()), str(mask_p.resolve())])
        print(f"  [OK] Saved {filename:<20} ({len(pairs):>4} rows) -> {csv_path.name}")



# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  Road Surface Crack Dataset Preprocessing")
    print("=" * 65)

    has_closeup = False
    has_dashcam = False

    closeup_pairs: list[tuple[Path, Path]] = []
    dashcam_pairs: list[tuple[Path, Path]] = []

    # 1. Check for Closeup folder (Stage 1)
    closeup_dirs = find_image_mask_dirs(CRACK_CLOSEUP_DIR)
    if closeup_dirs:
        img_d, mask_d = closeup_dirs
        closeup_pairs = load_pairs(img_d, mask_d, tag="Closeup (Stage 1)")
        if closeup_pairs:
            has_closeup = True
            report_stats(closeup_pairs, title="Closeup Dataset (Stage 1)")

    # 2. Check for Dashcam folder (Stage 2)
    dashcam_dirs = find_image_mask_dirs(CRACK_DASHCAM_DIR)
    if dashcam_dirs:
        img_d, mask_d = dashcam_dirs
        dashcam_pairs = load_pairs(img_d, mask_d, tag="Dashcam (Stage 2)")
        if dashcam_pairs:
            has_dashcam = True
            report_stats(dashcam_pairs, title="Dashcam Dataset (Stage 2)")

    # 3. Fallback: Check root data/crack/images and data/crack/masks
    if not has_closeup and not has_dashcam:
        root_dirs = find_image_mask_dirs(CRACK_DIR)
        if root_dirs:
            img_d, mask_d = root_dirs
            dashcam_pairs = load_pairs(img_d, mask_d, tag="Default Crack Dataset")
            if dashcam_pairs:
                has_dashcam = True
                report_stats(dashcam_pairs, title="Default Crack Dataset")

    if not has_closeup and not has_dashcam:
        print("\n[!] No image/mask pairs found.")
        print(f"    Expected either:")
        print(f"      1. {CRACK_CLOSEUP_DIR}/images and {CRACK_CLOSEUP_DIR}/masks")
        print(f"      2. {CRACK_DASHCAM_DIR}/images and {CRACK_DASHCAM_DIR}/masks")
        print(f"      3. {CRACK_IMG_DIR} and {CRACK_MASK_DIR}")
        return

    # Process and write splits
    print("\n--- Generating Split Manifests ---")

    # Closeup splits (Stage 1)
    if has_closeup and closeup_pairs:
        print("\nSplitting Closeup dataset:")
        c_splits = split_dataset(closeup_pairs)
        save_splits(c_splits, prefix="stage1_")
        save_splits(c_splits, prefix="closeup_")

    # Dashcam splits (Stage 2 / Default)
    if has_dashcam and dashcam_pairs:
        print("\nSplitting Dashcam dataset:")
        d_splits = split_dataset(dashcam_pairs)
        save_splits(d_splits, prefix="")            # standard train.csv, val.csv, test.csv
        save_splits(d_splits, prefix="dashcam_")

    # Combined dataset (if both exist)
    if has_closeup and has_dashcam and closeup_pairs and dashcam_pairs:
        print("\nCreating Combined dataset:")
        combined_splits = {
            "train": c_splits["train"] + d_splits["train"],
            "val"  : c_splits["val"] + d_splits["val"],
            "test" : d_splits["test"],  # evaluation primarily on target dashcam/drone domain
        }
        save_splits(combined_splits, prefix="combined_")

    print(f"\n[OK] Preprocessing complete! All manifests saved to: {CRACK_SPLIT_DIR}")


if __name__ == "__main__":
    main()


