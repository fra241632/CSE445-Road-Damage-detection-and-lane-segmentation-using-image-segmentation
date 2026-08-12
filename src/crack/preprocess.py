"""
src/crack/preprocess.py
------------------------
Preprocessing and EDA utilities for the crack detection dataset.

Steps performed:
  1. load_pairs()      – pair images with masks by filename stem
  2. report_stats()    – print resolution info + class imbalance
  3. split_dataset()   – 70/15/15 deterministic train/val/test split
  4. save_splits()     – write train.csv / val.csv / test.csv manifests

Usage (after download_data.py --extract has been run):
    python src/crack/preprocess.py
"""

import os
import sys
import csv
import random
import glob
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

CRACK_IMG_DIR   = config.CRACK_IMG_DIR
CRACK_MASK_DIR  = config.CRACK_MASK_DIR
CRACK_SPLIT_DIR = config.CRACK_SPLIT_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# 1. Pair loading
# ---------------------------------------------------------------------------
def load_pairs(images_dir: Path = CRACK_IMG_DIR,
               masks_dir:  Path = CRACK_MASK_DIR) -> list[tuple[Path, Path]]:
    """
    Match images with masks by filename stem (ignoring extension).
    Returns a sorted list of (image_path, mask_path) tuples.
    Raises AssertionError if any image has no corresponding mask.
    """
    images_dir = Path(images_dir)
    masks_dir  = Path(masks_dir)

    img_map  = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    mask_map = {p.stem: p for p in masks_dir.iterdir()  if p.suffix.lower() in IMAGE_EXTS}

    common = sorted(set(img_map) & set(mask_map))
    only_img  = set(img_map) - set(mask_map)
    only_mask = set(mask_map) - set(img_map)

    if only_img:
        print(f"[!] {len(only_img)} images have no matching mask — skipped.")
    if only_mask:
        print(f"[!] {len(only_mask)} masks have no matching image — skipped.")

    pairs = [(img_map[stem], mask_map[stem]) for stem in common]
    print(f"[✓] Loaded {len(pairs)} matched image-mask pairs.")
    return pairs


# ---------------------------------------------------------------------------
# 2. EDA statistics
# ---------------------------------------------------------------------------
def report_stats(pairs: list[tuple[Path, Path]], n_sample: int = 10) -> None:
    """
    Print per-image stats (resolution, crack pixel ratio) for up to n_sample pairs.
    Also prints dataset-level summary (min/max/mean resolution, overall imbalance).
    """
    if not pairs:
        print("[!] No pairs to report on.")
        return

    sample = pairs[:n_sample]
    heights, widths, fg_ratios = [], [], []

    print(f"\n{'File':<30} {'H×W':>12} {'Crack %':>10}")
    print("-" * 56)

    for img_path, mask_path in sample:
        img  = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"  [!] Could not read {img_path.name} or its mask — skipped.")
            continue
        h, w = img.shape[:2]
        fg   = float((mask > 127).mean())
        heights.append(h); widths.append(w); fg_ratios.append(fg)
        print(f"  {img_path.name:<28} {h}×{w:>5}   {fg:8.3%}")

    if heights:
        print("\n── Dataset-level summary ──────────────────────────────")
        print(f"  Resolutions : H {min(heights)}–{max(heights)} px   W {min(widths)}–{max(widths)} px")
        print(f"  Mean crack  : {np.mean(fg_ratios):.3%}  (background: {1-np.mean(fg_ratios):.3%})")
        print(f"  Total pairs : {len(pairs)}")
        print("────────────────────────────────────────────────────────\n")


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
    Returns dict with keys 'train', 'val', 'test'.
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    n       = len(shuffled)
    n_train = int(n * train)
    n_val   = int(n * val)

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
                output_dir: Path = CRACK_SPLIT_DIR) -> None:
    """
    Write train.csv / val.csv / test.csv, each with columns: image_path, mask_path.
    Paths are written as absolute strings so notebooks can load them from any cwd.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, pairs in splits.items():
        csv_path = output_dir / f"{split_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "mask_path"])
            for img_p, mask_p in pairs:
                writer.writerow([str(img_p.resolve()), str(mask_p.resolve())])
        print(f"[✓] Saved {split_name}.csv ({len(pairs)} rows) → {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Crack Dataset Preprocessing")
    print("=" * 60)

    if not CRACK_IMG_DIR.exists() or not any(CRACK_IMG_DIR.iterdir()):
        print(f"\n[!] No images found in {CRACK_IMG_DIR}")
        print("    Run: python src/crack/download_data.py --extract   first.")
        return

    pairs  = load_pairs()
    report_stats(pairs)
    splits = split_dataset(pairs)
    save_splits(splits)
    print("\n[✓] Preprocessing complete. Split manifests saved to:", CRACK_SPLIT_DIR)


if __name__ == "__main__":
    main()
