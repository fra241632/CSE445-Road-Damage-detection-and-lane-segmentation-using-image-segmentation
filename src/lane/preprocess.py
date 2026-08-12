"""
src/lane/preprocess.py
-----------------------
Preprocessing and EDA utilities for the TuSimple lane segmentation dataset.

Steps performed:
  1. parse_tusimple_json()  – convert TuSimple JSON annotations → binary PNG masks
  2. load_pairs()           – pair generated mask PNGs with their source frames
  3. report_stats()         – print resolution + lane pixel ratio
  4. split_dataset()        – 70/15/15 deterministic split
  5. save_splits()          – write train.csv / val.csv / test.csv manifests

TuSimple annotation format (per-line JSON in label_data_*.json):
  {
    "lanes":     [[-2, -2, ..., x_coord, ...], ...],  # x coords per row
    "h_samples": [160, 170, ..., 710],                # y coords
    "raw_file":  "clips/0313-1/6040/20.jpg"
  }
  A coord of -2 means the lane is not visible at that y position.

Usage:
    python src/lane/preprocess.py
"""

import os
import sys
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

LANE_IMG_DIR   = config.LANE_IMG_DIR
LANE_MASK_DIR  = config.LANE_MASK_DIR
LANE_JSON_DIR  = config.LANE_JSON_DIR
LANE_SPLIT_DIR = config.LANE_SPLIT_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
LANE_THICKNESS = 12   # pixels — thick enough to be learnable at 256×256


# ---------------------------------------------------------------------------
# 1. Parse TuSimple JSON → binary masks
# ---------------------------------------------------------------------------
def parse_tusimple_json(json_dir: Path = LANE_JSON_DIR,
                        images_dir: Path = LANE_IMG_DIR,
                        masks_dir:  Path = LANE_MASK_DIR) -> int:
    """
    Reads all TuSimple JSON label files, draws lane polylines onto blank
    binary masks, and saves them as PNG files in masks_dir.

    Returns the number of masks generated.

    The mask filename mirrors the frame filename (using '_' instead of '/'):
        clips/0313-1/6040/20.jpg → clips_0313-1_6040_20.png
    This flat naming allows easy glob-based pairing.
    """
    json_dir   = Path(json_dir)
    images_dir = Path(images_dir)
    masks_dir  = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"[!] No JSON files found in {json_dir}")
        print("    Run download_data.py --extract first.")
        return 0

    count = 0
    for json_path in json_files:
        print(f"[+] Parsing {json_path.name} ...")
        with open(json_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            ann = json.loads(line)

            raw_file  = ann["raw_file"]           # e.g. "clips/0313-1/6040/20.jpg"
            h_samples = ann["h_samples"]          # list of y coordinates
            lanes     = ann["lanes"]              # list of lists of x coordinates

            # Derive flat mask filename from raw_file path
            stem      = raw_file.replace("/", "_").replace("\\", "_")
            stem      = Path(stem).stem            # remove extension
            mask_name = stem + ".png"
            mask_path = masks_dir / mask_name

            # Find the corresponding image to get its dimensions
            # Try the flat images_dir first, then reconstruct relative path
            flat_img_name = Path(raw_file).name
            img_path_flat = images_dir / flat_img_name

            if img_path_flat.exists():
                img = cv2.imread(str(img_path_flat))
                if img is not None:
                    H, W = img.shape[:2]
                else:
                    H, W = 720, 1280   # TuSimple default
            else:
                H, W = 720, 1280       # TuSimple default

            # Draw lanes
            mask = np.zeros((H, W), dtype=np.uint8)
            for lane_xs in lanes:
                points = []
                for x, y in zip(lane_xs, h_samples):
                    if x >= 0:                      # -2 means not visible
                        points.append((int(x), int(y)))
                if len(points) >= 2:
                    for i in range(len(points) - 1):
                        cv2.line(mask, points[i], points[i + 1],
                                 color=255, thickness=LANE_THICKNESS)

            cv2.imwrite(str(mask_path), mask)
            count += 1

    print(f"[✓] Generated {count} binary mask PNGs in {masks_dir}")
    return count


# ---------------------------------------------------------------------------
# 2. Pair loading
# ---------------------------------------------------------------------------
def load_pairs(images_dir: Path = LANE_IMG_DIR,
               masks_dir:  Path = LANE_MASK_DIR) -> list[tuple[Path, Path]]:
    """
    Match images in images_dir with generated masks in masks_dir.

    TuSimple images are flat (.jpg files); generated masks are .png files
    whose stems are derived from the original relative path (with '/' → '_').
    This function handles the stem-matching heuristic.
    """
    images_dir = Path(images_dir)
    masks_dir  = Path(masks_dir)

    mask_map = {p.stem: p for p in masks_dir.iterdir() if p.suffix.lower() == ".png"}

    pairs = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = img_path.stem
        if stem in mask_map:
            pairs.append((img_path, mask_map[stem]))

    unmatched = len(list(images_dir.iterdir())) - len(pairs)
    print(f"[✓] Matched {len(pairs)} image-mask pairs ({unmatched} images unmatched).")
    return pairs


# ---------------------------------------------------------------------------
# 3. EDA statistics
# ---------------------------------------------------------------------------
def report_stats(pairs: list[tuple[Path, Path]], n_sample: int = 10) -> None:
    """Print per-image resolution and lane pixel ratio for up to n_sample pairs."""
    if not pairs:
        print("[!] No pairs to report on.")
        return

    sample = pairs[:n_sample]
    heights, widths, fg_ratios = [], [], []

    print(f"\n{'File':<35} {'H×W':>12} {'Lane %':>10}")
    print("-" * 60)

    for img_path, mask_path in sample:
        img  = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"  [!] Could not read {img_path.name} — skipped.")
            continue
        h, w = img.shape[:2]
        fg   = float((mask > 127).mean())
        heights.append(h); widths.append(w); fg_ratios.append(fg)
        print(f"  {img_path.name:<33} {h}×{w:>5}   {fg:8.3%}")

    if heights:
        print("\n── Dataset-level summary ──────────────────────────────")
        print(f"  Resolutions : H {min(heights)}–{max(heights)} px   W {min(widths)}–{max(widths)} px")
        print(f"  Mean lane   : {np.mean(fg_ratios):.3%}  (background: {1-np.mean(fg_ratios):.3%})")
        print(f"  Total pairs : {len(pairs)}")
        print("────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# 4. Split
# ---------------------------------------------------------------------------
def split_dataset(pairs: list[tuple[Path, Path]],
                  train: float = config.TRAIN_RATIO,
                  val:   float = config.VAL_RATIO,
                  test:  float = config.TEST_RATIO,
                  seed:  int   = config.RANDOM_SEED
                  ) -> dict[str, list[tuple[Path, Path]]]:
    """Deterministic 70/15/15 train/val/test split."""
    assert abs(train + val + test - 1.0) < 1e-6
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
# 5. Save manifests
# ---------------------------------------------------------------------------
def save_splits(splits: dict[str, list[tuple[Path, Path]]],
                output_dir: Path = LANE_SPLIT_DIR) -> None:
    """Write train.csv / val.csv / test.csv with absolute image_path, mask_path."""
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
    print("  Lane Dataset Preprocessing")
    print("=" * 60)

    if not LANE_JSON_DIR.exists() or not any(LANE_JSON_DIR.glob("*.json")):
        print(f"\n[!] No JSON annotation files found in {LANE_JSON_DIR}")
        print("    Run: python src/lane/download_data.py --extract   first.")
        return

    # Step 1: generate masks from JSON
    parse_tusimple_json()

    # Step 2: load pairs
    pairs = load_pairs()
    if not pairs:
        print("[!] No matched pairs found after mask generation. Check your directory layout.")
        return

    # Step 3: EDA
    report_stats(pairs)

    # Step 4 & 5: split + save
    splits = split_dataset(pairs)
    save_splits(splits)

    print("\n[✓] Preprocessing complete. Split manifests saved to:", LANE_SPLIT_DIR)


if __name__ == "__main__":
    main()
