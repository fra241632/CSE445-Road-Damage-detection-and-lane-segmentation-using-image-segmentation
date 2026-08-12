"""
src/lane/download_data.py
--------------------------
Guides the user through obtaining the TuSimple lane detection dataset,
then organizes raw files into the expected directory layout:

    data/lane/
        images/       ← all raw RGB frames (.jpg)
        masks/        ← generated binary lane masks (.png)
        annotations/  ← original TuSimple *.json label files

TuSimple is hosted on GitHub and requires Kaggle or direct download.
After running preprocess.py, masks/ will be populated from the JSON annotations.

Usage:
    python src/lane/download_data.py [--extract]
"""

import os
import sys
import zipfile
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

LANE_DIR       = config.LANE_DIR
LANE_IMG_DIR   = config.LANE_IMG_DIR
LANE_MASK_DIR  = config.LANE_MASK_DIR
LANE_JSON_DIR  = config.LANE_JSON_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

INSTRUCTIONS = """
========================================================
  TuSimple – Dataset Download Instructions
========================================================

Option A – via Kaggle (recommended, no account needed for download):
  https://www.kaggle.com/datasets/manideep1108/tusimple

  1. Download the dataset zip from Kaggle.
  2. Place the downloaded .zip file(s) in:
     {lane_dir}
  3. Re-run with --extract:
     python src/lane/download_data.py --extract

Option B – via the official TuSimple benchmark repo:
  https://github.com/TuSimple/tusimple-benchmark
  Follow the Google Drive links in the README to download:
      train_set.zip, test_set.zip (and test_label.json)
  Place them in {lane_dir} and re-run with --extract.

After extraction, run:
    python src/lane/preprocess.py
to convert the JSON lane annotations into binary mask PNGs.

Directory layout after this script + preprocess.py:
    data/lane/
        images/       ← all .jpg frames
        masks/        ← binary PNG masks (one per frame)
        annotations/  ← the original TuSimple JSON label files

========================================================
""".format(lane_dir=LANE_DIR)


# ---------------------------------------------------------------------------
# Extract helpers
# ---------------------------------------------------------------------------
def _collect_images(root: Path) -> list[Path]:
    """Recursively find all image files under root."""
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def _collect_jsons(root: Path) -> list[Path]:
    """Collect TuSimple JSON annotation files (label_data_*.json / test_label.json)."""
    return list(root.rglob("*.json"))


def extract_archives():
    """Unzip TuSimple archives and flatten images + JSONs into expected dirs."""
    LANE_DIR.mkdir(parents=True, exist_ok=True)
    LANE_IMG_DIR.mkdir(parents=True, exist_ok=True)
    LANE_MASK_DIR.mkdir(parents=True, exist_ok=True)
    LANE_JSON_DIR.mkdir(parents=True, exist_ok=True)

    zips = list(LANE_DIR.glob("*.zip"))
    if not zips:
        print(f"[!] No .zip archives found in {LANE_DIR}")
        print("    Follow the instructions above, place the archives, then re-run with --extract.")
        return

    for zp in zips:
        extract_to = LANE_DIR / zp.stem
        print(f"[+] Extracting {zp.name} → {extract_to} ...")
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(extract_to)

    # Collect and copy images
    all_imgs  = []
    all_jsons = []
    for zp in zips:
        extract_to = LANE_DIR / zp.stem
        all_imgs.extend(_collect_images(extract_to))
        all_jsons.extend(_collect_jsons(extract_to))

    print(f"[+] Copying {len(all_imgs)} images to {LANE_IMG_DIR} ...")
    for p in all_imgs:
        shutil.copy2(p, LANE_IMG_DIR / p.name)

    print(f"[+] Copying {len(all_jsons)} JSON files to {LANE_JSON_DIR} ...")
    for p in all_jsons:
        shutil.copy2(p, LANE_JSON_DIR / p.name)

    final_imgs  = len(list(LANE_IMG_DIR.iterdir()))
    final_jsons = len(list(LANE_JSON_DIR.iterdir()))
    print(f"\n[✓] Done. images/: {final_imgs} files   annotations/: {final_jsons} JSON files")
    print(f"[→] Now run: python src/lane/preprocess.py   to generate binary masks.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TuSimple dataset setup.")
    parser.add_argument("--extract", action="store_true",
                        help="Extract downloaded archives and organize files.")
    args = parser.parse_args()

    if args.extract:
        extract_archives()
    else:
        LANE_DIR.mkdir(parents=True, exist_ok=True)
        print(INSTRUCTIONS)


if __name__ == "__main__":
    main()
