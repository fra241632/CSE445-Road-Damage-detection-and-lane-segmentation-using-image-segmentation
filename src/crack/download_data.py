"""
src/crack/download_data.py
--------------------------
Guides the user through obtaining CRACK500 (and optionally DeepCrack),
then organizes the raw archives into the expected directory layout:

    data/crack/
        images/   ← all RGB crack images (.jpg / .png)
        masks/    ← corresponding binary masks (.png, same stem)

Usage:
    python src/crack/download_data.py [--extract]

Flags:
    --extract   After placing the archives in data/crack/, run this flag
                to automatically unzip and flatten into images/ and masks/.
"""

import os
import sys
import zipfile
import shutil
import argparse
from pathlib import Path

# Make config importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

CRACK_DIR      = config.CRACK_DIR
CRACK_IMG_DIR  = config.CRACK_IMG_DIR
CRACK_MASK_DIR = config.CRACK_MASK_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ---------------------------------------------------------------------------
# Download instructions
# ---------------------------------------------------------------------------
INSTRUCTIONS = """
========================================================
  CRACK500 – Dataset Download Instructions
========================================================

1. Fill out the request form to obtain CRACK500:
   https://github.com/fyangneil/pavement-crack-detection

   The authors typically share a Google Drive link by e-mail.
   Download the zip (usually named CRACK500.zip or similar).

2. (Optional) DeepCrack supplementary dataset:
   https://github.com/yhlleo/DeepCrack
   Click "Google Drive" link in the README → download DeepCrack.zip

3. Place the downloaded .zip file(s) in:
   {crack_dir}

4. Then re-run this script with the --extract flag:
   python src/crack/download_data.py --extract

The script will:
  - Unzip the archives
  - Flatten all images into:   {img_dir}
  - Flatten all masks  into:   {mask_dir}
  - Print a summary of pairs found

========================================================
""".format(
    crack_dir=CRACK_DIR,
    img_dir=CRACK_IMG_DIR,
    mask_dir=CRACK_MASK_DIR,
)

# ---------------------------------------------------------------------------
# CRACK500 layout helpers
# ---------------------------------------------------------------------------
# CRACK500 typically ships as:
#   CRACK500/
#     train/   image/ gt/
#     test/    image/ gt/
#     val/     image/ gt/
# DeepCrack ships as:
#   DeepCrack/
#     train_img/  train_lab/
#     test_img/   test_lab/

def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS

def _flatten_crack500(archive_root: Path) -> tuple[list[Path], list[Path]]:
    """Walk a CRACK500-style directory and collect (image, mask) pairs."""
    images, masks = [], []
    for split in ["train", "val", "test"]:
        img_dir  = archive_root / split / "image"
        mask_dir = archive_root / split / "gt"
        if not img_dir.exists():
            # some versions use 'images' / 'masks'
            img_dir  = archive_root / split / "images"
            mask_dir = archive_root / split / "masks"
        if img_dir.exists():
            images.extend(sorted(p for p in img_dir.iterdir() if _is_image(p)))
            masks.extend( sorted(p for p in mask_dir.iterdir() if _is_image(p)))
    return images, masks

def _flatten_deepcrack(archive_root: Path) -> tuple[list[Path], list[Path]]:
    """Walk a DeepCrack-style directory and collect (image, mask) pairs."""
    images, masks = [], []
    for img_sub, mask_sub in [("train_img", "train_lab"), ("test_img", "test_lab")]:
        img_dir  = archive_root / img_sub
        mask_dir = archive_root / mask_sub
        if img_dir.exists():
            images.extend(sorted(p for p in img_dir.iterdir() if _is_image(p)))
            masks.extend( sorted(p for p in mask_dir.iterdir() if _is_image(p)))
    return images, masks

# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------
def extract_archives():
    """Find all .zip files in CRACK_DIR, extract, then copy into images/ masks/."""
    CRACK_DIR.mkdir(parents=True, exist_ok=True)
    CRACK_IMG_DIR.mkdir(parents=True, exist_ok=True)
    CRACK_MASK_DIR.mkdir(parents=True, exist_ok=True)

    zips = list(CRACK_DIR.glob("*.zip"))
    if not zips:
        print(f"[!] No .zip archives found in {CRACK_DIR}")
        print("    Place the downloaded archives there first, then re-run with --extract.")
        return

    total_images, total_masks = [], []

    for zp in zips:
        extract_to = CRACK_DIR / zp.stem
        print(f"[+] Extracting {zp.name} → {extract_to} ...")
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(extract_to)

        # Try CRACK500 layout first
        imgs, masks = _flatten_crack500(extract_to)
        if not imgs:
            imgs, masks = _flatten_deepcrack(extract_to)
        if not imgs:
            # Generic fallback: any image in an 'image'/'images' subdir
            imgs  = [p for p in extract_to.rglob("*") if _is_image(p) and "mask" not in str(p).lower() and "gt" not in str(p).lower()]
            masks = [p for p in extract_to.rglob("*") if _is_image(p) and ("mask" in str(p).lower() or "gt" in str(p).lower())]

        print(f"    Found {len(imgs)} images and {len(masks)} masks.")
        total_images.extend(imgs)
        total_masks.extend(masks)

    # Copy into flat directories
    print(f"[+] Copying {len(total_images)} images to {CRACK_IMG_DIR} ...")
    for p in total_images:
        shutil.copy2(p, CRACK_IMG_DIR / p.name)

    print(f"[+] Copying {len(total_masks)} masks to {CRACK_MASK_DIR} ...")
    for p in total_masks:
        shutil.copy2(p, CRACK_MASK_DIR / p.name)

    # Summary
    final_imgs  = len(list(CRACK_IMG_DIR.iterdir()))
    final_masks = len(list(CRACK_MASK_DIR.iterdir()))
    print(f"\n[✓] Done. images/: {final_imgs} files   masks/: {final_masks} files")
    if final_imgs != final_masks:
        print("[!] WARNING: image and mask counts differ — check for naming mismatches.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CRACK500/DeepCrack dataset setup.")
    parser.add_argument("--extract", action="store_true",
                        help="Extract downloaded archives and organize into images/ masks/")
    args = parser.parse_args()

    if args.extract:
        extract_archives()
    else:
        CRACK_DIR.mkdir(parents=True, exist_ok=True)
        print(INSTRUCTIONS)

if __name__ == "__main__":
    main()
