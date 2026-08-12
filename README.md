# Road Damage Detection & Lane Segmentation Using Deep Learning

**CSE445 Machine Learning Project**
Group 02 | Section 07

---

## Overview

Two-task supervised semantic segmentation project:

| Task | Dataset | Model | Output |
|---|---|---|---|
| Crack Detection | CRACK500 + DeepCrack | U-Net | Binary crack mask |
| Lane Segmentation | TuSimple | U-Net | Binary lane mask |

Both tasks are framed as pixel-wise binary segmentation, trained with
a combined BCE + Dice loss to handle extreme class imbalance
(crack/lane pixels are only 2–8% of total image area).

---

## Repository Structure

```
ML Project/
├── config.py                  ← All paths & hyperparameters (edit here first)
├── requirements.txt
├── data/
│   ├── README.md              ← Dataset download instructions
│   ├── crack/
│   │   ├── images/            ← CRACK500 + DeepCrack images (after download)
│   │   ├── masks/             ← Binary crack masks
│   │   └── splits/            ← train.csv / val.csv / test.csv (70/15/15)
│   └── lane/
│       ├── images/            ← TuSimple frames
│       ├── masks/             ← Generated binary lane masks
│       ├── annotations/       ← TuSimple JSON label files
│       └── splits/            ← train.csv / val.csv / test.csv
├── src/
│   ├── crack/
│   │   ├── download_data.py   ← CRACK500/DeepCrack setup
│   │   └── preprocess.py      ← Pairing, EDA, split, CSV manifests
│   ├── lane/
│   │   ├── download_data.py   ← TuSimple setup
│   │   └── preprocess.py      ← JSON→mask, pairing, split, CSV manifests
│   └── shared/
│       ├── dataset.py         ← SegmentationDataset (used by both tasks)
│       ├── transforms.py      ← Albumentations train/val pipelines
│       ├── unet.py            ← U-Net architecture
│       ├── losses.py          ← BCEDiceLoss, FocalLoss
│       ├── metrics.py         ← IoU, Dice, Pixel Accuracy
│       └── trainer.py         ← Trainer with early stopping + LR scheduling
├── notebooks/
│   ├── 01_EDA_crack.ipynb
│   ├── 02_EDA_lane.ipynb
│   ├── 03_train_crack_unet.ipynb
│   ├── 04_train_lane_unet.ipynb
│   ├── 05_evaluate_crack.ipynb
│   └── 06_evaluate_lane.ipynb
└── experiments/
    ├── crack_unet_run1/       ← best_model.pth, train_log.csv, curves.png
    └── lane_unet_run1/        ← same structure
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
# Crack detection (CRACK500 + DeepCrack)
python src/crack/download_data.py
# Follow printed instructions, then:
python src/crack/download_data.py --extract

# Lane segmentation (TuSimple)
python src/lane/download_data.py
python src/lane/download_data.py --extract
```

### 3. Preprocess

```bash
python src/crack/preprocess.py   # generates split CSVs for crack
python src/lane/preprocess.py    # JSON→masks + split CSVs for lane
```

### 4. Run Notebooks (in order)

Open in Jupyter or Google Colab:

```
01_EDA_crack.ipynb        → EDA and augmentation preview
02_EDA_lane.ipynb         → TuSimple EDA
03_train_crack_unet.ipynb → Train U-Net on crack detection
04_train_lane_unet.ipynb  → Train U-Net on lane segmentation
05_evaluate_crack.ipynb   → Test set evaluation + failure analysis
06_evaluate_lane.ipynb    → Test set evaluation + failure analysis
```

### 5. Google Colab

Set `RUN_ENV=colab` (done automatically at the top of each notebook).
Mount your Drive and point `COLAB_ROOT` in `config.py` to your project folder.

---

## Architecture

**U-Net** (Ronneberger et al., 2015)
- Encoder: 4 × DoubleConv + MaxPool
- Bottleneck: DoubleConv
- Decoder: 4 × Upsample + skip-concat + DoubleConv
- Output: 1×1 Conv → Sigmoid → probability map [0, 1]
- ~7.8 M parameters with `base_features=32`, input 256×256

**Loss**: BCE + Dice (0.5 / 0.5 weighted)

**Optimiser**: Adam (lr=1e-4) + ReduceLROnPlateau (patience=5)

**Early stopping**: patience=10 epochs (monitored on val IoU)

---

## ML Concepts Demonstrated

| Concept | Location |
|---|---|
| Supervised learning | Labeled image–mask pairs |
| Dataset splitting (70/15/15) | `src/crack/preprocess.py`, `src/lane/preprocess.py` |
| Preprocessing & normalisation | `src/shared/transforms.py` |
| Data augmentation | `src/shared/transforms.py` |
| Backpropagation | PyTorch autograd in `src/shared/trainer.py` |
| Loss optimisation | `BCEDiceLoss`, Adam, LR scheduling |
| Overfitting analysis | Train vs val curves in notebooks 03 & 04 |
| Hyperparameter tuning | `config.py` + multiple experiment runs |
| Quantitative evaluation | IoU, Dice, Pixel Acc in notebooks 05 & 06 |

---

## Authors

Md. Sakif Chowdhury (2233359642) | Section 07 | Group 02 | CSE445
