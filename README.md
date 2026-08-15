# Road Surface Crack Detection Using Deep Learning

**CSE445 Machine Learning Project**  
Group 02 | Section 07  

---

## Overview

Supervised semantic segmentation project focused on pavement crack detection:

| Task | Dataset | Primary Model | Comparison Model | Output |
|---|---|---|---|---|
| Crack Detection | CRACK500 + DeepCrack | U-Net | DeepLabv3 (ResNet50) | Binary crack mask |

Framed as pixel-wise binary segmentation, trained with a combined BCE + Dice loss to handle extreme class imbalance (crack pixels are only 2–5% of total image area).

---

## Repository Structure

```
ML Project/
├── config.py                  ← All paths & hyperparameters (edit here first)
├── requirements.txt
├── data/
│   ├── README.md              ← Dataset download instructions
│   └── crack/
│       ├── images/            ← CRACK500 + DeepCrack images (after download)
│       ├── masks/             ← Binary crack masks
│       └── splits/            ← train.csv / val.csv / test.csv (70/15/15)
├── support/
│   ├── crack/
│   │   ├── download_data.py   ← CRACK500/DeepCrack setup
│   │   └── preprocess.py      ← Pairing, EDA, split, CSV manifests
│   └── shared/
│       ├── dataset.py         ← SegmentationDataset
│       ├── transforms.py      ← Albumentations train/val pipelines
│       ├── unet.py            ← U-Net architecture
│       ├── deeplabv3.py       ← DeepLabv3 architecture wrapper
│       ├── losses.py          ← BCEDiceLoss, FocalLoss
│       ├── metrics.py         ← IoU, Dice, Pixel Accuracy, Precision, Recall
│       └── trainer.py         ← Trainer with early stopping + LR scheduling
├── notebooks/
│   ├── 01_EDA_crack.ipynb
│   ├── 02_train_crack_unet.ipynb
│   ├── 03_evaluate_crack.ipynb
│   └── 04_compare_crack_deeplabv3.ipynb
└── experiments/
    ├── crack_unet_run1/       ← best_model.pth, train_log.csv, curves.png
    └── crack_deeplabv3_run1/  ← DeepLabv3 experiment outputs
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
python support/crack/download_data.py
# Follow printed instructions, then:
python support/crack/download_data.py --extract
```

### 3. Preprocess

```bash
python support/crack/preprocess.py   # generates split CSVs for crack dataset
```

### 4. Run Notebooks (in order)

Open in Jupyter or Google Colab:

```
01_EDA_crack.ipynb               → EDA and augmentation preview
02_train_crack_unet.ipynb        → Train U-Net on crack detection
03_evaluate_crack.ipynb          → Test set evaluation + failure analysis
04_compare_crack_deeplabv3.ipynb → Head-to-head comparison: U-Net vs DeepLabv3
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
| Dataset splitting (70/15/15) | `support/crack/preprocess.py` |
| Preprocessing & normalisation | `support/shared/transforms.py` |
| Data augmentation | `support/shared/transforms.py` |
| Backpropagation | PyTorch autograd in `support/shared/trainer.py` |
| Loss optimisation | `BCEDiceLoss`, Adam, LR scheduling |
| Overfitting analysis | Train vs val curves in notebook 02 |
| Hyperparameter tuning | `config.py` + experiment runs |
| Quantitative evaluation | IoU, Dice, Pixel Acc in notebooks 03 & 04 |

---

## Authors

Md. Sakif Chowdhury (2233359642) | Section 07 | Group 02 | CSE445


