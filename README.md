# Road Surface Crack Detection Using Deep Learning

**CSE445 Machine Learning Project**  
Group 02 | Section 07

---

## Overview

Supervised semantic segmentation project focused on pavement crack detection:

| Task | Dataset | Primary Model | Comparison Model | Output |
|---|---|---|---|---|
| Crack Detection | CRACK500 + DeepCrack (closeup) + Dashcam footage | U-Net (2-Stage) | DeepLabv3 (ResNet50) | Binary crack mask |

Framed as pixel-wise binary segmentation, trained with a combined BCE + Dice loss to handle extreme class imbalance (crack pixels are only 2–5% of total image area).

A **2-stage transfer learning** approach is used:
- **Stage 1**: Pre-train U-Net on high-contrast closeup crack images (CRACK500, DeepCrack)
- **Stage 2**: Fine-tune on real-world dashcam/drone survey footage at a lower learning rate

---

## Repository Structure

```
Road_Damage_Project/
├── config.py                        ← All paths & hyperparameters (edit here first)
├── main.py                          ← CLI entry point (--setup, --infer, --jupyter)
├── requirements.txt
├── data/
│   ├── README.md                    ← Dataset download instructions
│   ├── best/                        ← Demo/result videos
│   └── crack/
│       ├── closeup/                 ← Stage 1 data (CRACK500, DeepCrack, macro shots)
│       │   ├── images/
│       │   └── masks/
│       ├── dashcam/                 ← Stage 2 data (real-world dashcam/drone footage)
│       │   ├── images/
│       │   └── masks/
│       └── splits/                  ← Auto-generated CSV manifests (train/val/test)
├── support/
│   ├── crack/
│   │   ├── download_data.py         ← Dataset download helpers
│   │   ├── preprocess.py            ← Pairing, EDA, split, CSV manifests
│   │   └── train_2stage.py          ← 2-stage training script (Stage 1 → Stage 2)
│   └── shared/
│       ├── dataset.py               ← SegmentationDataset (reads CSV manifests)
│       ├── transforms.py            ← Albumentations train/val pipelines
│       ├── unet.py                  ← U-Net architecture (~7.8M params)
│       ├── deeplabv3.py             ← DeepLabv3 architecture wrapper
│       ├── losses.py                ← BCEDiceLoss, FocalLoss
│       ├── metrics.py               ← IoU, Dice, Pixel Accuracy, Precision, Recall
│       ├── inference.py             ← CrackPredictor (image & video inference)
│       └── trainer.py               ← Trainer with early stopping + LR scheduling
├── notebooks/
│   ├── 01_EDA_crack.ipynb           ← EDA and augmentation preview
│   ├── 02_train_crack_unet.ipynb    ← Train U-Net (2-stage) + learning curves
│   ├── 03_evaluate_crack.ipynb      ← Test set evaluation + failure analysis
│   ├── 04_compare_crack_deeplabv3.ipynb ← Head-to-head: U-Net vs DeepLabv3
│   └── 05_inference_crack.ipynb     ← Interactive image/video inference demo
└── experiments/
    ├── crack_stage1_unet/           ← Stage 1 checkpoint (best_model.pth, train_log.csv)
    └── crack_stage2_unet_finetuned/ ← Stage 2 checkpoint (best_model.pth, train_log.csv)
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Windows users**: If you have a GPU, also install the CUDA-enabled PyTorch version from https://pytorch.org/get-started/locally/

### 2. Organize Datasets

Place your datasets under `data/crack/` following this structure:

```
data/crack/
├── closeup/
│   ├── images/    ← CRACK500 / DeepCrack images
│   └── masks/     ← Corresponding binary masks
└── dashcam/
    ├── images/    ← Dashcam / drone survey images
    └── masks/     ← Corresponding binary masks
```

### 3. Preprocess (Generate Split CSVs)

```bash
python support/crack/preprocess.py
```

### 4. Train the Model

```bash
# Run full 2-stage training (Stage 1 → Stage 2)
python support/crack/train_2stage.py

# Or run a specific stage:
python support/crack/train_2stage.py --stage 1   # Stage 1 only
python support/crack/train_2stage.py --stage 2   # Stage 2 fine-tuning only
```

### 5. Run Notebooks (in order)

```bash
python main.py --jupyter
```

Then open in order:

```
01_EDA_crack.ipynb               → EDA and augmentation preview
02_train_crack_unet.ipynb        → Train U-Net on crack detection
03_evaluate_crack.ipynb          → Test set evaluation + failure analysis
04_compare_crack_deeplabv3.ipynb → Head-to-head comparison: U-Net vs DeepLabv3
05_inference_crack.ipynb         → Run crack detection on your own images/videos
```

### 6. Run Inference (CLI)

```bash
# On an image
python main.py --infer path/to/road_image.jpg

# On a video
python main.py --infer path/to/dashcam_video.mp4 --output result.mp4
```

### 7. Google Colab

1. Upload the project folder to Google Drive.
2. Open any notebook in Google Colab.
3. Set `RUN_ENV=colab` (done automatically at the top of each notebook).
4. Runs seamlessly on T4 GPU acceleration.

---

## Architecture

**U-Net** (Ronneberger et al., 2015)
- Encoder: 4 × DoubleConv + MaxPool
- Bottleneck: DoubleConv
- Decoder: 4 × Bilinear Upsample + skip-concat + DoubleConv
- Output: 1×1 Conv → Sigmoid → probability map [0, 1]
- ~7.8 M parameters with `base_features=32`, input 256×256

**Loss**: BCE + Dice (0.5 / 0.5 weighted)  
**Optimiser**: Adam + ReduceLROnPlateau  
**Early stopping**: patience-based, monitored on val IoU

---

## ML Concepts Demonstrated

| Concept | Location |
|---|---|
| Supervised learning | Labeled image–mask pairs |
| Transfer learning (2-stage) | `support/crack/train_2stage.py` |
| Dataset splitting (70/15/15) | `support/crack/preprocess.py` |
| Preprocessing & normalisation | `support/shared/transforms.py` |
| Data augmentation | `support/shared/transforms.py` |
| Backpropagation | PyTorch autograd in `support/shared/trainer.py` |
| Loss optimisation | `BCEDiceLoss`, Adam, LR scheduling |
| Overfitting analysis | Train vs val curves in notebook 02 |
| Hyperparameter tuning | `config.py` + experiment runs |
| Quantitative evaluation | IoU, Dice, Pixel Acc in notebooks 03 & 04 |
| Real-world inference | Image + video pipeline in `inference.py` |

---

## Authors

Md. Sakif Chowdhury (2233359642) | Araf Hussain (2111078642) | Dihan Shahriar (2031839042)  
Section 07 | Group 02 | CSE445
