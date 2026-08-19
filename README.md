# Road Surface Crack Detection Using Deep Learning

<p align="center">
  <img src="assets/demo.gif" alt="Road Surface Crack Detection Demo" width="700">
</p>

**CSE445 Machine Learning Project**  
Group 02 | Section 07

## Video Demonstrations

Watch real-time crack segmentation inference on real-world dashcam and survey footage:

| 🎬 **Demo Video 1: Dashcam Road Crack Detection** | 🎬 **Demo Video 2: High-Resolution Pavement Survey** |
|:---:|:---:|
| [![Road Crack Detection Demo 1](https://img.youtube.com/vi/-uLCMcTAiGo/0.jpg)](https://www.youtube.com/watch?v=-uLCMcTAiGo) | [![Road Crack Detection Demo 2](https://img.youtube.com/vi/SEnlQzeg_Fk/0.jpg)](https://www.youtube.com/watch?v=SEnlQzeg_Fk) |
| [▶ **Watch Demo 1 on YouTube**](https://www.youtube.com/watch?v=-uLCMcTAiGo) | [▶ **Watch Demo 2 on YouTube**](https://www.youtube.com/watch?v=SEnlQzeg_Fk) |

---

## Overview

Supervised semantic segmentation project focused on pavement crack detection:

| Task | Dataset | Primary Model | Comparison Model | Output |
|---|---|---|---|---|
| Crack Detection | CRACK500 + DeepCrack (closeup) + Dashcam footage | U-Net (2-Stage) | DeepLabv3 (ResNet50, 2-Stage) | Binary crack mask |

Framed as pixel-wise binary segmentation, trained with a combined BCE + Dice loss to handle extreme class imbalance (crack pixels are only 2–5% of total image area).

A **2-stage transfer learning** approach is used for both models:
- **Stage 1**: Pre-train on high-contrast closeup crack images (CRACK500 + DeepCrack)
- **Stage 2**: Fine-tune on real-world dashcam footage at a lower learning rate

## Results

### Validation Performance (Best Epochs)
| Model & Stage | Best Epoch | Val IoU | Val Dice | Val Pixel Acc |
|---|:---:|:---:|:---:|:---:|
| U-Net — Stage 1 (Closeup pre-train) | 7 | 0.5118 | 0.6536 | 95.78% |
| U-Net — Stage 2 (Dashcam fine-tune) | 73 | 0.1930 | 0.2881 | 99.52% |
| DeepLabv3 — Stage 1 (Closeup pre-train) | 8 | 0.5801 | 0.7136 | 97.09% |
| DeepLabv3 — Stage 2 (Dashcam fine-tune) | 48 | 0.1335 | 0.2124 | 99.36% |

### Held-Out Test Split Performance (Stage 2 Dashcam)
| Model | Test IoU | Test Dice | Test Precision | Test Recall | Test Pixel Acc | Params |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **U-Net (Two-Stage)** | **0.2034** | **0.3009** | **0.3316** | **0.3644** | **99.51%** | **~7.8M** |
| DeepLabv3 (Two-Stage) | 0.1393 | 0.2219 | 0.2233 | 0.3018 | 99.35% | ~42.0M |

> **Note on Stage 2 IoU**: Human annotators draw thick, coarse masks around thin cracks in dashcam footage. The model predicts pixel-precise boundaries. IoU penalizes this mismatch heavily — the segmentation is qualitatively accurate even where the number appears low.

---

## Repository Structure

```
Road_Damage_Project/
├── config.py                        ← All paths & hyperparameters (edit here first)
├── main.py                          ← CLI entry point (--setup, --infer, --jupyter)
├── requirements.txt
├── data/
│   ├── README.md                    ← Dataset download instructions & layout
│   └── crack/
│       ├── closeup/                 ← Stage 1 data (CRACK500, DeepCrack, macro shots)
│       ├── dashcam/                 ← Stage 2 data (real-world dashcam/drone footage)
│       └── splits/                  ← Auto-generated CSV manifests (train/val/test)
├── others/
│   ├── final_presentation.pptx      ← Final project presentation
│   ├── final_report.pdf             ← Final IEEE-format research report
│   ├── update_presentation.pptx     ← Midterm update presentation
│   ├── update_report.pdf            ← Midterm progress report
│   └── demo_video.mp4               ← 1-minute system demo video
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
│   ├── 04_compare_crack_deeplabv3.ipynb ← Head-to-head: U-Net vs DeepLabv3 (2-stage)
│   └── 05_inference_crack.ipynb     ← Interactive image/video inference demo
├── assets/
│   └── demo.gif                     ← Demo GIF shown on GitHub repository page
└── experiments/
    ├── crack_stage1_unet/                  ← U-Net Stage 1 (best_model.pth, train_log.csv)
    ├── crack_stage2_unet_finetuned/        ← U-Net Stage 2 (best_model.pth, train_log.csv)
    ├── crack_stage1_deeplabv3/             ← DeepLabv3 Stage 1 checkpoint & log
    ├── crack_stage2_deeplabv3_finetuned/   ← DeepLabv3 Stage 2 checkpoint & log
    └── *.png                               ← Generated figures and evaluation plots
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

### 5. Running in VS Code or Jupyter

* **In VS Code**:
  1. Open this repository folder in VS Code (`File > Open Folder...`).
  2. Select your Python interpreter (`Ctrl+Shift+P` → `Python: Select Interpreter`).
  3. Open any `.ipynb` notebook from the `notebooks/` directory and click **Run All** or run cells step-by-step.
  4. Use the integrated VS Code terminal to run training or inference scripts.

* **In Classic Jupyter**:
  ```bash
  python main.py --jupyter
  ```

### 6. Run Inference (CLI)

```bash
# Test on a single road image
python main.py --infer path/to/road_image.jpg

# Test on a driving video
python main.py --infer path/to/dashcam_video.mp4 --output result.mp4
```

### 7. Google Colab Execution

1. Upload the project folder to Google Drive.
2. Open any notebook in Google Colab.
3. Set `RUN_ENV=colab` (detected automatically at the top of each notebook).
4. Select `Runtime > Change runtime type > T4 GPU`.

---

## Architecture

**U-Net** (Ronneberger et al., 2015) — Primary Model
- Encoder: 4 × DoubleConv + MaxPool, channels `32 → 64 → 128 → 256`
- Bottleneck: DoubleConv at 16×16 (512 channels)
- Decoder: 4 × Bilinear Upsample + skip-concat + DoubleConv
- Output: 1×1 Conv → Sigmoid → probability map [0, 1]
- ~7.8 M parameters, input 256×256

**DeepLabv3** (ResNet-50 backbone) — Comparison Model
- Backbone: ResNet-50 pretrained on ImageNet (~40M parameters)
- ASPP: 5 parallel atrous branches (rates: 1, 6, 12, 18 + global average pooling)
- Output: Bilinear ×16 upsampling → binary mask

**Loss**: BCE + Dice (0.5 / 0.5 weighted)  
**Optimiser**: Adam + ReduceLROnPlateau  
**Early stopping**: patience-based, monitored on val IoU

---

## ML Concepts Demonstrated

| Concept | Location |
|---|---|
| Supervised learning | Labeled image–mask pairs |
| Transfer learning (2-stage) | `support/crack/train_2stage.py` |
| Multi-model comparison | `notebooks/04_compare_crack_deeplabv3.ipynb` |
| Dataset splitting (70/15/15) | `support/crack/preprocess.py` |
| Preprocessing & normalisation | `support/shared/transforms.py` |
| Data augmentation | `support/shared/transforms.py` |
| Backpropagation | PyTorch autograd in `support/shared/trainer.py` |
| Loss optimisation | `BCEDiceLoss`, Adam, LR scheduling |
| Overfitting analysis | Train vs val curves in notebook 02 |
| Hyperparameter tuning | `config.py` + experiment runs |
| Quantitative evaluation | IoU, Dice, Pixel Acc in notebooks 03 & 04 |
| Real-world video inference | Image + video pipeline in `support/shared/inference.py` |

---

## Authors

Md. Sakif Chowdhury (2233359642) | Araf Hussain (2111078642) | Dihan Shahriar (2031839042)  
Section 07 | Group 02 | CSE445
