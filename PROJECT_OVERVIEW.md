# Road Surface Crack Detection using Deep Image Segmentation

**Course**: CSE445 Machine Learning Project  
**Authors**: Md. Sakif Chowdhury (ID: 2233359642), Araf Hussain (ID: 2111078642) & Dihan Shahriar (ID: 2031839042) | Section 07 | Group 02

---

## 1. What This Project Is

This project is an end-to-end, supervised deep-learning system for **road-scene semantic segmentation**. Rather than detecting bounding boxes, semantic segmentation classifies **every individual pixel** in an image.

The system performs:
- **Pavement Crack Detection**: Identifies exact road surface crack boundaries at pixel resolution.
- **Real-World Video Inference**: Processes dashcam footage frame-by-frame and produces annotated output videos with crack overlays and bounding boxes.

### Live Video Demos (YouTube)
- **Demo 1 (Dashcam Road Survey)**: [https://www.youtube.com/watch?v=-uLCMcTAiGo](https://www.youtube.com/watch?v=-uLCMcTAiGo)
- **Demo 2 (High-Resolution Crack Segmentation)**: [https://www.youtube.com/watch?v=SEnlQzeg_Fk](https://www.youtube.com/watch?v=SEnlQzeg_Fk)

A **2-stage transfer learning** strategy is used: the model first pre-trains on high-contrast closeup crack images, then fine-tunes on real-world dashcam footage at a lower learning rate to bridge the domain gap.

---

## 2. Purpose & Motivation

- **Road Infrastructure Maintenance**: Manual road inspection is slow, expensive, and unsafe. Automated pixel-level crack detection enables highway authorities to detect road degradation early before structural failure occurs.
- **Solving Extreme Class Imbalance**: In road imagery, cracks occupy only **2–5%** of the total image area. Standard accuracy metrics fail under such imbalance. This project implements a specialised combined loss function ($\text{BCE} + \text{Dice}$) and region-wise metrics ($\text{IoU}$, $\text{Dice}$) to correctly address this.
- **Domain Gap**: Models trained only on clean macro-shots fail on noisy dashcam footage. The 2-stage fine-tuning scheme closes this gap systematically.

---

## 3. Datasets Used

| Stage | Dataset | Domain | Annotation Format | Train / Val / Test |
|---|---|---|---|---|
| **Stage 1 (Pre-train)** | CRACK500 + DeepCrack | Closeup macro-shots, high contrast | RGB images + grayscale binary masks | 2346 / 502 / 504 |
| **Stage 2 (Fine-tune)** | Dashcam Survey Footage | Wide-angle real-world driving | RGB images + grayscale binary masks | 1982 / 424 / 426 |

All images and masks are resized to **256 × 256 px**. Masks are binarized at threshold 127 (`pixel > 127 → 1.0`). Normalization uses ImageNet statistics: `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`.

---

## 4. How the System Works (End-to-End Pipeline)

```
 Raw Images & Annotations (Closeup + Dashcam)
           │
           ▼
 1. Preprocessing & Manifest Generation
    • Image-mask pairing by filename stem
    • Stratified 70% Train / 15% Val / 15% Test split (seed=42)
    • CSV manifests written to data/crack/splits/
           │
           ▼
 2. Dataset & Augmentation Pipeline
    • SegmentationDataset (PyTorch Dataset class)
    • Albumentations: HorizontalFlip (p=0.5), VerticalFlip (p=0.5),
      RandomRotate90 (p=0.5), Brightness/Contrast (p=0.3), GaussNoise (p=0.2)
    • ImageNet Normalisation + ToTensorV2
           │
           ▼
 3. Model Architecture & Loss Optimisation
    • Primary Model:    U-Net (~7.8M params, trained from scratch)
    • Comparison Model: DeepLabv3 (ResNet-50 backbone, ImageNet pretrained, ~40M params)
    • Loss Function:    BCEDiceLoss = 0.5 × BCE + 0.5 × Dice Loss
           │
           ▼
 4. Two-Stage Transfer Learning
    • Stage 1: Pre-train on closeup crack images (lr=2e-4, up to 40 epochs)
    • Stage 2: Fine-tune on dashcam footage (lr=1e-4, up to 75/50 epochs)
    • Adam Optimiser + ReduceLROnPlateau (patience=4, factor=0.5)
    • Early stopping monitors Val IoU
    • Best checkpoint auto-saved as best_model.pth
           │
           ▼
 5. Quantitative Evaluation & Model Comparison
    • Metrics: IoU, Dice Score, Pixel Accuracy, Precision, Recall
    • Training curves (loss & IoU per epoch)
    • Head-to-head comparison: U-Net vs DeepLabv3
           │
           ▼
 6. Inference & Deployment
    • CrackPredictor engine (images & videos)
    • Post-processing: threshold tuning (0.50 → 0.20), morphological closing,
      bounding box extraction with confidence overlay
    • Output: annotated video/image with 50% opacity red crack mask + yellow boxes
```

---

## 5. Technical Details of Key Components

### A. U-Net Architecture (~7.8M parameters)
- **Encoder**: 4 downsampling stages (DoubleConv + MaxPool). Channel widths: `32 → 64 → 128 → 256`.
- **Bottleneck**: DoubleConv at `16×16` resolution (512 channels).
- **Decoder**: 4 upsampling stages (Bilinear ↑2 + skip-concat + DoubleConv). Restores full resolution using skip connections to recover fine crack boundaries.
- **Output Head**: `1×1 Conv → Sigmoid` → probability map `[0, 1]`.

### B. DeepLabv3 Comparison Model (~42.0M parameters)
- **Backbone**: ResNet-50 (pretrained on ImageNet/COCO).
- **ASPP Module**: 5 parallel atrous branches (rates: 1, 12, 24, 36 + global average pooling) for multi-scale context capture.
- **Output**: Bilinear ×16 upsampling → binary mask.
- **Custom `load_state_dict()`**: Gracefully handles auxiliary classifier weight mismatches during checkpoint loading.

### C. BCEDiceLoss
$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$
- **BCE** provides stable pixel-level gradients, preventing early training collapse.
- **Dice Loss** ($1 - \text{Dice Coefficient}$) directly optimizes region overlap, countering background-pixel dominance.

### D. Evaluation Metrics
1. **IoU (Jaccard Index)**: Primary metric — $\frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$
2. **Dice Coefficient (F1)**: $\frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$
3. **Pixel Accuracy**: Fraction of correctly classified pixels.
4. **Precision**: True cracks / all predicted cracks.
5. **Recall**: True cracks detected / all actual cracks.

### E. Quantitative Results

#### Validation Performance (Best Epochs)
| Model & Stage | Best Epoch | Val IoU | Val Dice | Val Pixel Acc |
|---|:---:|:---:|:---:|:---:|
| U-Net — Stage 1 (Closeup pre-train) | **7** | **0.5118** | **0.6536** | **95.78%** |
| U-Net — Stage 2 (Dashcam fine-tune) | **73** | **0.1930** | **0.2881** | **99.52%** |
| DeepLabv3 — Stage 1 (Closeup pre-train) | **8** | **0.5801** | **0.7136** | **97.09%** |
| DeepLabv3 — Stage 2 (Dashcam fine-tune) | **48** | **0.1335** | **0.2124** | **99.36%** |

#### Held-Out Test Set Performance (Stage 2 Dashcam)
| Model | Test IoU | Test Dice | Test Precision | Test Recall | Test Pixel Acc | Parameters |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Two-Stage U-Net** | **0.2034** | **0.3009** | **0.3316** | **0.3644** | **99.51%** | **~7.8M** |
| Two-Stage DeepLabv3 | 0.1393 | 0.2219 | 0.2233 | 0.3018 | 99.35% | ~42.0M |

> **Note on Stage 2 IoU drop**: Human annotators draw coarse, thick masks around thin cracks in dashcam imagery. The model predicts fine, pixel-precise boundaries. IoU heavily penalizes this boundary misalignment, so the quantitative score drops even though qualitative detection is accurate.

---

## 6. Project Notebook Workflow

Execute the notebooks in `notebooks/` in order:

1. `01_EDA_crack.ipynb` — Exploratory Data Analysis & augmentation preview for crack detection.
2. `02_train_crack_unet.ipynb` — Trains U-Net (2-stage) and plots learning curves (train vs. val loss & IoU).
3. `03_evaluate_crack.ipynb` — Held-out test set evaluation for crack U-Net (IoU, Dice, Precision, Recall, error maps, failure analysis).
4. `04_compare_crack_deeplabv3.ipynb` — Trains DeepLabv3 (2-stage) and plots a head-to-head metric comparison against U-Net.
5. `05_inference_crack.ipynb` — Interactive sandbox to run crack detection on custom images and dashcam videos.

---

## 7. Project Folder Structure

```
Road_Damage_Project/
├── config.py                           ← All paths & hyperparameters (U-Net + DeepLabv3)
├── main.py                             ← CLI entry point (--setup, --infer, --jupyter)
├── requirements.txt
├── README.md                           ← Main project documentation
├── PROJECT_OVERVIEW.md                 ← In-depth technical specification
├── assets/
│   └── demo.gif                        ← Demo GIF embedded in GitHub README
├── data/
│   ├── README.md                       ← Dataset download instructions & layout
│   └── crack/
│       ├── closeup/                    ← Stage 1: CRACK500 + DeepCrack images & masks
│       ├── dashcam/                    ← Stage 2: real-world dashcam frames & masks
│       └── splits/                     ← CSV manifests (stage1_*.csv, dashcam_*.csv)
├── others/
│   ├── final_presentation.pptx         ← Final project presentation
│   ├── final_report.pdf                ← Final IEEE-format research report
│   ├── update_presentation.pptx        ← Midterm update presentation
│   ├── update_report.pdf               ← Midterm progress report
│   └── demo_video.mp4                  ← 1-minute system demo video
├── support/
│   ├── crack/
│   │   ├── preprocess.py               ← Pairs images/masks, generates split CSVs
│   │   ├── train_2stage.py             ← Runs Stage 1 → Stage 2 training (U-Net)
│   │   └── download_data.py            ← Dataset download helpers
│   └── shared/
│       ├── dataset.py                  ← SegmentationDataset (reads CSV manifests)
│       ├── transforms.py               ← Albumentations train/val pipelines
│       ├── unet.py                     ← U-Net model definition (~7.8M params)
│       ├── deeplabv3.py                ← DeepLabv3 wrapper (~42.0M params, ResNet-50)
│       ├── losses.py                   ← BCEDiceLoss, FocalLoss
│       ├── metrics.py                  ← IoU, Dice, Pixel Accuracy, Precision, Recall
│       ├── trainer.py                  ← Trainer: early stopping, LR scheduling, CSV logs
│       └── inference.py                ← CrackPredictor: image & video inference engine
├── notebooks/
│   ├── 01_EDA_crack.ipynb
│   ├── 02_train_crack_unet.ipynb
│   ├── 03_evaluate_crack.ipynb
│   ├── 04_compare_crack_deeplabv3.ipynb
│   └── 05_inference_crack.ipynb
└── experiments/
    ├── crack_stage1_unet/              ← U-Net Stage 1 checkpoint & log
    ├── crack_stage2_unet_finetuned/    ← U-Net Stage 2 checkpoint & log
    ├── crack_stage1_deeplabv3/         ← DeepLabv3 Stage 1 checkpoint & log
    ├── crack_stage2_deeplabv3_finetuned/ ← DeepLabv3 Stage 2 checkpoint & log
    └── *.png                           ← Generated diagrams and plot figures
```

---

## 8. How to Run Locally or on Google Colab

### Local Execution
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess data (generate split CSVs)
python support/crack/preprocess.py

# 3. Train U-Net (2-stage)
python support/crack/train_2stage.py

# 4. Run inference on an image
python main.py --infer path/to/image.jpg

# 5. Run inference on a video
python main.py --infer path/to/video.mp4 --output result.mp4

# 6. Open notebooks
python main.py --jupyter
```

### Google Colab Execution
1. Upload the project folder to Google Drive.
2. Open any notebook in Google Colab.
3. Set `RUN_ENV=colab` (done automatically at the top of each notebook).
4. Runs seamlessly on T4 GPU acceleration.
