# Road Surface Crack Detection using Deep Image Segmentation

**Course**: CSE445 Machine Learning Project  
**Author**: Md. Sakif Chowdhury (ID: 2233359642) | Section 07 | Group 02  

---

## 1. What This Project Is

This project is an end-to-end, supervised deep-learning system for **road-scene semantic segmentation**. Rather than just detecting bounding boxes around objects, semantic segmentation classifies **every individual pixel** in an image.

The system performs:
- **Pavement Crack Detection**: Identifies exact road surface crack boundaries at pixel resolution.

---

## 2. Purpose & Motivation

- **Road Infrastructure Maintenance**: Manual road inspection is slow, expensive, and unsafe. Automated pixel-level crack detection allows highway authorities to detect road degradation early before structural failure occurs.
- **Solving Extreme Class Imbalance**: In road imagery, cracks occupy only **2% to 5%** of total image area. Standard accuracy metrics fail under such imbalance. This project implements specialised loss functions ($\text{BCE} + \text{Dice}$) and region-wise metrics ($\text{IoU}$, $\text{Dice}$) to solve this problem.

---

## 3. Datasets Used

| Task | Primary Dataset | Supplementary Data | Annotation Format | Preprocessing Output |
|---|---|---|---|---|
| **Crack Detection** | **CRACK500** | **DeepCrack** | RGB images + grayscale binary masks | 70/15/15 split CSV manifests |

---

## 4. How the System Works (End-to-End Pipeline)

```
 Raw Images & Annotations
          │
          ▼
 1. Preprocessing & Manifest Generation
    • Pairing images and masks
    • Stem matching & validation
    • 70% Train / 15% Val / 15% Test deterministic split
          │
          ▼
 2. Dataset & Augmentation Pipeline
    • SegmentationDataset (PyTorch)
    • Albumentations: Horizontal Flip, Rotate 90°, Brightness/Contrast, Gauss Noise
    • ImageNet Normalisation
          │
          ▼
 3. Model Architecture & Loss Optimization
    • Primary Model: U-Net (~7.8M params)
    • Comparison Model: DeepLabv3 (ResNet50 backbone)
    • Loss Function: BCEDiceLoss (0.5 BCE + 0.5 Dice)
          │
          ▼
 4. Training Engine (Trainer)
    • Adam Optimiser (lr=1e-4)
    • ReduceLROnPlateau scheduler (monitors Val IoU)
    • Early Stopping (patience = 10 epochs)
    • Automatic checkpointing (best_model.pth)
          │
          ▼
 5. Quantitative Evaluation & Model Comparison
    • Metrics: IoU, Dice Score, Pixel Accuracy, Precision, Recall
    • Error Maps & Visual Overlays
    • Head-to-Head Comparison: U-Net vs DeepLabv3
```

---

## 5. Technical Details of Key Components

### A. U-Net Architecture
- **Encoder**: 4 downsampling stages (Double Convolution + Max Pooling). Extracts hierarchical visual features from low-level edges to high-level crack shapes.
- **Bottleneck**: Deepest feature representation at 1/16th spatial resolution (512 channels).
- **Decoder**: 4 upsampling stages (Bilinear Upsampling + Skip-connection concatenation + Double Convolution). Restores full resolution while recovering fine spatial detail using skip connections.
- **Output Head**: $1 \times 1$ Convolution with Sigmoid activation outputting probability map $[0, 1]$.

### B. DeepLabv3 (Comparison Model)
- Uses a **ResNet-50** backbone with **Atrous (Dilated) Spatial Pyramid Pooling (ASPP)**.
- Captures multi-scale contextual information without losing spatial resolution, serving as a benchmark comparison against U-Net for crack detection.

### C. BCEDiceLoss
$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$
- **Binary Cross-Entropy (BCE)** provides smooth pixel-level gradients.
- **Dice Loss** ($1 - \text{Dice}$) directly optimizes region overlap, preventing the model from predicting all-background.

### D. Evaluation Metrics
1. **Intersection over Union (IoU / Jaccard Index)**: Primary metric for segmentation ($\frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$).
2. **Dice Coefficient (F1 Score)**: Harmonic mean of precision and recall ($\frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$).
3. **Pixel Accuracy**: Percentage of correctly classified pixels.
4. **Precision**: Fraction of predicted positive pixels that are true cracks.
5. **Recall**: Fraction of actual crack pixels correctly detected.

---

## 6. Project Notebook Workflow

To run the complete project, execute the notebooks in `notebooks/` in order:

1. `01_EDA_crack.ipynb`: Exploratory Data Analysis & visual augmentation preview for crack detection.
2. `02_train_crack_unet.ipynb`: Trains U-Net on crack detection & plots learning curves (train vs. val loss & IoU).
3. `03_evaluate_crack.ipynb`: Held-out test set evaluation for crack U-Net (IoU, Dice, Precision, Recall, error maps).
4. `04_compare_crack_deeplabv3.ipynb`: Trains DeepLabv3 and plots a head-to-head metric comparison against U-Net.

---

## 7. How to Run Locally or on Google Colab

### Local Execution:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Organize & preprocess data
python main.py --setup

# 3. Open Jupyter Notebook
python main.py --jupyter
```

### Google Colab Execution:
1. Upload the project folder to Google Drive.
2. Open any notebook in Google Colab.
3. The environment auto-mounts Drive (`RUN_ENV=colab`) and runs seamlessly on T4 GPU acceleration.

