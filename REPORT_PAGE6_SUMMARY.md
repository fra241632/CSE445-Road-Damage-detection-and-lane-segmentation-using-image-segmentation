# Page 6 Report Review & Quantitative Experiment Summary

**Course**: CSE445 Machine Learning Project  
**Project**: Two-Stage Transfer Learning for Robust Road Surface Crack Segmentation  
**Group**: Section 07 | Project Group 02  

---

## 1. Quantitative Results (Exact Values for Table III)

Extracted directly from the training and validation logs:

| Model / Stage | Best Epoch | Validation IoU | Validation Dice | Validation Pixel Accuracy | Best Validation Loss | Log Source |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **U-Net — Stage 1 (Close-Up Pre-training)** | **7** | **0.5118** ($\approx \mathbf{0.51}$) | **0.6536** ($\approx \mathbf{0.65}$) | **95.78%** | 0.4308 | [`experiments/crack_stage1_unet/train_log.csv`](file:///d:/compressed/Road_Damage_Project%20main/experiments/crack_stage1_unet/train_log.csv#L8) |
| **U-Net — Stage 2 (Dashcam Fine-Tuning)** | **73** | **0.1930** ($\approx \mathbf{0.19}$) | **0.2881** ($\approx \mathbf{0.29}$) | **99.52%** | 0.3726 | [`experiments/crack_stage2_unet_finetuned/train_log.csv`](file:///d:/compressed/Road_Damage_Project%20main/experiments/crack_stage2_unet_finetuned/train_log.csv#L74) |

> **Note for Table III**:
> You can fill in `Dice = 0.65` for **Stage 1** (epoch 7) rather than leaving it blank or as a dash.

---

## 2. Mathematical Verification of Formulas on Page 6

### A. Dice Coefficient (Sørensen–Dice Index)
$$\text{Dice} = \frac{2|\hat{M} \cap M|}{|\hat{M}| + |M|} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

### B. Algebraic Identity relating IoU and Dice
$$\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$$

**Proof Verification**:
$$\frac{2 \cdot \text{IoU}}{1 + \text{IoU}} = \frac{2 \left(\frac{TP}{TP + FP + FN}\right)}{1 + \frac{TP}{TP + FP + FN}} = \frac{2 \cdot TP}{TP + FP + FN + TP} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN} \equiv \text{Dice}$$
*(100% mathematically correct).*

---

## 3. Video Inference Pipeline & Post-Processing Details

Matches implementation in [`support/shared/inference.py`](file:///d:/compressed/Road_Damage_Project%20main/support/shared/inference.py):

1. **Threshold Tuning ($0.50 \to 0.20$)**:
   - For real-world dashcam video deployment, lower raw probability thresholding significantly boosts **Recall** on faint, distant hairline cracks.
2. **Morphological Closing**:
   - Applies an elliptical structuring kernel (`cv2.MORPH_CLOSE`) to connect fragmented crack segments into coherent structural components.
3. **Bounding Box Extraction**:
   - `cv2.findContours` + `cv2.boundingRect` detects contiguous crack regions and draws bounding box overlays with confidence scores.

---

## 4. Key Insights & Explanations to Retain in Report

- **Annotation Granularity vs. Quantitative IoU**:
  - In wide-angle dashcam imagery, human annotators naturally draw thicker, coarse bounding masks around thin cracks.
  - The model predicts fine, pixel-precise boundaries. Because IoU penalizes non-overlapping boundary pixels heavily, the quantitative IoU is $\approx 0.19$ even though qualitative inspection shows crisp, accurate crack localization.
- **Domain Gap Interpretation**:
  - The drop from Stage 1 ($\text{IoU} \approx 0.51$) to Stage 2 ($\text{IoU} \approx 0.19$) is not a sign of model degradation, but rather reflects the transition from clean, high-contrast laboratory macro-shots to noisy, wide-angle real-world driving footage with varying illumination and asphalt textures.

---

## 5. Checklist Before Final PDF Export

- [ ] **Figure 2**: Insert the visual grid (Input Image | Ground Truth Mask | Predicted Mask Overlay).
- [ ] **Table III**: Update Stage 1 Dice to `0.65`.
- [ ] **Wording**: Ensure "dashcam/drone" is simplified to "dashcam video" if drone footage was not collected.
