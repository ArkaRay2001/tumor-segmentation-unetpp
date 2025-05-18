# 🧠 Brain Tumor Segmentation with UNet++ and CBAM

A PyTorch-based medical imaging project using a custom UNet++ model enhanced with CBAM (Convolutional Block Attention Module) and deep supervision. Designed for precise segmentation of brain tumor sub-regions: **enhancing tumor (ET)**, **edema (ED)**, and **necrotic core (NCR)**.

---

## 🚀 Key Features

- **UNet++ Architecture**: Nested skip connections for better semantic learning
- **Attention Module**: CBAM (Channel + Spatial) to highlight important features
- **Deep Supervision**: Multi-scale outputs for improved gradient flow and convergence
- **Hybrid Loss Function**: Combines Focal Loss, Dice Loss, and BCE to handle class imbalance
- **Medical-Specific Augmentations**: Rician noise, bias field simulation, intensity perturbations
- **Optimized Training**: Mixed-precision (AMP), gradient clipping, and OneCycleLR scheduler

---

## 📊 Performance Summary

| Metric       | Single Model | Checkpoint Ensemble |
|--------------|--------------|---------------------|
| Dice Score   | 0.82         | 0.84                |
| IoU          | 0.72         | 0.74                |

*(Based on validation set; update with your actual metrics from `logs/`)*

--
##🧪 Technical Highlights
✅ Medical Augmentation Pipeline
python
Copy
Edit
aug = BrainScanAugmentation(p=0.5)
image, mask = aug(image, mask)  # Applies Rician noise, bias field simulation, etc.
✅ CBAM: Channel + Spatial Attention
Improves model focus on relevant anatomical structures through dual attention.

✅ Deep Supervision
Enables gradient signals at multiple levels for faster, more stable convergence.
🛠 Requirements
1)Python 3.8+
2)PyTorch 1.12+
3)OpenCV
4)MONAI
5)h5py




