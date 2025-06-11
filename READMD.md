# 🦴 Pediatric & Adult Fracture Detection – Swin Transformer Pretraining

This repository contains the **pretraining phase** of a deep learning pipeline for detecting bone fractures in both pediatric and adult X-ray images. The pretrained model will be used as the **backbone** for further fine-tuning tasks such as AO classification and growth plate fracture detection.

---

## 🎯 Project Objective

- Build an image-only binary classifier to detect fractures (fracture vs. normal) using Swin Transformer.
- Use a diverse dataset including adult and pediatric data to pretrain a generalized fracture detection model.
- Later, fine-tune this model on pediatric-only data to classify **AO subtypes** and detect **growth plate-related fractures**.

---

## 🗂 Dataset Overview

| Dataset  | Target   | Count   | Fracture (1) | Normal (0) | Notes                            |
|----------|----------|---------|--------------|------------|----------------------------------|
| Kaggle   | Pediatric| 15,351  | 11,559       | 3,792      | Includes AO, Age, Sex, etc.     |
| Hospital | Adult    | 2,649   | 1,343        | 1,306      | Cropped images (radius/styloid/scaphoid) |
| MURA     | Adult    | 7,463   | 3,703        | 3,760      | Only fracture label available    |
| **Total**| Mixed    | **25,463**| **16,605**   | **8,858**  | Filtered with image + binary label only |

✅ Labels are filtered to include only `label ∈ [0, 1]`. AO and growth information are ignored in this pretraining phase.

---

## ⚙️ Preprocessing & Loading Strategy

- Resize: `384×384`
- Normalize: `mean=0.5, std=0.5` (for grayscale images)
- DataLoader: PyTorch `DataLoader` with custom `UnifiedFractureDataset`
- Split Strategy:
  - **Option A:** StratifiedKFold across full combined dataset
  - **Option B (Recommended):** Cross-Domain split (Train: MURA+Hospital, Val: Kaggle)

---

## 🧠 Model Architecture

```python
Swin Transformer Large (IN22K pretrained)
→ Linear(1536 → 256) → ReLU → Dropout
→ Linear(256 → 1) → BCEWithLogitsLoss
```


