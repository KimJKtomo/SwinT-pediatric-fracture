# 🦴 SwinT-Pediatric-Fracture: 소아 및 성인 X-ray 골절 진단 AI

본 리포지토리는 소아 및 성인 X-ray 영상에서 골절을 분류하는 Swin Transformer 기반의 딥러닝 모델 프리트레이닝 및 추론 파이프라인을 포함합니다.  
추후 AO classification, 성장판 관련 골절 검출 등 정밀 진단 작업을 위한 백본 모델로 사용됩니다.

---

## 🎯 Project Objective

- Swin Transformer 기반 **Binary Classifier (Fracture vs. Normal)** 구축
- 소아 및 성인 데이터를 활용한 **범용 골절 검출 모델** 사전학습
- 향후:
  - AO subtype classification
  - Growth plate-related fracture detection
  - Clinical deployment (Grad-CAM + GUI)

---

## 🗂 Dataset Overview (2025 기준 최신)

| Dataset            | 대상     | 총 수량   | Fracture (1) | Normal (0) | 비고                                               |
|--------------------|----------|-----------|--------------|------------|----------------------------------------------------|
| Kaggle             | 소아     | 20,328    | 13,551       | 6,777      | AO, Age, Gender 등 풍부한 메타데이터 포함         |
| 병원               | 성인     | 2,649     | 1,343        | 1,306      | 부위별 Cropped 이미지 (radius, scaphoid, styloid) |
| MURA (XR_WRIST)    | 성인     | 9,752     | 3,987        | 5,765      | 골절 여부만 존재 (AO 등 없음)                     |
| **총합**           | 혼합     | **32,729**| **18,881**   | **13,848** | Binary label (`label ∈ [0, 1]`) 기준으로 정제됨    |

✅ *현재 단계에서는 AO 및 growth info는 무시되고 binary label만 사용됩니다.*

---

## ⚙️ Preprocessing & Loading

- **Resize**: 384×384  
- **Normalize**: mean = 0.5, std = 0.5 (Grayscale)  
- **Loader**: `UnifiedFractureDataset` (PyTorch Custom)  

### Split Options

- **Option A**: StratifiedKFold (전체 데이터 기반)
- **Option B (추천)**: Cross-Domain (Train: MURA + Hospital / Val: Kaggle)

---

## 🧠 Model Architecture

```text
Swin Transformer Large (IN22K pretrained)
→ Linear(1536 → 256) → ReLU → Dropout
→ Linear(256 → 1) → BCEWithLogitsLoss
```

> Optimizer: AdamW  
> Scheduler: CosineAnnealingLR  
> Loss: Weighted BCEWithLogitsLoss (fracture class imbalance 반영)

---

## 📊 Performance Snapshot

- ✅ **Validation F1-score**: 0.74 (Cross-domain 기준)
- ✅ Threshold tuning, Youden Index 기반 최적화 적용
- ✅ 각 파트별 성능 분석 및 confusion matrix 포함

---

## 🔍 Visualization (Grad-CAM)

| Grad-CAM 예시 | 설명 |
|---------------|------|
| ![](examples/gradcam_radius.png) | Radius 부위 골절 focus |
| ![](examples/gradcam_scaphoid.png) | Scaphoid 부위 골절 |

> Grad-CAM 결과는 `test/test_single.py` 실행 시 자동 저장됩니다.

---

## 🛠 Inference Demo

### Requirements

```bash
Python >= 3.8  
torch >= 1.12  
timm, opencv-python, matplotlib  
```

### Run Inference (단일 이미지)

```bash
python test/test_single.py \
    --img path/to/image.png \
    --part radius \
    --weight weights/swinT_radius_epoch40.pt
```

- 결과는 `results/`에 예측 이미지 및 Grad-CAM 함께 저장됩니다.  
- AO 분류 모델 또는 Multimodal 모델은 `/20250619_AO/` 디렉터리에 포함 예정입니다.

---

## 🧪 Training (Pretraining Phase)

```bash
python train/train_fx_single.py --config config/train_config.yaml
```

MLflow 및 Ray를 통한 실험 추적 자동화가 포함되어 있습니다.  
(MLOps 환경: Ray, MLflow, Docker 기반 → 재현 가능한 실험 및 결과 버전 관리 지원)

---

## 🧩 Roadmap

- [x] Pretrain fracture classifier (MURA + Kaggle + Hospital)
- [x] Add Grad-CAM visualization
- [x] AO classification with Salter-Harris subtype
- [ ] Growth plate fracture detection
- [ ] Web-based GUI viewer integration
- [ ] Clinical trial deployment

---

## 👤 Contact

- 개발자: 김진규 (Jin-Gyu Kim)  
- GitHub: [KimJKtomo](https://github.com/KimJKtomo)  
- Email: kimgk3793@naver.com

---

## 📄 License

본 리포지토리는 연구 및 비상업적 용도에 한해 자유롭게 활용 가능합니다.
