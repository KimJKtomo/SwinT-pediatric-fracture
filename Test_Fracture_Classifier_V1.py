import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ✅ gradcam utils import
import sys
sys.path.append("/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250605_new")
from gradcam_visualize_batch_v1 import get_target_layers, reshape_transform, heatmap_filter
from pytorch_grad_cam import HiResCAM
from SwinT_ImageOnly_Classifier import SwinTImageClassifier

# ✅ Config
excel_path = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/엑셀_손목골절연구(전체결과값)v3_0714.xlsx"
img_dir = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/images"
weights_path = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250605_new/swinT_pretrained_fx_masked_focal_best.pt"
output_csv = "fracture_eval_result_from_excel.csv"
output_cam_dir = "gradcam_fx_images"
os.makedirs(output_cam_dir, exist_ok=True)
threshold = 0.4

# ✅ Load Excel
df = pd.read_excel(excel_path, skiprows=2)
df = df[['filename', 'radius', 'styloid', 'scaphoid']].dropna()
df['fracture_label'] = df[['radius', 'styloid', 'scaphoid']].max(axis=1).astype(int)

# ✅ Load Model
class Args:
    model_name = "swin_large_patch4_window12_384_in22k"
    pretrained = False
    num_classes = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinTImageClassifier(Args()).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# ✅ Grad-CAM 설정
target_layers = get_target_layers(model)
cam = HiResCAM(model=model, target_layers=target_layers,
               use_cuda=(device.type == 'cuda'),
               reshape_transform=reshape_transform)

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ✅ Inference + Grad-CAM
results = []
y_true, y_pred = [], []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
    fname = os.path.splitext(row['filename'])[0] + ".jpg"
    img_path = os.path.join(img_dir, fname)
    true_label = int(row['fracture_label'])

    if not os.path.exists(img_path):
        print(f"[⚠️] Image not found: {img_path}")
        continue

    try:
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            pred = int(prob > threshold)
    except Exception as e:
        print(f"[❌] Inference failed: {img_path} | {e}")
        continue

    y_true.append(true_label)
    y_pred.append(pred)

    results.append({
        "filename": fname,
        "true_label": true_label,
        "pred_label": pred,
        "probability": prob
    })

    # Grad-CAM 저장
    try:
        original = cv2.imread(img_path)
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        _, _, cam_result = heatmap_filter(grayscale_cam, original)
        cam_name = f"{fname}_gt{true_label}_pred{pred}_prob{prob:.2f}.jpg"
        cv2.imwrite(os.path.join(output_cam_dir, cam_name), cam_result)
    except Exception as e:
        print(f"[❌] Grad-CAM failed: {img_path} | {e}")

# ✅ Save & Report
result_df = pd.DataFrame(results)
result_df.to_csv(output_csv, index=False)
print(f"\n✅ Saved CSV: {output_csv}")

print("\n[Classification Report]")
print(classification_report(y_true, y_pred, digits=4, zero_division=0))
print("[Confusion Matrix]")
print(confusion_matrix(y_true, y_pred))
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f} | F1-score: {f1_score(y_true, y_pred):.4f}")
