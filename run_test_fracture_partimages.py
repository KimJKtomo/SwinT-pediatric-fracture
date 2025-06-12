import os
import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from torchvision import transforms
from SwinT_ImageOnly_Classifier import SwinTImageClassifier
from gradcam_visualize_batch_v1 import get_target_layers, reshape_transform, heatmap_filter
from pytorch_grad_cam import HiResCAM

# === Config ===
excel_path = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/엑셀_손목골절연구(전체결과값)v3_0714.xlsx"
img_dir = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/images"
model_path = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250605_new/swinT_pretrained_fx_masked_focal_best.pt"
output_dir = "fracture_eval_cam_outputs"
os.makedirs(output_dir, exist_ok=True)
threshold = 0.4

# === Load Excel ===
df = pd.read_excel(excel_path, skiprows=2)
df = df[['filename', 'radius', 'styloid', 'scaphoid']].dropna()
df['fracture_label'] = df[['radius', 'styloid', 'scaphoid']].max(axis=1).astype(int)

# === Load Model ===
class Args:
    model_name = "swin_large_patch4_window12_384_in22k"
    pretrained = False
    num_classes = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinTImageClassifier(Args()).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Grad-CAM Setup ===
target_layers = get_target_layers(model)
cam = HiResCAM(model=model, target_layers=target_layers,
               use_cuda=(device.type == 'cuda'),
               reshape_transform=reshape_transform)

# === Image Preprocessing ===
def preprocess_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (384, 384))
    input_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])(rgb)
    return image, rgb / 255.0, input_tensor.unsqueeze(0)

# === Inference + CAM
y_true, y_pred, filenames, probs = [], [], [], []
missing_count = 0

for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
    fname = os.path.splitext(row['filename'])[0] + ".jpg"
    img_path = os.path.join(img_dir, fname)
    if not os.path.exists(img_path):
        missing_count += 1
        continue

    try:
        original, rgb_norm, input_tensor = preprocess_image(img_path)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > threshold else 0
    except Exception as e:
        print(f"[⚠️] Failed on {fname}: {e}")
        continue

    label = int(row['fracture_label'])
    y_true.append(label)
    y_pred.append(pred)
    probs.append(prob)
    filenames.append(fname)

    # === Grad-CAM 저장
    try:
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        _, _, cam_result = heatmap_filter(grayscale_cam, original)
        cam_name = f"{fname}_gt{label}_pred{pred}_prob{prob:.2f}.jpg"
        cv2.imwrite(os.path.join(output_dir, cam_name), cam_result)
    except Exception as e:
        print(f"[❌] Grad-CAM failed for {fname}: {e}")

# === 결과 출력 및 저장
print(f"\n✅ Processed: {len(y_true)} samples | Missing images: {missing_count}")

if len(y_true) == 0:
    print("❌ No valid samples for evaluation.")
else:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    print("\n[Classification Report]")
    print(report)
    print("[Confusion Matrix]")
    print(cm)
    print(f"Accuracy: {acc:.4f} | F1-score: {f1:.4f}")

    result_df = pd.DataFrame({
        "filename": filenames,
        "true_label": y_true,
        "pred_label": y_pred,
        "probability": probs
    })
    result_df.to_csv("fracture_eval_result_from_excel.csv", index=False)
    print("✅ Saved: fracture_eval_result_from_excel.csv")
