# run_test_fracture_classifier_final.py

import os
import glob
import torch
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from torchvision import transforms
from SwinT_ImageOnly_Classifier import SwinTImageClassifier

def load_model(weights_path, device):
    class Args:
        model_name = "swin_large_patch4_window12_384_in22k"
        pretrained = False
        num_classes = 1
    model = SwinTImageClassifier(Args()).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def preprocess_image(img_path):
    image = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (384, 384))
    input_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])(rgb_img)
    return input_tensor.unsqueeze(0), image

def run_test(img_dir, model, device, threshold=0.40):
    image_paths = glob.glob(os.path.join(img_dir, '*.png')) + glob.glob(os.path.join(img_dir, '*.jpg'))
    y_true, y_pred, y_prob, filenames = [], [], [], []

    for img_path in tqdm(image_paths, desc="Testing"):
        # 라벨 추론 방식: 파일명에 'posi' → 1, 'nega' → 0
        fname_lower = os.path.basename(img_path).lower()
        if "posi" in fname_lower:
            label = 1
        elif "nega" in fname_lower:
            label = 0
        else:
            continue  # 예측 불가능 파일

        input_tensor, _ = preprocess_image(img_path)
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).cpu().item()
            pred = 1 if prob > threshold else 0

        y_true.append(label)
        y_pred.append(pred)
        y_prob.append(prob)
        filenames.append(os.path.basename(img_path))

    return y_true, y_pred, y_prob, filenames

def save_results(y_true, y_pred, y_prob, filenames, output_path):
    df = pd.DataFrame({
        "filename": filenames,
        "true_label": y_true,
        "pred_label": y_pred,
        "prob": y_prob
    })
    df.to_csv(output_path, index=False)
    return df

def print_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, digits=4))
    print("[Confusion Matrix]")
    print(cm)
    print(f"Accuracy: {acc:.4f} | F1-score: {f1:.4f}")

if __name__ == '__main__':
    # 🔍 경로 설정
    img_dir = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/img_png"
    weights_path = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250605_new/swinT_pretrained_fx_masked_focal_best.pt"
    output_csv = "test_results_masked.csv"
    threshold = 0.40

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights_path, device)
    y_true, y_pred, y_prob, filenames = run_test(img_dir, model, device, threshold=threshold)
    df = save_results(y_true, y_pred, y_prob, filenames, output_csv)
    print_metrics(y_true, y_pred)
