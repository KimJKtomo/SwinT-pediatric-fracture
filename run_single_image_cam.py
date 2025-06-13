# run_single_image_cam.py

import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from SwinT_ImageOnly_Classifier import SwinTImageClassifier
from gradcam_visualize_batch_v1 import get_target_layers, reshape_transform, heatmap_filter
from pytorch_grad_cam import HiResCAM

# === 설정 ===
image_path = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/MCF20000.jpg"  # 예: "/mnt/data/sample_fracture.png"
model_path = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250605_new/swinT_pretrained_fx_masked_focal_best.pt"
threshold = 0.4
output_cam_path = "MCF20000_cam_output.jpg"

# === 모델 로딩 ===
class Args:
    model_name = "swin_large_patch4_window12_384_in22k"
    pretrained = False
    num_classes = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinTImageClassifier(Args()).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Grad-CAM 설정 ===
target_layers = get_target_layers(model)
cam = HiResCAM(model=model, target_layers=target_layers,
               use_cuda=(device.type == 'cuda'),
               reshape_transform=reshape_transform)

# === 이미지 전처리 ===
def preprocess_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(original, (384, 384))
    input_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])(resized)
    return image, resized / 255.0, input_tensor.unsqueeze(0)

# === Inference + Grad-CAM ===
try:
    original, rgb_norm, input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob > threshold else 0

    print(f"🧠 Fracture Prediction: {pred} (probability: {prob:.4f})")

    # Grad-CAM 생성 및 저장
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    _, _, cam_result = heatmap_filter(grayscale_cam, original)
    cv2.imwrite(output_cam_path, cam_result)
    print(f"✅ Grad-CAM saved to: {output_cam_path}")

except Exception as e:
    print(f"❌ Error: {e}")
