# gradcam_visualize_batch_v1.py - Swin Transformer + Grad-CAM for all images in folder
import os
import torch
import cv2
import glob
import numpy as np
from torchvision import transforms
from SwinT_ImageOnly_Classifier import SwinTImageClassifier
from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from cam import reshape_transform, heatmap_filter

def load_model(weights_path, device):
    class Args:
        model_name = "swin_large_patch4_window12_384_in22k"
        pretrained = False
        num_classes = 1
    model = SwinTImageClassifier(Args()).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def get_target_layers(model):
    target_layers = []
    for name, module in model.backbone.named_modules():
        if 'norm' in name and 'layers.2' in name and 'blocks' in name:
            target_layers.append(module)
    return target_layers

def preprocess_image(img_path):
    image = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (384, 384))
    input_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])(rgb_img)
    return image, rgb_img / 255.0, input_tensor.unsqueeze(0)

def process_folder():
    img_dir = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/test_images"
    weights = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250605_new/swinT_pretrained_fx_masked_focal_best.pt"
    output_dir = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/cam_outputs"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, device)
    target_layers = get_target_layers(model)

    cam = HiResCAM(model=model, target_layers=target_layers,
                   use_cuda=device.type == 'cuda',
                   reshape_transform=reshape_transform)

    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg"))

    for img_path in image_paths:
        try:
            original_image, rgb_img, input_tensor = preprocess_image(img_path)
            input_tensor = input_tensor.to(device)
            grayscale_cam = cam(input_tensor=input_tensor)[0]

            _, heatmap, cam_result = heatmap_filter(grayscale_cam, original_image)
            base = os.path.splitext(os.path.basename(img_path))[0]

            cv2.imwrite(os.path.join(output_dir, f"{base}_input.jpg"), original_image)
            # cv2.imwrite(os.path.join(output_dir, f"{base}_heatmap.jpg"), heatmap)
            cv2.imwrite(os.path.join(output_dir, f"{base}_cam.jpg"), cam_result)
            print(f"✅ Saved: {base}_cam.jpg")
        except Exception as e:
            print(f"[❌] Error on {img_path}: {e}")

if __name__ == "__main__":
    process_folder()

