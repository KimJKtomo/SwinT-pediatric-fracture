import os
import torch
import pandas as pd
import cv2
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from unified_dataset_0605 import UnifiedFractureDataset
from SwinT_ImageOnly_Classifier import SwinTImageClassifier
from gradcam_visualize_batch_v1 import get_target_layers, reshape_transform, heatmap_filter
from pytorch_grad_cam import HiResCAM

# ✅ 1. Load Kaggle dataset (only 'open' source)
from load_new_dxmodule_0605_v2 import get_combined_dataset_v2
df = get_combined_dataset_v2(image_only=True, fracture_only=True)
kaggle_df = df[df['source'] == 'open'].copy()

# ✅ 2. Check and extract uncertain cases
if 'diagnosis_uncertain' not in kaggle_df.columns:
    raise ValueError("❌ 'diagnosis_uncertain' column not found in Kaggle dataset.")

test_df = kaggle_df[kaggle_df['diagnosis_uncertain'] == 1.0].reset_index(drop=True)

# ✅ 3. Transforms and dataset
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
test_dataset = UnifiedFractureDataset(test_df, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ✅ 4. Load pretrained SwinT model
class Args:
    model_name = "swin_large_patch4_window12_384_in22k"
    pretrained = False
    num_classes = 1

weights_path = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250605_new/swinT_pretrained_fx_sampler_focal_best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinTImageClassifier(Args()).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# ✅ 5. Set Grad-CAM
target_layers = get_target_layers(model)
cam = HiResCAM(model=model, target_layers=target_layers, use_cuda=device.type == 'cuda', reshape_transform=reshape_transform)
os.makedirs("uncertain_df_cam_outputs", exist_ok=True)

# ✅ 6. Inference loop with Grad-CAM
results = []
for i, (image, _) in enumerate(tqdm(test_loader, desc="Uncertain Inference")):
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()
        pred = int(prob > 0.5)

    img_path = test_df.loc[i, "image_path"]
    filename = os.path.basename(img_path)
    results.append({
        "filename": filename,
        "image_path": img_path,
        "ai_probability": prob,
        "ai_pred_label": pred
    })

    # Grad-CAM 저장
    try:
        original = cv2.imread(img_path)
        rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (384, 384)) / 255.0
        grayscale_cam = cam(input_tensor=image)[0]
        _, _, cam_result = heatmap_filter(grayscale_cam, original)
        cam_name = f"{i:04d}_{filename}_prob{prob:.2f}_pred{pred}.jpg"
        cv2.imwrite(os.path.join("uncertain_df_cam_outputs", cam_name), cam_result)
    except Exception as e:
        print(f"[⚠️] Grad-CAM failed for {img_path}: {e}")

# ✅ 7. Save results
result_df = pd.DataFrame(results)
result_df.to_csv("uncertain_df_ai_predictions.csv", index=False)
print("✅ Saved: uncertain_df_ai_predictions.csv")
