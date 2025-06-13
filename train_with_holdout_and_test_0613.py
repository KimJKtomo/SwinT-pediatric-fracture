import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import mlflow
from torch.optim.lr_scheduler import CosineAnnealingLR

from load_new_dxmodule_0605_v3 import get_combined_dataset_v3
from SwinT_ImageOnly_Classifier import SwinTImageClassifier
from unified_dataset_0605 import UnifiedFractureDataset

from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from cam import reshape_transform, heatmap_filter

# =========================
# Configuration
# =========================
class Args:
    model_name = "swin_large_patch4_window12_384_in22k"
    pretrained = True
    num_classes = 1
    batch_size = 8
    lr = 5e-5
    num_epochs = 100
    num_workers = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    threshold_sweep = [0.4, 0.45, 0.48, 0.5]
    patience = 20

args = Args()

# =========================
# Focal Loss
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# =========================
# Side Marker Masking
# =========================
def mask_side_markers(image):
    if isinstance(image, torch.Tensor):
        _, h, w = image.shape
        image[:, :, :int(w * 0.15)] = 0
        image[:, :, int(w * 0.85):] = 0
    return image

class MaskedFractureDataset(UnifiedFractureDataset):
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        image = mask_side_markers(image)
        return image, label

# =========================
# Test Set Holdout
# =========================
def extract_holdout_test_set(df):
    test_parts = []
    for source, (n_pos, n_neg) in {
        'open': (30, 30),
        'clinical': (15, 15),
        'mura': (15, 15)
    }.items():
        df_source = df[df['source'] == source]
        pos = df_source[df_source['label'] == 1].sample(n=n_pos, random_state=42)
        neg = df_source[df_source['label'] == 0].sample(n=n_neg, random_state=42)
        test_parts.append(pd.concat([pos, neg], ignore_index=True))
    test_df = pd.concat(test_parts, ignore_index=True)
    trainval_df = df[~df['image_path'].isin(test_df['image_path'])].reset_index(drop=True)
    return trainval_df, test_df

# =========================
# Training
# =========================
def train(trainval_df):
    mlflow.start_run(run_name="train_with_holdout_verbose")
    mlflow.log_params(vars(args))

    train_df = trainval_df[trainval_df['split'] == 'train'].reset_index(drop=True)
    val_df = trainval_df[trainval_df['split'] == 'val'].reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
        transforms.RandomErasing(p=0.3)
    ])

    train_labels = train_df['label'].tolist()
    class_weights = 1.0 / torch.tensor([train_labels.count(0), train_labels.count(1)]).float()
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(MaskedFractureDataset(train_df, transform), batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(MaskedFractureDataset(val_df, transform), batch_size=args.batch_size, shuffle=False)

    model = SwinTImageClassifier(args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    criterion = FocalLoss()

    best_f1 = 0
    patience_counter = 0

    for epoch in range(args.num_epochs):
        model.train()
        total_loss, all_preds, all_targets = 0, [], []

        for x, y in tqdm(train_loader, desc=f"[Epoch {epoch}] Train"):
            x, y = x.to(args.device), y.float().to(args.device)
            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds += torch.sigmoid(out).detach().cpu().numpy().flatten().tolist()
            all_targets += y.cpu().numpy().flatten().tolist()

        scheduler.step()
        train_bin = [int(p > 0.5) for p in all_preds]
        train_f1 = f1_score(all_targets, train_bin)
        train_acc = accuracy_score(all_targets, train_bin)
        train_cm = confusion_matrix(all_targets, train_bin)

        print(f"\n[Epoch {epoch}] Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"[Epoch {epoch}] Train Accuracy: {train_acc:.4f} | F1-score: {train_f1:.4f}")
        print(f"[Epoch {epoch}] Train Confusion Matrix:\n{train_cm}")

        mlflow.log_metric("train_loss", total_loss / len(train_loader), step=epoch)
        mlflow.log_metric("train_f1", train_f1, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)

        # Validation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"[Epoch {epoch}] Val"):
                x, y = x.to(args.device), y.float().to(args.device)
                out = model(x)
                val_preds += torch.sigmoid(out).cpu().numpy().flatten().tolist()
                val_trues += y.cpu().numpy().flatten().tolist()

        for t in args.threshold_sweep:
            val_bin = [int(p > t) for p in val_preds]
            val_f1 = f1_score(val_trues, val_bin)
            val_acc = accuracy_score(val_trues, val_bin)
            val_cm = confusion_matrix(val_trues, val_bin)

            print(f"[Epoch {epoch}] Val @ threshold {t:.2f} → Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"[Epoch {epoch}] Val Confusion Matrix:\n{val_cm}")

            mlflow.log_metric(f"val_f1_t{str(t).replace('.', '_')}", val_f1, step=epoch)
            mlflow.log_metric(f"val_acc_t{str(t).replace('.', '_')}", val_acc, step=epoch)

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), "swinT_best_holdout.pt")
                mlflow.pytorch.log_model(model, "best_model")
                print(f"✅ Saved best model @ threshold {t}")
            else:
                patience_counter += 1

        if patience_counter > args.patience:
            print("⏹️ Early Stopping")
            break

    mlflow.end_run()

# =========================
# Final Test + Grad-CAM
# =========================
def final_test_and_gradcam(test_df):
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = UnifiedFractureDataset(test_df, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = SwinTImageClassifier(args).to(args.device)
    model.load_state_dict(torch.load("swinT_best_holdout.pt", map_location=args.device))
    model.eval()

    y_true, y_pred, y_prob, filenames = [], [], [], []
    cam = HiResCAM(model=model, target_layers=[model.backbone.norm],
                   reshape_transform=reshape_transform, use_cuda=args.device == 'cuda')
    os.makedirs("cam_results", exist_ok=True)

    for i, (x, y) in enumerate(dataloader):
        x = x.to(args.device)
        with torch.no_grad():
            out = model(x)
            prob = torch.sigmoid(out).item()
            pred = int(prob > 0.5)

        y_true.append(int(y.item()))
        y_pred.append(pred)
        y_prob.append(prob)
        filenames.append(os.path.basename(test_df.iloc[i]['image_path']))

        grayscale_cam = cam(input_tensor=x)[0]
        orig = x[0].cpu().permute(1, 2, 0).numpy()
        orig = (orig * 0.5 + 0.5)
        cam_img = show_cam_on_image(orig, grayscale_cam, use_rgb=True)
        cv2.imwrite(f"cam_results/{i:03d}_label{int(y.item())}_pred{pred}.jpg", cam_img)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n[Final Evaluation]")
    print(classification_report(y_true, y_pred, digits=4))
    print(cm)
    mlflow.log_metric("test_acc", acc)
    mlflow.log_metric("test_f1", f1)

    df_result = pd.DataFrame({
        'filename': filenames,
        'true_label': y_true,
        'pred_label': y_pred,
        'probability': y_prob
    })
    df_result.to_csv("final_test_results.csv", index=False)
    mlflow.log_artifact("final_test_results.csv")

    for fname in os.listdir("cam_results"):
        mlflow.log_artifact(os.path.join("cam_results", fname), artifact_path="gradcam")

# =========================
# Main
# =========================
if __name__ == "__main__":
    full_df = get_combined_dataset_v3(image_only=True, fracture_only=True)
    trainval_df, test_df = extract_holdout_test_set(full_df)
    train(trainval_df)
    final_test_and_gradcam(test_df)
