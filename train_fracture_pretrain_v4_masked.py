import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score
from torchvision import transforms
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from torch.optim.lr_scheduler import CosineAnnealingLR

from load_new_dxmodule_0605_v3 import get_combined_dataset_v3
from unified_dataset_0605 import UnifiedFractureDataset
from SwinT_ImageOnly_Classifier import SwinTImageClassifier

# =======================
# Config and Arguments
# =======================
class Args:
    model_name = "swin_large_patch4_window12_384_in22k"
    pretrained = True
    num_classes = 1
    batch_size = 8
    lr = 5e-5
    num_epochs = 100
    num_workers = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    threshold_sweep = [0.4, 0.43, 0.45, 0.48, 0.50]
    patience = 100


# =======================
# Focal Loss
# =======================
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

# =======================
# Side Marker Masking
# =======================
def mask_side_markers(image):
    """이미지 좌/우측 15%를 마스킹하여 R/L 마커 제거"""
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

# =======================
# Main Training Function
# =======================
def train():
    args = Args()
    mlflow.start_run(run_name="v4_masked_focal_cosine_aug")
    mlflow.log_params(vars(args))

    df = get_combined_dataset_v3(image_only=True, fracture_only=True)
    df = df[df['split'].isin(['train', 'val']) & df['label'].isin([0, 1])].reset_index(drop=True)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
    ])

    train_labels = train_df['label'].tolist()
    class_counts = torch.tensor([train_labels.count(0), train_labels.count(1)])
    class_weights = 1.0 / class_counts.float()
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(MaskedFractureDataset(train_df, transform), batch_size=args.batch_size,
                              sampler=sampler, num_workers=args.num_workers)
    val_loader = DataLoader(MaskedFractureDataset(val_df, transform), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    model = SwinTImageClassifier(args).to(args.device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    best_f1, best_threshold = 0, 0.5
    patience_counter = 0

    for epoch in range(args.num_epochs):
        model.train()
        total_loss, all_preds, all_targets = 0, [], []

        for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch}] Training"):
            images, labels = images.to(args.device), labels.float().to(args.device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds += torch.sigmoid(outputs).detach().cpu().numpy().flatten().tolist()
            all_targets += labels.cpu().numpy().flatten().tolist()

        scheduler.step()

        train_bin_preds = [1 if p > 0.5 else 0 for p in all_preds]
        train_f1 = f1_score(all_targets, train_bin_preds)
        train_acc = accuracy_score(all_targets, train_bin_preds)

        print(f"\n[Epoch {epoch}] Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"[Epoch {epoch}] Train Accuracy: {train_acc:.4f} | F1-score: {train_f1:.4f}")

        mlflow.log_metric("train_loss", total_loss / len(train_loader), step=epoch)
        mlflow.log_metric("train_f1", train_f1, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"[Epoch {epoch}] Validation"):
                images, labels = images.to(args.device), labels.float().to(args.device)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                val_preds += probs.cpu().numpy().flatten().tolist()
                val_trues += labels.cpu().numpy().flatten().tolist()

        improved = False
        for t in args.threshold_sweep:
            val_bin_preds = [1 if p > t else 0 for p in val_preds]
            val_f1 = f1_score(val_trues, val_bin_preds)
            val_acc = accuracy_score(val_trues, val_bin_preds)
            val_cm = confusion_matrix(val_trues, val_bin_preds)
            recall_0 = recall_score(val_trues, val_bin_preds, pos_label=0)
            recall_1 = recall_score(val_trues, val_bin_preds, pos_label=1)

            print(f"[Epoch {epoch}] Val @ threshold {t:.2f} → Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"Recall_0 (정상): {recall_0:.4f} | Recall_1 (골절): {recall_1:.4f}")
            print(f"Confusion Matrix:\n{val_cm}")

            mlflow.log_metric(f"val_f1_t{str(t).replace('.', '_')}", val_f1, step=epoch)
            mlflow.log_metric(f"val_acc_t{str(t).replace('.', '_')}", val_acc, step=epoch)
            mlflow.log_metric(f"val_recall0_t{str(t).replace('.', '_')}", recall_0, step=epoch)
            mlflow.log_metric(f"val_recall1_t{str(t).replace('.', '_')}", recall_1, step=epoch)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_threshold = t
                patience_counter = 0
                improved = True
                torch.save(model.state_dict(), "swinT_pretrained_fx_v4_masked_best.pt")
                mlflow.pytorch.log_model(model, "best_model")
                print(f"✅ Best model saved @ threshold {t:.2f}")

        if not improved:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("⏹️ Early stopping triggered.")
                break

    print(f"\n🎯 Best F1-score: {best_f1:.4f} at threshold {best_threshold:.2f}")
    mlflow.end_run()

if __name__ == "__main__":
    train()
