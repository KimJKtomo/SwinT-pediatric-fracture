# train_fracture_focal_v1.py (with EarlyStopping + MLflow + full print logging)
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from torchvision import transforms
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from load_new_dxmodule_0605_v2 import get_combined_dataset_v2
from unified_dataset_0605 import UnifiedFractureDataset
from SwinT_ImageOnly_Classifier import SwinTImageClassifier

class Args:
    model_name = "swin_large_patch4_window12_384_in22k"
    pretrained = True
    num_classes = 1
    batch_size = 8
    lr = 1e-4
    num_epochs = 50
    num_workers = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    threshold_sweep = [0.4, 0.45, 0.50, 0.55 ,0.6]
    patience = 7  # EarlyStopping

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

def train():
    args = Args()
    mlflow.start_run(run_name="focal_loss_pretrain")
    mlflow.log_params(vars(args))

    df = get_combined_dataset_v2(image_only=True, fracture_only=True)
    df = df[df['split'].isin(['train', 'val']) & df['label'].isin([0, 1])].reset_index(drop=True)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_loader = DataLoader(UnifiedFractureDataset(train_df, transform), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(UnifiedFractureDataset(val_df, transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SwinTImageClassifier(args).to(args.device)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

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

        train_bin_preds = [1 if p > 0.5 else 0 for p in all_preds]
        train_f1 = f1_score(all_targets, train_bin_preds)
        train_acc = accuracy_score(all_targets, train_bin_preds)
        train_cm = confusion_matrix(all_targets, train_bin_preds)

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

            print(f"[Epoch {epoch}] Val @ threshold {t:.2f} → Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"[Epoch {epoch}] Val Confusion Matrix:\n{val_cm}")

            metric_name = f"val_f1_t{str(t).replace('.', '_')}"
            mlflow.log_metric(metric_name, val_f1, step=epoch)
            mlflow.log_metric(f"val_acc_t{str(t).replace('.', '_')}", val_acc, step=epoch)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_threshold = t
                patience_counter = 0
                improved = True
                torch.save(model.state_dict(), "swinT_pretrained_fx_focal_best_0610.pt")
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
